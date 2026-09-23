import sys
import os
import datetime
import logging
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
from utils.StableAdamW import StableAdamW

from model import DSANet
from xd_test import test
from utils.dataset import XDDataset
from utils.tools import get_prompt_text, get_batch_label
import xd_option

torch.cuda.set_per_process_memory_fraction(0.85)
torch.cuda.empty_cache()

def setup_logger(log_dir):
    """配置日志系统，双向输出到终端和文件"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    now = datetime.datetime.now()
    time_str = now.strftime('%Y%m%d_%H%M%S')
    temp_log_file = os.path.join(log_dir, f'temp_train_{time_str}.txt')

    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # 文件 Handler
    fh = logging.FileHandler(temp_log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 终端 Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger, temp_log_file, time_str

def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat((instance_logits, tmp))

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss

def CLASM_EVENT(logits, labels, lengths, device, epsilon=0.1):
    num_classes = logits.shape[2]
    instance_logits = torch.zeros(0).to(device)

    labels_sum = labels.sum(dim=1, keepdim=True).clamp(min=1e-6)
    labels_sm = (1 - epsilon) * (labels / labels_sum) + epsilon / num_classes
    labels_sm = labels_sm.to(device)

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels_sm * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def CLASM_BKG(logits, labels, lengths, device, epsilon=0.1):
    num_classes = logits.shape[2]
    instance_logits = torch.zeros(0).to(device)

    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)
    labels2 = torch.full(labels.shape, 0.01, device=labels.device)
    labels2[:, 0] = 1
    labels2_sum = labels2.sum(dim=1, keepdim=True).clamp(min=1e-6)
    labels2 = (1 - epsilon) * (labels2 / labels2_sum) + epsilon / num_classes
    labels2 = labels2.to(device)

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels2 * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

from torch.optim.lr_scheduler import _LRScheduler
class WarmCosineScheduler(_LRScheduler):

    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
        iters = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((warmup_schedule, schedule))
        super(WarmCosineScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [self.final_value for base_lr in self.base_lrs]
        else:
            return [self.schedule[self.last_epoch] for base_lr in self.base_lrs]

class ConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, logits1, original_features, reconstructed_features, lengths):
        recon_error_score = 1.0 - F.cosine_similarity(original_features, reconstructed_features, dim=-1)
        recon_error_score = recon_error_score / 2.0
        classifier_prob_score = torch.sigmoid(logits1.squeeze(-1))
        B, N = logits1.shape[0], logits1.shape[1]
        mask = torch.arange(N, device=logits1.device)[None, :] < lengths[:, None]
        valid_recon_scores = recon_error_score[mask]
        valid_classifier_scores = classifier_prob_score[mask]
        consistency_loss = self.mse_loss(valid_classifier_scores, valid_recon_scores)
        return consistency_loss

consistency_loss_fn = ConsistencyLoss()

def get_gradient_norms(model):
    total_norm = 0
    max_norm = 0
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            max_norm = max(max_norm, param_norm)
            gradients[name] = param_norm
    total_norm = total_norm ** 0.5
    return total_norm, max_norm, gradients


def get_dsa_weight(args, epoch_idx, batch_idx, num_batches_per_epoch):
    if not args.use_dsa or args.dsa_loss_weight <= 0:
        return 0.0

    progress_epoch = epoch_idx + batch_idx / max(1, num_batches_per_epoch)
    if progress_epoch < args.dsa_start_epoch:
        return 0.0

    if args.dsa_warmup_epochs <= 0:
        return args.dsa_loss_weight

    ratio = (progress_epoch - args.dsa_start_epoch) / args.dsa_warmup_epochs
    ratio = max(0.0, min(1.0, ratio))
    if args.dsa_warmup_mode == 'cosine':
        ratio = 0.5 * (1.0 - np.cos(np.pi * ratio))

    return args.dsa_loss_weight * ratio

def train(model, train_loader, test_loader, args, label_map: dict, device):
    # 初始化日志记录器
    logger, temp_log_file, time_str = setup_logger(args.log_dir)
    
    logger.info("="*60)
    logger.info("Training Started: XD-Violence Anomaly Detection")
    logger.info("="*60)
    logger.info("Configurations:")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info("="*60)
    logger.info(
        f"DSA schedule: target={args.dsa_loss_weight}, start_epoch={args.dsa_start_epoch}, "
        f"warmup_epochs={args.dsa_warmup_epochs}, mode={args.dsa_warmup_mode}, scale={args.dsa_scale}"
    )
    logger.info("="*60)

    model.to(device)

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    refiner_params = []
    main_model_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'video_anomaly_refiner' in name:
            refiner_params.append(param)
        else:
            main_model_params.append(param)

    optimizer_refiner = StableAdamW(
        [{'params': refiner_params}],
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        amsgrad=True,
        eps=1e-10
    )
    total_epochs = args.max_epoch
    num_batches_per_epoch = len(train_loader)
    total_iters_refiner = total_epochs * num_batches_per_epoch
    scheduler_refiner = WarmCosineScheduler(
        optimizer_refiner,
        base_value=args.lr,
        final_value=args.lr * 0.1,
        total_iters=total_iters_refiner,
        warmup_iters=100
    )
    optimizer_main = torch.optim.AdamW(
        [{'params': main_model_params}],
        lr=args.lr
    )
    if args.main_scheduler == 'step_warmcosine':
        total_iters_main = total_epochs * num_batches_per_epoch
        warmup_iters_main = int(args.warmup_ratio * num_batches_per_epoch)
        scheduler_main = WarmCosineScheduler(
            optimizer_main,
            base_value=args.lr,
            final_value=args.lr * args.min_lr_ratio,
            total_iters=total_iters_main,
            warmup_iters=warmup_iters_main
        )
    else:
        scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_main,
            T_max=args.max_epoch
        )

    prompt_text = get_prompt_text(label_map)
    ap_best = 0.0 # 统一追踪 AP2
    epoch = 0
    global_step = 0
    no_improve_evals = 0
    stop_training = False

    if args.use_checkpoint == True:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_main.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        global_step = checkpoint.get('global_step', 0)
        ap_best = checkpoint['ap']
        logger.info("Loaded checkpoint info:")
        logger.info(f"  epoch: {epoch+1}")
        logger.info(f"  global_step: {global_step}")
        logger.info(f"  ap: {ap_best:.4f}")

    def evaluate_and_update_best(eval_tag, e, current_global_step, count_patience=True):
        nonlocal ap_best, no_improve_evals
        if hasattr(model, '_text_features_cache'):
            model._text_features_cache = None
        logger.info(eval_tag)
        AUC1, AP1, AUC2, AP2, mAP = test(
            model, test_loader, args.visual_length, prompt_text,
            gt, gtsegments, gtlabels, args.DNP_use, args, device, logger=logger
        )
        model.train()

        if AP2 > ap_best:
            logger.info(f"🔥 New best AP2 found: {AP2:.6f} (Previous: {ap_best:.6f})")
            ap_best = AP2
            no_improve_evals = 0
            checkpoint = {
                'epoch': e,
                'global_step': current_global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_main.state_dict(),
                'ap': ap_best
            }
            torch.save(checkpoint, args.checkpoint_path)
        elif count_patience:
            no_improve_evals += 1
        return AUC1, AP1, AUC2, AP2, mAP

    highfreq_enabled = args.eval_interval > 0 and args.eval_until_epoch > 0

    # SWA：从指定 epoch 起对模型权重做运行平均（训练结束存 *_swa.pth 并复测）
    swa_state, swa_n = None, 0
    swa_start = args.swa_start if args.swa_start > 0 else max(1, args.max_epoch // 2)
    for e in range(args.max_epoch):
        DNP_use = args.DNP_use
        model.train()
        loss_total1 = 0
        loss_total2 = 0
        loss_total4 = 0
        loss_total5 = 0
        loss_aux_total = 0
        loss_ent_total = 0   
        grad_norm_total = 0
        grad_max_norm_total = 0

        for i, item in enumerate(train_loader):
            step = 0
            visual_feat, text_labels, feat_lengths = item
            visual_feat = visual_feat.to(device)
            feat_lengths = feat_lengths.to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)

            if DNP_use == True:
                text_features, logits1, logits2, logits3, logits4, DNP = model(visual_feat, None, prompt_text, feat_lengths, DNP_use, scale=args.dsa_scale, video_labels=text_labels)
            else:
                text_features, logits1, logits2, logits3, logits4 = model(visual_feat, None, prompt_text, feat_lengths, DNP_use, scale=args.dsa_scale)

            loss1 = CLAS2(logits1, text_labels, feat_lengths, device)
            loss_total1 += loss1.item()

            loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            loss_total2 += loss2.item()

            if DNP_use == True:
                consistency_loss = consistency_loss_fn(
                    logits1=logits1,
                    original_features=DNP['original_features'],
                    reconstructed_features=DNP['reconstructed_features'],
                    lengths=feat_lengths
                )
                g_loss = DNP['g_loss']
                
                pseudo_labels = DNP.get('pseudo_labels', None)
                if pseudo_labels is not None:
                    logits2_prob = F.softmax(logits2, dim=-1)
                    mask2d = torch.arange(logits2.size(1), device=device)[None, :] < feat_lengths[:, None]
                    if getattr(args, 'acc_loss_mode', 'dist') == 'normal_only':
                        # A2：ACC 一致性只作用于 normality 通道（第 0 类），异常类间分布放开，
                        # 保留特征级增益（logits1/AP2）同时切断对细粒度类别分布的压平（mAP 通道）
                        loss_aux = F.l1_loss(logits2_prob[..., 0][mask2d], pseudo_labels[..., 0][mask2d])
                    else:
                        mask = mask2d.unsqueeze(-1).expand_as(logits2_prob)
                        loss_aux = F.l1_loss(logits2_prob[mask], pseudo_labels[mask])
                else:
                    loss_aux = torch.tensor(0.0, device=device)
            else:
                consistency_loss = torch.tensor(0.0, device=device)
                g_loss = torch.tensor(0.0, device=device)
                loss_aux = torch.tensor(0.0, device=device)

            # DSA 关闭时 logits3/logits4 为 None，对应 loss4/loss5 置零
            if logits3 is not None:
                loss4 = CLASM_EVENT(logits3, text_labels, feat_lengths, device)
                loss5 = CLASM_BKG(logits4, text_labels, feat_lengths, device)
            else:
                loss4 = torch.tensor(0.0, device=device)
                loss5 = torch.tensor(0.0, device=device)
            loss_total4 += loss4.item()
            loss_total5 += loss5.item()

            loss3 = torch.zeros(1).to(device)
            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            loss3 = loss3 / 6

            # R3：每帧熵惩罚——不指定押哪类，只浓缩已有质量（对症 @0.1 类别指派钝化；
            # 与 ACC 锚无关，独立权重 acc_entropy，作用于 logits2 的帧级分布）
            if getattr(args, 'acc_entropy', 0) > 0:
                probs_e = F.softmax(logits2, dim=-1)
                ent = -(probs_e * torch.log(probs_e.clamp_min(1e-8))).sum(dim=-1)
                mask_e = torch.arange(logits2.size(1), device=device)[None, :] < feat_lengths[:, None]
                loss_ent = ent[mask_e].mean()
            else:
                loss_ent = torch.tensor(0.0, device=device)
            loss_aux_total += loss_aux.item()
            loss_ent_total += loss_ent.item()

            current_dsa_weight = get_dsa_weight(args, e, i, num_batches_per_epoch)
            dsa_loss = current_dsa_weight * (loss4 + loss5)
            if DNP_use == True:
                loss = loss1 + loss2 * args.loss2_weight + loss3 + dsa_loss + consistency_loss + g_loss + loss_aux * args.loss_aux_weight + args.acc_entropy * loss_ent
            else:
                loss = loss1 + loss2 + loss3 + dsa_loss + loss_aux * args.loss_aux_weight + args.acc_entropy * loss_ent

            optimizer_main.zero_grad()
            optimizer_refiner.zero_grad()
            loss.backward()

            total_norm, max_norm, gradients = get_gradient_norms(model)
            grad_norm_total += total_norm
            grad_max_norm_total += max_norm

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer_main.step()
            optimizer_refiner.step()
            scheduler_refiner.step()
            if args.main_scheduler == 'step_warmcosine':
                scheduler_main.step()
            global_step += 1
            step += i * train_loader.batch_size
            if step % 4800 == 0 and step != 0:
                current_lr_main = optimizer_main.param_groups[0]['lr']
                log_items = [
                    f"epoch: {e+1}",
                    f"step: {step}",
                    f"lr: {current_lr_main:.6f}",
                    f"grad_norm: {total_norm:.6f}",
                    f"loss1: {loss_total1 / (i+1):.6f}",
                    f"loss2: {loss_total2 / (i+1):.6f}",
                    f"loss3: {loss3.item():.6f}",
                    f"loss4: {loss_total4 / (i+1):.6f}",
                    f"loss5: {loss_total5 / (i+1):.6f}",
                    f"dsa_weight: {current_dsa_weight:.6f}",
                    f"dsa_scale: {args.dsa_scale:.3f}",
                    f"dsa_loss: {dsa_loss.item():.6f}",
                    f"loss_aux: {loss_aux_total / (i+1):.6f}",
                    f"loss_ent: {loss_ent_total / (i+1):.6f}",
                ]
                if DNP_use:
                    log_items.append(f"consistency_loss: {consistency_loss.item():.6f}")
                    log_items.append(f"g_loss: {g_loss.item():.6f}")
                
                logger.info(" | ".join(log_items))

            if highfreq_enabled and (e + 1) <= args.eval_until_epoch and global_step % args.eval_interval == 0:
                evaluate_and_update_best(
                    f"--- Step Evaluation epoch {e+1} global_step {global_step} dsa_weight {current_dsa_weight:.6f} ---",
                    e,
                    global_step
                )
                if args.early_stop_patience > 0 and no_improve_evals >= args.early_stop_patience:
                    logger.info(f"Early stopping triggered: no AP2 improvement for {no_improve_evals} evals")
                    stop_training = True
                    break

        if args.main_scheduler == 'epoch_cosine':
            scheduler_main.step()

        # SWA 累积：epoch 达到起点后纳入权重平均
        if args.swa and (e + 1) >= swa_start:
            if swa_state is None:
                swa_state = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
                swa_n = 1
            else:
                swa_n += 1
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        swa_state[k].add_((v.detach().float() - swa_state[k]) / swa_n)
                    else:
                        swa_state[k] = v.detach().clone()

        evaluate_and_update_best(f"--- Epoch {e+1} Evaluation ---", e, global_step, count_patience=False)

        if os.path.exists(args.checkpoint_path):
            checkpoint = torch.load(args.checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])

        if stop_training:
            break

    checkpoint = torch.load(args.checkpoint_path)
    torch.save(checkpoint['model_state_dict'], args.model_path)

    # SWA：保存平均权重并复测（AP/mAP 明细由 test 内部逐行打印）
    if args.swa and swa_state is not None:
        swa_path = args.model_path.replace('.pth', '_swa.pth')
        cur_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: v.to(cur_state[k].dtype) for k, v in swa_state.items()})
        torch.save(model.state_dict(), swa_path)
        logger.info(f"SWA 权重（{swa_n} 个 epoch 的平均）已保存: {swa_path}")
        _, _, _, swa_ap2, _ = test(model, testloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, DNP_use, device, args, logger=logger)
        logger.info(f"SWA 复测: AP2={swa_ap2:.6f}")
        model.load_state_dict(cur_state)
    
    # 训练结束后清理并重命名日志文件
    logger.info("="*60)
    logger.info(f"Training Complete. Best AP2: {ap_best:.6f}")
    logger.info("="*60)
    
    # 关闭 handler 以便释放文件锁
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
        
    final_log_name = f"{time_str}_{ap_best:.4f}.txt"
    final_log_path = os.path.join(args.log_dir, final_log_name)
    os.rename(temp_log_file, final_log_path)
    print(f"✅ Final training log saved as: {final_log_path}")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = xd_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'})

    train_dataset = XDDataset(args.visual_length, args.train_list, False, label_map)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = XDDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = DSANet(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, args, device)
    train(model, train_loader, test_loader, args, label_map, device)