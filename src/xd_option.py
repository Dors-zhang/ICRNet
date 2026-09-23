import argparse

parser = argparse.ArgumentParser(description='DSANet')
parser.add_argument('--seed', default=234, type=int)

parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--visual-layers', default=1, type=int)
parser.add_argument('--attn-window', default=64, type=int)
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)
parser.add_argument('--classes-num', default=7, type=int)

parser.add_argument('--max-epoch', default=10, type=int)
parser.add_argument('--model-path', default='/home/zhangyuanfang/anaconda3/VadCLIP-main/model/model_xdDS.pth')
parser.add_argument('--use-checkpoint', default=False, type=bool)
parser.add_argument('--checkpoint-path', default='/home/zhangyuanfang/anaconda3/VadCLIP-main/model/checkpoint_xdDS.pth')
parser.add_argument('--batch-size', default=96, type=int)
parser.add_argument('--train-list', default='/home/zhangyuanfang/anaconda3/VadCLIP-main/list/xd_CLIP_rgb.csv')
parser.add_argument('--test-list', default='/home/zhangyuanfang/anaconda3/VadCLIP-main/list/xd_CLIP_rgbtest.csv')
parser.add_argument('--gt-path', default='/home/zhangyuanfang/anaconda3/VadCLIP-main/list/gt.npy')
parser.add_argument('--gt-segment-path', default='/home/zhangyuanfang/anaconda3/VadCLIP-main/list/gt_segment.npy')
parser.add_argument('--gt-label-path', default='/home/zhangyuanfang/anaconda3/VadCLIP-main/list/gt_label.npy')

parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--scheduler-rate', default=0.1)
parser.add_argument('--scheduler-milestones', default=[3, 6, 10])

#DNP
parser.add_argument('--decoder_depth', type=int, default=8)
parser.add_argument('--normal_selection_ratio', type=float, default=0.8)
parser.add_argument('--DNP_use', default=True, type=lambda v: str(v).lower() not in ('false', '0', 'no', ''))
parser.add_argument('--num_prototypes', type=int, default=16)

parser.add_argument('--loss2_weight', type=float, default=5.0)

parser.add_argument('--temp', default=1.0, type=float)

#Adapter
parser.add_argument('--text_adapt_until', default=1, type=int)
parser.add_argument('--t_w', default=0.6, type=float)

# ACC 模块参数
parser.add_argument('--acc_threshold', default=0.7, type=float, help='Threshold for binarizing adjacency matrix')
parser.add_argument('--acc_eta', default=0.5, type=float, help='Cross-modal correction factor')
parser.add_argument('--loss_aux_weight', default=0.1, type=float, help='Weight for ACC auxiliary loss')
parser.add_argument('--acc_loss_mode', default='dist', choices=['dist', 'normal_only'], type=str,
                    help='ACC 一致性损失作用面：dist=全类别分布(原始)；normal_only=只约束 normality 通道(A2 改进，切断 mAP 类别压平通道)')
parser.add_argument('--acc_pseudo', default='cluster', choices=['cluster', 'class_aware', 'selective'], type=str,
                    help='伪标签模式：cluster=簇均值锚(原始)；class_aware=B1全段类别锚(已证压平时间结构)；selective=B2选择性类别锚(top-k帧锚GT类/其余锚normal)')
parser.add_argument('--swa', default=False, type=lambda v: str(v).lower() not in ('false', '0', 'no', ''), help='SWA 权重平均：从第 swa_start 个 epoch 起对权重做运行平均，训练结束保存 *_swa.pth 并复测')
parser.add_argument('--acc_entropy', default=0.0, type=float, help='R3 每帧熵惩罚权重 β（0=关）：不指定押哪类，只浓缩每帧已有类别质量，对症 @0.1 指派钝化')
parser.add_argument('--swa_start', default=0, type=int, help='SWA 起始 epoch（1 基）；0 = 自动取总 epoch 数的一半')
parser.add_argument('--acc_graph', default='threshold', choices=['threshold', 'knn', 'kmeans'], type=str,
                    help='ACC 亲和图构建：threshold=全局阈值(窄锥体特征下退化为单巨簇/视频均值锚)；knn=互近邻kNN图(时间链仍单簇,保留实验);kmeans=每视频k-means聚K簇(K=acc_knn_k,局部锚定)')
parser.add_argument('--acc_knn_k', default=8, type=int,
                    help='kNN 图每帧近邻数（仅 acc_graph=knn 时生效）')

# DSA 解耦语义对齐消融开关：控制 logits3/logits4 分支及其 loss4/loss5
# 默认 True = 保留 DSA（与原模型一致）；False = 关闭解耦对齐，仅保留主对齐 logits2/loss2
parser.add_argument('--use_dsa', default=True, type=lambda v: str(v).lower() not in ('false', '0', 'no', ''),
                    help='DSA 解耦语义对齐 (logits3/logits4 与 loss4/loss5)')
parser.add_argument('--dsa_loss_weight', default=1.0, type=float,
                    help='Target weight for DSA auxiliary losses loss4/loss5 after warmup; 1.0 = original, 0.5 = weakened')
parser.add_argument('--dsa-start-epoch', default=0.0, type=float,
                    help='Epoch progress before applying DSA loss; 1.0 means skip DSA loss for the whole first epoch')
parser.add_argument('--dsa-warmup-epochs', default=0.0, type=float,
                    help='Number of epochs to ramp DSA loss weight from 0 to dsa_loss_weight after dsa-start-epoch; 0 disables warmup')
parser.add_argument('--dsa-warmup-mode', default='linear', choices=['linear', 'cosine'],
                    help='Schedule shape for DSA loss warmup')
parser.add_argument('--dsa-scale', default=10.0, type=float,
                    help='Scale used in DSA abnormal/normal attention weighting; original implicit value is 10.0')

# 消融开关：MTA / Adapter / ACC（默认全开 = 与原模型一致）
_bool = lambda v: str(v).lower() not in ('false', '0', 'no', '')
parser.add_argument('--use_adapter', default=True, type=_bool, help='CLIP 文本 Adapter')
parser.add_argument('--use_acc', default=True, type=_bool, help='ACC 伪标签辅助损失（依赖 DNP）')
parser.add_argument('--exp_tag', default='', type=str, help='实验名，用于日志命名')

# Logging
parser.add_argument('--log-dir', default='/home/zhangyuanfang/anaconda3/VadCLIP-main/src/logs', type=str)

# XD round4: 高频验证 / 早停 / 主优化器调度策略
parser.add_argument('--eval-interval', default=0, type=int,
                    help='High-frequency validation interval in optimizer steps; 0 disables step validation')
parser.add_argument('--eval-until-epoch', default=0, type=int,
                    help='Run high-frequency validation through this epoch number; 0 disables step validation')
parser.add_argument('--early-stop-patience', default=0, type=int,
                    help='Stop after this many high-frequency evaluations without AP2 improvement; 0 disables early stopping')
parser.add_argument('--main-scheduler', default='epoch_cosine', choices=['epoch_cosine', 'step_warmcosine'],
                    help='Scheduler for the main optimizer')
parser.add_argument('--warmup-ratio', default=0.0, type=float,
                    help='Warmup length as a fraction of one epoch for step_warmcosine')
parser.add_argument('--min-lr-ratio', default=0.1, type=float,
                    help='Final LR ratio for step_warmcosine')