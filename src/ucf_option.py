import argparse

parser = argparse.ArgumentParser(description='DSANet')
parser.add_argument('--seed', default=234, type=int)

parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--visual-layers', default=2, type=int)
parser.add_argument('--attn-window', default=8, type=int)
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)
parser.add_argument('--classes-num', default=14, type=int)

parser.add_argument('--max-epoch', default=20, type=int)
parser.add_argument('--model-path', default='/home/zhangyuanfang/anaconda3/VadCLIP-main/model/model_ucf.pth')
parser.add_argument('--use-checkpoint', default=False, type=bool)
parser.add_argument('--checkpoint-path', default='/home/zhangyuanfang/anaconda3/VadCLIP-main/model/checkpoint.pth')
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--train-list', default='/home/zhangyuanfang/anaconda3/VadCLIP-main/list/ucf_CLIP_rgb.csv')
parser.add_argument('--test-list', default='/home/zhangyuanfang/anaconda3/VadCLIP-main/list/ucf_CLIP_rgbtest.csv')
parser.add_argument('--gt-path', default='/home/zhangyuanfang/anaconda3/VadCLIP-main/list/gt_ucf.npy')
parser.add_argument('--gt-segment-path', default='/home/zhangyuanfang/anaconda3/VadCLIP-main/list/gt_segment_ucf.npy')
parser.add_argument('--gt-label-path', default='/home/zhangyuanfang/anaconda3/VadCLIP-main/list/gt_label_ucf.npy')

parser.add_argument('--lr', type=float, default=7e-5)

#DNP
parser.add_argument('--decoder_depth', type=int, default=8)
parser.add_argument('--normal_selection_ratio', type=float, default=0.8)
parser.add_argument('--num_prototypes', type=int, default=16)
parser.add_argument('--DNP_use', default=True, type=lambda v: str(v).lower() not in ('false', '0', 'no', ''))

#Adapter
parser.add_argument('--text_adapt_until', default=3, type=int)
parser.add_argument('--t_w', default=0.1, type=float)
parser.add_argument('--adapter_strength', default=1.0, type=float,
                    help='Scale Adapter residual strength; effective weight = t_w * adapter_strength')

parser.add_argument('--temp', default=5.0, type=float)

parser.add_argument('--loss2_weight', type=float, default=1.1)

# ACC 模块参数（model.py 初始化时读取 acc_threshold/acc_eta，必须提供）
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

# 消融开关：MTA / Adapter / ACC（默认全开 = 与原模型一致）
_bool = lambda v: str(v).lower() not in ('false', '0', 'no', '')
parser.add_argument('--use_adapter', default=True, type=_bool, help='CLIP 文本 Adapter')
parser.add_argument('--use_acc', default=True, type=_bool, help='ACC 伪标签辅助损失（依赖 DNP）')
parser.add_argument('--exp_tag', default='', type=str, help='实验名，用于日志命名')

# Logging
parser.add_argument('--log-dir', default='/home/zhangyuanfang/anaconda3/VadCLIP-main/src/logs', type=str)