# ICRNet 投稿代码包（2026-09-24）

论文：Intra-Video Consistency Refinement Network (ICRNet)
锁定结果（单 seed 234，复测口径）：**XD-Violence AP 86.62 / mAP AVG 30.25；UCF-Crime AUC 88.94 / mAP AVG 12.21**

## 目录结构

```
├── README.md
├── checkpoints/
│   ├── abl_km8_s234.pth        # XD-Violence 终版模型（k-means K=8, λ_acc=0.1）
│   └── abl_adp05_ucf_s234.pth  # UCF-Crime 终版模型（K=1 视频均值锚, λ_acc=0.02, adapter 0.5）
└── src/
    ├── model.py                # ICRNet/DSANet 模型（ACC 伪标签在 generate_pseudo_labels）
    ├── xd_train.py / xd_test.py / xd_option.py     # XD-Violence 训练/测试/配置
    ├── ucf_train.py / ucf_test.py / ucf_option.py  # UCF-Crime 训练/测试/配置
    ├── clip/                   # vendored CLIP（encode_text 签名为本仓库定制，勿用官方包替换）
    └── utils/                  # 数据集、损失组件、mAP 评测等依赖模块
```

## 环境

- Python 3.10；PyTorch（单卡 RTX 5090 验证）；numpy / pandas / scikit-learn / opencv-python / ftfy
- 关键：CLIP 为仓库内置 vendored 版本（`src/clip/`），其 `encode_text(text, token)` 签名非官方标准

## 数据准备

训练/测试脚本通过 csv 索引特征文件（`path,label` 两列，path 为预提取特征 .npy 绝对路径）：
- XD-Violence：CLIP ViT-B/16 RGB 帧特征（`XDTrainClipFeatures/`、`XDTestClipFeatures/`）+ `list/annotations.txt`（帧级 GT，仅 test）
- UCF-Crime：CLIP RGB 帧特征 + `list/Temporal_Anomaly_Annotation.txt`
- csv 默认路径见 `xd_option.py` / `ucf_option.py` 的 `--train-list/--test-list/--gt-path` 等参数，请改为本机路径

## 复现论文主实验

测试（直接用附带 checkpoint，出论文表数字）：

```bash
cd src
python xd_test.py --model-path ../checkpoints/abl_km8_s234.pth          # XD: AP 86.62, AVG 30.25
python ucf_test.py --model-path ../checkpoints/abl_adp05_ucf_s234.pth   # UCF: AUC 88.94, AVG 12.21
```

训练（论文超参，固定 seed 234）：

```bash
# XD-Violence：K=8, λ_acc=0.1, adapter 1.0（默认）
python xd_train.py --seed 234 --acc_graph kmeans --acc_knn_k 8 --loss_aux_weight 0.1 \
  --model-path <输出路径>.pth --checkpoint-path <断点路径>.pth --log-dir logs

# UCF-Crime：K=1（视频均值锚）, λ_acc=0.02, adapter 0.5
python ucf_train.py --seed 234 --acc_graph kmeans --acc_knn_k 1 --loss_aux_weight 0.02 --adapter_strength 0.5 \
  --model-path <输出路径>.pth --checkpoint-path <断点路径>.pth --log-dir logs
```

注意：`--acc_knn_k` 在 `--acc_graph kmeans` 模式下即簇数 K（见 `model.py` 的 `K = min(self.acc_knn_k, L)`）；同 seed 训练管线确定，两次运行结果逐位一致。

## 其他说明

- 优化器：主分支 AdamW + 余弦退火，重建分支 StableAdamW；XD 10 epoch / UCF 20 epoch，按粗粒度指标保留最优 checkpoint
- 消融与分析（w/o ACC、K 扫描、λ 扫描、E1/E3 特征分析）由同一脚本的超参组合复现，分析脚本（analysis_acc.py / analysis_plot.py）不在本包内，需要可联系作者
