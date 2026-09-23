from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch import nn
from functools import partial
from torch.nn.init import trunc_normal_
from clip import clip
from utils.layers import GraphConvolution, DistanceAdj
from utils.adapter_modules import SimpleAdapter, SimpleProj
from utils.descriptions import DESCRIPTIONS_ORI, DESCRIPTIONS_ORI_XD
from utils.dnp_vision_transformer import Aggregation_Block, Prototype_Block


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, query_feat, key_value_feat):
        attn_output, _ = self.cross_attn(query=query_feat, key=key_value_feat, value=key_value_feat)
        x = self.ln1(query_feat + attn_output)
        ffn_output = self.ffn(x)
        out = self.ln2(x + ffn_output)
        return out


class CrossModalFusionTransformer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, query_feat, key_value_feat):
        x = query_feat
        for layer in self.layers:
            x = layer(query_feat=x, key_value_feat=key_value_feat)
        return x


class CLIP_Adapter(nn.Module):
    def __init__(self, clipmodel, device, text_adapt_until=3, t_w=0.1, adapter_strength=1.0):
        super(CLIP_Adapter, self).__init__()
        self.clipmodel = clipmodel
        self.text_adapt_until = text_adapt_until
        self.t_w = t_w
        self.adapter_strength = adapter_strength
        self.device = device
        self.text_adapter = nn.ModuleList(
            [SimpleAdapter(512, 512) for _ in range(text_adapt_until)] +
            [SimpleProj(512, 512, relu=True)]
        )
        self._init_weights_()

    def _init_weights_(self):
        for p in self.text_adapter.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_text(self, text, adapt_text=True):
        if not adapt_text:
            # 消融 Adapter：走原始 CLIP 文本编码（含 text_projection），不加任何 adapter 残差。
            # 注意本项目 vendored CLIP 的 encode_text(text, token) 要求第一个参数是已嵌入的 token，
            # 第二个参数是原始 token id（用于 EOT 位置 argmax）。
            cast_dtype = self.clipmodel.token_embedding.weight.dtype
            embedded = self.clipmodel.token_embedding(text).to(cast_dtype)
            return self.clipmodel.encode_text(embedded, text)
        cast_dtype = self.clipmodel.token_embedding.weight.dtype
        x = self.clipmodel.token_embedding(text).to(cast_dtype)
        x = x + self.clipmodel.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)
        for i in range(len(self.clipmodel.transformer.resblocks)):
            x = self.clipmodel.transformer.resblocks[i](x)
            if i < self.text_adapt_until:
                adapt_out = self.text_adapter[i](x)
                adapt_out = (
                    adapt_out * x.norm(dim=-1, keepdim=True) /
                    (adapt_out.norm(dim=-1, keepdim=True) + 1e-6)
                )
                effective_t_w = self.t_w * self.adapter_strength
                x = effective_t_w * adapt_out + (1 - effective_t_w) * x
        x = x.permute(1, 0, 2)
        x = self.clipmodel.ln_final(x)
        eot_indices = text.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eot_indices]
        x = self.text_adapter[-1](x)
        return x


class SGNM(nn.Module):
    def __init__(self, feature_dim=512, num_prototypes=64, num_heads=8,
                 extractor_depth=1, decoder_depth=8, normal_selection_ratio=0.125):
        super().__init__()
        self.normal_selection_ratio = normal_selection_ratio
        self.video_prototypes = nn.Parameter(torch.randn(num_prototypes, feature_dim))
        self.dnp_extractor = nn.ModuleList([
            Aggregation_Block(
                dim=feature_dim, num_heads=num_heads, mlp_ratio=4.,
                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8)
            ) for _ in range(extractor_depth)
        ])
        self.bottleneck = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(feature_dim, feature_dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(feature_dim * 4, feature_dim))
        ]))
        self.decoder = nn.ModuleList([
            Prototype_Block(
                dim=feature_dim, num_heads=num_heads, mlp_ratio=4.,
                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8)
            ) for _ in range(decoder_depth)
        ])

    def gather_loss(self, query, keys):
        distribution = 1. - F.cosine_similarity(query.unsqueeze(2), keys.unsqueeze(1), dim=-1)
        distance, _ = torch.min(distribution, dim=2)
        gather_loss = distance.mean()
        return gather_loss

    def forward(self, visual_features, logits1, normal_selection_ratio=0.125):
        normal_selection_ratio = self.normal_selection_ratio
        B, N, D = visual_features.shape
        with torch.no_grad():
            anomaly_scores = torch.sigmoid(logits1.squeeze(-1))
            num_normal_frames = int(N * normal_selection_ratio)
            _, indices = torch.topk(anomaly_scores, k=N, largest=False, dim=1)
            normal_indices = indices[:, :num_normal_frames]
            selected_normal_features = torch.gather(
                visual_features, 1,
                normal_indices.unsqueeze(-1).expand(-1, -1, D)
            )
        agg_prototype = self.video_prototypes.unsqueeze(0).expand(B, -1, -1)
        for blk in self.dnp_extractor:
            agg_prototype = blk(agg_prototype, selected_normal_features)
        dynamic_normal_patterns = agg_prototype
        g_loss = self.gather_loss(selected_normal_features, dynamic_normal_patterns)
        bottleneck_features = visual_features
        for blk in self.bottleneck:
            bottleneck_features = blk(bottleneck_features)
        reconstructed_features = bottleneck_features
        for blk in self.decoder:
            reconstructed_features = blk(reconstructed_features, dynamic_normal_patterns)
        return reconstructed_features, g_loss


class DSANet(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 args,
                 device):
        super().__init__()
        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device

        # ACC 参数
        self.acc_threshold = args.acc_threshold
        self.acc_eta = args.acc_eta
        # ACC 图构建方式：threshold=全局阈值（CLIP 特征窄锥体下退化为全连通/单巨簇）
        # knn=互近邻 kNN 图（相对近邻关系，聚类真正生效，2026-09-12 新增）
        self.acc_graph = getattr(args, 'acc_graph', 'threshold')
        self.acc_knn_k = getattr(args, 'acc_knn_k', 8)
        # B1 伪标签模式（2026-09-15 新增）：cluster=簇均值锚（原始）；
        # class_aware=类别感知锚：异常视频锚=GT 类 one-hot、正常视频锚=normal，
        # 使用训练标签 y 而非新增标注；解决均值锚抹平类别信息、与强 adapter 相克的问题
        self.acc_pseudo = getattr(args, 'acc_pseudo', 'cluster')

        # DSA 解耦语义对齐消融开关
        self.use_dsa = getattr(args, 'use_dsa', True)
        # Adapter / ACC 消融开关
        self.use_adapter = getattr(args, 'use_adapter', True)
        self.use_acc = getattr(args, 'use_acc', True)

        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )

        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()

        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.classifier = nn.Linear(visual_width, 1)

        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        adapter_strength = getattr(args, 'adapter_strength', 1.0)
        self.clip_adapter = CLIP_Adapter(self.clipmodel, self.device, args.text_adapt_until, args.t_w, adapter_strength)

        self.video_anomaly_refiner = SGNM(
            feature_dim=visual_width,
            num_prototypes=args.num_prototypes,
            num_heads=8,
            extractor_depth=1,
            decoder_depth=args.decoder_depth,
            normal_selection_ratio=args.normal_selection_ratio
        )

        self._text_features_cache = None
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)
        trainable_modules = nn.ModuleList([self.video_anomaly_refiner])
        for m in trainable_modules.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def build_attention_mask(self, attn_window):
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(self.visual_length / attn_window)):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0
        return mask

    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1))
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2 / (x_norm_x + 1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2
        return output

    def encode_video(self, images, padding_mask, lengths):
        images = images.to(torch.float)
        position_ids = torch.arange(self.visual_length, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings

        x, _ = self.temporal((images, None))
        x = x.permute(1, 0, 2)

        adj = self.adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1])
        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))

        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))

        x = torch.cat((x1, x2), 2)
        x = self.linear(x)
        return x

    def get_text_features(self, text):
        if not self.training and self._text_features_cache is not None:
            return self._text_features_cache
        category_features = []
        if len(text) == 14:
            DESCRIPTIONS = DESCRIPTIONS_ORI
        else:
            DESCRIPTIONS = DESCRIPTIONS_ORI_XD
        for class_name, descriptions in DESCRIPTIONS.items():
            tokens = clip.tokenize(descriptions).to(self.device)
            text_features = self.clip_adapter.encode_text(tokens, adapt_text=self.use_adapter)
            mean_feature = text_features.mean(dim=0)
            mean_feature = mean_feature / mean_feature.norm().clamp_min(1e-6)
            category_features.append(mean_feature)
        text_features_ori = torch.stack(category_features, dim=0)
        if not self.training:
            self._text_features_cache = text_features_ori
        return text_features_ori

    # ==================== 新增：ACC 伪标签生成 ====================
    def generate_pseudo_labels(self, visual_features, cross_logits, lengths, video_labels=None, frame_scores=None):
        """
        visual_features: [B, T, D]
        cross_logits: [B, T, C]   (已经过 softmax 的概率)
        lengths: [B]
        video_labels: [B, C] 视频级标签（class_aware/selective 模式使用，可为 None）
        frame_scores: [B, T] 帧异常分数 sigmoid(logits1)（selective 模式使用，可为 None）
        返回: pseudo_labels [B, T, C]   (软标签，每个分量的平均概率)
        """
        # B2：选择性类别锚——对症 B1 暴露的病根（视频级常数锚压平视频内时间结构，
        # mAP@高IoU 崩塌）。异常视频内按当前异常分数取 top-k 帧锚定 GT 类，
        # 其余帧锚定 normal；正常视频全段锚 normal。k=int(L/16+1) 与 CLAS2 的
        # MIL top-k 同式（语义自洽）。帧间对比由此保留并受显式监督。
        if getattr(self, 'acc_pseudo', 'cluster') == 'selective' and video_labels is not None and frame_scores is not None:
            pseudo = torch.zeros_like(cross_logits)
            normal_anchor = torch.zeros(cross_logits.shape[-1], device=cross_logits.device, dtype=pseudo.dtype)
            normal_anchor[0] = 1.0
            for b in range(cross_logits.shape[0]):
                L = int(lengths[b].item())
                labs = video_labels[b]
                if (labs[0] > 0) or (labs.sum() == 0):
                    pseudo[b, :L] = normal_anchor
                else:
                    gt_anchor = (labs / labs.sum().clamp_min(1e-6)).to(pseudo.dtype)
                    scores = frame_scores[b, :L]
                    k = min(L, int(L / 16 + 1))
                    top_idx = scores.topk(k).indices
                    sel = torch.zeros(L, dtype=torch.bool, device=scores.device)
                    sel[top_idx] = True
                    pseudo[b, :L][sel] = gt_anchor
                    pseudo[b, :L][~sel] = normal_anchor
            return pseudo
        # B1：类别感知伪标签——锚携带视频级 GT 类别信息（训练标签 y，非新增标注）。
        # 异常视频：锚 = GT 异常类 one-hot（多类视频为均匀混合）；正常视频：锚 = normal one-hot。
        # 语义变化：一致性损失从"拉向平均分布"变为"拉向正确的类"，与强 adapter 互补而非相克。
        if getattr(self, 'acc_pseudo', 'cluster') == 'class_aware' and video_labels is not None:
            pseudo = torch.zeros_like(cross_logits)
            anchors = video_labels / video_labels.sum(dim=1, keepdim=True).clamp_min(1e-6)
            for b in range(cross_logits.shape[0]):
                L = int(lengths[b].item())
                pseudo[b, :L] = anchors[b].to(pseudo.dtype)
            return pseudo
        B, T, C = cross_logits.shape
        device = visual_features.device
        pseudo_labels = torch.zeros_like(cross_logits)

        # 视觉相似度
        vis_sim = torch.matmul(visual_features, visual_features.transpose(1, 2))  # [B,T,T]
        vis_norm = visual_features.norm(dim=-1, keepdim=True)
        vis_sim = vis_sim / (vis_norm @ vis_norm.transpose(1, 2) + 1e-8)

        # 跨模态一致性：对每对 (i,j)，取 min(prob_i[c], prob_j[c]) 的最大值
        cross_expand_i = cross_logits.unsqueeze(2)   # [B,T,1,C]
        cross_expand_j = cross_logits.unsqueeze(1)   # [B,1,T,C]
        cross_min = torch.min(cross_expand_i, cross_expand_j)  # [B,T,T,C]
        cross_consistency = cross_min.max(dim=-1)[0]  # [B,T,T]

        # 增强相似度
        enhanced_sim = vis_sim * (1 + self.acc_eta * cross_consistency)

        # 构建邻接矩阵
        if self.acc_graph == 'kmeans':
            # k-means 聚类：每视频的帧特征直接聚成 K 簇，帧锚定到所在簇的均值分布。
            # 动机：全局阈值在 CLIP 特征窄锥体下退化为单巨簇（伪标签=视频均值）；
            # 互近邻 kNN 也会被时间链串成一簇（相邻帧互为近邻）。k-means 直接给出
            # K 个局部锚；K=1 精确退化为视频均值（原始行为），K 成为天然消融轴。
            # 初始化取等间隔帧（确定性，不消耗 RNG 流），25 轮迭代。
            pseudo_labels = torch.zeros_like(cross_logits)
            vis_normed = visual_features / visual_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            for b in range(B):
                L = int(lengths[b].item())
                K = min(self.acc_knn_k, L)
                x = vis_normed[b, :L]                      # [L, D]
                init_idx = torch.linspace(0, L - 1, K).long().to(x.device)
                cent = x[init_idx].clone()                 # [K, D]
                for _ in range(25):
                    assign = torch.cdist(x, cent).argmin(dim=1)  # [L]
                    new_cent = cent.clone()
                    for k in range(K):
                        m = assign == k
                        if m.any():
                            new_cent[k] = x[m].mean(dim=0)
                    if torch.allclose(new_cent, cent, atol=1e-6):
                        break
                    cent = new_cent
                assign = torch.cdist(x, cent).argmin(dim=1)
                for k in range(K):
                    m = assign == k
                    if m.any():
                        pseudo_labels[b, :L][m] = cross_logits[b, :L][m].mean(dim=0)
            return pseudo_labels
        elif self.acc_graph == 'knn':
            # kNN 图：每帧取增强相似度 top-k 近邻（排除自身），再取互近邻（mutual kNN）。
            # 注意：实测视频特征沿时间平滑，相邻帧互为近邻导致链式连通，仍多为单簇（保留供实验）。
            adj = torch.zeros_like(enhanced_sim)
            for b in range(B):
                L = int(lengths[b].item())
                if L <= 1:
                    continue  # 单帧视频无邻居，保持单例
                k = min(self.acc_knn_k, L - 1)
                sim_b = enhanced_sim[b, :L, :L].clone()
                sim_b.fill_diagonal_(float('-inf'))  # 排除自身
                _, idx = sim_b.topk(k, dim=1)        # [L, k] 每帧的 k 近邻
                a = torch.zeros(L, L, device=enhanced_sim.device)
                a.scatter_(1, idx, 1.0)              # 有向 kNN
                a = ((a + a.t()) > 0).float()        # 互近邻：仅双向都在 top-k 时连边
                adj[b, :L, :L] = a
        else:
            # 原始方式：全局阈值二值化
            adj = (enhanced_sim > self.acc_threshold).float()  # [B,T,T]

        # 对每个样本单独处理（并查集）
        for b in range(B):
            L = lengths[b].item()
            # 只考虑有效长度
            adj_b = adj[b, :L, :L].cpu().numpy()  # 转为 numpy 方便操作
            # 并查集
            parent = list(range(L))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x, y):
                rx, ry = find(x), find(y)
                if rx != ry:
                    parent[ry] = rx

            for i in range(L):
                for j in range(L):
                    if adj_b[i, j] > 0.5:
                        union(i, j)

            # 收集每个连通分量的帧索引
            comp_dict = {}
            for i in range(L):
                root = find(i)
                if root not in comp_dict:
                    comp_dict[root] = []
                comp_dict[root].append(i)

            # 为每个分量生成伪标签（取分量内 cross_logits 的平均）
            for indices in comp_dict.values():
                comp_logits = cross_logits[b, indices, :]  # [len, C]
                comp_mean = comp_logits.mean(dim=0)        # [C]
                pseudo_labels[b, indices, :] = comp_mean

        return pseudo_labels
    # ==================== ACC 结束 ====================

    def forward(self, visual, padding_mask, text, lengths, DNP_use, scale=10, video_labels=None):
        visual_features = self.encode_video(visual, padding_mask, lengths)
        logits1 = self.classifier(visual_features + self.mlp2(visual_features))

        text_features_ori = self.get_text_features(text)

        text_features = text_features_ori
        logits_attn = logits1.permute(0, 2, 1)
        visual_attn = logits_attn @ visual_features
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        visual_attn = visual_attn.expand(visual_attn.shape[0], text_features_ori.shape[0], visual_attn.shape[2])
        text_features = text_features_ori.unsqueeze(0)
        text_features = text_features.expand(visual_attn.shape[0], text_features.shape[1], text_features.shape[2])
        text_features = text_features + visual_attn
        text_features = text_features + self.mlp1(text_features)

        visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        text_features_norm = text_features_norm.permute(0, 2, 1)
        logits2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07

        # ==================== DSA 解耦语义对齐分支 ====================
        # 将视觉特征按异常分数解耦为异常/正常两部分，分别与文本对齐 (logits3/logits4)
        if self.use_dsa:
            logits = torch.sigmoid(logits1)
            abn_logits = (scale * logits).exp() - 1
            abn_logits = F.normalize(abn_logits, p=1, dim=1)
            nor_logits = (scale * (1. - logits)).exp() - 1
            nor_logits = F.normalize(nor_logits, p=1, dim=1)

            abn_feat = torch.matmul(abn_logits.permute(0, 2, 1), visual_features)
            abn_feat_ori = abn_feat
            nor_feat = torch.matmul(nor_logits.permute(0, 2, 1), visual_features)
            nor_feat_ori = nor_feat

            nor_text_features = text_features_ori.unsqueeze(0)
            nor_text_features = nor_text_features.expand(abn_feat.shape[0], nor_text_features.shape[1], nor_text_features.shape[2])

            nor_text_features_norm = nor_text_features / nor_text_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            nor_text_features_norm = nor_text_features_norm.permute(0, 2, 1)
            nor_visual_features_norm = nor_feat_ori / nor_feat_ori.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            abn_visual_features_norm = abn_feat_ori / abn_feat_ori.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            logits3 = abn_visual_features_norm @ nor_text_features_norm.type(abn_visual_features_norm.dtype) / 0.07
            logits4 = nor_visual_features_norm @ nor_text_features_norm.type(nor_visual_features_norm.dtype) / 0.07
        else:
            logits3 = None
            logits4 = None
        # ==================== DSA 结束 ====================

        if DNP_use:
            DNP = {}
            reconstructed_features, g_loss = self.video_anomaly_refiner(visual_features, logits1)
            DNP['reconstructed_features'] = reconstructed_features
            DNP['g_loss'] = g_loss
            DNP['original_features'] = visual_features

            # ==================== 新增：生成 ACC 伪标签 ====================
            # use_acc 关闭时不生成伪标签，pseudo_labels=None -> 训练侧 loss_aux 自动置零
            if self.use_acc:
                with torch.no_grad():
                    # 对 logits2 做 softmax 得到概率分布
                    logits2_prob = F.softmax(logits2, dim=-1)
                    # B2：帧异常分数（selective 模式的 top-k 选取依据）
                    frame_scores = torch.sigmoid(logits1)[..., 0] if logits1 is not None else None
                    pseudo_labels = self.generate_pseudo_labels(visual_features, logits2_prob, lengths, video_labels, frame_scores)
                DNP['pseudo_labels'] = pseudo_labels
            else:
                DNP['pseudo_labels'] = None
            # ==================== ACC 结束 ====================

            return text_features_ori, logits1, logits2, logits3, logits4, DNP
        else:
            return text_features_ori, logits1, logits2, logits3, logits4