# Mean-Teacher 半监督 COD 的动态原型库融合方案

## 摘要
- 目标是在 `BaseLine` 的 Mean-Teacher 框架中，融合 `RISE` 的空库初始化思想、`EASE` 的前/背景双库检索与 `M(x)` 生成、`CRLN` 的“原型参与纠正预测 + 融合模块单独监督”策略，且统一基于 decoder `p3` 特征。
- 原型库改为动态规模：每个 epoch 清空；每张有标签图在 `p3` 空间提取 `1` 个前景原型和 `1` 个背景原型；直接拼接成当轮全局库；不设固定库容量，不做 KMeans，不跨 epoch 累积。
- 监督改为两条路径：
  - 主模型：监督 student 主输出 `\hat y` 和原型图 `H(x)`
  - 融合标量 `μ`：只用 labeled 数据，通过纠正预测 `\hat y_r` 的损失单独更新
- 推理与 evaluator 同样接入原型库，最终输出使用纠正后的 `y_final = y + (1-\mu)H(x)`。

## 核心设计
### 1. 原型库构建
- 输入特征固定为 decoder `p3`，即 `p3 ∈ R^{B×C×h×w}`，`C/h/w` 动态读取，不能写死。
- 每个 epoch 开始时清空库：
  \[
  \mathcal P_{fg} = \emptyset,\quad \mathcal P_{bg} = \emptyset
  \]
- 对每张 labeled 图：
  1. 将 `gt` resize 到 `p3` 空间
  2. `fg_mask = gt > 0.5`, `bg_mask = 1 - fg_mask`
  3. 对 `p3` 做 masked average pooling，分别得到 `1` 个前景原型、`1` 个背景原型
  4. 若某类像素数 `< prototype_min_pixels`，则跳过该类
  5. 将所有有效图像原型直接拼接成全局库
- 库规模是动态的：
  \[
  |\mathcal P_{fg}| = N_{fg}^{valid},\quad |\mathcal P_{bg}| = N_{bg}^{valid}
  \]
  其中 `N_valid` 为当前 epoch 已见到的有效 labeled 图数。

### 2. 原型检索
- 对当前 batch 的 `p3` 展平得到查询特征：
  \[
  Q \in R^{B\times HW\times C}
  \]
- 分别与前景/背景原型库做余弦相似度：
  \[
  Sim_{fg} = QP_{fg}^\top,\quad Sim_{bg} = QP_{bg}^\top
  \]
- 逐行取 Top-k，`k=16`：
  \[
  S^{fg}_{topk} = TopK(Sim_{fg}),\quad S^{bg}_{topk} = TopK(Sim_{bg})
  \]
- 用温度系数 `T=prototype_sim_temperature` 做加权融合：
  \[
  \alpha^{fg} = Softmax(S^{fg}_{topk}/T),\quad
  fu_{fg} = \sum_j \alpha^{fg}_j \odot S^{fg}_{topk,j}
  \]
  \[
  \alpha^{bg} = Softmax(S^{bg}_{topk}/T),\quad
  fu_{bg} = \sum_j \alpha^{bg}_j \odot S^{bg}_{topk,j}
  \]

### 3. 从 `fu_fg/fu_bg` 到 `H(x)`
- 用 `p3` 经过轻量 MLP/1×1 conv 生成动态权重：
  \[
  \alpha(x)=\sigma(MLP(p3)),\quad \beta(x)=1-\alpha(x)
  \]
- 基于稳定边界的 `M(x)`：
  \[
  \theta_{img}=KDE\_Min((fu_{fg}-fu_{bg}).detach())
  \]
  \[
  M(x)=\sigma\left(\frac{\alpha(x)fu_{fg}-(1-\alpha(x))fu_{bg}-\theta_{img}}{\tau}\right)
  \]
- 上采样到原图得到：
  \[
  H(x)=Upsample(M(x))
  \]

### 4. CRLN式融合与监督
- 引入单个全局可学习标量 `μ ∈ [0,1]`
- `μ` 只通过 labeled 数据的纠正预测损失更新，不由 unlabeled loss 更新
- 有标签：
  \[
  \hat y_r = \hat y + (1-\mu)\cdot H(x)
  \]
- 无标签：
  \[
  \bar y_r = \bar y + (1-\mu)\cdot H(x)
  \]
- 其中：
  - `\hat y` 是 student 主输出概率图
  - `\bar y` 是 teacher 伪标签概率图
- 为保持概率合法，融合后统一：
  \[
  y_r = clamp(y_r, 0, 1)
  \]

## 关键代码骨架
### 1. 配置
`BaseLine/config/base/prototype.py`
```python
prototype_enable = True
prototype_feature_level = "p3"
prototype_source_branch = "student"

prototype_bank_policy = "per_image_masked_pool_dynamic"
prototype_bank_clear_each_epoch = True
prototype_bank_rebuild_interval = 1
prototype_min_pixels = 16

prototype_topk = 16
prototype_sim_temperature = 0.05
prototype_tau = 0.07
prototype_theta_method = "kde_min"
prototype_theta_fallback = 0.0

prototype_alpha_hidden_ratio = 4

prototype_mu_init = 0.5
prototype_mu_min = 0.0
prototype_mu_max = 1.0
prototype_mu_lr = 1e-3
prototype_mu_weight_decay = 0.0

prototype_loss_weight_h = 0.3

prototype_checkpoint_policy = "save_and_load"
prototype_eval_policy = "checkpoint_then_rebuild"
```

`BaseLine/config/mkcfg.py`
```python
COMMON_CONFIG_DIR = 'config/base/common.py'
MODEL_CONFIG_DIR = 'config/base/model.py'
PROTOTYPE_CONFIG_DIR = 'config/base/prototype.py'

class Config:
    def __init__(self, run_cfg: str):
        logger.key_info("Initialize config...")
        self.merge_from_file(COMMON_CONFIG_DIR)
        self.merge_from_file(MODEL_CONFIG_DIR)
        self.merge_from_file(PROTOTYPE_CONFIG_DIR)
        self.merge_from_file(run_cfg)
        logger.success_info("Config merged from {}.".format(run_cfg))
```

### 2. 动态原型库
`BaseLine/Paired_Background_Guidance/prototype_bank.py`
```python
import torch
import torch.nn.functional as F

class DynamicPrototypeBank:
    def __init__(self, min_pixels=16):
        self.min_pixels = min_pixels
        self.proto_fg = None
        self.proto_bg = None

    def begin_epoch(self):
        self.proto_fg = None
        self.proto_bg = None

    @staticmethod
    def _masked_avg_pool(feat, mask, min_pixels):
        # feat: [C, h, w], mask: [1, h, w]
        valid = mask.sum()
        if valid.item() < min_pixels:
            return None
        vec = (feat * mask).sum(dim=(1, 2)) / (valid + 1e-6)
        return F.normalize(vec, dim=0)

    def append_from_labeled_batch(self, p3, gt):
        # p3: [B, C, h, w], gt: [B, 1, H, W]
        gt_small = F.interpolate(gt.float(), size=p3.shape[-2:], mode="nearest")
        fg_list, bg_list = [], []

        for i in range(p3.shape[0]):
            feat_i = p3[i]
            fg_mask = (gt_small[i] > 0.5).float()
            bg_mask = 1.0 - fg_mask

            fg_proto = self._masked_avg_pool(feat_i, fg_mask, self.min_pixels)
            bg_proto = self._masked_avg_pool(feat_i, bg_mask, self.min_pixels)

            if fg_proto is not None:
                fg_list.append(fg_proto)
            if bg_proto is not None:
                bg_list.append(bg_proto)

        if fg_list:
            fg_new = torch.stack(fg_list, dim=0)
            self.proto_fg = fg_new if self.proto_fg is None else torch.cat([self.proto_fg, fg_new], dim=0)

        if bg_list:
            bg_new = torch.stack(bg_list, dim=0)
            self.proto_bg = bg_new if self.proto_bg is None else torch.cat([self.proto_bg, bg_new], dim=0)

    def retrieve(self, p3, topk=16, temperature=0.05):
        # p3: [B, C, h, w]
        B, C, h, w = p3.shape
        q = F.normalize(p3.flatten(2).transpose(1, 2), dim=-1)  # [B, HW, C]

        def _sim_to_fu(query, proto):
            if proto is None or proto.numel() == 0:
                sim = query.new_zeros(B, h * w, 1)
                fu = query.new_zeros(B, 1, h, w)
                return sim, fu

            proto = F.normalize(proto, dim=-1)                  # [N, C]
            sim = torch.matmul(query, proto.t())                # [B, HW, N]
            k = min(topk, proto.shape[0])
            top_vals = torch.topk(sim, k=k, dim=-1).values      # [B, HW, k]
            attn = torch.softmax(top_vals / temperature, dim=-1)
            fu = (attn * top_vals).sum(dim=-1, keepdim=True)    # [B, HW, 1]
            fu = fu.transpose(1, 2).reshape(B, 1, h, w)
            return sim, fu

        sim_bg, fu_bg = _sim_to_fu(q, self.proto_bg)
        sim_fg, fu_fg = _sim_to_fu(q, self.proto_fg)

        return {
            "sim_bg": sim_bg,
            "sim_fg": sim_fg,
            "fu_bg": fu_bg,
            "fu_fg": fu_fg,
        }

    def state_dict(self):
        return {
            "proto_fg": self.proto_fg,
            "proto_bg": self.proto_bg,
        }

    def load_state_dict(self, state):
        self.proto_fg = state.get("proto_fg", None)
        self.proto_bg = state.get("proto_bg", None)
```

### 3. `M(x)` 与 `H(x)`
`BaseLine/Paired_Background_Guidance/prototype_interaction.py`
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

def kde_min_threshold(diff_map, fallback=0.0):
    # diff_map: [B, 1, h, w], detached before call
    theta_list = []
    for i in range(diff_map.shape[0]):
        arr = diff_map[i, 0].reshape(-1).detach().cpu().numpy()
        if arr.size < 8 or float(arr.max() - arr.min()) < 1e-8:
            theta_list.append(fallback)
            continue
        try:
            kde = gaussian_kde(arr)
            x_vals = np.linspace(arr.min(), arr.max(), 2048)
            kde_vals = kde(x_vals)
            minima, _ = find_peaks(-kde_vals)
            theta = x_vals[minima[0]] if len(minima) > 0 else fallback
        except Exception:
            theta = fallback
        theta_list.append(theta)
    theta = torch.tensor(theta_list, device=diff_map.device, dtype=diff_map.dtype)
    return theta.view(-1, 1, 1, 1)

class DynamicPrototypeInteraction(nn.Module):
    def __init__(self, in_channels, hidden_ratio=4, tau=0.07, theta_fallback=0.0):
        super().__init__()
        hidden = max(in_channels // hidden_ratio, 16)
        self.alpha_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1)
        )
        self.tau = tau
        self.theta_fallback = theta_fallback

    def forward(self, p3, fu_fg, fu_bg, image_hw):
        alpha = torch.sigmoid(self.alpha_head(p3))
        base_diff = (fu_fg - fu_bg).detach()
        theta = kde_min_threshold(base_diff, fallback=self.theta_fallback)
        m_low = torch.sigmoid((alpha * fu_fg - (1.0 - alpha) * fu_bg - theta) / self.tau)
        h_map = F.interpolate(m_low, size=image_hw, mode="bilinear", align_corners=True)
        return {
            "alpha": alpha,
            "theta": theta,
            "m_low": m_low,
            "H": h_map,
        }
```

### 4. 可学习 `μ`
`BaseLine/Paired_Background_Guidance/prototype_fusion.py`
```python
import math
import torch
import torch.nn as nn

class LearnableMuFusion(nn.Module):
    def __init__(self, mu_init=0.5, mu_min=0.0, mu_max=1.0):
        super().__init__()
        self.mu_min = mu_min
        self.mu_max = mu_max
        x = (mu_init - mu_min) / (mu_max - mu_min + 1e-8)
        x = min(max(x, 1e-4), 1 - 1e-4)
        self.raw_mu = nn.Parameter(torch.tensor(math.log(x / (1 - x)), dtype=torch.float32))

    def mu(self):
        return self.mu_min + (self.mu_max - self.mu_min) * torch.sigmoid(self.raw_mu)

    def fuse_labeled(self, y_hat, H):
        mu = self.mu()
        return torch.clamp(y_hat + (1.0 - mu) * H, 0.0, 1.0)

    def fuse_unlabeled(self, y_bar, H):
        mu = self.mu().detach()
        return torch.clamp(y_bar + (1.0 - mu) * H, 0.0, 1.0)
```

### 5. 模型输出 `p3`
`BaseLine/models/talnet.py`
```python
# ModelEMA
def forward(self, x, ema=False, return_features=False):
    net = self.teacher if ema else self.student
    return net(x, return_features=return_features)

# TalNet
def forward_ori(self, x, return_features=False):
    (x1, x2, x3, x4) = self.forward_enc(x)
    if self.config.squeeze_block:
        x4 = self.squeeze_module(x4)
    features = [x, x1, x2, x3, x4]
    if self.training and self.config.out_ref:
        features.append(laplacian(torch.mean(x, dim=1).unsqueeze(1), kernel_size=5))
    return self.decoder(features, return_features=return_features)

def forward(self, x, return_features=False):
    return self.forward_ori(x, return_features=return_features)
```

`Decoder.forward`
```python
def forward(self, features, return_features=False):
    ...
    p3 = self.decoder_block3(_p3)
    ...
    p1_out = self.conv_out1(_p1)

    outs = []
    if self.config.ms_supervision:
        outs.extend([m4, m3, m2])
    outs.append(p1_out)

    feature_dict = {
        "p3": p3,
        "main_logit": p1_out,
        "image_hw": x.shape[-2:],
    }

    if self.config.out_ref and self.training:
        base_ret = ([outs_gdt_pred, outs_gdt_label], outs)
    else:
        base_ret = outs

    if return_features:
        return base_ret, feature_dict
    return base_ret
```

### 6. 训练整合
在 `SemiSupervisedTrainer` 中新增：
```python
from Paired_Background_Guidance.prototype_bank import DynamicPrototypeBank
from Paired_Background_Guidance.prototype_interaction import DynamicPrototypeInteraction
from Paired_Background_Guidance.prototype_fusion import LearnableMuFusion
```

初始化：
```python
self.prototype_bank = DynamicPrototypeBank(min_pixels=config.prototype_min_pixels)
self.prototype_interaction = None   # lazy init by p3 channels
self.mu_fusion = LearnableMuFusion(
    mu_init=config.prototype_mu_init,
    mu_min=config.prototype_mu_min,
    mu_max=config.prototype_mu_max,
).to(self.device)
self.mu_optimizer = torch.optim.Adam(
    self.mu_fusion.parameters(),
    lr=config.prototype_mu_lr,
    weight_decay=config.prototype_mu_weight_decay,
)
```

懒初始化交互头：
```python
def _ensure_proto_modules(self, p3):
    if self.prototype_interaction is None:
        self.prototype_interaction = DynamicPrototypeInteraction(
            in_channels=p3.shape[1],
            hidden_ratio=self.config.prototype_alpha_hidden_ratio,
            tau=self.config.prototype_tau,
            theta_fallback=self.config.prototype_theta_fallback,
        ).to(self.device)
```

epoch 开始：
```python
self.prototype_bank.begin_epoch()
```

labeled batch 主流程：
```python
(sup_outs, sup_feat) = self.model(inputs_sup, return_features=True)
p3_sup = sup_feat["p3"]
main_logit_sup = sup_feat["main_logit"]
self._ensure_proto_modules(p3_sup)

# 1. 先用 labeled p3 + gt 建库
self.prototype_bank.append_from_labeled_batch(p3_sup.detach(), gts_sup.detach())

# 2. 再检索并生成 H
ret_sup = self.prototype_bank.retrieve(
    p3_sup,
    topk=self.config.prototype_topk,
    temperature=self.config.prototype_sim_temperature,
)
proto_sup = self.prototype_interaction(
    p3_sup, ret_sup["fu_fg"], ret_sup["fu_bg"], image_hw=gts_sup.shape[-2:]
)
H_sup = proto_sup["H"]

# 3. 主模型损失：baseline 主输出 + H(x)
student_prob_sup = main_logit_sup.sigmoid()
loss_sup_main = self.pix_loss([main_logit_sup], gts_sup)
loss_sup_h = self.prob_loss(H_sup, gts_sup) * self.config.prototype_loss_weight_h
loss_model = loss_sup_main + loss_sup_h
```

`μ` 单独监督：
```python
# 注意：只更新 mu，不更新 student / prototype interaction
self.mu_optimizer.zero_grad()

with torch.no_grad():
    y_hat = student_prob_sup.detach()
    H_detach = H_sup.detach()

y_hat_r = self.mu_fusion.fuse_labeled(y_hat, H_detach)
loss_mu = self.prob_loss(y_hat_r, gts_sup)

loss_mu.backward()
self.mu_optimizer.step()
```

无标签 batch：
```python
with torch.no_grad():
    tea_outs, tea_feat = self.model(inputs_unsup, ema=True, return_features=True)
    y_bar = tea_feat["main_logit"].sigmoid()

(stu_outs_u, stu_feat_u) = self.model(inputs_unsup, return_features=True)
p3_u = stu_feat_u["p3"]
main_logit_u = stu_feat_u["main_logit"]

ret_u = self.prototype_bank.retrieve(
    p3_u,
    topk=self.config.prototype_topk,
    temperature=self.config.prototype_sim_temperature,
)
proto_u = self.prototype_interaction(
    p3_u, ret_u["fu_fg"], ret_u["fu_bg"], image_hw=inputs_unsup.shape[-2:]
)
H_u = proto_u["H"]

y_bar_r = self.mu_fusion.fuse_unlabeled(y_bar, H_u)

student_prob_u = main_logit_u.sigmoid()
loss_unsup = self.prob_loss(student_prob_u, y_bar_r) * 0.1
```

总损失：
```python
self.model_optimizer.zero_grad()
loss = loss_model + loss_unsup
loss.backward()
self.model_optimizer.step()
```

### 7. 概率损失
`BaseLine/engine/loss.py`
```python
class ProbLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lambdas = config.lambdas_pix_last
        self.criterions = {
            "bce": nn.BCELoss(),
            "iou": IoULoss(),
            "ssim": SSIMLoss(),
        }

    def forward(self, probs, gt):
        loss = 0.0
        for name, criterion in self.criterions.items():
            if self.lambdas.get(name, 0) > 0:
                loss = loss + criterion(probs, gt) * self.lambdas[name]
        return loss
```

### 8. checkpoint / 推理 / evaluator
保存：
```python
model_dict = {
    "model": self.model.module.state_dict() if self.config.distributed_train else self.model.state_dict(),
    "optimizer": self.model_optimizer.state_dict(),
    "lr_scheduler": self.model_lr_scheduler.state_dict(),
    "epoch": epoch,
    "prototype_bank": self.prototype_bank.state_dict(),
    "mu_fusion": self.mu_fusion.state_dict(),
}
```

恢复：
```python
if "prototype_bank" in checkpoint:
    self.prototype_bank.load_state_dict(checkpoint["prototype_bank"])
if "mu_fusion" in checkpoint:
    self.mu_fusion.load_state_dict(checkpoint["mu_fusion"])
```

推理/evaluator 最终输出：
```python
with torch.no_grad():
    outs, feat = self.model(inputs, ema=ema, return_features=True)
    p3 = feat["p3"]
    main_prob = feat["main_logit"].sigmoid()

    ret = self.prototype_bank.retrieve(
        p3,
        topk=self.config.prototype_topk,
        temperature=self.config.prototype_sim_temperature,
    )
    proto = self.prototype_interaction(
        p3, ret["fu_fg"], ret["fu_bg"], image_hw=main_prob.shape[-2:]
    )
    pred_final = self.mu_fusion.fuse_labeled(main_prob, proto["H"])
```

如果 checkpoint 没有原型库：
- 用当前 `split=5%` 的 labeled indices 创建 labeled loader
- 用 checkpoint 模型前向一次 labeled 集
- 按训练同样的 `append_from_labeled_batch()` 逻辑在线重建库
- 再开始测试

## 实验设置
### 训练阶段
- Warm-up：
  - 仍然每个 epoch 建库
  - 不启用原型检索监督，不更新 `μ`
  - 只跑 baseline 的 supervised 分支
- Semi-supervised：
  - labeled 分支：`loss_sup_main + loss_sup_h`
  - `μ`：仅由 `loss_mu` 更新
  - unlabeled 分支：teacher 生成 `\bar y`，student `p3` 生成 `H(x)`，融合得到 `\bar y_r`

### 推荐超参数
- `prototype_topk = 16`
- `prototype_sim_temperature = 0.05`
- `prototype_tau = 0.07`
- `prototype_min_pixels = 16`
- `prototype_mu_init = 0.5`
- `prototype_mu_lr = 1e-3`
- `prototype_loss_weight_h = 0.3`

### 评价指标
沿用当前 `BaseLine` evaluator：
- `MAE`
- `maxFm`
- `wFmeasure`
- `Smeasure`
- `meanEm`
- `meanFm`

建议额外记录：
- `μ` 的 epoch 变化曲线
- 每个 epoch 的 `|P_fg| / |P_bg|`
- `H(x)` 与 `gt` 的单独监督损失
- `\hat y_r` 相比 `\hat y` 的增益

## 消融实验
- Baseline Mean-Teacher
- Baseline + 动态原型库，不加 `H` 监督
- Baseline + `H` 监督，不加 `μ`
- Baseline + `μ` 融合，但 `μ` 固定为 0.5
- Baseline + 完整方案（动态库 + `H` + 可学习 `μ`）
- 去掉 `KDE_min`，改成 `theta=0`
- 前景库 only / 背景库 only / 前背景双库

## 测试计划
- 形状测试：不同 backbone 下 `p3` 通道变化时，原型模块工作正常
- 空库测试：epoch 初始时无原型，`H(x)` 应退化为零图，不报错
- 动态规模测试：库长度随 labeled 图数增长，不固定，不跨 epoch 继承
- 反向传播测试：
  - `μ` 只在 `loss_mu.backward()` 后有梯度
  - unlabeled loss 不更新 `μ`
- 推理一致性测试：训练期在线 evaluator 与离线 evaluator 的最终输出路径一致
- 分布式测试：多卡原型拼接后所有 rank 库一致

## 假设与默认决策
- 原型库动态规模采用“每张 labeled 图每类 1 个原型”的最稳版本，不走像素级大库
- `μ` 是单个全局标量，不做 per-image 或 per-class 版本
- `\hat y_r` 和 `\bar y_r` 都在概率空间融合，不在 logit 空间融合
- `μ` 监督完全独立于主模型更新，严格对齐你要求的“仅通过 labeled data 单独更新 `μ`”
- 推理和 evaluator 使用纠正后的最终输出，而不是仅用 student/teacher 主输出
