# SAM 现有代码可复用点审计包：面向 SVB-PLR 接入

## 0. 背景与结论

当前 CBM-PFI 仓库中已经存在一套可运行方向明确的 SAM 伪标签细化代码，主要位于：

```text
SAM/protoSAMprompt/
utils/utils2.py
utils/sam_pseudo_logging.py
config/base/sam.py
```

重新上传 `utils/utils2.py` 后，之前 `SAM/protoSAMprompt/sam_refiner.py` 和 `SAM/protoSAMprompt/sam2_refiner.py` 的关键依赖已经补齐。因此，现有代码可以直接复用为 **SVB-PLR 第一版的 SAM refinement backend**：

```text
teacher p_final / coarse pseudo mask
  -> box / point / mask prompt
  -> SAM or SAM2
  -> refined pseudo label
  -> unlabeled student supervision
```

但当前代码还不能直接完整实现 SVB-PLR，因为它尚未使用 CBM retrieval evidence，也没有输出 `conf_ref`、没有 boundary-only fusion、没有 weak-to-strong 几何对齐。

推荐定位：

```text
现有 SAM 代码 = 可复用的 SAM 后端与 teacher-mask prompt refinement 基座
SVB-PLR 还需新增 = CBM 证据解析 + 边界限制融合 + confidence map + weighted loss + weak/strong 对齐
```

---

## 1. 相关文件总览

### 1.1 SAM 统一伪标签细化入口

```text
SAM/protoSAMprompt/train_pseudo_refiner.py
```

主要职责：

- 根据配置构建 SAM1 或 SAM2 refiner。
- 接收 `images` 与 `pseudo_probs`。
- 将 teacher pseudo probability 转成 coarse mask。
- 调用 SAM 后端细化。
- 将 SAM mask 与 teacher probability 做 conservative fusion。
- 提供 no-op fallback。

关键类与函数：

```python
class _NoOpPseudoLabelRefiner
class _BaseSamPseudoLabelRefiner
class Sam1PseudoLabelRefiner
class Sam2PseudoLabelRefiner
class SamPseudoLabelRefiner
def build_sam_pseudo_label_refiner(config, device, logger=None)
```

最适合直接接入 trainer 的入口：

```python
build_sam_pseudo_label_refiner(config, device, logger=None)
```

当前接口形态：

```python
refiner(images, pseudo_probs, epoch=None, step=None) -> refined_pseudo_probs
```

---

### 1.2 SAM1 prompt refinement backend

```text
SAM/protoSAMprompt/sam_refiner.py
```

关键函数：

```python
def sam_input_prepare(
    image,
    pred_masks,
    image_embeddings=None,
    resize_transform=None,
    use_point=True,
    use_box=True,
    use_mask=True,
    add_neg=True,
    margin=0.0,
    gamma=1.0,
    strength=15,
)
```

作用：

- 从 coarse mask 生成 SAM 输入字典。
- 组合 box prompt、point prompt、mask prompt。
- 调用 `utils/utils2.py` 中的 prompt 生成工具。

```python
def sam_refiner(
    image,
    coarse_masks,
    sam,
    resize_transform=None,
    use_point=True,
    use_box=True,
    use_mask=True,
    add_neg=True,
    iters=5,
    margin=0.0,
    gamma=4.0,
    strength=30,
    use_samhq=False,
    ddp=False,
    is_train=False,
    coarse_threshold=0.5,
)
```

作用：

- 将 coarse mask 二值化。
- 对图像做 SAM resize/preprocess。
- 提取 SAM image embedding。
- 迭代生成 prompt 并调用 `forward_with_image_embeddings`。
- 对 multimask 输出选择 predicted IoU 最高的 mask。
- 返回 refined binary mask 与 low-res logits。

可直接复用能力：

```text
teacher coarse mask -> box + positive/negative points + mask prompt -> SAM1 refined mask
```

---

### 1.3 SAM2 prompt refinement backend

```text
SAM/protoSAMprompt/sam2_refiner.py
```

关键类：

```python
class Sam2PromptRefiner
```

关键方法：

```python
def __call__(
    self,
    image,
    coarse_masks,
    use_point=True,
    use_box=True,
    use_mask=True,
    add_neg=True,
    iters=1,
    gamma=4.0,
    strength=30,
    coarse_threshold=0.5,
)
```

内部已实现：

- SAM2 模型构建与冻结：

```python
build_sam2(...)
SAM2ImagePredictor(...)
```

- coarse mask 处理：

```python
_prepare_masks(...)
```

- box prompt：

```python
_mask_to_box(...)
```

- mask prompt：

```python
_mask_to_logits(...)
```

- point prompt：

```python
_points_from_mask(...)
```

- bf16 autocast：

```python
_autocast_context(...)
```

可直接复用能力：

```text
teacher coarse mask -> box + point + mask prompt -> SAM2 predictor -> best mask
```

---

### 1.4 Prompt 工具函数

```text
utils/utils2.py
```

这是 SAM prompt 生成的底层依赖。重新上传后，SAM1/SAM2 refiner 的关键依赖已补齐。

#### prepare_image

```python
def prepare_image(image, transform, device)
```

作用：

- 使用 SAM 的 `ResizeLongestSide` 对 numpy image 做 resize。
- 转成 torch tensor。
- 从 HWC 转成 CHW。

复用点：

```text
SAM1 输入图像预处理，可直接保留。
```

#### extract_bboxes_expand

```python
def extract_bboxes_expand(image_embeddings, mask, margin=0, img_path=None)
```

作用：

- 从 binary mask 计算 bounding box。
- 可选利用 SAM image embedding 相似度决定是否扩框。
- 返回：

```python
boxes, box_masks, areas, expand_list
```

复用点：

```text
teacher pseudo mask -> box prompt
```

对 SVB-PLR 的价值：

- 第一版可直接用于 box prompt。
- 后续可把 CBM foreground/background evidence 融入 box 生成逻辑，例如用 `Y_ctx` 限制 box 扩张。

#### extract_points

```python
def extract_points(pred_masks, add_neg=True, use_mask=True, gamma=1.0)
```

作用：

- 对前景 mask 做 distance transform，取最中心点作为 positive point。
- 若 `add_neg=True`，在 bbox 内的背景区域取 negative point。
- 可选生成 Gaussian distance map，用于 mask prompt。
- 返回：

```python
point_coords, point_labels, gaus_dt
```

复用点：

```text
teacher pseudo mask -> positive / negative point prompts
```

对 SVB-PLR 的价值：

- 第一版可直接使用。
- 后续可替换为 CBM-guided point sampling：

```text
positive points: p_t high + CBM fg score high + context consistency high
negative points: p_t low + CBM bg_near score high
boundary points: B_query / U_map high uncertainty band
```

#### extract_mask

```python
def extract_mask(pred_masks, gaus_dt, target_size, is01, strength=15, device=0, expand_list=0)
```

作用：

- 将 binary mask 转成 SAM mask input。
- resize 到 SAM low-res mask prompt 尺寸。
- 使用 Gaussian distance map 平滑 mask prompt。
- 支持不同 `strength` 控制 mask prompt 强度。

复用点：

```text
teacher pseudo mask -> SAM mask prompt / low-res logits
```

对 SVB-PLR 的价值：

- 可以直接作为 teacher mask prompt。
- 后续可把 `refine_band` 乘到 mask prompt 或只在边界带内强化。

---

### 1.5 SAM refinement 日志

```text
utils/sam_pseudo_logging.py
```

关键类：

```python
class SamPseudoRefineLogger
```

可复用能力：

- 初始化日志：

```python
log_init(...)
```

- batch 级统计：

```python
new_batch_stats(...)
log_batch(...)
```

- teacher/SAM/fused mask 差异统计：

```python
mask_change_metrics(...)
```

对 SVB-PLR 的价值：

- 可直接复用为 debug 日志。
- 后续建议扩展记录：

```text
refine_band area
CBM agreement
SAM reliability
conf_ref mean
boundary-only changed ratio
```

---

### 1.6 SAM 配置

```text
config/base/sam.py
```

已有配置：

```python
use_sam_pseudo_refine = False
sam_pseudo_backend = "sam1"  # sam1 | sam2
sam_pseudo_checkpoint = "SAM/sam_hq_vit_h.pth"
sam_pseudo_model_type = "vit_h"
sam_pseudo_threshold = 0.5
sam_pseudo_fusion_alpha = 0.5
sam_pseudo_iters = 1
sam_pseudo_use_point = True
sam_pseudo_use_box = True
sam_pseudo_use_mask = True
sam_pseudo_add_neg = True
sam_pseudo_margin = 0.0
sam_pseudo_gamma = 4.0
sam_pseudo_strength = 30
sam_pseudo_log_enable = False
sam_pseudo_log_interval = 300

sam2_checkpoint = "SAM/sam2.1_hiera_large.pt"
sam2_model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_multimask_output = True
sam2_use_bfloat16 = True
```

注意：

当前 `config/mkcfg.py` 只合并：

```python
config/base/common.py
config/base/model.py
config/base/cbm.py
run_cfg
```

因此 `config/base/sam.py` 目前不会自动进入训练配置。若要训练时启用，需要修改配置加载器或在 run config 中显式重复 SAM 配置。

---

## 2. 当前代码最适合的复用方式

### 2.1 第一版：最小可跑 SAM pseudo refinement

目标：

```text
不动 CBM 主路径，不动 memory，不引入复杂 weak/strong。
只把当前 teacher pseudo label 用 SAM refine 后作为 unlabeled gt_replace。
```

当前无标签分支位置：

```text
engine/solver.py
SemiSupervisedTrainer.train_epoch
```

当前逻辑：

```python
inputs = unsup_batch[0].to(self.device)
with torch.no_grad():
    teacher_preds = self.model(inputs, ema=True, use_memory=use_memory)
    p_labels = teacher_preds[-1].sigmoid()

self._train_batch(
    unsup_batch,
    gt_replace=p_labels,
    loss_alpha=float(getattr(self.config, "cbm_unsup_loss_alpha", 0.1)),
    use_memory=use_memory,
    enable_cbm_loss=False,
    branch_name="Unsup",
)
```

最小接入后目标形态：

```python
inputs = unsup_batch[0].to(self.device)
with torch.no_grad():
    teacher_preds, aux_t = self.model(
        inputs,
        ema=True,
        use_memory=use_memory,
        return_aux=True,
    )
    p_t = aux_t["p_final"] if aux_t.get("p_final") is not None else teacher_preds[-1].sigmoid()
    p_labels = self.sam_pseudo_refiner(
        inputs,
        p_t,
        epoch=epoch,
        step=self.global_step,
    )

self._train_batch(
    unsup_batch,
    gt_replace=p_labels,
    loss_alpha=float(getattr(self.config, "cbm_unsup_loss_alpha", 0.1)),
    use_memory=use_memory,
    enable_cbm_loss=False,
    branch_name="Unsup",
)
```

这个版本可以最大化复用：

```text
build_sam_pseudo_label_refiner
Sam1PseudoLabelRefiner
Sam2PseudoLabelRefiner
sam_refiner
Sam2PromptRefiner
utils2 prompt functions
SamPseudoRefineLogger
```

---

### 2.2 第二版：加入 confidence map，但仍不做完整 CBM prompt

现有 `build_sam_pseudo_label_refiner` 只返回 refined pseudo probability，没有返回 `conf_ref`。

可以新增轻量 confidence：

```python
def binary_reliability(p):
    return torch.abs(p - 0.5) * 2.0
```

或者 teacher/SAM agreement：

```python
conf_ref = 1.0 - torch.abs(p_ref - p_t)
conf_ref = conf_ref.clamp(min=sam_min_reliability, max=1.0)
```

然后需要训练 loss 支持 pixel-wise weight。当前仓库只有：

```text
engine/loss.py::PixLoss.forward(scaled_preds, gt)
```

无 `weighted_seg_loss`。因此这一版需要新增或扩展 loss 接口。

---

### 2.3 第三版：CBM-aware SVB-PLR

当前 CBM `aux` 没有 `aux["retrieval"]` 嵌套结构，而是把 retrieval evidence 展开在顶层：

```python
aux_t = {
    "Y_map": ...,
    "Y_ctx": ...,
    "R_map": ...,
    "R_ctx": ...,
    "U_map": ...,
    "valid_map": ...,
    "cons_map": ...,
    "prob3": ...,
    "B_query": ...,
    "boundary_mask": ...,
    "z_mem3": ...,
    "gate3": ...,
    "p_final": ...,
    "p_main": ...,
}
```

因此 SVB-PLR 中的 `retrieval_aux` adapter 可以直接使用 `aux_t`：

```python
retrieval_aux = aux_t
```

或整理成文档中的结构：

```python
retrieval_aux = {
    "Y_ctx": aux_t.get("Y_ctx"),
    "U_map": aux_t.get("U_map"),
    "cons_map": aux_t.get("cons_map"),
    "gate3": aux_t.get("gate3"),
    "B3": aux_t.get("B_query"),
    "valid_map": aux_t.get("valid_map"),
}
```

CBM evidence 可用于：

```text
S_fg = Y_ctx[:, 0:1] + Y_ctx[:, 1:2]
S_bg = Y_ctx[:, 2:3] + Y_ctx[:, 3:4]
M_bd = Y_ctx[:, 1:2] - Y_ctx[:, 2:3]
U_map = retrieval uncertainty
cons_map = context consistency
gate3 = correction gate
B_query = predicted boundary map
```

然后构建：

```text
refine_band = normalize(B_query + uncertainty + image gradient) > threshold
```

最终只允许 SAM 在边界带内融合：

```python
beta = sam_beta_max * R_sam * refine_band
p_ref = (1.0 - beta) * p_t + beta * sam_mask
```

---

## 3. 当前代码与完整 SVB-PLR 的差距

### 3.1 没有 CBM-guided prompt generation

当前 prompt 主要来自 teacher coarse mask：

```text
coarse mask -> bbox
coarse mask -> positive / negative points
coarse mask -> mask prompt
```

还没有使用：

```text
Y_ctx
U_map
cons_map
gate3
B_query
valid_map
```

因此它是 teacher-mask-guided SAM refinement，不是完整 CBM-guided SVB-PLR。

### 3.2 没有 confidence map 输出

当前输出：

```python
refined_pseudo_probs
```

SVB-PLR 需要：

```python
p_ref, conf_ref, sam_aux
```

其中 `conf_ref` 用于无标签 loss 加权。

### 3.3 没有 boundary-only fusion

当前 `_BaseSamPseudoLabelRefiner.__call__` 中融合方式是全图：

```python
fused = alpha * sam_mask + (1.0 - alpha) * teacher_prob
```

SVB-PLR 更合适：

```python
beta = sam_beta_max * R_sam * refine_band
p_ref = (1.0 - beta) * p_t + beta * sam_mask
```

这样可以避免 SAM 修改 teacher 已经高置信的非边界区域。

### 3.4 没有 weak-to-strong 几何对齐

当前训练器中 unlabeled teacher 和 student 使用同一个 `unsup_batch[0]`。

没有：

```text
img_u_w
img_u_s
geom
apply_geom(pseudo_w, geom)
apply_geom(conf_w, geom)
```

因此当前最小版本只能做 same-view SAM pseudo refinement。

### 3.5 没有 weighted_seg_loss

当前无标签 loss 复用 `PixLoss`，只支持整体 scalar：

```python
loss_pix = self.pix_loss(scaled_preds, gt) * loss_alpha
```

SVB-PLR 需要 pixel-wise weight：

```python
weighted_seg_loss(pred, target, weight)
```

---

## 4. 推荐实现顺序

### Step 1：配置接入

目标：

```text
让 config/base/sam.py 真正被 Config 加载，或把必要 SAM 配置写进 run config。
```

当前风险：

```text
config/base/sam.py 存在，但 Config 默认不合并它。
```

### Step 2：Trainer 初始化 SAM refiner

目标：

```python
from SAM.protoSAMprompt.train_pseudo_refiner import build_sam_pseudo_label_refiner

self.sam_pseudo_refiner = build_sam_pseudo_label_refiner(
    self.config,
    self.device,
    logger=self.logger,
)
```

建议位置：

```text
engine/solver.py::SemiSupervisedTrainer.__init__
```

### Step 3：无标签 teacher pseudo label 后处理

建议位置：

```text
engine/solver.py::SemiSupervisedTrainer.train_epoch
teacher_preds = self.model(inputs, ema=True, use_memory=use_memory)
p_labels = teacher_preds[-1].sigmoid()
```

第一版替换为：

```python
teacher_preds, aux_t = self.model(
    inputs,
    ema=True,
    use_memory=use_memory,
    return_aux=True,
)
p_t = aux_t["p_final"] if aux_t.get("p_final") is not None else teacher_preds[-1].sigmoid()
p_labels = self.sam_pseudo_refiner(inputs, p_t, epoch=epoch, step=self.global_step)
```

### Step 4：从 teacher-mask refinement 升级为 CBM-aware refinement

新增 adapter：

```python
def build_retrieval_aux_from_cbm_aux(aux_t):
    return {
        "Y_ctx": aux_t.get("Y_ctx"),
        "U_map": aux_t.get("U_map"),
        "cons_map": aux_t.get("cons_map"),
        "gate3": aux_t.get("gate3"),
        "B3": aux_t.get("B_query"),
        "valid_map": aux_t.get("valid_map"),
    }
```

新增 `refine_band`：

```python
refine_band = f(B3, U_map, image_gradient, cons_map)
```

将原全图融合替换为边界带融合：

```python
p_ref = p_t * (1 - beta) + sam_mask * beta
```

### Step 5：增加 confidence-weighted unlabeled loss

新增：

```python
conf_ref = binary_reliability(p_ref)
```

或：

```python
conf_ref = teacher_sam_agreement * cbm_agreement * stability
```

再实现：

```python
weighted_seg_loss(pred, target, weight)
```

---

## 5. 最小可复用代码路径清单

### 必须复用

```text
SAM/protoSAMprompt/train_pseudo_refiner.py
SAM/protoSAMprompt/sam_refiner.py
SAM/protoSAMprompt/sam2_refiner.py
utils/utils2.py
utils/sam_pseudo_logging.py
```

### 可复用配置

```text
config/base/sam.py
```

但需要确认它被 `Config` 合并。

### 需要读取的 CBM aux 来源

```text
models/talnet.py::TalNet.forward_cbm_pfi
CBM/engine.py::CBMPFIEngine.apply_p3_hook
CBM/engine.py::CBMPFIEngine.apply_final_fusion
CBM/core/outputs.py::build_used_aux
```

### 训练接入位置

```text
engine/solver.py::SemiSupervisedTrainer.__init__
engine/solver.py::SemiSupervisedTrainer.train_epoch
engine/solver.py::SemiSupervisedTrainer._train_batch
engine/loss.py::PixLoss.forward
```

---

## 6. 最终判断

`utils/utils2.py` 补齐后，现有 SAM 代码不再只是零散实验代码，而是可以作为 SVB-PLR 的第一版工程基座：

```text
可直接用：
teacher p_final -> SAM prompt refinement -> refined pseudo label

需要新增：
CBM evidence -> prompt/filter/refine_band/confidence -> weighted unlabeled loss
```

建议不要一开始就重写 SAM 调用。更稳妥的做法是：

```text
1. 复用 build_sam_pseudo_label_refiner 跑通 same-view SAM refine；
2. 确认训练稳定与显存开销；
3. 再逐步加入 CBM-guided refine_band；
4. 最后加入 conf_ref 与 weighted_seg_loss。
```

这样可以把风险拆开：

```text
SAM 后端可用性
伪标签变化是否合理
CBM evidence 是否能有效约束 SAM
confidence-weighted loss 是否提升稳定性
```

