# SV-UME 可复用代码点审计：CBM-PFI + SVB-PLR

> 目标文档：`C:\Users\UserY\Downloads\SV_UME_SAM_refined_Unlabeled_Memory_Expansion_CBM_PFI.md`
>
> 当前仓库根目录：`C:\Users\UserY\Desktop\BaseLine\CBM-PFI`
>
> 本文用于上传给 GPT 继续分析 SV-UME（SAM-refined Unlabeled Memory Expansion）实现方案。重点是梳理当前 CBM 模块与 SVB-PLR 已实现点、可复用位置、需要新增的位置，以及应避免破坏的训练行为。

## 1. 结论摘要

当前代码已经具备 SV-UME 的两个关键前置基础：

1. **CBM-PFI 的 labeled dense boundary memory 主链路已经存在**：
   - labeled memory 构建；
   - 四类 region token：`fg_core / fg_boundary / bg_near / bg_far`；
   - image-level router；
   - pointwise dense retrieval；
   - contextual aggregation；
   - p3 correction；
   - final logit fusion；
   - teacher forward 返回 `p_final` 与 retrieval evidence。

2. **SVB-PLR 的 SAM-refined pseudo-label 主链路已经基本实现**：
   - 从 CBM aux 整理 retrieval evidence；
   - CBM-guided hybrid prompt generation；
   - 复用现有 SAM1/SAM2 backend；
   - prompt expert selector；
   - SAM-CBM reliability filter；
   - conformal calibrator；
   - cache；
   - visualizer；
   - trainer 中无标签分支接入；
   - pixel-wise confidence weighted unsup loss。

但 **SV-UME 自身的 unlabeled memory expansion 尚未实现**。当前还缺少：

- unlabeled candidate token builder；
- matched auxiliary unlabeled memory；
- novel auxiliary unlabeled pool；
- hard ambiguous pool；
- core/aux separated retriever；
- aux memory gated fusion；
- Stage 3b trainer 写入逻辑；
- SV-UME 专用 losses；
- SV-UME memory logging / visualization / checkpoint state。

因此，SV-UME 最合适的实现策略是：**复用 SVB-PLR 生成的 `p_ref/conf_ref/sam_aux` 作为 unlabeled memory candidate 的输入，复用 CBM memory/retrieval/fusion 的接口模式，但新增独立 aux memory，不污染当前 labeled core memory。**

## 2. SV-UME 文档需求拆解

SV-UME 文档中的核心目标可以概括为：

1. 使用 SAM-refined pseudo label 作为 unlabeled memory expansion 的候选监督。
2. 不将 unlabeled tokens 直接混入 labeled core memory，避免污染可靠 labeled memory。
3. 将 unlabeled memory 分为：
   - `core_labeled`：原始 labeled memory；
   - `matched_aux_unlabeled`：与 labeled prototypes/regions 高一致的 unlabeled tokens；
   - `novel_aux_unlabeled`：疑似新模式，先进入 pending pool；
   - `hard_ambiguous`：高不确定或冲突样本，只做诊断或 hard mining，不直接参与强融合。
4. 分阶段训练：
   - Stage 3a：epoch 16-20，只使用 SVB-PLR refined pseudo label 训练 student，不扩展 memory；
   - Stage 3b：epoch 21-30，启用 SV-UME，筛选可靠 unlabeled tokens 加入 aux memory；
   - 后续阶段可逐步提高 aux memory 权重。
5. Candidate reliability 需要结合：
   - SAM-CBM reliability；
   - teacher agreement；
   - CBM retrieval evidence；
   - boundary/refine band；
   - global/region/token consistency；
   - conformal calibration。

## 3. 当前 CBM-PFI 已实现点

### 3.1 主模型入口

| 文件 | 符号 | 当前能力 |
|---|---|---|
| `models/talnet.py` | `Network.forward` | 支持 `ema=True/False`、`use_memory`、`return_aux`，teacher/student 共用封装入口。 |
| `models/talnet.py` | `Network.forward_cbm_pfi` | CBM-PFI 主 forward，调用 CBM hook，支持返回 `(scaled_preds, aux)`。 |
| `CBM/engine.py` | `CBMEngine.apply_p3_hook` | 在 p3 层执行 boundary query、memory retrieval、context aggregation、p3 correction。 |
| `CBM/engine.py` | `CBMEngine.apply_final_fusion` | 将 corrected p3 相关信息融合回 final prediction，生成 `p_final`。 |

### 3.2 DenseBoundaryMemory

| 文件 | 符号 | 当前能力 |
|---|---|---|
| `CBM/memory/bank.py` | `DenseBoundaryMemory` | labeled dense boundary memory 容器。保存 image keys、point keys、values、meta。 |
| `CBM/memory/bank.py` | `DenseBoundaryMemory.append_batch` | 从 labeled batch 的 `x3/p3/gt/img_ids/reliability` 构建 region tokens。 |
| `CBM/memory/bank.py` | `DenseBoundaryMemory.finalize` | 合并并截断 memory，按 region cap 控制容量。 |
| `CBM/memory/bank.py` | `DenseBoundaryMemory.get_image_keys` | 提供 image-level keys 给 global router。 |
| `CBM/memory/bank.py` | `DenseBoundaryMemory.get_sub_memory` | 根据 router 选中 image ids 取出子 memory。 |
| `CBM/memory/bank.py` | `DenseBoundaryMemory.to_state_dict/load_state_dict` | 支持 checkpoint 保存/恢复 memory。 |
| `CBM/memory/builder.py` | `LabeledMemoryBuilder` | labeled memory rebuild 的训练期构建器。 |

### 3.3 Region/value 语义

| 文件 | 符号 | 当前能力 |
|---|---|---|
| `CBM/memory/labels.py` | `REGION_NAMES` | 已定义四类 region：`fg_core, fg_boundary, bg_near, bg_far`。 |
| `CBM/memory/labels.py` | `VALUE_LAYOUT` | value 含 region one-hot、`bg/fg/sdf/reliability`。 |
| `CBM/memory/labels.py` | `build_gt_regions` | 基于 GT 构建四类 region mask。 |
| `CBM/memory/labels.py` | `sample_tokens_from_region` | 从 region mask 中采样 token 坐标。 |

这部分和 SV-UME 高度对齐。SV-UME 的候选 token 也应沿用这四类 region 语义，只是 `gt` 来源从 labeled GT 换成 `p_ref/conf_ref/refine_band`。

### 3.4 Retrieval 与 correction

| 文件 | 符号 | 当前能力 |
|---|---|---|
| `CBM/retrieval/global_router.py` | `GlobalMemoryRouter` | 根据 query image key 选择相似 memory image ids。 |
| `CBM/retrieval/pointwise.py` | `PointwiseBoundaryRetriever` | 对 boundary/query positions 做 dense token retrieval，输出 `Y_map/R_map/U_map/valid_map`。 |
| `CBM/context/aggregator.py` | `ContextualBoundaryAggregator` | 将 retrieval map 聚合为 contextual evidence：`Y_ctx/R_ctx/cons_map`。 |
| `CBM/correction/p3_correction.py` | `BoundaryCorrectionHead` | 用 CBM evidence 修正 p3 logits，输出 `gate3` 等。 |
| `CBM/correction/logit_fusion.py` | `BoundaryLogitFusion` | 将 corrected p3 influence 融合到 final logit。 |

SV-UME 的 separated retrieval 可以复用这些类的思想，但建议新增包装层，分别计算 core memory 与 aux memory 的 retrieval evidence，再做可靠性门控融合。

### 3.5 Teacher aux / retrieval evidence

| 文件 | 符号 | 当前能力 |
|---|---|---|
| `CBM/core/outputs.py` | `build_used_aux` | 返回 `B_query/gate3/Y_map/Y_ctx/R_map/R_ctx/U_map/valid_map/cons_map/prob3` 等 evidence。 |
| `CBM/core/outputs.py` | `build_fallback_aux` | memory 不可用时返回 fallback aux。 |
| `models/talnet.py` | `_make_fallback_aux` | 主模型 fallback aux，包含 `p_final/p_main/B_query/gate3` 等。 |

SV-UME 可以直接使用这些 evidence 做 candidate scoring：

- `p_final`：teacher soft pseudo-label 初始来源；
- `Y_ctx`：region support；
- `R_ctx`：retrieval reliability；
- `U_map`：retrieval uncertainty；
- `valid_map`：有效 retrieval 区域；
- `cons_map`：context consistency；
- `B_query/B3`：boundary query/boundary evidence；
- `gate3`：CBM correction gate；
- `prob3`：低层 prediction prior。

## 4. 当前 SVB-PLR 已实现点

### 4.1 配置

| 文件 | 符号 | 当前能力 |
|---|---|---|
| `config/base/sam.py` | SVB-PLR config | 已包含 `use_svb_plr`、SAM backend、prompt、boundary band、reliability、fusion、cache、visualization、loss、ablation mode。 |
| `config/mkcfg.py` | base config loading | 已加载 `config/base/sam.py`。 |

注意：`use_svb_plr=False` 是总开关默认值，用于保证 baseline 不变。

### 4.2 CBM aux adapter

| 文件 | 符号 | 当前能力 |
|---|---|---|
| `SAM/SAM_refinement/cbm_aux_adapter.py` | `build_retrieval_aux_from_cbm_aux` | 将 CBM aux 统一整理成 retrieval_aux。兼容 `aux["retrieval"]` 与顶层展开 key。 |
| `SAM/SAM_refinement/cbm_aux_adapter.py` | `validate_retrieval_aux` | 检查必要 evidence 是否存在。 |
| `SAM/SAM_refinement/cbm_aux_adapter.py` | `has_valid_cbm_evidence` | 判断是否存在有效 CBM evidence。 |

SV-UME 可直接复用该 adapter，避免在 trainer 里硬编码 aux key。

### 4.3 SVB utility functions

| 文件 | 符号 | 当前能力 |
|---|---|---|
| `SAM/SAM_refinement/svb_utils.py` | `normalize_01` | 归一化到 0-1。 |
| `SAM/SAM_refinement/svb_utils.py` | `image_gradient_magnitude` | 计算 image/prob map 梯度幅值。 |
| `SAM/SAM_refinement/svb_utils.py` | `binary_reliability` | 从 binary probability 估计 confidence。 |
| `SAM/SAM_refinement/svb_utils.py` | `soft_iou` | soft mask IoU。 |
| `SAM/SAM_refinement/svb_utils.py` | `soft_boundary_alignment` | mask 与 refine band 边界对齐度。 |
| `SAM/SAM_refinement/svb_utils.py` | `sample_topk_points` | 从 score/mask 采样 top-k prompt 点。 |
| `SAM/SAM_refinement/svb_utils.py` | `compute_connected_component_boxes` | 从 mask 生成 component boxes。 |

SV-UME candidate scoring、boundary candidate generation、spatial diversity sampling 可以继续复用这些工具。

### 4.4 CBM-guided prompt generator

| 文件 | 符号 | 当前能力 |
|---|---|---|
| `SAM/SAM_refinement/cbm_prompt_generator.py` | `CBMPromptGenerator` | 从 `teacher_prob + retrieval_aux` 生成 SAM prompts 与 `refine_band/evidence`。 |
| `SAM/SAM_refinement/cbm_prompt_generator.py` | `parse_cbm_evidence` | 解析 `Y_ctx/U_map/cons_map/gate3/B3/valid_map` 并 resize 到 teacher 空间。 |
| `SAM/SAM_refinement/cbm_prompt_generator.py` | `build_refinement_band` | 结合 boundary、uncertainty、gradient、consistency、gate 生成 refine band。 |

SV-UME 可复用：

- `refine_band` 作为 unlabeled boundary candidate 区域；
- `evidence` 中的 `S_fg_up/S_bg_up/M_bd_up/U_up/cons_up/gate_up/B3_up/valid_up` 作为 candidate token reliability 的输入；
- positive/negative/boundary points 作为 SAM prompt 诊断与可视化。

### 4.5 SAM backend adapter

| 文件 | 符号 | 当前能力 |
|---|---|---|
| `SAM/SAM_refinement/sam_backend_adapter.py` | `ExistingSAMBackendAdapter` | 封装现有 `SAM/protoSAMprompt` 的 SAM1/SAM2 backend，统一 `predict` 输出。 |
| `SAM/protoSAMprompt/sam_refiner.py` | `Sam1PseudoLabelRefiner` | SAM1 pseudo label refinement 基础实现。 |
| `SAM/protoSAMprompt/sam2_refiner.py` | `Sam2PseudoLabelRefiner` | SAM2 pseudo label refinement 基础实现。 |
| `SAM/protoSAMprompt/train_pseudo_refiner.py` | `build_sam_pseudo_label_refiner` | 旧 SAM pseudo refiner 构建入口。 |

SV-UME 不需要重新写 SAM 调用。应该继续通过 `SAMVerifiedBoundaryPseudoLabelRefinement.refine` 间接调用 SAM。

### 4.6 Prompt expert selector

| 文件 | 符号 | 当前能力 |
|---|---|---|
| `SAM/SAM_refinement/prompt_expert_selector.py` | `PromptExpertSelector` | 支持 `box`、`box_point`、`mask`、`boundary` 四类 expert prompts。 |
| `SAM/SAM_refinement/prompt_expert_selector.py` | `select` | 根据 teacher IoU、boundary alignment、CBM agreement、over-seg penalty 与 SAM score 选择最佳 mask。 |

SV-UME 可复用 selector 的 `selector_aux/expert_scores` 作为 candidate reliability 的一个来源。例如：

- prompt experts 一致时，candidate 更可靠；
- expert 分歧大时，candidate 进入 hard ambiguous pool；
- boundary expert 高分可提升 `fg_boundary/bg_near` token 置信度。

### 4.7 SAM-CBM reliability filter

| 文件 | 符号 | 当前能力 |
|---|---|---|
| `SAM/SAM_refinement/sam_reliability_filter.py` | `SAMCBMReliabilityFilter` | 融合 teacher agreement、CBM agreement、SAM stability、conformal reliability，输出 `p_ref/conf_ref/filter_aux`。 |

SV-UME 最核心可复用输出：

- `p_ref`：soft refined pseudo-label；
- `conf_ref`：pixel-wise confidence；
- `R_sam`：SAM-CBM 综合可靠性；
- `beta`：融合强度；
- `refine_band`：主要变化区域；
- `fg_support/bg_support`：CBM foreground/background support。

这些可以直接用于 unlabeled memory candidate 筛选。

### 4.8 Conformal calibrator

| 文件 | 符号 | 当前能力 |
|---|---|---|
| `SAM/SAM_refinement/conformal_sam_calibrator.py` | `ConformalSAMCalibrator` | 在 labeled set 上估计 SAM boundary nonconformity 分布，提供 `estimate_reliability`。 |

SV-UME 可复用 conformal reliability 作为是否允许 unlabeled token 入 aux memory 的保守门控。

### 4.9 SVB-PLR 总控

| 文件 | 符号 | 当前能力 |
|---|---|---|
| `SAM/SAM_refinement/svb_plr.py` | `SAMVerifiedBoundaryPseudoLabelRefinement` | 总控：prompt generation -> SAM backend -> prompt selection -> reliability filter -> cache/visualization。 |
| `SAM/SAM_refinement/svb_plr.py` | `refine` | 输入 `images/teacher_prob/retrieval_aux`，输出 `p_ref/conf_ref/sam_aux`。 |

SV-UME 应优先复用 `refine` 输出，而不是重新拼装 SAM pipeline。

### 4.10 Cache 与 visualizer

| 文件 | 符号 | 当前能力 |
|---|---|---|
| `SAM/SAM_refinement/svb_cache.py` | `SVBPLRCache` | 基于 `image_id + epoch + backend + prompt_mode + teacher hash` 缓存 `p_ref/conf_ref/sam_mask/refine_band/R_sam/beta`。 |
| `SAM/SAM_refinement/sam_refine_visualizer.py` | `SamRefineVisualizer` | 保存 3x4 panel，可视化 teacher、SAM mask、p_ref、evidence、points、R_sam、conf_ref、difference。 |

SV-UME 可以扩展 visualizer，新增：

- selected unlabeled tokens overlay；
- matched/novel/hard pool 标记；
- aux memory retrieval map；
- core vs aux retrieval contribution map。

### 4.11 Weighted unsup loss 与 trainer 接入

| 文件 | 符号 | 当前能力 |
|---|---|---|
| `engine/loss.py` | `weighted_seg_loss` | 支持 soft target + pixel-wise weight 的 BCE + IoU/structure-style weighted loss。 |
| `engine/solver.py` | `_train_batch(... gt_replace_conf=None ...)` | 无标签分支可使用 `conf_ref` 作为 pixel-wise loss weight。 |
| `engine/solver.py` | `SemiSupervisedTrainer._init_svb_plr` | 当 `use_svb_plr=True` 时初始化 SVB-PLR。 |
| `engine/solver.py` | `SemiSupervisedTrainer._prepare_svb_epoch` | conformal fit 的 epoch hook。 |
| `engine/solver.py` | unsup teacher branch | teacher 生成 aux，调用 SVB-PLR refine，得到 `p_ref/conf_ref`。 |
| `engine/solver.py` | `_record_svb_aux` | 记录 `used_sam/refine_band/beta/R_sam/conf_ref/changed_ratio` 等日志。 |

SV-UME 的 Stage 3a 基本已经具备。Stage 3b 需要在同一无标签分支中增加 candidate extraction 与 aux memory update。

## 5. SV-UME 可复用映射表

| SV-UME 需求 | 当前可复用模块 | 复用方式 | 仍需新增 |
|---|---|---|---|
| SAM-refined pseudo-label | `SAMVerifiedBoundaryPseudoLabelRefinement.refine` | 直接获得 `p_ref/conf_ref/sam_aux` | 无 |
| CBM evidence 读取 | `build_retrieval_aux_from_cbm_aux` | 从 teacher aux 整理 retrieval evidence | 无 |
| Boundary/refine band | `CBMPromptGenerator.build_refinement_band` | `sam_aux["prompt_pack"]["refine_band"]` | 无 |
| Reliability map | `SAMCBMReliabilityFilter` | 使用 `conf_ref/R_sam/beta` | 需要转为 token-level score |
| Candidate region 定义 | `CBM/memory/labels.py` | 沿用四类 region 语义 | 需要 soft pseudo-label 版本 region builder |
| Labeled memory 构建模式 | `LabeledMemoryBuilder` + `DenseBoundaryMemory.append_batch` | 参考接口和 value layout | 新增 unlabeled candidate builder |
| Core memory 容器 | `DenseBoundaryMemory` | 保持 labeled core 不变 | 新增 aux memory 容器 |
| Global matching | `GlobalMemoryRouter` | 判断 unlabeled image 与 labeled core 是否匹配 | 新增 matched/novel 分类逻辑 |
| Dense retrieval | `PointwiseBoundaryRetriever` | 可复用 retrieval 计算 | 新增 core/aux separated retriever |
| Context fusion | `ContextualBoundaryAggregator` | 可复用 context aggregation | 新增 core/aux evidence merge |
| Correction/fusion | `BoundaryCorrectionHead/BoundaryLogitFusion` | 可复用最终融合模式 | 新增 aux gated fusion 权重 |
| Loss 权重 | `weighted_seg_loss` | 继续使用 `conf_ref` 降低低可靠区域 loss | 需要 aux memory regularization losses |
| Cache/visualization | `SVBPLRCache/SamRefineVisualizer` | 可扩展诊断 | 新增 SV-UME memory token 可视化 |

## 6. 建议新增模块

### 6.1 `CBM/memory/unlabeled_candidate_builder.py`

建议新增类：

```python
class UnlabeledMemoryCandidateBuilder(nn.Module):
    @torch.no_grad()
    def build_candidates(
        self,
        x3,
        p3,
        p_ref,
        conf_ref,
        sam_aux,
        retrieval_aux,
        image_ids,
        epoch=None,
        step=None,
    ):
        ...
```

职责：

1. 将 `p_ref/conf_ref/refine_band/R_sam/beta/evidence` resize 到 p3 空间。
2. 基于 soft pseudo-label 生成四类 candidate masks：
   - `fg_core`：`p_ref` 高、`conf_ref` 高、远离 boundary；
   - `fg_boundary`：`p_ref` 高或中等、处于 refine band/boundary band；
   - `bg_near`：背景但靠近 boundary；
   - `bg_far`：稳定远背景；
3. 计算 token reliability：
   - pixel confidence；
   - SAM-CBM reliability；
   - teacher/SAM agreement；
   - CBM evidence consistency；
   - region-level agreement；
   - optional conformal score；
4. 按 region 采样，并保留 metadata：
   - `image_id`;
   - `region_id`;
   - `xy`;
   - `reliability`;
   - `source="unlabeled_sam_refined"`;
   - `epoch/step`;
   - `pool_type` placeholder。

### 6.2 `CBM/memory/expanded_bank.py`

建议新增类：

```python
class AuxiliaryDenseBoundaryMemory(nn.Module):
    def append_candidates(self, candidates): ...
    def finalize(self): ...
    def get_aux_memory(self, pool="matched"): ...
    def to_state_dict(self): ...
    def load_state_dict(self, state): ...
```

或新增总控：

```python
class ExpandedDenseBoundaryMemory(nn.Module):
    def __init__(self, core_memory, cfg):
        self.core_labeled = core_memory
        self.matched_aux_unlabeled = AuxiliaryDenseBoundaryMemory(...)
        self.novel_aux_unlabeled = PendingNovelMemory(...)
        self.hard_ambiguous = HardAmbiguousPool(...)
```

硬性原则：

- `core_labeled` 不被 unlabeled token 修改；
- aux memory 独立容量控制；
- checkpoint 中明确区分 core/aux；
- 默认关闭时完全不改变当前 CBM retrieval。

### 6.3 `CBM/memory/unlabeled_consistency.py`

建议新增函数：

```python
compute_global_consistency(...)
compute_region_consistency(...)
compute_token_consistency(...)
assign_unlabeled_pool(...)
```

用途：

- `global_consistency`：unlabeled image key 与 labeled core image keys 的相似度；
- `region_consistency`：candidate region 与 core region prototype 的一致性；
- `token_consistency`：candidate key 与 retrieved core token 的一致性；
- `assign_unlabeled_pool`：
  - 高 global/region/token consistency -> `matched_aux_unlabeled`;
  - 中高可靠但与 core 不匹配 -> `novel_aux_unlabeled`;
  - 高冲突/高不确定 -> `hard_ambiguous`;
  - 低可靠 -> discard。

### 6.4 `CBM/retrieval/separated_retriever.py`

建议新增类：

```python
class SeparatedCoreAuxRetriever(nn.Module):
    def forward(
        self,
        p3,
        B_query,
        boundary_mask,
        core_memory,
        aux_memory=None,
        ...
    ):
        return {
            "core": core_aux,
            "aux": aux_aux,
            "merged": merged_aux,
            "aux_gate": aux_gate,
        }
```

融合建议：

```text
Y_merged = Y_core + gamma_aux * aux_gate * Y_aux
R_merged = max_or_weighted(R_core, R_aux)
U_merged = uncertainty_aware_merge(U_core, U_aux)
valid_merged = valid_core OR valid_aux
```

`gamma_aux` 应使用 schedule，从 0 逐步增大，避免早期 aux memory 干扰。

### 6.5 `CBM/losses/aux_memory_losses.py`

建议新增：

- `aux_memory_consistency_loss`：student prediction 与 aux retrieval evidence 一致；
- `anchor_core_distillation_loss`：aux retrieval 不应破坏 core memory prediction；
- `novel_pool_separation_loss`：novel pool 与 core prototypes 保持可分；
- `hard_ambiguous_contrast_loss`：仅用于 hard mining 或诊断，默认可关闭。

### 6.6 Trainer 接入点

建议修改 `engine/solver.py`，但必须受新开关保护：

```python
if cfg.use_sv_ume and epoch >= cfg.sv_ume_start_epoch:
    candidates = self.sv_ume_candidate_builder.build_candidates(
        x3=teacher_or_student_features,
        p3=teacher_p3,
        p_ref=p_ref,
        conf_ref=conf_ref,
        sam_aux=sam_aux,
        retrieval_aux=retrieval_aux,
        image_ids=image_ids,
        epoch=epoch,
        step=self.global_step,
    )
    self.aux_memory.append_candidates(candidates)
```

推荐位置：

1. teacher weak view 生成 `p_ref/conf_ref/sam_aux` 之后；
2. weak-to-strong 对齐之前或之后均可，但 memory candidate 应优先使用 teacher weak view 原始空间；
3. student unsup loss 之前完成候选提取；
4. 不要让 `p_ref/conf_ref` 参与反向传播；
5. 不要将 refined pseudo-label 写入 labeled core memory。

## 7. 建议新增配置

建议新增 `config/base/sv_ume.py`，并在 `config/mkcfg.py` 中加载。也可以临时放到 `config/base/sam.py`，但长期建议独立文件。

```python
# SV-UME main
use_sv_ume = False
sv_ume_start_epoch = 21
sv_ume_end_epoch = 30
sv_ume_update_interval = 1

# stage policy
sv_ume_require_svb_plr = True
sv_ume_use_core_labeled_anchor = True
sv_ume_do_not_update_core_memory = True

# candidate thresholds
sv_ume_min_pixel_conf = 0.65
sv_ume_min_r_sam = 0.55
sv_ume_min_beta = 0.05
sv_ume_min_global_consistency = 0.35
sv_ume_min_region_consistency = 0.45
sv_ume_min_token_consistency = 0.45

# region sampling
sv_ume_tokens_per_image_fg_core = 64
sv_ume_tokens_per_image_fg_boundary = 128
sv_ume_tokens_per_image_bg_near = 128
sv_ume_tokens_per_image_bg_far = 64
sv_ume_spatial_nms_radius = 3

# aux memory capacity
sv_ume_max_matched_aux_tokens = 32768
sv_ume_max_novel_aux_tokens = 8192
sv_ume_max_hard_ambiguous_tokens = 8192

# aux retrieval fusion
sv_ume_use_aux_retrieval = True
sv_ume_aux_gamma_start = 0.0
sv_ume_aux_gamma_end = 0.5
sv_ume_aux_gamma_warmup_epochs = 5
sv_ume_aux_gate_by_reliability = True

# loss
sv_ume_use_aux_memory_loss = False
sv_ume_lambda_aux_mem = 0.05
sv_ume_lambda_anchor = 0.05
sv_ume_lambda_novel = 0.01

# diagnostics
sv_ume_log_interval = 50
sv_ume_vis_interval = 500
sv_ume_save_candidate_debug = True
```

默认 `use_sv_ume=False`，确保当前训练行为不变。

## 8. 当前缺失清单

### 8.1 必须新增

1. `UnlabeledMemoryCandidateBuilder`
2. soft pseudo-label 四类 region builder
3. token-level reliability scorer
4. global/region/token consistency scorer
5. matched/novel/hard pool assignment
6. auxiliary unlabeled memory container
7. separated core/aux retriever
8. aux retrieval gated merge
9. Stage 3b trainer update hook
10. aux memory checkpoint save/load
11. SV-UME logging
12. SV-UME visualization

### 8.2 建议新增

1. aux memory losses
2. hard ambiguous mining strategy
3. candidate cache
4. pool aging/decay/pruning
5. per-image duplicate suppression
6. prompt expert disagreement score
7. SAM cache 与 SV-UME candidate cache 的关联 key

### 8.3 当前存在但需要谨慎使用

1. `DenseBoundaryMemory.append_batch` 当前假设输入是 labeled GT，不建议直接用 unlabeled pseudo-label 调它写 core memory。
2. 当前 `CBMEngine` 只持有一个 `self.memory`，不区分 core/aux。
3. 当前 retriever 输出没有区分 core evidence 与 aux evidence。
4. 当前 trainer 的 memory rebuild 仍面向 labeled loader。
5. 当前 SVB-PLR refine 输出已经 detach，不适合作为可学习分支，但适合作为 memory candidate source。
6. 当前 weak-to-strong 几何对齐如果 dataloader 没有 `geom`，会 fallback 到 same-view；SV-UME memory candidate 建议以 weak teacher view 为准。

## 9. 推荐实现路线

### Step 1：配置与 no-op 接入

- 新增 `config/base/sv_ume.py`，默认 `use_sv_ume=False`；
- `SemiSupervisedTrainer.__init__` 中在开关打开时初始化 SV-UME 组件；
- 开关关闭时保证完全 baseline。

### Step 2：Candidate builder

- 输入 SVB-PLR 的 `p_ref/conf_ref/sam_aux`；
- 生成四类 region candidate masks；
- 从 p3 feature 上采样/下采样对齐；
- 输出候选 token table。

### Step 3：Aux memory container

- 新增 aux memory，不改 `DenseBoundaryMemory` core 行为；
- checkpoint 显式保存 `core_labeled` 与 `aux_unlabeled`；
- 先只写 matched pool，novel/hard pool 可以只缓存不参与 retrieval。

### Step 4：Separated retrieval

- 先跑 core retrieval；
- 再可选跑 aux retrieval；
- 使用 reliability gate 与 gamma schedule 融合；
- aux 输出写入 aux keys，例如 `Y_aux_map/R_aux_map/U_aux_map/aux_gate`。

### Step 5：Trainer Stage 3b

- 在无标签 teacher branch 生成 `p_ref/conf_ref` 后调用 candidate builder；
- 将高可靠 candidates 写入 aux memory；
- 不更新 DenseBoundaryMemory core；
- 不让 SAM 或 pseudo-label 参与反向传播。

### Step 6：诊断与 ablation

建议 ablation：

1. Baseline CBM-PFI；
2. CBM-PFI + SVB-PLR loss only；
3. SV-UME matched aux only；
4. SV-UME matched + separated retrieval；
5. SV-UME matched + novel pending；
6. SV-UME full。

建议日志：

- `sv_ume_candidates_total`;
- `sv_ume_candidates_kept`;
- `sv_ume_matched_count`;
- `sv_ume_novel_count`;
- `sv_ume_hard_count`;
- `sv_ume_reject_count`;
- `sv_ume_aux_memory_size`;
- `sv_ume_aux_gamma`;
- `sv_ume_aux_gate_mean`;
- `sv_ume_core_aux_agreement`;
- `sv_ume_aux_retrieval_valid_ratio`。

## 10. 最小伪代码

```python
# teacher weak branch
teacher_preds, aux_t = model(
    img_u_w,
    ema=True,
    use_memory=use_memory,
    return_aux=True,
)

p_t = aux_t.get("p_final", teacher_preds[-1].sigmoid())
retrieval_aux = build_retrieval_aux_from_cbm_aux(aux_t)

# Stage 3a/3b: SAM-refined pseudo-label
p_ref, conf_ref, sam_aux = svb_plr.refine(
    images=img_u_w,
    teacher_prob=p_t,
    retrieval_aux=retrieval_aux,
    image_ids=image_ids,
    epoch=epoch,
    step=global_step,
)

# Stage 3b only: SV-UME candidate extraction
if cfg.use_sv_ume and epoch >= cfg.sv_ume_start_epoch:
    candidates = sv_ume_candidate_builder.build_candidates(
        x3=aux_t_or_feature["x3"],
        p3=aux_t_or_feature["p3"],
        p_ref=p_ref,
        conf_ref=conf_ref,
        sam_aux=sam_aux,
        retrieval_aux=retrieval_aux,
        image_ids=image_ids,
        epoch=epoch,
        step=global_step,
    )

    assigned = sv_ume_consistency.assign_pools(
        candidates=candidates,
        core_memory=model.cbm.memory,
        aux_memory=sv_ume_aux_memory,
    )

    sv_ume_aux_memory.append_candidates(assigned["matched_aux_unlabeled"])
    sv_ume_aux_memory.append_pending(assigned["novel_aux_unlabeled"])
    sv_ume_aux_memory.append_hard(assigned["hard_ambiguous"])

# student unsup loss remains SVB-PLR weighted pseudo-label loss
pseudo_s = apply_geom_or_same_view(p_ref)
conf_s = apply_geom_or_same_view(conf_ref)
_train_batch(
    unsup_batch,
    gt_replace=pseudo_s,
    gt_replace_conf=conf_s,
    branch_name="Unsup",
)
```

## 11. 风险与约束

1. **不要污染 labeled core memory**：SV-UME 必须新增 aux memory，而不是直接把 unlabeled tokens append 到 `DenseBoundaryMemory` core。
2. **不要让 SAM 参与反向传播**：继续保持 SVB-PLR 全链路 `torch.no_grad()`。
3. **不要用 refined pseudo-label 重建 labeled memory**：memory rebuild loader 仍应只用 labeled data。
4. **先保守启用 matched aux**：novel pool 可先只收集和可视化，不参与 retrieval。
5. **要记录丢弃样本**：拒绝的 candidates 对调阈值很关键。
6. **需要处理空 mask**：COD 中无标签 pseudo-label 可能为空或极小，candidate builder 必须安全返回空列表。
7. **需要防止同一图像重复写入**：aux memory key 应包含 `image_id/epoch/teacher hash`，或按 image_id 更新替换。
8. **需要容量控制与老化机制**：unlabeled memory 如果只增不减会引入噪声累积。

## 12. 推荐给 GPT 的分析问题

上传本文后，可以要求 GPT 重点分析：

1. 如何从 `p_ref/conf_ref/refine_band/R_sam/evidence` 构造最稳健的 four-region unlabeled candidates？
2. `matched_aux_unlabeled / novel_aux_unlabeled / hard_ambiguous` 的阈值和判别公式如何设计？
3. core/aux separated retrieval 应该如何融合，才能提升边界质量而不破坏 labeled memory？
4. SV-UME 的最小可实现版本应该包含哪些模块，哪些模块可以后置？
5. 如何设计 ablation，证明 SV-UME 的提升来自 SAM-refined memory expansion，而不是单纯 SVB-PLR pseudo-label loss？

