# 半监督 COD 双原型库融合改造完整方案（训练 + 推理 + Evaluator 全链路版）

## Summary
- 在现有 `BaseLine` mean-teacher 框架中加入“双原型库纠偏分支”，并将其覆盖到训练、在线验证、离线推理和 evaluator。
- 原型特征默认使用 backbone 原始第三层 `f3`，但实现上不再假设固定通道数或固定空间尺寸；通过 `config.base.prototype.py` 指定使用哪一层，并在模型初始化时动态推断该层的 `channels / height / width / stride`。
- 原型库构建只在主训练流的监督 batch 中进行，不额外切 `eval()` 再跑一遍 backbone。
- warm-up 和半监督阶段的每一个 epoch 都完整建库：
  - 当前 epoch：监督 batch 在线收集本 epoch 原型
  - epoch 末：聚合为 `next_bank`
  - 下一 epoch：激活为 `active_bank`
- 推理和 evaluator 引入原型库后，默认使用 `rectified_logit` 作为最终预测；同时保留配置开关，允许回退到 baseline 输出做对照。

## Public Interfaces
- `ModelEMA.forward`
  - 改为 `forward(x, ema=False, **kwargs)`
  - `ema=False` 走 student，`ema=True` 走 teacher，其他参数全部透传
- `TalNet.forward`
  - `forward(x)`：保持旧行为，返回原 `scaled_preds`
  - `forward(x, return_aux=True, proto_enable=False, proto_bank=None, proto_gamma=1.0, proto_infer_output="rectified")`：返回字典
- `TalNet.forward(..., return_aux=True)` 返回字段固定为：
  - `scaled_preds`
  - `gdt`
  - `raw_feats`
  - `proto`
  - `rectified_logit`
  - `final_logit`
- `raw_feats`
  - 固定包含 `{"f1","f2","f3","f4"}`
  - 全部来自 backbone 原始输出，不包含半尺度复算拼接结果
- `proto`
  - 固定返回 `sim_fg / sim_bg / fu_fg / fu_bg / alpha / theta / proto_logit / m_prob / proto_logit_full`
- `final_logit`
  - 训练时：
    - 若 `proto_enable=False` 或 bank 不可用，`final_logit = scaled_preds[-1]`
    - 若 `proto_enable=True`，`final_logit = rectified_logit`
  - 推理时：
    - 根据配置决定取 `rectified_logit`、`scaled_preds[-1]` 或 `m_prob`

## Configuration
- 在 [mkcfg.py](c:\IT\AI\CV\COD\SCOD\Prototype_model_use\BaseLine\config\mkcfg.py) 中新增：
  - `PROTOTYPE_CONFIG_DIR = 'config/base/prototype.py'`
  - merge 顺序固定为：
    - `common.py`
    - `model.py`
    - `prototype.py`
    - `run_cfg`
- 在 [prototype.py](c:\IT\AI\CV\COD\SCOD\Prototype_model_use\BaseLine\config\base\prototype.py) 新增固定配置：
  - `proto_enabled = True`
  - `proto_feature_name = "f3"`
  - `proto_topk = 16`
  - `proto_sim_temperature = 0.07`
  - `proto_tau = 0.2`
  - `proto_gamma = 1.0`
  - `proto_sup_m_weight = 0.5`
  - `proto_sup_rect_weight = 1.0`
  - `proto_unsup_weight = 0.1`
  - `proto_kde_points = 256`
  - `proto_infer_enabled = True`
  - `proto_infer_output = "rectified"`
  - `proto_eval_use_bank = True`
  - `proto_checkpoint_save_bank = True`
  - `proto_checkpoint_bank_policy = "next_bank"`
- `proto_feature_name`
  - 默认为 `f3`
  - 实现必须支持切换到 `f1/f2/f4`
  - 所有通道数和空间尺寸都从动态推断得到，不能写死

## Prototype Modules
- 原型相关实现统一放在 [Paired_Background_Guidance](c:\IT\AI\CV\COD\SCOD\Prototype_model_use\BaseLine\Paired_Background_Guidance) 目录下。
- 因为目录结构后续再定，首版全部集中在 [__init__.py](c:\IT\AI\CV\COD\SCOD\Prototype_model_use\BaseLine\Paired_Background_Guidance\__init__.py)：
  - `kde_min_threshold`
  - `PrototypeRectifier`
  - `PrototypeBankManager`
  - `serialize_prototype_bank`
  - `deserialize_prototype_bank`
- `kde_min_threshold(diff_map, num_points)`
  - 输入 `[B,1,Hf,Wf]`
  - 每张图 flatten 后做 `gaussian_kde`
  - 在 `num_points=256` 个采样点估计密度
  - 取最高两个峰之间的最低谷作为 `theta_img`
  - 若峰不足 2、方差过小、数值退化或 KDE 异常，回退 `0.0`
  - 输出 `[B,1,1,1]`，并 `detach`
- `PrototypeRectifier`
  - `in_channels` 取自 `raw_feat_specs[proto_feature_name]["channels"]`
  - `topk / sim_temperature / tau` 取自配置
  - `alpha_gate = Conv1x1 -> BN -> ReLU -> Conv1x1`
  - 末层卷积 `weight=0, bias=0`
- `PrototypeRectifier.forward(proto_feat, proto_fg, proto_bg, input_size)`
  - `proto_feat=[B,C,Hf,Wf]`
  - `q = normalize(proto_feat.flatten(2).transpose(1,2), dim=-1)`，即 `[B,Hf*Wf,C]`
  - `Sim_fg = q @ proto_fg.T`
  - `Sim_bg = q @ proto_bg.T`
  - `S_topk = topk(Sim, min(config.proto_topk, N), dim=-1)`
  - `alpha_topk = softmax(S_topk / config.proto_sim_temperature, dim=-1)`
  - `fu = sum(alpha_topk * S_topk, dim=-1, keepdim=True)`
  - `fu_fg/fu_bg` reshape 为 `[B,1,Hf,Wf]`
  - `alpha = sigmoid(alpha_gate(proto_feat))`
  - `theta_img = kde_min_threshold((fu_fg - fu_bg).detach(), config.proto_kde_points)`
  - `proto_logit = (alpha * fu_fg - (1-alpha) * fu_bg - theta_img) / config.proto_tau`
  - `m_prob = sigmoid(proto_logit)`
  - `proto_logit_full = bilinear(proto_logit, size=input_size, align_corners=True)`
- `PrototypeBankManager`
  - 维护：
    - `active_bank`
    - `next_bank`
    - `epoch_buffers`
  - `active_bank/next_bank` 固定结构：
    - `{"fg": Tensor[N_fg,C], "bg": Tensor[N_bg,C], "ready": bool, "feature_name": str, "feature_shape": tuple}`
  - `epoch_buffers` 固定结构：
    - `{"fg": list, "bg": list}`
- `enqueue_from_batch(proto_feat, gt)`
  - 只在监督 batch 调用
  - `gt` resize 到 `proto_feat.shape[-2:]`
  - `fg_mask = (gt > 0.5).float()`
  - `bg_mask = 1 - fg_mask`
  - 每张图最多生成 1 个前景原型和 1 个背景原型
  - 立即 `detach().cpu()` 存入 `epoch_buffers`
  - 不缓存整张 feature map
- `finalize_epoch(distributed=False)`
  - 本 rank 先把 buffer 堆成 CPU tensor
  - DDP 时 `all_gather_object`
  - 每个 rank 得到一致的 `next_bank`
  - 将 bank 移回当前 device，并再次 `normalize`
- `activate_next_epoch()`
  - 若 `next_bank` 存在，则替换 `active_bank`
  - 清空 `next_bank`
- `serialize_prototype_bank(bank)`
  - 保存到 checkpoint 前，把 `fg/bg` 搬到 CPU
  - 连同 `feature_name / feature_shape / ready` 一并写入
- `deserialize_prototype_bank(bank_state, device)`
  - 从 checkpoint 恢复 bank
  - 迁移回当前 device

## Dynamic Feature Selection
- `TalNet` 初始化时，用一次 `torch.no_grad()` dummy forward 推断 `raw_feat_specs`
- `raw_feat_specs` 固定至少保存：
  - `channels`
  - `height`
  - `width`
  - `stride_h`
  - `stride_w`
- 原型分支统一取：
  - `proto_feat = raw_feats[self.config.proto_feature_name]`
- 因此：
  - 默认实验仍可用 `f3`
  - 但实现上支持任意指定层
  - `PrototypeRectifier` 和 `PrototypeBankManager` 都从 `proto_feat` 动态取 `C/H/W`

## Training Timeline
- `reset_trainer()` 中新增：
  - `self.prototype_bank_manager = PrototypeBankManager(...)`
  - `self.prototype_bank = self.prototype_bank_manager.get_active_bank()`
- `launch_train()` 每个 epoch 的固定顺序：
  1. `activate_next_epoch()`
  2. `self.prototype_bank = self.prototype_bank_manager.get_active_bank()`
  3. `reset_epoch_buffers()`
  4. `train_epoch()`
  5. `finalize_epoch(distributed=config.distributed_train)`
  6. 保存 checkpoint 时，把用于该 epoch 推理的 bank 一并写入
  7. 进行 online evaluation 时，直接使用同一份 checkpoint bank
- 因为你要求 warm-up 和半监督阶段每个 epoch 都要建库，所以：
  - `epoch 1~tot_epochs` 每轮都执行 `reset -> collect -> finalize`
  - 没有任何 epoch 跳过建库
- `active_bank` 的语义固定为：
  - `epoch t` 训练中使用的是 `epoch t-1` 构建出的 bank
- `eval/checkpoint` 使用的 bank 固定为：
  - `epoch t` 训练结束后刚刚 `finalize_epoch()` 得到的 `next_bank`
  - 也就是和当前 epoch 训练后模型参数同周期的 bank
  - 配置名固定为 `proto_checkpoint_bank_policy = "next_bank"`

## Supervised And Unsupervised Flow
- 训练器拆成：
  - `_train_supervised_batch(batch, epoch)`
  - `_train_unsupervised_batch(batch, epoch)`
- `_train_supervised_batch(batch, epoch)` 固定顺序：
  1. `proto_flag = (epoch > sup_only_train_epoch and self.prototype_bank["ready"])`
  2. 调 student：
     - `out = self.model(images, ema=False, return_aux=True, proto_enable=proto_flag, proto_bank=self.prototype_bank, proto_gamma=config.proto_gamma)`
  3. 若 `out["gdt"]` 存在，按旧逻辑算 `loss_gdt`
  4. `proto_feat = out["raw_feats"][config.proto_feature_name]`
  5. `enqueue_from_batch(proto_feat, gt)`
  6. `loss_sup_base = PixLoss(out["scaled_preds"], gt)`
  7. 若 `proto_flag=True`
     - `loss_sup_m = PixLoss([out["proto"]["proto_logit_full"]], gt)`
     - `loss_sup_rect = PixLoss([out["rectified_logit"]], gt)`
     - `loss_sup = loss_sup_base + config.proto_sup_m_weight * loss_sup_m + config.proto_sup_rect_weight * loss_sup_rect`
  8. 若 `proto_flag=False`
     - `loss_sup = loss_sup_base`
  9. 若 `loss_gdt` 存在，再加上去
  10. 反向与优化
- `_train_unsupervised_batch(batch, epoch)` 固定顺序：
  1. teacher 分支：
     - `with torch.no_grad(): teacher_out = self.model(inputs, ema=True, return_aux=True, proto_enable=True, proto_bank=self.prototype_bank, proto_gamma=config.proto_gamma)`
  2. `pseudo_rect = sigmoid(teacher_out["rectified_logit"]).detach()`
  3. student 分支：
     - `student_scaled_preds = self.model(inputs, ema=False)`
  4. `loss_unsup = PixLoss(student_scaled_preds, pseudo_rect) * config.proto_unsup_weight`
  5. 反向与优化
- warm-up 阶段：
  - `epoch 1~sup_only_train_epoch`
  - 只做 baseline 监督训练和原型收集
  - 不启用原型检索
  - 不计算 `loss_sup_m`
  - 不计算 `loss_sup_rect`
  - 不做纠偏伪标签
- 半监督阶段：
  - `epoch > sup_only_train_epoch`
  - 监督分支启用 rectified supervision
  - 无监督分支启用 rectified pseudo label
  - 仍然继续在线建库

## Checkpoint And Inference
- 保存 checkpoint 时新增字段：
  - `prototype_bank`
  - `prototype_meta`
- `prototype_bank` 固定保存：
  - 当前 epoch 训练结束后刚构建出的 `next_bank`
  - 而不是训练期间正在使用的 `active_bank`
- 原因：
  - evaluator / 离线推理应该使用“与当前 epoch 模型参数同周期”的 bank
  - 这份 bank 是该 epoch 训练完成后，用当前 student 特征收集得到的
- [build_model.py](c:\IT\AI\CV\COD\SCOD\Prototype_model_use\BaseLine\models\build_model.py) 的 `build_model_eval` 固定改为：
  - `load_state_dict(..., strict=False)`
  - 同时读取 checkpoint 中的 `prototype_bank`
  - 把恢复后的 bank 附加到 model 上，例如：
    - `model.eval_prototype_bank = deserialize_prototype_bank(...)`
    - `model.eval_prototype_ready = True/False`
- 若 checkpoint 没有 `prototype_bank`
  - 且 `config.proto_eval_use_bank=True`
  - evaluator 打 warning
  - 自动回退到 baseline 输出
- 这样旧 checkpoint 仍能评估，不会直接崩

## Evaluator And Offline Inference
- [evaluator.py](c:\IT\AI\CV\COD\SCOD\Prototype_model_use\BaseLine\engine\evaluator.py) 固定改为支持 bank 推理
- `Evaluator.__init__` / `from_exists`
  - 新增 `prototype_bank=None`
  - 若未显式传入，则优先从 `model.eval_prototype_bank` 获取
- `inference_on_dataset()` 固定推理逻辑：
  - 若 `config.proto_infer_enabled=True` 且 `prototype_bank.ready=True`
    - 调：
      - `out = self.model(inputs, ema=ema, return_aux=True, proto_enable=True, proto_bank=self.prototype_bank, proto_gamma=config.proto_gamma)`
    - 根据 `config.proto_infer_output` 选择输出：
      - `"rectified"`：`pred = sigmoid(out["rectified_logit"])`
      - `"baseline"`：`pred = sigmoid(out["scaled_preds"][-1])`
      - `"m_prob"`：`pred = interpolate(out["proto"]["m_prob"], input_size)`
  - 否则：
    - 保持旧逻辑 `self.model(inputs, ema=ema)[-1].sigmoid()`
- online evaluation during training
  - `evaluate_online(epoch)` 不再只传 model
  - 还要传本 epoch checkpoint 对应的 eval bank
  - 即训练结束后 `finalize_epoch()` 产生的 `next_bank`
- [evaluate.py](c:\IT\AI\CV\COD\SCOD\Prototype_model_use\BaseLine\scripts\evaluate.py)
  - 不再只关心 model
  - 要允许 evaluator 自动使用 checkpoint 中恢复出的 `prototype_bank`
- [tool.py](c:\IT\AI\CV\COD\SCOD\Prototype_model_use\BaseLine\scripts\tool.py)
  - `extract_pseudo_labels()` 也属于离线推理路径
  - 固定与 evaluator 保持一致：
    - 若 checkpoint 中有 bank 且 `config.proto_infer_enabled=True`
    - 则用 `rectified_logit` 生成伪标签
    - 否则回退 baseline

## Logging And Metrics
- 训练期新增记录：
  - `loss_sup_base`
  - `loss_sup_m`
  - `loss_sup_rect`
  - `loss_unsup`
  - `bank_fg_size`
  - `bank_bg_size`
  - `theta_mean`
  - `theta_std`
  - `M_mean`
  - `pseudo_rect_mean`
- 推理期建议额外记录：
  - `proto_eval_bank_fg_size`
  - `proto_eval_bank_bg_size`
  - `proto_infer_output_mode`
  - `proto_eval_bank_ready`
- 评测指标本身不变：
  - `MAE`
  - `maxFm`
  - `wFmeasure`
  - `Smeasure`
  - `meanEm`
  - `meanFm`

## Files To Change
- [BaseLine/config/mkcfg.py](c:\IT\AI\CV\COD\SCOD\Prototype_model_use\BaseLine\config\mkcfg.py)
  - 增加 `prototype.py` 合并入口
- [BaseLine/config/base/prototype.py](c:\IT\AI\CV\COD\SCOD\Prototype_model_use\BaseLine\config\base\prototype.py)
  - 新增文件
  - 放原型分支训练/推理/eval 配置
- [BaseLine/Paired_Background_Guidance/__init__.py](c:\IT\AI\CV\COD\SCOD\Prototype_model_use\BaseLine\Paired_Background_Guidance\__init__.py)
  - 实现 bank、rectifier、KDE、checkpoint bank 序列化
- [BaseLine/models/talnet.py](c:\IT\AI\CV\COD\SCOD\Prototype_model_use\BaseLine\models\talnet.py)
  - 动态推断 raw feat specs
  - 暴露 raw feats
  - 接入 rectifier
  - 统一 aux/proto/final_logit 接口
- [BaseLine/engine/solver.py](c:\IT\AI\CV\COD\SCOD\Prototype_model_use\BaseLine\engine\solver.py)
  - 拆分监督/无监督 batch
  - 每 epoch 建库
  - 保存 checkpoint bank
  - online evaluation 使用 eval bank
- [BaseLine/models/build_model.py](c:\IT\AI\CV\COD\SCOD\Prototype_model_use\BaseLine\models\build_model.py)
  - `build_model_eval` 加载 checkpoint bank
  - `strict=False`
- [BaseLine/engine/evaluator.py](c:\IT\AI\CV\COD\SCOD\Prototype_model_use\BaseLine\engine\evaluator.py)
  - 支持带原型库推理
  - 支持选择 rectified/baseline/m_prob 输出
- [BaseLine/scripts/evaluate.py](c:\IT\AI\CV\COD\SCOD\Prototype_model_use\BaseLine\scripts\evaluate.py)
  - 保证离线评估能自动读取并使用 checkpoint bank
- [BaseLine/scripts/tool.py](c:\IT\AI\CV\COD\SCOD\Prototype_model_use\BaseLine\scripts\tool.py)
  - 伪标签导出路径与 evaluator 一致接入 bank

## Test Plan
- 接口兼容性：
  - `self.model(inputs)` 返回与 baseline 一致
  - `self.model(inputs, return_aux=True)` 返回新字典
- 动态特征检查：
  - `proto_feat = raw_feats[config.proto_feature_name]`
  - 代码层面不允许假设固定 `C/H/W`
  - 只要求满足 `proto_feat.shape == [B,C,Hf,Wf]`
- 训练时序检查：
  - warm-up 和半监督阶段每个 epoch 都执行建库
  - 当前 epoch 只写 `epoch_buffers` 和 `next_bank`
  - 下一 epoch 才激活为 `active_bank`
- checkpoint 检查：
  - 每个保存的 checkpoint 都带 `prototype_bank`
  - `build_model_eval` 能恢复 bank
  - 旧 checkpoint 无 bank 时能自动回退 baseline
- evaluator 检查：
  - `proto_infer_enabled=True` 且 bank 可用时，输出来自 `rectified_logit`
  - bank 不可用时自动回退 baseline
- DDP 检查：
  - 所有 rank 的 `next_bank` 行数一致
  - 所有 rank 的 bank 内容一致
- 梯度检查：
  - `enqueue_from_batch` 中存入 buffer 的原型不带梯度
  - `theta_img.requires_grad == False`
  - `pseudo_rect.requires_grad == False`
- 实验设置：
  - 数据：`TR-COD10K + TR-CAMO`
  - 标注比例：`5%`
  - 无标签比例：`95%`
  - 测试集：`CHAMELEON / TE-COD10K / TE-CAMO / NC4K`
  - 总 epoch：`30`
  - warm-up：`1~15`
  - semi-supervised：`16~30`
- 必做消融：
  - `Baseline`
  - `Baseline + 每 epoch 在线建库`
  - `+ rectified supervision`
  - `+ rectified pseudo label`
  - `+ inference/evaluator 使用 prototype bank`

## Assumptions
- 默认实验层仍是 `f3`，但实现必须支持通过 `config.proto_feature_name` 动态切换。
- 原型库保留的是“图像级 pooled prototype”的全集，不是像素级全集。
- 原型模块首版全部集中在 `Paired_Background_Guidance/__init__.py`，后续拆目录只做重构，不改外部接口。
- 推理和 evaluator 现在默认引入原型库，并默认使用 `rectified_logit`，但保留配置回退到 baseline 的能力。
