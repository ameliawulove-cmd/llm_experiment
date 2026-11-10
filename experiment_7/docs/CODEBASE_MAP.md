# Exp7 代码库地图（Codebase Map）

> 这份文档列出 Experiment 7 所有 Python 文件，说明每个文件的用途、输入/输出、关键函数，以及是否还在使用。适合第一次接触这个项目的人快速定位代码。

---

## 分类总览

| 类别 | 文件数 | 说明 |
|------|--------|------|
| 核心实验脚本 | 7 | 主实验流程（exp7_neuro.py 等） |
| 验证脚本 V1 | 4 | 历史验证（有污染，结论需谨慎） |
| 确认性验证 V2 | 14 | 干净独立验证（主要参考） |
| 辅助/报告工具 | 3 | 生成文档、构建 stub 等 |

---

## 一、核心实验脚本（项目根目录）

### 1. `exp7_neuro.py` — 主脚本（1761行）

**类别：** 计算模型拟合 + Linear Probe + 组件分析 + 探索分析  
**状态：** ✅ Active（但 7C 输出文件缺失，该部分未完成）  
**复现最终结果需要：** 是（7A、7B 是核心结果）

| 项目 | 内容 |
|------|------|
| **主要目的** | 7A：用强化学习模型拟合行为数据；7B：训练 Linear Probe 检测各层编码信息；7C：分析注意力层和 MLP 层贡献；7E：PCA + logistic regression 探索 |
| **输入** | `data/exp6_full/trials.jsonl`，`data/exp6_activations/*.npy` |
| **输出** | `results/exp7a/model_fits.json`，`results/exp7b/probe_results.json` |
| **关键函数** | `load_trials()` 读行为数据；`build_targets()` 构造预测目标（choice_tendency 等）；`ols_probe()` OLS 线性回归 probe；`episode_split()` 按 episode 分训练/测试集；`corr()` 计算 Pearson r；`_rw_nll()` / `_recency_nll()` / `_wsls_nll()` 三个模型的负对数似然；`command_fit_models()` 运行 7A；`command_probe_7b()` 运行 7B；`command_analyze_7c()` 运行 7C（输出缺失）；`command_discover_7e()` 运行 7E |
| **注意事项** | 7C 的输出文件缺失，`command_analyze_7c()` 和 `command_collect_comps()` 尚未成功运行 |

---

### 2. `exp7d_dynamics.py` — 激活动态分析（1590行）

**类别：** 因果干预 / 激活扰动传播  
**状态：** ✅ Active  
**复现最终结果需要：** 否（辅助分析，不是 V2 主要证据）

| 项目 | 内容 |
|------|------|
| **主要目的** | 通过 forward hook 给某层 hidden state 加扰动，观察扰动如何传播到后续层，以及对最终 A/B 选择 logit 的影响 |
| **输入** | `data/exp6_activations/acts_choice.npy`，`results/exp7d/probe_weights.npz`，`results/exp7h/R_choice_tendency_L27_k16.npy` |
| **输出** | `results/exp7d/full_results.npz`，`results/exp7d/probe_weights.npz`，`results/exp7d/directed_results.npz` |
| **关键函数** | `command_fit_probes()` 训练各层 probe；`run_perturbed()` 在 `blocks[l_inj-1]` 注册 forward hook 实现加法扰动；`compute_metrics()` 计算扰动后的指标；`build_directions()` 构造扰动方向；`command_full()` 全面分析；`command_directed()` 方向性扰动；`command_rand_subspace()` 随机子空间对照 |
| **注意事项** | 这里的扰动是**加法（additive）**，和 V2 验证用的**替换（replacement）**方式不同，直接对比时要注意 |

---

### 3. `exp7f_sae.py` — 稀疏自编码器（594行）

**类别：** 特征提取  
**状态：** ✅ Active（训练完成）  
**复现最终结果需要：** 否（下游 7G 用到，但 7G 本身也不是核心结论）

| 项目 | 内容 |
|------|------|
| **主要目的** | 在 L27（第27层）的 hidden state 上训练稀疏自编码器，提取稀疏可解释特征 |
| **模型结构** | SAE，encoder/decoder 各一层，hidden dim = 4096，L1 正则化 |
| **输入** | `data/exp6_activations/acts_choice.npy`，`data/exp6_activations/acts_declare_a.npy` |
| **输出** | `results/exp7f/sae_choice_L27.pt`（PyTorch checkpoint） |
| **关键函数** | `SparseAutoencoder`（模型类）；`train_sae()` 训练主函数；`command_train()` 命令入口；`command_analyze()` 分析已训练模型 |

---

### 4. `exp7g_lagrangian.py` — Lagrangian 特征扰动（645行）

**类别：** 因果干预（特征空间）  
**状态：** ✅ Active  
**复现最终结果需要：** 否

| 项目 | 内容 |
|------|------|
| **主要目的** | 在 SAE 的特征空间做约束优化（Lagrangian 对偶上升），找出能影响 probe 分数的最小扰动 |
| **输入** | SAE checkpoint（`results/exp7f/`），激活数组，probe 权重 |
| **输出** | `results/exp7g/lagrangian_results.json` |
| **关键函数** | `solve_lagrangian()` Lagrangian 对偶上升主函数；`command_run()` 命令入口 |
| **⚠️ 重要注意** | 优化的目标是 **probe 分数的代理目标**，不是模型的直接输出（logit）。因此结果不能直接说明对模型实际行为的影响。 |

---

### 5. `exp7h_das.py` — DAS 训练（522行）

**类别：** DAS（Distributed Alignment Search）训练  
**状态：** ✅ Active（已训练，R 矩阵已冻结）  
**复现最终结果需要：** 是（R 矩阵是 V2 验证的输入）

| 项目 | 内容 |
|------|------|
| **主要目的** | 学习一个 16 维正交子空间 R ∈ ℝ^{1024×16}，使得在这个子空间里做"替换干预"（interchange intervention）能预测性地移动模型输出 |
| **输入** | `data/exp6_activations/acts_choice.npy`，`data/exp6_full/trials.jsonl` |
| **输出** | `results/exp7h/R_choice_tendency_L27_k16.npy`（已冻结），`results/exp7h/das_results.json` |
| **关键函数** | `RotationBasis`（可学习正交矩阵）；`train_das()` 训练主函数；`sample_pairs()` 采样 source/base 配对；`fit_ridge_probe()` 训练 ridge probe；`command_train()` 训练入口；`command_evaluate()` 评估入口 |
| **⚠️ 重要注意** | 训练/测试分割是**按行索引**：前1200行（episodes seq_0000 到 seq_0039）训练，后300行测试。这和 7B probe 用的 episode 级分割不同。 |
| **干预公式** | h_int = h_b + R R^T (h_s - h_b)，这是**替换**操作（replacement），不是加法 |

---

### 6. `exp7i_integrate.py` — 结果整合（332行）

**类别：** 结果整合  
**状态：** ❌ Superseded（已被科学审计推翻）  
**复现最终结果需要：** 否（结论无效，不要引用）

| 项目 | 内容 |
|------|------|
| **主要目的** | 基于规则自动整合各实验结果，生成 "CONFIRMED" / "PARTIAL" 等 verdict |
| **输入** | 各实验的 results/ 文件 |
| **输出** | `results/exp7i/integration_report.json` |
| **⚠️ 审计结论** | 其输出的 "CONFIRMED" 结论超出证据支持范围。0.995 IIA 实际是训练集污染的方向性符号一致率；83% 实际是校准阈值越过率，不是行为翻转。**不要引用 7I 的结论。** |

---

### 7. `exp7j_circuits.py` — 电路分析（656行）

**类别：** 电路分析（Mechanistic Interpretability）  
**状态：** ⚠️ Active 但有已知 bug  
**复现最终结果需要：** 否

| 项目 | 内容 |
|------|------|
| **主要目的** | Logit lens（在每一层应用最终的 unembedding matrix 预测 token）+ activation patching（替换某层激活观察输出变化） |
| **输入** | 模型权重，行为数据 |
| **输出** | `results/exp7j/` |
| **关键函数** | `apply_rms_norm()` 应用 RMS 归一化；`command_logit_lens()` 运行 logit lens；`command_collect_patch()` 运行 activation patching |
| **⚠️ 已知 bug** | `build_corrupted_prompt()` 存在 **partial swap bug**：构造 corrupted prompt 时，历史标签（A/B 的奖励历史）未完全重映射，导致 corrupted prompt 实际上只是部分损坏。结果仅供参考。 |

---

## 二、辅助/报告工具（项目根目录 & REPORTS/）

| 文件 | 类别 | 主要目的 | 状态 |
|------|------|----------|------|
| `build_das_stub.py` | 工具 | 构建 DAS stub 文件，辅助调试 | ✅ Active |
| `build_exp7_gdoc.py` | 报告工具 | 生成 Google Doc 格式报告 | ✅ Active |
| `REPORTS/build_exp7_pptx.py` | 报告工具 | 构建 PowerPoint 演示文稿 | ✅ Active |

---

## 三、历史验证脚本 V1（`results/validation/`）

> ⚠️ **这些脚本发现了重要错误，但自身结果也有污染问题。** 仅用于理解错误来源，不能作为正面证据。

### `validate_01_generation.py`（383行）

**类别：** 验证  
**状态：** ⚠️ 结果需谨慎解读（训练集污染）

| 项目 | 内容 |
|------|------|
| **主要目的** | 测试 additive hook steering 能否让模型翻转选择 |
| **关键发现** | 83% 校准阈值越过率，但 argmax flip = **0%** |
| **污染问题** | 在训练集数据（前1200行）上测试，不是独立验证集 |
| **结论** | 83% 不是行为翻转，只是 logit 相对于阈值的越过率 |

---

### `validate_02_random_iia.py`（339行）

**类别：** 验证  
**状态：** ⚠️ 结果需谨慎解读（训练集污染 + 精度问题）

| 项目 | 内容 |
|------|------|
| **主要目的** | 测试 DAS 交换干预的 IIA（Interchange Intervention Accuracy） |
| **关键发现** | 方向性符号一致率 = 0.995 |
| **污染问题** | 在训练集数据上测试；float16（激活缓存）和 float32（模型推断）精度不匹配 |
| **结论** | 0.995 不是真正的 IIA，只是训练集上的方向性符号一致率 |

---

### `validate_03_letter_prior.py`（165行）

**类别：** 验证  
**状态：** ✅ 有效结果

| 项目 | 内容 |
|------|------|
| **主要目的** | 测量模型对 "A" vs "B" token 的系统性偏好（字母先验） |
| **关键发现** | 字母先验 = **+4.016 logits**（std 0.067）——这是一个干净有效的结果 |
| **说明** | 解释了为什么所有实验的 argmax flip = 0%：先验太强，任何合理的干预都无法翻转最终选择 |

---

### `validate_04_audit.py`

**类别：** 审计工具  
**状态：** ✅ Active  
**说明：** 审计辅助脚本，用于交叉检查其他验证脚本的结果

---

## 四、确认性验证 V2（`VALIDATION_EXP7_V2/scripts/`）

> ✅ **这是实验中最干净的验证数据来源。** 在20个新 episode（未参与任何训练）上运行，使用范数匹配随机控制，10-cluster bootstrap 统计检验。

### 主要验证脚本

| 文件 | 行数 | 主要目的 | 是否复现需要 |
|------|------|----------|--------------|
| `confirm_v2.py` | 280 | V2 主验证脚本：replacement hook，范数匹配随机控制，10-cluster bootstrap | ✅ 是（主要结果来源） |
| `core.py` | 119 | 纯数值工具函数，无模型推断：`metrics()` 计算所有指标，`bootstrap()` bootstrap 统计 | ✅ 是 |
| `model_adapter.py` | 64 | 模型接口（tokenizer 准备等），封装模型加载 | ✅ 是 |
| `report_results.py` | 103 | 生成 VALIDATION_RESULTS.md，验证 hash 后拒绝覆写（防止篡改） | ✅ 是 |

**confirm_v2.py 的关键设计：**
- 干预方式：**replacement hook**（h_int = h_b + R R^T (h_s - h_b)），和 7D 的加法扰动不同
- 控制组：10 个范数匹配的随机正交子空间（排除"任意大向量都有效果"的可能）
- 统计检验：10-cluster bootstrap，配对差异 CI
- 数据：20 个全新 episode（seq_0040 到 seq_0059），与训练集零重叠

### 预检诊断脚本（按顺序运行）

| 文件 | 主要目的 |
|------|----------|
| `phase_0_5_sanity.py` | 基础 sanity check：检查数据加载、模型输出格式 |
| `phase_0_5b_diagnostic.py` | 诊断：检查激活缓存和实时推断的一致性 |
| `phase_0_5c_intervention.py` | 单点测试：确认 hook 注册和替换逻辑正确 |
| `phase_0_5d_propagation.py` | 确认干预信号能传播到输出层 |

### 开发和审计辅助脚本

| 文件 | 主要目的 |
|------|----------|
| `analyze.py` | 分析 V2 结果文件 |
| `check_duplicates.py` | 检查配对数据中是否有重复 |
| `dev_pairs.py` | 开发配对采样逻辑 |
| `dev_sanity.py` | 开发阶段 sanity check |
| `freeze_artifacts.py` | 冻结关键文件（生成 hash），防止意外修改 |
| `manifest.py` | 管理文件 manifest 和 hash 验证 |

### 单元测试（`VALIDATION_EXP7_V2/tests/`）

| 文件 | 主要目的 |
|------|----------|
| `test_core.py` | 测试 core.py 中的数值函数（metrics、bootstrap） |
| `test_hook.py` | 测试 hook 注册和干预传播逻辑 |

---

## 五、文件与实验的对应关系

```
行为数据收集（Behavioral Data Collection）
  └─ 无单独脚本（exp6 阶段完成）→ trials.jsonl

激活提取（Activation Extraction）
  └─ exp7_neuro.py（command_collect_t2, command_collect_comps）→ *.npy

7A 计算模型拟合
  └─ exp7_neuro.py（command_fit_models）→ model_fits.json

7B Linear Probe
  └─ exp7_neuro.py（command_probe_7b）→ probe_results.json ★核心结果

7C 组件分析
  └─ exp7_neuro.py（command_analyze_7c）→ ⚠️ 输出缺失，未完成

7D 激活动态
  └─ exp7d_dynamics.py（command_full, command_directed）→ full_results.npz

7E 探索分析
  └─ exp7_neuro.py（command_discover_7e）→ 探索结果

7F 稀疏自编码器
  └─ exp7f_sae.py（command_train）→ sae_choice_L27.pt

7G Lagrangian 优化
  └─ exp7g_lagrangian.py（command_run）→ lagrangian_results.json
  ⚠️ 优化 probe 代理目标，非直接输出

7H DAS 训练
  └─ exp7h_das.py（command_train）→ R_choice_tendency_L27_k16.npy ★核心结果

7I 整合分析
  └─ exp7i_integrate.py → ❌ 已被审计推翻，不可引用

7J 电路分析
  └─ exp7j_circuits.py → ⚠️ 有 partial swap bug

历史验证 V1
  └─ results/validation/validate_01~04 → ⚠️ 训练集污染，结论需修正

确认性验证 V2
  └─ VALIDATION_EXP7_V2/scripts/confirm_v2.py → ★V2 因果敏感性证据
```

---

## 六、复现最终结论所需的最小文件集

如果你只想复现论文中可以报告的核心结论，需要以下文件：

| 文件 | 用途 |
|------|------|
| `exp7_neuro.py` | 复现 7B Linear Probe r = 0.969 |
| `exp7h_das.py` | 复现 DAS 训练（或直接用冻结的 R 矩阵） |
| `results/exp7h/R_choice_tendency_L27_k16.npy` | 冻结的 R 矩阵（V2 验证的输入） |
| `VALIDATION_EXP7_V2/scripts/confirm_v2.py` | 复现 V2 因果敏感性证据 |
| `VALIDATION_EXP7_V2/scripts/core.py` | V2 数值计算依赖 |
| `VALIDATION_EXP7_V2/scripts/model_adapter.py` | V2 模型接口依赖 |
| `results/validation/validate_03_letter_prior.py` | 复现字母先验测量 |

**不需要（也不应该引用）：**  
`exp7i_integrate.py`，`validate_01_generation.py`，`validate_02_random_iia.py` 的结论

---

*文档由 Claude Code 生成，2026-11-10*
