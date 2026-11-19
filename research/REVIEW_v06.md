# REVIEW_v06 — 对 v0.6（qEHVI P0 修复 / qNEHVI / mfbo / 复孔 alpha）的审查

> 审查者：algo-research ｜ 日期：2026-11-05 ｜ 快照：18:04（`qehvi.py` 8.0KB、`qnehvi.py` 6.3KB、`mfbo.py` 12.7KB、`bayes.py` 22.5KB、`study.py` 26.5KB）
> 格式：**`文件:行` → 问题 → 文献依据 → 修复建议 → 优先级**。P0 = 正确性/结果不可信；P1 = 显著影响样本效率或与宣称不符；P2 = 工程质量。
> **已确认修复（v0.5 → v0.6）**：qEHVI 的 `tell` 现在对**全部**观测调用父类（`qehvi.py:79`，缺失目标记为 NaN 行），`_skipped` 自动统计（`:83`）；无完整观测/无点支配 ref 时显式 `ValueError`（`:124-138`）；局部惩罚尺度复用 `models[0]`（`:152-154`，不再拟合第三个 GP）；`Study` 已用 `supports_fidelity`/`ask_with_fidelity`（`study.py:338-351`）并把 fidelity 写进 pending/suggestions/results（`:477-493,650-659`）；复孔逐点 alpha 已接入（`bayes.py:208-223,434-445`）。

---

## 0. 总结

| 编号 | 位置 | 一句话 | 优先级 |
|---|---|---|---|
| P0-1 | `qehvi.py:104-110`（qnehvi 继承） | 数据推导 ref 用 `np.min(signed)`，而 `_Y` 含缺失目标的 **NaN 行** → ref 变成 NaN；随后 `partition_2d` 过滤掉所有点 → 抛"无观测支配 ref"的误导性错误。**只要有一条缺目标观测且没给显式 ref_point，qEHVI/qNEHVI 必挂** | **P0** |
| P0-2 | `qnehvi.py:85-95` | `_noise_std` 用**标量** `objective_std` 同时填两列；`noise_std` 默认 0、且没有"无噪声信息"的告警 → 用户以为在跑带噪 qNEHVI，实际可能退化为 qEHVI | **P0** |
| P1-1 | `qnehvi.py:132-139` | baseline 用**全部** valid 观测（无 `baseline_max`/支配剪枝）→ n_obs 大时每次 ask `O(n_base×n_obs)`；且被支配点不改变前沿只浪费算力 | P1 |
| P1-2 | `qehvi.py:117` | 候选池 local 中心仍来自父类 `self._y`（主目标）；多目标前沿覆盖偏向主目标端（v0.5 P1 未改） | P1 |
| P1-3 | `qehvi.py:104-110` | ref 回退值在首次 `_model_ask`（≥3 个完整观测）即冻结；对预算紧的多目标 study，窗口可能由 3 个点决定（v0.4 ParEGO 同型问题） | P1 |
| P1-4 | `mfbo.py:203` | `best = np.max(self._y)` 混用所有保真度的观测；低保真系统性偏低（bias>0），但噪声/异常时可能污染 EI 的基线 → 采集值系统性偏移 | P1 |
| P1-5 | `mfbo.py:233` | 局部惩罚用硬编码 `scale=0.15`，且"撞点"时只把候选索引 `+1` 移动、**不重新按采集值选点** → 可能选到明显更差的点 | P1 |
| P1-6 | `mfbo.py:174-191` | 初始设计按 `levels[index % len]` 轮转（含全保真）而非"全用最低档"；`n_initial` 默认 `2*(feature_dim+1)` 未考虑 fidelity 维 | P1 |
| P1-7 | `bayes.py:434-445` + `dataset.py:53` | 复孔 alpha 已实现，但 `objective_std` 是**标量** → 多目标时 qEHVI/qNEHVI 的第 2 个目标仍在用主目标噪声（`_fit(X, Y[:,1])` 的 alpha 来自 `_noise_var`） | P1 |
| P2-1 | `qehvi.py:104-110` | `_ref` 若为 NaN 会写进 `state_dict`（`json.dumps` 产出非法 JSON `NaN`） | P2 |
| P2-2 | `qnehvi.py:99-166` | `_model_ask` 在 2 目标 + baseline 循环里重复 fit（`_fit(X, Y[:,0])` 又一次只为 lengthscale）；应复用 ensemble 的模型 | P2 |
| P2-3 | `qnehvi.py:132-139` | 候选 `points` 在 baseline 循环外抽一次（公共随机数）：**无偏且降方差**，但需在文档里写明这是有意的（否则读者以为 bug） | P2 |
| P2-4 | `mfbo.py:146-147` | `ask()` 丢弃 fidelity（只在 `ask_with_fidelity` 里返回）；直接用 `ask()` 的调用方（benchmark 脚本）会把建议按全保真执行 | P2 |
| P2-5 | `mfbo.py:109-121` | `fidelity_levels` 用逗号字符串参数解析；非法输入静默跳过（只保证 1.0 存在）→ CLI 拼错时用户无感 | P2 |
| P2-6 | `bayes.py:437-445` | `alpha` 数组的 padding 逻辑假设 fantasy 点总在末尾；`batch_strategy` 仍含 `constant_liar`（会伪造观测）时该假设成立，但若将来加"插入式"候选需重算 | P2 |
| P2-7 | `study.py:338-351` | `ask_with_fidelity` 的 pending 有 fidelity，但 `remaining()` 仍按**次数**计；`cost_spent` 只在 report 里（v0.6 未确认是否实现）→ 多保真的"预算"语义在 CLI 层仍模糊 | P2 |

**总体**：v0.5 的两个 P0 修得干净（显式报错 + 计数一致 + 复用 GP），qNEHVI 结构与 Round 6 规格一致（baseline 抽样 → 分区 → DHV），`Study` 的 fidelity 链路完整。**新暴露的 P0-1/P0-2 都是"错误路径"**：缺目标观测 + 无显式 ref → NaN ref；无噪声信息 → 静默退化为 qEHVI。两者都不影响现有测试（测试都有完整目标与显式 ref），但会在真实 campaign 第一轮就出问题。

---

## 1. P0 详述

### P0-1 `qehvi.py:104-110` — NaN ref（缺目标观测 + 无显式 ref_point）

```python
if self._ref is None:
    signed = signed_scores(self._Y, self.objective)     # ← _Y 可能含 NaN 行
    lows = np.min(signed, axis=0)                       # ← NaN 传播
    highs = np.max(signed, axis=0)
    span = np.maximum(highs - lows, 1e-9)
    self._ref = lows - 0.05 * span                      # ← ref 全 NaN
```

v0.6 的 `tell` 为了计数一致，把**缺失目标的观测也写进 `_Y`（NaN 行）**（`qehvi.py:76`）——这是对的；但 `_signed_references` 没有过滤 NaN。只要有一条缺目标观测且用户没给 `ref_point`：

1. `ref = [nan, nan]`；
2. `partition_2d` 的 `points[np.all(points > ref)]` 对 NaN 比较恒 False → `boxes=[]`；
3. 抛 `ValueError("no observation dominates objective.ref_point=None")` —— 错误信息误导（ref 是自动推导的，且数据里明明有高值点），用户会去查数据而不是查 NaN。

**文献依据**：R21（EHVI 的 ref 必须严格被支配；BoTorch 在 ref 不合法时显式抛错）、R42（HV/ref 口径）、R17（配置/数据错误应在使用点给出可行动的错误信息）。

**修复**（3 行）

```python
def _signed_references(self):
    if self.objective.ref_point:
        ...
    if self._ref is None:
        valid = np.all(np.isfinite(self._Y), axis=1)      # ← 只用完整观测
        if valid.sum() < max(2, self.objective.n_objectives):
            raise ValueError("qEHVI: need at least 2 complete objective vectors "
                             "to derive a reference point; provide Objective.ref_point")
        signed = signed_scores(self._Y[valid], self.objective)
        ...
```

同时：`state_dict` 里若 `_ref` 含 NaN，改为存 `None` 并 warning（P2-1）。

**验收**：`test_qehvi_missing_objective_without_ref_raises_clear_error`（一条缺目标 → 错误信息含 "complete objective vectors" 或 "ref_point"）；`test_qehvi_missing_objective_with_ref_works`（显式 ref → 正常 ask）。

### P0-2 `qnehvi.py:85-95` — "带噪"可能静默退化为无噪

```python
std = float(observation.objective_std) if observation.objective_std > 0 else float(self.params["noise_std"])
rows.append([std, std])
```

- `objective_std` 是**标量**（`dataset.py:53`），所以两目标都用同一个 σ；
- 默认 `noise_std=0.0`；若用户既没给复孔 std、也没显式设 `noise_std`，则 **baseline 抽样 = 确定值** → qNEHVI ≡ qEHVI，但 router/技能层仍会因为它声明 `noisy` 而选它；
- 没有任何 warning 提示"没有噪声信息，正在退化为 qEHVI"。

**文献依据**：R12（qNEHVI 的收益完全来自 baseline 噪声；无噪声时与 qEHVI 等价）、R10（带噪 EI 需要噪声模型）、R2（复孔/alpha）。

**修复**

```python
def tell(self, observations):
    ...
    if noise.max() <= 0 and any(std is None):   # 无复孔 std 且未设 noise_std
        warnings.warn("qNEHVI: no observation-noise information (objective_std=0, noise_std=0); "
                      "the acquisition reduces to qEHVI — provide replicates or set noise_std")
```
- 更好的做法：在 `Study.create` 时校验"算法声明 noisy 且 noise∈{medium,high}"时，若 `objective_std` 与 `noise_std` 都缺，则要求用户明确（否则 router 退到 qEHVI）。
- 长期：`objective_std` 改为 `dict[str,float]`（P1-7）。

**验收**：`test_qnehvi_warns_when_no_noise_information`；`test_qnehvi_noise_std_changes_acquisition`（`noise_std=0` vs `2.0` 的采集排序不同，见 `qNEHVI_ACCEPTANCE.md` #3）。

---

## 2. P1 详述

### P1-1 `qnehvi.py:132-139` — baseline 未剪枝

每个 baseline 样本都对**全部** valid 观测做 `non_dominated_mask`（O(n²)，v0.6 `pareto.non_dominated_mask` 对 n>500 已慢）。修复：每次 ask 前先取"非支配 ∪ 每目标 top-k ∪ 最近 20 个"作为 `X_baseline`（上限 `baseline_max=200`），并在 `ParamSpec` 暴露 `baseline_max`；`n_obs≤200` 时行为不变（不影响测试）。
**依据**：R12（qNEHVI 实践中 `X_baseline` 会剪枝）、R21。

### P1-2 `qehvi.py:117` — 候选池局部中心用主目标

`_candidate_pool` 来自父类，local 中心 = `argsort(-self._y)[:5]`（主目标）。2D 前沿覆盖会偏向主目标端。修复：qEHVI 覆写 `_candidate_pool`，中心取"当前 Pareto 集的点 + 每个目标端 top-2"；或把 `self._y` 临时换成"归一化目标 1 与目标 2 的交替极值"。**依据**：R21（候选分布影响前沿覆盖）、R11（ParEGO 的权重轮换动机）。

### P1-3 `qehvi.py:104-110` — ref 回退冻结过早

与 v0.4 ParEGO 的 P0-1 同型：首次 `_model_ask`（≥3 完整观测）就冻结数据推导 ref。修复：数据不足 `max(3, 2m)` 时**不冻结**（每次重算）；冻结时写 `decisions`。**依据**：R42（ref 固定用于跨轮比较，但必须"有意义"）、`PAREGO_SPEC.md` §2.2。

### P1-4 `mfbo.py:203` — EI 基线混用保真度

```python
best = float(np.max(self._y))     # ← 混合了所有 fidelity 的观测
```
低保真观测系统性偏低（bias>0），通常 max 来自全保真；但噪声下低保证"幸运高值"会污染基线，使 EI 低估真实改进。修复：`best = max(y[obs.fidelity == max(level)])`，若无全保真点则用 `max(y) - max_bias(level)`（或直接用当前保真度分组的 max）。**依据**：R41（BOCA 的偏差处理）、R2（噪声）。

### P1-5 `mfbo.py:233` — 局部惩罚的硬编码与"索引 +1"替换

- `scale=0.15` 固定，未用 ARD lengthscale；
- 撞点时 `candidate_index = (candidate_index + 1) % len(candidates)` 只移动一个索引，**不再看采集值** → 可能选到非常差的点（尤其候选池排序是 Sobol 随机顺序）。
修复：与 `bayes` 一致，用采集值乘惩罚后重新 `argmax`（一次算完，不需要循环）。**依据**：R35（local penalization）。

### P1-6 `mfbo.py:174-191` — 初始设计含全保真

`levels[index % len(levels)]` 轮转，初始设计就花全保真成本。MVP 规格（`MULTIFIDELITY_MVP.md` §4）建议**初始全部用最低档**（低成本探索），全保真留给 `min_full_fraction`。修复：`queue.append((config, min(levels)))`；若担心"没有全保真信息"，在 `_initial_design` 末尾追加 1–2 个全保真点。**依据**：R41、R39（低成本先筛）。

### P1-7 `bayes.py:434-445` + `dataset.py:53` — 多目标复孔噪声

`_noise_var` 来自标量 `objective_std`（主目标）。`qehvi._fit(X, Y[:, 1])`（第 2 目标）用同一个 alpha 数组 → 第 2 目标噪声被错配。修复：`Observation.objective_std` 改 `dict[str,float]`；`_fit` 增加 `target_name` 参数；`qehvi/qnehvi` 按目标取噪声。**依据**：R2（逐点/逐目标噪声）、R10（带噪约束/目标）。

---

## 3. P2（择机）

- **P2-1** `qehvi.py:104-110`：NaN ref 被 `state_dict` 序列化（Python `json` 写 `NaN`，严格 JSON 解析器会拒绝）；存 `None`。
- **P2-2** `qnehvi.py:99-166`：保存 `models` 列表，复用 `models[0]` 求 `_local_penalty_scale`（现在又 fit 了一次）。
- **P2-3** `qnehvi.py:132-139`：`points` 在 baseline 循环外抽样是"公共随机数"（无偏、降方差）；在 docstring 写明。
- **P2-4** `mfbo.py:146-147`：`ask()` 丢 fidelity；在 docstring 写"请用 `ask_with_fidelity`（`Study` 已自动）"；benchmark 脚本用 `ask_with_fidelity`。
- **P2-5** `mfbo.py:109-121`：非法 `fidelity_levels` 静默跳过 → `warnings.warn`。
- **P2-6** `bayes.py:437-445`：alpha padding 依赖"fantasy 点在末尾"；加断言 `len(alpha) == len(X)` 并在 `_fit` 统一重算（更稳）。
- **P2-7** `study.py:338-351`：`remaining()` 按次数；多保真应增加 `remaining_cost()`（`cost_spent` 已部分实现，需 CLI/report 暴露）。

---

## 4. 证据与验证

| 检查 | 方法 | 结果 |
|---|---|---|
| qEHVI P0 修复 | 读 `qehvi.py:69-83,113-154` | `tell` 全量给父类 ✓；ref 空/无完整观测显式 `ValueError` ✓；复用 `models[0]` ✓ |
| NaN ref 风险 | 读 `qehvi.py:104-110` + `_Y` 写入路径（`:74-76`） | 未过滤 NaN → ref NaN（P0-1） |
| qNEHVI 结构 | 读 `qnehvi.py:78-166` | 嵌套 MC（baseline 外/候选内）、复用 `partition_2d`/`dhv_2d_batch` ✓；无 `baseline_max`（P1-1）；无噪声告警（P0-2） |
| mfbo 链路 | 读 `mfbo.py:127-241` + `study.py:338-351` | `supports_fidelity`/`ask_with_fidelity`/pending/results 全链路 ✓；`best` 混用保真度（P1-4）；初始设计含全保真（P1-6） |
| 复孔 alpha | 读 `dataset.py:53-55`、`bayes.py:208-223,434-445` | 逐点 alpha=(σ²/k)/var(y) ✓；标量 std（P1-7） |

---

## 5. 修复顺序建议

1. **P0-1**（NaN ref，3 行）→ **P0-2**（噪声缺失告警）——都在第一轮 campaign 就会踩。
2. **P1-4/P1-6**（mfbo 的基线混保真、初始设计）——多保真主路径，改动小。
3. **P1-7**（std 字典化）——qNEHVI/带噪约束的前置。
4. **P1-1/P1-2/P1-3**（baseline 剪枝、候选池中心、ref 冻结时机）——样本效率。
5. **P1-5**（惩罚重选）与 P2 清理。

**回归门槛**：现有测试全绿 + `qNEHVI_ACCEPTANCE.md` 的 5 条（无噪 HV_ratio≥0.90、收敛 Spearman≥0.85、含噪 HV_ratio≥0.80、批量最小区间距≥0.05、ref 必填/持久化）；`moo_yield_cost` 的 qEHVI HV_ratio 不劣化（当前 ≈0.95）。

---

## 6. 参考

R12（qNEHVI：嵌套 MC/噪声 baseline）、R21（qEHVI：分区/ref）、R42（HV/固定 ref）、R10（带噪 EI）、R41（BOCA：保真度作为输入/偏差）、R35（local penalization）、R2（复孔/alpha）、R32（benchmark 协议）、R17（配置/数据校验）。
