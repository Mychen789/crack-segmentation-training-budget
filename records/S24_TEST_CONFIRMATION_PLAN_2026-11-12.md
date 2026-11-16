# S24 一次性测试确认计划与就绪审计

- 日期：2026-11-12
- 状态：**历史草案；2026-11-13固定方案已执行，核心测试回传独立审核通过**
- 证据范围：协议、冻结/授权记录、配置、脚本和文件清单。
- 本轮未打开测试图像或标签，未运行测试推理，未生成测试预测或指标。
- 本草案登记一个可评审的确认问题，不修改或追溯替换 v0.3/v0.4/v0.5、S21、S22 的冻结记录。
- 执行分工：写作设备负责设计、核对与稿件同步；原训练设备负责资产清点、validation 核验以及未来冻结后的测试执行。无需把测试图像、标签或整套权重搬到写作设备。

### 当前进度（2026-11-13 更新）

核心test现已完成66个固定模型的107张/30组推理，回传COMPLETE且无错误。逐组混淆计数独立重算得到38/55对改变、Mmax=0.0256999，删除min后30/45，满足预先固定的描述性门槛。方案逐字节一致、选中权重身份及执行记录一致，10,000次重采样区间复算一致。完整结果见 `S24_CORE_TEST_RESULTS_2026-11-13.md`。下面的“下一步”及待核实状态为执行前历史快照，不作为当前待办。

前三轮检查均已完成并独立核对：315条正式运行历史一致，66份权重内部轮次/身份/哈希一致，61个源码文件哈希一致；实际validation复现66/66通过，最大偏差为0，数据54张/15组。完整报告及审核记录分别为 artifacts/s24/s24_validation_reproduction.json 与 s24_validation_audit.json。核心test方案及执行包现已固定，详见 `S24_CORE_TEST_FREEZE_2026-11-13.md` 与 `artifacts/s24/s24_frozen_test_plan.json`；下一步由原设备按说明显式启动，test尚未读取或评估。冻结CSV仅按split元数据统计，核心test为107张/30组；这一步没有打开图像或标签。单模型validation基线41/55与原文MA(5)窗口基线25/55分别记录。以下旧资产状态属于首次清点前快照，以本段最新进度为准。

## 1. 审计结论

写作设备没有正式模型权重，符合当前设备分工；测试准备应在原训练设备进行。本机另有的 `D:\新论文\baseline_platform` 是早期平台，只保留工程检查运行，不能替代论文的正式训练矩阵。

`tools_v3/给写作设备_原设备迁移说明_2026-11-08.md` 指向原训练设备的 `E:\新论文\baseline_platform`、`E:\新论文\s21_runs` 和 `E:\新论文\s22_runs`。这些是远程原设备上的清点目标；迁移记录说明原设备保有正式权重，下一步由清点助手核实当前完整性。

现有协议仍锁定 test 与 external_test。用户已授权继续补强，可继续完成资产准备、分析设计和执行包。正式测试须在模型清单、估计对象、脚本和本次修订形成实际冻结记录后进行，不能以旧的 TRAIN_VALIDATION_ONLY 授权启动。

## 2. 已核实的本地资产

| 项目 | 本轮核实结果 |
|---|---|
| `paper_package/data/run_summary.csv`、逐轮 validation 数据与原始 result JSON | 可用，可据此生成待索取 checkpoint 清单和选点记录 |
| `paper_package` 内 `.pt/.pth/.ckpt/.safetensors` | 未发现 |
| `D:\新论文\baseline_platform\runs` | 仅有 `overfit_max_btsp_b16_n8_e30_s2026` 与 `smoke_crack500_max_b16_s2026_64x16_e1` 两个运行目录 |
| 上述 runs 中的权重 | 仅 smoke 目录下 `best.pt`、`latest.pt`，各 23,368,521 字节；没有论文正式矩阵权重 |
| 本机旧模型源码 | 存在 UNet、ResNet18-UNet 等源码；未发现该目录下的 ConvNeXt 模型文件 |
| 旧配置 | `D:\新论文\baseline_platform\configs\crack500_train_val.json` 与 `deepcrack537_train_val.json`，mode 均为 `LOCAL_TRAIN_VAL_ONLY`，splits 均为 train/val |
| 原设备最新正式平台 | 原迁移记录给出路径与哈希；本机不可访问 |
| 核心分组清单 | 配置所指的两个 CSV 存在，文件哈希与冻结配置一致；本轮只核对文件哈希，没有列出样本或打开图像/标签 |
| test 执行入口 | 现有 `tools/reevaluate_checkpoints.py` 是 validation-only 工具，不能直接作为测试入口；旧平台有 `SplitGuard`，需独立的显式 token 与 digest |

核实过的核心分组清单：

- `D:\新论文\phase2_data_audit\manifests\crack500_group_clean_30s_groupv3_v7_r2_iter3_seed20260821.csv`，SHA-256 `f9f186d45d9db9600666fdad9fce8f43ff79e4cf36f350b7122bd874688ba825`。
- `D:\新论文\phase2_data_audit\manifests\deepcrack537_group_clean_groupv3_v7_r2_iter1_seed20260821.csv`，SHA-256 `6ea69131264f046b5a6f2aa65d2aa58e3699746e3945082f8c1dacaa21da7099`。

不能把本机旧代码当成正式代码使用。本机旧训练脚本 SHA-256 为 `d8e13e78146b0c5f821c679a74023a25f594ab937db9e9eb15e83215adc9cb52`，写作包内训练脚本为 `eb7e7cb2c59dd9a41bff80c9114acb718d63c782fa00d83c0aa318e69fd0969b`。迁移记录中的 S22 训练脚本为 `3bde1fa9b4018969fc8c6ad4d9c26cedb99272a3c75a747a804dd16f28224a89`。这些不是可互换的文件。

## 3. 既有锁与必须保留的时序

- `records/preregistered_protocol_v0.4.md` 第 1.3 节规定，主张集冻结前不得动用剩余的一次性 test；第 6 节保留 test 与 external_test 锁。
- 其继承的 v0.3 第 5 节要求，validation Go/No-Go 关闭、实现哈希不变且形成单独 test-unlock 记录后，才读取核心 test。
- `artifacts/preregistered_protocol_v0.4_freeze.json` 为已冻结的方法学版本。
- `artifacts/preregistered_protocol_v0.5_freeze.json` 仍为 `PENDING_SIGNATURE`、`effective: false`，不能把它当成已经生效的测试授权。
- `train_val_execution_authorization_v1.json` 与 `training_contract_freeze_v1.json` 只授权 TRAIN_VALIDATION_ONLY。

本次研究问题已由原 BTSP 假设转向预算敏感性，测试前须新增一个独立修订，明确不重开已判为未获支持的 H1，并将测试确认的对象、条件和解释范围写定。历史冻结文件保持原样。

## 4. 拟冻结的确认问题与最小矩阵

**确认问题：**在 validation 预先选定模型之后，DeepCrack537 / UNet 从 30 轮到 90 轮造成的算子间差值变化，是否也出现在从未用于开发的核心 test 上？

最小矩阵建议固定为 DeepCrack537 / UNet、恒定学习率、既有 11 个互异算子、种子 2026/2027/2028，以及 30 与 90 轮两个预算，共 66 个预算–算子–种子 checkpoint。两个预算下算子和种子必须完整，不能按 validation 效果只选若干对。external_test/CFD 不在本次范围内。

这是现有主效应的测试确认，不提供两个数据集、所有骨干和所有调度的独立复现。若要加入 ResNet18-UNet 或余弦调度，应在本次测试首次访问前一并写入冻结修订；不能先看最小矩阵结果再决定是否加入。

## 5. checkpoint 选择与估计对象

MA(5) 最大值是五个逐轮 validation 指标的均值，不是一份 checkpoint 的指标。已有记录 `decision_log_2026-09-15_ma5_checkpoint_selection.md` 与 `decision_log_2026-09-23_correctness_audit_and_metric_repair.md` 已明确指出这一差别。旧规则生成的 best.pt 还可能不对应修订后的 MA(5) 选点。

本草案建议固定一个可执行的模型端点：在每个预算 B 内，依据已存在的 validation 历史确定 MA(5) 最大窗口的右端轮次，平局取最早轮次；在 test 上只评估该轮次的单个模型。不得用 test 重新选择 epoch，不得在 test 上求逐轮最大值。

该端点应命名为“validation-MA(5)-selected checkpoint test score”，不能称为“test MA(5)”。它回答选定模型的测试表现是否具有预算敏感性，不等同于逐轮 test 曲线的 MA(5) 敏感性。

执行前须生成实际 checkpoint manifest，至少含：运行名、dataset、backbone、operator、seed、预算、validation 选择轮次、checkpoint 内记录轮次、相对路径、字节数、SHA-256、模型/训练配置哈希。只有选点轮次与权重元数据一致时纳入。资产不匹配时记录缺失，不得临时换 latest.pt 或另一个 best.pt。

若原设备没有选中轮次的权重，不能从 result.json 恢复模型。需要先制定公平再现训练方案及新的记录，或收缩成完整可用的预先冻结矩阵；不得用现有 smoke 权重替代，也不得在第一次 test 后补训练、补种子。

## 6. 主指标与统计量草案

- 主指标：原生网格、固定阈值 0.5、组内汇总混淆计数再对组等权的 foreground IoU，与论文指标实现一致。
- 每个算子、每个预算取三个预定种子的 test score 均值。
- 对每个算子对计算 `Δ_test(B) = score_test(a,B) − score_test(b,B)`，再计算 `shift_test = Δ_test(90) − Δ_test(30)`。
- 主要描述端点：55 对中 `|shift_test| > 0.005` 的比例 Fcross_test 及最大绝对变化 Mmax_test。三档类别改变和符号反转单独报告，不将 Fcross_test 当成排名反转率。
- 拟议描述性复现判据：Fcross_test ≥ 0.25。此阈值参照 S21 的描述性复现门槛，**属于本次新修订的拟议规则，不是旧协议中已有的测试判据**。需在首次 test 前正式写定；未达标也完整报告，不据结果调整阈值。
- 预定附加报告：逐种子 Fcross_test、55 对完整差值、三档变化计数、Fflip_test、去掉 min 后同口径结果。所有结果一次生成，不能先看主结果再挑附加分析。
- 区间方案草案：10,000 次配对重采样，随机种子 424242，以 group 为样本单位，种子按同一索引在全部算子和两个预算间配对重采样；每次重算算子均值和全部端点。冻结脚本必须明确实现两层的先后次序。55 对共享 11 个算子，不按 55 个独立观测做显著性检验。
- 本次不根据测试结果改变 reporting rule、δ、训练预算、算子集或结论叙述强度的预定解释规则。

## 7. 首次访问前的实际依赖

| 依赖 | 当前状态 | 可继续的准备工作 |
|---|---|---|
| 66 个正式 checkpoint 及对应 result/config | 本机缺失；原设备记录说明部分存在，但未完成逐文件核实 | 从 run_summary 与 result 生成具体索取清单；原设备仅清点、复制、计算哈希，不访问 test |
| checkpoint 选择轮次与权重内元数据匹配 | 未核实 | 拿到资产后只读核验，并在 validation 上核对选中模型得分；不读取 test |
| 与正式权重匹配的完整模型/指标代码 | 本机旧平台不匹配，写作包只有部分脚本 | 从正式平台取完整源码和环境版本，核对历史哈希链；先以合成输入与 validation 检查 |
| 核心 split 清单身份 | 两个清单哈希与冻结配置一致 | 保留原清单；test 数量、组数、原始图像/标签存在性本轮未核实 |
| 本次独立测试读取/评估脚本 | 缺失 | 编写具备 deny-by-default split guard、checkpoint manifest 校验、原子写出和断点幂等的入口；只用合成数据与 validation 验证 |
| 冻结修订与单独 test-unlock 记录 | 缺失 | 绑定本计划定稿、主张—证据表、checkpoint manifest、源码/配置/指标/统计脚本 SHA-256，形成有时间戳的不可覆盖记录 |

现阶段可直接在原设备执行清点与 validation 核验，不需要因写作设备缺少权重而停止补强。测试推理应在远程资产清单核实、端点确定并形成冻结修订后开展。

## 8. 原训练设备的只读清点交接

交接文件：`paper_package/tools_v3/s24_inventory_original_device.ps1`。它不导入项目源码，不加载 checkpoint 内部对象，不运行推理，不枚举 raw 数据目录，只读取预定 train/validation run 的 result.json 与权重文件元数据，并计算代码/配置哈希。权重哈希通过开关启用。

将此助手复制到原设备 `E:\新论文\tools_v3\s24_inventory_original_device.ps1` 后，在原设备 PowerShell 运行：

```powershell
& 'E:\新论文\tools_v3\s24_inventory_original_device.ps1' `
  -ProjectRoot 'E:\新论文' `
  -Python 'E:\新论文\.venv\Scripts\python.exe' `
  -OutputDir 'E:\新论文\s24_inventory_20261112' `
  -HashCheckpoints
```

输出目录必须是新目录；若已存在，换一个明确的新名字。助手只枚举三个运行根目录：`baseline_platform/runs`、`s21_runs`、`s22_runs`。结果是 `s24_asset_inventory.json`，运行前后不会改变这些目录中的任何文件。

第一轮只需回传该 JSON。它应包含：

1. 正式运行名、dataset、backbone、operator、seed、预算和完成状态。
2. validation foreground IoU 的逐轮 history、从 history 计算的 30/90 轮 MA(5) 选中 epoch，以及训练记录中的 best_epoch_ma5/选点语义。
3. 运行目录中全部 `.pt/.pth/.ckpt/.safetensors` 的路径、字节数、修改时间和 SHA-256。这能区分只留 best/latest 与留有每轮权重；**不能仅凭文件名确认权重内部 epoch**。
4. 正式平台 src、tools、config 文件的路径和哈希，以及冻结/授权 JSON 的文件哈希。脚本不打开配置所引用的图像或标签。
5. 缺失目录/结果、history 不完整等问题清单。

随后在原设备核对目标 66 个 checkpoint 的内部 epoch、model schema、config/hash 与 selection epoch，并使用已有 validation 数据验证能否载入和复现该单模型的 validation 分数。这一步须使用与正式权重匹配的可信源码；不得调用 test loader。只回传核验表、validation 结果和必要源码/config/result 文件，测试样本留在原设备。

如果仅存 best/latest，应逐预算对照：30 轮 run 的 best.pt 只能在其内部 epoch 等于该 run 的 MA(5) 选点时使用；90 轮 run 同理。不能用 90 轮 best.pt 代表 30 轮端点。若选中的 epoch 未保存，先记录缺口，再制定公平的 train/validation 再现方案，不临时替换模型。

## 9. 未来执行与报告约定

准备齐全后，在同一个冻结批次内一次性评估全部预定模型。每个模型保存 group 级混淆计数及必要预测摘要，并记录首次 test 访问时间、操作者、硬件、环境、代码哈希和 manifest 哈希。技术性中断只允许按同一 checkpoint 哈希和脚本恢复尚未完成的项，不做选模或改变配置。

首次结果出来后，以预定口径报告“复现/未达到拟议判据”及不确定性，保留原 validation 结果。不把测试确认倒写成早期已经预注册的分析，也不通过额外试验筛选有利结果。无论正负，测试访问事实与本修订的时间都写入中英文稿及统一修订记录。
