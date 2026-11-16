# 预注册研究协议 v0.3

日期：2026-08-26  
状态：FROZEN METHODOLOGY / EXECUTION LOCKED（方法学已冻结，执行仍锁定）  
实施状态：未启动拟议方法训练；未读取或生成 validation 预测；确认性 test 与 CFD 外测不得解锁。  
继承来源：

- preregistered_protocol_candidate_v0.2.md，SHA-256 a2027a95ff434cf5730e58d2d235a72c66fef97ed078a6570117b714d274deed；
- p1_method_novelty_alignment_freeze_v1.json，SHA-256 ed72b8ecc4309af15922c775f047f63554b3853fe29733f221ae7dc94d9c61df。

本协议是训练前控制版本。v0.2 中关于数据构建、group-clean 划分、CNR/宽度阈值、退化、统计单位、发表与再分发边界的详细条款继续有效；凡与本协议在方法定义、假设、投稿门槛、术语或最近邻对照上冲突，以 v0.3 为准。

本协议不构成法律意见，也不证明方法有效。所有结果性主张必须由协议规定的证据产生。

## 0. v0.3 变更摘要

相对 v0.2，本版在任何拟议方法训练或结果读取前完成以下变更：

1. 将 V1 统一定义为双侧尾统计池化（Bilateral Tail-Statistic Pooling, BTSP）。
2. 关闭 2×2、top-2/bottom-2 条件下 mean 的代数冗余，不再设置独立中心输出分支。
3. H1 改为直接检验 V1 下尾相对参数匹配上尾单支的独立贡献；V0 只作 hard-extrema 诊断。
4. H2 保留低 image-domain CNR 主要条件、细宽度次要条件、四类退化等权 Macro-rAUC 和 −0.003 总体非劣界。
5. V0 不再具有独立投稿资格；最低投稿版本为同时通过 H1 与 H2 的 V1。
6. Avg-TopK、T-Max-Avg 和最强公平非拟议池化基线进入投稿资格必需对照。
7. 结果前标题删除 Robust；只有 H1 与 H2 均通过后才可加入 Robust 或 Low-Contrast。
8. 投稿梯队固定为 SPIC 主投、PRL 强结果压缩路线、MVA 工程备选；期刊偏好不得改变事实主张。
9. 数据角色、数据哈希、第二骨干、效应门槛、统计单位、退化宏平均和 test 锁不变。

## 1. 一句话论证、标题与术语

### 1.1 一句话论证

> 在二维 RGB 表面裂缝二值语义分割中，本研究检验局部上、下尾秩摘要的双侧融合是否能相对于上尾单支、hard extrema 与既有 Max–Avg/Top-K 池化保留更多低 CNR 细裂缝证据，并评估其在受控退化和跨骨干条件下的稳定性；证据限于预注册数据、模型和指标。

这是待检验论证，不是结果陈述。

### 1.2 结果条件化标题

预结果工作标题：

> Bilateral Tail-Statistic Pooling for Two-Dimensional RGB Surface Crack Segmentation

只有 H1 与 H2 均通过后，才允许使用：

> Robust Bilateral Tail-Statistic Pooling for Low-Contrast Surface Crack Segmentation

若 H2 的低 CNR 主要条件失败，标题、摘要和结论不得出现 Robust、Low-Contrast 或等价措辞。

### 1.3 目标期刊梯队

1. Signal Processing: Image Communication：当前主投；突出图像特征聚合、低 CNR/退化稳定性和完整实验。
2. Pattern Recognition Letters：结果强且能压缩时冲刺；必须正面区分其既有裂缝 Mixed Pooling。
3. Machine Vision and Applications：工程备选；突出稀疏细长缺陷、跨场景、效率与失败边界。

期刊梯队只改变篇幅和叙事重点，不得改变数据角色、效应门槛、指标或结果解释。

### 1.4 规范术语

| 规范术语 | 本协议定义 | 禁止或限制用法 |
|---|---|---|
| 二维 RGB 表面裂缝二值语义分割 | RGB 图像上裂缝/背景像素分类 | 不称 BEV、LiDAR、多模态或 3D 感知 |
| V0 静态三统计量池化 | Max、Mean、Min 的输入无关可训练融合 | 只作诊断，不称主要创新 |
| BTSP | 2×2 窗口内 top-2 mean 与 bottom-2 mean 的双侧融合 | 不称新的基础池化家族 |
| 上尾单支 | 仅使用 top-2 mean；等价于本设置的 Avg-TopK（K=2） | H1 必须直接比较 |
| 静态 BTSP | 融合权重训练后固定，不依赖当前输入 | 不称内容自适应 |
| 内容相关 BTSP | 融合权重由当前局部描述生成 | 只有通过 H3 才构成附加贡献 |
| 鲁棒 BTSP | H1、H2 均通过后的结果性称谓 | 结果前只称 BTSP |
| image-domain CNR proxy | 由 GT 前景与邻域背景计算的图像域困难度代理 | 不等同物理 SNR 或成因 |
| 外部边界测试 | CFD 上不调参的边界式评估 | 不与核心区域式 FgIoU 宏平均 |

结果前禁止“首次提出 Top-K/尾统计池化”“首次将混合池化用于裂缝分割”“三个独立尾—中心分支”“通用池化家族”“SOTA”“鲁棒”“轻量”“实时”等无证据表述。

## 2. 预注册假设与顺序门控

### H1：下尾统计具有独立贡献

问题：控制模块结构和参数量后，V1 中加入 bottom-2 mean 是否优于只含 top-2 mean 的上尾单支？

- 方法：V1 静态 BTSP。
- 主对照：参数匹配上尾单支。该对照保留相同模块和参数登记，但将下尾输入替换为上尾输入，因此输出严格退化为 \(U_2\)，不得引入其他信息。
- V0 Max–Mean–Min 对参数匹配 Max+Mean 只作诊断支持，不决定 H1。
- 主要估计量：

\[
\Delta_{H1}=FgIoU(V1)-FgIoU(Matched\ UpperTail).
\]

支持 H1 必须同时满足：

1. CRACK500 group-clean 与 DeepCrack537 group-clean 的效应方向均为正；
2. 两核心数据集等权宏平均提升不低于 0.005 FgIoU；
3. 分组配对 bootstrap 95% CI 下限大于 0；
4. 至少 4/5 个配对种子方向一致；
5. 删除或复制下尾后性能可重复下降，不能只以学习到的 \(\alpha\) 数值证明贡献；
6. 增益不能由插入位置、训练预算、参数量或单数据集调参解释。

### H2：BTSP 改善低 CNR 与受控退化稳定性

问题：相对 hard-extrema V0，V1 是否在保持干净总体非劣的同时，主要改善低 image-domain CNR，并在细宽度和四类退化上给出一致支持？

- 方法：V1 静态 BTSP。
- 主要对照：V0。
- 最近邻资格对照：Avg-TopK、T-Max-Avg 与最强公平非拟议池化基线。
- 主要困难分层：低 CNR。
- 有序次要分层：细/中/粗相对宽度。
- 退化汇总：gamma、Gaussian blur、Gaussian noise、JPEG 四类全部等权进入 Macro-rAUC。

对数据集 \(d\)：

\[
\Delta^{(d)}_{H2,low}
=\overline{FgIoU}^{V1}_{low}
-\overline{FgIoU}^{V0}_{low}.
\]

两核心数据集主要效应：

\[
\Delta_{H2,low}
=\frac{\Delta^{(CRACK500)}_{H2,low}
+\Delta^{(DeepCrack537)}_{H2,low}}{2}.
\]

支持 H2 必须同时满足：

1. 干净总体 FgIoU 相对 V0 非劣，差值 95% CI 下限高于 −0.003；
2. \(\Delta_{H2,low}\ge0.005\)，95% CI 下限大于 0，且两个核心数据集方向均为正；
3. 有序次要点估计满足 \(\Delta_{fine}\ge\Delta_{medium}\ge\Delta_{coarse}\)，并报告 \(\Delta_{fine}-\Delta_{coarse}\) 的 95% CI；该证据不能替代低 CNR 主要条件；
4. 四类退化全部进入等权 Macro-rAUC，V1−V0 宏平均差值为正并报告 95% CI；
5. V1 相对 T-Max-Avg 与最强公平非拟议池化基线的两数据集等权宏平均点估计为正，且任一核心数据集不存在超过 0.003 FgIoU 的不利差值；
6. 不删除不利退化、数据集、种子或失败样本。

H1 已覆盖 V1 对 Avg-TopK/上尾单支的直接检验。

### H3：内容相关融合具有附加价值

问题：让 BTSP 融合权重随当前局部特征变化，是否比静态 BTSP 提供稳定、可解释且成本受控的附加收益？

- 方法：V2 内容相关 BTSP。
- 主对照：V1，以及参数匹配但不依赖输入的静态门控。
- 支持条件：两核心数据集等权宏平均 FgIoU 至少提高 0.003；95% CI 下限大于 0；两数据集及第二骨干方向均为正；门值随 CNR、宽度或退化出现可重复变化；参数匹配静态门控不能获得同等增益；P50 延迟和峰值显存相对 V1 增量均不超过 10%。

### 2.1 顺序门控

H1→H2→H3 顺序推进：

- H1 未通过：V1 不具备论文主方法资格，V0 也不得替代为主创新；停止确认性方法扩张。
- H2 未通过：不得使用 Robust/Low-Contrast 主张，也不得把 V2 包装为主贡献；只能降级为技术报告、负结果或终止。
- H3 未通过：回退到通过 H1 与 H2 的 V1。
- 新注意力、新损失、边界模块、Transformer 或 Mamba/SSM 不能绕过门控。

## 3. 方法版本边界

### 3.1 V0：hard-extrema 诊断基线

\[
y=q_{max}\max(x)+q_{mean}\operatorname{mean}(x)+q_{min}\min(x),
\qquad q=\operatorname{softmax}(\theta).
\]

- 每个下采样层使用共享静态权重；
- 权重不依赖当前输入；
- 只检验 hard lower-tail 信息是否具有诊断价值；
- V0 不具有独立投稿资格。

### 3.2 V1：静态 BTSP

对 2×2、stride 2、无 padding 的局部窗口，将响应按升序记为

\[
x_{(1)}\le x_{(2)}\le x_{(3)}\le x_{(4)}.
\]

冻结：

\[
U_2=\frac{x_{(3)}+x_{(4)}}{2},\qquad
L_2=\frac{x_{(1)}+x_{(2)}}{2}.
\]

恒有：

\[
C_2=\frac{U_2+L_2}{2}=\operatorname{mean}(x),\qquad
D_2=U_2-L_2.
\]

因此 mean 不是独立第三输出分支。V1 输出固定为：

\[
y_l=\alpha_l U_2+(1-\alpha_l)L_2,\qquad
\alpha_l=\sigma(\theta_l).
\]

实现边界：

- 每个下采样层仅学习一个标量 \(\theta_l\)，在通道和空间位置间共享；
- \(\theta_l\) 初始化为 0，即 \(\alpha_l=0.5\)；
- 四个下采样层各自具有独立 \(\theta_l\)；
- 主插入位置为现有 MaxPool 注册位置，即编码块输出之后；所有公平基线使用相同位置；
- pre-activation 或 normalization-adjacent 位置只作预注册机制敏感性，不能替代主结果；
- 不同时加入注意力、新损失、边界模块或额外训练预算；
- TailMeanPool2d 原语保持不变，源文件 SHA-256 为 8954238024e9ab9598c2c848b0482af672ec07d754381b8685a6748c06c3a842。

top-2 单支与 Avg-TopK（K=2）等价，必须作为直接对照。

### 3.3 V2：内容相关 BTSP

V2 只有在 validation 阶段 V1 达到 Go 条件后才允许进入候选实现。局部描述固定为：

\[
C_2=(U_2+L_2)/2,\qquad
D_2=U_2-L_2,\qquad
S_2=\sqrt{\operatorname{Var}(x)+10^{-6}}.
\]

每个下采样层使用四个共享标量：

\[
\alpha_l(x)=\sigma(w_C C_2+w_D D_2+w_S S_2+b).
\]

权重在通道和空间位置间共享，门值随当前局部窗口变化。禁止高容量 MLP、独立空间注意、Transformer、Mamba/SSM 或额外分支。

参数匹配静态对照保留同样四个标量，但不接收 \(C_2,D_2,S_2\)，只生成输入无关常数门值。V2 与静态对照必须使用相同训练预算。

## 4. 数据角色与冻结产物

### 4.1 固定角色

| 数据集 | 角色 | 允许支撑的结论 |
|---|---|---|
| CRACK500 group-clean | 核心开发与确认之一 | retained-crop raster、道路表面域内证据 |
| DeepCrack537 group-clean | 第二核心确认 | 多表面/跨场景证据，不等同纯道路证据 |
| CFD labeled 001–118 | 冻结 external boundary test | 道路边界外测，不参与核心 FgIoU 宏平均 |
| GAPs384 | HOLD | 不训练、不验证、不测试、不进入叙事 |

CrackLS315 与 OmniCrack30k 不进入最低实施路线；未来加入必须创建新协议版本。

### 4.2 核心冻结绑定

| 对象 | 路径 | SHA-256 |
|---|---|---|
| CRACK500 30 s group-clean manifest | phase2_data_audit/manifests/crack500_group_clean_30s_groupv3_v7_r2_iter3_seed20260821.csv | f9f186d45d9db9600666fdad9fce8f43ff79e4cf36f350b7122bd874688ba825 |
| DeepCrack537 group-clean manifest | phase2_data_audit/manifests/deepcrack537_group_clean_groupv3_v7_r2_iter1_seed20260821.csv | 6ea69131264f046b5a6f2aa65d2aa58e3699746e3945082f8c1dacaa21da7099 |
| CRACK500 10 s grouping sensitivity | phase2_data_audit/manifests/crack500_grouping_sensitivity_10s_bound_to_30s_v1.csv | 89a20fb7f74166d222f1c5b527b21d9a8d6b05ffe9eff8144559d57e52d80fcf |
| CFD boundary manifest | phase2_data_audit/manifests/cfd_labeled001_118_boundary_v1.csv | 1c415ee296b93a36e5cfb37691352ffd4942f18da9713744e297b6c543959cc6 |
| CRACK500 difficulty thresholds | baseline_platform/artifacts/difficulty_thresholds/crack500_train_group_difficulty_v1.json | f83f8c5fa21e01de5a415a65a5cbdb230834ab9d1e08d2806422b6e55e26a6c6 |
| DeepCrack537 difficulty thresholds | baseline_platform/artifacts/difficulty_thresholds/deepcrack537_train_group_difficulty_v1.json | b330df73f6cdb7c6d547e129b9326d53957599baffd713e2c3d06982028544b3 |
| 第二骨干与 k 冻结 | baseline_platform/artifacts/pretrain_method_freeze_v1.json | e71b53b5c187631bf05ce861141d185e7170665acc8528b811f9b2a68255efa0 |
| 效应门槛冻结 | baseline_platform/artifacts/effect_size_decision_freeze_v1.json | e312669f0901b386d439de6374c22624f06bbb51a7f52b4878cd7ed0b05a818c |
| P1 方法—创新修订冻结 | baseline_platform/artifacts/p1_method_novelty_alignment_freeze_v1.json | ed72b8ecc4309af15922c775f047f63554b3853fe29733f221ae7dc94d9c61df |

v0.2 的 group/source-family 构建、native-grid 评估、CFD 042/065 双敏感性和数据发表/再分发条款原样继承。

## 5. 防泄漏、开发阶段与 test 锁

- group/source-family 划分必须先于裁剪、缩放和增强；
- CRACK500 以 30 s acquisition-sequence group 为主统计和 bootstrap 单位；10 s 只重算敏感性 CI，不生成新 split；
- DeepCrack537 以 source family 为 split、统计和 bootstrap 单位；
- 归一化、类别权重、CNR/宽度阈值只能由 train groups 计算；
- validation 只用于预注册 Go/No-Go、checkpoint 和有限 V2 资格判断；
- test、CFD 与 GAPs384 不用于方法、epoch、阈值、退化或叙事选择；
- 预测阈值主值固定为 0.5；validation 最优阈值只能作标明的附加敏感性；
- checkpoint 固定为最高 validation FgIoU，平局选择最早 epoch；
- 确认种子固定为 2026、2027、2028、2029、2030，配对方法共享初始化、数据顺序与增强随机流；
- 方法实现、训练预算、指标脚本、统计脚本和本协议执行授权全部冻结后，才允许启动拟议方法训练；
- validation Go 条件关闭、实现哈希不变且作者形成单独 test-unlock 记录后，才允许读取核心 test；
- CFD 只能在核心方法与主张完全冻结后运行。

任何实现修复、故障重跑或协议偏离必须写入版本化 decision log；不得因结果不利而重跑、删种子、改分层或改主张。

## 6. 网络、基线、指标与统计

### 6.1 网络

- 主骨干：vanilla 四层 U-Net，base channels 32；
- 第二骨干：pool_adapter_resnet18_unet_base32_v1，无预训练权重；
- loss：固定 BCE + Dice；
- 两骨干所有比较使用同一池化插入位置，不改解码器或 loss。

optimizer、learning rate、schedule、epoch、有效 batch size 和增强细节尚未冻结，因此本协议不授权训练。

### 6.2 P0 同构对照

1. MaxPool、AvgPool、MinPool、stride-2 convolution；
2. 固定 0.5 Max–Avg 与静态可学习 Max–Avg；
3. Avg-TopK，即 V1 上尾单支；
4. T-Max-Avg；
5. V0 及参数匹配去下尾变体；
6. V1 BTSP 及参数匹配上尾单支；
7. V2 及参数匹配静态门控；
8. 在非拟议方法中表现最强的公平池化基线。

P1 最近邻包括 PRL Mixed Pooling、Gated Scale Pooling、MAM、MinMax/HPD、SoftPool、OWA/CombPool。至少引用并进入补充比较；不能复现时必须提供可审计理由，且不得作覆盖性优越主张。

所有比较共享 split、输入、预训练状态、loss、增强、optimizer、schedule、epoch、seed、有效 batch size、checkpoint、插入位置和调参预算。不得把整网结构变化归因于单个池化算子。

### 6.3 指标

主指标：

\[
FgIoU=\frac{TP}{TP+FP+FN}.
\]

- CRACK500 在 retained-crop native raster 上计数后按 30 s group 汇总；
- DeepCrack537 在完整图像上计数后按 source family 汇总；
- 两数据集分别估计后等权宏平均，不按像素、crop、图像或族大小跨数据集加权；
- Pixel Accuracy 只能附带，不能支撑 H1–H3。

次指标：Dice/F1、Precision、Recall、AUPRC、Boundary-F1、clDice，以及 Params、MACs、峰值显存、模型大小和 P50/P95 延迟。

CFD 固定概率阈值 0.5，按 Berkeley correspondPixels 一对一匹配，最大距离为图像对角线的 0.01；主报等图像 Boundary-F1/Precision/Recall，并并列 full118、exclude-042、exclude-065。CFD 不报告 ODS/OIS/AP。

### 6.4 困难分层与退化

- CNR 与宽度阈值只由各数据集 train groups 冻结；
- 低 CNR 为 H2 主要条件；
- 细/中/粗宽度为有序次要条件；
- gamma、Gaussian blur、Gaussian noise、JPEG 四类退化全部进入等权 Macro-rAUC；
- 不得因某类退化不利而删除或改权。

### 6.5 统计

- 五个配对确认种子：2026–2030；
- 两级配对 bootstrap：先重采样种子，再在种子内重采样来源组；
- bootstrap 10,000 次，seed 424242；
- 每次重算组内混淆、指标、方法差值、分层效应和跨数据集等权宏平均；
- CRACK500 主分析使用 30 s group，10 s 只作预注册敏感性；
- H1→H2→H3 控制主要假设顺序；探索性次指标使用 Holm–Bonferroni 或明确标为探索性；
- 主要解释依赖效应量、95% CI、跨种子和跨数据集一致性，不只依赖 p 值。

## 7. Go/No-Go 与最低投稿门槛

### 7.1 V1 下尾资格门

validation Go：

- CRACK500 三个探索种子至少 2/3 个 V1 优于参数匹配上尾单支；
- 平均 FgIoU 提升至少 0.005；
- 删除下尾后低 CNR 指标可重复下降；
- 增益不由单个异常运行或训练不稳定驱动。

No-Go：

- 提升小于种子波动；
- 下尾删除无损或融合长期退化为上尾单支；
- 不能超过 Avg-TopK 或 T-Max-Avg；
- 只改善 Pixel Accuracy。

### 7.2 V1→V2 资格门

validation Go：

- V1 总体相对 V0 非劣；
- 低 CNR 探索效应达到 0.005；
- 四类退化 Macro-rAUC 方向为正；
- 三个探索种子至少 2/3 同方向；
- 无明显异常点敏感性或训练不稳定。

No-Go：

- 只有单数据集特调有效；
- 干净总体下降超过 0.003；
- 低 CNR 主要方向失败；
- BTSP 不优于 hard extrema 或 T-Max-Avg；
- 下尾无独立贡献。

### 7.3 最低投稿门槛

- V0 不具有独立投稿资格；
- V1 为投稿主方法：H1 与 H2 必须同时通过；
- V2 为投稿主方法：H1、H2、H3 必须全部通过；
- 无论主方法为何，还必须满足：两核心数据集及第二骨干方向一致；Avg-TopK、T-Max-Avg 和最强公平基线比较完整；效率成本与收益相称；至少一种失败模式明确报告；
- CFD 只能补充外部边界证据，不能替代核心假设；
- 若只在一个数据集取得小幅提升，降级为 Technical Note、负结果报告或终止，不包装为通用池化方法。

## 8. 效率、机制与失败审计

### 8.1 效率

- 固定 GPU、驱动、CUDA、cuDNN、PyTorch 和功耗模式；
- 输入固定 1×3×512×512，FP32 主报；
- batch 1 主报、batch 8 附加；
- 100 次预热、1,000 次正式测量、GPU 同步、独立进程重启 3 次；
- 报告 Params、MACs、模型大小、峰值显存、P50/P95 模型延迟和端到端延迟。

### 8.2 机制与失败

- V0 记录 hard-min 零值率；
- V1 记录每层 \(\alpha_l\)、上/下尾分支幅度与下尾删除效应；
- V2 记录门值均值、方差、熵及其随 CNR、宽度和退化的变化；
- 比较主插入位置与预注册机制敏感性位置；
- 报告断裂、阴影、标线、接缝、污渍、纹理噪声等失败类型；
- 深层下尾响应不解释为暗裂缝、凹陷或物理深度；
- 可视化不能替代公平消融和统计检验。

## 9. 学术结果发表与再分发边界

技术完整性、非商业学术汇总结果发表和原始数据再分发分别判定：

- CRACK500：非商业学术汇总结果 GO WITH ATTRIBUTION；原图、掩膜、数据包再分发 HOLD；样例图与权重 REVIEW。
- DeepCrack537：非商业研究/教育汇总结果 GO WITH CITATION；数据集再分发 HOLD；样例图与权重 REVIEW。
- CFD：非商业学术汇总结果 GO WITH ATTRIBUTION；原始数据再分发 HOLD；外测仍受执行锁控制。
- 代码仓库不得包含第三方原始数据副本；Data Availability 指向原作者入口并规范引用。

公开数据不等于自动获得转载或再分发许可，但不妨碍在遵守来源条款和引用要求的前提下发表自行产生的非商业汇总实验结果。

## 10. 冻结链与执行开放项

本协议的 SHA-256 只记录在外部冻结登记 baseline_platform/artifacts/preregistered_protocol_v0.3_freeze.json 中，避免自引用哈希。

方法学冻结已关闭：

- 任务和数据角色；
- H1–H3、效应门槛和顺序门控；
- V0/V1/V2 定义及 BTSP 代数边界；
- P0/P1 对照和最低投稿门槛；
- 指标、分层、退化、统计单位与 bootstrap；
- 目标期刊梯队和结果条件化标题；
- 核心 test 与 CFD 外测锁。

执行仍保持 LOCKED，必须依次关闭：

1. 实现 V1、参数匹配上尾单支、T-Max-Avg 与必要注册接口，完成单元和运行时测试并冻结源文件哈希；
2. 若进入 V2，实现本协议定义的四标量内容门控及参数匹配静态对照；
3. 冻结 optimizer、learning rate、schedule、epoch、有效 batch size、增强和完整训练配置；
4. 完成 10 s grouping、困难阈值、CFD 视觉确认与 Berkeley 官方匹配内核的剩余复核；
5. 生成独立 train/validation execution authorization；没有该登记不得启动训练；
6. validation Go/No-Go 完成后，另行生成 test-unlock 记录；没有该记录不得读取核心 test；
7. 核心方法和主张全部冻结后，另行授权 CFD 外测。

任何冻结后偏离必须在结果读取前写入版本化 decision log，说明原因、受影响假设和是否需要全量重跑；不得根据结果追溯性修改假设、数据角色、统计单位、CNR 阈值、退化宏平均或投稿门槛。
