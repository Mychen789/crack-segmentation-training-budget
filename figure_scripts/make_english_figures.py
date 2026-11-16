"""Translate copies of the six established figure scripts without changing data."""
from pathlib import Path
import shutil, runpy, sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ZH_WORK = ROOT / 'draft_zh/docx_src_jcce/figure_scripts'
TOOLS = ROOT / 'tools_fig'
HERE.mkdir(exist_ok=True, parents=True)

files = {
    'fig1_concept.py': ZH_WORK / 'fig1_concept.py',
    'fig2_architecture.py': TOOLS / 'fig2_architecture.py',
    'fig4_ranks.py': ZH_WORK / 'fig4_ranks.py',
    'fig5_moves_lifts.py': TOOLS / 'fig5_moves_lifts.py',
    'fig6_schedules.py': TOOLS / 'fig6_schedules.py',
    'fig7_rules.py': TOOLS / 'fig7_rules.py',
}

shared = {
    '固定 max+avg': 'Fixed max+avg', '可学 max+avg': 'Learned max+avg',
    '对角划分': 'Diagonal partition', '步长卷积': 'Strided conv.',
    '训练轮数': 'Training epoch', '验证集前景 IoU（MA(5)）': 'Validation foreground IoU, MA(5)',
    '预算 B（轮）': 'Budget B (epochs)', '两个算子的学习曲线': 'Learning curves',
    '同一对比较在不同预算下的结论': 'Comparison across budgets',
    '±0.005 最小有意义差值': 'Minimum meaningful difference: ±0.005',
    '（实线）': ' (solid)', '（虚线）': ' (dashed)',
    '配对差 Δ（': 'Paired difference Δ (', '）': ')',
    '其余 9 个算子': 'Other 9 operators', 'UNet：学习曲线': 'UNet learning curves',
    '：各预算下的名次': ': ranks across budgets', '名次（1 为最好)': 'Rank (1 = best)',
    ' 轮': ' epochs',
    '配对差变化：高预算 − 低预算（×10⁻³)': 'Shift in paired difference (×10⁻³)',
    '各算子的提升：高预算 − 低预算（×10⁻³)': 'Gain from extra training (×10⁻³)',
    '配对差变化：90 轮 − 30 轮（×10⁻³)': 'Shift in paired difference (×10⁻³)',
    '各算子的提升：90 轮 − 30 轮（×10⁻³)': 'Gain from 30 to 90 epochs (×10⁻³)',
    '结论改变': 'Changed', '提升极差': 'Gain range',
    '55 个算子对的配对差变化': 'Shifts for 55 operator pairs',
    '11 个算子各自的提升': 'Gains for 11 operators',
    '超过 0.005（Changed)': '>0.005 (changed)', '不超过 0.005': '≤0.005',
    '高预算下变号': 'Sign flip at high budget', '90 epochs时变号': 'Sign flip at 90 epochs',
    '11 个算子的均值': 'Mean of 11 operators',
    '恒定学习率\\nMA(5) 读数': 'Constant learning rate\\nMA(5) read-out',
    '余弦调度\\nMA(5) 读数': 'Cosine schedule\\nMA(5) read-out',
    '余弦调度\\n最后一轮读数': 'Cosine schedule\\nLast-epoch read-out',
}

architecture = {
    '被比较的下采样算子': 'Compared operator', '编码块': 'Encoder block',
    'UNet：每级两层卷积': 'UNet: 2 convolutions/level',
    'ResNet18-UNet：E₀ 一层卷积，': 'ResNet18: 1 convolution at E₀;',
    'E₁–E₄ 各两个残差块': '2 residual blocks at E₁–E₄',
    'ConvNeXt-UNet：每级两个': 'ConvNeXt: 2 blocks/level',
    'ConvNeXt 块': '', '输入图像': 'Input image', '上采样': 'Upsample',
    '解码块': 'Decoder', '跳跃连接（拼接)': 'Skip connections (concatenation)',
    '解码通路': 'Decoder path', '1×1 卷积': '1×1 conv.',
    '输出掩膜（示意)': 'Output mask (schematic)',
    '每次训练在 4 个位置使用同一种算子，共 11 种': 'One of 11 operators is used at all 4 locations in each run',
}

rules = {
    '现行规则（基线)': 'Current rule (baseline)',
    '收敛门控，': 'Convergence gate, ', '各自收敛点，6 组': 'Individual convergence, 6 settings',
    '末窗均值，': 'Tail mean, ', '门控加配对符号，4 组': 'Gate + paired signs, 4 settings',
    '指数拟合，全段，4 组': 'Exponential fit, all, 4 settings',
    '指数拟合，后半段，4 组': 'Exponential fit, half, 4 settings',
    '幂律拟合，全段，': 'Power-law fit, all, ',
    '幂律拟合，后半段，4 组': 'Power-law fit, half, 4 settings',
    '两点外推，': 'Two-point extrapolation, ', '，κ': ', κ',
    '覆盖率\\n': 'Coverage\\n', '不稳定度 ÷ R0\\n': 'Instability / R0\\n',
    '保真度\\n': 'Fidelity\\n', '区分度\\n': 'Discriminability\\n',
    '未设': 'Not set', '基线': 'Baseline', '"是"': '"Yes"', '"否"': '"No"',
    '四条\\n全部达标': 'All four\\ncriteria met', '第一轮': 'Round 1', '第二轮': 'Round 2',
}

for name, source in files.items():
    s = source.read_text(encoding='utf-8')
    for old, new in shared.items(): s=s.replace(old,new)
    for old, new in (architecture if name=='fig2_architecture.py' else rules if name=='fig7_rules.py' else {}).items(): s=s.replace(old,new)
    # Full-width export gives English panel titles space without altering the plotted values.
    if name=='fig1_concept.py':
        s=s.replace('FULL_W * 0.82', 'FULL_W')
        s=s.replace('Minimum meaningful difference: ±0.005','±0.005 meaningful difference')
    if name=='fig4_ranks.py':
        s=s.replace('figsize=(FULL_W, 64 * MM)', 'figsize=(FULL_W * 1.12, 68 * MM)')
        s=s.replace('wspace=0.62', 'wspace=0.82')
        s=s.replace('f"{title}: ranks across budgets"','f"{title} ranks"')
    if name=='fig2_architecture.py':
        s=s.replace('12.5, 9.5, 12.5, 5.5', '12.5, 9.5, 16.5, 5.5')
    if name in ('fig5_moves_lifts.py','fig6_schedules.py'):
        s=s.replace('ncol=5', 'ncol=3').replace('bbox_to_anchor=(0.55, -0.02)', 'bbox_to_anchor=(0.55, -0.10)')
        s=s.replace('Mean of 11 operators','Mean')
        s=s.replace('Sign flip at high budget','Sign flip').replace('Sign flip at 90 epochs','Sign flip')
        s=s.replace('配对差变化：90 epochs − 30 epochs（×10⁻³)', 'Shift in paired difference (×10⁻³)')
        s=s.replace('各算子的提升：90 epochs − 30 epochs（×10⁻³)', 'Gain from 30 to 90 epochs (×10⁻³)')
    if name=='fig7_rules.py':
        s=s.replace('figsize=(FULL_W * 0.78, 92 * MM)', 'figsize=(FULL_W * 1.05, 92 * MM)')
        s=s.replace('ax.text(-3.55, mid, name,', 'ax.text(-4.5, mid, name,')
        s=s.replace('ax.set_xlim(-3.6, ncol + 0.2)', 'ax.set_xlim(-4.6, ncol + 0.2)')
    (HERE/name).write_text(s,encoding='utf-8')

for name in ('figstyle.py','style3.py'):
    s=(TOOLS/name).read_text(encoding='utf-8')
    if name=='figstyle.py':
        s=s.replace('ROOT = Path(__file__).resolve().parents[1]', 'ROOT = Path(__file__).resolve().parents[2]')
        s=s.replace('OUT = ROOT / "figures"', 'OUT = ROOT / "draft_en" / "figures"')
    for old,new in shared.items(): s=s.replace(old,new)
    s=s.replace('["Times New Roman", "SimSun"]','["Times New Roman"]')
    (HERE/name).write_text(s,encoding='utf-8')

sys.path.insert(0,str(HERE))
for name in files:
    runpy.run_path(str(HERE/name),run_name='__main__')
    print('Rendered',name)
