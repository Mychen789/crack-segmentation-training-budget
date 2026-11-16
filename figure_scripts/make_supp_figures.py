"""English copies of the two supplemental figure scripts (Figs. S1 and S2); plotted data are unchanged."""
from pathlib import Path
import runpy, sys

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parents[1] / 'tools_fig'

fig3 = {
    '"合成裂缝图像（示意）"': '"Synthetic crack image (schematic)"',
    '"局部放大（24 × 24 像素）"': '"Local crop (24 × 24 pixels)"',
    '"非负响应图（示意）"': '"Nonnegative response (schematic)"',
    '"排序"': '"sort"',
    '"较小两值为 0"': '"Two smallest = 0"',
    'f"全图 {share_zero:.0%} 的窗口差值为 0"': 'f"Zero difference in {share_zero:.0%} of windows"',
    '"局部响应图经 2 × 2、步长 2 下采样后的输出（蓝框对应上方窗口）"':
        '"Outputs after 2 × 2, stride-2 downsampling of the response (box marks the window above)"',
}
fig8 = {
    '"结论相对 90 轮改变的算子对"': '"Pairs changed relative to 90 epochs"',
    '"检查判为可报告的算子对"': '"Pairs reportable by the check"',
    '"通过检查的训练"': '"Runs passing the check"',
    'f"B ≥ {b47}：\\n结论基本不变"': 'f"B ≥ {b47}:\\nstable"',
    'f"B ≥ {b70}：检查才放行过半的对"': 'f"B ≥ {b70}:\\nhalf pass"',
    '"预算 B（轮）"': '"Budget B (epochs)"',
    '"占比（算子对或训练）"': '"Proportion of pairs or runs"',
}
for name, table in (('fig3_operator_detail.py', fig3), ('fig8_check.py', fig8)):
    s = (TOOLS / name).read_text(encoding='utf-8')
    for old, new in table.items():
        assert old in s, (name, old)
        s = s.replace(old, new)
    assert not any('一' <= ch <= '鿿' for line in s.splitlines() if not line.lstrip().startswith(('#', '"""', 'Rules')) for ch in line), name
    (HERE / name).write_text(s, encoding='utf-8')

sys.path.insert(0, str(HERE))
for name in ('fig3_operator_detail.py', 'fig8_check.py'):
    runpy.run_path(str(HERE / name), run_name='__main__')
    print('Rendered', name)
