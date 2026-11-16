"""把包内平铺的 data/runs/<名>.json 还原成 data/runs_flat/<名>/result.json。

两个重算脚本（budget_robust_reporting.py 及其 round2）预期的是
<runs_dir>/<run 名>/result.json 这个布局，而包里为了少一层目录是平铺的。
本脚本只做目录还原，不改任何内容。

画图不需要跑这个 —— data/per_epoch_validation.csv 已含全部逐轮数据。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
SRC = PKG / "data" / "runs"
DST = PKG / "data" / "runs_flat"


def main() -> int:
    if not SRC.is_dir():
        print(f"找不到 {SRC}", file=sys.stderr)
        return 1
    DST.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in sorted(SRC.glob("*.json")):
        target = DST / src.stem / "result.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        count += 1
    print(f"已还原 {count} 支 run 到 {DST}")
    print("现在可以这样重算：")
    print(f'  python tools/budget_robust_reporting.py --runs-dir "{DST}" '
          f'--cell design_dc537_unet --budget-low 30 --budget-high 90')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
