"""Fig. 6 (S21): budget sensitivity on DeepCrack537 / UNet under the constant and the cosine schedule.

Same layout as the moves/lifts figure: (a) the 55 pairwise moves from 30 to 90 epochs,
(b) the 11 per-operator lifts. Rows: constant LR with the MA(5) reading, cosine with the
MA(5) reading, cosine with the last-epoch reading. Cosine runs are records/S21.
"""

import json
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

from style3 import BAND, CHANGED, FULL_W, MM, OPER, ROOT, THRESH, C, panel, save

OPS = ["max", "avg", "min", "avg_top2", "fixed_max_avg", "learned_max_avg", "t_max_avg",
       "v0_max_mean_min", "btsp", "diag_partition_matched", "stride_conv"]
SEEDS = [2026, 2027, 2028]
COS = ROOT / "data" / "runs-cosine" / "s21_collect"
CONST = ROOT / "data" / "runs"


def curve(path, budget=None):
    h = json.loads(path.read_text(encoding="utf-8"))["history"]
    c = np.array([e["validation"]["foreground_iou"] for e in h])
    return c[:budget] if budget else c


def ma5(c):
    return np.convolve(c, np.ones(5) / 5, mode="valid").max()


def means(path_of, reader):
    return {o: np.mean([reader(path_of(o, s)) for s in SEEDS]) for o in OPS}


const = lambda o, s: CONST / f"b90_unet_gr_deepcrack537_{o}_s{s}.json"
cos = lambda b: (lambda o, s: COS / f"cos{b}_unet_gr_deepcrack537_{o}_s{s}.json")
rows = [
    ("Constant learning rate\nMA(5) read-out", means(const, lambda p: ma5(curve(p, 30))), means(const, lambda p: ma5(curve(p)))),
    ("Cosine schedule\nMA(5) read-out", means(cos(30), lambda p: ma5(curve(p))), means(cos(90), lambda p: ma5(curve(p)))),
    ("Cosine schedule\nLast-epoch read-out", means(cos(30), lambda p: curve(p)[-1]), means(cos(90), lambda p: curve(p)[-1])),
]

rng = np.random.default_rng(0)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_W, 52 * MM), sharey=True,
                               gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.08})
fig.subplots_adjust(left=0.17, right=0.9, bottom=0.3)
t = THRESH * 1e3
n = len(rows)

for k, (label, lo, hi) in enumerate(rows):
    y0 = n - 1 - k
    mv, flip = [], []
    for a, b in combinations(OPS, 2):
        d_lo, d_hi = lo[a] - lo[b], hi[a] - hi[b]
        if d_lo < 0:
            d_lo, d_hi = -d_lo, -d_hi
        mv.append((d_hi - d_lo) * 1e3)
        flip.append(d_hi < 0)
    mv, flip = np.array(mv), np.array(flip)
    cross = np.abs(mv) > t
    yj = y0 + rng.uniform(-0.22, 0.22, len(mv))
    for sel, fc, ec, mk in [(~cross & ~flip, "white", C["dark"], "o"),
                            (~cross & flip, "white", C["dark"], "v"),
                            (cross & ~flip, CHANGED, CHANGED, "o"),
                            (cross & flip, CHANGED, CHANGED, "v")]:
        ax1.scatter(mv[sel], yj[sel], s=10, marker=mk, facecolor=fc, edgecolor=ec,
                    linewidth=0.5, zorder=3)
    ax1.text(23.5, y0, f"{cross.sum()}/55", va="center", ha="right", fontsize=6.5, color=CHANGED)

    lift = np.array([(hi[o] - lo[o]) * 1e3 for o in OPS])
    ax2.plot([lift.min(), lift.max()], [y0, y0], color=C["light"], lw=5, solid_capstyle="round", zorder=1)
    ax2.scatter(lift, np.full_like(lift, y0), s=11, facecolor="white", edgecolor=C["dark"],
                linewidth=0.6, zorder=3)
    ax2.scatter([lift[OPS.index("min")]], [y0], s=13, color=OPER, zorder=4)
    ax2.plot([lift.mean()] * 2, [y0 - 0.25, y0 + 0.25], color=C["ink"], lw=1.0, zorder=2)
    ax2.text(39.5, y0, f"{np.ptp(lift):.1f}", va="center", ha="right", fontsize=6.5)
    ax1.text(-26, y0, label, va="center", ha="right", fontsize=6.5, linespacing=1.25)

ax1.axvspan(-t, t, color=BAND, lw=0, zorder=0)
ax1.axvline(0, color=C["mid"], lw=0.5, zorder=1)
ax1.set_xlim(-24, 24)
ax1.set_xticks([-20, -10, 0, 10])
ax1.spines["bottom"].set_bounds(-20, 12)
ax1.set_xlabel("Shift in paired difference (×10⁻³)", x=0.42)
ax1.text(23.5, n - 0.45, "Changed", ha="right", va="bottom", fontsize=6.3, color=C["dark"])
panel(ax1, "a", "Shifts for 55 operator pairs")

ax2.set_xlim(-10, 40)
ax2.set_xticks([-10, 0, 10, 20, 30])
ax2.spines["bottom"].set_bounds(-10, 34)
ax2.axvline(0, color=C["mid"], lw=0.5, zorder=0)
ax2.set_xlabel("Gain from 30 to 90 epochs (×10⁻³)", x=0.42)
ax2.text(39.5, n - 0.45, "Gain range", ha="right", va="bottom", fontsize=6.3, color=C["dark"])
panel(ax2, "b", "Gains for 11 operators")

for ax in (ax1, ax2):
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)

from matplotlib.lines import Line2D
handles = [
    Line2D([], [], ls="", marker="o", ms=3.6, mfc=CHANGED, mec=CHANGED, label=">0.005 (changed)"),
    Line2D([], [], ls="", marker="o", ms=3.6, mfc="white", mec=C["dark"], mew=0.6, label="≤0.005"),
    Line2D([], [], ls="", marker="v", ms=3.6, mfc="white", mec=C["dark"], mew=0.6, label="Sign flip"),
    Line2D([], [], ls="", marker="o", ms=3.8, mfc=OPER, mec=OPER, label="min"),
    Line2D([], [], color=C["ink"], lw=1.0, label="Mean"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.55, -0.10),
           handlelength=1.2, columnspacing=1.4)

save(fig, "fig6_schedules")
