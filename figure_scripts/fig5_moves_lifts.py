"""Fig. 5: pairwise moves (a) and the per-operator lifts that produce them (b), four cells.

Eq. (1): move(a, b) = lift(a) - lift(b), so the spread in (b) bounds the moves in (a).
"""

import matplotlib.pyplot as plt
import numpy as np

from style3 import (BAND, CELLS, CHANGED, FULL_W, MM, OPER, THRESH, C, arm_means, panel,
                    load_histories, operators, pairs, save, tag)

h = load_histories()
rng = np.random.default_rng(0)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_W, 60 * MM), sharey=True,
                               gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.08})
fig.subplots_adjust(left=0.2, right=0.9, bottom=0.26)
t = THRESH * 1e3
n = len(CELLS)

for k, (cell, label, lo, hi) in enumerate(CELLS):
    y0 = n - 1 - k
    m_lo, m_hi = arm_means(h, cell, lo), arm_means(h, cell, hi)
    ops = operators(h, cell)
    lift = {o: (m_hi[o] - m_lo[o]) * 1e3 for o in ops}

    # (a) every pair, oriented so that the low-budget difference is >= 0
    mv, flip = [], []
    for a, b in pairs(ops):
        d_lo, d_hi = m_lo[a] - m_lo[b], m_hi[a] - m_hi[b]
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
    ax1.text(23.5, y0, f"{cross.sum()}/55", va="center", ha="right", fontsize=6.5,
             color=CHANGED if cross.sum() else C["ink"])

    # (b) the 11 lifts
    v = np.array(list(lift.values()))
    ax2.plot([v.min(), v.max()], [y0, y0], color=C["light"], lw=5, solid_capstyle="round", zorder=1)
    ax2.scatter(v, np.full_like(v, y0), s=11, facecolor="white", edgecolor=C["dark"],
                linewidth=0.6, zorder=3)
    ax2.scatter([lift["min"]], [y0], s=13, color=OPER, zorder=4)
    ax2.plot([v.mean()] * 2, [y0 - 0.25, y0 + 0.25], color=C["ink"], lw=1.0, zorder=2)
    ax2.text(39.5, y0, f"{v.max() - v.min():.1f}", va="center", ha="right", fontsize=6.5)

    ds, bb = label.split(" / ")
    ax1.text(-26, y0, f"{ds} / {bb}\n{lo} → {hi} epochs", va="center", ha="right",
             fontsize=6.5, linespacing=1.25)

# (a) axes
ax1.axvspan(-t, t, color=BAND, lw=0, zorder=0)
ax1.axvline(0, color=C["mid"], lw=0.5, zorder=1)
ax1.set_xlim(-24, 24)
ax1.set_xticks([-20, -10, 0, 10])
ax1.spines["bottom"].set_bounds(-20, 12)
ax1.set_xlabel("Shift in paired difference (×10⁻³)", x=0.42)
ax1.text(23.5, n - 0.45, "Changed", ha="right", va="bottom", fontsize=6.3, color=C["dark"])
ax1.text(0, -0.75, "±0.005", ha="center", va="center", fontsize=6.0, color=C["dark"])
panel(ax1, "a", "Shifts for 55 operator pairs")

# (b) axes
ax2.set_xlim(0, 40)
ax2.set_xticks([0, 10, 20, 30])
ax2.spines["bottom"].set_bounds(0, 34)
ax2.set_xlabel("Gain from extra training (×10⁻³)", x=0.42)
ax2.text(39.5, n - 0.45, "Gain range", ha="right", va="bottom", fontsize=6.3, color=C["dark"])
panel(ax2, "b", "Gains for 11 operators")

for ax in (ax1, ax2):
    ax.set_ylim(-1.0, n - 0.3)
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

save(fig, "fig5_moves_lifts")
