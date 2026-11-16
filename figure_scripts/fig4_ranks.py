"""Fig. 4: learning curves (a) and operator ranks across budgets as heatmaps (b, c), DeepCrack537."""

import statistics as st

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from style3 import (FOCUS, FULL_W, MM, OP_LABEL, OPER, PAL, SEEDS, C,
                    arm_means, load_histories, ma5_series, operators, panel, save)

h = load_histories()
fig = plt.figure(figsize=(FULL_W * 1.12, 68 * MM))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1], wspace=0.82)

# (a) seed-averaged MA(5) curves, UNet; highlight the Fig. 1 pair (stride conv vs min)
cell = "unet_deepcrack537_90"
ax = fig.add_subplot(gs[0])
style = {"stride_conv": (FOCUS, 1.1), "min": (OPER, 1.1)}
ends = {}
for o in operators(h, cell):
    cs = [ma5_series(h[(cell, o, s)]) for s in SEEDS]
    ep = sorted(cs[0])
    y = [st.mean(c[e] for c in cs) for e in ep]
    col, lw = style.get(o, (C["light"], 0.6))
    ax.plot(ep, y, color=col, lw=lw,
            ls=(0, (4, 2)) if o == "min" else "-",
            zorder=3 if o in style else 2)
    ends[o] = y[-1]
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], color=FOCUS, lw=1.1, label="Strided conv."),
                   Line2D([], [], color=OPER, lw=1.1, ls=(0, (4, 2)), label="min"),
                   Line2D([], [], color=C["light"], lw=0.8, label="Other 9 operators")],
          loc="lower right", fontsize=6.0, handlelength=1.4)
for b in (30, 90):
    ax.axvline(b, color=C["mid"], lw=0.5, ls=(0, (3, 2)), zorder=1)
ax.set_xlim(5, 90)
ax.set_xticks([5, 30, 60, 90])
ax.set_ylim(0.58, 0.69)
ax.set_xlabel("Training epoch")
ax.set_ylabel("Validation foreground IoU, MA(5)")
panel(ax, "a", "UNet learning curves")

# (b, c) rank of every operator at every budget
budgets = list(range(10, 91, 5))
ramp = LinearSegmentedColormap.from_list("rank", ["#163F80", PAL["blue"], PAL["blue_mid"], PAL["blue_soft"], "#F3F7FC"])
cmap = ListedColormap(ramp(np.linspace(0, 1, 11)))
for k, (cell, title) in enumerate([("unet_deepcrack537_90", "UNet"),
                                   ("rn18_deepcrack537_90", "ResNet18-UNet")]):
    ax = fig.add_subplot(gs[k + 1])
    ops = operators(h, cell)
    R = {o: [] for o in ops}
    for B in budgets:
        m = arm_means(h, cell, B)
        order = sorted(ops, key=lambda o: -m[o])
        for o in ops:
            R[o].append(order.index(o) + 1)
    rows = sorted(ops, key=lambda o: R[o][-1])            # ordered by rank at 90 epochs
    M = np.array([R[o] for o in rows])
    im = ax.imshow(M, cmap=cmap, vmin=0.5, vmax=11.5, aspect="auto",
                   extent=(budgets[0] - 2.5, budgets[-1] + 2.5, len(rows) - 0.5, -0.5))
    for i, o in enumerate(rows):
        for j, B in enumerate(budgets):
            ax.text(B, i, M[i, j], ha="center", va="center", fontsize=4.3,
                    color="white" if M[i, j] <= 3 else C["ink"])
    bi = rows.index("min")
    ax.add_patch(plt.Rectangle((budgets[0] - 2.5, bi - 0.5), budgets[-1] - budgets[0] + 5, 1,
                               fill=False, ec=C["ink"], lw=1.0))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([OP_LABEL[o] for o in rows], fontsize=6.0)
    ax.get_yticklabels()[bi].set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([10, 30, 50, 70, 90])
    ax.set_xlabel("Budget B (epochs)")
    for sp in ax.spines.values():
        sp.set_visible(False)
    panel(ax, "bc"[k], f"{title} ranks")

cax = fig.add_axes([0.915, 0.25, 0.008, 0.5])
cb = fig.colorbar(im, cax=cax, ticks=[1, 6, 11])
cb.ax.tick_params(labelsize=6, length=1.5)
cb.outline.set_visible(False)
cb.set_label("Rank (1 = best)", fontsize=6.2)

save(fig, "fig4_ranks")
