"""Fig. 1: one real operator pair whose conclusion changes with the budget (DeepCrack537 / UNet)."""

import statistics as st

import matplotlib.pyplot as plt

from style3 import (BAND, FOCUS, FULL_W, MM, OP_LABEL, OPER, PAL, SEEDS, THRESH, C,
                    load_histories, ma5_best, ma5_series, panel, save)

CELL = "unet_deepcrack537_90"
A, B_ = "stride_conv", "min"
h = load_histories()


def mean_curve(op):
    cs = [ma5_series(h[(CELL, op, s)]) for s in SEEDS]
    ep = sorted(cs[0])
    return ep, [st.mean(c[e] for c in cs) for e in ep]


def reading(op, budget):
    return st.mean(ma5_best(h[(CELL, op, s)], budget)[0] for s in SEEDS)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_W, 56 * MM),
                               gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.38})

# (a) learning curves and the two readings
col = {A: FOCUS, B_: OPER}
for op in (A, B_):
    ep, y = mean_curve(op)
    ax1.plot(ep, y, color=col[op], lw=1.1,
             ls="-" if op == A else (0, (4, 2)))
    # labels sit beside the curves mid-way, clear of the reading dots at B = 90
    ax1.text(44 if op == A else 56, 0.6635 if op == A else 0.6315,
             OP_LABEL[op] + (" (solid)" if op == A else " (dashed)"),
             color=FOCUS if op == A else PAL["orange_dark"],
             fontsize=6.5, va="center", ha="center")
for budget in (30, 90):
    ra, rb = reading(A, budget), reading(B_, budget)
    ax1.axvline(budget, color=C["mid"], lw=0.5, ls=(0, (3, 2)), zorder=0)
    # the two readings (seed mean of each seed's MA(5) maximum over the first B epochs) as dots on
    # the budget line; they can sit above the mean curve once the seeds peak at different epochs
    ax1.plot([budget, budget], [rb, ra], color=C["ink"], lw=0.8, zorder=3)
    ax1.plot([budget], [ra], "o", ms=3.2, color=FOCUS, zorder=4)
    ax1.plot([budget], [rb], "D", ms=3.2, color=OPER, zorder=4)
    ax1.text(budget - 1.2, ra + 0.0035, f"Δ = {ra - rb:.3f}", ha="right", va="bottom", fontsize=6.3)
    ax1.text(budget + 0.8, 0.583, f"B = {budget}", fontsize=6.2, color=C["dark"])
ax1.set_xlim(5, 100)
ax1.set_xticks([5, 30, 60, 90])
ax1.set_ylim(0.58, 0.675)
ax1.set_xlabel("Training epoch")
ax1.set_ylabel("Validation foreground IoU, MA(5)")
panel(ax1, "a", "Learning curves")

# (b) the paired difference as a function of the budget
budgets = list(range(10, 91))
d = [reading(A, b) - reading(B_, b) for b in budgets]
ax2.axhspan(-THRESH, THRESH, color=BAND, lw=0, zorder=0)
ax2.text(12, -THRESH + 0.0007, "±0.005 meaningful difference", ha="left", va="bottom", fontsize=6.2, color=C["dark"])
ax2.axhline(0, color=C["mid"], lw=0.5)
ax2.plot(budgets, d, color=C["ink"], lw=1.0)
for b in (30, 90):
    v = d[budgets.index(b)]
    ax2.plot([b], [v], "o", ms=3.5, color=C["ink"])
    if b == 30:
        ax2.text(b + 2, v + 0.0025, f"{v:.3f}", fontsize=6.3, ha="left")
    else:
        ax2.text(b - 1, v - 0.0042, f"{v:.3f}", fontsize=6.3, ha="right", va="top")
ax2.set_xlim(10, 92)
ax2.set_xticks([10, 30, 50, 70, 90])
ax2.set_ylim(-0.008, 0.04)
ax2.set_xlabel("Budget B (epochs)")
ax2.set_ylabel(f"Paired difference Δ ({OP_LABEL[A]} − {OP_LABEL[B_]})")
panel(ax2, "b", "Comparison across budgets")

save(fig, "fig1_concept")
