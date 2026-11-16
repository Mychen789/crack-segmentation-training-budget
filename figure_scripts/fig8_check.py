"""Fig. 8: the edge check (g = 2) and the realised budget sensitivity across budgets (DeepCrack537)."""

import matplotlib.pyplot as plt

from style3 import (CHANGED, FULL_W, MM, SEEDS, THRESH, C, load_histories, ma5_best,
                    operators, pairs, panel, save)

G = 2
h = load_histories()
fig, axes = plt.subplots(1, 2, figsize=(FULL_W * 0.86, 60 * MM), sharey=True)
fig.subplots_adjust(wspace=0.1)

for k, (ax, (cell, title)) in enumerate(zip(axes, [
        ("unet_deepcrack537_90", "UNet"),
        ("rn18_deepcrack537_90", "ResNet18-UNet")])):
    ops = operators(h, cell)
    prs = pairs(ops)
    ref = {o: sum(ma5_best(h[(cell, o, s)], 90)[0] for s in SEEDS) / 3 for o in ops}
    budgets = list(range(10, 91))
    run_pass, pair_rep, f_cross = [], [], []
    for B in budgets:
        val, ok = {}, {}
        for o in ops:
            for s in SEEDS:
                v, arg = ma5_best(h[(cell, o, s)], B)
                val[(o, s)], ok[(o, s)] = v, arg <= B - G
        m = {o: sum(val[(o, s)] for s in SEEDS) / 3 for o in ops}
        run_pass.append(sum(ok.values()) / len(ok))
        pair_rep.append(sum(all(ok[(a, s)] and ok[(b, s)] for s in SEEDS) for a, b in prs) / len(prs))
        f_cross.append(sum(abs((m[a] - m[b]) - (ref[a] - ref[b])) > THRESH for a, b in prs) / len(prs))

    ax.plot(budgets, run_pass, color=C["light"], lw=0.9, zorder=2)
    ax.plot(budgets, f_cross, color=CHANGED, lw=1.2, zorder=3)
    ax.plot(budgets, pair_rep, color=C["ink"], lw=1.2, zorder=4)
    if k == 0:
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], color=CHANGED, lw=1.2, label="Pairs changed relative to 90 epochs"),
                           Line2D([], [], color=C["ink"], lw=1.2, label="Pairs reportable by the check"),
                           Line2D([], [], color=C["light"], lw=0.9, label="Runs passing the check")],
                  loc="upper left", fontsize=6.0, handlelength=1.4, borderaxespad=0.2)
    else:
        b47 = next(b for b, f in zip(budgets, f_cross) if b >= 40 and max(f_cross[budgets.index(b):]) < 0.04)
        ax.axvline(b47, color=CHANGED, lw=0.5, ls=(0, (2, 2)))
        ax.text(b47 + 1, 0.26, f"B ≥ {b47}:\nstable", color=CHANGED, fontsize=6.0,
                ha="left", linespacing=1.2)
        b70 = next(b for b, p in zip(budgets, pair_rep) if p >= 0.5)
        ax.axvline(b70, color=C["ink"], lw=0.5, ls=(0, (2, 2)))
        ax.text(b70 - 1, 0.99, f"B ≥ {b70}:\nhalf pass", color=C["ink"], fontsize=6.0,
                ha="right", va="top")
    ax.set_xlim(10, 90)
    ax.set_xticks([10, 30, 50, 70, 90])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Budget B (epochs)")
    if k == 0:
        ax.set_ylabel("Proportion of pairs or runs")
    panel(ax, "ab"[k], f"DeepCrack537 / {title}")

save(fig, "fig8_check")
