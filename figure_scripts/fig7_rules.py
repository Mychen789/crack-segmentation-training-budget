"""Fig. 7: every rule-hyperparameter group against the four preregistered criteria (design cell).

A pass/fail matrix: each cell shows the value, its background whether the preset bar was met.
"""

import csv
import math

import matplotlib.pyplot as plt

from style3 import (C, DATA, FAIL_BG, FULL_W, MM, NA_BG, PASS_BG, save)

rows = []
with open(DATA / "tables" / "table4_reporting_rules.csv", encoding="utf-8-sig") as f:
    rows = [r for r in csv.DictReader(f) if r["cell"] == "design_dc537_unet"]


def num(v):
    try:
        x = float(v)
    except ValueError:
        return None
    return None if math.isnan(x) else x


R0_F = 0.45454545454545453
R0_S = 1.7115924288079656

groups = [
    ("R0  Current rule (baseline)", lambda r: r["rule"] == "R0" and r["round"] == "1"),
    ("R1  Convergence gate, g ∈ {3, 5, 8, 12}", lambda r: r["rule"] == "R1"),
    ("R2  Individual convergence, 6 settings (w, δ)", lambda r: r["rule"] == "R2"),
    ("R3  Tail mean, k = 5", lambda r: r["rule"] == "R3" and '"tail": 5}' in r["params"]),
    ("R3  Tail mean, k = 10", lambda r: r["rule"] == "R3" and '"tail": 10}' in r["params"]),
    ("R4  Gate + paired signs, 4 settings", lambda r: r["rule"] == "R4"),
    ("R5  Exponential fit, all, 4 settings", lambda r: r["rule"] == "R5" and '"exp"' in r["params"] and '"all"' in r["params"]),
    ("R5  Exponential fit, half, 4 settings", lambda r: r["rule"] == "R5" and '"exp"' in r["params"] and '"half"' in r["params"]),
    ("R5  Power-law fit, all, κ = 0.05", lambda r: r["rule"] == "R5" and '"pow"' in r["params"] and '"all"' in r["params"] and '"kappa": 0.05' in r["params"]),
    ("R5  Power-law fit, all, κ = 0.10", lambda r: r["rule"] == "R5" and '"pow"' in r["params"] and '"all"' in r["params"] and '"kappa": 0.1}' in r["params"]),
    ("R5  Power-law fit, half, 4 settings", lambda r: r["rule"] == "R5" and '"pow"' in r["params"] and '"half"' in r["params"]),
    ("R6  Two-point extrapolation, r = 0.5, κ = 0.05", lambda r: r["rule"] == "R6" and '"ratio": 0.5, "kappa": 0.05' in r["params"]),
    ("R6  Two-point extrapolation, r = 0.5, κ = 0.10", lambda r: r["rule"] == "R6" and '"ratio": 0.5, "kappa": 0.1}' in r["params"]),
    ("R6  Two-point extrapolation, r = 0.7, κ = 0.05", lambda r: r["rule"] == "R6" and '"ratio": 0.7, "kappa": 0.05' in r["params"]),
    ("R6  Two-point extrapolation, r = 0.7, κ = 0.10", lambda r: r["rule"] == "R6" and '"ratio": 0.7, "kappa": 0.1}' in r["params"]),
]

table = []
for label, pick in groups:
    r = [r for r in rows if pick(r)][0]
    f = num(r["F_cross"])
    table.append((label, r["round"], {
        "C": num(r["C_adm"]),
        "F": None if f is None else f / R0_F,
        "A": num(r["A_fid"]),
        "S": num(r["S_disc"]) if r["round"] == "2" else None,
    }))
table[0][2]["S"] = R0_S

# key, header, bar, sense
crit = [("C", "Coverage\n≥ 0.5", 0.5, "ge"),
        ("F", "Instability / R0\n≤ 0.5", 0.5, "le"),
        ("A", "Fidelity\n≥ 0.9", 0.9, "ge"),
        ("S", "Discriminability\n≥ 0.5", 0.5, "ge")]

n = len(table)
fig, ax = plt.subplots(figsize=(FULL_W * 1.05, 92 * MM))
cw, x0 = 1.0, 0.0
ncol = len(crit) + 1
for i, (label, rnd, v) in enumerate(table):
    y = i
    ok_all = True
    for j, (key, _, bar, sense) in enumerate(crit):
        x = v[key]
        if i == 0:
            bg, txt = NA_BG, f"{x:.2f}"
        elif key == "S" and rnd == "1":
            bg, txt, ok_all = NA_BG, "Not set", False
        elif x is None:
            bg, txt, ok_all = NA_BG, "—", False
        else:
            ok = x >= bar if sense == "ge" else x <= bar
            ok_all &= ok
            bg, txt = (PASS_BG if ok else FAIL_BG), f"{x:.2f}"
        ax.add_patch(plt.Rectangle((j, y - 0.5), cw, 1, fc=bg, ec="white", lw=1.2))
        ax.text(j + 0.5, y, txt, ha="center", va="center", fontsize=6.3,
                color=C["mid"] if bg == NA_BG and i else C["ink"])
        if bg in (PASS_BG, FAIL_BG):    # redundant mark so the cell reads without colour
            ax.text(j + 0.86, y, "√" if bg == PASS_BG else "×", ha="center", va="center",
                    fontsize=5.6, color=C["dark"])
    j = len(crit)
    if i == 0:
        bg, txt = NA_BG, "Baseline"
    else:
        bg, txt = (PASS_BG, "Yes") if ok_all else (FAIL_BG, "No")
    ax.add_patch(plt.Rectangle((j + 0.15, y - 0.5), cw, 1, fc=bg, ec="white", lw=1.2))
    ax.text(j + 0.65, y, txt, ha="center", va="center", fontsize=6.3, fontweight="bold")
    ax.text(-0.15, y, label, ha="right", va="center", fontsize=6.3)

for j, (_, head, _, _) in enumerate(crit):
    ax.text(j + 0.5, -0.75, head, ha="center", va="bottom", fontsize=6.3, linespacing=1.25)
ax.text(len(crit) + 0.65, -0.75, "All four\ncriteria met", ha="center", va="bottom", fontsize=6.3,
        linespacing=1.25, fontweight="bold")
for sep, mid, name in [(0.5, 3.0, "Round 1"), (5.5, (6 + n - 1) / 2, "Round 2")]:
    ax.axhline(sep, xmin=0, xmax=1, color=C["dark"], lw=0.6)
    # the round label sits at the vertical middle of its group of rows
    ax.text(-4.5, mid, name, ha="left", va="center", fontsize=6.0, color=C["mid"])

ax.set_xlim(-4.6, ncol + 0.2)
ax.set_ylim(n - 0.5, -1.6)
ax.axis("off")

save(fig, "fig7_rules")
