"""Shared style, data access and export for the manuscript figures."""

from __future__ import annotations

import csv
import itertools
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "draft_en" / "figures"

for _f in ("C:/Windows/Fonts/Noto Sans SC (TrueType).otf",
           "C:/Windows/Fonts/Noto Sans SC Bold (TrueType).otf"):
    fm.fontManager.addfont(_f)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans SC"],
    "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "figure.dpi": 150,
})

MM = 1 / 25.4
FULL_W = 180 * MM

C = {
    "ink": "#272727",
    "dark": "#4D4D4D",
    "mid": "#8A8A8A",
    "light": "#CFCECE",
    "faint": "#EDEDED",
    "signal": "#B64342",
    "signal_soft": "#F6CFCB",
    "blue": "#0F4D92",
    "blue_soft": "#B4C0E4",
    "teal": "#42949E",
    "pass": "#DDF3DE",
}

EQUIV = {"upper_tail_matched", "upper_tail_capacity_matched", "v0_max_mean_matched"}
SEEDS = ("2026", "2027", "2028")
THRESH = 0.005

# (cell id in data, label, low budget, high budget)
CELLS = [
    ("unet_deepcrack537_90", "DeepCrack537 / UNet", 30, 90),
    ("rn18_deepcrack537_90", "DeepCrack537 / ResNet18-UNet", 30, 90),
    ("unet_crack500_30", "CRACK500 / UNet", 15, 30),
    ("rn18_crack500_30", "CRACK500 / ResNet18-UNet", 15, 30),
]

OP_LABEL = {
    "max": "max", "avg": "avg", "min": "min", "avg_top2": "Avg-Top2",
    "fixed_max_avg": "Fixed max+avg", "learned_max_avg": "Learned max+avg",
    "t_max_avg": "T-Max-Avg", "v0_max_mean_min": "max/mean/min",
    "diag_partition_matched": "Diagonal partition", "btsp": "BTSP",
    "stride_conv": "Strided conv.",
}


def load_histories() -> dict:
    """{(cell, pooling, seed): {epoch: fg_iou}} for the 11 distinct operators."""
    h: dict = defaultdict(dict)
    with open(DATA / "per_epoch_validation.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["pooling"] in EQUIV:
                continue
            h[(r["cell"], r["pooling"], r["seed"])][int(r["epoch"])] = float(r["foreground_iou"])
    return h


def ma5_series(seq: dict) -> dict:
    return {e: sum(seq[e - k] for k in range(5)) / 5 for e in sorted(seq) if e >= 5}


def ma5_best(seq: dict, budget: int) -> tuple[float, int]:
    """Trailing-MA(5) maximum within the first `budget` epochs, and its epoch."""
    best, arg = None, None
    for e in range(5, budget + 1):
        v = sum(seq[e - k] for k in range(5)) / 5
        if best is None or v > best:
            best, arg = v, e
    return best, arg


def operators(h: dict, cell: str) -> list[str]:
    return sorted({k[1] for k in h if k[0] == cell})


def arm_means(h: dict, cell: str, budget: int) -> dict:
    return {o: st.mean(ma5_best(h[(cell, o, s)], budget)[0] for s in SEEDS)
            for o in operators(h, cell)}


def pairs(ops):
    return list(itertools.combinations(ops, 2))


def save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def panel_tag(ax, tag: str) -> None:
    ax.text(-0.2, 1.2, tag, transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", ha="left", va="bottom")
