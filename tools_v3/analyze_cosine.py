"""S21 analysis: budget sensitivity of operator comparisons under a cosine schedule.

Reads data/runs_cosine/cos{30,90}_unet_gr_deepcrack537_{op}_s{seed}.json and, for the
cross-schedule endpoint, the constant-LR runs data/runs/b90_unet_gr_deepcrack537_*.json.
Prints every endpoint of records/S21 section 5 and the preregistered verdict of section 6.

Run:  python tools_v3/analyze_cosine.py [--cos-dir DIR]
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

PKG = Path(__file__).resolve().parents[1]
OPS = ["max", "avg", "min", "avg_top2", "fixed_max_avg", "learned_max_avg", "t_max_avg",
       "v0_max_mean_min", "btsp", "diag_partition_matched", "stride_conv"]
SEEDS = [2026, 2027, 2028]
THRESH = 0.005


def curve(path, budget=None):
    h = json.loads(Path(path).read_text(encoding="utf-8"))["history"]
    iou = [e["validation"]["foreground_iou"] for e in h]
    return np.array(iou[:budget] if budget else iou)


def ma5_max(c):
    ma = np.convolve(c, np.ones(5) / 5, mode="valid")      # ma[i] ends at epoch i + 5
    return ma.max(), int(ma.argmax()) + 5


def readings(paths, reader):
    """{op: [per-seed reading]} for one budget."""
    return {o: [reader(paths[(o, s)]) for s in SEEDS] for o in OPS}


def deltas(r):
    m = {o: np.mean(v) for o, v in r.items()}
    return np.array([m[a] - m[b] for a, b in combinations(OPS, 2)]), m


def f_cross(lo, hi, t=THRESH):
    return float(np.mean(np.abs(deltas(hi)[0] - deltas(lo)[0]) > t))


def endpoints(lo, hi, label, rng):
    d_lo, m_lo = deltas(lo)
    d_hi, m_hi = deltas(hi)
    move = d_hi - d_lo
    lift = np.array([m_hi[o] - m_lo[o] for o in OPS])
    per_seed = []
    for k in range(len(SEEDS)):
        a = {o: [v[k]] for o, v in lo.items()}
        b = {o: [v[k]] for o, v in hi.items()}
        per_seed.append(f_cross(a, b))
    boot = []
    for _ in range(10000):
        pick = rng.integers(0, len(OPS), len(OPS))
        boot.append(np.mean([abs(lift[pick[i]] - lift[pick[j]]) > THRESH
                             for i, j in combinations(range(len(OPS)), 2) if pick[i] != pick[j]]))
    print(f"\n== {label}")
    print(f"F_cross = {np.mean(np.abs(move) > THRESH):.3f} ({int(np.sum(np.abs(move) > THRESH))}/55)")
    print(f"F_flip  = {np.mean(np.sign(d_hi) != np.sign(d_lo)):.3f}")
    print(f"M_max   = {np.abs(move).max():.4f}   lift mean {lift.mean():.4f}  range {np.ptp(lift):.4f}")
    print("F_cross per seed = " + " / ".join(f"{x:.3f}" for x in per_seed))
    print("F_cross at 0.0025 / 0.005 / 0.01 = " + " / ".join(
        f"{np.mean(np.abs(move) > t):.3f}" for t in (0.0025, 0.005, 0.01)))
    lo95, hi95 = np.percentile(boot, [2.5, 97.5])
    print(f"operator bootstrap 95% = {lo95:.3f}-{hi95:.3f}")
    return float(np.mean(np.abs(move) > THRESH))


def edge_share(paths, budget):
    hits = [ma5_max(curve(paths[(o, s)], budget))[1] >= budget - 1 for o in OPS for s in SEEDS]
    return np.mean(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cos-dir", default=str(PKG / "data" / "runs_cosine"))
    args = ap.parse_args()
    cos_dir = Path(args.cos_dir)
    cos = {b: {(o, s): cos_dir / f"cos{b}_unet_gr_deepcrack537_{o}_s{s}.json"
               for o in OPS for s in SEEDS} for b in (30, 90)}
    missing = [str(p) for b in cos for p in cos[b].values() if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} cosine runs missing, e.g. {missing[0]}")
    const = {(o, s): PKG / "data" / "runs" / f"b90_unet_gr_deepcrack537_{o}_s{s}.json"
             for o in OPS for s in SEEDS}

    rng = np.random.default_rng(424242)
    main_reader = lambda p: ma5_max(curve(p))[0]
    last_reader = lambda p: curve(p)[-1]
    c30, c90 = readings(cos[30], main_reader), readings(cos[90], main_reader)
    primary = endpoints(c30, c90, "cosine 30 -> 90, MA(5) max (PRIMARY)", rng)
    endpoints(readings(cos[30], last_reader), readings(cos[90], last_reader),
              "cosine 30 -> 90, last-epoch value", rng)

    k30 = readings(const, lambda p: ma5_max(curve(p, 30))[0])
    k90 = readings(const, main_reader)
    endpoints(k30, k90, "constant 30 -> 90, MA(5) max (reference, should give 0.455)", rng)

    d_c90, m_c90 = deltas(c90)
    d_k90, m_k90 = deltas(k90)
    tau = kendalltau([m_c90[o] for o in OPS], [m_k90[o] for o in OPS]).statistic
    print("\n== cosine 90 vs constant 90")
    print(f"pairs whose difference disagrees by > 0.005: {np.mean(np.abs(d_c90 - d_k90) > THRESH):.3f}")
    print(f"Kendall tau_b of rankings: {tau:.3f}")
    print(f"peak-at-edge share: cos30 {edge_share(cos[30], 30):.2f}, cos90 {edge_share(cos[90], 90):.2f}")

    verdict = ("persists" if primary >= 0.25 else "largely removed" if primary <= 0.10
               else "reduced but present")
    print(f"\nPREREGISTERED VERDICT (S21 section 6): F_cross(cos) = {primary:.3f} -> {verdict}")


if __name__ == "__main__":
    main()
