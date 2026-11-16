"""S22 analysis: budget sensitivity on a ConvNeXt-style UNet (DeepCrack537).

Reads data/runs_s22/b90_cnx_gr_deepcrack537_{op}_s{seed}.json and, as references, the
constant-LR runs data/runs/b90_{unet,rn18}_gr_deepcrack537_*.json. The 30-epoch readings
are truncations of the 90-epoch runs (paper section 2.4). Prints every endpoint of
records/S22 section 5 except C0, which needs the activation probe on checkpoints.

Run:  python tools_v3/analyze_s22.py [--s22-dir DIR] [--dry-run]
      --dry-run treats the ResNet18-UNet runs as if they were S22 runs, to check the
      script before any S22 result exists.
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
P1_LIMIT = 0.015
LO, HI = 30, 90


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def curve(run, budget):
    iou = [e["validation"]["foreground_iou"] for e in run["history"]]
    assert len(iou) >= budget, f"run has {len(iou)} epochs, need {budget}"
    return np.array(iou[:budget])


def ma5_max(c):
    ma = np.convolve(c, np.ones(5) / 5, mode="valid")      # ma[i] ends at epoch i + 5
    return ma.max(), int(ma.argmax()) + 5


def runs(folder, tag):
    out = {}
    for o in OPS:
        for s in SEEDS:
            p = folder / f"b90_{tag}_gr_deepcrack537_{o}_s{s}.json"
            if not p.exists():
                raise SystemExit(f"missing run: {p}")
            r = load(p)
            assert r["status"] == "COMPLETE" and not r["test_or_external_test_used"], p
            out[(o, s)] = r
    return out


def readings(rs, budget):
    """{op: [per-seed MA(5) max over the first `budget` epochs]}."""
    return {o: [ma5_max(curve(rs[(o, s)], budget))[0] for s in SEEDS] for o in OPS}


def means(r):
    return {o: float(np.mean(v)) for o, v in r.items()}


def deltas(m):
    return np.array([m[a] - m[b] for a, b in combinations(OPS, 2)])


def f_cross_from(lo, hi, t=THRESH):
    return float(np.mean(np.abs(deltas(means(hi)) - deltas(means(lo))) > t))


def report(rs, label, rng):
    lo, hi = readings(rs, LO), readings(rs, HI)
    m_lo, m_hi = means(lo), means(hi)
    move = deltas(m_hi) - deltas(m_lo)
    lift = np.array([m_hi[o] - m_lo[o] for o in OPS])
    per_seed = [f_cross_from({o: [v[k]] for o, v in lo.items()},
                             {o: [v[k]] for o, v in hi.items()}) for k in range(len(SEEDS))]
    boot = []
    for _ in range(10000):
        pick = rng.integers(0, len(OPS), len(OPS))
        boot.append(np.mean([abs(lift[pick[i]] - lift[pick[j]]) > THRESH
                             for i, j in combinations(range(len(OPS)), 2) if pick[i] != pick[j]]))
    edge = np.mean([ma5_max(curve(rs[(o, s)], LO))[1] >= LO - 1 for o in OPS for s in SEEDS])

    fc = float(np.mean(np.abs(move) > THRESH))
    print(f"\n== {label}")
    print(f"F_cross = {fc:.3f} ({int(np.sum(np.abs(move) > THRESH))}/55)")
    print(f"F_flip  = {np.mean(np.sign(deltas(m_hi)) != np.sign(deltas(m_lo))):.3f}")
    print(f"M_max   = {np.abs(move).max():.4f}   lift mean {lift.mean():.4f}  range {np.ptp(lift):.4f}")
    print("F_cross per seed = " + " / ".join(f"{x:.3f}" for x in per_seed))
    print("F_cross at 0.0025 / 0.005 / 0.01 = " + " / ".join(
        f"{np.mean(np.abs(move) > t):.3f}" for t in (0.0025, 0.005, 0.01)))
    lo95, hi95 = np.percentile(boot, [2.5, 97.5])
    print(f"operator bootstrap 95% = {lo95:.3f}-{hi95:.3f}")
    print(f"30-epoch peak in last two epochs (fails g = 2 check): {edge:.2f}")

    gap = m_lo["max"] - m_lo["min"]
    order = sorted(OPS, key=lambda o: m_hi[o] - m_lo[o], reverse=True)
    print(f"P1  max - min at {LO} epochs = {gap:.4f}  (limit {P1_LIMIT}) -> "
          f"{'holds' if gap <= P1_LIMIT else 'fails'}")
    print(f"P2  rank of min's lift = {order.index('min') + 1} of 11 -> "
          f"{'holds' if order[0] != 'min' else 'fails'}")
    return {"fc": fc, "m_hi": m_hi, "rs": rs}


def time_ratio(rs):
    sec = {o: np.mean([rs[(o, s)]["elapsed_seconds"] for s in SEEDS]) for o in OPS}
    return {o: sec[o] / sec["max"] for o in OPS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s22-dir", default=str(PKG / "data" / "runs_s22"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ref_dir = PKG / "data" / "runs"
    unet, rn18 = runs(ref_dir, "unet"), runs(ref_dir, "rn18")
    cnx = rn18 if args.dry_run else runs(Path(args.s22_dir), "cnx")
    if not args.dry_run:
        params = {r["parameter_count"] for r in cnx.values()}
        print(f"convnext_unet parameter count: {sorted(params)}")

    rng = np.random.default_rng(424242)
    report(unet, "reference: DeepCrack537 / UNet (should give 0.455, 25/55)", rng)
    report(rn18, "reference: DeepCrack537 / ResNet18-UNet", rng)
    res = report(cnx, ("DRY RUN (ResNet18-UNet in place of S22)" if args.dry_run
                       else "S22: DeepCrack537 / convnext_unet (PRIMARY)"), rng)

    m = res["m_hi"]
    for name, ref in (("UNet", unet), ("ResNet18-UNet", rn18)):
        mr = means(readings(ref, HI))
        tau = kendalltau([m[o] for o in OPS], [mr[o] for o in OPS]).statistic
        print(f"Kendall tau_b of 90-epoch rankings vs {name}: {tau:.3f}")
    print("per-run time relative to max: " + ", ".join(
        f"{o} {v:.2f}" for o, v in time_ratio(cnx).items()))
    print("\nC0 (zero share at the four pooling inputs) is not computed here: run the "
          "activation probe on the 90-epoch checkpoints and record it in records/S22 section 8.")


if __name__ == "__main__":
    main()
