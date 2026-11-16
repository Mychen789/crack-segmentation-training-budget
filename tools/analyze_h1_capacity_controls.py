"""Read-only comparison of static BTSP against its three upper-tail controls.

Added 2026-09-21 alongside
``phase1_protocol_audit/decision_log_2026-09-21_h1_control_capacity_match.md``.

The frozen H1 control ``upper_tail_matched`` matches the parameter count of
static BTSP but registers a scalar whose gradient is analytically zero, so it
degenerates to ``avg_top2`` and its fusion weight never leaves its initial
value.  This tool reports the frozen contrast beside the two capacity-matched
arms added by that record, so the difference between "parameter-matched" and
"capacity-matched" is visible in one table.

It modifies nothing, reads only ``runs/*/result.json``, and never touches test
or external-test material.  Metric is the trailing five-epoch mean selection
fixed by ``decision_log_2026-09-15_ma5_checkpoint_selection.md``; the raw
maximum is reported beside it because the two disagree by about the size of the
effect under discussion.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import random
import statistics as st

BUNDLE_ROOT = Path(__file__).resolve().parents[1]

PROPOSED = "btsp"
# Ordered with the primary control first.  Until 2026-09-23 the primary was
# upper_tail_capacity_matched; it was then found to be the identical function to
# t_max_avg and, more seriously, to start from a different function than BTSP
# (0.75 r1 + 0.25 r2 against exact average pooling at alpha 0.5), so its contrast
# confounds the lower tail with a different starting point.
CONTROLS = (
    ("diag_partition_matched", "PRIMARY: matched in count, capacity and initial function"),
    ("upper_tail_matched", "frozen H1 control, dead scalar"),
    ("upper_tail_capacity_matched", "same function as t_max_avg, initial function unmatched"),
    ("t_max_avg", "identical operator to the row above, initialised at 0.7"),
    ("avg_top2", "unmatched upper tail, no parameter"),
)
DATASETS = ("crack500", "deepcrack537")

# Registered in the 2026-09-21 decision log, before the new arms were run.
# Filed 2026-09-21 before the arm was run.  It is kept verbatim and still
# checked, because a registration is not revised after the fact; but see the
# 2026-09-23 correction above for why that arm is no longer the primary control.
PREDICTION = {
    "arm": "upper_tail_capacity_matched",
    "sign": "negative on both core datasets",
    "macro_magnitude": (0.003, 0.010),
}


def _load(runs_dir: Path) -> dict[tuple[str, str, str], dict]:
    table: dict[tuple[str, str, str], dict] = {}
    for result in sorted(runs_dir.glob("formal_gn_*/result.json")):
        try:
            payload = json.loads(result.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("status") != "COMPLETE":
            continue
        if payload.get("epochs_completed") != payload.get("epochs_target"):
            continue
        dataset = next(
            (d for d in DATASETS if d in str(payload.get("dataset_id", ""))), None
        )
        if dataset is None:
            continue
        key = (dataset, str(payload.get("pooling")), str(payload.get("seed")))
        table[key] = payload
    return table


def _metric(payload: dict, raw: bool) -> float:
    return float(
        payload["best_foreground_iou"] if raw else payload["best_foreground_iou_ma5"]
    )


def _bootstrap(diffs: list[float], replicates: int, seed: int) -> tuple[float, float]:
    if len(diffs) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    means = sorted(
        st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(replicates)
    )
    lo = means[int(0.025 * replicates)]
    hi = means[min(int(0.975 * replicates), replicates - 1)]
    return lo, hi


def _contrast(
    table: dict, control: str, seeds: list[str], raw: bool
) -> tuple[dict[str, list[float]], list[float]]:
    per_dataset: dict[str, list[float]] = {}
    pooled: list[float] = []
    for dataset in DATASETS:
        diffs = []
        for seed in seeds:
            a = table.get((dataset, PROPOSED, seed))
            b = table.get((dataset, control, seed))
            if a is None or b is None:
                continue
            diffs.append(_metric(a, raw) - _metric(b, raw))
        if diffs:
            per_dataset[dataset] = diffs
            pooled.extend(diffs)
    return per_dataset, pooled


def _scalars(table: dict, arm: str, seeds: list[str]) -> str:
    for dataset in DATASETS:
        for seed in seeds:
            payload = table.get((dataset, arm, seed))
            if payload and payload.get("learned_pooling_scalars"):
                values = payload["learned_pooling_scalars"]
                spread = max(values) - min(values)
                return (
                    " ".join(f"{v:.3f}" for v in values)
                    + f"   spread {spread:.3f}"
                )
    return "no registered scalar"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=BUNDLE_ROOT / "runs")
    parser.add_argument("--seeds", nargs="+", default=["2026", "2027", "2028"])
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=424242)
    parser.add_argument("--raw-max", action="store_true")
    args = parser.parse_args()

    table = _load(args.runs_dir.resolve())
    if not table:
        print(f"no complete formal runs under {args.runs_dir}")
        return 1

    label = "raw maximum" if args.raw_max else "trailing MA(5)"
    print(f"Selection metric: {label} foreground IoU")
    print(f"Seeds: {', '.join(args.seeds)}")
    print()

    print(f"{'arm':30s} {'trained fusion scalars':42s} role")
    print("-" * 110)
    for arm, role in ((PROPOSED, "proposed method"),) + CONTROLS:
        print(f"{arm:30s} {_scalars(table, arm, args.seeds):42s} {role}")
    print()

    print(f"BTSP minus control, {label}")
    print("-" * 110)
    header = f"{'control':30s}"
    for dataset in DATASETS:
        header += f"{dataset:>16s}"
    header += f"{'macro':>10s}{'bootstrap 95% CI':>26s}{'n':>4s}"
    print(header)
    print("-" * 110)

    for control, _role in CONTROLS:
        per_dataset, pooled = _contrast(table, control, args.seeds, args.raw_max)
        if not pooled:
            print(f"{control:30s}{'not yet run':>16s}")
            continue
        row = f"{control:30s}"
        macro_parts = []
        for dataset in DATASETS:
            diffs = per_dataset.get(dataset)
            if diffs:
                mean = st.mean(diffs)
                macro_parts.append(mean)
                row += f"{mean:>+16.4f}"
            else:
                row += f"{'-':>16s}"
        macro = st.mean(macro_parts) if macro_parts else float("nan")
        lo, hi = _bootstrap(pooled, args.bootstrap_replicates, args.bootstrap_seed)
        row += f"{macro:>+10.4f}"
        row += f"   [{lo:+.4f}, {hi:+.4f}]".rjust(26)
        row += f"{len(pooled):>4d}"
        print(row)

    print()
    print("Per-seed detail")
    print("-" * 110)
    for control, _role in CONTROLS:
        per_dataset, _ = _contrast(table, control, args.seeds, args.raw_max)
        if not per_dataset:
            continue
        for dataset, diffs in per_dataset.items():
            signs = sum(1 for d in diffs if d > 0)
            print(
                f"{control:30s} {dataset:14s} "
                + " ".join(f"{d:+.4f}" for d in diffs)
                + f"   mean {st.mean(diffs):+.4f}"
                + (f"  sd {st.pstdev(diffs):.4f}" if len(diffs) > 1 else "")
                + f"  positive {signs}/{len(diffs)}"
            )

    print()
    arm = PREDICTION["arm"]
    per_dataset, pooled = _contrast(table, arm, args.seeds, args.raw_max)
    print(f"Registered prediction for {arm}, filed 2026-09-21 before the arm was run")
    print(f"  expected: {PREDICTION['sign']}, macro magnitude in "
          f"{PREDICTION['macro_magnitude'][0]:.3f} to "
          f"{PREDICTION['macro_magnitude'][1]:.3f}")
    if not pooled:
        print("  observed: arm not yet run")
        return 0
    macro = st.mean(st.mean(v) for v in per_dataset.values())
    lo, hi = PREDICTION["macro_magnitude"]
    both_negative = all(st.mean(v) < 0 for v in per_dataset.values())
    in_band = lo <= abs(macro) <= hi
    print(f"  observed: macro {macro:+.4f}, "
          f"both datasets negative: {both_negative}, "
          f"magnitude in band: {in_band}")
    if both_negative and in_band:
        print("  verdict: consistent with the registered prediction")
    else:
        print("  verdict: DEPARTS from the registered prediction; report as such, "
              "do not quietly adopt")
    print()
    print("H1 remains recorded as not supported "
          "(decision_log_2026-09-17_h1_not_supported.md). "
          "No arm here may be used to restate it as supported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
