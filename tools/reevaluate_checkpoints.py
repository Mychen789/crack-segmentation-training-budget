"""Recompute every metric for finished runs from their saved checkpoint.

Why this exists.  The secondary metrics recorded during training were measured
with an absolute 2 px boundary tolerance on a grid resampled to short side 512.
Neither number appears in the pre-registered protocol or in any decision log,
and an absolute tolerance is not comparable across datasets whose annotations
differ in stroke width.  The protocol fixes the matching distance at a fraction
of the image diagonal instead.

A second reason.  During training the checkpoint is selected on the trailing
five-epoch mean of the primary metric, and every other metric is then read off
the same epoch, so the secondary metrics inherit the primary metric's selection
without carrying their own optimism estimate.  Recomputing all metrics from one
checkpoint makes them mutually consistent.

The primary metric implementation was audited separately and is unchanged, so
re-running here is a measurement pass, not a retraining pass.  Only the
validation split is read and no checkpoint is modified.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
import time

import torch
from torch.utils.data import DataLoader

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = BUNDLE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from surface_crack.config import (  # noqa: E402
    load_baseline_data_config,
    load_train_val_records,
)
from surface_crack.data import SurfaceCrackDataset  # noqa: E402
from surface_crack.extended_metrics import (  # noqa: E402
    BOUNDARY_DIAGONAL_FRACTION,
    GroupedExtendedSegmentationMetrics,
)
from surface_crack.metrics import GroupedBinarySegmentationMetrics  # noqa: E402
from surface_crack.models import UNet  # noqa: E402
from surface_crack.research_pooling import install_research_pooling  # noqa: E402
from surface_crack.spatial import (  # noqa: E402
    crop_prediction_to_original,
    pad_image_and_mask,
)

DATASET_CONFIGS = {
    "crack500": "configs/crack500_train_val.json",
    "deepcrack537": "configs/deepcrack537_train_val.json",
}


def _dataset_of(payload: dict) -> str:
    identifier = str(payload.get("dataset_id", ""))
    for name in DATASET_CONFIGS:
        if name in identifier:
            return name
    raise ValueError(f"cannot map dataset_id {identifier!r} to a config")


def _loader_for(dataset: str, cache: dict) -> tuple[DataLoader, int]:
    if dataset not in cache:
        config = load_baseline_data_config(
            (BUNDLE_ROOT / DATASET_CONFIGS[dataset]).resolve(strict=True)
        )
        records = [r for r in load_train_val_records(config) if r.split == "val"]
        cache[dataset] = (
            DataLoader(
                SurfaceCrackDataset(records, mask_spec=config.mask_spec),
                batch_size=1,
                shuffle=False,
                num_workers=0,
            ),
            len(records),
        )
    return cache[dataset]


def _evaluate(model, loader, device, amp_dtype, fraction) -> dict[str, float]:
    model.eval()
    core = GroupedBinarySegmentationMetrics(prediction_space="logits")
    extended = GroupedExtendedSegmentationMetrics(
        prediction_space="logits",
        boundary_tolerance_diagonal_fraction=fraction,
    )
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=True):
                padded, _, metadata = pad_image_and_mask(image, mask, multiple=16)
                logits = crop_prediction_to_original(model(padded), metadata)
            logits = logits.float()
            group = [batch["group_id"][0]]
            core.update(logits, mask, group)
            extended.update(logits, mask, group)
    result = {k: float(v) for k, v in core.compute().items()}
    result.update({k: float(v) for k, v in extended.compute().items()})
    return result


def _summarize(path: Path) -> int:
    """Print per-dataset and per-operator tables from a previous run of this tool."""

    import statistics as st

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["runs"]
    if not rows:
        print(f"{path} holds no runs")
        return 1
    datasets = sorted({r["dataset"] for r in rows})

    print(f"source   {path}")
    print(f"boundary tolerance   {payload['boundary_diagonal_fraction']} of the diagonal")
    print(f"checkpoint           {payload['checkpoint']}")
    print()

    metrics = ("foreground_iou", "dice", "boundary_f1", "cldice", "auprc")
    print("Per dataset, mean over every operator and seed")
    header = f"{'dataset':16s}" + "".join(f"{m:>14s}" for m in metrics)
    print(header)
    print("-" * len(header))
    for dataset in datasets:
        subset = [r for r in rows if r["dataset"] == dataset]
        line = f"{dataset:16s}"
        for metric in metrics:
            line += f"{st.mean(r['recomputed'][metric] for r in subset):>14.4f}"
        print(line)

    print()
    print("Per operator, two-dataset equal-weight macro of the recomputed metrics")
    header = f"{'operator':30s}" + "".join(f"{m:>14s}" for m in metrics)
    print(header)
    print("-" * len(header))
    operators = sorted({r["pooling"] for r in rows})
    table = []
    for operator in operators:
        macro = {}
        for metric in metrics:
            per_dataset = [
                st.mean(
                    r["recomputed"][metric]
                    for r in rows
                    if r["pooling"] == operator and r["dataset"] == dataset
                )
                for dataset in datasets
                if any(r["pooling"] == operator and r["dataset"] == dataset for r in rows)
            ]
            macro[metric] = st.mean(per_dataset) if per_dataset else float("nan")
        table.append((macro["foreground_iou"], operator, macro))
    for _, operator, macro in sorted(table, reverse=True):
        print(f"{operator:30s}" + "".join(f"{macro[m]:>14.4f}" for m in metrics))

    print()
    print("Primary metric check: recomputed against the value recorded at training")
    print("(they need not agree; the checkpoint is the trailing-mean window right")
    print("end, not the trailing mean itself)")
    gaps = [
        r["recomputed"]["foreground_iou"] - r["as_trained"]["foreground_iou_ma5"]
        for r in rows
        if r["as_trained"]["foreground_iou_ma5"] is not None
    ]
    if gaps:
        print(
            f"  n={len(gaps)}  mean {st.mean(gaps):+.4f}  "
            f"sd {st.pstdev(gaps):.4f}  "
            f"range [{min(gaps):+.4f}, {max(gaps):+.4f}]"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summarize",
        type=Path,
        help="read a previous output JSON and print tables instead of evaluating",
    )
    parser.add_argument("--runs-dir", type=Path, default=BUNDLE_ROOT / "runs")
    parser.add_argument("--pattern", default="formal_gn_*")
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument(
        "--boundary-diagonal-fraction",
        type=float,
        default=BOUNDARY_DIAGONAL_FRACTION,
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.summarize:
        return _summarize(args.summarize.resolve(strict=True))

    install_research_pooling()
    runs_dir = args.runs_dir.resolve()
    device = torch.device(args.device)
    results = sorted(runs_dir.glob(f"{args.pattern}/result.json"))
    if args.limit:
        results = results[: args.limit]
    if not results:
        print(f"no runs matched {args.pattern} under {runs_dir}")
        return 1

    cache: dict = {}
    rows = []
    started = time.time()
    for index, path in enumerate(results, start=1):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETE":
            continue
        if str(payload.get("backbone")) != "unet":
            print(f"skip {path.parent.name}: only the unet backbone is supported")
            continue
        checkpoint = path.parent / args.checkpoint
        if not checkpoint.is_file():
            print(f"skip {path.parent.name}: {args.checkpoint} is missing")
            continue

        dataset = _dataset_of(payload)
        loader, count = _loader_for(dataset, cache)
        model = UNet(
            in_channels=3,
            base_channels=int(payload["base_channels"]),
            pooling=str(payload["pooling"]),
            norm=str(payload.get("norm") or "none"),
        ).to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        amp_dtype = (
            torch.bfloat16 if payload.get("amp_dtype") == "bfloat16" else torch.float16
        )
        metrics = _evaluate(
            model, loader, device, amp_dtype, args.boundary_diagonal_fraction
        )
        rows.append(
            {
                "run": path.parent.name,
                "dataset": dataset,
                "pooling": str(payload["pooling"]),
                "seed": int(payload["seed"]),
                "checkpoint": args.checkpoint,
                "checkpoint_epoch": int(state.get("epoch", -1)),
                "val_images": count,
                "boundary_diagonal_fraction": args.boundary_diagonal_fraction,
                "recomputed": metrics,
                "as_trained": {
                    "foreground_iou_ma5": payload.get("best_foreground_iou_ma5"),
                    "foreground_iou_raw_max": payload.get("best_foreground_iou"),
                },
            }
        )
        elapsed = time.time() - started
        print(
            f"[{index}/{len(results)}] {path.parent.name}"
            f"  fg_iou={metrics['foreground_iou']:.4f}"
            f"  bF1={metrics['boundary_f1']:.4f}"
            f"  clDice={metrics['cldice']:.4f}"
            f"  ({elapsed / index:.1f}s/run)",
            flush=True,
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {
                    "record_type": "recomputed_validation_metrics",
                    "boundary_diagonal_fraction": args.boundary_diagonal_fraction,
                    "checkpoint": args.checkpoint,
                    "note": (
                        "Recomputed from saved checkpoints after the boundary "
                        "tolerance moved from an unregistered absolute 2 px on a "
                        "512 short-side grid to the pre-registered fraction of the "
                        "image diagonal on the native grid. The checkpoint is the "
                        "trailing-mean window right end, not the trailing mean "
                        "itself, so the primary metric here need not equal "
                        "best_foreground_iou_ma5."
                    ),
                    "runs": rows,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
