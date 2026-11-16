"""Read-only test of how much the learned BTSP fusion weight actually matters.

The branch probe shows the bottom-2 mean is exactly zero at most window
positions, so BTSP is ``alpha * top2_mean`` over most of the feature map.  A
constant gain in front of a convolution followed by GroupNorm is largely
cancelled, which would make the learned scalar close to inert.

This replaces the trained scalar at every downsampling site with a grid of
values and reports the validation foreground IoU for each.  A flat curve means
the parameter the whole V1 design rests on does not change what the network
computes; a peaked curve means it does.

Nothing is written and only the validation split is read.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import sys

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
from surface_crack.metrics import GroupedBinarySegmentationMetrics  # noqa: E402
from surface_crack.models import UNet  # noqa: E402
from surface_crack.resnet_unet import PoolAdapterResNet18UNet  # noqa: E402
from surface_crack.research_pooling import install_research_pooling  # noqa: E402
from surface_crack.spatial import (  # noqa: E402
    crop_prediction_to_original,
    pad_image_and_mask,
)


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def _evaluate(model, loader, device, amp_dtype) -> float:
    model.eval()
    metrics = GroupedBinarySegmentationMetrics(prediction_space="logits")
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=True):
                padded, _, metadata = pad_image_and_mask(image, mask, multiple=16)
                logits = crop_prediction_to_original(model(padded), metadata)
            metrics.update(logits.float(), mask, [batch["group_id"][0]])
    return float(metrics.compute()["foreground_iou"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default=BUNDLE_ROOT / "runs")
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99],
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    install_research_pooling()
    runs_dir = args.runs_dir.resolve()
    payload = json.loads(
        (runs_dir / args.run_name / "result.json").read_text(encoding="utf-8")
    )
    pooling = str(payload["pooling"])
    norm = str(payload.get("norm") or "none")

    data_config = load_baseline_data_config(args.config.resolve(strict=True))
    records = [r for r in load_train_val_records(data_config) if r.split == "val"]
    loader = DataLoader(
        SurfaceCrackDataset(records, mask_spec=data_config.mask_spec),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(args.device)
    backbone = str(payload.get("backbone") or "unet")
    if backbone == "unet":
        model = UNet(
            in_channels=3,
            base_channels=int(payload["base_channels"]),
            pooling=pooling,
            norm=norm,
        )
    elif backbone == "pool_adapter_resnet18_unet":
        model = PoolAdapterResNet18UNet(
            in_channels=3,
            base_channels=int(payload["base_channels"]),
            pooling=pooling,
            norm=norm,
        )
    else:
        raise SystemExit(f"unsupported backbone: {backbone!r}")
    model = model.to(device)
    state = torch.load(
        runs_dir / args.run_name / args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(state["model_state_dict"])

    sites = [layer for layer in model.downsamplers if hasattr(layer, "theta")]
    if not sites:
        raise SystemExit(f"{pooling} registers no fusion scalar to sweep")
    trained = [float(layer.theta.detach().cpu()) for layer in sites]
    amp_dtype = torch.bfloat16 if payload.get("amp_dtype") == "bfloat16" else torch.float16

    baseline = _evaluate(model, loader, device, amp_dtype)
    trained_alphas = [1.0 / (1.0 + math.exp(-t)) for t in trained]
    print(f"run       {args.run_name}")
    print(f"pooling   {pooling}   norm {norm}   sites {len(sites)}")
    print(f"trained   alpha " + " ".join(f"{a:.3f}" for a in trained_alphas))
    print(f"reported  {payload.get('best_foreground_iou_ma5')}")
    print()
    print(f"{'forced alpha':>14}{'val fg IoU':>13}{'delta vs trained':>19}")
    print("-" * 46)
    print(f"{'trained':>14}{baseline:>13.4f}{0.0:>+19.4f}")

    for alpha in args.alphas:
        with torch.no_grad():
            for layer in sites:
                layer.theta.fill_(_logit(alpha))
        score = _evaluate(model, loader, device, amp_dtype)
        print(f"{alpha:>14.2f}{score:>13.4f}{score - baseline:>+19.4f}")

    with torch.no_grad():
        for layer, value in zip(sites, trained):
            layer.theta.fill_(value)
    print()
    print("A flat column means the fusion weight does not change what the")
    print("network computes, so the design argument it carries is inert.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
