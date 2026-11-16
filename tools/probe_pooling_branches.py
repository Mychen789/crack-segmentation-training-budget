"""Read-only mechanism probe for the BTSP branches (Phase B).

Question this answers: under GroupNorm followed by ReLU, roughly half of the
activations entering a downsampling point are exactly zero, so the bottom-2 mean
of a 2x2 window may be exactly zero over most of the feature map.  If it is,
static BTSP reduces to ``alpha * top2_mean`` at those positions, i.e. a scaled
``avg_top2``, and the scale can be absorbed by the next convolution.  That would
refute the V1 design argument mechanistically rather than statistically.

It loads a finished checkpoint, replays the validation split, and reports per
downsampling point:

* the fraction of window positions whose bottom-2 mean is exactly zero,
* the fraction whose whole 2x2 window is zero,
* the magnitude ratio between the lower and upper branches,
* the fraction of windows holding a tie, which is what makes ``topk`` gradient
  routing non-deterministic under bfloat16.

Nothing is written and no split other than validation is touched.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import torch
from torch import Tensor, nn
from torch.nn import functional as F
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
from surface_crack.models import UNet  # noqa: E402
from surface_crack.resnet_unet import PoolAdapterResNet18UNet  # noqa: E402
from surface_crack.research_pooling import install_research_pooling  # noqa: E402
from surface_crack.spatial import (  # noqa: E402
    crop_prediction_to_original,
    pad_image_and_mask,
)


class BranchStatistics:
    """Running counts for one downsampling point."""

    def __init__(self) -> None:
        self.windows = 0
        self.lower_zero = 0
        self.window_all_zero = 0
        self.upper_zero = 0
        self.tied = 0
        self.lower_sum = 0.0
        self.upper_sum = 0.0
        self.input_zero = 0
        self.input_total = 0

    def update(self, x: Tensor) -> None:
        # fp32 so that the zero test is about the activations, not about bf16.
        x = x.detach().float()
        batch, channels, height, width = x.shape
        height -= height % 2
        width -= width % 2
        x = x[..., :height, :width]
        windows = (
            x.reshape(batch, channels, height // 2, 2, width // 2, 2)
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(batch, channels, -1, 4)
        )
        ordered, _ = torch.sort(windows, dim=-1, descending=True)
        upper = ordered[..., :2].mean(dim=-1)
        lower = ordered[..., 2:].mean(dim=-1)

        self.windows += int(upper.numel())
        self.lower_zero += int((lower == 0).sum())
        self.upper_zero += int((upper == 0).sum())
        self.window_all_zero += int((ordered[..., 0] == 0).sum())
        self.lower_sum += float(lower.abs().sum())
        self.upper_sum += float(upper.abs().sum())
        # A tie anywhere in the window makes topk's choice of index arbitrary.
        adjacent_equal = ordered[..., :-1] == ordered[..., 1:]
        self.tied += int(adjacent_equal.any(dim=-1).sum())
        self.input_zero += int((x == 0).sum())
        self.input_total += int(x.numel())

    def report(self) -> dict[str, float]:
        if not self.windows:
            return {}
        return {
            "lower_branch_exactly_zero": self.lower_zero / self.windows,
            "upper_branch_exactly_zero": self.upper_zero / self.windows,
            "whole_window_zero": self.window_all_zero / self.windows,
            "window_holds_a_tie": self.tied / self.windows,
            "input_activation_zero": self.input_zero / max(self.input_total, 1),
            "lower_over_upper_magnitude": (
                self.lower_sum / self.upper_sum if self.upper_sum else float("nan")
            ),
            "windows": float(self.windows),
        }


def _build(backbone: str, base_channels: int, pooling: str, norm: str):
    """Both frozen backbone families, at whatever normalisation the run used."""

    if backbone == "unet":
        return UNet(
            in_channels=3, base_channels=base_channels, pooling=pooling, norm=norm
        )
    if backbone == "pool_adapter_resnet18_unet":
        return PoolAdapterResNet18UNet(
            in_channels=3, base_channels=base_channels, pooling=pooling, norm=norm
        )
    raise SystemExit(f"unsupported backbone: {backbone!r}")


def _load_run(runs_dir: Path, run_name: str) -> dict:
    result = runs_dir / run_name / "result.json"
    return json.loads(result.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default=BUNDLE_ROOT / "runs")
    parser.add_argument(
        "--checkpoint",
        default="best.pt",
        help="checkpoint file inside the run directory",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="validation images to replay, 0 = all"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    install_research_pooling()
    runs_dir = args.runs_dir.resolve()
    payload = _load_run(runs_dir, args.run_name)
    pooling = str(payload["pooling"])
    norm = str(payload.get("norm") or "none")
    base_channels = int(payload["base_channels"])

    backbone = str(payload.get("backbone") or "unet")

    data_config = load_baseline_data_config(args.config.resolve(strict=True))
    records = [r for r in load_train_val_records(data_config) if r.split == "val"]
    if args.limit:
        records = records[: args.limit]
    loader = DataLoader(
        SurfaceCrackDataset(records, mask_spec=data_config.mask_spec),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(args.device)
    model = _build(backbone, base_channels, pooling, norm).to(device)
    checkpoint_path = runs_dir / args.run_name / args.checkpoint
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    statistics = [BranchStatistics() for _ in model.downsamplers]
    handles = []
    for index, layer in enumerate(model.downsamplers):
        def hook(_module, inputs, _index=index):
            statistics[_index].update(inputs[0])

        handles.append(layer.register_forward_pre_hook(hook))

    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            padded, _, metadata = pad_image_and_mask(image, mask, multiple=16)
            crop_prediction_to_original(model(padded), metadata)
    for handle in handles:
        handle.remove()

    scalars = [
        float(layer.alpha.detach().cpu())
        if hasattr(layer, "alpha")
        else float(layer.t.detach().cpu())
        if hasattr(layer, "t")
        else float("nan")
        for layer in model.downsamplers
    ]

    print(f"run        {args.run_name}")
    print(f"pooling    {pooling}   norm {norm}   checkpoint {args.checkpoint}")
    print(f"validation {len(records)} images on {device}")
    print()
    header = (
        f"{'site':>5}{'alpha':>8}{'lower=0':>10}{'window=0':>10}"
        f"{'upper=0':>9}{'tie':>8}{'input=0':>9}{'|L|/|U|':>10}"
    )
    print(header)
    print("-" * len(header))
    rows = []
    for index, stat in enumerate(statistics):
        report = stat.report()
        if not report:
            continue
        rows.append(report)
        print(
            f"{index + 1:>5}"
            f"{scalars[index]:>8.3f}"
            f"{report['lower_branch_exactly_zero']:>10.1%}"
            f"{report['whole_window_zero']:>10.1%}"
            f"{report['upper_branch_exactly_zero']:>9.1%}"
            f"{report['window_holds_a_tie']:>8.1%}"
            f"{report['input_activation_zero']:>9.1%}"
            f"{report['lower_over_upper_magnitude']:>10.4f}"
        )
    print()
    print("lower=0    bottom-2 mean is exactly zero, so the lower branch carries")
    print("           nothing at that position and BTSP reduces to alpha*top2_mean")
    print("tie        at least two of the four values are equal, which makes the")
    print("           topk index choice, and therefore gradient routing, arbitrary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
