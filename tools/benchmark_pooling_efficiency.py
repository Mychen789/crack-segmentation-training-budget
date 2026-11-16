"""Dataset-free efficiency benchmark for one backbone/pooling combination."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import time
from typing import Any

import torch
from torch import Tensor, nn


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = BUNDLE_ROOT / "src"
import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from surface_crack.models import UNet  # noqa: E402
from surface_crack.resnet_unet import PoolAdapterResNet18UNet  # noqa: E402
from surface_crack.research_pooling import install_research_pooling  # noqa: E402


BACKBONES = ("unet", "pool_adapter_resnet18_unet")
POOLING = (
    "max",
    "avg",
    "min",
    "stride_conv",
    "fixed_max_avg",
    "learned_max_avg",
    "v0_max_mean_min",
    "v0_max_mean_matched",
    "avg_top2",
    "upper_tail_matched",
    "t_max_avg",
    "btsp",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=BACKBONES, required=True)
    parser.add_argument("--pooling", choices=POOLING, required=True)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--batch-size", type=int, choices=(1, 8), default=1)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeat-id", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _validate_positive(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _build_model(backbone: str, *, base_channels: int, pooling: str) -> nn.Module:
    install_research_pooling()
    if backbone == "unet":
        model_class = UNet
    elif backbone == "pool_adapter_resnet18_unet":
        model_class = PoolAdapterResNet18UNet
    else:
        raise ValueError(f"unknown backbone: {backbone!r}")
    return model_class(in_channels=3, base_channels=base_channels, pooling=pooling)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of an empty sequence")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must lie in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _conv_macs(model: nn.Module, sample: Tensor) -> int:
    total = 0

    def hook(module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        nonlocal total
        if not isinstance(output, Tensor):
            raise TypeError("complexity hook expects a tensor output")
        if isinstance(module, nn.Conv2d):
            kernel = module.kernel_size[0] * module.kernel_size[1]
            per_output = module.in_channels // module.groups * kernel
            total += output.numel() * per_output
        elif isinstance(module, nn.ConvTranspose2d):
            input_tensor = inputs[0]
            kernel = module.kernel_size[0] * module.kernel_size[1]
            per_input = module.out_channels // module.groups * kernel
            total += input_tensor.numel() * per_input

    handles = [
        module.register_forward_hook(hook)
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d))
    ]
    try:
        with torch.inference_mode():
            model(sample)
    finally:
        for handle in handles:
            handle.remove()
    return total


def _model_forward_times(
    model: nn.Module,
    sample: Tensor,
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> list[float]:
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            times: list[float] = []
            for _ in range(iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(sample)
                end.record()
                torch.cuda.synchronize(device)
                times.append(float(start.elapsed_time(end)))
            return times

        times = []
        for _ in range(iterations):
            started = time.perf_counter()
            model(sample)
            times.append((time.perf_counter() - started) * 1000.0)
        return times


def _end_to_end_times(
    model: nn.Module,
    cpu_sample: Tensor,
    *,
    device: torch.device,
    iterations: int,
) -> list[float]:
    times: list[float] = []
    with torch.inference_mode():
        for _ in range(iterations):
            started = time.perf_counter()
            device_sample = cpu_sample.to(device)
            probabilities = torch.sigmoid(model(device_sample))
            probabilities.cpu()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append((time.perf_counter() - started) * 1000.0)
    return times


def benchmark(
    *,
    backbone: str,
    pooling: str,
    base_channels: int,
    batch_size: int,
    input_size: int,
    warmup: int,
    iterations: int,
    repeat_id: int,
    device_name: str,
) -> dict[str, Any]:
    for value, name in (
        (base_channels, "base_channels"),
        (batch_size, "batch_size"),
        (input_size, "input_size"),
        (warmup, "warmup"),
        (iterations, "iterations"),
        (repeat_id, "repeat_id"),
    ):
        _validate_positive(value, name)
    if input_size % 16:
        raise ValueError("input_size must be divisible by 16")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")

    device = torch.device(device_name)
    model = _build_model(
        backbone, base_channels=base_channels, pooling=pooling
    ).to(device=device, dtype=torch.float32).eval()
    cpu_sample = torch.randn(batch_size, 3, input_size, input_size, dtype=torch.float32)
    sample = cpu_sample.to(device)
    convolution_macs = _conv_macs(model, sample)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model_times = _model_forward_times(
        model,
        sample,
        device=device,
        warmup=warmup,
        iterations=iterations,
    )
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    end_to_end_times = _end_to_end_times(
        model,
        cpu_sample,
        device=device,
        iterations=iterations,
    )

    protocol_shape = (
        device.type == "cuda"
        and input_size == 512
        and batch_size in (1, 8)
        and warmup >= 100
        and iterations >= 1000
        and repeat_id in (1, 2, 3)
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    state_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in model.state_dict().values()
    )
    return {
        "schema_version": "1.0",
        "benchmark_type": "DATASET_FREE_MODEL_EFFICIENCY",
        "status": "PROTOCOL_SHAPE_RUN" if protocol_shape else "ENGINEERING_SMOKE",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "backbone": backbone,
        "pooling": pooling,
        "base_channels": base_channels,
        "batch_size": batch_size,
        "input_shape": [batch_size, 3, input_size, input_size],
        "dtype": "float32",
        "warmup": warmup,
        "iterations": iterations,
        "repeat_id": repeat_id,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "parameter_count": parameter_count,
        "state_bytes": state_bytes,
        "convolution_macs_per_batch": convolution_macs,
        "model_latency_ms": {
            "mean": statistics.fmean(model_times),
            "p50": _percentile(model_times, 0.50),
            "p95": _percentile(model_times, 0.95)
        },
        "end_to_end_latency_ms": {
            "mean": statistics.fmean(end_to_end_times),
            "p50": _percentile(end_to_end_times, 0.50),
            "p95": _percentile(end_to_end_times, 0.95)
        },
        "peak_allocated_bytes": peak_memory,
        "data_samples_used": 0,
        "validation_predictions_generated": False,
        "test_or_external_test_used": False
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = _arguments()
    result = benchmark(
        backbone=args.backbone,
        pooling=args.pooling,
        base_channels=args.base_channels,
        batch_size=args.batch_size,
        input_size=args.input_size,
        warmup=args.warmup,
        iterations=args.iterations,
        repeat_id=args.repeat_id,
        device_name=args.device,
    )
    _write_json_atomic(args.output.resolve(), result)
    print(
        f"status={result['status']} model_p50_ms={result['model_latency_ms']['p50']:.4f} "
        f"output={args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
