"""Portable train/validation-only pooling experiment with epoch resume."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
MA_WINDOW = 5
SRC_ROOT = BUNDLE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from surface_crack.config import (  # noqa: E402
    load_baseline_data_config,
    load_train_val_records,
)
from surface_crack.data import SurfaceCrackDataset  # noqa: E402
from surface_crack.extended_metrics import GroupedExtendedSegmentationMetrics  # noqa: E402
from surface_crack.losses import (  # noqa: E402
    DEFAULT_SKELETON_ITERATIONS,
    segmentation_loss,
)
from surface_crack.metrics import GroupedBinarySegmentationMetrics  # noqa: E402
from surface_crack.models import UNet  # noqa: E402
from surface_crack.resnet_unet import PoolAdapterResNet18UNet  # noqa: E402
from surface_crack.research_pooling import install_research_pooling  # noqa: E402
from surface_crack.spatial import (  # noqa: E402
    crop_prediction_to_original,
    pad_image_and_mask,
)


POOLING_CHOICES = (
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
    "upper_tail_capacity_matched",
    "diag_partition_matched",
    "btsp_v2_content",
    "t_max_avg",
    "btsp",
)
BACKBONE_CHOICES = ("unet", "pool_adapter_resnet18_unet")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pooling", choices=POOLING_CHOICES, required=True)
    parser.add_argument("--backbone", choices=BACKBONE_CHOICES, default="unet")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=BUNDLE_ROOT / "runs")
    parser.add_argument("--run-name")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--amp-dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--experiment-stage",
        choices=("engineering", "exploratory_validation", "confirmatory_validation"),
        default="engineering",
    )
    parser.add_argument("--authorization", type=Path)
    # 0 = 不裁剪，只测量。测量本身对数值零影响 (clip_grad_norm_ 传 inf 不改梯度)。
    parser.add_argument("--grad-clip", type=float, default=0.0)
    # 0 = 不做 warmup，schedule 保持契约冻结的 constant，对现有 run 零影响。
    parser.add_argument("--warmup-epochs", type=int, default=0)
    # 2026-10-29（S21）：constant = 预热后保持 base lr，是冻结契约的行为，默认值，
    # 对既有 run 零影响。cosine = 预热结束后按全局步余弦衰减，在本次训练最后一步
    # 降到 0；周期以本次 --epochs 为终点，因此不同预算是相互独立的训练，不能截断互推。
    parser.add_argument("--schedule", choices=("constant", "cosine"), default="constant")
    # none = 原版无归一化基线，默认值，保持既有 run 的数值行为不变。
    # none 与 group 适用于主骨干；batch 只适用于第二骨干，是它的冻结配置。
    parser.add_argument("--norm", choices=("none", "group", "batch"), default="none")
    # 0 = 不加中心线项，损失与冻结契约逐位相同，对既有 run 零影响。
    # 大于 0 时损失变为 BCE + Dice + w * soft-clDice，属训练契约偏离，
    # 需单独的偏离记录与预注册权重，不得在验证集上调 w。
    parser.add_argument("--cldice-weight", type=float, default=0.0)
    parser.add_argument(
        "--cldice-iterations", type=int, default=DEFAULT_SKELETON_ITERATIONS
    )
    return parser.parse_args()


FORMAL_STAGES = frozenset({"exploratory_validation", "confirmatory_validation"})


def _authorization_sha256(args: argparse.Namespace) -> str | None:
    """正式阶段的守门：没有有效授权记录就不许打正式阶段的戳。

    summarize_train_val_runs.py 要 result.json 同时带 experiment_stage 与
    train_val_execution_authorization_sha256 才判正式资格。这两个字段以前根本没人写，
    所以 2026-09-08 那轮 36 支跑完仍全判 missing_formal_validation_stage 且无法追认。
    这里失败闭合：--experiment-stage 一旦不是 engineering，就必须给出一份 status 为
    TRAIN_VAL_EXECUTION_AUTHORIZED、且 test 与 external_test 均未授权的记录，
    否则当场拒跑。本函数只读记录、只算哈希，不含任何能解锁 test 的路径或令牌。
    """

    if args.experiment_stage not in FORMAL_STAGES:
        if args.authorization is not None:
            raise ValueError(
                "--authorization is only meaningful for a formal experiment stage"
            )
        return None
    if args.authorization is None:
        raise ValueError(
            f"--experiment-stage {args.experiment_stage} requires --authorization"
        )
    path = args.authorization.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"authorization record not found: {path}")
    record = json.loads(path.read_text(encoding="utf-8-sig"))
    if record.get("status") != "TRAIN_VAL_EXECUTION_AUTHORIZED":
        raise ValueError(
            f"authorization record is not train/validation authorized: {path}"
        )
    if record.get("test_authorized") is not False:
        raise ValueError(f"authorization record must keep test locked: {path}")
    if record.get("external_test_authorized") is not False:
        raise ValueError(
            f"authorization record must keep external_test locked: {path}"
        )
    stages = record.get("authorized_stages")
    if isinstance(stages, list) and args.experiment_stage not in stages:
        raise ValueError(
            f"stage {args.experiment_stage!r} is not in authorized_stages {stages!r}"
        )
    return _sha256(path)


def _validate_arguments(args: argparse.Namespace) -> None:
    for value, name in (
        (args.epochs, "epochs"),
        (args.base_channels, "base_channels"),
        (args.learning_rate, "learning_rate"),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if args.workers < 0:
        raise ValueError("workers must be non-negative")
    if args.max_train_samples < 0 or args.max_val_samples < 0:
        raise ValueError("sample limits must be non-negative")
    if args.cldice_weight < 0:
        raise ValueError("cldice_weight must be non-negative")
    if args.cldice_iterations < 0:
        raise ValueError("cldice_iterations must be non-negative")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _limited(records: list[object], limit: int, seed: int) -> list[object]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    return shuffled if limit == 0 else shuffled[:limit]


def _segmentation_loss(
    logits: Tensor,
    target: Tensor,
    *,
    cldice_weight: float = 0.0,
    cldice_iterations: int = DEFAULT_SKELETON_ITERATIONS,
) -> Tensor:
    """BCE + soft Dice, optionally plus soft-clDice.

    2026-09-23: the body moved to ``surface_crack.losses.segmentation_loss`` so
    that the centreline term could be unit tested on its own.  With
    ``cldice_weight == 0`` the result is bit-identical to the frozen loss, which
    ``tests/test_losses.py`` pins, so runs without the flag are unaffected.

    The original note on the float32 Dice reduction moved with it: CRACK500 的
    裁剪块是 640x360 = 230400 像素，fp16 上限 65504，平均预测概率一旦超过
    0.2843，probabilities.sum() 就溢出成 inf，Dice 项被钉死在 1.0 且梯度失效。
    """

    return segmentation_loss(
        logits,
        target,
        cldice_weight=cldice_weight,
        cldice_iterations=cldice_iterations,
    )


def _build_model(
    backbone: str, *, base_channels: int, pooling: str, norm: str = "none"
) -> nn.Module:
    if backbone == "unet":
        if norm == "batch":
            # 主骨干冻结时刻意不含归一化：有效批量 1 时最深特征图是 1x1，
            # BatchNorm 与 InstanceNorm 在那里没有定义，GroupNorm 有。
            raise ValueError(
                "--norm batch is not available for the unet backbone; batch size 1 "
                "leaves BatchNorm undefined at the 1x1 deepest feature map"
            )
        return UNet(
            in_channels=3,
            base_channels=base_channels,
            pooling=pooling,
            norm=norm,
        )
    if backbone == "pool_adapter_resnet18_unet":
        # 2026-09-23 之前这里拒绝任何非 none 的 norm，理由是"第二骨干尚未接 norm
        # 开关"。但拒绝并没有让两个骨干变得可比：第二骨干全程硬编码 BatchNorm，
        # 而有效批量是 1，训练时它等效 InstanceNorm、评估时依赖单图估出的
        # running statistics，两个阶段都与主骨干（norm=group）不同。
        # 现在 norm 已贯通第二骨干，--norm group 才真正让两者可比。
        # 该骨干的 norm=none 是"完全没有归一化的残差网络"，不是冻结配置，
        # 冻结配置是 batch，所以这里要求显式给出而不做隐式映射。
        if norm == "none":
            raise ValueError(
                "for pool_adapter_resnet18_unet pass --norm batch for the frozen "
                "configuration or --norm group to make it comparable with the unet "
                "backbone; --norm none would leave a residual network unnormalised"
            )
        return PoolAdapterResNet18UNet(
            in_channels=3, base_channels=base_channels, pooling=pooling, norm=norm
        )
    raise ValueError(f"unsupported backbone: {backbone!r}")


def _native_forward(model: nn.Module, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    padded_image, _, metadata = pad_image_and_mask(image, mask, multiple=16)
    padded_logits = model(padded_image)
    return crop_prediction_to_original(padded_logits, metadata), mask


def _validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    # 关掉 AMP 时这个值用不上，给默认值是为了不改动已冻结的
    # test_train_validation_metrics_binding.py（它按 amp_enabled=False 四参调用）。
    amp_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, float | int]:
    model.eval()
    metrics = GroupedBinarySegmentationMetrics(prediction_space="logits")
    extended = GroupedExtendedSegmentationMetrics(prediction_space="logits")
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                logits, native_mask = _native_forward(model, image, mask)
            group_ids = [batch["group_id"][0]]
            logits = logits.float()
            metrics.update(logits, native_mask, group_ids)
            extended.update(logits, native_mask, group_ids)
    result = metrics.compute()
    result.update(extended.compute())
    return result


def _learned_pooling_scalars(model: nn.Module) -> list[float]:
    values: list[float] = []
    for layer in model.downsamplers:
        if hasattr(layer, "alpha"):
            values.append(float(layer.alpha.detach().cpu()))
        elif hasattr(layer, "t"):
            values.append(float(layer.t.detach().cpu()))
        elif hasattr(layer, "weights"):
            # 2026-09-23: V0 的两支把混合权重命名为 weights，既不是 alpha 也不是 t，
            # 于是 v0_max_mean_min 与 v0_max_mean_matched 的这一列一直是空列表，
            # 它们学到了什么从未被记录过。softmax 权重是每层三个，按层顺序摊平。
            values.extend(float(v) for v in layer.weights.detach().cpu())
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_checkpoint_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _rng_state(generator: torch.Generator) -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_generator": generator.get_state(),
    }


def _restore_rng_state(state: dict[str, object], generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    generator.set_state(state["loader_generator"])


def main() -> None:
    args = _arguments()
    _validate_arguments(args)
    _seed_everything(args.seed)
    install_research_pooling()

    config_path = args.config.resolve(strict=True)
    data_config = load_baseline_data_config(config_path)
    records = load_train_val_records(data_config)
    train_records = _limited(
        [record for record in records if record.split == "train"],
        args.max_train_samples,
        args.seed,
    )
    val_records = _limited(
        [record for record in records if record.split == "val"],
        args.max_val_samples,
        args.seed + 1,
    )
    if not train_records or not val_records:
        raise RuntimeError("both train and validation records are required")

    train_dataset = SurfaceCrackDataset(train_records, mask_spec=data_config.mask_spec)
    val_dataset = SurfaceCrackDataset(val_records, mask_spec=data_config.mask_spec)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    authorization_sha256 = _authorization_sha256(args)
    amp_enabled = bool(args.amp and device.type == "cuda")
    # AMP 默认 bfloat16 而非 float16。models.py 的 UNet 不含任何归一化层
    # (BatchNorm/GroupNorm/InstanceNorm/LayerNorm 全无)，配上有效批量 1 与常数 lr 1e-3，
    # 前向激活会偶发越过 fp16 上限 65504，logits 变 inf，验证阶段 metrics 直接抛
    # "predictions must contain only finite values"，整支 run 中途死掉。
    # 2026-09-08 的 12 算子 x 3 种子矩阵有 7/36 支这样死亡，跨全部算子族，非算子缺陷。
    # bfloat16 的指数范围与 fp32 相同，溢出消失；sm_120 上吞吐与 fp16 相同。
    # 冻结契约只写 "amp": "enabled_when_cuda_is_available"，未钉 dtype，故属数值稳定性
    # 修复而非超参改动。见 decision_log_2026-09-09_amp_dtype_bfloat16.md。
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    model = _build_model(
        args.backbone,
        base_channels=args.base_channels,
        pooling=args.pooling,
        norm=args.norm,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    # bfloat16 指数范围同 fp32，不需要梯度缩放，只有 fp16 才开 scaler。
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and amp_dtype is torch.float16
    )

    run_name = args.run_name or (
        f"{data_config.dataset_id}_{args.backbone}_{args.pooling}_s{args.seed}"
    )
    run_dir = args.output_dir.resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = run_dir / "latest.pt"
    best_path = run_dir / "best.pt"
    result_path = run_dir / "result.json"

    start_epoch = 0
    best_fg_iou = float("-inf")
    best_fg_iou_ma = float("-inf")
    best_epoch_ma = 0
    fg_iou_window: list[float] = []
    history: list[dict[str, object]] = []
    elapsed_before = 0.0
    if args.resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {latest_path}")
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        expected = {
            "dataset_id": data_config.dataset_id,
            "backbone": args.backbone,
            "pooling": args.pooling,
            "base_channels": args.base_channels,
            "seed": args.seed,
            "config_sha256": _sha256(config_path),
            "amp_dtype": args.amp_dtype if amp_enabled else None,
            "experiment_stage": args.experiment_stage,
            # 2026-09-23 补入。以前这三项不校验：改 max_*_samples 会改变
            # len(train_loader)，既换了训练预算又打乱 warmup 的 global_step 推算；
            # resume 时把 warmup_epochs 改成 0，lr 会停在恢复出来的半程 warmup 值上
            # 并永远保持；grad_clip 换了则梯度处理换了而无任何痕迹。
            "warmup_epochs": args.warmup_epochs,
            "grad_clip": args.grad_clip,
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "norm": args.norm,
            "cldice_weight": args.cldice_weight,
            "schedule": args.schedule,
        }
        legacy_defaults = {
            "warmup_epochs": 0,
            "grad_clip": 0.0,
            "max_train_samples": None,
            "max_val_samples": None,
            "norm": "none",
            "cldice_weight": 0.0,
            "schedule": "constant",
        }
        for key, value in expected.items():
            if key == "backbone":
                observed = checkpoint.get(key, "unet")
            elif key == "experiment_stage":
                observed = checkpoint.get(key, "engineering")
            elif key in legacy_defaults:
                observed = checkpoint.get(key, legacy_defaults[key])
            elif key == "amp_dtype":
                # 2026-09-09 之前的检查点没有这个键，一律按 float16 认，
                # 让 fp16 权重被 bf16 续跑时当场撞停而不是静默混精度。
                observed = checkpoint.get(key, "float16")
            else:
                observed = checkpoint.get(key)
            if observed != value:
                raise RuntimeError(
                    f"resume mismatch for {key}: {observed!r} != {value!r}"
                )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        best_fg_iou = float(checkpoint["best_fg_iou"])
        best_fg_iou_ma = float(checkpoint.get("best_fg_iou_ma", float("-inf")))
        best_epoch_ma = int(checkpoint.get("best_epoch_ma", 0))
        fg_iou_window = list(checkpoint.get("fg_iou_window", []))
        history = list(checkpoint["history"])
        elapsed_before = float(checkpoint.get("elapsed_seconds", 0.0))
        _restore_rng_state(checkpoint["rng_state"], generator)
        print(f"resumed={latest_path} completed_epochs={start_epoch}", flush=True)
    elif latest_path.exists():
        raise FileExistsError(
            f"run already has a checkpoint; pass --resume or choose --run-name: {run_dir}"
        )

    if start_epoch >= args.epochs:
        print(f"already_complete={start_epoch}/{args.epochs} result={result_path}")
        return

    session_started = time.perf_counter()
    for epoch_index in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        grad_norms: list[float] = []
        warmup_steps = args.warmup_epochs * len(train_loader)
        clipped_steps = 0
        nonfinite_grad_steps = 0
        for step_in_epoch, batch in enumerate(train_loader):
            # 线性 warmup：按全局步而非按轮插值，前 warmup_epochs 轮内 lr 从 0 升到 base。
            # global_step 由 epoch_index 推出，不进检查点，--resume 后自动落在正确位置。
            if warmup_steps > 0:
                global_step = epoch_index * len(train_loader) + step_in_epoch + 1
                if global_step <= warmup_steps:
                    scaled = args.learning_rate * global_step / warmup_steps
                    for group in optimizer.param_groups:
                        group["lr"] = scaled
            # S21：余弦段只在 --schedule cosine 时进入；constant 不经过这里，上面的
            # 预热与此前逐位相同。total_steps 由 --epochs 推出，resume 后同样落在原位。
            if args.schedule == "cosine":
                total_steps = args.epochs * len(train_loader)
                global_step = epoch_index * len(train_loader) + step_in_epoch + 1
                if global_step > warmup_steps:
                    progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
                    scaled = args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
                    for group in optimizer.param_groups:
                        group["lr"] = scaled
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                logits, native_mask = _native_forward(model, image, mask)
                loss = _segmentation_loss(
                    logits,
                    native_mask,
                    cldice_weight=args.cldice_weight,
                    cldice_iterations=args.cldice_iterations,
                )
            scaler.scale(loss).backward()
            # 先解缩放再量范数：scaler 开着时梯度带着缩放因子，不解开量出来的是假值。
            # scaler 关闭 (bfloat16) 时 unscale_ 是空操作。step() 会识别已解缩放，不会重复解。
            scaler.unscale_(optimizer)
            step_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip if args.grad_clip > 0 else float("inf"),
                )
            )
            if math.isfinite(step_norm):
                grad_norms.append(step_norm)
                if args.grad_clip > 0 and step_norm > args.grad_clip:
                    clipped_steps += 1
                scaler.step(optimizer)
            else:
                nonfinite_grad_steps += 1
                # 2026-09-23: 以前这里只计数不跳过。fp16 路径有 GradScaler 兜底，
                # 它自己会在梯度非有限时跳过 step；bfloat16 路径 scaler 是禁用的，
                # scaler.step 退化成无条件 optimizer.step()，于是一个 NaN/Inf 梯度
                # 会被直接写进权重，整支 run 从此污染，而唯一痕迹只是这个计数。
                # 已核对：截至本次修改，84 支正式 run 的该计数全为 0，无既有结果受影响。
                optimizer.zero_grad(set_to_none=True)
            scaler.update()
            running_loss += float(loss.detach().cpu())

        epoch_number = epoch_index + 1
        epoch_loss = running_loss / len(train_loader)
        validation = _validate(model, val_loader, device, amp_enabled, amp_dtype)
        fg_iou = float(validation["foreground_iou"])
        # checkpoint 选择改用拖尾 MA(5)，不再用单轮最大值。
        # 单轮 val FgIoU 的轮间 σ 中位数 0.0074，而 best-of-N 等于取 30 个带噪观测的最大值，
        # 奇偶折半实测其选择性乐观中位数 +0.0058 = H1 门槛 0.005 的 1.2 倍，
        # 即两个完全相同的算子仅凭噪声尖峰即可造出超过门槛的假差距。
        # 拖尾 MA(5) 把这个乐观降到 +0.0015 (0.3 倍门槛)，且只用过去轮次，训练中可算。
        # 见 decision_log_2026-09-15_ma5_checkpoint_selection.md。
        fg_iou_window.append(fg_iou)
        smoothed = (
            sum(fg_iou_window[-MA_WINDOW:]) / MA_WINDOW
            if len(fg_iou_window) >= MA_WINDOW
            else None
        )
        is_best = smoothed is not None and smoothed > best_fg_iou_ma
        if is_best:
            best_fg_iou_ma = smoothed
            best_epoch_ma = epoch_number
        if fg_iou > best_fg_iou:
            best_fg_iou = fg_iou
        history.append(
            {
                "epoch": epoch_number,
                "train_loss": epoch_loss,
                "grad_norm_max": max(grad_norms) if grad_norms else None,
                "grad_norm_mean": (
                    sum(grad_norms) / len(grad_norms) if grad_norms else None
                ),
                "grad_norm_p99": (
                    sorted(grad_norms)[int(len(grad_norms) * 0.99)]
                    if len(grad_norms) >= 100
                    else None
                ),
                "clipped_steps": clipped_steps,
                "nonfinite_grad_steps": nonfinite_grad_steps,
                # S21：本轮最后一步使用的学习率，用于核对调度。只记录，不参与计算。
                "lr_end": float(optimizer.param_groups[0]["lr"]),
                "validation": validation,
            }
        )
        elapsed = elapsed_before + time.perf_counter() - session_started
        checkpoint_payload = {
            "epoch": epoch_number,
            "dataset_id": data_config.dataset_id,
            "backbone": args.backbone,
            "pooling": args.pooling,
            "base_channels": args.base_channels,
            "seed": args.seed,
            "config_sha256": _sha256(config_path),
            "amp_dtype": args.amp_dtype if amp_enabled else None,
            "experiment_stage": args.experiment_stage,
            # 2026-09-23：这五项进检查点是为了让 --resume 能真的校验它们。
            # 2026-09-23 之前写出的检查点没有这些键，resume 守卫按各自的
            # 历史默认值回退，因此旧检查点仍可续跑。
            "warmup_epochs": args.warmup_epochs,
            "grad_clip": args.grad_clip,
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "norm": args.norm,
            "cldice_weight": args.cldice_weight,
            "cldice_iterations": args.cldice_iterations,
            "schedule": args.schedule,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_fg_iou": best_fg_iou,
            "best_fg_iou_ma": best_fg_iou_ma,
            "best_epoch_ma": best_epoch_ma,
            "fg_iou_window": fg_iou_window,
            "history": history,
            "elapsed_seconds": elapsed,
            "rng_state": _rng_state(generator),
            "test_or_external_test_used": False,
        }
        _save_checkpoint_atomic(latest_path, checkpoint_payload)
        if is_best:
            shutil.copy2(latest_path, best_path)

        result: dict[str, object] = {
            "status": "COMPLETE" if epoch_number == args.epochs else "RUNNING",
            "dataset_id": data_config.dataset_id,
            "backbone": args.backbone,
            "config_sha256": _sha256(config_path),
            "pooling": args.pooling,
            "seed": args.seed,
            "epochs_completed": epoch_number,
            "epochs_target": args.epochs,
            "base_channels": args.base_channels,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            # 2026-09-23：--resume 时 optimizer.load_state_dict 会用检查点里的
            # param_groups 覆盖刚按命令行设好的 lr 与 weight_decay，而上面两行记的
            # 是命令行值，于是一份声称 lr=3e-4 的 result.json 可能实际跑在 1e-3 上。
            # 下面两行记录本轮结束时 optimizer 里的真实值，两者不一致即说明发生了覆盖。
            # 上面两行保持命令行语义不变，因为 run_training.py 的设置比对依赖它，
            # 而 warmup 期间 optimizer 里的 lr 是缩放后的中间值。
            "learning_rate_effective": float(optimizer.param_groups[0]["lr"]),
            "weight_decay_effective": float(optimizer.param_groups[0]["weight_decay"]),
            "optimizer_state_restored": bool(args.resume),
            "experiment_stage": args.experiment_stage,
            "train_val_execution_authorization_sha256": authorization_sha256,
            "amp": amp_enabled,
            "amp_dtype": args.amp_dtype if amp_enabled else None,
            "grad_clip": args.grad_clip,
            "warmup_epochs": args.warmup_epochs,
            "schedule": args.schedule,
            "norm": args.norm,
            # 2026-09-23：损失配置进 result.json，否则无法从产物分辨一支 run
            # 用的是冻结契约的 BCE+Dice 还是加了中心线项的版本。
            "cldice_weight": args.cldice_weight,
            "cldice_iterations": args.cldice_iterations,
            "device": str(device),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "history": history,
            "best_foreground_iou_ma5": best_fg_iou_ma if best_fg_iou_ma != float("-inf") else None,
            "best_epoch_ma5": best_epoch_ma or None,
            "best_foreground_iou": best_fg_iou,
            "learned_pooling_scalars": _learned_pooling_scalars(model),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "elapsed_seconds": elapsed,
            "latest_checkpoint": "latest.pt",
            # 2026-09-23：以前这里无条件写 "best.pt"。轮数少于 5 时 MA(5) 取不到值，
            # is_best 恒为 False，best.pt 从不被创建，而 result.json 仍指向它。
            "best_checkpoint": "best.pt" if best_path.is_file() else None,
            # best.pt 存的是 MA(5) 窗口右端那一轮的权重，不是"取得该 MA(5) 值的对象"。
            # 二者不是同一组权重：用 best.pt 做推理得到的单轮分数与本文件报告的
            # best_foreground_iou_ma5 会系统性不符。任何后续 test 阶段推理都必须
            # 先读这一条，不能把两个数当成同一个量。
            "best_checkpoint_selection": (
                "trailing_ma5_argmax_window_right_end" if best_path.is_file() else None
            ),
            "test_or_external_test_used": False,
        }
        _write_json_atomic(result_path, result)
        print(
            f"epoch={epoch_number}/{args.epochs} train_loss={epoch_loss:.6f} "
            f"val_fg_iou={fg_iou:.6f} best={best_fg_iou:.6f}",
            flush=True,
        )

    print(f"result_path={result_path}", flush=True)


if __name__ == "__main__":
    main()
