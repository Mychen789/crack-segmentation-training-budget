# Decision log: fp32 reduction for the soft Dice term

Date: 2026-09-08 (recorded retroactively on 2026-09-09)
Decision: KEEP THE FIX; QUARANTINE EVERY RUN PRODUCED BEFORE IT

## Problem

Under `torch.autocast(dtype=torch.float16)` the soft Dice term reduced in fp16.
A CRACK500 crop is 640 x 360 = 230400 pixels and the fp16 maximum is 65504, so
`probabilities.sum()` overflows to inf once the mean predicted probability
exceeds about 0.2843. At random initialisation sigmoid outputs are near 0.5, so
the overflow is reached immediately: the Dice term pins at 1.0, its gradient
vanishes, and the corrupted weights later make the validation forward emit
non-finite logits, which `metrics.py` rejects.

## Decision

`_segmentation_loss` in `baseline_platform/tools/train_pooling_experiment.py`
now casts to fp32 before the Dice reduction. `binary_cross_entropy_with_logits`
already sits on the autocast fp32 list and was left unchanged.

Runs produced before the fix were moved to
`baseline_platform/runs/_contaminated_fp16dice/` and are not eligible for any
comparison. Their `result.json` files are retained as evidence; their
checkpoints were deleted on 2026-09-09 to reclaim disk (see
`baseline_platform/runs/_logs/cleanup_20260909.md`).

## Audit chain

`PACKAGE_CONTENTS_SHA256.json` records the pre-change tool as
`d8e13e78146b0c5f821c679a74023a25f594ab937db9e9eb15e83215adc9cb52`.
The intermediate hash of the fp32-Dice version was not recorded before the
2026-09-09 bfloat16 change overwrote the file, and it is not reconstructed or
claimed here. The current hash is recorded in
`decision_log_2026-09-09_amp_dtype_bfloat16.md`.

## Scope

- Evidence used: training and validation data only.
- Changes H1, H2, H3, primary metric, stratification thresholds or Go/No-Go: no.
- Fair re-run of all affected arms required: yes.
- Authorises formal train/validation execution: no.
- Authorises test or external-test access: no.
