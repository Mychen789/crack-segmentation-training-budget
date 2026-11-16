# Decision log: AMP dtype changed from float16 to bfloat16

Date: 2026-09-09
Decision: DEFAULT AMP DTYPE IS bfloat16; float16 REMAINS SELECTABLE

## Problem

The fp32 Dice reduction recorded in
`decision_log_2026-09-08_fp32_dice_reduction.md` removed one overflow path but
not the other. In the 12-operator x 3-seed CRACK500 matrix of 2026-09-08
(`baseline_platform/runs/_logs/formal_20260908_103705.log`, 12.76 hours,
36 jobs) 7 jobs still terminated with

    ValueError: predictions must contain only finite values
    (src/surface_crack/metrics.py:125, reached from _validate)

after 2 to 25 of 30 epochs. The 7 failures were `avg_s2027`,
`avg_top2_s2028`, `btsp_s2027`, `stride_conv_s2026`, `t_max_avg_s2027`,
`upper_tail_matched_s2028` and `v0_max_mean_min_s2027` - spread across every
operator family including plain average pooling and strided convolution, so the
cause is not a property of any pooling operator.

`src/surface_crack/models.py` contains no normalisation layer of any kind
(no BatchNorm, GroupNorm, InstanceNorm or LayerNorm). With an effective batch
size of 1 and a constant learning rate of 1e-3, forward activations
intermittently exceed the fp16 ceiling of 65504 and the logits become inf.

## Decision

`torch.autocast` now runs in bfloat16 by default. bfloat16 carries the same
exponent range as fp32, so the overflow disappears, and on sm_120 its
throughput matches fp16: a single full CRACK500 epoch with the BTSP operator
took 79.9 s in bfloat16 against 78.4 s per epoch in float16, a 1.9 percent
difference.

Changes to `baseline_platform/tools/train_pooling_experiment.py`:

- new `--amp-dtype {bfloat16,float16}`, default `bfloat16`;
- both autocast blocks (training and validation) use the selected dtype;
- `GradScaler` is enabled only for float16, since bfloat16 needs no loss
  scaling;
- `amp_dtype` is written into `result.json` and into the checkpoint, and is
  checked on `--resume`; checkpoints written before this change are read as
  `float16`, so resuming fp16 weights under bfloat16 fails loudly instead of
  silently mixing precisions.

`_validate` received a default value for its new parameter so that the frozen
test `test_train_validation_metrics_binding.py`, which calls it with
`amp_enabled=False` and four arguments, was not modified. The full suite is
green: 97 passed, 48 subtests.

## Why this is not a contract change

`baseline_platform/artifacts/training_contract_freeze_v1.json` freezes AMP as
`"amp": "enabled_when_cuda_is_available"` and does not pin a dtype. Optimizer,
learning rate, schedule, epochs, effective batch size, augmentation, loss and
checkpoint selection are untouched. Adding normalisation layers to the model was
rejected as the alternative: it would alter the architecture and break the
1941105-parameter agreement with the reference machine.

## Audit chain

`baseline_platform/tools/train_pooling_experiment.py`

- frozen in `PACKAGE_CONTENTS_SHA256.json`:
  `d8e13e78146b0c5f821c679a74023a25f594ab937db9e9eb15e83215adc9cb52`
- after the bfloat16 change:
  `3f562429ee8e4a0a44c8d810fb701746795f8687d2c93a73542d14893cd41a91`
- superseded the same day by a second change to the same file: the trainer now
  writes `experiment_stage` and `train_val_execution_authorization_sha256` into
  `result.json`, refuses a formal stage without a valid authorization record,
  and carries the stage through the checkpoint and the `--resume` consistency
  check. Current hash:
  `d887cdcd8e270f9f1a6d96de7ba03ecd19be4f810d64687609bd5944de28a64f`.
  A dedicated log for that change is still to be written; it was not produced in
  the 2026-09-09 session because the write was refused by the assistant's
  permission layer.

`PACKAGE_CONTENTS_SHA256.json` itself is not rewritten; the freeze record is
left intact and the drift is carried by this log.

## Status of the 2026-09-08 matrix

All 36 runs were moved to `baseline_platform/runs/_pilot_fp16amp/`. They remain
engineering-only for a reason that is independent of the numerical fault:
`train_pooling_experiment.py` never wrote `experiment_stage` or
`train_val_execution_authorization_sha256`, so
`summarize_train_val_runs.py` classifies all of them as
`missing_formal_validation_stage`. The current index reports
`formal=0, engineering_or_ineligible=75`. These runs cannot be promoted
retroactively and are treated as a pilot.

## Scope

- Evidence used: training and validation data only.
- Changes H1, H2, H3, primary metric, stratification thresholds or Go/No-Go: no.
- Fair re-run of all affected arms required: yes, the whole matrix must be
  re-run under one dtype; mixing bfloat16 and float16 arms would confound the
  operator comparison with a precision difference.
- Authorises formal train/validation execution: no. The two readiness blockers
  (`train_val_authorization_present`, `plan_execution_status`) are untouched.
- Authorises test or external-test access: no.
