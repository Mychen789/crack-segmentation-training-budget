# Decision log: pre-formal engineering checks

Date: 2026-08-29  
Status: RECORDED BEFORE FORMAL TRAIN/VALIDATION EXECUTION  
Scope: engineering smoke and train-only overfit diagnostics

## What changed

The historical implementation-status sentence in `preregistered_protocol_v0.3.md`
states that no validation predictions had been generated at the time of protocol
freeze. After that freeze, the user explicitly authorized a minimal engineering
smoke check on the current device.

The smoke check used:

- dataset configuration: CRACK500 train/validation only;
- method: MaxPool, not the proposed BTSP method;
- model size: base channels 16;
- sample limits: 64 train and 16 validation samples;
- budget: one epoch;
- seed: 2026;
- purpose: data loading, CUDA/AMP, checkpoint, result writing, and resume only.

It generated validation metrics for the 16-sample engineering subset. Therefore,
the historical statement "no validation predictions generated" is no longer a
current-state description, although it remains true as a statement of the state
at protocol freeze.

A second diagnostic trained MaxPool and BTSP separately on the same fixed eight
training samples for 30 epochs and evaluated only those same training samples.
It did not open validation, test, or external-test data.

## Scientific-use restriction

- Neither diagnostic is a formal experiment.
- Neither diagnostic may enter a paper table, effect estimate, method choice,
  hyperparameter choice, Go/No-Go decision, or result-dependent narrative.
- The overfit diagnostic cannot be cited as evidence that BTSP outperforms
  MaxPool because it is a tiny train-only engineering check.
- No test or external-test sample was requested or opened.
- The formal train/validation execution remains locked.

## Artifacts

- `baseline_platform/runs/smoke_crack500_max_b16_s2026_64x16_e1/result.json`
- `baseline_platform/runs/overfit_max_btsp_b16_n8_e30_s2026/summary.json`
- `baseline_platform/artifacts/train_val_result_index_v0.1.json`

The result index classifies the existing smoke run as engineering/ineligible and
contains zero formal results.

## Impact on preregistered hypotheses

H1, H2, and H3 remain untested. No effect threshold, dataset role, statistical
unit, seed set, degradation plan, title gate, test lock, or CFD lock is changed.

## Required action before formal execution

Generate a separate, hash-bound train/validation execution authorization only
after the complete training contract and implementation hashes are frozen. This
decision log does not authorize any further validation experiment.

