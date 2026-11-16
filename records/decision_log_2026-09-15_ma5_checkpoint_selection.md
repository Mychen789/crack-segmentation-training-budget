# Decision log: trailing five-epoch mean replaces the raw validation maximum

Date: 2026-09-15
Decision: CHANGE THE CHECKPOINT-SELECTION AND REPORTING RULE; RECOMPUTE, DO NOT RETRAIN

## The defect in the frozen rule

`training_contract_freeze_v1.json` froze
`checkpoint: {selection_metric: validation_group_macro_foreground_iou,
direction: maximum, tie_break: earliest_epoch}`.

Taking the maximum over 30 epochs means taking the maximum of 30 noisy
observations. Measured across the two authorised matrices, the per-epoch
validation FgIoU has a median epoch-to-epoch standard deviation of 0.0074, while
the pre-registered H1 threshold is 0.005. The rule therefore reports a quantity
whose noise exceeds the effect it must resolve.

Selection optimism was estimated without any new training, by splitting each
run's epoch sequence into odd and even subsequences, selecting on one half and
reading the matched position on the other. Over 55 non-diverged runs from v1 and
v2:

| rule | median selection optimism | multiple of the 0.005 threshold |
|---|---|---|
| raw maximum (frozen rule) | +0.0058 | 1.2 |
| max of trailing mean, window 3 | +0.0031 | 0.6 |
| **max of trailing mean, window 5** | **+0.0015** | **0.3** |
| max of trailing mean, window 7 | -0.0036 | -0.7 |
| max of centred mean, window 5 | +0.0001 | 0.0 |

Two identical operators can therefore differ by more than the H1 threshold on
noise alone under the frozen rule.

## What changed

The reported epoch is chosen by the maximum of a trailing five-epoch mean of the
validation foreground IoU, and every reported validation metric is averaged over
the same window, so the primary metric and the secondaries describe one
consistent estimate. Runs shorter than five epochs fall back to the old rule
rather than emitting a meaningless partial average.

A centred window measures marginally better (+0.0001) but cannot be computed
during training. The trailing window was chosen so that the trainer applies the
same rule live that the summariser applies afterwards; +0.0015 is accepted as
the price of that consistency.

- `baseline_platform/tools/train_pooling_experiment.py` selects `best.pt` by the
  trailing mean and records `best_foreground_iou_ma5` and `best_epoch_ma5`.
  `best_foreground_iou` is still written as the raw maximum, for continuity.
- `baseline_platform/tools/summarize_train_val_runs.py` reports the smoothed
  metrics and adds a `foreground_iou_raw_max` column, so any earlier table can
  be reconciled against the new one.

Nothing else moves: optimizer, learning rate, weight decay, epochs, effective
batch size, augmentation, loss and the warmup schedule are untouched.

## Effect on the existing data

The rule is computed from `history`, so both authorised matrices were recomputed
without retraining. Over the 23 non-diverged runs of v2:

| operator | n | trailing MA(5) | between-seed sd | old raw maximum | change |
|---|---|---|---|---|---|
| `v0_max_mean_matched` | 1 | 0.5611 | — | 0.5704 | -0.0094 |
| `btsp` | 3 | 0.5610 | 0.0015 | 0.5641 | -0.0031 |
| `max` | 1 | 0.5589 | — | 0.5661 | -0.0072 |
| `learned_max_avg` | 3 | 0.5543 | 0.0126 | 0.5656 | -0.0113 |
| `v0_max_mean_min` | 2 | 0.5528 | 0.0157 | 0.5692 | -0.0163 |
| `fixed_max_avg` | 2 | 0.5508 | 0.0103 | 0.5643 | -0.0135 |
| `min` | 2 | 0.5502 | 0.0027 | 0.5550 | -0.0049 |
| `avg` | 2 | 0.5430 | 0.0227 | 0.5647 | -0.0218 |
| `upper_tail_matched` | 2 | 0.5408 | 0.0050 | 0.5513 | -0.0105 |
| `stride_conv` | 3 | 0.4693 | 0.0314 | 0.5043 | -0.0350 |

Every value falls, but not uniformly: the reduction tracks the between-seed
standard deviation almost exactly. `btsp`, the least dispersed arm at 0.0015,
loses 0.0031; `avg` at 0.0227 loses 0.0218; `stride_conv` at 0.0314 loses 0.0350.
The frozen rule was subsidising unstable arms in proportion to their instability.

This reorders the table. `btsp` moves from seventh to second and `avg` from fifth
to ninth. The H1 contrast `btsp` minus `upper_tail_matched` widens from +0.0128
to +0.0202, four times the threshold, because the noisier control arm was the
larger beneficiary of the old rule.

These figures are not a result. They come from the subset of runs that did not
diverge, which is a post-hoc selection on outcome, with one to three seeds per
operator. They are reported here only to document what the rule change does.

## Scope

- Evidence used: training and validation data only.
- Changes H1, H2, H3, the primary metric definition, stratification thresholds
  or Go/No-Go: the checkpoint selection rule changes as described; the metric
  itself, group-macro foreground IoU, is unchanged, as are all hypotheses and
  thresholds.
- Fair re-run of all affected arms required: no. The rule is computed from
  recorded history and was applied to both existing matrices without retraining.
  Checkpoint files written under the old rule may point at a different epoch than
  the reported one; any run whose `best.pt` is later needed for test-stage
  inference must be retrained under this rule first.
- Authorises formal train/validation execution: no.
- Authorises test or external-test access: no.

## Audit chain

`baseline_platform/tools/train_pooling_experiment.py`

- after `--warmup-epochs`:
  `6e54d4bcf6b22e23e4aa3bb2d6e78c03276c8bbda9f29b68b50bb45ee6d8e757`
- after this change:
  `34c1622b60b38f1fa04d432e6d1a61159ca9f99f51d75f6d436f0bfe8e641808`

`baseline_platform/tools/summarize_train_val_runs.py`

- as frozen in `PACKAGE_CONTENTS_SHA256.json`:
  `63490a7e79dc8914ca852c191769b4d46bffe32c55cb4f8419684ba8ba62a45c`
- after this change:
  `90cfbd8d066b5d02c231f7630fc108db83ae7df52d6484e693ef21edd4d66107`

This is the first change to `summarize_train_val_runs.py` since the package was
frozen, so the manifest value is also its immediately preceding state.
