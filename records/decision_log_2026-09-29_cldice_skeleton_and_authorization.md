# Decision log: replace the clDice skeleton, and re-sign the execution authorization

Date: 2026-09-29
Decision: MAKE ZHANG-SUEN THINNING THE clDice SKELETON AND RECOMPUTE; RE-SIGN THE
EXECUTION AUTHORIZATION AGAINST THE TRAINER HASH THAT MOVED ON 2026-09-25
Stage: validation-only. No test or external-test sample was opened or requested.

## 1. The authorization binding is restored

`tools/train_pooling_experiment.py` was edited on 2026-09-25 to thread a
normalisation switch into the second backbone, which moved its digest from
`a6ac343b` to `4c09615b` and lapsed the binding in
`train_val_execution_authorization_v1.json`. **No formal training could run.**

The project owner re-signed on 2026-09-29. The record's own digest moves from
`15bfe951` to `49df0a8a`, and `configs/formal_train_val_plan_v1.json` was updated
to pin it, giving the chain `b27a390e` to `26b967fc` to `15bfe951` to `49df0a8a`.
Readiness re-checks at 22 of 22 with no blockers.

Scope is unchanged: train and val only, test and external_test stay locked, and
the record carries no token that could open them. This unblocks G2, the second
backbone.

## 2. The clDice skeleton was not measuring connectivity

`_morphological_skeleton` is a union of erosion residuals. That is a correct
morphological construction and it is **not** connectivity-preserving, which is
the one property clDice is named for.

Measured on real validation annotations, component count of the mask against
component count of its skeleton:

| dataset | mask components | morphological skeleton | Zhang-Suen |
|---|---|---|---|
| CRACK500, six images | 2 to 27 | 43 to 439, median 325 | **identical to the mask, every image** |
| DeepCrack537, six images | 1 to 4 | 81 to 277, median 141 | **identical to the mask, every image** |

A mask of 27 connected components became a skeleton of 354. **clDice computed on
that skeleton was measuring something, but not connectivity.**

## 3. What replaces it

`_zhang_suen_thinning`, the 1984 two-subiteration algorithm, which deletes a
pixel only when its removal cannot disconnect the local neighbourhood. Verified:

- component count reproduces the mask's on all twelve real images tested and on
  three synthetic shapes including a ring and a two-component figure;
- idempotent, a second pass changes nothing;
- an already one-pixel-wide line is returned unchanged;
- skeleton density falls from a median 13.9% to 10.7% of mask area on CRACK500
  and 22.1% to 16.4% on DeepCrack537, closer to what a centreline should be.

`SKELETON_CHOICES = ("thinning", "morphological")`. The default is `thinning`
from this date; `morphological` is kept so pre-2026-09-29 numbers can be
regenerated. `GroupedExtendedSegmentationMetrics`, `cldice_counts_from_masks`
and `cldice` all take the choice and validate it.

## 4. What changes

clDice moves for every run. Spot checks on `btsp` seed 2026:

| dataset | morphological | thinning |
|---|---|---|
| CRACK500 | 0.8332 | 0.8437 |
| DeepCrack537 | 0.8472 | 0.8793 |

All 84 runs are recomputed into
`artifacts/recomputed_validation_metrics_v2.json`; the v1 file stays as the
record of the previous skeleton. Dataset means over all 42 runs each:

| dataset | morphological | thinning | change |
|---|---|---|---|
| CRACK500 | 0.8251 | 0.8362 | +0.0111 |
| DeepCrack537 | 0.8468 | **0.8773** | **+0.0305** |

DeepCrack537 moves more, consistent with its thinner annotations giving the
skeleton a larger share of the mask.

**Nothing else moved.** Foreground IoU, dice, boundary F1 and AUPRC are
bit-identical between v1 and v2, maximum difference 0.00e+00 on all four, which
confirms the skeleton feeds clDice and nothing else.

The operator ordering does not change character. Two-dataset macro clDice over
the eleven distinct operators spans 0.0193 against a within-arm seed standard
deviation of 0.0054, a ratio of 3.56, below the 5.19 that foreground IoU gives.
`btsp` sits sixth of eleven. **Fixing the skeleton did not make the operators
separate**, which is C3 again on a different metric.

**The restriction recorded on 2026-09-23 is lifted.** clDice may now be cited as
a connectivity measure. The cross-dataset restriction is **not** lifted: skeleton
density still varies with annotation stroke width, so clDice remains
non-comparable across the two datasets, and C8's second source stands.

## 5. What this does not do

- It does not make clDice a topology-preserving guarantee in the sense of
  homotopy equivalence. Zhang-Suen preserves connectivity empirically on the
  shapes tested; it is not proven here to preserve every topological invariant.
- It does not change the soft-clDice **loss** in `losses.py`, which uses a
  differentiable soft skeleton for training and is a separate construction.
- It does not re-open H2. The stratified primary condition is still unsupportable
  on the frozen split, see S8.

## 6. Files changed

| file | in the v0.4 freeze | changed |
|---|---|---|
| `src/surface_crack/extended_metrics.py` | yes | yes |
| `tests/test_extended_metrics.py` | no | yes, six tests added |
| `artifacts/train_val_execution_authorization_v1.json` | no | re-signed |
| `configs/formal_train_val_plan_v1.json` | no | authorization pointer |

Test suite: 129 before, **135** after. The v0.4 freeze record's bound-source list
needs the new `extended_metrics.py` digest; that is amended alongside this record.

## 7. Scope

- Evidence used: training and validation data only.
- Changes H1, H2, H3, the primary metric, thresholds or Go/No-Go: no.
- Fair re-run of affected arms required: no. Recomputation is a measurement pass.
- Authorises formal train/validation execution: the re-signing in section 1 does,
  for train and val only.
- Authorises test or external-test access: no.

## 8. Reproduction

    cd baseline_platform
    ../.venv/Scripts/python.exe -m pytest tests/test_extended_metrics.py -q
    ../.venv/Scripts/python.exe tools/reevaluate_checkpoints.py \
        --pattern "formal_gn_*" \
        --output-json artifacts/recomputed_validation_metrics_v2.json

Passing `skeleton="morphological"` to any of the three entry points reproduces
the previous numbers.
