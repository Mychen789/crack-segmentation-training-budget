# Decision log: correctness audit, boundary metric repair, and the lower-tail deletion test

Date: 2026-09-23
Decision: REPAIR THE BOUNDARY TOLERANCE TO THE PRE-REGISTERED RULE AND RECOMPUTE
THE SECONDARY METRICS FROM CHECKPOINTS; CORRECT TWO CLAIMS IN THE 2026-09-21
RECORD; PATCH FOUR LATENT TRAINER DEFECTS
Stage: validation-only. No test or external-test sample was opened or requested.

Three independent audits covered the pooling and training path, the metric and
data path, and the existing test and audit coverage. Every claim below was
re-verified directly before being written down. The primary metric is unaffected
and no run was retrained.

## 1. Two corrections to the 2026-09-21 record

### 1.1 The capacity-matched control is the same function as `t_max_avg`

`CapacityMatchedUpperTailPool2d.forward` and `TMaxAvgPool2d.forward` are
line-for-line identical. Measured at alpha 0.3, 0.5, 0.7 and 0.9, their outputs
are bit-identical, maximum difference 0.0. Expanded over the window order
statistics both equal

    (1 + a) / 2 * r1 + (1 - a) / 2 * r2

so they are one single-parameter family at two initial values, 0.5 and 0.7. The
2026-09-21 record said "its functional family coincides with `t_max_avg` ... the
two are not interchangeable", which understates it. They are the same operator.

### 1.2 The control does not start from the same function as BTSP

This is the more serious of the two.

| arm | function at alpha = 0.5 | max difference from average pooling |
|---|---|---|
| `btsp` | exactly average pooling | 0.0000 |
| `upper_tail_capacity_matched` | 0.75 r1 + 0.25 r2 | 2.1265 |

The 2026-09-21 design matched the initial value of the fusion logit, θ =
logit(0.5), across the two arms. It did not match the function those parameters
produce. BTSP starts at average pooling; the control starts at a strongly
max-weighted statistic. The measured contrast therefore mixes "the lower tail"
with "a different starting point", and the −0.0061 figure cannot be carried into
the paper as written.

This is the same class of error that C6 was written to expose: a quantity was
matched because it was easy to count, while the quantity that governs the
comparison was left free. It was committed while writing C6. That is recorded
here rather than quietly repaired.

### 1.3 What replaces it

`diag_partition_matched` is the only control that matches static BTSP on all
three of parameter count, live fusion capacity, and initial function: at alpha =
0.5 it is also exactly average pooling. It should become the primary control for
C6. Its measured contrast is BTSP **+0.0021** with the MA(5) interval crossing
zero, that is, **under a fully matched control BTSP shows no advantage**. That
is a cleaner statement than the upper-tail one and does not depend on the
initialisation confound.

### 1.4 A third instance of the same defect, which strengthens C6

`MatchedMaxMeanPool2d` returns `w0 * max + (w1 + w2) * avg` under a softmax, so
it is identically `w0 * max + (1 - w0) * avg`, that is `LearnableMaxAvgPool2d`.
Measured maximum difference at equal alpha is 1.2e-07, and the gradients of
`logits[1]` and `logits[2]` are exactly equal, confirming the zero-gradient
direction. It registers three parameters and has one effective degree of
freedom. Its initial value is 1/3 while `learned_max_avg` starts at 0.5.

| instance | registered | effective | origin |
|---|---|---|---|
| `upper_tail_matched` | 1 | 0 | literal implementation of a protocol clause |
| `v0_max_mean_matched` | 3 | 1 | implementation choice |
| `upper_tail_capacity_matched` | 1 | 1, but initial function unmatched | the 2026-09-21 record |

Gap G6 of the C6 document was "only one instance, which cannot establish a
pattern". There are three, all inside one project that pre-registers its
comparisons and was actively hunting this class of error, and one of them was
committed in the act of documenting it. G6 narrows substantially on this
evidence alone.

## 2. The lower-tail deletion test, which protocol H1 condition 5 requires

Condition 5 of H1 reads: performance must repeatably fall when the lower tail is
deleted or duplicated, and the learned value of alpha alone does not establish a
contribution. **That test had never been run.** It is run here by forcing alpha
to 0.99 at every downsampling site of a trained model, which removes the lower
branch at inference time, and re-measuring validation foreground IoU.

| dataset | s2026 | s2027 | s2028 | mean |
|---|---|---|---|---|
| CRACK500 | −0.0107 | −0.0150 | **+0.0017** | −0.0080 |
| DeepCrack537 | −0.0001 | **+0.0039** | −0.0015 | **+0.0008** |

Two-dataset macro mean −0.0036, below the 0.005 threshold. Four of six runs fall
and two rise, and the two datasets disagree in sign at the dataset-mean level:
on DeepCrack537 removing the lower tail is on average slightly beneficial.

**Condition 5 is not met.** The fall is not repeatable.

One limitation must travel with this result. Removing a branch at inference time
from a network trained with it is not the same as training without it, because
the network may have adapted around the branch. The training-time equivalent
already exists in the matrix: `avg_top2` is BTSP without the lower branch,
trained from scratch, and it scores 0.6317 against BTSP's 0.6290. Both the
inference-time ablation and the training-time comparison point the same way.

## 3. Why the lower branch carries so little: it is usually exactly zero

> **Correction, 2026-09-25.** This section's mechanism holds for the UNet used
> here and does not transfer unchanged to a residual backbone. See
> `S9_LOWER_TAIL_STARVATION_DECOMPOSITION_2026-09-25.md`: initialisation accounts
> for about half the zeros, training raises the deepest site from 49.8% to 90.8%,
> and a residual encoder's post-addition ReLU sees a non-zero-mean identity path
> so its deep sites sit at 24% to 36% instead. The rate is independent of the
> pooling operator, which is confirmed across five arms, but it is **not**
> independent of backbone or of training. That record registers four falsifiable
> predictions for the second backbone before it is run.

Each DoubleConv applies GroupNorm and then ReLU, so the activations entering a
downsampling site are the positive part of a zero-mean tensor, so a large
fraction of them are exactly zero. Measured, that fraction is 59% to 66% across
the four sites on CRACK500 and rises with depth on DeepCrack537, from 49% at the
first site to 89% at the fourth. The bottom-2 mean of a 2x2 window is then
exactly zero whenever at least two of the four values are zero.

Measured on the validation split from the trained checkpoints, fraction of
window positions whose bottom-2 mean is exactly zero:

| site | CRACK500 | DeepCrack537 |
|---|---|---|
| 1 | 62.2% | 52.0% |
| 2 | 73.1% | 70.3% |
| 3 | 68.9% | 83.1% |
| 4 | 65.5% | 90.8% |

At those positions BTSP is `alpha * top2_mean`, a scaled `avg_top2`, and a
constant gain in front of a convolution followed by GroupNorm is largely
absorbed. The lower branch is therefore inactive over most of the feature map,
which is the mechanism behind section 2 rather than a separate finding.

The same table carries a second reading. The fraction of windows holding a tie
runs 63% to 73% on CRACK500 and 52% to 91% on DeepCrack537, almost all of it
from repeated zeros. `torch.topk` gives no guarantee about which of several tied
indices it returns, and the backward pass routes gradients to the indices it
chose, so gradient routing is arbitrary at the majority of positions. Combined
with the absence of `torch.use_deterministic_algorithms` and with
`ConvTranspose2d` backward being non-deterministic on CUDA, bit-identical
reproduction under a fixed seed is not guaranteed by construction. The
reproductions observed on 2026-09-07 and in S6 therefore record what happened,
not what is assured.

## 4. Boundary F1 repaired to the pre-registered rule

### 4.1 What was wrong

The implementation used an absolute 2 px tolerance on a grid resampled to short
side 512. **Neither number appears in the protocol or in any decision log.** The
protocol fixes the matching distance at 0.01 of the image diagonal, which is 7 px
on both core datasets, so the implementation was about five times stricter than
the rule it was supposed to follow.

The consequence was measured on real annotations. Median stroke width is 19.8 px
on CRACK500 and 4.7 px on DeepCrack537, a factor of 4.2. Under controlled
perturbation of ground truth at matched region overlap, boundary F1 came out
0.000 on CRACK500 and 0.984 on DeepCrack537. The metric is close to a step
function near its tolerance: a 1 px error scores about 0.99, 3 px scores about 0.

### 4.2 The repair

`boundary_counts_from_masks` now takes `tolerance_diagonal_fraction`, exclusive
with the absolute `tolerance`, and computes the pixel tolerance per image from
the diagonal. Class and standalone-function defaults, which previously disagreed
with each other, are now both the native grid with fraction 0.01. Recomputed
from checkpoints:

Both columns are means over all 42 runs of that dataset, so they are the same
kind of quantity:

| dataset | 2 px on a 512 grid | 0.01 diagonal on the native grid |
|---|---|---|
| CRACK500 | 0.3573 | **0.8280** |
| DeepCrack537 | 0.8694 | **0.9451** |

The gap between the two datasets closes from 2.43-fold to 1.14-fold.

clDice barely moves, 0.8261 to 0.8251 on CRACK500 and 0.8483 to 0.8468 on
DeepCrack537, as it must: the boundary tolerance is not an input to it. Its own
problem is the skeleton, section 4.4.

### 4.3 What the repair does not fix

A fraction of the diagonal scales with image size, not with annotation stroke
width, so it does not make the two datasets measure the same thing. A test now
pins this limitation rather than a guarantee: a wide bar and a narrow bar
displaced in proportion to their own widths still score more than 0.25 apart.
DeepCrack537 at 0.946 is also approaching saturation. **Cross-dataset boundary
F1 comparison remains invalid**, and the protocol's own rule does not rescue it.
That is itself an instance of main line B: a metric that looks standard, is
implemented faithfully, and is still not comparable.

Two further properties are now documented in the docstring rather than left
implicit: the neighbourhood is a Chebyshev square, so the diagonal tolerance is
sqrt(2) times nominal, and matching is many-to-one dilation rather than Berkeley
one-to-one correspondence, so precision and recall are upper bounds.

### 4.4 clDice is not measuring topology

> **Lifted, 2026-09-29.** The morphological skeleton was replaced by Zhang-Suen
> thinning, whose component count reproduces the mask's exactly on every image
> tested. See `decision_log_2026-09-29_cldice_skeleton_and_authorization.md`.
> **clDice may now be cited as a connectivity measure.** The cross-dataset
> restriction below is *not* lifted: skeleton density still varies with
> annotation stroke width.

The morphological skeleton fragments a single connected crack into as many as
147 components, and skeleton density moves from 4.6% to 63% of mask area with
stroke width. clDice is named for connectivity but is not measuring it here. **No
change is made in this record**; until the skeleton is replaced by a
connectivity-preserving thinning, clDice must be described as a tolerance-style
region similarity and must not be cited as topological evidence.

## 4.5 What the recomputation produced

All 84 runs were re-measured from their checkpoints, 42 per dataset, with every
metric taken from the same weights.

| dataset | fg IoU | dice | boundary F1 | clDice | AUPRC |
|---|---|---|---|---|---|
| CRACK500 | 0.6025 | 0.7459 | **0.8280** | 0.8251 | 0.8262 |
| DeepCrack537 | 0.6582 | 0.7842 | **0.9451** | 0.8468 | 0.8430 |

`upper_tail_matched` and `avg_top2` agree to every printed digit on all five
recomputed metrics, which is the fifth independent confirmation of their
identity.

Two results deserve to be carried forward.

### The checkpoint does not score what the run reports

Recomputed primary metric minus the reported trailing mean, over 84 runs: mean
+0.0002, sd 0.0066, range −0.0169 to +0.0126. The mean is near zero, so the
trailing mean is not biased, but individual runs differ from their own reported
figure by up to 0.017, more than three times the 0.005 acceptance threshold.
Any test-stage inference from `best.pt` will therefore not reproduce the
validation number the paper quotes. This is the practical consequence of the
selection semantics noted in section 5.

### The contrast's significance depends on the reporting rule

BTSP minus control, two-dataset macro, under the two rules:

| control | trailing MA(5) | recomputed checkpoint |
|---|---|---|
| `diag_partition_matched` | +0.0021 [−0.0009, +0.0058] | +0.0035 [−0.0007, +0.0076] |
| `upper_tail_matched` | −0.0027 [−0.0063, +0.0013] | −0.0060 [−0.0125, +0.0007] |
| `upper_tail_capacity_matched` | **−0.0061 [−0.0110, −0.0014]** | −0.0068 [−0.0148, +0.0025] |
| `t_max_avg` | **−0.0068 [−0.0107, −0.0019]** | −0.0003 [−0.0060, +0.0059] |

Two contrasts that exclude zero under the frozen selection rule include it under
the recomputed one. Part of that is expected and must be said plainly: a single
checkpoint carries one epoch of validation noise, whose between-epoch standard
deviation this project measured at about 0.0074, while a five-epoch trailing
mean cuts that by roughly the square root of five. **Wider intervals under the
recomputed rule are therefore an artefact of the rule, not evidence of
fragility.**

What noise does not explain is the movement of a point estimate. The
`t_max_avg` contrast moves from −0.0068 to −0.0003, a shift of 0.0065, larger
than the 0.005 threshold the study was designed around. The other three move by
0.0007 to 0.0033. So one of the four conclusions is genuinely sensitive to a
choice, which checkpoint to evaluate, that was never pre-registered.

`diag_partition_matched`, the control that matches on all three of count,
capacity and initial function, gives the same answer under both rules: BTSP
shows no advantage, interval crossing zero. **It is the only contrast here that
is stable across the reporting rule**, which is a further reason to make it the
primary.

### The ranking is as unstable across reporting rules as it is across datasets

Kendall tau-b between the two rules' operator orderings, over all 14 registered
names and their 90 pairs:

| comparison | tau-b | discordant pairs |
|---|---|---|
| CRACK500 against DeepCrack537, S1 | 0.500 | reported there |
| trailing MA(5) against recomputed checkpoint | **0.600** | 18 of 90 |

The second row is measured on **the same weights and the same data**, with only
the reporting rule differing, and it still agrees no better than the comparison
across two different datasets. The largest movement is five places:
`t_max_avg` sits second under one rule and seventh under the other, and `max`
fourth against ninth. `btsp` moves from tenth to eighth.

An internal check: `avg_top2` and `upper_tail_matched` are the same operator, and
they move together by exactly four places, as they must.

The reshuffling is consistent with noise rather than with a real reordering, and
that is the finding rather than a caveat against it. Measured over all 84 formal
runs on their last ten epochs, medians:

| quantity | value |
|---|---|
| between-epoch standard deviation of a single reading | 0.0083 |
| standard deviation of the trailing MA(5) sequence | 0.0042 |
| gap between adjacent operators, two-dataset macro | **0.0009** |

**The gap between adjacent operators is one fifth of the within-arm noise.** The
smallest adjacent gap is exactly 0.0000, which is `avg_top2` against
`upper_tail_matched` and confirms the identity again; the largest is 0.0108,
which is `min` against the arm above it.

> **Correction, 2026-09-29.** The sentence that stood here, "ordering adjacent
> operators is therefore not a measurement", compared the wrong two quantities
> and is withdrawn. A within-arm standard deviation governs a *marginal* mean; it
> does not govern a *paired* contrast, where the dataset and seed effects cancel
> on subtraction. Measured: `btsp` minus `fixed_max_avg` over the six pairs is
> −0.0075, −0.0039, −0.0058, −0.0023, −0.0052, −0.0044, all six the same sign,
> with a paired standard deviation of **0.0016**, while each arm's own six runs
> have a standard deviation of about **0.027**, seventeen times larger.
>
> Under the paired bootstrap the study resolves far more than the marginal noise
> suggests: **29 of the 55 pairs among the eleven distinct operators have
> intervals clear of zero**, 19 of 45 once `min` is excluded. `btsp` is
> distinguishably worse than `learned_max_avg`, `t_max_avg`, `max` and
> `fixed_max_avg`, and distinguishably better than `min`.
>
> What survives from this section is the narrower and still useful claim: the
> *ranking* is unstable across reporting rules and across datasets, because many
> adjacent pairs are not separated even under pairing. What does not survive is
> the stronger claim that nothing is measurable.

A side observation on the MA(5) rule itself. It reduces the standard deviation by
1.96 times rather than the 2.24 that five independent epochs would give, because
the first-order autocorrelation of validation foreground IoU over the last twenty
epochs is +0.266. **Averaging correlated epochs buys less than the square root of
the window**, which belongs with C5's assessment of that rule.

That is C3's claim, now quantified on a second axis, with the ordering statistic
S1 already uses, and against noise measured from this project's own runs rather
than cited.

This belongs in main line B beside C5. C5 already records that the raw maximum
is sensitive to the epoch budget while the trailing mean is not. This adds a
second axis: which weights you evaluate, at a fixed budget, can move a contrast
by more than the acceptance threshold and can move a rank by five places.

## 5. Four latent trainer defects patched

None of these changed any recorded result; all 84 formal runs were checked.

1. **Non-finite gradients were counted but not skipped.** The fp16 path is
   covered by GradScaler, which skips the step itself, but under bfloat16 the
   scaler is disabled and `scaler.step` degrades to an unconditional
   `optimizer.step()`, so one NaN would enter the weights with nothing but a
   counter to show for it. The step is now skipped and gradients cleared.
   Verified: `nonfinite_grad_steps` is 0 across all 84 runs, so no result is
   affected.
2. **The V0 arms' learned weights were never recorded.** They expose `weights`
   rather than `alpha` or `t`, so `learned_pooling_scalars` returned an empty
   list for `v0_max_mean_min` and `v0_max_mean_matched` in every run to date.
   Both now record 12 values, four sites by three softmax weights.
3. **`--resume` did not validate five settings.** `warmup_epochs`, `grad_clip`,
   `max_train_samples`, `max_val_samples` and `norm` are now checked, and are
   written into the checkpoint so that they can be. Checkpoints written before
   today lack the keys and fall back to their historical defaults, so they stay
   resumable.
4. **`learning_rate` and `weight_decay` could be misreported.** On resume,
   `optimizer.load_state_dict` overwrites both from the checkpoint while
   `result.json` recorded the command-line values. The requested values are kept
   under the original keys, because the matrix runner compares against them, and
   `learning_rate_effective`, `weight_decay_effective` and
   `optimizer_state_restored` now record what the optimizer actually held.

Also declared rather than repaired: `best.pt` holds the weights of the right-hand
end of the trailing-mean window, not the weights that achieved the reported
trailing mean, and the two need not agree. `best_checkpoint` is now null when the
file was never created, which happens whenever a run is shorter than the window,
and `best_checkpoint_selection` names the rule.

## 6. What changes and what does not

- No hypothesis, threshold, primary metric definition, stratification, Go/No-Go
  rule, data role or test lock is modified. H1 stands as not supported.
- The primary metric implementation is unchanged. Its aggregation, threshold,
  native-grid handling and group isolation were audited and are correct, and
  recomputation reproduces `foreground_iou` exactly.
- Secondary metrics change for every run. The 84-row index must be regenerated
  and the four documents that quote boundary F1 revised.
- Test suite: 103 tests before, 108 after, all passing. The boundary contract
  test that pinned the 2 px and 512 defaults is replaced by tests pinning the
  new contract, the image-scale invariance it does buy, and the stroke-width
  non-invariance it does not.

## 7. Scope

- Evidence used: training and validation data only.
- Changes H1, H2, H3, the primary metric, thresholds or Go/No-Go: no.
- Fair re-run of all affected arms required: no. Recomputation is a measurement
  pass over saved checkpoints.
- Authorises formal train/validation execution: no.
- Authorises test or external-test access: no.

## 8. Reproduction

    cd baseline_platform
    ../.venv/Scripts/python.exe -m pytest tests/ -q
    ../.venv/Scripts/python.exe tools/probe_pooling_branches.py \
        --run-name formal_gn_deepcrack537_unet_btsp_s2026 \
        --config configs/deepcrack537_train_val.json
    ../.venv/Scripts/python.exe tools/probe_alpha_sensitivity.py \
        --run-name formal_gn_crack500_unet_btsp_s2026 \
        --config configs/crack500_train_val.json
    ../.venv/Scripts/python.exe tools/reevaluate_checkpoints.py \
        --pattern "formal_gn_*" \
        --output-json artifacts/recomputed_validation_metrics_v1.json

New read-only tools: `tools/probe_pooling_branches.py` for the branch-zero and
tie rates, `tools/probe_alpha_sensitivity.py` for the deletion test of section 2,
`tools/reevaluate_checkpoints.py` for section 4.2.
