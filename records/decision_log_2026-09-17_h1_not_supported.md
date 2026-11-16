# Decision log: H1 is not supported under stabilised training

Date: 2026-09-17
Decision: RECORD H1 AS NOT SUPPORTED ON CRACK500; REFRAME THE PAPER AROUND THE
METHODOLOGICAL FINDING (main line B of `P2_STORY_REFRAME_AND_EVIDENCE_2026-09-15.md`)
Stage: validation-only. No test or external-test sample was opened or requested.

## The v3 matrix

Third authorised matrix, `--norm group --warmup-epochs 5`, run names
`formal_gn_crack500_unet_<op>_s<seed>`, 2026-09-16 09:41 to 2026-09-17 05:40
(19.98 h, 35 complete, 1 killed by the environment at epoch 23 and refilled on
2026-09-17, exit 0). `summarize_train_val_runs.py` reports `formal=36`.

The stabilisation holds at full scale:

| | v1 (bare) | v2 (warmup) | v3 (warmup + GroupNorm) |
|---|---|---|---|
| runs reaching 30 epochs | 36 | 36 | 36 |
| diverged (peak − last-3 mean > 0.05) | 6 | 22 | **0** |
| largest such drop | — | — | 0.0265 |
| `grad_norm_max`, median / max | 184.6 / 638507.3 | 141.9 / 272385.9 | **34.2 / 54.7** |
| non-finite gradient steps | 0 | 0 | 0 |

The operator table collapses as training becomes healthy. Trailing MA(5) means
over arms with all seeds usable, lowest arm excluded to remove the `stride_conv`
and `avg` outliers that dominate the v1/v2 spans:

| | span across operators | `btsp` rank |
|---|---|---|
| v1 | 0.0465 | 2 / 12 |
| v2 | 0.0137 | 4 / 9 |
| v3 | **0.0078** | **10 / 12** |

Including every arm the spans are 0.1222 / 0.1057 / 0.0082; the v2 figure is held
open by `stride_conv` alone, so both conventions are stated rather than one
chosen silently.

## H1 across the three matrices

`btsp` minus `upper_tail_matched`, the parameter-matched control
(7765989 parameters each under `--norm group`), trailing MA(5), per seed:

| matrix | per-seed Δ | mean | sd | same sign |
|---|---|---|---|---|
| v1 | +0.0630 / −0.0659 / +0.0698 | +0.0223 | 0.0764 | 2/3 |
| v2 | +0.0054 / +0.0181 / +0.0238 | +0.0158 | 0.0094 | **3/3** |
| v3 | −0.0023 / +0.0001 / −0.0059 | **−0.0027** | 0.0030 | 2/3 |

## Why v2 looked positive

This is the part worth preserving. Under v2 the contrast had every surface
feature the pre-registered protocol asks for: all three seeds same sign, mean
four times the 0.005 threshold on the non-diverged subset (+0.0202), and a
between-seed sd an order of magnitude tighter than v1.

The cause is visible in the divergence column:

| seed | `btsp` drop | `upper_tail_matched` drop |
|---|---|---|
| 2026 | 0.0098 | **0.2348** |
| 2027 | 0.0139 | **0.0773** |
| 2028 | 0.0161 | **0.0632** |

In v2 all three `btsp` seeds survived intact while all three control seeds were
damaged by gradient explosion. The contrast measured which arm resisted collapse,
not which pooling operator represents cracks better. Removing the control seed
that diverged worst leaves 3 `btsp` seeds against 2 controls and a group-mean gap
of +0.0202; restricting to the two seeds present in both arms gives +0.0210, so
the imbalance is not the mechanism — the damage itself is.

Once GroupNorm removes the explosions the advantage has nothing to act on and the
sign reverses. Had the study stopped at v2, a false positive would have been
reported, and every pre-registered acceptance criterion would have been met.

## Selection bias does not shrink when training becomes healthy

Selection optimism was re-estimated with the odd/even split of
`decision_log_2026-09-15_ma5_checkpoint_selection.md`, verified against it first:
restricting v1+v2 to runs with drop ≤ 0.09 reproduces that log's 55 runs and its
raw-maximum optimism of +0.0058 exactly; at drop ≤ 0.08 (53 runs) the MA(5)
figure of +0.0015 also reproduces. The implementation is therefore comparable
across matrices.

Applying it per matrix, at a common drop ≤ 0.05 (v3 has no diverged run, so its
value is independent of any threshold):

| matrix | n | raw-max optimism (median) | MA(5) optimism (median) |
|---|---|---|---|
| v1 | 30 | 0.0093 | 0.0017 |
| v2 | 14 | 0.0055 | 0.0019 |
| **v3** | 36 | 0.0098 | **0.0043** |

The correction degrades exactly where it was supposed to help. Across the twelve
v3 arms the MA(5) optimism averages **0.0043, or 0.86 of the 0.005 threshold**,
ranging from 0.0010 (`min`) to 0.0080 (`btsp`) — an eightfold spread between arms
of the same experiment.

The mechanism is straightforward once stated: while a curve is still climbing,
both halves of the split peak at the end and selection has little freedom. Once
training converges and the curve flattens, the location of the maximum is decided
by noise alone, and selection optimism is at its largest. The problem is worst in
precisely the regime everyone reports from.

Two further points bear on H1 specifically. First, the odd/even estimate is a
lower bound: each half offers half the epochs to choose from, so the full-length
selection has more room than the estimate measures. Second, the paired design
cancels most but not all of it — the mean gap between the two arms' optimism is
+0.0017 against a paired difference of −0.0027 (ratio 0.63), but that gap has an
sd of 0.0051 across seeds, larger than its own mean, reaching +0.0076 on s2027.
Per seed the cancellation is unreliable.

## What this means

H1 is not supported on CRACK500. The measured direction is negative, only 2 of 3
seeds agree, and the effect is smaller than the selection optimism carried by the
primary metric.

Estimating required sample size from the v3 paired sd of 0.0030 (two-sided
α = 0.05, power 0.8): about 3 paired seeds to detect the 0.005 threshold, about
10 to detect the observed −0.0027. The sd is estimated from 3 seeds and is itself
unstable, so these are order-of-magnitude figures. They show that falsification is
within reach at this budget while confirmation is not, because the observed effect
points the wrong way.

## What changes and what does not

- No hypothesis, threshold, metric definition, stratification, Go/No-Go rule,
  data role or test lock is modified by this record.
- No run was retrained. Every figure above is computed from recorded `history`.
- The paper's main line moves from A to B as already set out in
  `P2_STORY_REFRAME_AND_EVIDENCE_2026-09-15.md` §3. That document's §2.2
  ("no real difference between operators" not established) is superseded for the
  stabilised regime: across v3 the between-arm span is 0.0078 while within-arm
  seed sd runs 0.0007–0.0065, so the two are now the same order, whereas under v2
  the span was roughly three times the within-arm noise.
- `upper_tail_matched` and `avg_top2` remain bit-identical across all three seeds
  in v3, in both MA(5) and raw-maximum terms. Eleven distinct operators, not
  twelve, for the third consecutive matrix.
- DeepCrack537 remains untouched. Its value as out-of-sample confirmation of a
  claim formed on CRACK500 is unchanged and should not be spent until the claim
  for main line B is itself frozen.

## Scope

- Evidence used: training and validation data only.
- Changes H1, H2, H3, the primary metric definition, stratification thresholds or
  Go/No-Go: no. This records an outcome against the existing H1, not a change to
  it.
- Fair re-run of all affected arms required: no.
- Authorises formal train/validation execution: no.
- Authorises test or external-test access: no.

## Reproduction

New read-only analysis tools, none of them modifying a frozen file:

- `baseline_platform/tools/analyze_three_round_evidence.py` — three-matrix tables,
  selection bias, training pathology, H1 across matrices.
- `baseline_platform/tools/calibrate_selection_bias.py` — threshold sweep
  reproducing the 55-run subset and the +0.0058 / +0.0015 figures of the MA(5)
  decision log.
- `baseline_platform/tools/h1_bias_decomposition.py` — per-arm optimism and the
  paired-cancellation residual.

Matrices on disk: `runs_archive_v1_nowarmup/`, `runs_archive_v2_nogroupnorm/`,
`runs/formal_gn_*`. The training configuration of each matrix is self-recorded in
its `result.json` (`warmup_epochs`, `norm`, `amp_dtype`, `grad_clip`), so the
three regimes are distinguishable without relying on directory names.

The run killed at epoch 23 on 2026-09-17 is archived unmodified at
`runs/_failed_max_s2028_20260917/`. Its first 23 epochs are bit-identical to the
refill, which is consistent with the determinism established on 2026-09-07 and
confirms the interruption was environmental rather than numerical. Corroborated
on 2026-09-18: the DeepCrack537 matrix on the same machine was killed twice in the
same way, once with the harness reporting explicitly that it stopped the process
because the system was low on memory (free physical memory 3.9 of 31.2 GB,
committed 53.2 of 54.6 GB). Memory pressure is therefore the probable cause of the
2026-09-17 kill as well, though it remains inference rather than a captured error;
the exit code alone does not distinguish it from a CUDA fault or a driver reset.
It carries
`status=RUNNING` and is therefore excluded from the formal table by
`_formal_eligibility`; `formal=36` counts the refill.
