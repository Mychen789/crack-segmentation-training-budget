# Decision log: the H1 control is parameter-matched but not capacity-matched

Date: 2026-09-21
Decision: ADD TWO CAPACITY-MATCHED CONTROLS AS POST-HOC STRENGTHENING ARMS;
LEAVE THE PRE-REGISTERED H1 OUTCOME OF 2026-09-17 EXACTLY AS RECORDED
Stage: validation-only. No test or external-test sample was opened or requested.

> **Correction, 2026-09-23.** Two claims in this record are wrong and one figure
> must not be used as written. See
> `decision_log_2026-09-23_correctness_audit_and_metric_repair.md` section 1.
>
> 1. `upper_tail_capacity_matched` is not merely in the same functional family
>    as `t_max_avg`; the two forward passes are line-for-line identical and
>    produce bit-identical output at equal alpha. They are one operator at two
>    initial values.
> 2. The arm does not start from the same function as static BTSP. At alpha 0.5
>    BTSP is exactly average pooling while this arm is `0.75 r1 + 0.25 r2`. This
>    record matched the initial value of the fusion logit, not the function it
>    produces, so the **−0.0061 contrast below confounds the lower tail with a
>    different starting point and must not be carried into the paper**.
> 3. The primary control for C6 becomes `diag_partition_matched`, the only arm
>    matching BTSP on parameter count, live capacity and initial function. Its
>    result is BTSP **+0.0021** with the interval crossing zero.
>
> The reasoning about the frozen control's dead scalar, the branch identity with
> `avg_top2`, the learned-scalar evidence and the registered prediction procedure
> are unaffected. The error corrected here is of the same class this record was
> written to expose, and was committed while writing it.

## What the frozen control is

Protocol v0.3 fixes the H1 primary control twice, and both statements are
deliberate rather than accidental.

Section 1.4, terminology table:

> 上尾单支 | 仅使用 top-2 mean；等价于本设置的 Avg-TopK（K=2） | H1 必须直接比较

Section 2, H1:

> 主对照：参数匹配上尾单支。该对照保留相同模块和参数登记，但将下尾输入替换为
> 上尾输入，因此输出严格退化为 U₂，不得引入其他信息。

`MatchedUpperTailPool2d` implements that clause literally:

    return alpha * upper + (1.0 - alpha) * upper

The implementation is correct against the protocol. This record does not treat
it as a defect in the code.

## What the frozen control cannot do

The clause holds the parameter COUNT constant and lets the output degenerate to
U₂. Measured consequences, all reproducible on CPU in under a second:

| property | `btsp` | `upper_tail_matched` | `avg_top2` |
|---|---|---|---|
| UNet base32 `--norm group` parameters | 7765989 | 7765989 | 7765985 |
| gradient at any pool site | live | **exactly 0.0** | no parameter |
| output vs `avg_top2` on identical input | differs | **bit-identical** | — |

The registered scalar is never an optimisation degree of freedom. Its gradient
is zero analytically, not merely small, because the derivative of
`a*u + (1-a)*u` with respect to the fusion logit vanishes for every input. So
the arm occupies a row in the parameter table without occupying one in the
search space.

The completed matrix records this directly, so the argument does not rest on the
derivative alone. Every run writes its trained fusion scalars to
`learned_pooling_scalars`. On DeepCrack537, after thirty epochs:

| arm | s2026 | s2027 | s2028 |
|---|---|---|---|
| `btsp` | 0.709 0.643 0.608 0.465 | 0.754 0.688 0.584 0.487 | 0.749 0.670 0.611 0.511 |
| `upper_tail_matched` | 0.5 0.5 0.5 0.5 | 0.5 0.5 0.5 0.5 | 0.5 0.5 0.5 0.5 |
| `t_max_avg` | 0.790 0.749 0.711 0.665 | 0.818 0.734 0.683 0.660 | 0.789 0.732 0.712 0.582 |

The control's four scalars sit at their initial value to the printed precision
in all three seeds, while both live arms move their deepest and shallowest sites
apart by more than 0.2. The parameter was registered, optimised over, and never
moved, because it could not.

That makes the pre-registered H1 estimator vary two things at once:

1. the lower-tail input, which is the quantity H1 exists to test, and
2. the presence of a usable fusion degree of freedom, which H1 condition 6
   explicitly forbids as an explanation of the gain.

The 2026-09-17 record already noted the downstream symptom, that
`upper_tail_matched` and `avg_top2` are bit-identical across all three v3 seeds
and that the matrix therefore holds eleven distinct operators, not twelve. This
record states the cause and closes it. The same identity was re-confirmed on
2026-09-18 for DeepCrack537: all six pairs agree on foreground IoU, dice,
precision, recall, boundary F1, clDice, AUPRC and best epoch.

## The two added controls

Both keep the module shape of static BTSP, one sigmoid scalar per downsampling
site, the fusion logit initialised so that alpha is 0.5, and 7765989 total
parameters under UNet base32 with `--norm group`, an exact match to `btsp`.

**`upper_tail_capacity_matched`** mixes the rank-1 maximum with the top-2 mean
of the same 2x2 window. Both branches are upper order statistics, so no
lower-tail value can enter and the protocol requirement that no other
information be introduced is preserved in substance. The two branches differ, so
the fusion gradient is generically non-zero. This is the arm that isolates the
lower tail alone, and it is the one this record proposes as the primary
comparator for any future restatement of H1.

Its functional family coincides with `t_max_avg`, which is already in the
matrix, but the two are not interchangeable. `t_max_avg` initialises its mixing
weight at 0.7 and carries a literature-baseline role, while this arm initialises
at 0.5 to match BTSP exactly. Running both also measures initialisation
sensitivity within one functional family, which the matrix currently cannot.

**`diag_partition_matched`** mixes the two diagonal means of the same window. It
splits that window into two complementary two-element means by fixed POSITION
instead of by VALUE rank. At alpha 0.5 it equals average pooling exactly, which
is also where static BTSP starts, so the two arms share module shape, parameter
count, capacity and initial function, and differ only in whether the two-plus-two
partition is ordered. It is not upper-tail only and therefore cannot serve as the
H1 primary control; it is a structural control for main line B. If `btsp` and
this arm agree, then value ordering buys nothing beyond mixing two complementary
half-window means.

## Why this is not moving the goalposts

Redefining a control after a hypothesis has failed is a methodological hazard
and is treated as one here. Three facts bound it.

First, the direction. The replacement control is strictly harder to beat than
the frozen one, because it has a live fusion scalar the frozen one lacks. The
change makes H1 harder for the proposed method, not easier. Goalpost-moving runs
the other way.

Second, the expected outcome is registered here, before the arms are run, in
keeping with `S1_PREREGISTERED_PREDICTIONS_2026-09-18.md`. Against the frozen
control the v3 paired mean is -0.0027. Against `t_max_avg`, which shares the new
control's functional family, the two-dataset macro gap is -0.0067. The
prediction is therefore that `btsp` minus `upper_tail_capacity_matched` is
negative on both core datasets, with a macro magnitude between 0.003 and 0.010.
A positive result would contradict this registration and must be reported as
such rather than quietly adopted.

Third, the status of the original result. The pre-registered H1 outcome stands
as recorded on 2026-09-17: not supported. Nothing here reopens it, and neither
new arm may be used to restate H1 as supported. Both are exploratory arms
serving main line B of `P2_STORY_REFRAME_AND_EVIDENCE_2026-09-15.md`, whose
subject is how a pre-registered contrast can look clean and still be wrong. A
control that matches a parameter count while contributing no capacity is a
second instance of exactly that subject, alongside the v2 divergence artefact.

## Results

Executed 2026-09-21 17:51 to 2026-09-22 00:19. Twelve runs, two arms times three
seeds times two core datasets, all reaching 30 of 30 epochs with zero non-finite
gradient steps and `formal_eligible=true`. DeepCrack537 took 0.50 h. CRACK500 was
interrupted once at 2 of 6 by a host memory reaper while the session was idle,
not by any training fault; the partial run carried `status=RUNNING` and was
therefore excluded from the formal table by `_formal_eligibility` and re-run from
the start. The remaining four completed under the project watchdog in 2.23 h with
one restart, the initial launch. The formal table now holds 84 rows, fourteen
operators times three seeds times two datasets, and no test or external-test
sample was opened.

BTSP minus control, two-dataset equal-weight macro:

| control | CRACK500 | DeepCrack537 | macro | bootstrap 95% CI |
|---|---|---|---|---|
| `upper_tail_matched`, frozen | -0.0027 | -0.0027 | -0.0027 | [-0.0063, +0.0013] |
| `upper_tail_capacity_matched` | -0.0018 | -0.0105 | **-0.0061** | **[-0.0110, -0.0014]** |
| `diag_partition_matched` | +0.0004 | +0.0038 | +0.0021 | [-0.0009, +0.0058] |
| `t_max_avg`, reference | -0.0033 | -0.0102 | -0.0068 | [-0.0107, -0.0019] |

Trailing MA(5), the frozen selection rule. Under the raw maximum the same
ordering holds: -0.0023 for the frozen control with a CI crossing zero, -0.0065
for the capacity-matched control with a CI of [-0.0120, -0.0016]. The conclusion
does not depend on the selection metric.

Three things follow.

First, the frozen control was masking about half of the deficit. Replacing a dead
scalar with a live one moves the estimate from -0.0027 to -0.0061 and moves the
interval off zero. The pre-registered contrast was not merely noisy, it was
biased toward the proposed method, because the control was denied a degree of
freedom that the proposed method had.

Second, the capacity-matched control and `t_max_avg` agree closely, -0.0061
against -0.0068, as they should: they share a functional family and differ only
in initialisation. The agreement is a consistency check on the new arm rather
than a second independent result, and it should be reported that way.

Third, the direction of the two new arms differs, and the pair is more
informative than either alone. BTSP beats the position-partition control by
+0.0021, so value ordering does buy something over an arbitrary two-plus-two
split of the same window, though the MA(5) interval crosses zero. BTSP loses to
the upper-tail control by -0.0061 with an interval clear of zero. Ordering helps;
spending the ordered capacity on the lower tail does not. The lower tail is not
merely inert here, it is a worse use of the same parameter than the upper tail.

One asymmetry must be stated rather than averaged away. The macro is carried by
DeepCrack537. On CRACK500 the capacity-matched contrast is -0.0018 with two of
three seeds positive, which on its own supports nothing; on DeepCrack537 it is
-0.0105 with zero of three positive. The two core datasets agree in sign at the
macro level and disagree in strength by a factor of six. Any statement built on
the macro alone would overstate what CRACK500 shows.

### Hash chain after execution

A signature binds what existed when it was made, so three recorded digests no
longer match their files and each difference is accounted for here rather than
left for a later reader to discover.

| object | recorded | recorded where | current | why it moved |
|---|---|---|---|---|
| authorization | `15bfe951` | the twelve new `result.json` | `15bfe951` | unchanged, chain intact |
| this decision log | `36eddce2` | authorization, freeze v2 | moves on every append | Results appended after signing |
| freeze v2 | `f5b930f8` | authorization | `f712c741` | binding status set to satisfied after signing |

This document's own digest is not pinned in the table, because stating it inside
the document changes it. It is `36eddce2` as signed, and whatever the file
currently hashes to thereafter.

The authorization file was deliberately not rewritten after the runs completed.
Its own digest is recorded inside all twelve new results, so editing it would
break the link those runs assert. The two moved digests both belong to documents
that were appended to after the signature, which is the expected direction: the
signature covers the rationale as it stood when the owner signed, and the
outcome was added below it. Anyone verifying the chain should check the
authorization against the runs, and read the two appended documents as
supersets of what was signed rather than as replacements.

### Prediction check

The prediction filed above, before these arms were run, was that `btsp` minus
`upper_tail_capacity_matched` would be negative on both core datasets with a
macro magnitude between 0.003 and 0.010. Observed: negative on both, macro
magnitude 0.0061. The registration is satisfied on both clauses. This is recorded
as a successful pre-registration, not as support for H1. `tools/analyze_h1_capacity_controls.py`
re-checks the registration on every invocation and prints a departure verdict if a
later re-run moves the estimate out of the band.

## What changes and what does not

- No hypothesis, threshold, metric definition, stratification, Go/No-Go rule,
  data role or test lock is modified. H1 keeps its frozen wording and its
  recorded outcome.
- No existing operator changed behaviour. `research_pooling.py` was edited by
  addition only. `tests/test_research_pooling.py` gains a regression guard that
  pins the outputs of `btsp`, `upper_tail_matched`, `avg_top2`, `t_max_avg`,
  `max`, `avg` and `min` on a fixed input, and a test that pins the zero
  gradient of `upper_tail_matched` so the defect cannot be fixed silently.
- No run was retrained and no recorded result was recomputed. All 72 runs of the
  v3 matrix remain valid and interpretable under their original names; the twelve
  runs of the new arms were added beside them, taking the formal table from 72
  rows to 84.
- Test suite: 97 tests at implementation freeze v1, 103 now, all passing. The
  six added tests are listed under Reproduction.

## Files changed and the sign-off this needs

Two files under `implementation_freeze_v1.json` were edited, so per its
change rule a new freeze accompanies this record as
`artifacts/implementation_freeze_v2.json`.

| file | freeze v1 | now |
|---|---|---|
| `src/surface_crack/research_pooling.py` | `cd66e872` | `2830bd30` |
| `tools/train_pooling_experiment.py` | `d8e13e78` | `a6ac343b` |

`train_pooling_experiment.py` is also a bound artifact of
`train_val_execution_authorization_v1.json`, where it was bound as `32b840c5`.
Its hash moved, so the binding lapsed and the two new arms could not be trained
until it was restored. The project owner re-signed the authorization on
2026-09-21 in the form used for the 2026-09-18 DeepCrack537 amendment, and the
arms were run afterwards. The edit itself is two names added to the
`POOLING_CHOICES` tuple; no training logic, contract field, schedule, metric or
split behaviour was touched, and the twelve existing choices keep their positions
and spellings. Scope was unchanged by the re-signing: train and val only, with
test and external_test still locked and no token in the record that could open
them. The authorization hash chain is `b27a390e` to `26b967fc` to `15bfe951`, and
`configs/formal_train_val_plan_v1.json` was updated to pin the new value.

Two hashes in `implementation_freeze_v1.json` were already stale before this
record and are not affected by it: `models.py`, superseded by
`decision_log_2026-09-15_groupnorm_stabilisation.md`, and
`summarize_train_val_runs.py`, re-bound on 2026-09-09. Both match the
authorization record and neither was edited here.

Re-signing will not disturb the completed matrix, and this was checked rather
than assumed. The runner compares only `epochs_target`, `base_channels`,
`learning_rate`, `weight_decay`, `warmup_epochs`, `norm` and `backbone` when
deciding whether a run is superseded or already complete, so no training setting
moves and all seventy-two runs stay skippable. `summarize_train_val_runs.py`
checks only that `train_val_execution_authorization_sha256` is a well-formed
digest, not that it equals the current file, so eligibility is likewise
unaffected.

One consequence should be recorded rather than discovered later. The formal
table will then span three authorization hashes: the thirty-six CRACK500 runs
carry `b27a390e`, the thirty-six DeepCrack537 runs carry `26b967fc` after the
2026-09-18 amendment, and the six new arms will carry whatever the re-signed
record hashes to. The summary mixes them silently. The precedent is the
2026-09-18 amendment, which accepted a two-hash table on the stated ground that
the five artifacts bound earlier were re-hashed and found byte-identical, so no
earlier result was affected. The same ground holds here for every file except
the trainer, and the trainer edit adds two enum members without touching
training logic. Any table that reports the new arms beside the old ones should
state the three hashes rather than present one authorization.

## Scope

- Evidence used: training and validation data only.
- Changes H1, H2, H3, the primary metric definition, stratification thresholds or
  Go/No-Go: no.
- Fair re-run of all affected arms required: no. The new arms are additive, and
  the matrix runner skips the twelve completed operators.
- Authorises formal train/validation execution: no. See the sign-off paragraph.
- Authorises test or external-test access: no.

## Reproduction

    cd baseline_platform
    ../.venv/Scripts/python.exe -m pytest tests/ -q

Added to `tests/test_research_pooling.py`:

- `test_matched_upper_scalar_is_dead_by_construction`
- `test_capacity_matched_upper_keeps_the_scalar_alive`
- `test_capacity_matched_upper_cannot_see_the_lower_tail`
- `test_diagonal_partition_starts_at_exact_mean_like_btsp`
- `test_diagonal_partition_reads_position_where_btsp_reads_rank`
- `test_legacy_operator_outputs_are_unchanged`

The lower-tail closure test is the load-bearing one. It depresses the bottom two
values of every window, leaves the top two untouched, and asserts that
`upper_tail_capacity_matched` returns a bit-identical output while
`BilateralTailStatisticPool2d` does not.

Planned matrix once the authorization is re-signed: `.vscode/run_training.py`
carries both new names, so a run adds six trainings per dataset and skips the
thirty-six already on disk. Budget at v3 rates is about 2.0 h for CRACK500 and
about 0.4 h for DeepCrack537.
