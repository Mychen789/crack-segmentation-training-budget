# Decision log: five-epoch linear LR warmup (training-contract deviation)

Date: 2026-09-14
Decision: DEVIATE FROM THE FROZEN CONSTANT SCHEDULE; RE-RUN THE WHOLE MATRIX

## Why the frozen contract had to change

`training_contract_freeze_v1.json` froze `schedule: {name: constant, warmup_epochs: 0}`
on 2026-08-29, before any training had been run. The first authorised matrix
(36 runs, 2026-09-09/10, 13.35 h, all formal-eligible) showed that this contract
cannot produce a readable answer to H1.

Measured on CRACK500 / UNet base32:

- The same architecture and data, differing only in seed, reaches two stable
  outcomes. `btsp` reached best FgIoU 0.5012 (s2026) and 0.5060 (s2028) but only
  0.4408 (s2027).
- The gap is not a speed difference. A 150-epoch probe of `btsp_s2027` stayed
  flat across the whole run (epochs 31-45 mean 0.4288, 51-70 0.4314,
  81-100 0.4330, 131-150 0.4292; best 0.4434, only 0.0026 above its 30-epoch
  value) while its training loss stalled at 0.5204 against 0.1721 for s2026.
- The paired H1 difference has a standard deviation of 0.074 across three seeds
  (+0.0660, -0.0589, +0.0725). Detecting the pre-registered minimum effect of
  0.005 FgIoU at that dispersion would need on the order of 840 paired seeds.
  The protocol plans five.

The experiment as contracted therefore cannot measure the effect it was designed
to measure. More seeds do not help; the training setup had to be fixed first.

## What changed

`schedule` becomes a five-epoch linear LR warmup followed by the frozen constant
rate. Implemented as `--warmup-epochs` in
`baseline_platform/tools/train_pooling_experiment.py`, interpolating per
optimizer step rather than per epoch. The default is 0, so every run recorded
before this date keeps its exact numerical behaviour.

Nothing else moves: optimizer, learning rate, weight decay, epochs, effective
batch size, augmentation, loss and checkpoint selection are untouched.

## Evidence that it works

`probe_warmup_btsp_s2027`, 45 epochs, engineering stage, five warmup epochs,
against the same seed without warmup and against the previously best seed:

| run | epochs 30-45 mean FgIoU | peak | best |
|---|---|---|---|
| btsp s2027 + warmup | 0.5333 | 0.5541 | 0.5658 |
| btsp s2027 no warmup | 0.4286 | 0.4377 | 0.4434 |
| btsp s2026 no warmup | 0.4855 | 0.5026 | 0.5026 |

Warmup did not merely rescue the bad seed: it raised it 0.105 above its own
unwarmed result and 0.048 above the best run previously obtained in this
project. Convergence is also much earlier, reaching 0.5041 at epoch 2 and 0.5654
at epoch 10.

## A second, separate failure mode, not addressed here

The warmup probe collapsed in its final epoch: training loss rose from 0.2964 to
0.4526 and validation FgIoU fell from 0.5423 to 0.3323 at epoch 45, with
`grad_norm_max` reaching 585.7 against a whole-run median of 34.6, a 16.9-fold
spike, and `grad_norm_mean` rising from 1.4 to 3.55.

This is a genuine gradient explosion and is a different fault from the early
basin trapping that warmup fixes: the earlier epoch-6 event in the unwarmed run
carried no gradient anomaly at all (`grad_norm_max` 34.85, below its own epoch-1
value of 167.84). Gradient clipping is therefore now evidence-backed, but it is
not adopted in this deviation because the event occurred at epoch 45, outside
the contracted 30-epoch budget. `--grad-clip` remains available and defaults to
measurement only. If the re-run matrix shows collapses inside 30 epochs, a
separate decision log will adopt clipping.

## Consequences

- All 36 runs of the 2026-09-09/10 matrix are superseded. They are moved to
  `baseline_platform/runs_archive_v1_nowarmup/` and remain the v1 exploratory
  record; their `result.json` files are kept, their checkpoints deleted.
  They are not mixed into the new result index.
- The matrix is re-run in full under the new schedule. No arm is carried over.
- Evidence used: training and validation data only.
- Changes H1, H2, H3, primary metric, stratification thresholds or Go/No-Go: no.
- Authorises test or external-test access: no.

## Audit chain

`baseline_platform/tools/train_pooling_experiment.py`

- after the formal-stage metadata wiring:
  `d887cdcd8e270f9f1a6d96de7ba03ecd19be4f810d64687609bd5944de28a64f`
- after `--grad-clip` and gradient-norm logging:
  `6cfa054eb3358c4f473ce745851afb1291c48c19bbdc75d2343dabc417542f28`
- after `--warmup-epochs` (this change):
  `6e54d4bcf6b22e23e4aa3bb2d6e78c03276c8bbda9f29b68b50bb45ee6d8e757`
