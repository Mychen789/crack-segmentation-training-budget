# Decision log: wire normalisation into the second backbone, and confirm its parameter count

Date: 2026-09-25
Decision: THREAD `norm` THROUGH `PoolAdapterResNet18UNet` SO THE TWO BACKBONES CAN
BE COMPARED; CORRECT THE FROZEN ANALYTIC PARAMETER COUNT AGAINST THE RUNTIME
Stage: validation-only. No test or external-test sample was opened or requested.

This closes the code-side blocker on gap G2, the single-architecture gap, which
the novelty assessment identified as what stands between the branch-zero
mechanism finding and a method-facing contribution.

## 1. The guard was hiding the incomparability rather than preventing it

`tools/train_pooling_experiment.py` rejected any `--norm` other than `none` for
this backbone, with the note "宁可当场拒绝，也不要静默给出一个与主骨干不同归一化
设置的模型，那会让两个骨干的结果不可比".

The intent was right and the effect was not. The second backbone hard-codes
`nn.BatchNorm2d` at all 29 normalisation sites, and the effective batch size is
1, so:

- during training it behaves as InstanceNorm, statistics coming from one image;
- at evaluation it uses running statistics estimated from single images;
- the main backbone has run GroupNorm since 2026-09-15.

The two were therefore already incomparable in both phases. The guard blocked
the one flag that could have fixed it.

## 2. What changed

`src/surface_crack/resnet_unet.py` now takes `norm`, threaded through
`ResidualBasicBlock`, `DecoderBlock` and the stem, with
`NORM_CHOICES = ("batch", "group", "none")`. GroupNorm is constructed by the
same `_make_norm` helper the main backbone uses, so the two share one
construction.

| setting | meaning |
|---|---|
| `batch` | the frozen configuration, default, numerics unchanged |
| `group` | what makes this backbone comparable with the main one |
| `none` | no normalisation at all; not a sensible residual baseline |

Measured: `batch` gives 29 BatchNorm and 0 GroupNorm layers, `group` the
reverse, and the two have **identical parameter counts**, so a cross-backbone
comparison does not confound normalisation with capacity.

The trainer now refuses each backbone the other's setting rather than accepting
it silently: `--norm batch` is rejected for the main backbone, because batch size
1 leaves BatchNorm undefined at the 1x1 deepest feature map, which is why that
backbone was frozen without normalisation in the first place; `--norm none` is
rejected for the second, because that is not its frozen configuration and an
unnormalised residual network is not a baseline anyone intended.

## 3. The frozen analytic parameter count was low by 22 percent

`artifacts/pretrain_method_freeze_v1.json` carries
`analytic_trainable_parameter_count_for_rgb_base32 = 11067041` with status
`ANALYTIC_EXPECTATION_PENDING_TORCH_RUNTIME_CONFIRMATION`, and
`torch_runtime_tests: PENDING_DEPENDENCY_NOT_INSTALLED`. Torch is installed now,
so the expectation can be checked. It is wrong.

| component | parameters |
|---|---|
| stem | 928 |
| encoder stages | 11,150,720 |
| downsamplers | 0 |
| transposed-convolution upsamplers | 696,800 |
| decoder stages | 2,352,000 |
| classifier | 33 |
| **total** | **14,200,481** |

The gap against the analytic expectation is 3,133,440, and the decoder totals
3,048,800, which is 97 percent of it. The most likely cause is that the analytic
count omitted most of the decoder. **This closes an open item that the freeze
record itself flagged; it is not a newly discovered hidden error.** No result is
affected, because this backbone has never been trained.

The corrected figure is pinned by
`tests/test_second_backbone_tail_statistics.py::test_runtime_parameter_count_corrects_the_frozen_analytic_expectation`.

## 4. What this does not do

- **It does not run the second backbone.** G2 stays open until a matrix runs on
  it. This record only removes the code-side blocker.
- It does not change the main backbone's numerics. `--norm none` and
  `--norm group` behave exactly as before for it.
- It does not settle which setting a cross-backbone comparison should use.
  `group` on both is the only setting that matches, so it is the obvious choice,
  but the frozen `batch` configuration remains available and any comparison must
  state which was used.
- BatchNorm at batch size 1 remains a defect of the frozen configuration rather
  than something this record repairs. It is now avoidable, not removed.

## 5. What changes and what does not

- No hypothesis, threshold, primary metric, stratification, Go/No-Go rule, data
  role or test lock is modified.
- No existing run is affected. The second backbone has never been trained, and
  the main backbone's behaviour is unchanged at its two existing settings.
- Test suite: 121 before, **129** after. Six tests were added for the
  normalisation switch and the corrected count, and
  `test_both_frozen_backbone_families_build_and_forward` now builds each family
  at its own frozen setting instead of relying on one shared default.

## 6. Files changed and the binding this affects

| file | in freeze v1 | in the v0.4 signature | changed here |
|---|---|---|---|
| `src/surface_crack/resnet_unet.py` | yes | no | yes |
| `tools/train_pooling_experiment.py` | yes | yes | yes |

`preregistered_protocol_v0.4_freeze.json` was signed on 2026-09-25 against an
earlier hash of the trainer, so its bound-source list is amended in place with a
note naming this record. The protocol text itself is untouched, and no run has
recorded the freeze record's own digest, so nothing downstream breaks.

`train_val_execution_authorization_v1.json` binds the trainer too. **Its binding
now lapses again and must be re-signed before either backbone is trained**, in
the same form as the 2026-09-18 and 2026-09-21 amendments. The edit adds one
enum member and two guards; no training logic, contract field, schedule, metric
or split behaviour was touched.

## 7. Scope

- Evidence used: training and validation data only; §3 needs no data at all.
- Changes H1, H2, H3, the primary metric, thresholds or Go/No-Go: no.
- Fair re-run of affected arms required: no.
- Authorises formal train/validation execution: no.
- Authorises test or external-test access: no.

## 8. Reproduction

    cd baseline_platform
    ../.venv/Scripts/python.exe -m pytest tests/test_second_backbone_tail_statistics.py tests/test_train_pooling_entry.py -q
