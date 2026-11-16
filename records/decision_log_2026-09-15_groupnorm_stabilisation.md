# Decision log: GroupNorm in the UNet encoder/decoder (implementation-freeze deviation)

Date: 2026-09-15
Decision: ADD GROUPNORM BEHIND AN OPT-IN FLAG; ADOPT IT FOR THE NEXT MATRIX

## Why the frozen implementation had to change

`src/surface_crack/models.py` was frozen on 2026-08-29 with no normalization
layer anywhere in the network. Its `DoubleConv` docstring states the reason:
normalization was omitted so that batch size 1 stays valid down to the deepest
feature map, which is 1x1 for a 16x16 input.

Two authorised matrices (26.7 h, 72 runs) showed three failure modes, all
consistent with the absence of normalization:

1. Early trapping in a bad basin. Same architecture and data, seed only
   differing, reached 0.4408 and 0.5060 FgIoU in v1.
2. Mid-training gradient explosion. In v2, 13 of 36 runs blew up inside the
   contracted 30 epochs, peaking at epochs 8-21 and then falling 0.11-0.24
   without recovery. Effective training length varied from 8 to 30 epochs, so
   the arms were not comparable.
3. Extreme sensitivity to hyperparameters generally.

Warmup (2026-09-14) fixed the first. Gradient clipping was tested against the
second and rescued one of three probe arms; its mechanism was diagnosed as
insufficient, because the collapse is not a single-step outlier but an upward
drift of the entire gradient distribution (`grad_norm_mean` 1.7 -> 2.4 -> 5.3
before the `max_s2027` collapse, clipped steps 28 -> 70 -> 140), and AdamW
already normalises step size by sqrt(v), so norm clipping mostly reweights
direction.

The stated criterion for adopting a stabiliser, fixed before this probe was
run: all three known-explosion arms complete 30 epochs AND peak minus the mean
of the last three epochs is below 0.05.

## What changed

`src/surface_crack/models.py` gains `NORM_CHOICES = ("none", "group")` and a
`norm` argument on `UNet` and `DoubleConv`. When `norm="group"`, each of the 18
convolutions in the encoders, bottleneck and decoders is followed by
`nn.GroupNorm(num_groups=gcd(8, channels), num_channels=channels)` before its
ReLU. The default is `"none"`, so every run recorded before this date keeps its
exact numerical behaviour, and the migration-equivalence record still
reproduces: `UNet(pooling="max", base_channels=16, norm="none")` is still
1941105 parameters, matching the reference machine.

GroupNorm is compatible with the constraint the docstring describes. It
normalises over channel groups within one sample and never over the batch, so
batch size 1 and a 1x1 deepest feature map are both valid. This was confirmed by
a forward pass at 16x16 with batch 1, which produces finite output. BatchNorm
and InstanceNorm remain excluded for exactly the reason the docstring gives.

`tools/train_pooling_experiment.py` exposes `--norm {none,group}`, default
`none`, and records it in `result.json`. The second backbone
`pool_adapter_resnet18_unet` is not wired and raises on `--norm group` rather
than silently returning a model with a different normalisation setting from the
main backbone.

Nothing else moves: optimizer, learning rate, weight decay, epochs, effective
batch size, augmentation, loss, warmup schedule and checkpoint selection are
untouched.

## Effect on parameter matching

The H1 contrast requires `btsp` and the matched upper-tail control to have equal
parameter counts. GroupNorm adds 5888 parameters at base_channels 32, and the
increment is identical for all twelve operators, so every existing matched pair
stays matched:

| operator | norm=none | norm=group | delta |
|---|---|---|---|
| `btsp` | 7760101 | 7765989 | +5888 |
| `upper_tail_matched` | 7760101 | 7765989 | +5888 |
| `avg_top2`, `max`, `avg`, `min`, `fixed_max_avg` | 7760097 | 7765985 | +5888 |
| `t_max_avg`, `learned_max_avg` | 7760101 | 7765989 | +5888 |
| `v0_max_mean_matched`, `v0_max_mean_min` | 7760109 | 7765997 | +5888 |
| `stride_conv` | 8543457 | 8549345 | +5888 |

## Evidence

The three arms that collapsed in v2, re-run at seed 2027 with five warmup
epochs and `--norm group`, against the same three arms without it and with
clipping at 10:

| arm | setting | epochs | peak | last-3 mean | drop | max grad norm |
|---|---|---|---|---|---|---|
| `max` | v2, warmup only | 30 | 0.5630 | 0.3406 | 0.2224 | 14144 |
| `max` | + clip 10 | 30 | 0.5657 | 0.3398 | 0.2259 | 2775 |
| `max` | **+ GroupNorm** | 30 | **0.6104** | **0.5946** | **0.0158** | **25.7** |
| `avg` | v2, warmup only | 30 | 0.5705 | 0.3283 | 0.2422 | 272386 |
| `avg` | + clip 10 | 30 | 0.5670 | 0.5354 | 0.0317 | 207 |
| `avg` | **+ GroupNorm** | 30 | **0.6046** | **0.5914** | **0.0131** | **21.5** |
| `t_max_avg` | v2, warmup only | 30 | 0.5704 | 0.3414 | 0.2289 | 183 |
| `t_max_avg` | + clip 10 | 30 | 0.5491 | 0.4790 | 0.0701 | 486 |
| `t_max_avg` | **+ GroupNorm** | 30 | **0.6050** | **0.5912** | **0.0138** | **25.9** |

All three pass the pre-stated criterion, by an order of magnitude rather than
marginally. No non-finite gradient step occurred in any of the three runs, and
the largest gradient norm observed over all 90 epochs is 25.9, against a v2
maximum of 272386.

The accuracy level also rises, from roughly 0.565 to roughly 0.605 peak, but
that is not what the probe was run to establish and it is measured at one seed
per arm.

## Consequences

- The v2 matrix is superseded. It stays as the v2 exploratory record and is not
  mixed into any later index.
- The next matrix must be run in full with `--norm group`. No arm carries over,
  because a normalised and an unnormalised network are different models.
- The implementation freeze hash for `models.py` is broken. The manifest value
  is superseded as recorded below. The frozen file remains byte-recoverable in
  behaviour through `norm="none"`, which is the default.
- Evidence used: training and validation data only.
- Changes H1, H2, H3, primary metric, stratification thresholds or Go/No-Go: no.
- Authorises test or external-test access: no.
- Authorises formal train/validation execution: no. The existing authorisation
  record is re-pointed at the new hashes; it grants nothing new.

## Audit chain

`baseline_platform/src/surface_crack/models.py`

- as frozen in `PACKAGE_CONTENTS_SHA256.json`:
  `24802c14845653457e07affe7aacf1a2f707e530132eb7c428790281219275f4`
- after this change:
  `c402e7c1092ee8b3230fa0db3f8646f8921503cb03aa516bab8210366c487cee`

This is the first change to `models.py` since the package was frozen, so the
manifest value is also its immediately preceding state.

`baseline_platform/tools/train_pooling_experiment.py`

- after the MA(5) checkpoint-selection change:
  `34c1622b60b38f1fa04d432e6d1a61159ca9f99f51d75f6d436f0bfe8e641808`
- after `--norm` (this change):
  `32b840c5b54d4f53723a8ae457a40d4778121cb6d99c08547617aedd5e84f7c4`
