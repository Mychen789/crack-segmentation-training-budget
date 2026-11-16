# Data, code, and supplemental materials: Effects of Training Duration on Comparisons of Downsampling Operators for Crack Segmentation

Mengyao Chen, Baohua Guo, Xiao Yu, and David Bassir

This repository accompanies the manuscript *Effects of Training Duration on Comparisons of Downsampling Operators for Crack Segmentation* (submitted to the *Journal of Computing in Civil Engineering*). It contains the Supplemental Materials cited in the manuscript, the validation records of all training runs, the core-test confirmation records, the preregistration and amendment documents, and the scripts used to recompute the reported numbers.

The original image datasets are not redistributed. CRACK500 (Yang et al. 2020; Zhang et al. 2016) and DeepCrack537 (Liu et al. 2019) are available from their original releases. Model weights are not included.

## Contents

| Path | Content | Manuscript reference |
|---|---|---|
| `supplemental/Supplemental_Materials.pdf` | Tables S1–S8 and Figs. S1–S2 | Supplemental Materials |
| `supplemental/figures/` | Figs. S1 and S2 as PDF, PNG, and SVG | Figs. S1–S2 |
| `data/per_epoch_validation.csv` | Per-epoch validation metrics of the 216 main runs (10,440 rows; foreground IoU, Dice, boundary F1, clDice, AUPRC, precision, recall, training loss) | Sections on budget sensitivity |
| `data/run_summary.csv` | One row per main run: MA(5) read-out, peak epoch, run time, parameter count | Table 4, Table S3 |
| `data/runs/` | Per-run JSON records of the main constant-learning-rate runs | Tables 4, S2, S3 |
| `data/runs-cosine/` | Records of the 66 cosine-schedule runs | Table 4, Fig. 5 |
| `data/runs_s22/` | Records of the 33 ConvNeXt-UNet runs | Table 4 |
| `data/tables/` | Derived tables, including operator omission (`s23_*`) and core-test pairs and operator scores (`s24_*`) | Tables S4, S5 |
| `data/rule_eval_raw/` | Raw outputs of the two reporting-rule evaluation rounds | Fig. 6, Tables S6–S8 |
| `data/figures_data/` | Source data exported for early figure drafts (BTSP rank by budget, pairwise moves, MA(5) trajectories) | — |
| `artifacts/` | Frozen protocol, implementation, and training-contract records with SHA-256 bindings | Table 2 |
| `artifacts/s24/` | Core-test freeze, execution freeze, checkpoint metadata, validation reproduction, test confirmation (including 1,980 group confusion records), and independent audit | Core-test confirmation, Table S5 |
| `records/` | Preregistered protocol versions v0.3–v0.5, decision logs documenting every registered amendment, and the protocols of the reporting-rule, cosine-schedule, ConvNeXt-UNet, and core-test analyses | Table 2 |
| `train_pooling_experiment.py` | Training and evaluation script | Methods |
| `tools/`, `tools_v3/` | Analysis scripts (reporting rules, cosine and ConvNeXt analyses, operator omission, core-test statistics and audit) | Results |
| `figure_scripts/` | Scripts that draw the English figures | Figs. 1–6, S1–S2 |

Most protocol and decision-log documents in `records/` were written in Chinese during the study and are provided unchanged so that their SHA-256 bindings remain valid. The JSON records in `artifacts/` are in English.

## Key definitions

- The read-out of a run is the maximum five-epoch moving average, MA(5), of validation foreground IoU within the budget B.
- The paired difference of operators a and b at budget B is the difference of their three-seed mean read-outs. A comparison changes when the paired difference shifts by more than δ = 0.005 IoU between the low and high budgets.
- Fcross is the proportion of the 55 operator pairs whose comparison changes. Mmax is the largest absolute shift.

## Recomputation

The scripts require Python 3.10 or later with NumPy, SciPy, and pandas; training additionally requires PyTorch. Paths inside some records refer to the original workstation (for example `E:\...`) and are kept unchanged because the records are hash-bound. Typical entry points:

- `tools_v3/analyze_s23_operator_omission.py` recomputes Table S4.
- `tools_v3/s24_test_statistics.py` and `tools_v3/analyze_s24_test_report.py` recompute and audit the core-test results in Table S5.
- `tools/budget_robust_reporting.py` and `tools/budget_robust_reporting_round2.py` evaluate the reporting rules R0–R6.

## Licenses

- Data, documents, figures, and supplemental materials: Creative Commons Attribution 4.0 International (CC BY 4.0), see `LICENSE-DATA`.
- Code (`*.py`): MIT License, see `LICENSE`.

## Citation

If you use these materials, please cite the article and this dataset (see `CITATION.cff`).
