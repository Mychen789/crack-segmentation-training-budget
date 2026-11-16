"""把论文所需的数据、表格、图数据、记录与工具导出成一个自洽的包。

只读已落盘产物。不训练，不写 run 目录，不触碰 test 或 external_test。
导出的全部数字可由包内 tools/ 下的脚本在包内重算。
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import shutil
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "baseline_platform" / "runs"
PKG = ROOT / "paper_package_2026-10-25"

OPERATORS = ("btsp", "diag_partition_matched", "t_max_avg", "avg_top2",
             "learned_max_avg", "fixed_max_avg", "v0_max_mean_min",
             "max", "avg", "min", "stride_conv")
SEEDS = ("2026", "2027", "2028")
EFFECT = 0.005

# 六个格：(标签, run 名 glob, 预算)
CELLS = (
    ("unet_crack500_30",      "formal_gn_crack500_unet_*",     30),
    ("unet_deepcrack537_30",  "formal_gn_deepcrack537_unet_*", 30),
    ("rn18_crack500_30",      "g2_rn18_gn_crack500_*",         30),
    ("rn18_deepcrack537_30",  "g2_rn18_gn_deepcrack537_*",     30),
    ("unet_deepcrack537_90",  "b90_unet_gr_deepcrack537_*",    90),
    ("rn18_deepcrack537_90",  "b90_rn18_gr_deepcrack537_*",    90),
)

RECORDS = (
    "PAPER_SKELETON_v1_2026-10-25.md",
    "preregistered_protocol_v0.3.md",
    "preregistered_protocol_v0.4.md",
    "preregistered_protocol_v0.5.md",
    "P3_CONSOLIDATED_CLAIM_EVIDENCE_2026-09-25.md",
    "S5b_SYSTEMATIC_PROTOCOL_2026-10-10.md",
    "S5c_SYSTEMATIC_RESULTS_2026-10-10.md",
    "S5d_SYSTEMATIC_PROTOCOL_ROUND2_2026-10-10.md",
    "S7_CRACK500_CONVERGENCE_PROBE_2026-09-25.md",
    "S8_DIFFICULTY_STRATA_POWER_2026-09-25.md",
    "S9_LOWER_TAIL_STARVATION_DECOMPOSITION_2026-09-25.md",
    "S11_STRATIFIED_METRICS_2026-09-25.md",
    "S12_G2_SECOND_BACKBONE_STAGE_ONE_2026-09-29.md",
    "S13_LITERATURE_GROUNDING_FOR_EXPERIMENTAL_FINDINGS_2026-10-10.md",
    "S14_G2_CRACK500_SECOND_BACKBONE_RESULTS_2026-10-12.md",
    "S15_DEEPCRACK537_90EPOCH_RESULTS_2026-10-25.md",
    "S16_BUDGET_ROBUST_REPORTING_PROTOCOL_2026-10-25.md",
    "S17_BUDGET_ROBUST_REPORTING_RESULTS_2026-10-25.md",
    "S18_BUDGET_ROBUST_REPORTING_PROTOCOL_ROUND2_2026-10-25.md",
    "S19_BUDGET_ROBUST_REPORTING_RESULTS_ROUND2_2026-10-25.md",
    "S20_BUDGET_SENSITIVITY_DECOMPOSITION_2026-10-25.md",
)

TOOLS = (
    "budget_robust_reporting.py",
    "budget_robust_reporting_round2.py",
    "export_paper_package.py",
    "probe_pooling_branches.py",
    "probe_alpha_sensitivity.py",
    "reevaluate_checkpoints.py",
    "analyze_h1_capacity_controls.py",
    "benchmark_pooling_efficiency.py",
)

ARTIFACTS = (
    "preregistered_protocol_v0.4_freeze.json",
    "preregistered_protocol_v0.5_freeze.json",
    "implementation_freeze_v2.json",
    "train_val_execution_authorization_v1.json",
    "recomputed_validation_metrics_v2.json",
    "stratified_validation_metrics_v1.json",
    "training_contract_freeze_v1.json",
)


def ma5(values):
    return [None if i + 1 < 5 else sum(values[i - 4:i + 1]) / 5
            for i in range(len(values))]


def ma5_argmax(values):
    best_value, best_epoch = float("-inf"), 0
    for index, smoothed in enumerate(ma5(values)):
        if smoothed is not None and smoothed > best_value:
            best_value, best_epoch = smoothed, index + 1
    return (best_value, best_epoch) if best_epoch else (None, None)


def load_cell(pattern):
    """(算子, 种子) -> payload，只收 COMPLETE。"""
    out = {}
    for result in sorted(RUNS.glob(f"{pattern}/result.json")):
        payload = json.loads(result.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETE":
            continue
        out[(payload["pooling"], str(payload["seed"]))] = payload
    return out


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> int:
    for sub in ("data/runs", "data/tables", "records", "tools",
                "refs/pdf", "figures_data", "artifacts"):
        (PKG / sub).mkdir(parents=True, exist_ok=True)

    cells = {label: load_cell(pattern) for label, pattern, _ in CELLS}

    # ---------------- 1. 原始 run 产物 ----------------
    copied = 0
    for label, pattern, _ in CELLS:
        for result in sorted(RUNS.glob(f"{pattern}/result.json")):
            payload = json.loads(result.read_text(encoding="utf-8"))
            if payload.get("status") != "COMPLETE":
                continue
            shutil.copy2(result, PKG / "data/runs" / f"{result.parent.name}.json")
            copied += 1
    # S6 与 S7 的收敛性探针，正文 §4 与 §6.5 引用
    for result in sorted(RUNS.glob("probe_long_*/result.json")):
        shutil.copy2(result, PKG / "data/runs" / f"{result.parent.name}.json")
        copied += 1
    print(f"data/runs  复制 {copied} 个 result.json")

    # ---------------- 2. 逐轮长表，一切图表的母数据 ----------------
    rows = []
    for label, pattern, _ in CELLS:
        for key, payload in sorted(cells[label].items()):
            for entry in payload["history"]:
                v = entry["validation"]
                rows.append([label, payload["backbone"], payload["dataset_id"],
                             payload["epochs_target"], payload["pooling"],
                             payload["seed"], entry["epoch"],
                             v["foreground_iou"], v["dice"], v["boundary_f1"],
                             v["cldice"], v["auprc"], v["precision"], v["recall"],
                             entry["train_loss"], entry.get("nonfinite_grad_steps")])
    write_csv(PKG / "data/per_epoch_validation.csv",
              ["cell", "backbone", "dataset", "epochs_target", "pooling", "seed",
               "epoch", "foreground_iou", "dice", "boundary_f1", "cldice", "auprc",
               "precision", "recall", "train_loss", "nonfinite_grad_steps"], rows)
    print(f"data/per_epoch_validation.csv  {len(rows)} 行（图表母数据）")

    # ---------------- 3. run 级汇总 ----------------
    rows = []
    for label, pattern, budget in CELLS:
        for (op, seed), payload in sorted(cells[label].items()):
            rows.append([label, payload["backbone"], payload["dataset_id"], budget,
                         op, seed, payload["best_foreground_iou_ma5"],
                         payload["best_epoch_ma5"], payload["best_foreground_iou"],
                         payload["elapsed_seconds"], payload["parameter_count"],
                         payload["train_samples"], payload["val_samples"],
                         payload["train_val_execution_authorization_sha256"][:16]])
    write_csv(PKG / "data/run_summary.csv",
              ["cell", "backbone", "dataset", "epochs_target", "pooling", "seed",
               "best_fg_iou_ma5", "best_epoch_ma5", "best_fg_iou_raw",
               "elapsed_seconds", "parameter_count", "train_samples", "val_samples",
               "authorization_sha256_prefix"], rows)
    print(f"data/run_summary.csv  {len(rows)} 行")

    # ---------------- 4. 表 2：六格 × 11 算子排名 ----------------
    rows = []
    for label, pattern, budget in CELLS:
        means = {op: statistics.mean(cells[label][(op, s)]["best_foreground_iou_ma5"]
                                     for s in SEEDS) for op in OPERATORS}
        order = sorted(OPERATORS, key=lambda o: -means[o])
        top = order[0]
        for rank, op in enumerate(order, start=1):
            rows.append([label, budget, rank, op, round(means[op], 6),
                         round(means[top] - means[op], 6),
                         "yes" if means[top] - means[op] > EFFECT else "no"])
    write_csv(PKG / "data/tables/table2_rankings_six_cells.csv",
              ["cell", "epochs", "rank", "pooling", "fg_iou_ma5_mean",
               "gap_to_top", "gap_exceeds_0.005"], rows)
    print("data/tables/table2_rankings_six_cells.csv")

    # ---------------- 5. 图 1：BTSP 名次对预算 ----------------
    rows = []
    for label, pattern, budget in CELLS:
        means = {op: statistics.mean(cells[label][(op, s)]["best_foreground_iou_ma5"]
                                     for s in SEEDS) for op in OPERATORS}
        order = sorted(OPERATORS, key=lambda o: -means[o])
        rows.append([label, label.split("_")[0], "_".join(label.split("_")[1:-1]),
                     budget, order.index("btsp") + 1, round(means["btsp"], 6),
                     order[0], round(means[order[0]], 6)])
    write_csv(PKG / "figures_data/fig1_btsp_rank_vs_budget.csv",
              ["cell", "backbone", "dataset", "epochs", "btsp_rank",
               "btsp_fg_iou_ma5", "top_operator", "top_fg_iou_ma5"], rows)
    print("figures_data/fig1_btsp_rank_vs_budget.csv")

    # ---------------- 6. 表 3 与 图 2：四格预算敏感性 ----------------
    budget_pairs = (("unet_deepcrack537_90", "unet_deepcrack537_90", 30, 90),
                    ("rn18_deepcrack537_90", "rn18_deepcrack537_90", 30, 90),
                    ("unet_crack500_30",     "unet_crack500_30",     15, 30),
                    ("rn18_crack500_30",     "rn18_crack500_30",     15, 30))
    rows, pair_rows = [], []
    for label, _, low, high in budget_pairs:
        cell = cells[label]
        traj = {k: [e["validation"]["foreground_iou"] for e in v["history"]]
                for k, v in cell.items()}
        crossings = flips = 0
        moves = []
        for a, b in itertools.combinations(OPERATORS, 2):
            d_low = statistics.mean(ma5_argmax(traj[(a, s)][:low])[0]
                                    - ma5_argmax(traj[(b, s)][:low])[0] for s in SEEDS)
            d_high = statistics.mean(ma5_argmax(traj[(a, s)][:high])[0]
                                     - ma5_argmax(traj[(b, s)][:high])[0] for s in SEEDS)
            move = abs(d_high - d_low)
            moves.append(move)
            if move > EFFECT:
                crossings += 1
            if d_low * d_high < 0:
                flips += 1
            pair_rows.append([label, low, high, a, b, round(d_low, 6),
                              round(d_high, 6), round(move, 6),
                              "yes" if move > EFFECT else "no",
                              "yes" if d_low * d_high < 0 else "no"])
        total = len(moves)
        # 收敛性指标给四个口径，因为"贴边"有两种常见定义，混用会读错：
        #   last  = argmax 恰好落在该预算的最后一轮
        #   w2    = argmax 落在最后两轮之内（即 >= B-1）
        def frac(budget, margin):
            hit = sum(1 for k in traj
                      if ma5_argmax(traj[k][:budget])[1] >= budget - margin)
            return round(hit / len(traj), 4)
        rows.append([label, low, high, total, crossings, round(crossings / total, 4),
                     flips, round(flips / total, 4), round(max(moves), 6),
                     round(statistics.median(moves), 6), len(traj),
                     frac(low, 0), frac(low, 1), frac(high, 0), frac(high, 1)])
    write_csv(PKG / "data/tables/table3_budget_sensitivity.csv",
              ["cell", "budget_low", "budget_high", "n_pairs", "n_cross_0.005",
               "F_cross", "n_flip", "F_flip", "M_max", "move_median", "n_runs",
               "frac_argmax_last_epoch_at_low", "frac_argmax_within2_at_low",
               "frac_argmax_last_epoch_at_high", "frac_argmax_within2_at_high"], rows)
    write_csv(PKG / "figures_data/fig2_pairwise_moves.csv",
              ["cell", "budget_low", "budget_high", "op_a", "op_b", "delta_low",
               "delta_high", "abs_move", "crosses_0.005", "sign_flip"], pair_rows)
    print(f"data/tables/table3_budget_sensitivity.csv  +  "
          f"figures_data/fig2_pairwise_moves.csv（{len(pair_rows)} 对）")

    # ---------------- 7. 图 3：MA(5) 轨迹与 argmax 位置 ----------------
    rows = []
    for label in ("unet_deepcrack537_90",):
        for op in ("btsp", "t_max_avg", "min", "learned_max_avg"):
            for seed in SEEDS:
                values = [e["validation"]["foreground_iou"]
                          for e in cells[label][(op, seed)]["history"]]
                smoothed = ma5(values)
                for budget in (30, 90):
                    _, peak = ma5_argmax(values[:budget])
                    for epoch, (raw, sm) in enumerate(zip(values, smoothed), start=1):
                        if epoch > budget:
                            break
                        rows.append([label, op, seed, budget, epoch, raw,
                                     "" if sm is None else round(sm, 6),
                                     "yes" if epoch == peak else "no",
                                     "yes" if peak >= budget else "no"])
    write_csv(PKG / "figures_data/fig3_ma5_trajectories.csv",
              ["cell", "pooling", "seed", "budget", "epoch", "fg_iou", "ma5",
               "is_argmax_at_this_budget", "argmax_at_edge"], rows)
    print(f"figures_data/fig3_ma5_trajectories.csv  {len(rows)} 行")

    # ---------------- 8. 表 4 / 图 4：规程 × 判据 ----------------
    scratch = Path(r"C:\Users\123\AppData\Local\Temp\claude"
                   r"\D--cmy------migration-package-preformal-v1"
                   r"\d8f42865-1ae1-477a-8864-b0067f08f5fe\scratchpad")
    rows = []
    for stem, rnd in (("s17_design", 1), ("s17_valid_dc537_rn18", 1),
                      ("s17_valid_c500_unet", 1), ("s17_valid_c500_rn18", 1),
                      ("s19_design", 2), ("s19_valid_dc537_rn18", 2),
                      ("s19_valid_c500_unet", 2), ("s19_valid_c500_rn18", 2)):
        src = scratch / f"{stem}.json"
        if not src.is_file():
            print(f"  缺 {stem}.json，跳过")
            continue
        blob = json.loads(src.read_text(encoding="utf-8"))
        for row in blob["rows"]:
            rows.append([rnd, blob["cell"], blob["budget_low"], blob["budget_high"],
                         row["rule"], json.dumps(row.get("params", {}), ensure_ascii=False),
                         row["reportable_pairs"], round(row["C_adm"], 4),
                         row["F_cross"], row["F_flip"], row["M_max"], row["A_fid"],
                         row.get("S_disc", ""), row.get("E_abs", ""),
                         row.get("meets", "")])
    write_csv(PKG / "data/tables/table4_reporting_rules.csv",
              ["round", "cell", "budget_low", "budget_high", "rule", "params",
               "reportable_pairs", "C_adm", "F_cross", "F_flip", "M_max", "A_fid",
               "S_disc", "E_abs", "meets_criteria"], rows)
    shutil.copytree(scratch, PKG / "data/rule_eval_raw", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("*.py", "*.output"))
    print(f"data/tables/table4_reporting_rules.csv  {len(rows)} 行")

    # ---------------- 8b. 表 5：抬升分解与恒等式核验（S20） ----------------
    # 恒等式：Delta(a,b,high) - Delta(a,b,low) = lift(a) - lift(b)
    # 故 M_max = max(lift) - min(lift)。两种算法独立算出并比对。
    rows, ident = [], []
    for label, _, low, high in budget_pairs:
        cell = cells[label]
        traj = {k: [e["validation"]["foreground_iou"] for e in v["history"]]
                for k, v in cell.items()}
        lift = {op: statistics.mean(ma5_argmax(traj[(op, s)][:high])[0]
                                    - ma5_argmax(traj[(op, s)][:low])[0]
                                    for s in SEEDS) for op in OPERATORS}
        for op in sorted(OPERATORS, key=lambda o: -lift[o]):
            rows.append([label, low, high, op, round(lift[op], 6)])
        via_lift = max(abs(lift[a] - lift[b])
                       for a, b in itertools.combinations(OPERATORS, 2))
        direct = max(abs(statistics.mean(ma5_argmax(traj[(a, s)][:high])[0]
                                         - ma5_argmax(traj[(b, s)][:high])[0]
                                         for s in SEEDS)
                         - statistics.mean(ma5_argmax(traj[(a, s)][:low])[0]
                                           - ma5_argmax(traj[(b, s)][:low])[0]
                                           for s in SEEDS))
                     for a, b in itertools.combinations(OPERATORS, 2))
        values = list(lift.values())
        ident.append([label, low, high, round(statistics.mean(values), 6),
                      round(max(values) - min(values), 6),
                      round(statistics.stdev(values), 6),
                      round(direct, 6), round(via_lift, 6),
                      f"{abs(direct - via_lift):.2e}"])
    write_csv(PKG / "data/tables/table5_lift_decomposition.csv",
              ["cell", "budget_low", "budget_high", "pooling", "lift"], rows)
    write_csv(PKG / "data/tables/table5b_identity_check.csv",
              ["cell", "budget_low", "budget_high", "lift_mean", "lift_range",
               "lift_sd", "M_max_direct", "M_max_via_lift_range", "abs_diff"], ident)
    print("data/tables/table5_lift_decomposition.csv + table5b_identity_check.csv")

    # ---------------- 9. 表 1：run 清单 ----------------
    rows = []
    for label, pattern, budget in CELLS:
        payloads = list(cells[label].values())
        rows.append([label, payloads[0]["backbone"], payloads[0]["dataset_id"], budget,
                     len({p["pooling"] for p in payloads}), len(SEEDS), len(payloads),
                     payloads[0]["parameter_count"], payloads[0]["train_samples"],
                     payloads[0]["val_samples"],
                     sum(e.get("nonfinite_grad_steps") or 0
                         for p in payloads for e in p["history"])])
    write_csv(PKG / "data/tables/table1_run_inventory.csv",
              ["cell", "backbone", "dataset", "epochs", "n_operators", "n_seeds",
               "n_runs", "parameter_count", "train_samples", "val_samples",
               "nonfinite_grad_steps_total"], rows)
    print("data/tables/table1_run_inventory.csv")

    # ---------------- 10. 记录、工具、产物 ----------------
    audit = ROOT / "phase1_protocol_audit"
    for name in RECORDS:
        src = audit / name
        if src.is_file():
            shutil.copy2(src, PKG / "records" / name)
        else:
            print(f"  记录缺失：{name}")
    for name in sorted(p.name for p in audit.glob("decision_log_*.md")):
        shutil.copy2(audit / name, PKG / "records" / name)
    for name in TOOLS:
        src = ROOT / "baseline_platform" / "tools" / name
        if src.is_file():
            shutil.copy2(src, PKG / "tools" / name)
    for name in ARTIFACTS:
        src = ROOT / "baseline_platform" / "artifacts" / name
        if src.is_file():
            shutil.copy2(src, PKG / "artifacts" / name)
    print(f"records {len(list((PKG/'records').iterdir()))} 个  "
          f"tools {len(list((PKG/'tools').iterdir()))} 个  "
          f"artifacts {len(list((PKG/'artifacts').iterdir()))} 个")

    # ---------------- 11. 清单与哈希 ----------------
    manifest = []
    for path in sorted(PKG.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.csv":
            manifest.append([str(path.relative_to(PKG)).replace("\\", "/"),
                             path.stat().st_size,
                             hashlib.sha256(path.read_bytes()).hexdigest()])
    write_csv(PKG / "MANIFEST.csv", ["path", "bytes", "sha256"], manifest)
    total = sum(r[1] for r in manifest)
    print(f"MANIFEST.csv  {len(manifest)} 个文件，合计 {total/1048576:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
