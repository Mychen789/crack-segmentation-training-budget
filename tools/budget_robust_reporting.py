"""按 S16 协议评估五个报数规程的预算稳健性。

只读已落盘的 run 轨迹，不训练、不写 run 目录、不触碰 test 或 external_test。

规程、超参网格、三条判据与达标线全部取自
`phase1_protocol_audit/S16_BUDGET_ROBUST_REPORTING_PROTOCOL_2026-10-25.md`，
本文件不引入协议之外的任何选项。

硬约束（S16 §2）：每个规程只能看一支 run 的前 B 轮，不得使用更长的轨迹。
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

MA_WINDOW = 5                 # 与 train_pooling_experiment.py 的 MA_WINDOW 相同
EFFECT_THRESHOLD = 0.005      # 协议的效应门槛，本文件中是常量，不是可调项

OPERATORS = (
    "btsp", "diag_partition_matched", "t_max_avg", "avg_top2", "learned_max_avg",
    "fixed_max_avg", "v0_max_mean_min", "max", "avg", "min", "stride_conv",
)
SEEDS = ("2026", "2027", "2028")

INADMISSIBLE = None           # 规程返回 None 表示不可报告


# --------------------------------------------------------------------------
# 轨迹读取
# --------------------------------------------------------------------------

def load_trajectories(runs_dir: Path, pattern: str) -> dict[tuple[str, str], list[float]]:
    """(算子, 种子) -> 逐轮 validation foreground_iou。"""
    out: dict[tuple[str, str], list[float]] = {}
    for result in sorted(runs_dir.glob(f"{pattern}/result.json")):
        payload = json.loads(result.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETE":
            continue
        key = (payload["pooling"], str(payload["seed"]))
        out[key] = [e["validation"]["foreground_iou"] for e in payload["history"]]
    return out


# --------------------------------------------------------------------------
# 规程，五个。每个只看 values[:budget]
# --------------------------------------------------------------------------

def _ma5(values: list[float]) -> list[float | None]:
    """MA5[i] 对应第 i+1 轮；不足窗口长度者为 None。与训练器定义一致。"""
    out: list[float | None] = []
    for i in range(len(values)):
        if i + 1 >= MA_WINDOW:
            out.append(sum(values[i + 1 - MA_WINDOW:i + 1]) / MA_WINDOW)
        else:
            out.append(None)
    return out


def _ma5_argmax(values: list[float]) -> tuple[float, int] | None:
    """返回 (最佳 MA5 值, 1 基轮号)。严格大于比较，故并列取最早轮，与训练器一致。"""
    best_value = float("-inf")
    best_epoch = 0
    for index, smoothed in enumerate(_ma5(values)):
        if smoothed is not None and smoothed > best_value:
            best_value = smoothed
            best_epoch = index + 1
    return None if best_epoch == 0 else (best_value, best_epoch)


def rule_r0(values: list[float], budget: int, **_: object) -> float | None:
    """现行规则：前 budget 轮内拖尾 MA(5) 的 argmax 值。"""
    found = _ma5_argmax(values[:budget])
    return None if found is None else found[0]


def rule_r1(values: list[float], budget: int, gate: int = 5, **_: object) -> float | None:
    """收敛门控：argmax 轮距预算末端不足 gate 轮即不可报告。"""
    found = _ma5_argmax(values[:budget])
    if found is None:
        return INADMISSIBLE
    value, epoch = found
    return INADMISSIBLE if epoch > budget - gate else value


def rule_r2(values: list[float], budget: int, window: int = 5,
            delta: float = 0.001, **_: object) -> float | None:
    """各自收敛点：首个 MA5[e] - MA5[e-window] < delta 的轮，报该处 MA5。"""
    smoothed = _ma5(values[:budget])
    for index in range(len(smoothed)):
        current = smoothed[index]
        earlier = smoothed[index - window] if index - window >= 0 else None
        if current is None or earlier is None:
            continue
        if current - earlier < delta:
            return current
    return INADMISSIBLE


def rule_r3(values: list[float], budget: int, tail: int = 5, **_: object) -> float | None:
    """末窗均值：前 budget 轮的最后 tail 轮原始 FgIoU 的均值，无 argmax。"""
    window = values[:budget][-tail:]
    return statistics.mean(window) if len(window) == tail else INADMISSIBLE


RULES = {
    "R0": rule_r0,
    "R1": rule_r1,
    "R2": rule_r2,
    "R3": rule_r3,
    "R4": rule_r1,   # R4 复用 R1 的臂级门控，另加对级的符号一致性，见 _pair_delta
}

# S16 §4 的超参网格，不增不减
GRIDS: dict[str, list[dict[str, object]]] = {
    "R0": [{}],
    "R1": [{"gate": g} for g in (3, 5, 8, 12)],
    "R2": [{"window": w, "delta": d} for w in (5, 10) for d in (0.0005, 0.001, 0.002)],
    "R3": [{"tail": k} for k in (5, 10)],
    "R4": [{"gate": g} for g in (3, 5, 8, 12)],
}


# --------------------------------------------------------------------------
# 对级量：配对差与可报告性
# --------------------------------------------------------------------------

def _pair_delta(traj: dict, rule_name: str, params: dict, budget: int,
                left: str, right: str) -> float | None:
    """算子对的配对差（三种子均值）。任一臂任一种子不可报告则整对不可报告。

    R4 另要求三个种子的配对差符号一致（3/3），否则整对不可报告。
    """
    rule = RULES[rule_name]
    per_seed = []
    for seed in SEEDS:
        if (left, seed) not in traj or (right, seed) not in traj:
            return None
        a = rule(traj[(left, seed)], budget, **params)
        b = rule(traj[(right, seed)], budget, **params)
        if a is INADMISSIBLE or b is INADMISSIBLE:
            return None
        per_seed.append(a - b)
    if rule_name == "R4":
        positives = sum(1 for x in per_seed if x > 0)
        if positives not in (0, len(per_seed)):
            return None
    return statistics.mean(per_seed)


def evaluate(traj: dict, rule_name: str, params: dict,
             budget_low: int, budget_high: int,
             reference: dict[tuple[str, str], float]) -> dict:
    """S16 §5 的三条判据。reference 为参照预算下 R0 的配对差，用于 A_fid。"""
    moves: list[float] = []
    crossings = flips = 0
    fid_match = fid_total = 0
    reportable = 0

    for left, right in itertools.combinations(OPERATORS, 2):
        low = _pair_delta(traj, rule_name, params, budget_low, left, right)
        high = _pair_delta(traj, rule_name, params, budget_high, left, right)
        if low is None or high is None:
            continue
        reportable += 1
        movement = abs(high - low)
        moves.append(movement)
        if movement > EFFECT_THRESHOLD:
            crossings += 1
        if low * high < 0:
            flips += 1
        ref = reference.get((left, right))
        if ref is not None and ref != 0.0 and low != 0.0:
            fid_total += 1
            if (low > 0) == (ref > 0):
                fid_match += 1

    total_pairs = len(list(itertools.combinations(OPERATORS, 2)))
    return {
        "rule": rule_name,
        "params": params,
        "reportable_pairs": reportable,
        "C_adm": reportable / total_pairs,
        "F_cross": crossings / reportable if reportable else float("nan"),
        "F_flip": flips / reportable if reportable else float("nan"),
        "M_max": max(moves) if moves else float("nan"),
        "A_fid": fid_match / fid_total if fid_total else float("nan"),
    }


def reference_deltas(traj: dict, budget: int) -> dict[tuple[str, str], float]:
    """参照：R0 在收敛预算下的配对差（S16 §5.3 与 §6.2）。"""
    out = {}
    for left, right in itertools.combinations(OPERATORS, 2):
        delta = _pair_delta(traj, "R0", {}, budget, left, right)
        if delta is not None:
            out[(left, right)] = delta
    return out


def meets_criteria(row: dict, f_cross_r0: float) -> bool:
    """S16 §5.4：三条同时满足。"""
    if row["reportable_pairs"] == 0:
        return False
    return (
        row["F_cross"] <= 0.5 * f_cross_r0
        and row["C_adm"] >= 0.50
        and row["A_fid"] >= 0.90
    )


# --------------------------------------------------------------------------

CELLS = {
    "design_dc537_unet":   ("b90_unet_gr_deepcrack537_*", 90),
    "valid_dc537_rn18":    ("b90_rn18_gr_deepcrack537_*", 90),
    "valid_c500_unet":     ("formal_gn_crack500_unet_*", 30),
    "valid_c500_rn18":     ("g2_rn18_gn_crack500_*", 30),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="baseline_platform/runs")
    parser.add_argument("--cell", choices=sorted(CELLS), required=True)
    parser.add_argument("--budget-low", type=int, required=True)
    parser.add_argument("--budget-high", type=int, required=True)
    parser.add_argument("--only-rule", default=None,
                        help="只跑一个规程，并可配合 --params 指定冻结超参")
    parser.add_argument("--params", default=None,
                        help="JSON，例如 '{\"gate\": 5}'，给定则不扫网格")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    pattern, reference_budget = CELLS[args.cell]
    traj = load_trajectories(Path(args.runs_dir), pattern)
    print(f"cell={args.cell}  runs={len(traj)}  "
          f"预算对=({args.budget_low}, {args.budget_high})  参照预算={reference_budget}")

    reference = reference_deltas(traj, reference_budget)
    baseline = evaluate(traj, "R0", {}, args.budget_low, args.budget_high, reference)
    f_cross_r0 = baseline["F_cross"]
    print(f"R0 基线：F_cross={f_cross_r0:.3f}  达标线=F_cross<={0.5 * f_cross_r0:.3f}")
    print()

    header = f"{'规程':6s} {'超参':26s} {'C_adm':>7s} {'F_cross':>8s} {'F_flip':>7s} {'M_max':>8s} {'A_fid':>7s} {'达标':>5s}"
    print(header)
    print("-" * len(header))

    rows = []
    rule_names = [args.only_rule] if args.only_rule else list(RULES)
    for name in rule_names:
        grid = [json.loads(args.params)] if args.params else GRIDS[name]
        for params in grid:
            row = evaluate(traj, name, params, args.budget_low, args.budget_high, reference)
            row["meets"] = meets_criteria(row, f_cross_r0) if name != "R0" else None
            rows.append(row)
            mark = "" if row["meets"] is None else ("达标" if row["meets"] else "—")
            print(f"{name:6s} {json.dumps(params, ensure_ascii=False):26s} "
                  f"{row['C_adm']:7.3f} {row['F_cross']:8.3f} {row['F_flip']:7.3f} "
                  f"{row['M_max']:8.4f} {row['A_fid']:7.3f} {mark:>5s}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"cell": args.cell, "budget_low": args.budget_low,
                        "budget_high": args.budget_high,
                        "reference_budget": reference_budget,
                        "f_cross_r0": f_cross_r0, "rows": rows},
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n已写出 {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
