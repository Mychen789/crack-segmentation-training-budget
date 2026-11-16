"""按 S18 协议评估外推类报数规程 R5 与 R6 的预算稳健性。

第一轮的共用部分（轨迹读取、MA(5)、R0、配对差、F_cross/F_flip/M_max/C_adm/A_fid）
从 `budget_robust_reporting` 导入，该模块已在 v0.5 冻结记录中绑定哈希，本文件不改动它。

硬约束（S18 §0 第 1 条）：规程只对前 B 轮的 MA(5) 曲线拟合，
不得使用任何更长轨迹的信息。唯一的例外是 §3.5 的 E_abs，
那是**评估者**用 90 轮真值检查规程准不准，规程本身没有用到它。

拟合实现（S18 §6 要求记录的实现细节）：
`scipy.optimize.curve_fit`，method 为默认的 trf（给了 bounds），
p0 由数据确定性地导出，maxfev 固定 20000，无随机成分，故逐次复现。
拟合失败（RuntimeError / 超出 maxfev）一律按 INADMISSIBLE 处理。
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from budget_robust_reporting import (
    EFFECT_THRESHOLD, INADMISSIBLE, MA_WINDOW, OPERATORS, SEEDS,
    _ma5, _ma5_argmax, load_trajectories, rule_r0,
)

MAXFEV = 20000


# --------------------------------------------------------------------------
# R5：饱和曲线拟合，报渐近值
# --------------------------------------------------------------------------

def _still_rising(smoothed: list[float | None], budget: int) -> bool:
    """曲线在 B 处是否仍在上升，即 MA(5) 的 argmax 是否贴在预算末端。

    S18 §6.2.1：两点比较 MA5[B] > MA5[B/2] 检测不出局部峰值（B=90 时峰值在
    第 72 轮左右但 y90 仍高于 y45，25/33 支如此），故改用 argmax 的位置，
    与第一轮 R1 的门控同口径。
    """
    best_epoch = 0
    best_value = float("-inf")
    for index, value in enumerate(smoothed[:budget]):
        if value is not None and value > best_value:
            best_value = value
            best_epoch = index + 1
    return best_epoch >= budget


def _model_exp(e, a, c, tau):
    return a - c * np.exp(-e / tau)


def _model_pow(e, a, c, p):
    return a - c * np.power(e, -p)


_MODELS = {
    "exp": (_model_exp, ([0.0, 0.0, 0.5], [2.0, 2.0, 1000.0])),
    "pow": (_model_pow, ([0.0, 0.0, 0.01], [2.0, 2.0, 10.0])),
}


def _fit_asymptote(epochs: list[int], values: list[float], form: str):
    """返回 (渐近值 A, R²)，拟合失败返回 (None, None)。确定性。"""
    model, (lower, upper) = _MODELS[form]
    x = np.asarray(epochs, dtype=float)
    y = np.asarray(values, dtype=float)
    if len(x) < 4:
        return None, None
    # p0 确定性地由数据导出：A 猜末值略上方，C 猜观测跨度，形状参数取网格中段
    span = max(float(y.max() - y.min()), 1e-6)
    p0 = [float(y[-1]) + span, span, 20.0 if form == "exp" else 1.0]
    p0 = [min(max(v, lo), hi) for v, lo, hi in zip(p0, lower, upper)]
    try:
        popt, _ = curve_fit(model, x, y, p0=p0, bounds=(lower, upper), maxfev=MAXFEV)
    except (RuntimeError, ValueError):
        return None, None
    residual = y - model(x, *popt)
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(popt[0]), r2


def rule_r5(values: list[float], budget: int, form: str = "exp",
            fit_range: str = "all", rho: float = 0.80,
            kappa: float = 0.05, **_: object) -> float | None:
    smoothed = _ma5(values[:budget])
    points = [(i + 1, v) for i, v in enumerate(smoothed) if v is not None]
    if not points:
        return INADMISSIBLE
    if fit_range == "half":
        cutoff = math.ceil(budget / 2)
        points = [(e, v) for e, v in points if e >= cutoff]
    epochs = [e for e, _ in points]
    series = [v for _, v in points]
    if len(series) < 4:
        return INADMISSIBLE

    asymptote, r2 = _fit_asymptote(epochs, series, form)
    if asymptote is None or r2 is None or r2 < rho:
        return INADMISSIBLE

    observed_max = max(v for v in smoothed if v is not None)
    if not _still_rising(smoothed, budget):
        # S18 §6.1：曲线已过峰，单调饱和模型的渐近值本就该低于峰值，
        # 此时规程退化为 R0（报观测 MA(5) 最大值），守卫 2 不适用。
        return observed_max
    if asymptote < observed_max:                      # 守卫 2，仅在仍上升时适用
        return INADMISSIBLE
    if asymptote - observed_max > kappa:              # 守卫 3
        return INADMISSIBLE
    return asymptote


# --------------------------------------------------------------------------
# R6：两点外推，无曲线拟合
# --------------------------------------------------------------------------

def rule_r6(values: list[float], budget: int, ratio: float = 0.5,
            kappa: float = 0.05, **_: object) -> float | None:
    smoothed = _ma5(values[:budget])
    half = math.ceil(budget / 2)
    if budget - 1 >= len(smoothed) or half - 1 < 0:
        return INADMISSIBLE
    y_b = smoothed[budget - 1]
    y_half = smoothed[half - 1]
    if y_b is None or y_half is None:
        return INADMISSIBLE

    observed_max = max(v for v in smoothed if v is not None)
    if not _still_rising(smoothed, budget):
        # S18 §6.1：已过峰则退化为 R0
        return observed_max

    asymptote = y_b + (y_b - y_half) * ratio / (1.0 - ratio)
    if asymptote < observed_max:
        return INADMISSIBLE
    if asymptote - observed_max > kappa:
        return INADMISSIBLE
    return asymptote


RULES2 = {"R5": rule_r5, "R6": rule_r6}

GRIDS2: dict[str, list[dict[str, object]]] = {
    "R5": [{"form": f, "fit_range": r, "rho": rho, "kappa": k}
           for f in ("exp", "pow") for r in ("all", "half")
           for rho in (0.80, 0.90) for k in (0.05, 0.10)],
    "R6": [{"ratio": r, "kappa": k} for r in (0.5, 0.7) for k in (0.05, 0.10)],
}


# --------------------------------------------------------------------------
# 对级与格级评估
# --------------------------------------------------------------------------

def _score(traj, rule_name, params, budget, operator, seed):
    if rule_name == "R0":
        return rule_r0(traj[(operator, seed)], budget)
    return RULES2[rule_name](traj[(operator, seed)], budget, **params)


def _pair_delta2(traj, rule_name, params, budget, left, right):
    per_seed = []
    for seed in SEEDS:
        if (left, seed) not in traj or (right, seed) not in traj:
            return None
        a = _score(traj, rule_name, params, budget, left, seed)
        b = _score(traj, rule_name, params, budget, right, seed)
        if a is INADMISSIBLE or b is INADMISSIBLE:
            return None
        per_seed.append(a - b)
    return statistics.mean(per_seed)


def _operator_spread(traj, rule_name, params, budget):
    """S18 §3.4：三个种子全部可报告的算子，其分数均值的极差。"""
    means = []
    for operator in OPERATORS:
        values = [_score(traj, rule_name, params, budget, operator, s)
                  for s in SEEDS if (operator, s) in traj]
        if len(values) != len(SEEDS) or any(v is INADMISSIBLE for v in values):
            continue
        means.append(statistics.mean(values))
    return (max(means) - min(means)) if len(means) >= 2 else None


def evaluate2(traj, rule_name, params, budget_low, budget_high,
              reference, reference_budget, spread_reference):
    moves, crossings, flips = [], 0, 0
    fid_match = fid_total = reportable = 0
    for left, right in itertools.combinations(OPERATORS, 2):
        low = _pair_delta2(traj, rule_name, params, budget_low, left, right)
        high = _pair_delta2(traj, rule_name, params, budget_high, left, right)
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

    total = len(list(itertools.combinations(OPERATORS, 2)))
    spread = _operator_spread(traj, rule_name, params, budget_low)
    s_disc = (spread / spread_reference) if (spread and spread_reference) else float("nan")

    # S18 §3.5：E_abs 只在有 90 轮真值的格上可算，规程本身未使用该信息
    e_abs = float("nan")
    if reference_budget >= 90:
        errors = []
        for operator in OPERATORS:
            for seed in SEEDS:
                if (operator, seed) not in traj:
                    continue
                estimate = _score(traj, rule_name, params, budget_low, operator, seed)
                if estimate is INADMISSIBLE:
                    continue
                truth = rule_r0(traj[(operator, seed)], reference_budget)
                if truth is not None:
                    errors.append(abs(estimate - truth))
        if errors:
            e_abs = statistics.median(errors)

    return {
        "rule": rule_name, "params": params, "reportable_pairs": reportable,
        "C_adm": reportable / total,
        "F_cross": crossings / reportable if reportable else float("nan"),
        "F_flip": flips / reportable if reportable else float("nan"),
        "M_max": max(moves) if moves else float("nan"),
        "A_fid": fid_match / fid_total if fid_total else float("nan"),
        "S_disc": s_disc, "E_abs": e_abs,
    }


def meets2(row, f_cross_r0):
    """S18 §3.1 至 §3.4：四条同时满足。"""
    if row["reportable_pairs"] == 0:
        return False
    checks = (
        row["F_cross"] <= 0.5 * f_cross_r0,
        row["C_adm"] >= 0.50,
        row["A_fid"] >= 0.90,
        row["S_disc"] >= 0.50,
    )
    return all(c is True or c == True for c in checks) and not any(
        isinstance(v, float) and math.isnan(v)
        for v in (row["F_cross"], row["A_fid"], row["S_disc"]))


CELLS = {
    "design_dc537_unet": ("b90_unet_gr_deepcrack537_*", 90),
    "valid_dc537_rn18":  ("b90_rn18_gr_deepcrack537_*", 90),
    "valid_c500_unet":   ("formal_gn_crack500_unet_*", 30),
    "valid_c500_rn18":   ("g2_rn18_gn_crack500_*", 30),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="baseline_platform/runs")
    parser.add_argument("--cell", choices=sorted(CELLS), required=True)
    parser.add_argument("--budget-low", type=int, required=True)
    parser.add_argument("--budget-high", type=int, required=True)
    parser.add_argument("--only-rule", default=None)
    parser.add_argument("--params", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    pattern, reference_budget = CELLS[args.cell]
    traj = load_trajectories(Path(args.runs_dir), pattern)
    print(f"cell={args.cell}  runs={len(traj)}  "
          f"预算对=({args.budget_low}, {args.budget_high})  参照预算={reference_budget}")

    reference = {}
    for left, right in itertools.combinations(OPERATORS, 2):
        delta = _pair_delta2(traj, "R0", {}, reference_budget, left, right)
        if delta is not None:
            reference[(left, right)] = delta

    spread_reference = _operator_spread(traj, "R0", {}, args.budget_high)
    baseline = evaluate2(traj, "R0", {}, args.budget_low, args.budget_high,
                         reference, reference_budget, spread_reference)
    f_cross_r0 = baseline["F_cross"]
    print(f"R0 基线：F_cross={f_cross_r0:.3f}  达标线 F_cross<={0.5*f_cross_r0:.3f}  "
          f"区分度参照极差={spread_reference:.4f}")
    print()

    header = (f"{'规程':5s} {'超参':48s} {'C_adm':>6s} {'F_cross':>8s} {'F_flip':>7s} "
              f"{'M_max':>7s} {'A_fid':>6s} {'S_disc':>7s} {'E_abs':>7s} {'达标':>4s}")
    print(header); print("-" * len(header))

    rows = [baseline]
    print(f"{'R0':5s} {'(基线)':48s} {baseline['C_adm']:6.3f} {baseline['F_cross']:8.3f} "
          f"{baseline['F_flip']:7.3f} {baseline['M_max']:7.4f} {baseline['A_fid']:6.3f} "
          f"{1.0:7.3f} {baseline['E_abs']:7.4f} {'':>4s}")

    names = [args.only_rule] if args.only_rule else list(RULES2)
    for name in names:
        grid = [json.loads(args.params)] if args.params else GRIDS2[name]
        for params in grid:
            row = evaluate2(traj, name, params, args.budget_low, args.budget_high,
                            reference, reference_budget, spread_reference)
            row["meets"] = meets2(row, f_cross_r0)
            rows.append(row)
            print(f"{name:5s} {json.dumps(params, ensure_ascii=False):48s} "
                  f"{row['C_adm']:6.3f} {row['F_cross']:8.3f} {row['F_flip']:7.3f} "
                  f"{row['M_max']:7.4f} {row['A_fid']:6.3f} {row['S_disc']:7.3f} "
                  f"{row['E_abs']:7.4f} {'达标' if row['meets'] else '—':>4s}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"cell": args.cell, "budget_low": args.budget_low,
             "budget_high": args.budget_high, "reference_budget": reference_budget,
             "f_cross_r0": f_cross_r0, "spread_reference": spread_reference,
             "rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n已写出 {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
