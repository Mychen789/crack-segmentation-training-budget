"""S23: post hoc, descriptive leave-one-operator-out sensitivity analysis.

ANALYSIS DEFINITION (fixed in this script before generation, 2026-11-12):
Use only existing train/validation records. Never load test data or checkpoints.
For each of the four main settings, compute a run's read-out as the maximum
complete trailing five-epoch average of foreground validation IoU within the
budget. Use budgets (30, 90) for DeepCrack537 and (15, 30) for CRACK500. Average
the three seed read-outs for each operator before subtracting low from high.
For cosine schedules, use the separately trained 30- and 90-epoch runs.
For ConvNeXt-UNet, truncate the existing 90-epoch runs at 30 and 90 epochs.

For all 11 distinct operators, and then after omitting each operator in turn,
report the strict count |gain(a)-gain(b)| > 0.005 over all unordered remaining
pairs, denominator, Fcross, maximum absolute shift Mmax, and gain mean/range.
Also repeat the strongest main setting separately for each observed seed.
Do not refit rules, select thresholds, resample data, or perform hypothesis tests.
These are descriptive sensitivities conditional on the observed runs, seeds,
settings and operator set. They are neither new confirmatory results nor
population-inference confidence intervals.

Independent implementation: standard-library CSV reader and explicit five-value
means, without importing existing plotting or analysis code. Main-setting full
results are checked against data/tables/table3_budget_sensitivity.csv. Per-
operator gains are also checked against table5_lift_decomposition.csv.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'tables'
OPS = ('max', 'avg', 'min', 'avg_top2', 'fixed_max_avg', 'learned_max_avg',
       't_max_avg', 'v0_max_mean_min', 'btsp', 'diag_partition_matched', 'stride_conv')
SEEDS = (2026, 2027, 2028)
DELTA = 0.005
MAIN = {
    'unet_deepcrack537_90': (30, 90),
    'rn18_deepcrack537_90': (30, 90),
    'unet_crack500_30': (15, 30),
    'rn18_crack500_30': (15, 30),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def ma5_readout(values, budget):
    assert len(values) >= budget >= 5
    assert all(math.isfinite(v) for v in values[:budget])
    # Complete windows only, each ending no later than the selected budget.
    return max(math.fsum(values[end - 5:end]) / 5 for end in range(5, budget + 1))


def constant_readouts():
    raw = ROOT / 'data' / 'per_epoch_validation.csv'
    grouped = defaultdict(dict)
    for row in read_csv(raw):
        cell, op, seed = row['cell'], row['pooling'], int(row['seed'])
        if cell not in MAIN or op not in OPS or seed not in SEEDS:
            continue
        key = cell, op, seed
        epoch = int(row['epoch'])
        assert epoch not in grouped[key], ('duplicate epoch', key, epoch)
        grouped[key][epoch] = float(row['foreground_iou'])
    settings = {}
    for cell, (low, high) in MAIN.items():
        readings = {}
        for op in OPS:
            for seed in SEEDS:
                epochs = grouped[(cell, op, seed)]
                assert sorted(epochs) == list(range(1, high + 1)), (cell, op, seed)
                curve = [epochs[e] for e in range(1, high + 1)]
                readings[(op, seed)] = (ma5_readout(curve, low), ma5_readout(curve, high))
        settings[cell] = {'budgets': (low, high), 'readings': readings}
    return settings, [raw]


def read_validation_history(path):
    run = json.loads(path.read_text(encoding='utf-8'))
    assert run['status'] == 'COMPLETE', path
    assert not run.get('test_or_external_test_used', False), path
    history = run['history']
    assert [int(x['epoch']) for x in history] == list(range(1, len(history) + 1)), path
    return [float(x['validation']['foreground_iou']) for x in history]


def add_supplementary(settings, source_paths):
    cosine = ROOT / 'data' / 'runs-cosine' / 's21_collect'
    readings = {}
    for op in OPS:
        for seed in SEEDS:
            values = []
            for budget in (30, 90):
                path = cosine / f'cos{budget}_unet_gr_deepcrack537_{op}_s{seed}.json'
                source_paths.append(path)
                curve = read_validation_history(path)
                assert len(curve) == budget, path
                values.append(ma5_readout(curve, budget))
            readings[(op, seed)] = tuple(values)
    settings['unet_deepcrack537_cosine'] = {'budgets': (30, 90), 'readings': readings}
    readings = {}
    for op in OPS:
        for seed in SEEDS:
            path = ROOT / 'data' / 'runs_s22' / f'b90_cnx_gr_deepcrack537_{op}_s{seed}.json'
            source_paths.append(path)
            curve = read_validation_history(path)
            assert len(curve) == 90, path
            readings[(op, seed)] = (ma5_readout(curve, 30), ma5_readout(curve, 90))
    settings['cnx_deepcrack537_90'] = {'budgets': (30, 90), 'readings': readings}


def aggregate(readings, seeds):
    low = {o: math.fsum(readings[(o, s)][0] for s in seeds) / len(seeds) for o in OPS}
    high = {o: math.fsum(readings[(o, s)][1] for s in seeds) / len(seeds) for o in OPS}
    return low, high


def statistics(gains, operators):
    shifts = [abs(gains[a] - gains[b]) for a, b in combinations(operators, 2)]
    n_cross = sum(x > DELTA for x in shifts)
    gain_values = [gains[o] for o in operators]
    mmax = max(shifts)
    gain_range = max(gain_values) - min(gain_values)
    assert math.isclose(mmax, gain_range, rel_tol=0, abs_tol=1e-15)
    return {'n_operators': len(operators), 'n_pairs': len(shifts), 'n_cross': n_cross,
            'Fcross': n_cross / len(shifts), 'Mmax': mmax,
            'gain_mean': math.fsum(gain_values) / len(gain_values),
            'gain_min': min(gain_values), 'gain_max': max(gain_values), 'gain_range': gain_range}


def omission_rows(setting, data, seeds):
    low, high = aggregate(data['readings'], seeds)
    gains = {op: high[op] - low[op] for op in OPS}
    rows = []
    for omitted in (None, *OPS):
        keep = tuple(op for op in OPS if op != omitted)
        rows.append({'setting': setting, 'seed_scope': 'mean_3' if len(seeds) == 3 else str(seeds[0]),
                     'budget_low': data['budgets'][0], 'budget_high': data['budgets'][1],
                     'omitted_operator': omitted or 'none', **statistics(gains, keep)})
    return rows, gains


def verify_full(rows, gains_by_setting):
    table3 = ROOT / 'data' / 'tables' / 'table3_budget_sensitivity.csv'
    table5 = ROOT / 'data' / 'tables' / 'table5_lift_decomposition.csv'
    full = {row['setting']: row for row in rows if row['omitted_operator'] == 'none'}
    checks = []
    for old in read_csv(table3):
        key = old['cell']
        if key not in MAIN:
            continue
        new = full[key]
        assert new['n_pairs'] == int(old['n_pairs'])
        assert new['n_cross'] == int(old['n_cross_0.005'])
        assert abs(new['Fcross'] - float(old['F_cross'])) <= 0.000051
        assert abs(new['Mmax'] - float(old['M_max'])) <= 0.00000051
        checks.append({'setting': key, 'reference': table3.relative_to(ROOT).as_posix(), 'status': 'matched'})
    for old in read_csv(table5):
        key, op = old['cell'], old['pooling']
        if key in MAIN and op in OPS:
            assert abs(gains_by_setting[key][op] - float(old['lift'])) <= 0.00000051, (key, op)
    assert len(checks) == 4
    # Existing manuscript-reported independent replications.
    assert full['unet_deepcrack537_cosine']['n_cross'] == 27
    assert full['cnx_deepcrack537_90']['n_cross'] == 2
    return checks, [table3, table5]


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    settings, source_paths = constant_readouts()
    add_supplementary(settings, source_paths)
    rows, gains = [], {}
    for key, data in settings.items():
        current, gains[key] = omission_rows(key, data, SEEDS)
        rows.extend(current)
    checks, references = verify_full(rows, gains)
    per_seed = []
    strongest = 'unet_deepcrack537_90'
    for seed in SEEDS:
        current, _ = omission_rows(strongest, settings[strongest], (seed,))
        per_seed.extend(current)
    summaries = []
    for key in settings:
        current = [r for r in rows if r['setting'] == key]
        loo = [r for r in current if r['omitted_operator'] != 'none']
        summaries.append({'setting': key, 'full': current[0],
                          'without_min': next(r for r in loo if r['omitted_operator'] == 'min'),
                          'leave_one_out_n_cross_range': [min(r['n_cross'] for r in loo), max(r['n_cross'] for r in loo)],
                          'leave_one_out_Fcross_range': [min(r['Fcross'] for r in loo), max(r['Fcross'] for r in loo)],
                          'leave_one_out_Mmax_range': [min(r['Mmax'] for r in loo), max(r['Mmax'] for r in loo)]})
    payload = {
        'analysis_id': 'S23_operator_omission',
        'analysis_type': 'post hoc descriptive sensitivity conditional on observed validation runs',
        'definition': __doc__,
        'threshold': DELTA, 'seeds': SEEDS, 'operators': OPS,
        'script_sha256': sha256(Path(__file__)),
        'input_files': [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha256(p)}
                        for p in source_paths + references],
        'reference_checks': checks,
        'summaries': summaries,
        'mean_three_seed_results': rows,
        'strongest_setting_per_seed_results': per_seed,
        'gains_by_setting': gains,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 's23_operator_omission.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write_csv(OUT / 's23_operator_omission.csv', rows)
    write_csv(OUT / 's23_operator_omission_per_seed.csv', per_seed)
    for summary in summaries:
        full, no_min = summary['full'], summary['without_min']
        print(f"{summary['setting']}: full {full['n_cross']}/{full['n_pairs']}; "
              f"without min {no_min['n_cross']}/{no_min['n_pairs']}; "
              f"all LOO count range {summary['leave_one_out_n_cross_range']}, "
              f"Fcross range {summary['leave_one_out_Fcross_range']}")
    print('Reference checks matched all four main settings and their per-operator gains.')


if __name__ == '__main__':
    main()
