"""Applicability-filtered model C experiment (rule applicability-v1).

Keeps the original time split and every original artefact.  The only change
for the new model is *which trips are eligible* (train / validation / test and
the vehicle-history source).  Features, hyper-parameters, rounds and early
stopping are copied from the original C run.  Nothing here tunes thresholds.
"""

import argparse
import json
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import analyze
from analysis.train_fuel_model import (
    FEATURE_COLUMNS,
    find_existing_index,
)
from analysis.prepare_dashboard_model import _api_metric
from analysis.trip_applicability import (
    APPLICABILITY_CONFIG,
    EXCLUSION_CODES,
    RULE_VERSION,
    evaluate_applicability,
    flatten_applicability,
)
from analysis.vehicle_history_experiment import (
    DEFAULT_BASELINE_RUN,
    DEFAULT_OUTPUT_ROOT,
    HISTORY_FEATURES,
    SPLITS,
    _baseline_predictions,
    _evaluate,
    _prepare_rows,
    _safe_json_value,
    _sample_key,
    build_vehicle_history_features,
)

BASE = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_EXPERIMENT = DEFAULT_OUTPUT_ROOT / 'vehicle_history_20261008_191019'
# Fixed in advance; applied only to eligible (>= 5 km) trips.
DISTANCE_GROUPS = ((5.0, 10.0), (10.0, 20.0), (20.0, 50.0), (50.0, float('inf')))


def _group_label(low, high):
    return f'{low:g}+ km' if high == float('inf') else f'{low:g}-{high:g} km'


def compute_applicability(index_path, keys):
    """Analyse each requested trip from the read-only index."""
    results = {}
    with sqlite3.connect(f'file:{Path(index_path).as_posix()}?mode=ro', uri=True) as db:
        for ordinal, (vehicle, journey) in enumerate(sorted(keys), 1):
            raw = [
                json.loads(item[0])
                for item in db.execute(
                    'SELECT raw FROM points WHERE vehicle=? AND journey=? ORDER BY seq',
                    (vehicle, journey),
                )
            ]
            analysis = analyze(raw, detail=True, feature_only=True)
            decision = evaluate_applicability(analysis)
            decision['fuel_l'] = analysis['summary'].get('fuel_l')
            results[(vehicle, journey)] = decision
            if ordinal % 500 == 0:
                print(f'applicability {ordinal}/{len(keys)}', flush=True)
    return results


def label_counts(decisions):
    """Per-label counts plus a de-duplicated exclusion total (labels overlap)."""
    counts = {code: 0 for code in EXCLUSION_CODES}
    excluded = 0
    for decision in decisions:
        reasons = decision['exclusion_reasons']
        for reason in reasons:
            counts[reason] += 1
        excluded += bool(reasons)
    return {
        'per_reason_counts_overlapping': counts,
        'excluded_unique_trips': excluded,
        'eligible_trips': len(decisions) - excluded,
        'total_trips': len(decisions),
    }


def retention(rows, decisions):
    kept = [d['model_eligible'] for d in decisions]

    def total(key, mask):
        return float(sum(float(d[key] or 0) for d, k in zip(decisions, mask) if k))

    mask_all = [True] * len(rows)
    return {
        'trips_total': len(rows),
        'trips_kept': int(sum(kept)),
        'trip_retention': float(np.mean(kept)) if kept else None,
        'distance_km_total': total('distance_km', mask_all),
        'distance_km_kept': total('distance_km', kept),
        'fuel_l_total': total('fuel_l', mask_all),
        'fuel_l_kept': total('fuel_l', kept),
    }


def _ratio(a, b):
    return a / b if b else None


def breakdown_by_group(rows, actual, prediction_sets):
    table = []
    distances = np.asarray([float(r['distance_km']) for r in rows])
    for low, high in DISTANCE_GROUPS:
        mask = (distances >= low) & (distances < high)
        entry = {'group': _group_label(low, high), 'n': int(mask.sum())}
        for name, pred in prediction_sets.items():
            entry[f'{name}_mae'] = (
                float(np.mean(np.abs(pred[mask] - actual[mask]))) if mask.any() else None
            )
        table.append(entry)
    return table


def breakdown_by_vehicle(rows, actual, prediction_sets):
    vehicles = np.asarray([r['vehicle'] for r in rows])
    table = []
    for vehicle in sorted(set(vehicles)):
        mask = vehicles == vehicle
        entry = {'vehicle': vehicle, 'n': int(mask.sum())}
        for name, pred in prediction_sets.items():
            entry[f'{name}_mae'] = float(np.mean(np.abs(pred[mask] - actual[mask])))
        table.append(entry)
    return table


def run(index_path=None, source_experiment=DEFAULT_SOURCE_EXPERIMENT,
        baseline_run=DEFAULT_BASELINE_RUN, output_root=DEFAULT_OUTPUT_ROOT):
    import xgboost as xgb

    source = Path(source_experiment)
    metadata = json.loads((Path(baseline_run) / 'metadata.json').read_text(encoding='utf-8'))
    parameters = metadata['model']['parameters']
    rounds = metadata['model']['num_boost_round']
    early = metadata['model']['early_stopping_rounds']

    split_rows = _prepare_rows(
        pd.read_csv(source / 'trip_features_with_vehicle_history.csv')
    )
    keys = {_sample_key(r) for s in SPLITS for r in split_rows[s]}
    index_file, _info = find_existing_index(index_path)
    decisions = compute_applicability(index_file, keys)

    for split in SPLITS:
        for row in split_rows[split]:
            row['applicability'] = decisions[_sample_key(row)]

    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = f'applicability_{RULE_VERSION}_{datetime.now():%Y%m%d_%H%M%S}'
    out = out_root / stamp
    out.mkdir()

    # Eligible subsets (original split assignment kept as-is).
    eligible = {
        s: [r for r in split_rows[s] if r['applicability']['model_eligible']]
        for s in SPLITS
    }
    if any(not eligible[s] for s in SPLITS):
        raise ValueError('some split has no eligible trips')

    # New history features from eligible past training trips only.
    for split in SPLITS:
        history = build_vehicle_history_features(
            eligible['train'], eligible[split], frozen=(split != 'train')
        )
        for row, feat in zip(eligible[split], history):
            row['new_' + HISTORY_FEATURES[0]] = feat[HISTORY_FEATURES[0]]
            row['new_' + HISTORY_FEATURES[1]] = feat[HISTORY_FEATURES[1]]
            row['new_' + HISTORY_FEATURES[2]] = feat[HISTORY_FEATURES[2]]

    def matrix(rows, label=True):
        frame = pd.DataFrame(rows)
        x = frame[FEATURE_COLUMNS].copy()
        for name in HISTORY_FEATURES:
            x[name] = frame['new_' + name]
        return xgb.DMatrix(
            x,
            label=frame['target_l_per_100km'] if label else None,
            feature_names=FEATURE_COLUMNS + HISTORY_FEATURES,
        )

    d_train, d_val, d_test = (
        matrix(eligible['train']), matrix(eligible['validation']), matrix(eligible['test'])
    )
    model = xgb.train(
        parameters, d_train, num_boost_round=rounds,
        evals=[(d_val, 'validation')], early_stopping_rounds=early, verbose_eval=False,
    )
    best = int(model.best_iteration)
    rng = (0, best + 1)
    model.save_model(out / 'xgboost_applicable_c.json')

    # Original predictions for the SAME eligible rows.
    results = {}
    predictions = {}
    for split in ('validation', 'test'):
        csv = pd.read_csv(source / f'{split}_predictions.csv')
        csv['k'] = list(zip(csv.vehicle.astype(str), csv.journey.astype(str)))
        csv_map = {k: rec for k, rec in zip(csv['k'], csv.to_dict('records'))}
        rows = eligible[split]
        lookup = [csv_map[_sample_key(r)] for r in rows]
        actual = np.asarray([r['target_l_per_100km'] for r in rows], float)
        if not np.allclose(actual, [x['actual_l_per_100km'] for x in lookup]):
            raise ValueError('target mismatch with original predictions')
        d = d_val if split == 'validation' else d_test
        preds = {
            'original_A': np.asarray([x['vehicle_median_baseline_prediction_l_per_100km'] for x in lookup], float),
            'new_A': _baseline_predictions(eligible['train'], rows),
            'original_C': np.asarray([x['history_xgboost_prediction_l_per_100km'] for x in lookup], float),
            'new_C': np.asarray(model.predict(d, iteration_range=rng), float),
        }
        vehicles = [r['vehicle'] for r in rows]
        predictions[split] = (rows, actual, preds)
        results[split] = {
            'n': len(rows),
            'methods': {n: _evaluate(actual, p, vehicles) for n, p in preds.items()},
            'by_vehicle': breakdown_by_vehicle(rows, actual, preds),
            'by_distance_group': breakdown_by_group(rows, actual, preds),
        }
        pd.DataFrame(
            {
                'vehicle': vehicles,
                'journey': [r['journey'] for r in rows],
                'actual_l_per_100km': actual,
                **{f'{n}_prediction': p for n, p in preds.items()},
            }
        ).to_csv(out / f'{split}_eligible_predictions.csv', index=False)

        # Analysis A: original C on all trips vs eligible subset.
        full = csv.reset_index(drop=True)
        full_actual = full.actual_l_per_100km.to_numpy(float)
        full_pred = full.history_xgboost_prediction_l_per_100km.to_numpy(float)
        elig_pred = preds['original_C']
        results[split]['scope_change_original_C'] = {
            'all_trips': _evaluate(full_actual, full_pred, full.vehicle.astype(str).tolist()),
            'eligible_subset': results[split]['methods']['original_C'],
        }

    # Retention / label counts per split.
    summary = {}
    for split in SPLITS:
        decs = [r['applicability'] for r in split_rows[split]]
        summary[split] = {
            'retention': retention(split_rows[split], decs),
            'labels': label_counts(decs),
        }
    all_decs = [r['applicability'] for s in SPLITS for r in split_rows[s]]
    summary['all'] = {
        'retention': retention(
            [r for s in SPLITS for r in split_rows[s]], all_decs
        ),
        'labels': label_counts(all_decs),
        'label_counts_including_eligible': {
            code: sum(code in d['labels'] for d in all_decs)
            for code in (*EXCLUSION_CODES, 'eligible_transport')
        },
    }
    for entry in summary.values():
        r = entry['retention']
        r['distance_retention'] = _ratio(r['distance_km_kept'], r['distance_km_total'])
        r['fuel_retention'] = _ratio(r['fuel_l_kept'], r['fuel_l_total'])

    val_new = results['validation']['methods']['new_C']['mae_l_per_100km']
    val_old = results['validation']['methods']['original_C']['mae_l_per_100km']
    decision = {
        'rule': 'update dashboard model only if new C validation MAE < original C validation MAE on the same eligible validation trips; test is never used',
        'validation_mae_original_C': val_old,
        'validation_mae_new_C': val_new,
        'update_dashboard_model': bool(val_new < val_old),
    }

    # Frozen eligible-only history for later inference use.
    frozen = {}
    for vehicle in sorted({r['vehicle'] for r in eligible['train']}):
        values = [r['target_l_per_100km'] for r in eligible['train'] if r['vehicle'] == vehicle]
        frozen[vehicle] = {
            'median_l_per_100km': float(np.median(values)),
            'completed_training_trip_count': len(values),
        }
    payload = {
        'rule_version': RULE_VERSION,
        'config': {k: _safe_json_value(v) for k, v in APPLICABILITY_CONFIG.items()},
        'source_experiment': str(source),
        'baseline_run': str(baseline_run),
        'new_model': {
            'best_iteration': best,
            'prediction_tree_count': best + 1,
            'parameters': parameters,
            'num_boost_round': rounds,
            'early_stopping_rounds': early,
        },
        'summary': summary,
        'results': results,
        'decision': decision,
        'frozen_eligible_training_history': {
            'global_training_median_l_per_100km': float(
                np.median([r['target_l_per_100km'] for r in eligible['train']])
            ),
            'training_vehicle_history': frozen,
        },
        'notes': [
            'Test split is a follow-up tracking evaluation, not a clean holdout.',
            'Original C predictions are read from saved CSVs and use original history.',
        ],
    }
    (out / 'evaluation.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=_safe_json_value),
        encoding='utf-8',
    )
    pd.DataFrame(
        [
            {
                'split': s, 'vehicle': r['vehicle'], 'journey': r['journey'],
                'start': r['start'], 'end': r['end'],
                'distance_km': r['distance_km'], 'fuel_l': r['applicability']['fuel_l'],
                **flatten_applicability(r['applicability']),
            }
            for s in SPLITS for r in split_rows[s]
        ]
    ).to_csv(out / 'trip_applicability.csv', index=False)
    bundle_dir = out / 'dashboard_bundle'
    bundle_dir.mkdir()
    shutil.copyfile(out / 'xgboost_applicable_c.json', bundle_dir / 'model.json')
    (bundle_dir / 'frozen_training_history.json').write_text(
        json.dumps(
            {
                'feature_method': 'Eligible (applicability rule) completed training trips only; '
                'training rows use trips ending strictly before the current start, '
                'validation/test and new trips use frozen eligible-training medians.',
                'strict_completion_rule': 'historical.end < current.start',
                'rule_version': RULE_VERSION,
                **payload['frozen_eligible_training_history'],
            },
            ensure_ascii=False, indent=2, allow_nan=False,
        ),
        encoding='utf-8',
    )
    pd.DataFrame(
        [
            {
                'vehicle': r['vehicle'], 'journey': r['journey'], 'split': s,
                **{c: r.get(c) for c in FEATURE_COLUMNS},
                **{h: r['new_' + h] for h in HISTORY_FEATURES},
            }
            for s in SPLITS for r in eligible[s]
        ]
    ).to_csv(bundle_dir / 'trip_features_with_vehicle_history.csv', index=False)
    import xgboost
    (bundle_dir / 'bundle.json').write_text(
        json.dumps(
            {
                'model_method': 'applicable_history_xgboost',
                'model_target': 'l_per_100km',
                'model_file': 'model.json',
                'prediction_tree_count': best + 1,
                'feature_names': FEATURE_COLUMNS + HISTORY_FEATURES,
                'xgboost_version': xgboost.__version__,
                'model_role': 'experimental_trip_completion_estimate',
                'rule_version': RULE_VERSION,
                'applicability_config': {
                    k: _safe_json_value(v) for k, v in APPLICABILITY_CONFIG.items()
                },
                'validation_metrics': {
                    'l_per_100km': _api_metric(results['validation']['methods']['new_C'])
                },
                'comparison_validation_mae_l_per_100km': {
                    'original_A': results['validation']['methods']['original_A']['mae_l_per_100km'],
                    'new_A': results['validation']['methods']['new_A']['mae_l_per_100km'],
                    'original_C': results['validation']['methods']['original_C']['mae_l_per_100km'],
                    'new_C': results['validation']['methods']['new_C']['mae_l_per_100km'],
                },
                'test_follow_up_metrics': {
                    'l_per_100km': _api_metric(results['test']['methods']['new_C'])
                },
                'selection_reason': (
                    '適用範圍篩選後重訓的 C 模型；依相同合格驗證行程的 MAE 優於原 C 而採用，'
                    '未使用測試結果。'
                ),
                'shap_method': 'XGBoost exact TreeSHAP',
                'shap_background': 'Path-dependent',
                'split_counts': {s: len(eligible[s]) for s in SPLITS},
            },
            ensure_ascii=False, indent=2, allow_nan=False,
        ),
        encoding='utf-8',
    )
    print(json.dumps({'output': str(out), 'decision': decision}, ensure_ascii=False))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index')
    args = parser.parse_args()
    run(args.index)


if __name__ == '__main__':
    main()
