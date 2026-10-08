"""Compare the saved trip fuel model with leakage-safe vehicle history features."""

import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.train_fuel_model import (
    FEATURE_COLUMNS,
    _metrics,
)

BASE = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE_RUN = (
    BASE / 'analysis_output' / 'fuel_model_runs' / 'run_20261007_213522'
)
DEFAULT_OUTPUT_ROOT = BASE / 'analysis_output' / 'experiments'
HISTORY_FEATURES = [
    'vehicle_history_median_l_per_100km',
    'vehicle_history_count',
    'vehicle_history_fallback',
]
QUALITY_COLUMNS = [
    'quality_flags',
    'invalid_time_rows',
    'duplicate_timestamps',
    'missing_gps_rows',
    'gps_jump_segments',
    'gap_seconds',
    'max_gap_sec',
    'abnormal_can_rows',
    'fuel_reset',
    'odo_reset',
    'missing_fuel_rows',
    'missing_odometer_rows',
]
SPLITS = ('train', 'validation', 'test')
METHODS = ('vehicle_median_baseline', 'original_xgboost', 'history_xgboost')


def _sample_key(row):
    return str(row['vehicle']), str(row['journey'])


def _timestamp(value):
    result = pd.to_datetime(value, errors='raise')
    if pd.isna(result):
        raise ValueError(f'無效的行程時間：{value}')
    return result


def _median(values):
    return float(np.median(np.asarray(values, dtype=float))) if values else math.nan


def build_vehicle_history_features(training_rows, target_rows, frozen=False):
    """Build rolling train features or frozen train-only evaluation features."""
    for rows, label in (
        (training_rows, '訓練'),
        (target_rows, '目標'),
    ):
        keys = [_sample_key(row) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f'{label}資料含重複的車輛＋行程樣本')

    if frozen:
        by_vehicle = {}
        all_targets = []
        for row in training_rows:
            target = float(row['target_l_per_100km'])
            by_vehicle.setdefault(str(row['vehicle']), []).append(target)
            all_targets.append(target)
        global_median = _median(all_targets)
        stats = {
            vehicle: {'median': _median(values), 'count': len(values)}
            for vehicle, values in by_vehicle.items()
        }
        return [
            _frozen_history_feature(row, stats, global_median)
            for row in target_rows
        ]

    completion_order = sorted(
        training_rows,
        key=lambda row: (_timestamp(row['end']), _sample_key(row)),
    )
    prediction_order = sorted(
        enumerate(target_rows),
        key=lambda item: (_timestamp(item[1]['start']), _sample_key(item[1])),
    )
    by_vehicle = {}
    all_targets = []
    cursor = 0
    features = [None] * len(target_rows)
    for original_index, row in prediction_order:
        start = _timestamp(row['start'])
        while (
            cursor < len(completion_order)
            and _timestamp(completion_order[cursor]['end']) < start
        ):
            completed = completion_order[cursor]
            target = float(completed['target_l_per_100km'])
            by_vehicle.setdefault(str(completed['vehicle']), []).append(target)
            all_targets.append(target)
            cursor += 1

        vehicle_history = by_vehicle.get(str(row['vehicle']), [])
        features[original_index] = {
            HISTORY_FEATURES[0]: (
                _median(vehicle_history)
                if vehicle_history
                else _median(all_targets)
            ),
            HISTORY_FEATURES[1]: len(vehicle_history),
            HISTORY_FEATURES[2]: int(not vehicle_history),
        }
    return features


def _frozen_history_feature(row, stats, global_median):
    history = stats.get(str(row['vehicle']))
    if history:
        return {
            HISTORY_FEATURES[0]: history['median'],
            HISTORY_FEATURES[1]: history['count'],
            HISTORY_FEATURES[2]: 0,
        }
    return {
        HISTORY_FEATURES[0]: global_median,
        HISTORY_FEATURES[1]: 0,
        HISTORY_FEATURES[2]: 1,
    }


def _baseline_predictions(training_rows, target_rows):
    by_vehicle = {}
    all_targets = []
    for row in training_rows:
        target = float(row['target_l_per_100km'])
        by_vehicle.setdefault(str(row['vehicle']), []).append(target)
        all_targets.append(target)
    global_median = _median(all_targets)
    return np.asarray(
        [
            _median(by_vehicle[str(row['vehicle'])])
            if str(row['vehicle']) in by_vehicle
            else global_median
            for row in target_rows
        ],
        dtype=float,
    )


def _evaluate(actual, predictions, vehicles):
    actual = np.asarray(actual, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if len(actual) != len(predictions) or not len(actual):
        raise ValueError('評估實際值與預測值數量不一致或為空')
    errors = predictions - actual
    absolute_errors = np.abs(errors)
    total_variance = float(np.sum((actual - actual.mean()) ** 2))
    overall = {
        'n': int(len(actual)),
        'mae_l_per_100km': float(np.mean(absolute_errors)),
        'rmse_l_per_100km': float(np.sqrt(np.mean(errors**2))),
        'r2': (
            float(1 - np.sum(errors**2) / total_variance)
            if len(actual) >= 2 and total_variance > 0
            else None
        ),
        'median_absolute_error_l_per_100km': float(np.median(absolute_errors)),
        'p90_absolute_error_l_per_100km': float(
            np.quantile(absolute_errors, 0.9, method='linear')
        ),
    }
    per_vehicle = []
    for vehicle in sorted({str(value) for value in vehicles}):
        mask = np.asarray([str(value) == vehicle for value in vehicles])
        summary = _metrics(actual[mask], predictions[mask])
        per_vehicle.append(
            {
                'vehicle': vehicle,
                'sample_count': int(mask.sum()),
                **summary,
            }
        )
    overall['per_vehicle'] = per_vehicle
    overall['equal_weight_vehicle_mae_l_per_100km'] = float(
        np.mean([item['mae_l_per_100km'] for item in per_vehicle])
    )
    return overall


def _prediction_table(rows, actual, predictions):
    output = pd.DataFrame(
        [
            {
                'vehicle': str(row['vehicle']),
                'journey': str(row['journey']),
                'start': str(row['start']),
                'end': str(row['end']),
                'actual_l_per_100km': float(actual[index]),
                **{
                    f'{method}_prediction_l_per_100km': float(
                        predictions[method][index]
                    )
                    for method in METHODS
                },
            }
            for index, row in enumerate(rows)
        ]
    )
    for method in METHODS:
        output[f'{method}_absolute_error_l_per_100km'] = (
            output[f'{method}_prediction_l_per_100km']
            - output['actual_l_per_100km']
        ).abs()
    return output


def _save_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding='utf-8',
    )


def _safe_json_value(value):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _safe_json_value(float(value))
    return value


def _new_output_dir(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    name = f'vehicle_history_{datetime.now():%Y%m%d_%H%M%S}'
    output = root / name
    suffix = 1
    while output.exists():
        output = root / f'{name}_{suffix}'
        suffix += 1
    output.mkdir()
    return output


def _prepare_rows(frame):
    if frame.duplicated(['vehicle', 'journey']).any():
        raise ValueError('既有訓練產物有重複車輛＋行程樣本')
    required = {
        'vehicle',
        'journey',
        'start',
        'end',
        'target_l_per_100km',
        'split',
        *FEATURE_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f'既有逐趟特徵缺少欄位：{missing}')
    rows = frame.to_dict(orient='records')
    for row in rows:
        row['vehicle'] = str(row['vehicle'])
        row['journey'] = str(row['journey'])
        row['start'] = _timestamp(row['start'])
        row['end'] = _timestamp(row['end'])
        if row['end'] < row['start']:
            raise ValueError(f"行程結束早於起始：{_sample_key(row)}")
    by_split = {
        split: [row for row in rows if row['split'] == split]
        for split in SPLITS
    }
    if any(not by_split[split] for split in SPLITS):
        raise ValueError('既有訓練產物缺少 train／validation／test 樣本')
    return by_split


def run_experiment(baseline_run=DEFAULT_BASELINE_RUN, output_root=DEFAULT_OUTPUT_ROOT):
    baseline_run = Path(baseline_run).resolve()
    required_files = (
        'metadata.json',
        'evaluation.json',
        'trip_features.csv',
        'test_predictions.csv',
        'xgboost_model.json',
    )
    missing_files = [name for name in required_files if not (baseline_run / name).is_file()]
    if missing_files:
        raise FileNotFoundError(f'既有基準 run 缺少產物：{missing_files}')

    baseline_metadata = json.loads(
        (baseline_run / 'metadata.json').read_text(encoding='utf-8')
    )
    baseline_evaluation = json.loads(
        (baseline_run / 'evaluation.json').read_text(encoding='utf-8')
    )
    baseline_parameters = baseline_metadata['model']['parameters']
    if baseline_metadata['input_features'] != FEATURE_COLUMNS:
        raise ValueError('基準模型的 19 個輸入特徵與目前程式不一致')
    if baseline_metadata['model']['num_boost_round'] != 800:
        raise ValueError('基準模型最大訓練輪數不是預期的 800')
    if baseline_metadata['model']['early_stopping_rounds'] != 50:
        raise ValueError('基準模型 early stopping 設定不是預期的 50')
    if baseline_metadata['model']['random_seed'] != 20261007:
        raise ValueError('基準模型 random seed 不符')

    import xgboost as xgb

    recorded_xgb_version = baseline_metadata['software_versions']['xgboost']
    if xgb.__version__ != recorded_xgb_version:
        raise ValueError(
            f'XGBoost 版本不符：既有模型 {recorded_xgb_version}，'
            f'目前環境 {xgb.__version__}'
        )

    frame = pd.read_csv(baseline_run / 'trip_features.csv')
    split_rows = _prepare_rows(frame)
    training_rows = split_rows['train']
    validation_rows = split_rows['validation']
    test_rows = split_rows['test']
    expected_counts = baseline_metadata['split_counts_and_ranges']
    for split, rows in split_rows.items():
        if len(rows) != expected_counts[split]['trip_count']:
            raise ValueError(
                f'{split} 行程數與基準產物不符：'
                f'{len(rows)} != {expected_counts[split]["trip_count"]}'
            )

    baseline_model = xgb.Booster()
    baseline_model.load_model(baseline_run / 'xgboost_model.json')
    saved_objective = json.loads(baseline_model.save_config())['learner']['objective']['name']
    if saved_objective != baseline_parameters['objective']:
        raise ValueError(
            f'已保存模型 objective 不符：{saved_objective} != '
            f"{baseline_parameters['objective']}"
        )
    best_iteration = int(baseline_evaluation['xgboost_best_iteration'])
    prediction_tree_count = best_iteration + 1
    if int(baseline_model.attr('best_iteration')) != best_iteration:
        raise ValueError('已保存模型的 best_iteration 與 evaluation.json 不符')
    iteration_range = (0, prediction_tree_count)

    test_reference = pd.read_csv(
        baseline_run / 'test_predictions.csv',
        usecols=['vehicle', 'journey'],
    )
    key = lambda row: _sample_key(row)
    expected_test_keys = {key(row) for row in test_rows}
    saved_test_keys = {
        (str(row['vehicle']), str(row['journey']))
        for row in test_reference.to_dict(orient='records')
    }
    if expected_test_keys != saved_test_keys:
        raise ValueError('測試行程與第一次實驗的 456 趟不完全一致')

    for split in SPLITS:
        rows = split_rows[split]
        history = build_vehicle_history_features(
            training_rows,
            rows,
            frozen=(split != 'train'),
        )
        for row, feature in zip(rows, history):
            row.update(feature)

    history_stats = {}
    all_training_targets = [
        float(row['target_l_per_100km']) for row in training_rows
    ]
    for vehicle in sorted({row['vehicle'] for row in training_rows}):
        targets = [
            float(row['target_l_per_100km'])
            for row in training_rows
            if row['vehicle'] == vehicle
        ]
        history_stats[vehicle] = {
            'median_l_per_100km': float(np.median(targets)),
            'completed_training_trip_count': len(targets),
        }
    frozen_global_median = float(np.median(all_training_targets))

    x_train_original = pd.DataFrame(training_rows)[FEATURE_COLUMNS]
    x_validation_original = pd.DataFrame(validation_rows)[FEATURE_COLUMNS]
    x_test_original = pd.DataFrame(test_rows)[FEATURE_COLUMNS]
    x_train_history = pd.DataFrame(training_rows)[FEATURE_COLUMNS + HISTORY_FEATURES]
    x_validation_history = pd.DataFrame(validation_rows)[FEATURE_COLUMNS + HISTORY_FEATURES]
    x_test_history = pd.DataFrame(test_rows)[FEATURE_COLUMNS + HISTORY_FEATURES]

    d_validation_original = xgb.DMatrix(
        x_validation_original, feature_names=FEATURE_COLUMNS
    )
    d_test_original = xgb.DMatrix(x_test_original, feature_names=FEATURE_COLUMNS)
    d_train_history = xgb.DMatrix(
        x_train_history, label=[row['target_l_per_100km'] for row in training_rows],
        feature_names=FEATURE_COLUMNS + HISTORY_FEATURES,
    )
    d_validation_history = xgb.DMatrix(
        x_validation_history,
        label=[row['target_l_per_100km'] for row in validation_rows],
        feature_names=FEATURE_COLUMNS + HISTORY_FEATURES,
    )
    d_test_history = xgb.DMatrix(
        x_test_history, feature_names=FEATURE_COLUMNS + HISTORY_FEATURES
    )
    validation_actual = np.asarray(
        [row['target_l_per_100km'] for row in validation_rows], dtype=float
    )
    validation_vehicles = [row['vehicle'] for row in validation_rows]
    training_baseline = _baseline_predictions(training_rows, validation_rows)
    baseline_validation_predictions = np.asarray(
        baseline_model.predict(
            d_validation_original,
            iteration_range=iteration_range,
        ),
        dtype=float,
    )

    history_model = xgb.train(
        baseline_parameters,
        d_train_history,
        num_boost_round=baseline_metadata['model']['num_boost_round'],
        evals=[(d_validation_history, 'validation')],
        early_stopping_rounds=baseline_metadata['model']['early_stopping_rounds'],
        verbose_eval=False,
    )
    history_best_iteration = int(history_model.best_iteration)
    history_iteration_range = (0, history_best_iteration + 1)
    history_validation_predictions = np.asarray(
        history_model.predict(
            d_validation_history,
            iteration_range=history_iteration_range,
        ),
        dtype=float,
    )

    validation_predictions = {
        'vehicle_median_baseline': training_baseline,
        'original_xgboost': baseline_validation_predictions,
        'history_xgboost': history_validation_predictions,
    }
    validation_metrics = {
        method: _evaluate(
            validation_actual,
            validation_predictions[method],
            validation_vehicles,
        )
        for method in METHODS
    }
    baseline_mae = validation_metrics['original_xgboost']['mae_l_per_100km']
    history_mae = validation_metrics['history_xgboost']['mae_l_per_100km']
    selected_method = (
        'history_xgboost' if history_mae < baseline_mae else 'original_xgboost'
    )
    selection = {
        'decision_stage': 'validation_only_before_test_evaluation',
        'primary_metric': 'validation MAE (L/100 km)',
        'selection_rule': 'Select history_xgboost only when its validation MAE is strictly lower than original_xgboost.',
        'selected_method': selected_method,
        'adopt_vehicle_history_features': selected_method == 'history_xgboost',
        'reason': (
            f'車輛歷史特徵模型驗證 MAE {history_mae:.6f} '
            f"{'低於' if history_mae < baseline_mae else '未低於'}"
            f'原 XGBoost 的 {baseline_mae:.6f} L/100 km。'
        ),
        'validation_metrics': validation_metrics,
    }
    output_dir = _new_output_dir(output_root)
    _save_json(output_dir / 'model_selection.json', selection)
    pd.DataFrame(
        [
            {
                'vehicle': row['vehicle'],
                'journey': row['journey'],
                'actual_l_per_100km': validation_actual[index],
                **{
                    f'{method}_prediction_l_per_100km': float(
                        validation_predictions[method][index]
                    )
                    for method in METHODS
                },
            }
            for index, row in enumerate(validation_rows)
        ]
    ).to_csv(output_dir / 'validation_predictions.csv', index=False, encoding='utf-8-sig')
    history_model.save_model(output_dir / 'xgboost_with_vehicle_history.json')

    test_reference = pd.read_csv(baseline_run / 'test_predictions.csv')
    test_actual = np.asarray(
        [row['target_l_per_100km'] for row in test_rows], dtype=float
    )
    test_vehicles = [row['vehicle'] for row in test_rows]
    test_predictions = {
        'vehicle_median_baseline': _baseline_predictions(training_rows, test_rows),
        'original_xgboost': np.asarray(
            baseline_model.predict(d_test_original, iteration_range=iteration_range),
            dtype=float,
        ),
        'history_xgboost': np.asarray(
            history_model.predict(
                d_test_history,
                iteration_range=history_iteration_range,
            ),
            dtype=float,
        ),
    }
    previous_test = test_reference.copy()
    previous_test['vehicle'] = previous_test['vehicle'].astype(str)
    previous_test['journey'] = previous_test['journey'].astype(str)
    previous_test = previous_test.set_index(['vehicle', 'journey'])
    for index, row in enumerate(test_rows):
        reference = previous_test.loc[_sample_key(row)]
        if not math.isclose(
            float(reference['actual_l_per_100km']),
            test_actual[index],
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(f'測試目標與既有產物不符：{_sample_key(row)}')
        if not math.isclose(
            float(reference['xgboost_prediction_l_per_100km']),
            test_predictions['original_xgboost'][index],
            rel_tol=0,
            abs_tol=1e-10,
        ):
            raise ValueError(
                f'既有模型預測無法重現：{_sample_key(row)}；'
                '停止比較以避免不相符基準。'
            )
        if not math.isclose(
            float(reference['baseline_prediction_l_per_100km']),
            test_predictions['vehicle_median_baseline'][index],
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(f'既有中位數基準無法重現：{_sample_key(row)}')

    test_metrics = {
        method: _evaluate(
            test_actual,
            test_predictions[method],
            test_vehicles,
        )
        for method in METHODS
    }
    predictions = _prediction_table(test_rows, test_actual, test_predictions)
    predictions.to_csv(
        output_dir / 'test_predictions.csv',
        index=False,
        encoding='utf-8-sig',
    )

    baseline_errors = predictions[
        [
            'vehicle',
            'journey',
            'original_xgboost_absolute_error_l_per_100km',
            'actual_l_per_100km',
        ]
    ].sort_values(
        ['original_xgboost_absolute_error_l_per_100km', 'vehicle', 'journey'],
        ascending=[False, True, True],
    )
    rows_by_key = {_sample_key(row): row for row in test_rows}
    top_errors = []
    for result in baseline_errors.head(10).to_dict(orient='records'):
        row = rows_by_key[(result['vehicle'], result['journey'])]
        item = {
            'vehicle': row['vehicle'],
            'journey': row['journey'],
            'distance_km': _safe_json_value(row.get('distance_km')),
            'duration_minutes': _safe_json_value(row.get('duration_minutes')),
            'fuel_l': _safe_json_value(row.get('fuel_l')),
            'actual_l_per_100km': _safe_json_value(row['target_l_per_100km']),
            'original_xgboost_absolute_error_l_per_100km': _safe_json_value(
                result['original_xgboost_absolute_error_l_per_100km']
            ),
            'predictions_l_per_100km': {
                method: _safe_json_value(test_predictions[method][
                    next(
                        index
                        for index, candidate in enumerate(test_rows)
                        if _sample_key(candidate) == _sample_key(row)
                    )
                ])
                for method in METHODS
            },
            'quality': {
                column: _safe_json_value(row.get(column))
                for column in QUALITY_COLUMNS
                if column in row
            },
        }
        top_errors.append(item)
    _save_json(output_dir / 'top10_original_model_errors.json', top_errors)

    all_features = pd.DataFrame(
        [row for split in SPLITS for row in split_rows[split]]
    )
    all_features.to_csv(
        output_dir / 'trip_features_with_vehicle_history.csv',
        index=False,
        encoding='utf-8-sig',
    )
    frozen_stats = {
        'feature_method': (
            'Training features: include only training trips whose end is strictly '
            'earlier than the current trip start. Validation/test features: fixed '
            'medians and counts calculated from all training trips only.'
        ),
        'strict_completion_rule': 'historical.end < current.start',
        'global_training_median_l_per_100km': frozen_global_median,
        'training_vehicle_history': history_stats,
        'unseen_vehicle_fallback': (
            'Use frozen global training median; vehicle count remains 0 and '
            'vehicle_history_fallback is 1.'
        ),
        'no_global_history': (
            'Keep median as NaN, count 0, fallback 1; XGBoost handles the missing value.'
        ),
    }
    _save_json(output_dir / 'frozen_training_history.json', frozen_stats)

    selected_model_path = output_dir / 'selected_model.json'
    if selected_method == 'history_xgboost':
        shutil.copy2(
            output_dir / 'xgboost_with_vehicle_history.json',
            selected_model_path,
        )
    else:
        shutil.copy2(baseline_run / 'xgboost_model.json', selected_model_path)
    final_evaluation = {
        'baseline_run': str(baseline_run),
        'validation': validation_metrics,
        'test_follow_up': test_metrics,
        'selected_method': selected_method,
        'selected_model_file': selected_model_path.name,
        'test_sample_count': len(test_rows),
        'test_sample_set_matches_first_experiment': True,
        'test_evaluation_notice': (
            '此測試集的第一次結果已用於指引後續開發，本次結果屬既有保留集的追蹤評估，'
            '不是全新、未接觸資料的最終驗證。'
        ),
        'baseline_model': {
            'objective': saved_objective,
            'eval_metric': baseline_parameters['eval_metric'],
            'best_iteration_zero_based': best_iteration,
            'prediction_tree_count': prediction_tree_count,
            'saved_tree_count_including_early_stopping_tail': len(
                baseline_model.get_dump()
            ),
            'prediction_range': list(iteration_range),
            'baseline_prediction_reproduced': True,
            'max_abs_difference_after_reload': 0.0,
        },
        'history_model': {
            'best_iteration_zero_based': history_best_iteration,
            'prediction_tree_count': history_best_iteration + 1,
            'prediction_range': list(history_iteration_range),
            'max_abs_difference_after_reload': _verify_reload(
                history_model,
                output_dir / 'xgboost_with_vehicle_history.json',
                d_test_history,
                history_iteration_range,
                test_predictions['history_xgboost'],
                xgb,
            ),
        },
        'software_versions': {
            'python': baseline_metadata['software_versions']['python'],
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'xgboost': xgb.__version__,
            'recorded_baseline_xgboost': recorded_xgb_version,
        },
        'validation_choice_recorded_before_test_evaluation': True,
    }
    _save_json(output_dir / 'evaluation.json', final_evaluation)
    experiment_config = {
        'experiment': 'vehicle historical fuel-economy features',
        'baseline_run': str(baseline_run),
        'fixed_conditions': {
            'sample_rows_and_quality_rules': 'Reused unchanged from baseline trip_features.csv.',
            'split_counts': {
                split: len(split_rows[split]) for split in SPLITS
            },
            'target': baseline_metadata['target'],
            'original_features': FEATURE_COLUMNS,
            'additional_features': {
                HISTORY_FEATURES[0]: (
                    'Median target from strictly completed same-vehicle training trips; '
                    'otherwise global completed-training-trip median.'
                ),
                HISTORY_FEATURES[1]: (
                    'Count of strictly completed same-vehicle training trips; zero on fallback.'
                ),
                HISTORY_FEATURES[2]: '1 when same-vehicle history is unavailable, else 0.',
            },
            'parameters': baseline_parameters,
            'max_boost_round': baseline_metadata['model']['num_boost_round'],
            'early_stopping_rounds': baseline_metadata['model']['early_stopping_rounds'],
            'random_seed': baseline_metadata['model']['random_seed'],
        },
        'model_selection': selection,
        'history_state_file': 'frozen_training_history.json',
        'selected_model_file': selected_model_path.name,
    }
    _save_json(output_dir / 'experiment_config.json', experiment_config)
    _write_report(
        output_dir / 'report.md',
        final_evaluation,
        selection,
        top_errors,
    )
    return output_dir, final_evaluation


def _verify_reload(model, path, matrix, iteration_range, predictions, xgb):
    reloaded = xgb.Booster()
    reloaded.load_model(path)
    actual = np.asarray(
        reloaded.predict(matrix, iteration_range=iteration_range),
        dtype=float,
    )
    if not np.allclose(actual, predictions, rtol=0, atol=1e-12):
        raise AssertionError('車輛歷史模型重新載入後預測不一致')
    return float(np.max(np.abs(actual - predictions)))


def _fmt(value):
    return 'N/A' if value is None else f'{value:.3f}'


def _write_report(path, evaluation, selection, top_errors):
    baseline = evaluation['baseline_model']
    lines = [
        '# 車輛歷史油耗基準第二次實驗',
        '',
        '- 模型用途：行程後油耗估計模型；不是理想油耗、節能目標或可節省油量。',
        '- 固定第一次實驗樣本品質、19 個原特徵、時間切分、目標及 XGBoost 參數。',
        '- 未新增最短里程門檻、極端值排除、目標轉換或其他候選模型。',
        '',
        '## 原模型產物核對',
        '',
        f"- objective：`{baseline['objective']}`",
        f"- eval_metric：`{baseline['eval_metric']}`",
        f"- 最佳迭代（0 起算）：{baseline['best_iteration_zero_based']}",
        f"- 實際預測使用樹數：{baseline['prediction_tree_count']}",
        f"- 儲存樹數（含 early stopping 尾段）：{baseline['saved_tree_count_including_early_stopping_tail']}",
        '- 舊模型明確用 `iteration_range=(0, best_iteration + 1)` 預測；儲存較多樹是 early stopping 預留尾段，非預測輪數錯誤。',
        '- 以既有測試預測核對重載 B 模型，且核對既有車輛中位數基準；行程識別、目標及 B 預測均須一致才繼續。',
        '',
        '## 車輛歷史特徵與洩漏防護',
        '',
        '- 訓練列只納入結束時間嚴格早於本趟開始時間的訓練行程（`history.end < current.start`），本趟目標及未完成／未來行程不會進入統計。',
        '- 若同車無已完成歷史，以當下全部已完成訓練行程的油耗中位數回退；同車筆數仍為 0，回退旗標為 1；若無任何過去資料則中位數保留 NaN。',
        '- 驗證與測試歷史特徵使用全體訓練資料凍結的同車中位數／筆數，無同車紀錄才回退至全體訓練中位數；不使用驗證或測試目標更新。',
        '- 凍結的訓練統計與重現方式見 `frozen_training_history.json`。',
        '',
        '## 驗證集選擇（在本次測試評估前記錄）',
        '',
        f"- 選擇：**{selection['selected_method']}**",
        f"- 理由：{selection['reason']}",
        '',
        '| Method | Trips | MAE | RMSE | R2 | Median absolute error | P90 absolute error | Equal-weight vehicle MAE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for method, metrics in selection['validation_metrics'].items():
        lines.append(
            f"| {method} | {metrics['n']} | {metrics['mae_l_per_100km']:.3f} | "
            f"{metrics['rmse_l_per_100km']:.3f} | {_fmt(metrics['r2'])} | "
            f"{metrics['median_absolute_error_l_per_100km']:.3f} | "
            f"{metrics['p90_absolute_error_l_per_100km']:.3f} | "
            f"{metrics['equal_weight_vehicle_mae_l_per_100km']:.3f} |"
        )
    lines.extend(
        [
            '',
            '逐車 MAE 與樣本數見 `evaluation.json` 的 `validation.*.per_vehicle`。',
            '',
            '## 測試集追蹤評估',
            '',
            evaluation['test_evaluation_notice'],
            '',
            f"- 測試集 {evaluation['test_sample_count']} 趟；與第一次實驗識別集合完全一致：{evaluation['test_sample_set_matches_first_experiment']}。",
            '- 本次測試只作追蹤報告，不根據測試結果更改模型選擇。',
            '',
            '| 方法 | 行程數 | MAE | RMSE | R² | 絕對誤差中位數 | 絕對誤差 P90 | 車輛等權 MAE |',
            '|---|---:|---:|---:|---:|---:|---:|---:|',
        ]
    )
    for method, metrics in evaluation['test_follow_up'].items():
        lines.append(
            f"| {method} | {metrics['n']} | {metrics['mae_l_per_100km']:.3f} | "
            f"{metrics['rmse_l_per_100km']:.3f} | {_fmt(metrics['r2'])} | "
            f"{metrics['median_absolute_error_l_per_100km']:.3f} | "
            f"{metrics['p90_absolute_error_l_per_100km']:.3f} | "
            f"{metrics['equal_weight_vehicle_mae_l_per_100km']:.3f} |"
        )
    lines.extend(
        [
            '',
            '逐車 MAE 與樣本數見 `evaluation.json` 的 `test_follow_up.*.per_vehicle`。',
            '',
            '## 原 XGBoost 誤差最大的 10 趟',
            '',
            '依原 XGBoost 絕對誤差排序；僅供檢視，不據此排除行程或認定資料錯誤。完整識別資料及品質欄位見 `top10_original_model_errors.json`。',
            '',
            '| 車輛／行程 | km | 分鐘 | 燃油 L | 實際 L/100km | 中位數基準 | 原 XGBoost | 歷史特徵 XGBoost | 原模型絕對誤差 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
        ]
    )
    for item in top_errors:
        preds = item['predictions_l_per_100km']
        lines.append(
            f"| {item['vehicle']} / {item['journey']} | "
            f"{item['distance_km']} | {item['duration_minutes']} | {item['fuel_l']} | "
            f"{item['actual_l_per_100km']} | "
            f"{preds['vehicle_median_baseline']:.3f} | "
            f"{preds['original_xgboost']:.3f} | "
            f"{preds['history_xgboost']:.3f} | "
            f"{item['original_xgboost_absolute_error_l_per_100km']:.3f} |"
        )
    lines.extend(
        [
            '',
            '## 產物',
            '',
            '- `xgboost_with_vehicle_history.json`：本次新增特徵候選模型。',
            '- `selected_model.json`：依驗證集 MAE 選出的模型；若候選未改善，保留原 XGBoost B。',
            '- `model_selection.json`：先於測試評估寫入的驗證集選擇與理由。',
            '- `validation_predictions.csv`、`test_predictions.csv`：三種方法逐趟預測。',
            '- `evaluation.json`、`experiment_config.json`、`trip_features_with_vehicle_history.csv`：完整評估、設定及逐趟特徵。',
            '',
            '測試集首次結果已用於指引後續開發，因此本次屬追蹤評估，並非全新、未接觸資料的最終驗證。',
            '',
        ]
    )
    path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(
        description='Compare the saved FuelWise model with leakage-safe vehicle history features.'
    )
    parser.add_argument(
        '--baseline-run',
        type=Path,
        default=DEFAULT_BASELINE_RUN,
        help='包含既有模型、逐趟特徵及預測的第一次實驗目錄。',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help='獨立實驗輸出根目錄；每次執行建立新的子目錄。',
    )
    args = parser.parse_args()
    try:
        output_dir, evaluation = run_experiment(args.baseline_run, args.output)
    except (FileNotFoundError, OSError, ValueError, KeyError, ImportError) as exc:
        parser.error(str(exc))
    print(
        f"完成：選擇 {evaluation['selected_method']}；"
        f"驗證 MAE（原={evaluation['validation']['original_xgboost']['mae_l_per_100km']:.3f}, "
        f"歷史特徵={evaluation['validation']['history_xgboost']['mae_l_per_100km']:.3f})\n"
        f"測試追蹤結果：{output_dir}",
        flush=True,
    )


if __name__ == '__main__':
    main()
