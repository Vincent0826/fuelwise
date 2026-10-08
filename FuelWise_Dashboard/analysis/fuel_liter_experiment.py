"""Evaluate one total-fuel target candidate against the saved L/100 km models."""

import json
import math
import sqlite3
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import analyze
from analysis.train_fuel_model import FEATURE_COLUMNS
from analysis.vehicle_history_experiment import HISTORY_FEATURES

BASE = Path(__file__).resolve().parent.parent
HISTORY_EXPERIMENT = (
    BASE / 'analysis_output' / 'experiments' / 'vehicle_history_20261008_172441'
)
BASELINE_RUN = BASE / 'analysis_output' / 'fuel_model_runs' / 'run_20261007_213522'
EXPERIMENT_ROOT = BASE / 'analysis_output' / 'experiments'
DASHBOARD_BUNDLE = EXPERIMENT_ROOT / 'dashboard_model'
METHODS = (
    'vehicle_median_baseline',
    'original_xgboost',
    'history_xgboost',
    'total_fuel_xgboost',
)
DISTANCE_BANDS = (
    ('0 < distance < 1 km', lambda distance: (distance > 0) & (distance < 1)),
    ('1 <= distance < 5 km', lambda distance: (distance >= 1) & (distance < 5)),
    ('5 <= distance < 20 km', lambda distance: (distance >= 5) & (distance < 20)),
    ('distance >= 20 km', lambda distance: distance >= 20),
)


def _safe(value):
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def _write_json(path, value):
    path.write_text(
        json.dumps(_safe(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding='utf-8',
    )


def _metrics(actual, predicted, vehicles):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    absolute = np.abs(error)
    total_variance = float(np.sum((actual - actual.mean()) ** 2))
    per_vehicle = []
    for vehicle in sorted({str(value) for value in vehicles}):
        mask = np.asarray([str(value) == vehicle for value in vehicles])
        per_vehicle.append(
            {
                'vehicle': vehicle,
                'sample_count': int(mask.sum()),
                'mae': float(np.mean(absolute[mask])),
            }
        )
    return {
        'n': int(len(actual)),
        'mae': float(np.mean(absolute)),
        'rmse': float(np.sqrt(np.mean(error**2))),
        'r2': (
            float(1 - np.sum(error**2) / total_variance)
            if len(actual) > 1 and total_variance > 0
            else None
        ),
        'median_absolute_error': float(np.median(absolute)),
        'p90_absolute_error': float(np.quantile(absolute, 0.9, method='linear')),
        'equal_weight_vehicle_mae': float(
            np.mean([row['mae'] for row in per_vehicle])
        ),
        'per_vehicle': per_vehicle,
    }


def _load_experiment_rows():
    features = pd.read_csv(HISTORY_EXPERIMENT / 'trip_features_with_vehicle_history.csv')
    predictions = {
        split: pd.read_csv(HISTORY_EXPERIMENT / f'{split}_predictions.csv')
        for split in ('validation', 'test')
    }
    features['vehicle'] = features['vehicle'].astype(str)
    features['journey'] = features['journey'].astype(str)
    for split, predicted in predictions.items():
        predicted['vehicle'] = predicted['vehicle'].astype(str)
        predicted['journey'] = predicted['journey'].astype(str)
        actual = features.loc[features['split'] == split]
        if set(zip(actual.vehicle, actual.journey)) != set(
            zip(predicted.vehicle, predicted.journey)
        ):
            raise ValueError(f'{split} 樣本與已保存 A/B/C 預測不一致')
        predictions[split] = actual[['vehicle', 'journey']].merge(
            predicted,
            on=['vehicle', 'journey'],
            how='left',
            validate='one_to_one',
            sort=False,
        )
    return features, predictions


def _distance_diagnostics(features, predictions, train_predictions):
    report = {}
    for split in ('train', 'validation', 'test'):
        rows = features.loc[features['split'] == split].copy()
        saved_predictions = predictions.get(split)
        if saved_predictions is not None:
            rows = rows.merge(
                saved_predictions,
                on=['vehicle', 'journey'],
                how='left',
                validate='one_to_one',
            )
        else:
            rows = rows.merge(
                train_predictions,
                on=['vehicle', 'journey'],
                how='left',
                validate='one_to_one',
            )
        groups = []
        for name, mask_fn in DISTANCE_BANDS:
            distance = rows['distance_km'].to_numpy(dtype=float)
            mask = mask_fn(distance)
            group = rows.loc[mask]
            fuel = group['fuel_l'].to_numpy(dtype=float)
            l100 = group['target_l_per_100km'].to_numpy(dtype=float)
            result = {
                'distance_band': name,
                'trip_count': int(len(group)),
                'split_share': float(len(group) / len(rows)) if len(rows) else None,
                'total_fuel_l': float(np.sum(fuel)),
                'fuel_median_l': float(np.median(fuel)) if len(fuel) else None,
                'fuel_p90_l': float(np.quantile(fuel, 0.9)) if len(fuel) else None,
                'fuel_max_l': float(np.max(fuel)) if len(fuel) else None,
                'l_per_100km_median': float(np.median(l100)) if len(l100) else None,
                'l_per_100km_p90': float(np.quantile(l100, 0.9)) if len(l100) else None,
                'l_per_100km_max': float(np.max(l100)) if len(l100) else None,
                'duration_median_minutes': (
                    float(group['duration_minutes'].median()) if len(group) else None
                ),
                'duration_p90_minutes': (
                    float(group['duration_minutes'].quantile(0.9)) if len(group) else None
                ),
                'idle_share_median_pct': (
                    float(group['idle_engine_share_pct'].median()) if len(group) else None
                ),
                'idle_share_p90_pct': (
                    float(group['idle_engine_share_pct'].quantile(0.9))
                    if len(group)
                    else None
                ),
            }
            for method in METHODS:
                error_column = f'{method}_error_l_per_100km'
                if error_column in group:
                    absolute_error = group[error_column].abs().to_numpy(dtype=float)
                else:
                    prediction_column = f'{method}_prediction_l_per_100km'
                    absolute_error = np.abs(
                        group[prediction_column].to_numpy(dtype=float) - l100
                    )
                result[f'{method}_mae_l_per_100km'] = (
                    float(np.mean(absolute_error)) if len(absolute_error) else None
                )
                result[f'{method}_absolute_error_sum_l_per_100km'] = float(
                    np.sum(absolute_error)
                )
            for method in METHODS:
                total = float(
                    np.sum(
                        np.abs(
                            rows[f'{method}_prediction_l_per_100km'].to_numpy(dtype=float)
                            - rows['target_l_per_100km'].to_numpy(dtype=float)
                        )
                    )
                )
                result[f'{method}_absolute_error_share'] = (
                    result[f'{method}_absolute_error_sum_l_per_100km'] / total
                    if total
                    else None
                )
            groups.append(result)
        report[split] = groups
    return report


def _counter_update_diagnostics(training, index_path):
    path = Path(index_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'既有 SQLite 索引不存在：{path}')
    examples = (
        (
            '約 0.1 km 且觀測到正向燃油增量',
            (training['distance_km'].between(0.09, 0.11))
            & (training['fuel_l'] > 0),
        ),
        (
            '1 至 5 km',
            (training['distance_km'] >= 1) & (training['distance_km'] < 5),
        ),
        (
            '5 至 20 km',
            (training['distance_km'] >= 5) & (training['distance_km'] < 20),
        ),
        ('至少 20 km', training['distance_km'] >= 20),
    )
    selected = []
    for label, mask in examples:
        candidates = training.loc[mask].sort_values(
            ['trip_id', 'vehicle', 'journey']
        )
        if candidates.empty:
            continue
        selected.append((label, candidates.iloc[0]))

    output = []
    with sqlite3.connect(f'file:{path.as_posix()}?mode=ro', uri=True) as connection:
        for label, row in selected:
            raw = [
                json.loads(item[0])
                for item in connection.execute(
                    'SELECT raw FROM points WHERE vehicle=? AND journey=? ORDER BY seq',
                    (str(row['vehicle']), str(row['journey'])),
                )
            ]
            if not raw:
                raise ValueError(
                    f"SQLite 索引找不到訓練樣本：{row['vehicle']} / {row['journey']}"
                )
            trip = analyze(raw, detail=True, feature_only=True)
            points = trip['points']
            intervals = {'fuel': [], 'odometer': []}
            all_interval_seconds = []
            for previous, current in zip(points, points[1:]):
                elapsed = current.get('sec', 0) - previous.get('sec', 0)
                if not math.isfinite(elapsed) or elapsed <= 0:
                    continue
                all_interval_seconds.append(float(elapsed))
                if elapsed > 120:
                    continue
                for name, key in (('fuel', 'fuel'), ('odometer', 'odo')):
                    start = previous.get(key)
                    end = current.get(key)
                    if start is None or end is None:
                        continue
                    delta = end - start
                    if math.isfinite(delta) and delta > 1e-8:
                        intervals[name].append(float(delta))
            quality = trip['quality']
            output.append(
                {
                    'sample_group': label,
                    'vehicle': str(row['vehicle']),
                    'journey': str(row['journey']),
                    'split': 'train',
                    'observation_count': len(points),
                    'distance_km': float(row['distance_km']),
                    'fuel_l': float(row['fuel_l']),
                    'duration_minutes': float(row['duration_minutes']),
                    'adjacent_interval_count_le_120s': sum(
                        elapsed <= 120
                        for elapsed in all_interval_seconds
                    ),
                    'positive_fuel_update_count_le_120s': len(intervals['fuel']),
                    'positive_fuel_update_amounts_l_le_120s': intervals['fuel'][:20],
                    'positive_odometer_update_count_le_120s': len(
                        intervals['odometer']
                    ),
                    'positive_odometer_update_amounts_km_le_120s': (
                        intervals['odometer'][:20]
                    ),
                    'max_adjacent_interval_seconds': (
                        max(all_interval_seconds) if all_interval_seconds else None
                    ),
                    'gaps_over_120_seconds': sum(
                        elapsed > 120 for elapsed in all_interval_seconds
                    ),
                    'fuel_counter_reset': bool(quality.get('fuel_reset')),
                    'odometer_counter_reset': bool(quality.get('odo_reset')),
                    'abnormal_can_rows': int(quality.get('abnormal_can_rows', 0)),
                    'duplicate_timestamps': int(
                        quality.get('duplicate_timestamps', 0)
                    ),
                    'interpretation': (
                        '觀測到的正向計數器變化與取樣間隔；不代表硬體解析度，'
                        '亦不單獨證明高油耗是異常或合理。'
                    ),
                }
            )
    return output


def _new_run_dir():
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    name = f'total_fuel_{datetime.now():%Y%m%d_%H%M%S}'
    path = EXPERIMENT_ROOT / name
    suffix = 1
    while path.exists():
        path = EXPERIMENT_ROOT / f'{name}_{suffix}'
        suffix += 1
    path.mkdir()
    return path


def run_experiment():
    import xgboost as xgb

    features, saved_predictions = _load_experiment_rows()
    config = json.loads(
        (HISTORY_EXPERIMENT / 'experiment_config.json').read_text(encoding='utf-8')
    )
    baseline_metadata = json.loads(
        (BASELINE_RUN / 'metadata.json').read_text(encoding='utf-8')
    )
    if xgb.__version__ != baseline_metadata['software_versions']['xgboost']:
        raise ValueError('XGBoost 版本與第一次實驗不同，拒絕不可比的訓練')
    if list(config['fixed_conditions']['original_features']) != FEATURE_COLUMNS:
        raise ValueError('原始特徵順序與既有模型不一致')

    training = features.loc[features['split'] == 'train'].copy()
    validation = features.loc[features['split'] == 'validation'].copy()
    test = features.loc[features['split'] == 'test'].copy()
    model_features = FEATURE_COLUMNS + HISTORY_FEATURES
    for split, rows in (('train', training), ('validation', validation), ('test', test)):
        if rows[model_features].shape[1] != len(model_features):
            raise ValueError(f'{split} 特徵欄位不足')

    parameters = dict(config['fixed_conditions']['parameters'])
    d_train = xgb.DMatrix(
        training[model_features],
        label=training['fuel_l'].to_numpy(dtype=float),
        feature_names=model_features,
    )
    d_validation = xgb.DMatrix(
        validation[model_features],
        label=validation['fuel_l'].to_numpy(dtype=float),
        feature_names=model_features,
    )
    d_test = xgb.DMatrix(test[model_features], feature_names=model_features)

    model = xgb.train(
        parameters,
        d_train,
        num_boost_round=int(config['fixed_conditions']['max_boost_round']),
        evals=[(d_validation, 'validation')],
        early_stopping_rounds=int(
            config['fixed_conditions']['early_stopping_rounds']
        ),
        verbose_eval=False,
    )
    tree_count = int(model.best_iteration) + 1
    validation_fuel_prediction = model.predict(
        d_validation,
        iteration_range=(0, tree_count),
    )
    validation_l100_prediction = (
        validation_fuel_prediction / validation['distance_km'].to_numpy(dtype=float) * 100
    )

    validation_predictions = saved_predictions['validation'].copy()
    validation_predictions['total_fuel_xgboost_prediction_l'] = (
        validation_fuel_prediction
    )
    validation_predictions['total_fuel_xgboost_prediction_l_per_100km'] = (
        validation_l100_prediction
    )
    actual_validation = validation['target_l_per_100km'].to_numpy(dtype=float)
    model_selection = {
        'decision_stage': 'validation_only_before_test_evaluation',
        'primary_metric': 'validation MAE (L/100 km)',
        'selection_rule': (
            'Select total_fuel_xgboost only when its validation L/100 km MAE '
            'is strictly lower than the previously selected history_xgboost.'
        ),
        'previously_selected_model': 'history_xgboost',
        'candidate': 'total_fuel_xgboost',
        'validation_mae_l_per_100km': {
            'vehicle_median_baseline': float(
                np.mean(
                    np.abs(
                        validation_predictions[
                            'vehicle_median_baseline_prediction_l_per_100km'
                        ].to_numpy(dtype=float)
                        - actual_validation
                    )
                )
            ),
            'original_xgboost': float(
                np.mean(
                    np.abs(
                        validation_predictions[
                            'original_xgboost_prediction_l_per_100km'
                        ].to_numpy(dtype=float)
                        - actual_validation
                    )
                )
            ),
            'history_xgboost': float(
                np.mean(
                    np.abs(
                        validation_predictions[
                            'history_xgboost_prediction_l_per_100km'
                        ].to_numpy(dtype=float)
                        - actual_validation
                    )
                )
            ),
            'total_fuel_xgboost': float(
                np.mean(np.abs(validation_l100_prediction - actual_validation))
            ),
        },
    }
    model_selection['selected_model'] = (
        'total_fuel_xgboost'
        if model_selection['validation_mae_l_per_100km']['total_fuel_xgboost']
        < model_selection['validation_mae_l_per_100km']['history_xgboost']
        else 'history_xgboost'
    )
    model_selection['adopt_total_fuel_candidate'] = (
        model_selection['selected_model'] == 'total_fuel_xgboost'
    )
    model_selection['selected_reason'] = (
        'D 的驗證集 L/100 km MAE 嚴格低於既有選定模型 C。'
        if model_selection['adopt_total_fuel_candidate']
        else 'D 未改善既有選定模型 C 的驗證集 L/100 km MAE，保留 C。'
    )

    output_dir = _new_run_dir()
    _write_json(output_dir / 'model_selection.json', model_selection)

    # The test labels are first read for evaluation after the validation choice is persisted.
    test_fuel_prediction = model.predict(d_test, iteration_range=(0, tree_count))
    test_l100_prediction = (
        test_fuel_prediction / test['distance_km'].to_numpy(dtype=float) * 100
    )
    test_predictions = saved_predictions['test'].copy()
    test_predictions['total_fuel_xgboost_prediction_l'] = test_fuel_prediction
    test_predictions['total_fuel_xgboost_prediction_l_per_100km'] = test_l100_prediction
    for method in METHODS[:3]:
        test_predictions[f'{method}_prediction_l'] = (
            test_predictions[f'{method}_prediction_l_per_100km']
            * test['distance_km'].to_numpy(dtype=float)
            / 100
        )
    test_predictions['total_fuel_xgboost_error_l'] = (
        test_fuel_prediction - test['fuel_l'].to_numpy(dtype=float)
    )
    test_predictions['total_fuel_xgboost_error_l_per_100km'] = (
        test_l100_prediction - test['target_l_per_100km'].to_numpy(dtype=float)
    )

    validation_predictions['total_fuel_xgboost_error_l'] = (
        validation_fuel_prediction - validation['fuel_l'].to_numpy(dtype=float)
    )
    validation_predictions['total_fuel_xgboost_error_l_per_100km'] = (
        validation_l100_prediction - validation['target_l_per_100km'].to_numpy(
            dtype=float
        )
    )
    train_rows_for_baseline = training.to_dict(orient='records')
    vehicle_medians = training.groupby('vehicle')['target_l_per_100km'].median()
    global_median = float(training['target_l_per_100km'].median())
    training_baseline_prediction = np.asarray(
        [
            float(vehicle_medians.get(str(row['vehicle']), global_median))
            for row in train_rows_for_baseline
        ],
        dtype=float,
    )
    training_dmatrix = xgb.DMatrix(
        training[model_features],
        feature_names=model_features,
    )
    training_prediction = model.predict(
        training_dmatrix,
        iteration_range=(0, tree_count),
    )
    train_predictions = pd.DataFrame(
        {
            'vehicle': training['vehicle'].astype(str).to_numpy(),
            'journey': training['journey'].astype(str).to_numpy(),
            'vehicle_median_baseline_prediction_l_per_100km': (
                training_baseline_prediction
            ),
            'original_xgboost_prediction_l_per_100km': np.nan,
            'history_xgboost_prediction_l_per_100km': np.nan,
            'total_fuel_xgboost_prediction_l_per_100km': (
                training_prediction / training['distance_km'].to_numpy(dtype=float) * 100
            ),
        }
    )
    baseline_model = xgb.Booster()
    baseline_model.load_model(BASELINE_RUN / 'xgboost_model.json')
    baseline_tree_count = int(baseline_model.attr('best_iteration')) + 1
    d_train_original = xgb.DMatrix(
        training[FEATURE_COLUMNS],
        feature_names=FEATURE_COLUMNS,
    )
    train_predictions['original_xgboost_prediction_l_per_100km'] = (
        baseline_model.predict(
            d_train_original,
            iteration_range=(0, baseline_tree_count),
        )
    )
    history_model = xgb.Booster()
    history_model.load_model(HISTORY_EXPERIMENT / 'xgboost_with_vehicle_history.json')
    history_tree_count = int(history_model.attr('best_iteration')) + 1
    train_predictions['history_xgboost_prediction_l_per_100km'] = (
        history_model.predict(
            training_dmatrix,
            iteration_range=(0, history_tree_count),
        )
    )
    train_predictions['total_fuel_xgboost_error_l_per_100km'] = (
        train_predictions['total_fuel_xgboost_prediction_l_per_100km']
        - training['target_l_per_100km'].to_numpy(dtype=float)
    )
    train_predictions['total_fuel_xgboost_prediction_l'] = training_prediction

    evaluation = {
        'sample_counts': {
            'train': int(len(training)),
            'validation': int(len(validation)),
            'test_follow_up': int(len(test)),
        },
        'test_caveat': (
            '此測試集的第一次結果已用於指引後續開發，本次結果屬既有保留集的追蹤評估，'
            '不是全新、未接觸資料的最終驗證。'
        ),
        'test_sample_keys_match_existing_experiment': True,
        'models': {},
    }
    for split, rows, table in (
        ('train', training, train_predictions),
        ('validation', validation, validation_predictions),
        ('test_follow_up', test, test_predictions),
    ):
        actual_l100 = rows['target_l_per_100km'].to_numpy(dtype=float)
        actual_l = rows['fuel_l'].to_numpy(dtype=float)
        vehicles = rows['vehicle'].astype(str).to_numpy()
        distances = rows['distance_km'].to_numpy(dtype=float)
        evaluation['models'][split] = {}
        for method in METHODS:
            if method == 'total_fuel_xgboost':
                pred_l = table['total_fuel_xgboost_prediction_l'].to_numpy(dtype=float)
                pred_l100 = table[
                    'total_fuel_xgboost_prediction_l_per_100km'
                ].to_numpy(dtype=float)
            else:
                pred_l100 = table[
                    f'{method}_prediction_l_per_100km'
                ].to_numpy(dtype=float)
                pred_l = pred_l100 * distances / 100
            evaluation['models'][split][method] = {
                'fuel_l': _metrics(actual_l, pred_l, vehicles),
                'l_per_100km': _metrics(actual_l100, pred_l100, vehicles),
            }

    distance_diagnostics = _distance_diagnostics(
        features,
        {
            'validation': validation_predictions,
            'test': test_predictions,
        },
        train_predictions,
    )
    counter_updates = _counter_update_diagnostics(
        training,
        baseline_metadata['index']['path'],
    )
    original_features = features[
        (features['distance_km'] > 0) & (features['distance_km'] < 0.2)
    ].copy()
    short_examples = original_features.sort_values(
        'target_l_per_100km',
        ascending=False,
    ).head(30)
    short_evidence = short_examples[
        [
            'vehicle',
            'journey',
            'split',
            'distance_km',
            'fuel_l',
            'target_l_per_100km',
            'duration_minutes',
            'idle_engine_share_pct',
            'fuel_min_positive_step_l',
            'odo_min_positive_step_km',
            'quality_flags',
            'abnormal_can_rows',
        ]
    ].to_dict(orient='records')

    model.save_model(output_dir / 'xgboost_total_fuel.json')
    _write_json(
        output_dir / 'experiment_config.json',
        {
            'experiment': 'total trip fuel target candidate D',
            'target': 'fuel_l; effective cumulative fuel counter difference in L',
            'features': model_features,
            'sample_split_source': str(
                HISTORY_EXPERIMENT / 'trip_features_with_vehicle_history.csv'
            ),
            'split_counts': evaluation['sample_counts'],
            'parameters': parameters,
            'max_boost_round': int(config['fixed_conditions']['max_boost_round']),
            'early_stopping_rounds': int(
                config['fixed_conditions']['early_stopping_rounds']
            ),
            'best_iteration_zero_based': int(model.best_iteration),
            'prediction_tree_count': tree_count,
            'early_stopping_metric': 'MAE in L',
            'final_candidate_selection_metric': 'validation MAE in L/100 km',
            'test_policy': 'One test follow-up after validation selection is saved.',
            'test_caveat': evaluation['test_caveat'],
            'training_diagnostics_note': (
                'Training-set metrics are in-sample descriptive diagnostics only; '
                'validation and test follow-up metrics are the generalization comparisons.'
            ),
        },
    )
    _write_json(output_dir / 'evaluation.json', evaluation)
    _write_json(output_dir / 'distance_diagnostics.json', distance_diagnostics)
    _write_json(output_dir / 'short_trip_evidence.json', short_evidence)
    _write_json(output_dir / 'counter_update_diagnostics.json', counter_updates)
    validation_predictions.to_csv(
        output_dir / 'validation_predictions.csv',
        index=False,
        encoding='utf-8-sig',
    )
    test_predictions.to_csv(
        output_dir / 'test_predictions.csv',
        index=False,
        encoding='utf-8-sig',
    )
    train_predictions.to_csv(
        output_dir / 'train_predictions.csv',
        index=False,
        encoding='utf-8-sig',
    )

    chosen_method = model_selection['selected_model']
    selected_model_path = (
        output_dir / 'xgboost_total_fuel.json'
        if chosen_method == 'total_fuel_xgboost'
        else HISTORY_EXPERIMENT / 'xgboost_with_vehicle_history.json'
    )
    shutil.copy2(selected_model_path, output_dir / 'selected_model.json')
    bundle = {
        'model_method': chosen_method,
        'model_target': 'fuel_l' if chosen_method == 'total_fuel_xgboost' else 'l_per_100km',
        'model_file': 'model.json',
        'prediction_tree_count': (
            tree_count
            if chosen_method == 'total_fuel_xgboost'
            else int(history_model.attr('best_iteration')) + 1
        ),
        'feature_names': model_features,
        'xgboost_version': xgb.__version__,
        'model_role': 'experimental_trip_completion_estimate',
        'validation_metrics': evaluation['models']['validation'][chosen_method],
        'test_follow_up_metrics': evaluation['models']['test_follow_up'][chosen_method],
        'comparison_validation_mae_l_per_100km': model_selection[
            'validation_mae_l_per_100km'
        ],
        'selection_reason': model_selection['selected_reason'],
        'trip_feature_file': 'trip_features_with_vehicle_history.csv',
        'history_statistics_file': 'frozen_training_history.json',
        'shap_method': 'XGBoost native exact TreeSHAP pred_contribs, tree-path-dependent',
        'shap_background': 'No external background dataset; fixed trained-tree path statistics.',
    }
    DASHBOARD_BUNDLE.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected_model_path, DASHBOARD_BUNDLE / 'model.json')
    shutil.copy2(
        HISTORY_EXPERIMENT / 'trip_features_with_vehicle_history.csv',
        DASHBOARD_BUNDLE / 'trip_features_with_vehicle_history.csv',
    )
    shutil.copy2(
        HISTORY_EXPERIMENT / 'frozen_training_history.json',
        DASHBOARD_BUNDLE / 'frozen_training_history.json',
    )
    _write_json(DASHBOARD_BUNDLE / 'bundle.json', bundle)
    report = [
        '# 行程總耗油目標候選 D：短距離診斷與驗證',
        '',
        '模型用途為行程後油耗估計；不代表因果、理想油耗或可節省油量。',
        '',
        '## 診斷結論',
        '',
        '距離 0–1 km 的驗證及測試 MAE 顯著高於較長距離組，且 L/100 km 是總耗油除以里程的比值。'
        '在 0.1 km 行程中，單一 0.5 L 的觀測增量會換算為 500 L/100 km。此結果支持只新增一次以總耗油 L 為目標的候選 D；'
        '這不保證換算後的 L/100 km 誤差一定改善。',
        '',
        '## 驗證集選擇（先於測試評估）',
        '',
        f"- 選擇：`{chosen_method}`。",
        f"- 理由：{model_selection['selected_reason']}",
        '',
        '| 方法 | 驗證 MAE L/100 km | 測試追蹤 MAE L/100 km | 驗證 MAE L | 測試追蹤 MAE L |',
        '|---|---:|---:|---:|---:|',
    ]
    for method in METHODS:
        report.append(
            f"| {method} | "
            f"{evaluation['models']['validation'][method]['l_per_100km']['mae']:.3f} | "
            f"{evaluation['models']['test_follow_up'][method]['l_per_100km']['mae']:.3f} | "
            f"{evaluation['models']['validation'][method]['fuel_l']['mae']:.3f} | "
            f"{evaluation['models']['test_follow_up'][method]['fuel_l']['mae']:.3f} |"
        )
    report.extend(
        [
            '',
            '## 距離分組',
            '',
            '完整分組指標、各模型誤差總量占比見 `distance_diagnostics.json`；逐車、每車樣本數及兩種單位的 MAE 見 `evaluation.json`。',
            '',
            '| 分割 | 距離 | 趟數 | 占比 | 燃油中位數／P90／最大 L | L/100 km 中位數／P90／最大 | 中位行程分鐘 | 中位停車引擎運轉比例 |',
            '|---|---|---:|---:|---:|---:|---:|---:|',
        ]
    )
    for split, groups in distance_diagnostics.items():
        for group in groups:
            report.append(
                f"| {split} | {group['distance_band']} | {group['trip_count']} | "
                f"{group['split_share']:.1%} | {group['fuel_median_l']:.3f} / "
                f"{group['fuel_p90_l']:.3f} / {group['fuel_max_l']:.3f} | "
                f"{group['l_per_100km_median']:.3f} / {group['l_per_100km_p90']:.3f} / "
                f"{group['l_per_100km_max']:.3f} | "
                f"{group['duration_median_minutes']:.2f} | "
                f"{group['idle_share_median_pct']:.2f}% |"
            )
    report.extend(
        [
            '',
            '### 距離組別模型誤差與占全部絕對誤差比例',
            '',
            '訓練組數字為訓練內描述性結果；驗證與測試數字分別來自固定驗證集及測試追蹤評估。',
            '',
            '| 分割 | 距離 | A MAE／占比 | B MAE／占比 | C MAE／占比 | D MAE／占比 |',
            '|---|---|---:|---:|---:|---:|',
        ]
    )
    for split, groups in distance_diagnostics.items():
        for group in groups:
            cells = []
            for method in METHODS:
                mae = group[f'{method}_mae_l_per_100km']
                share = group[f'{method}_absolute_error_share']
                cells.append(
                    '—'
                    if mae is None or share is None
                    else f'{mae:.3f}／{share:.1%}'
                )
            report.append(
                f"| {split} | {group['distance_band']} | "
                + ' | '.join(cells)
                + ' |'
            )
    report.extend(
        [
            '',
            '## 約 0.1 km 行程與計數器',
            '',
            '距離與耗油延續既有口徑：有效累積里程／燃油端點差，遇到計數器倒退或端點無效不修補。'
            '若資料品質旗標為 0、CAN 異常列為 0，且有 0.5 L 燃油增量與 0.1 km 里程增量，這些觀測本身不足以判定資料異常；'
            '短時間高怠速可與合理的低速移動／停車引擎運轉相容。未取得足以確認的外部真值時，仍標記為無法判定，'
            '不把最小觀測增量稱為硬體解析度，也不排除極端值。逐趟例子在 `short_trip_evidence.json`。',
            '少量訓練軌跡抽查正向計數器更新及相鄰量測間隔，見 `counter_update_diagnostics.json`。'
            '約 0.1 km 樣本可只有一次 0.5 L 和一次 0.1 km 的正向增量；較長行程則可能在多個有效區間分散觀測到正向更新。'
            '這只能說明目前抽樣記錄的稀疏性，不可推論硬體解析度。',
            '',
            '## 限制',
            '',
            '目前機器學習模型尚未優於歷史基準。',
            evaluation['test_caveat'],
            'D 的訓練與 early stopping 使用總耗油 L；候選選擇和跨模型比較使用同一驗證集的 L/100 km MAE。'
            'A/B/C 換算 L 預測時以各趟距離乘以 L/100 km 預測除以 100。'
            '本次不使用測試結果調整模型，未用全部資料重訓。',
        ]
    )
    (output_dir / 'report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(json.dumps({
        'output_dir': str(output_dir),
        'dashboard_bundle': str(DASHBOARD_BUNDLE),
        'selected_model': chosen_method,
        'model_selection': model_selection,
        'evaluation': {
            split: {
                method: {
                    'l100_mae': values['l_per_100km']['mae'],
                    'fuel_mae': values['fuel_l']['mae'],
                }
                for method, values in methods.items()
            }
            for split, methods in evaluation['models'].items()
        },
    }, ensure_ascii=False, indent=2))
    return output_dir


if __name__ == '__main__':
    run_experiment()
