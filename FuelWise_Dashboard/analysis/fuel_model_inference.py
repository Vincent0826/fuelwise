"""Load the selected trip model once and explain individual estimates."""

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.train_fuel_model import FEATURE_COLUMNS, extract_features
from analysis.trip_applicability import evaluate_applicability
from analysis.vehicle_history_experiment import HISTORY_FEATURES

BASE = Path(__file__).resolve().parent.parent
EXPERIMENT_ROOT = BASE / 'analysis_output' / 'experiments'
LEGACY_BUNDLE_DIR = EXPERIMENT_ROOT / 'dashboard_model'
APPLICABLE_BUNDLE_DIR = EXPERIMENT_ROOT / 'dashboard_model_applicable'
PACKAGED_BUNDLE_DIR = BASE / 'analysis' / 'model_bundle_applicable'
BUNDLE_DIR = next(
    (
        bundle_dir
        for bundle_dir in (
            PACKAGED_BUNDLE_DIR,
            APPLICABLE_BUNDLE_DIR,
            LEGACY_BUNDLE_DIR,
        )
        if (bundle_dir / 'bundle.json').is_file()
    ),
    LEGACY_BUNDLE_DIR,
)
MODEL_FEATURES = FEATURE_COLUMNS + HISTORY_FEATURES
FEATURE_LABELS = {
    'distance_km': ('行程距離', 'km'),
    'duration_minutes': ('行程歷時', 'min'),
    'speed_mean_kmh': ('平均車速', 'km/h'),
    'speed_std_kmh': ('車速標準差', 'km/h'),
    'rpm_mean': ('平均引擎轉速', 'rpm'),
    'engine_load_mean_pct': ('平均引擎負載', '%'),
    'coolant_temp_mean_c': ('平均冷卻水溫', '°C'),
    'idle_engine_share_pct': ('停車引擎運轉比例', '%'),
    'speed_abs_change_mean_kmh': ('平均速度變化量', 'km/h/觀測間隔'),
    'mean_abs_acceleration_m_s2': ('平均絕對加速度', 'm/s²'),
    'kinetic_increase_j_per_kg': ('正向動能變化代理值', 'J/kg'),
    'kinetic_increase_j_per_kg_km': ('單位距離動能變化代理值', 'J/kg/km'),
    'speed_stopped_time_pct': ('車速為零時間比例', '%'),
    'speed_0_20_time_pct': ('0–20 km/h 時間比例', '%'),
    'speed_20_40_time_pct': ('20–40 km/h 時間比例', '%'),
    'speed_40_60_time_pct': ('40–60 km/h 時間比例', '%'),
    'speed_60_80_time_pct': ('60–80 km/h 時間比例', '%'),
    'speed_80_100_time_pct': ('80–100 km/h 時間比例', '%'),
    'speed_over_100_time_pct': ('超過 100 km/h 時間比例', '%'),
    'vehicle_history_median_l_per_100km': ('訓練歷史油耗中位數', 'L/100 km'),
    'vehicle_history_count': ('同車訓練歷史趟數', '趟'),
    'vehicle_history_fallback': ('使用全體訓練中位數回退', '0／1'),
}
DRIVER_ACTIONS = {
    'idle_engine_share_pct': (
        '減少不必要怠速',
        '停等時若安全、合法且符合車隊規範，減少不必要的引擎運轉；勿因此影響交通安全。',
        'idle',
    ),
    'mean_abs_acceleration_m_s2': (
        '平順起步與加速',
        '提早觀察路況、平順起步，避免不必要的急加速。',
        'smooth_driving',
    ),
    'speed_abs_change_mean_kmh': (
        '減少頻繁加減速',
        '保持安全車距並提早收油，減少不必要的頻繁加減速。',
        'smooth_driving',
    ),
    'speed_std_kmh': (
        '維持穩定車速',
        '在路況與速限允許下平穩行駛；交通與路線也會影響車速變化。',
        'smooth_driving',
    ),
}


class ModelUnavailableError(RuntimeError):
    """The saved dashboard model or its artifacts are unavailable."""


class TripInferenceError(ValueError):
    """A trip cannot be scored with the saved model."""


def _finite_number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@lru_cache(maxsize=1)
def load_bundle():
    required = (
        BUNDLE_DIR / 'bundle.json',
        BUNDLE_DIR / 'model.json',
        BUNDLE_DIR / 'frozen_training_history.json',
        BUNDLE_DIR / 'trip_features_with_vehicle_history.csv',
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise ModelUnavailableError(
            '行程油耗實驗模型產物不完整，缺少：'
            + '、'.join(missing)
            + '。啟動網站時會自動由目前的 SQLite 索引建立；請查看啟動終端機的模型訓練錯誤。'
        )
    try:
        import xgboost as xgb

        metadata = json.loads(required[0].read_text(encoding='utf-8'))
        history = json.loads(required[2].read_text(encoding='utf-8'))
        if xgb.__version__ != metadata['xgboost_version']:
            raise ModelUnavailableError(
                f"模型需要 XGBoost {metadata['xgboost_version']}，"
                f"目前環境為 {xgb.__version__}。"
            )
        feature_names = metadata['feature_names']
        supported_feature_sets = (FEATURE_COLUMNS, MODEL_FEATURES)
        if feature_names not in supported_feature_sets:
            raise ModelUnavailableError('模型特徵名稱或順序與目前程式不一致。')
        model = xgb.Booster()
        model.load_model(required[1])
        if model.feature_names != feature_names:
            raise ModelUnavailableError('已保存模型的輸入特徵順序不正確。')
        tree_count = int(metadata['prediction_tree_count'])
        if tree_count < 1 or tree_count > model.num_boosted_rounds():
            raise ModelUnavailableError('模型預測樹數設定無效。')
        trip_features = pd.read_csv(
            required[3],
            usecols=['vehicle', 'journey', 'split', *feature_names],
            dtype={'vehicle': str, 'journey': str},
        )
        saved_features = {}
        for row in trip_features.to_dict(orient='records'):
            key = (str(row['vehicle']), str(row['journey']))
            if key in saved_features:
                raise ModelUnavailableError('逐趟模型特徵含重複車輛與行程識別。')
            saved_features[key] = {
                'split': str(row['split']),
                'features': {name: row[name] for name in feature_names},
            }
        return {
            'metadata': metadata,
            'history': history,
            'model': model,
            'tree_count': tree_count,
            'saved_features': saved_features,
        }
    except ModelUnavailableError:
        raise
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ModelUnavailableError(f'載入行程油耗模型失敗：{exc}') from exc


def _history_features_for_new_trip(vehicle, history):
    stats = history.get('training_vehicle_history', {})
    global_median = _finite_number(
        history.get('global_training_median_l_per_100km')
    )
    vehicle_stats = stats.get(str(vehicle))
    if vehicle_stats:
        median = _finite_number(vehicle_stats.get('median_l_per_100km'))
        count = int(vehicle_stats.get('completed_training_trip_count', 0))
        return {
            HISTORY_FEATURES[0]: median,
            HISTORY_FEATURES[1]: count,
            HISTORY_FEATURES[2]: 0,
        }
    return {
        HISTORY_FEATURES[0]: global_median,
        HISTORY_FEATURES[1]: 0,
        HISTORY_FEATURES[2]: 1,
    }


def _feature_values(analysis, vehicle, journey, bundle):
    saved = bundle['saved_features'].get((str(vehicle), str(journey)))
    if saved is not None:
        return saved['features'], saved['split'], 'saved_trip_features'

    features, _, _ = extract_features(analysis)
    missing_columns = sorted(set(FEATURE_COLUMNS) - set(features))
    if missing_columns:
        raise TripInferenceError(
            '本趟無法建立模型所需的特徵：' + '、'.join(missing_columns)
        )
    values = {
        name: _finite_number(features.get(name))
        for name in FEATURE_COLUMNS
    }
    values.update(_history_features_for_new_trip(vehicle, bundle['history']))
    return values, 'other', 'frozen_training_history'


def _predict_and_explain(model, values, feature_names, tree_count):
    import xgboost as xgb

    normalized = {
        name: _finite_number(values.get(name))
        for name in feature_names
    }
    if any(name not in values for name in feature_names):
        absent = [name for name in feature_names if name not in values]
        raise TripInferenceError('推論特徵缺少欄位：' + '、'.join(absent))
    if not any(value is not None for value in normalized.values()):
        raise TripInferenceError('本趟沒有任何可用的模型輸入特徵。')
    matrix = xgb.DMatrix(
        pd.DataFrame(
            [[np.nan if normalized[name] is None else normalized[name]
              for name in feature_names]],
            columns=feature_names,
        ),
        feature_names=feature_names,
    )
    prediction = float(
        model.predict(matrix, iteration_range=(0, tree_count))[0]
    )
    contributions = np.asarray(
        model.predict(
            matrix,
            pred_contribs=True,
            iteration_range=(0, tree_count),
        )[0],
        dtype=float,
    )
    if len(contributions) != len(feature_names) + 1:
        raise TripInferenceError('SHAP 貢獻數量與模型特徵不一致。')
    base_value = float(contributions[-1])
    shap_values = contributions[:-1]
    reconstructed = base_value + float(np.sum(shap_values))
    additivity_error = reconstructed - prediction
    tolerance = max(1e-5, abs(prediction) * 1e-5)
    if not math.isfinite(prediction) or not math.isfinite(reconstructed):
        raise TripInferenceError('模型或 SHAP 回傳非有限數值，無法展示估計。')
    if abs(additivity_error) > tolerance:
        raise TripInferenceError(
            f'SHAP 加總與模型輸出不一致（差 {additivity_error:.6g}）。'
        )
    return normalized, prediction, base_value, shap_values, additivity_error


def _fuel_saving_scenarios(
    model,
    values,
    shap_values,
    feature_names,
    tree_count,
    current_prediction,
    distance_km,
    target,
    bundle,
):
    """Estimate isolated model scenarios against the eligible training Q25."""
    current_l100 = (
        current_prediction / distance_km * 100
        if target == 'fuel_l'
        else current_prediction
    )
    if not math.isfinite(current_l100) or current_l100 <= 0:
        return []

    training = [
        row['features']
        for row in bundle['saved_features'].values()
        if row.get('split') == 'train'
    ]
    candidates = []
    for index, name in enumerate(feature_names):
        action = DRIVER_ACTIONS.get(name)
        current = _finite_number(values.get(name))
        shap_value = _finite_number(shap_values[index])
        if (
            action is None
            or current is None
            or shap_value is None
            or shap_value <= 0
        ):
            continue
        reference = np.asarray(
            [
                value
                for row in training
                if (value := _finite_number(row.get(name))) is not None
            ],
            dtype=float,
        )
        if not len(reference):
            continue
        benchmark = float(np.quantile(reference, 0.25))
        if not math.isfinite(benchmark) or current <= benchmark:
            continue

        scenario = dict(values)
        scenario[name] = benchmark
        _, counterfactual, _, _, _ = _predict_and_explain(
            model,
            scenario,
            feature_names,
            tree_count,
        )
        counterfactual_l100 = (
            counterfactual / distance_km * 100
            if target == 'fuel_l'
            else counterfactual
        )
        saving_l100 = current_l100 - counterfactual_l100
        if (
            not math.isfinite(counterfactual_l100)
            or counterfactual_l100 < 0
            or saving_l100 <= 0
        ):
            continue
        title, instruction, group = action
        candidates.append(
            {
                'key': name,
                'name': FEATURE_LABELS[name][0],
                'title': title,
                'instruction': instruction,
                'group': group,
                'unit': FEATURE_LABELS[name][1],
                'current_value': current,
                'benchmark_value': benchmark,
                'benchmark_description': '合格訓練行程第 25 百分位',
                'shap_contribution': shap_value,
                'estimate_before_l_per_100km': current_l100,
                'estimated_saving_pct': saving_l100 / current_l100 * 100,
                'estimated_saving_l_per_100km': saving_l100,
                'estimated_saving_l_for_trip': saving_l100 * distance_km / 100,
                'estimate_after_l_per_100km': counterfactual_l100,
            }
        )

    # Acceleration and speed-variation features overlap; report only the
    # strongest modeled scenario for that shared driving behavior.
    selected = {}
    for candidate in candidates:
        group = candidate['group']
        previous = selected.get(group)
        if previous is None or (
            candidate['estimated_saving_pct'],
            candidate['shap_contribution'],
        ) > (
            previous['estimated_saving_pct'],
            previous['shap_contribution'],
        ):
            selected[group] = candidate
    return sorted(
        selected.values(),
        key=lambda item: item['estimated_saving_pct'],
        reverse=True,
    )


def _explanation_items(values, shap_values, feature_names):
    contributions = [
        {
            'key': name,
            'name': FEATURE_LABELS.get(name, (name, ''))[0],
            'unit': FEATURE_LABELS.get(name, (name, ''))[1],
            'value': values[name],
            'shap_value': float(shap_values[index]),
        }
        for index, name in enumerate(feature_names)
    ]
    contributions.sort(key=lambda item: abs(item['shap_value']), reverse=True)
    visible = contributions[:6]
    remainder = contributions[6:]
    if remainder:
        visible.append(
            {
                'key': 'other_features',
                'name': f'其他特徵（{len(remainder)} 項合計）',
                'unit': '',
                'value': None,
                'shap_value': float(
                    sum(item['shap_value'] for item in remainder)
                ),
            }
        )
    return visible


def predict_trip(analysis, vehicle, journey):
    applicability = evaluate_applicability(analysis)
    if not applicability['model_eligible']:
        return {
            'available': True,
            'eligible': False,
            'applicability': applicability,
        }
    bundle = load_bundle()
    summary = analysis.get('summary', {})
    distance = _finite_number(summary.get('distance_km'))
    if distance is None or distance <= 0:
        raise TripInferenceError('有效里程不是正數，無法計算或比較 L/100 km。')
    metadata = bundle['metadata']

    values, split, feature_source = _feature_values(
        analysis,
        vehicle,
        journey,
        bundle,
    )
    normalized, raw_prediction, base_value, shap_values, additivity_error = (
        _predict_and_explain(
            bundle['model'],
            values,
            metadata['feature_names'],
            bundle['tree_count'],
        )
    )
    target = metadata['model_target']
    fuel_saving_scenarios = _fuel_saving_scenarios(
        bundle['model'],
        normalized,
        shap_values,
        metadata['feature_names'],
        bundle['tree_count'],
        raw_prediction,
        distance,
        target,
        bundle,
    )
    if target == 'fuel_l':
        estimated_fuel_l = raw_prediction
        estimated_l_per_100km = raw_prediction / distance * 100
        difference_unit = 'L'
        actual_value = _finite_number(summary.get('fuel_l'))
        estimate_value = estimated_fuel_l
    elif target == 'l_per_100km':
        estimated_l_per_100km = raw_prediction
        estimated_fuel_l = raw_prediction * distance / 100
        difference_unit = 'L/100 km'
        actual_value = _finite_number(summary.get('l100'))
        estimate_value = estimated_l_per_100km
    else:
        raise ModelUnavailableError(f'模型目標單位不支援：{target}')

    history_baseline_l100 = (
        _finite_number(values.get(HISTORY_FEATURES[0]))
        if split == 'train'
        else _finite_number(
            _history_features_for_new_trip(vehicle, bundle['history'])[
                HISTORY_FEATURES[0]
            ]
        )
    )
    actual_fuel_l = _finite_number(summary.get('fuel_l'))
    actual_l_per_100km = _finite_number(summary.get('l100'))
    missing_features = [
        name for name, value in normalized.items() if value is None
    ]
    limitations = [
        '根據已完成行程的資料估計油耗。',
        '特徵貢獻解釋模型判斷，不代表因果或保證可節省的油量。',
        '本模型使用歷史車隊資料；未包含載重、坡度、交通、天候或道路狀況。',
        '測試集已用於開發方向，本次測試指標屬追蹤評估，不是全新未接觸的最終驗證。',
    ]
    warnings = []
    if missing_features:
        warnings.append(
            '以下特徵在本趟缺值，模型按訓練時規則將其視為缺失：'
            + '、'.join(FEATURE_LABELS.get(name, (name, ''))[0]
                       for name in missing_features)
        )
    if raw_prediction < 0:
        warnings.append('模型原始估計為負值；未截斷，請勿視為可實際耗油量。')
    actual_minus_estimate = (
        actual_value - estimate_value if actual_value is not None else None
    )
    return {
        'available': True,
        'eligible': True,
        'applicability': applicability,
        'model': {
            'rule_version': metadata.get('rule_version'),
            'method': metadata['model_method'],
            'target': target,
            'role': metadata['model_role'],
            'xgboost_version': metadata['xgboost_version'],
            'tree_count': bundle['tree_count'],
            'validation_metrics': metadata['validation_metrics'],
            'validation_comparison_mae_l_per_100km': metadata[
                'comparison_validation_mae_l_per_100km'
            ],
            'test_follow_up_metrics': metadata.get('test_follow_up_metrics'),
            'selection_reason': metadata['selection_reason'],
        },
        'data_split': split,
        'feature_source': feature_source,
        'actual': {
            'fuel_l': actual_fuel_l,
            'l_per_100km': actual_l_per_100km,
        },
        'estimate': {
            'fuel_l': estimated_fuel_l,
            'l_per_100km': estimated_l_per_100km,
        },
        'estimated_difference': {
            'definition': 'actual_minus_estimate',
            'value': actual_minus_estimate,
            'unit': difference_unit,
        },
        'history_baseline': {
            'l_per_100km': history_baseline_l100,
            'unit': 'L/100 km',
        },
        'explanation': {
            'method': metadata['shap_method'],
            'background': metadata['shap_background'],
            'target_unit': 'L' if target == 'fuel_l' else 'L/100 km',
            'base_value': base_value,
            'prediction': raw_prediction,
            'additivity_error': additivity_error,
            'features': _explanation_items(
                normalized,
                shap_values,
                metadata['feature_names'],
            ),
        },
        'fuel_saving_scenarios': fuel_saving_scenarios,
        'missing_features': missing_features,
        'warnings': warnings,
        'limitations': limitations,
    }
