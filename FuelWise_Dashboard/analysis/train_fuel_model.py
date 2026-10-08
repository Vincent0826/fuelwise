"""Build and evaluate a time-split trip-level fuel economy model."""

import argparse
import json
import math
import os
import sqlite3
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import analyze

BASE = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = BASE / 'analysis_output' / 'fuel_model_runs'
RANDOM_SEED = 20261007
GAP_LIMIT_SECONDS = 120
ACCELERATION_MAX_INTERVAL_SECONDS = 30
SPEED_BANDS = [
    ('speed_stopped_time_pct', 'Valid time with CAN speed = 0', 'percent'),
    ('speed_0_20_time_pct', 'Valid time with 0 < CAN speed <= 20', 'percent'),
    ('speed_20_40_time_pct', 'Valid time with 20 < CAN speed <= 40', 'percent'),
    ('speed_40_60_time_pct', 'Valid time with 40 < CAN speed <= 60', 'percent'),
    ('speed_60_80_time_pct', 'Valid time with 60 < CAN speed <= 80', 'percent'),
    ('speed_80_100_time_pct', 'Valid time with 80 < CAN speed <= 100', 'percent'),
    ('speed_over_100_time_pct', 'Valid time with CAN speed > 100', 'percent'),
]
FEATURE_DEFINITIONS = {
    'distance_km': {
        'definition': 'Effective trip distance from the existing cumulative odometer endpoint difference.',
        'unit': 'km',
        'coverage': 'Valid adjacent odometer endpoint intervals <=120 seconds / elapsed trip seconds; target eligibility separately requires finite positive endpoint distance without a counter reset.',
    },
    'duration_minutes': {
        'definition': 'Elapsed seconds between the first and last valid trip timestamps divided by 60.',
        'unit': 'min',
        'coverage': 'Valid first and last timestamps; elapsed duration includes periods the existing analysis marks as long gaps.',
    },
    'speed_mean_kmh': {
        'definition': 'Time-weighted mean CAN speed using the existing previous-observation hold and valid intervals.',
        'unit': 'km/h',
        'coverage': 'Valid CAN speed interval seconds / elapsed trip seconds.',
    },
    'speed_std_kmh': {
        'definition': 'Population standard deviation of CAN speed, weighted by valid interval seconds.',
        'unit': 'km/h',
        'coverage': 'Valid CAN speed interval seconds / elapsed trip seconds.',
    },
    'rpm_mean': {
        'definition': 'Time-weighted mean engine RPM using the existing previous-observation hold and valid intervals.',
        'unit': 'rpm',
        'coverage': 'Valid CAN RPM interval seconds / elapsed trip seconds.',
    },
    'engine_load_mean_pct': {
        'definition': 'Time-weighted mean CAN engine load; engine load is not cargo weight.',
        'unit': 'percent',
        'coverage': 'Valid CAN engine-load interval seconds / elapsed trip seconds.',
    },
    'coolant_temp_mean_c': {
        'definition': 'Time-weighted mean CAN coolant temperature.',
        'unit': 'degC',
        'coverage': 'Valid CAN coolant-temperature interval seconds / elapsed trip seconds.',
    },
    'idle_engine_share_pct': {
        'definition': 'Existing idle definition: CAN speed = 0 and RPM > 0; idle seconds / intervals where speed and RPM are both known.',
        'unit': 'percent',
        'coverage': 'Intervals with both valid CAN speed and RPM / elapsed trip seconds.',
    },
    'speed_abs_change_mean_kmh': {
        'definition': 'Mean absolute speed change across adjacent valid CAN speed readings with 0 < interval <= 120 seconds.',
        'unit': 'km/h per adjacent observation',
        'coverage': 'Valid adjacent-speed interval seconds / elapsed trip seconds.',
    },
    'mean_abs_acceleration_m_s2': {
        'definition': 'Time-weighted mean absolute speed difference / elapsed seconds, converted from km/h to m/s; uses the existing <=30 second acceleration interval rule.',
        'unit': 'm/s^2',
        'coverage': 'Valid adjacent-speed interval seconds <=30 seconds / elapsed trip seconds.',
    },
    'kinetic_increase_j_per_kg': {
        'definition': 'Sum(max(0, (v_next^2 - v_prev^2) / 2)) for adjacent valid CAN speed observations, with speeds converted to m/s and intervals <=120 seconds.',
        'unit': 'J/kg',
        'coverage': 'Valid adjacent-speed interval seconds / elapsed trip seconds.',
        'limitation': 'A sampling-scale acceleration-activity proxy, not fuel energy; excludes vehicle mass, resistance, and drivetrain efficiency.',
    },
    'kinetic_increase_j_per_kg_km': {
        'definition': 'kinetic_increase_j_per_kg divided by effective trip distance.',
        'unit': 'J/kg/km',
        'coverage': 'Valid adjacent-speed interval seconds / elapsed trip seconds; included only for a valid positive trip distance.',
        'limitation': 'A sampling-scale acceleration-activity proxy, not fuel energy; excludes vehicle mass, resistance, and drivetrain efficiency.',
    },
}
for feature, definition, unit in SPEED_BANDS:
    FEATURE_DEFINITIONS[feature] = {
        'definition': definition,
        'unit': unit,
        'coverage': 'Valid CAN speed interval seconds / elapsed trip seconds.',
    }
FEATURE_COLUMNS = list(FEATURE_DEFINITIONS)
FORBIDDEN_FEATURE_TOKENS = (
    'fuel',
    'l100',
    'km_l',
    'target',
    'score',
    'rank',
    'journey',
    'vehicle',
    'source_row',
    'trip_id',
)


def finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def find_existing_index(index_path=None):
    """Select an existing read-only index; never invoke the Excel importer."""
    if index_path is not None:
        path = Path(index_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f'SQLite 索引不存在：{path}')
        candidates = [path]
    else:
        candidates = sorted(
            (BASE / '.cache').glob('*.sqlite'),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
    for path in candidates:
        try:
            with sqlite3.connect(f'file:{path.as_posix()}?mode=ro', uri=True) as con:
                row = con.execute(
                    'SELECT value FROM meta WHERE key=?', ('info',)
                ).fetchone()
                if row:
                    info = json.loads(row[0])
                    if info.get('source') and con.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='trips'"
                    ).fetchone():
                        return path, info
        except (sqlite3.Error, json.JSONDecodeError):
            continue
    raise FileNotFoundError(
        '找不到可用的既有 SQLite 索引。請先透過 FuelWise 建立索引；本腳本不會重新匯入 Excel。'
    )


def _weighted_mean(values, weights):
    mask = np.isfinite(values) & (weights > 0)
    total = float(weights[mask].sum())
    if not total:
        return None, 0.0
    return float(np.sum(values[mask] * weights[mask]) / total), total


def extract_features(analysis):
    """Derive non-fuel features from the normalized points returned by analytics."""
    summary = analysis['summary']
    quality = analysis['quality']
    points = analysis['points']
    duration = finite_number(summary.get('duration_sec')) or 0.0

    def column_values(name):
        return np.asarray(
            [finite_number(point.get(name)) for point in points],
            dtype=float,
        )

    dt = column_values('dt')
    dt = np.where(np.isfinite(dt) & (dt > 0), dt, 0.0)
    speeds = column_values('speed')
    rpms = column_values('rpm')
    loads = column_values('load')
    temps = column_values('temp')
    odometers = column_values('odo')
    speed_mean, speed_seconds = _weighted_mean(speeds, dt)
    if speed_seconds:
        speed_mask = np.isfinite(speeds) & (dt > 0)
        speed_std = float(
            np.sqrt(
                np.sum(dt[speed_mask] * (speeds[speed_mask] - speed_mean) ** 2)
                / speed_seconds
            )
        )
    else:
        speed_std = None
    rpm_mean, rpm_seconds = _weighted_mean(rpms, dt)
    load_mean, load_seconds = _weighted_mean(loads, dt)
    temp_mean, temp_seconds = _weighted_mean(temps, dt)

    elapsed_seconds = np.asarray(
        [finite_number(point.get('sec')) for point in points], dtype=float
    )
    interval_seconds = np.diff(elapsed_seconds)
    pair_valid = (
        (interval_seconds > 0)
        & (interval_seconds <= GAP_LIMIT_SECONDS)
        & np.isfinite(speeds[:-1])
        & np.isfinite(speeds[1:])
    )
    speed_changes = np.diff(speeds)
    valid_pair_seconds = float(interval_seconds[pair_valid].sum())
    abs_speed_change_mean = (
        float(np.abs(speed_changes[pair_valid]).mean())
        if pair_valid.any()
        else None
    )
    speed_ms = speeds / 3.6
    kinetic_increase = (
        float(
            np.maximum(
                0.0,
                (speed_ms[1:][pair_valid] ** 2 - speed_ms[:-1][pair_valid] ** 2)
                / 2.0,
            ).sum()
        )
        if pair_valid.any()
        else None
    )

    accel_valid = pair_valid & (
        interval_seconds <= ACCELERATION_MAX_INTERVAL_SECONDS
    )
    abs_accelerations = np.abs(
        speed_changes[accel_valid] / 3.6 / interval_seconds[accel_valid]
    )
    acceleration_mean, acceleration_seconds = _weighted_mean(
        abs_accelerations,
        interval_seconds[accel_valid],
    )

    known_idle_seconds = finite_number(summary.get('idle_known_sec')) or 0.0
    idle_seconds = finite_number(summary.get('idle_sec'))
    idle_share = (
        100 * idle_seconds / known_idle_seconds
        if idle_seconds is not None and known_idle_seconds > 0
        else None
    )

    features = {
        'distance_km': finite_number(summary.get('distance_km')),
        'duration_minutes': duration / 60 if duration > 0 else None,
        'speed_mean_kmh': speed_mean,
        'speed_std_kmh': speed_std,
        'rpm_mean': rpm_mean,
        'engine_load_mean_pct': load_mean,
        'coolant_temp_mean_c': temp_mean,
        'idle_engine_share_pct': idle_share,
        'speed_abs_change_mean_kmh': abs_speed_change_mean,
        'mean_abs_acceleration_m_s2': acceleration_mean,
        'kinetic_increase_j_per_kg': kinetic_increase,
        'kinetic_increase_j_per_kg_km': (
            kinetic_increase / summary['distance_km']
            if kinetic_increase is not None
            and finite_number(summary.get('distance_km')) is not None
            and summary['distance_km'] > 0
            else None
        ),
    }
    for feature, _, _ in SPEED_BANDS:
        if feature == 'speed_stopped_time_pct':
            mask = np.isfinite(speeds) & (speeds == 0)
        elif feature == 'speed_over_100_time_pct':
            mask = np.isfinite(speeds) & (speeds > 100)
        else:
            low, high = {
                'speed_0_20_time_pct': (0, 20),
                'speed_20_40_time_pct': (20, 40),
                'speed_40_60_time_pct': (40, 60),
                'speed_60_80_time_pct': (60, 80),
                'speed_80_100_time_pct': (80, 100),
            }[feature]
            mask = np.isfinite(speeds) & (speeds > low) & (speeds <= high)
        known_seconds = float(dt[mask].sum())
        features[feature] = (
            100 * known_seconds / speed_seconds if speed_seconds else None
        )

    odometer_pair_seconds = float(
        interval_seconds[
            (interval_seconds > 0)
            & (interval_seconds <= GAP_LIMIT_SECONDS)
            & np.isfinite(odometers[:-1])
            & np.isfinite(odometers[1:])
        ].sum()
    )
    coverages = {
        'distance_km': _coverage_pct(odometer_pair_seconds, duration),
        'duration_minutes': 100.0,
        'speed_mean_kmh': _coverage_pct(speed_seconds, duration),
        'speed_std_kmh': _coverage_pct(speed_seconds, duration),
        'rpm_mean': _coverage_pct(rpm_seconds, duration),
        'engine_load_mean_pct': _coverage_pct(load_seconds, duration),
        'coolant_temp_mean_c': _coverage_pct(temp_seconds, duration),
        'idle_engine_share_pct': _coverage_pct(known_idle_seconds, duration),
        'speed_abs_change_mean_kmh': _coverage_pct(valid_pair_seconds, duration),
        'mean_abs_acceleration_m_s2': _coverage_pct(acceleration_seconds, duration),
        'kinetic_increase_j_per_kg': _coverage_pct(valid_pair_seconds, duration),
        'kinetic_increase_j_per_kg_km': _coverage_pct(valid_pair_seconds, duration),
    }
    for feature, _, _ in SPEED_BANDS:
        coverages[feature] = _coverage_pct(speed_seconds, duration)
    feature_coverage = {
        f'coverage_pct__{feature}': coverage
        for feature, coverage in coverages.items()
    }
    quality_summary = {
        'quality_flags': summary.get('quality_flags'),
        'invalid_time_rows': quality.get('invalid_time_rows'),
        'duplicate_timestamps': quality.get('duplicate_timestamps'),
        'missing_gps_rows': quality.get('missing_gps_rows'),
        'gps_jump_segments': quality.get('gps_jump_segments'),
        'gap_seconds': quality.get('gap_seconds'),
        'max_gap_sec': quality.get('max_gap_sec'),
        'abnormal_can_rows': quality.get('abnormal_can_rows'),
        'can_status_available': quality.get('can_status_available'),
        'fuel_reset': quality.get('fuel_reset'),
        'odo_reset': quality.get('odo_reset'),
        'fuel_min_positive_step_l': quality.get('fuel_min_positive_step_l'),
        'odo_min_positive_step_km': quality.get('odo_min_positive_step_km'),
        'excluded_out_of_range': json.dumps(
            quality.get('excluded_out_of_range', {}),
            ensure_ascii=False,
            sort_keys=True,
        ),
        'missing_fuel_rows': quality.get('missing_per_column', {}).get(
            'can.engine.totalFuelUsed'
        ),
        'missing_odometer_rows': quality.get('missing_per_column', {}).get(
            'can.totalMileage'
        ),
    }
    return features, feature_coverage, quality_summary


def _coverage_pct(numerator, denominator):
    return min(100.0, 100.0 * numerator / denominator) if denominator > 0 else None


def _target_exclusion_reasons(analysis):
    summary = analysis['summary']
    quality = analysis['quality']
    points = analysis['points']
    reasons = []
    if quality.get('fuel_reset'):
        reasons.append('fuel_counter_reset')
    if quality.get('odo_reset'):
        reasons.append('odometer_counter_reset')
    if not points or points[0].get('fuel') is None or points[-1].get('fuel') is None:
        reasons.append('invalid_fuel_counter_endpoint')
    if not points or points[0].get('odo') is None or points[-1].get('odo') is None:
        reasons.append('invalid_odometer_endpoint')
    distance = finite_number(summary.get('distance_km'))
    fuel = finite_number(summary.get('fuel_l'))
    target = finite_number(summary.get('l100'))
    if distance is None:
        reasons.append('missing_or_nonfinite_distance')
    elif distance <= 0:
        reasons.append('zero_or_negative_distance')
    if fuel is None:
        reasons.append('missing_or_nonfinite_fuel')
    elif fuel < 0:
        reasons.append('negative_fuel')
    if target is None:
        reasons.append('missing_or_nonfinite_target')
    elif target < 0:
        reasons.append('negative_target')
    return sorted(set(reasons))


def load_training_rows(index_path, index_info):
    rows = []
    excluded = []
    with sqlite3.connect(
        f'file:{index_path.as_posix()}?mode=ro', uri=True
    ) as connection:
        trips = connection.execute(
            'SELECT id, vehicle, journey, summary FROM trips ORDER BY id'
        ).fetchall()
        for ordinal, (trip_id, vehicle, journey, summary_json) in enumerate(trips, 1):
            db_summary = json.loads(summary_json)
            raw_rows = [
                json.loads(item[0])
                for item in connection.execute(
                    'SELECT raw FROM points WHERE vehicle=? AND journey=? ORDER BY seq',
                    (vehicle, journey),
                )
            ]
            key = (str(vehicle), str(journey))
            try:
                analysis = analyze(raw_rows, detail=True, feature_only=True)
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                excluded.append(
                    {
                        'vehicle': key[0],
                        'journey': key[1],
                        'start': db_summary.get('start'),
                        'end': db_summary.get('end'),
                        'exclusion_reasons': f'analysis_failed: {exc}',
                    }
                )
                continue

            reasons = _target_exclusion_reasons(analysis)
            if reasons:
                excluded.append(
                    {
                        'vehicle': key[0],
                        'journey': key[1],
                        'start': analysis['summary'].get('start'),
                        'end': analysis['summary'].get('end'),
                        'exclusion_reasons': ';'.join(reasons),
                    }
                )
                continue

            features, coverages, quality = extract_features(analysis)
            if set(features) != set(FEATURE_COLUMNS):
                raise AssertionError('Feature columns do not match the declared schema.')
            start = pd.to_datetime(analysis['summary'].get('start'), errors='coerce')
            end = pd.to_datetime(analysis['summary'].get('end'), errors='coerce')
            if pd.isna(start) or pd.isna(end) or end < start:
                excluded.append(
                    {
                        'vehicle': key[0],
                        'journey': key[1],
                        'start': analysis['summary'].get('start'),
                        'end': analysis['summary'].get('end'),
                        'exclusion_reasons': 'invalid_trip_time_range',
                    }
                )
                continue
            row = {
                'trip_id': trip_id,
                'vehicle': key[0],
                'journey': key[1],
                'start': start.isoformat(sep=' '),
                'end': end.isoformat(sep=' '),
                'target_l_per_100km': float(analysis['summary']['l100']),
                'fuel_l': float(analysis['summary']['fuel_l']),
                **features,
                **coverages,
                **quality,
            }
            row['_start_dt'] = start
            row['_end_dt'] = end
            rows.append(row)
            if ordinal % 500 == 0:
                print(f'已整理 {ordinal:,}/{len(trips):,} 趟索引記錄', flush=True)

    keys = [(row['vehicle'], row['journey']) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('索引中存在重複的車輛＋journeyCode樣本，停止以避免重複樣本。')
    return rows, excluded, len(trips), index_info


def temporal_split(rows):
    """Split by start-time bands and discard trips crossing either time boundary."""
    if len(rows) < 3:
        raise ValueError('有效樣本不足3趟，無法建立訓練、驗證、測試三段時間切分。')
    ordered = sorted(rows, key=lambda row: (row['_start_dt'], row['_end_dt']))
    distinct_starts = sorted({row['_start_dt'] for row in ordered})
    if len(distinct_starts) < 3:
        raise ValueError('有效行程起始時間不足3個不同時間點，無法建立時間順序切分。')
    first_index = max(1, min(len(distinct_starts) - 2, math.floor(len(distinct_starts) * 0.70)))
    second_index = max(
        first_index + 1,
        min(len(distinct_starts) - 1, math.floor(len(distinct_starts) * 0.85)),
    )
    first_boundary = distinct_starts[first_index]
    second_boundary = distinct_starts[second_index]
    assignments = {}
    excluded = []
    for row in ordered:
        start, end = row['_start_dt'], row['_end_dt']
        key = (row['vehicle'], row['journey'])
        if end < start:
            excluded.append((key, 'invalid_trip_time_range'))
        elif start < first_boundary:
            if end <= first_boundary:
                assignments[key] = 'train'
            else:
                excluded.append((key, 'crosses_train_validation_boundary'))
        elif start < second_boundary:
            if end <= second_boundary:
                assignments[key] = 'validation'
            else:
                excluded.append((key, 'crosses_validation_test_boundary'))
        else:
            assignments[key] = 'test'

    sets = {
        split: [row for row in ordered if assignments.get((row['vehicle'], row['journey'])) == split]
        for split in ('train', 'validation', 'test')
    }
    if not all(sets.values()):
        raise ValueError(
            '嚴格時間切分及跨界行程排除後至少有一個集合為空；不會改用隨機切分或放寬邊界。'
        )
    return sets, excluded, first_boundary, second_boundary


def _metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = predicted - actual
    total_variance = float(np.sum((actual - actual.mean()) ** 2))
    return {
        'n': int(len(actual)),
        'mae_l_per_100km': float(np.mean(np.abs(errors))),
        'rmse_l_per_100km': float(np.sqrt(np.mean(errors**2))),
        'r2': (
            float(1 - np.sum(errors**2) / total_variance)
            if len(actual) >= 2 and total_variance > 0
            else None
        ),
    }


def _split_summary(rows):
    starts = [row['_start_dt'] for row in rows]
    ends = [row['_end_dt'] for row in rows]
    return {
        'trip_count': len(rows),
        'vehicle_count': len({row['vehicle'] for row in rows}),
        'start_date': min(starts).date().isoformat(),
        'end_date': max(ends).date().isoformat(),
    }


def _package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _assert_no_leakage():
    forbidden = [column for column in FEATURE_COLUMNS if any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS)]
    if forbidden:
        raise AssertionError(f'禁止的目標衍生或識別輸入特徵：{forbidden}')


def run_pipeline(index_path=None, output_root=DEFAULT_OUTPUT_ROOT):
    _assert_no_leakage()
    index_path, index_info = find_existing_index(index_path)
    rows, excluded, indexed_trip_count, index_info = load_training_rows(
        index_path, index_info
    )
    if len(rows) < 20:
        raise ValueError(
            f'只有 {len(rows)} 趟有效樣本；不足以依既定約70/15/15比例評估，停止且不改用其他切分。'
        )
    split_sets, split_excluded, first_boundary, second_boundary = temporal_split(rows)
    all_rows_by_key = {(row['vehicle'], row['journey']): row for row in rows}
    for key, reason in split_excluded:
        source = all_rows_by_key[key]
        excluded.append(
            {
                'vehicle': source['vehicle'],
                'journey': source['journey'],
                'start': source['start'],
                'end': source['end'],
                'exclusion_reasons': reason,
            }
        )

    split_by_key = {}
    for split, split_rows in split_sets.items():
        for row in split_rows:
            key = (row['vehicle'], row['journey'])
            split_by_key[key] = split
            row['split'] = split
    rows = [
        row
        for row in rows
        if (row['vehicle'], row['journey']) in split_by_key
    ]
    for row in rows:
        row['split'] = split_by_key[(row['vehicle'], row['journey'])]
    # Assignments are key based, so exact sample identity cannot appear in two sets.
    split_keys = {
        name: {(row['vehicle'], row['journey']) for row in split_rows}
        for name, split_rows in split_sets.items()
    }
    if (
        split_keys['train'] & split_keys['validation']
        or split_keys['train'] & split_keys['test']
        or split_keys['validation'] & split_keys['test']
    ):
        raise AssertionError('行程樣本跨集合重疊。')
    for left, right in (
        ('train', 'validation'),
        ('validation', 'test'),
    ):
        if max(row['_end_dt'] for row in split_sets[left]) > min(
            row['_start_dt'] for row in split_sets[right]
        ):
            raise AssertionError('集合間存在跨界行程的時間區間重疊。')

    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError(
            '缺少 XGBoost；請安裝 requirements.txt 中的 xgboost 依賴。'
        ) from exc

    train_rows = split_sets['train']
    validation_rows = split_sets['validation']
    test_rows = split_sets['test']
    x_train = pd.DataFrame(train_rows)[FEATURE_COLUMNS]
    y_train = np.asarray([row['target_l_per_100km'] for row in train_rows])
    x_validation = pd.DataFrame(validation_rows)[FEATURE_COLUMNS]
    y_validation = np.asarray(
        [row['target_l_per_100km'] for row in validation_rows]
    )
    x_test = pd.DataFrame(test_rows)[FEATURE_COLUMNS]
    y_test = np.asarray([row['target_l_per_100km'] for row in test_rows])
    per_vehicle_median = {}
    for row in train_rows:
        per_vehicle_median.setdefault(row['vehicle'], []).append(
            row['target_l_per_100km']
        )
    per_vehicle_median = {
        vehicle: float(np.median(values))
        for vehicle, values in per_vehicle_median.items()
    }
    global_median = float(np.median(y_train))
    baseline_predictions = np.asarray(
        [
            per_vehicle_median.get(row['vehicle'], global_median)
            for row in test_rows
        ],
        dtype=float,
    )

    parameters = {
        'objective': 'reg:squarederror',
        'max_depth': 3,
        'eta': 0.03,
        'min_child_weight': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'lambda': 10.0,
        'tree_method': 'hist',
        'eval_metric': 'mae',
        'seed': RANDOM_SEED,
        'nthread': max(1, min(4, os.cpu_count() or 1)),
        'verbosity': 0,
    }
    num_boost_round = 800
    early_stopping_rounds = 50
    d_train = xgb.DMatrix(x_train, label=y_train, feature_names=FEATURE_COLUMNS)
    d_validation = xgb.DMatrix(
        x_validation, label=y_validation, feature_names=FEATURE_COLUMNS
    )
    d_test = xgb.DMatrix(x_test, label=y_test, feature_names=FEATURE_COLUMNS)
    model = xgb.train(
        parameters,
        d_train,
        num_boost_round=num_boost_round,
        evals=[(d_validation, 'validation')],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    best_iteration = int(model.best_iteration)
    iteration_range = (0, best_iteration + 1)
    model_predictions = np.asarray(
        model.predict(d_test, iteration_range=iteration_range), dtype=float
    )
    output_root = Path(output_root).resolve()
    run_name = datetime.now().strftime('run_%Y%m%d_%H%M%S')
    output_dir = output_root / run_name
    suffix = 1
    while output_dir.exists():
        output_dir = output_root / f'{run_name}_{suffix}'
        suffix += 1
    output_dir.mkdir(parents=True)
    dataset_path = output_dir / 'trip_features.csv'
    pd.DataFrame(rows).drop(columns=['_start_dt', '_end_dt']).to_csv(
        dataset_path, index=False, encoding='utf-8-sig'
    )
    pd.DataFrame(excluded).to_csv(
        output_dir / 'excluded_trips.csv', index=False, encoding='utf-8-sig'
    )
    model_path = output_dir / 'xgboost_model.json'
    model.save_model(model_path)
    reloaded = xgb.Booster()
    reloaded.load_model(model_path)
    reloaded_predictions = np.asarray(
        reloaded.predict(d_test, iteration_range=iteration_range), dtype=float
    )
    if not np.allclose(model_predictions, reloaded_predictions, rtol=0, atol=1e-12):
        raise AssertionError('重新載入模型後預測不一致。')

    per_vehicle_test = []
    for vehicle in sorted({row['vehicle'] for row in test_rows}):
        positions = [i for i, row in enumerate(test_rows) if row['vehicle'] == vehicle]
        per_vehicle_test.append(
            {
                'vehicle': vehicle,
                'sample_count': len(positions),
                'baseline_mae_l_per_100km': _metrics(
                    y_test[positions], baseline_predictions[positions]
                )['mae_l_per_100km'],
                'xgboost_mae_l_per_100km': _metrics(
                    y_test[positions], model_predictions[positions]
                )['mae_l_per_100km'],
            }
        )
    test_predictions = pd.DataFrame(
        [
            {
                'vehicle': row['vehicle'],
                'journey': row['journey'],
                'start': row['start'],
                'end': row['end'],
                'actual_l_per_100km': y_test[index],
                'baseline_prediction_l_per_100km': baseline_predictions[index],
                'xgboost_prediction_l_per_100km': model_predictions[index],
                'baseline_error_prediction_minus_actual': baseline_predictions[index] - y_test[index],
                'xgboost_error_prediction_minus_actual': model_predictions[index] - y_test[index],
                'xgboost_absolute_error': abs(model_predictions[index] - y_test[index]),
                'xgboost_prediction_negative': bool(model_predictions[index] < 0),
            }
            for index, row in enumerate(test_rows)
        ]
    )
    test_predictions.to_csv(
        output_dir / 'test_predictions.csv', index=False, encoding='utf-8-sig'
    )
    split_report = {
        name: _split_summary(split_rows)
        for name, split_rows in split_sets.items()
    }
    evaluations = {
        'baseline_vehicle_training_median': _metrics(y_test, baseline_predictions),
        'xgboost': _metrics(y_test, model_predictions),
        'validation_xgboost': _metrics(
            y_validation,
            model.predict(d_validation, iteration_range=iteration_range),
        ),
        'xgboost_negative_prediction_count': int((model_predictions < 0).sum()),
        'xgboost_negative_prediction_fraction': float(
            np.mean(model_predictions < 0)
        ),
        'xgboost_best_iteration': best_iteration,
        'per_vehicle_test': per_vehicle_test,
    }
    metadata = {
        'task': 'trip-level L/100 km regression; predictive error is not driver waste or savings potential',
        'target': {
            'column': 'target_l_per_100km',
            'definition': '100 * effective cumulative fuel counter difference (L) / effective cumulative odometer difference (km)',
            'quality_source': 'analytics.analyze existing endpoint, CAN, and counter-reset checks',
        },
        'input_features': FEATURE_COLUMNS,
        'feature_definitions_and_units': FEATURE_DEFINITIONS,
        'feature_coverage_columns': {
            feature: f'coverage_pct__{feature}' for feature in FEATURE_COLUMNS
        },
        'excluded_input_columns': [
            'fuel_l', 'target_l_per_100km', 'L/100 km or km/L derivatives',
            'vehicle', 'journey', 'trip_id', 'source_row', 'scores or ranks',
        ],
        'split_method': {
            'kind': 'chronological start-time bands, approximately 70/15/15',
            'training_validation_boundary': first_boundary.isoformat(sep=' '),
            'validation_test_boundary': second_boundary.isoformat(sep=' '),
            'crossing_trip_policy': 'Drop any full trip interval that crosses either boundary; no random fallback.',
            'validation_use': 'Fixed XGBoost early stopping only; no test-based tuning.',
            'generalization_claim': 'Future trips under the observed fleet; not new-vehicle generalization.',
        },
        'split_counts_and_ranges': split_report,
        'indexed_trip_count': indexed_trip_count,
        'eligible_before_temporal_boundary_exclusion': len(rows) + len(split_excluded),
        'excluded_target_or_time_count': len(excluded) - len(split_excluded),
        'excluded_boundary_crossing_count': len(split_excluded),
        'exclusion_reason_counts': _reason_counts(excluded),
        'index': {
            'path': str(index_path),
            'file_size_bytes': index_path.stat().st_size,
            'file_mtime_ns': index_path.stat().st_mtime_ns,
            'source_name': index_info.get('source'),
            'source_row_count': index_info.get('rows'),
            'source_skipped_row_count': index_info.get('skipped_rows'),
        },
        'software_versions': {
            'python': sys.version.split()[0],
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'xgboost': _package_version('xgboost'),
        },
        'model': {
            'class': 'xgboost.Booster',
            'parameters': parameters,
            'num_boost_round': num_boost_round,
            'early_stopping_rounds': early_stopping_rounds,
            'random_seed': RANDOM_SEED,
            'missing_values': 'No learned imputation; XGBoost native missing-value routing.',
            'vehicle_id_used_as_feature': False,
        },
        'evaluation': evaluations,
        'model_reload_prediction_max_abs_difference': float(
            np.max(np.abs(model_predictions - reloaded_predictions))
        ),
    }
    (output_dir / 'feature_definitions.json').write_text(
        json.dumps(FEATURE_DEFINITIONS, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    (output_dir / 'evaluation.json').write_text(
        json.dumps(evaluations, ensure_ascii=False, indent=2, allow_nan=False),
        encoding='utf-8',
    )
    (output_dir / 'metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False),
        encoding='utf-8',
    )
    report = _render_report(metadata, evaluations, len(test_predictions))
    (output_dir / 'report.md').write_text(report, encoding='utf-8')
    return output_dir, metadata


def _reason_counts(excluded):
    counts = {}
    for row in excluded:
        for reason in row['exclusion_reasons'].split(';'):
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
    return counts


def _render_report(metadata, evaluations, test_count):
    split_lines = [
        f"| {name} | {summary['trip_count']} | {summary['vehicle_count']} | {summary['start_date']} – {summary['end_date']} |"
        for name, summary in metadata['split_counts_and_ranges'].items()
    ]
    exclusion_lines = [
        f'- `{reason}`: {count}'
        for reason, count in sorted(metadata['exclusion_reason_counts'].items())
    ] or ['- 無']
    baseline = evaluations['baseline_vehicle_training_median']
    xgb = evaluations['xgboost']
    per_vehicle = [
        f"| {row['vehicle']} | {row['sample_count']} | {row['baseline_mae_l_per_100km']:.3f} | {row['xgboost_mae_l_per_100km']:.3f} |"
        for row in evaluations['per_vehicle_test']
    ]
    return '\n'.join(
        [
            '# 行程層級油耗模型訓練與評估',
            '',
            '- 預測誤差僅代表模型預測與歷史實際值的差異，不代表駕駛浪費或可節省油量。',
            '- 評估依時間往後切分；不代表已驗證新車輛泛化。',
            '- 測試集所有符合目標品質規則的行程均保留，未按預測難度篩除。',
            '',
            '## 資料與切分',
            '',
            f"- 索引行程：{metadata['indexed_trip_count']:,}",
            f"- 進入時間切分前有效行程：{metadata['eligible_before_temporal_boundary_exclusion']:,}",
            f"- 目標／時間品質排除：{metadata['excluded_target_or_time_count']:,}",
            f"- 跨時間邊界排除：{metadata['excluded_boundary_crossing_count']:,}",
            f"- 測試預測行程：{test_count:,}",
            '',
            '| 集合 | 行程數 | 車輛數 | 日期範圍 |',
            '|---|---:|---:|---|',
            *split_lines,
            '',
            '### 排除原因',
            '',
            '同一趟可能有多項品質問題，因此下列原因計數可重疊；有效樣本排除總數按唯一行程計算。',
            '',
            *exclusion_lines,
            '',
            '## 方法與測試集結果',
            '',
            '| 方法 | MAE (L/100 km) | RMSE (L/100 km) | R² |',
            '|---|---:|---:|---:|',
            f"| 訓練集車輛中位數（未見車輛回退全體中位數） | {baseline['mae_l_per_100km']:.3f} | {baseline['rmse_l_per_100km']:.3f} | {_fmt_metric(baseline['r2'])} |",
            f"| XGBoost | {xgb['mae_l_per_100km']:.3f} | {xgb['rmse_l_per_100km']:.3f} | {_fmt_metric(xgb['r2'])} |",
            '',
            f"- XGBoost 負預測：{evaluations['xgboost_negative_prediction_count']}/{test_count}（{evaluations['xgboost_negative_prediction_fraction']:.1%}）；未截斷或隱藏。",
            f"- Early stopping 最佳迭代：{evaluations['xgboost_best_iteration']}",
            f"- 重新載入模型後最大預測差：{metadata['model_reload_prediction_max_abs_difference']:.3g}",
            '',
            '### 測試集各車輛 MAE',
            '',
            '| 車輛 | 行程數 | 基準 MAE | XGBoost MAE |',
            '|---|---:|---:|---:|',
            *per_vehicle,
            '',
            '## 輸入特徵',
            '',
            '| 特徵 | 單位 | 定義 |',
            '|---|---|---|',
            *[
                f"| `{feature}` | {definition['unit']} | {definition['definition']} |"
                for feature, definition in metadata['feature_definitions_and_units'].items()
            ],
            '',
            '累積單位質量正向動能變化僅是取樣尺度下的加速活動代理，不是實際燃油能量；不考慮車重、阻力或傳動效率。引擎負載不是貨物載重。',
            '',
            '逐趟特徵、品質涵蓋率、排除原因、模型設定及資料索引識別資訊見同目錄的 CSV/JSON 檔案。',
            '',
        ]
    )


def _fmt_metric(value):
    return '不適用（樣本不足或實際值無變異）' if value is None else f'{value:.3f}'


def main():
    parser = argparse.ArgumentParser(
        description='直接使用既有 FuelWise SQLite 索引訓練與評估行程層級油耗模型。'
    )
    parser.add_argument(
        '--index',
        type=Path,
        help='既有 SQLite 索引路徑；預設使用 .cache 中最新可用索引。',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help='輸出根目錄；本次結果會寫入新的時間戳子目錄。',
    )
    args = parser.parse_args()
    try:
        output_dir, metadata = run_pipeline(args.index, args.output)
    except (FileNotFoundError, OSError, sqlite3.Error, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(
        f"完成：train={metadata['split_counts_and_ranges']['train']['trip_count']:,}, "
        f"validation={metadata['split_counts_and_ranges']['validation']['trip_count']:,}, "
        f"test={metadata['split_counts_and_ranges']['test']['trip_count']:,}\n"
        f"輸出：{output_dir}",
        flush=True,
    )


if __name__ == '__main__':
    main()
