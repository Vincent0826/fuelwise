"""Pure, testable trip analytics. No LLM-generated numbers."""
import json
import math
import re

import numpy as np
import pandas as pd

FIELDS = {
    'lat': 'gps.latitude',
    'lon': 'gps.longitude',
    'speed': 'can.canSpeed',
    'gps_speed': 'gps.speed',
    'odo': 'can.totalMileage',
    'fuel': 'can.engine.totalFuelUsed',
    'rpm': 'can.engine.rpm',
    'load': 'can.engine.engineLoad',
    'temp': 'can.engine.engineCoolantTemp',
    'limit': 'gps.speedLimit',
    'battery': 'line.battery',
    'fuel_level': 'line.fuelLevel',
    'can_status': 'can.canStatus',
}
SEGMENT_WINDOW_SECONDS = 15 * 60
MIN_SEGMENT_FUEL_UPDATES = 3
QUALITY_REASON_LABELS = {
    'non_positive_interval': '非正取樣間隔',
    'long_gap': '相鄰觀測間隔超過120秒，已排除',
    'invalid_can': '區間端點CAN狀態無效',
    'missing_fuel': '燃油計數器端點缺值',
    'cross_window_fuel': '有效燃油增量跨越時間窗邊界，未分配',
    'fuel_counter_reset': '燃油計數器倒退，該區間未計算',
    'missing_odometer': '里程計數器端點缺值',
    'odo_counter_reset': '里程計數器倒退，該區間未計算',
    'missing_speed': '車速缺值',
    'missing_rpm': 'RPM缺值',
}
EVENT_NAMES = dict(
    enumerate(
        [
            '行駛', '怠速', '熄火', '引擎啟動', '圍籬', '急加速', '急減速', '超速',
            'DTC故障碼', '引擎過熱', '引擎過載', '疲勞駕駛', '未繫安全帶', '使用通訊',
            '抽菸', '分心', '鏡頭遮蔽', '駕駛消失', '預警防護', '車道偏離',
        ],
        1,
    )
)

def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat(sep=' ')
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    return 12742000 * np.arcsin(
        np.sqrt(
            np.clip(
                np.sin(dp / 2) ** 2
                + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2,
                0,
                1,
            )
        )
    )


def analyze_segments(a, valid_can, gap_limit=120, window_seconds=SEGMENT_WINDOW_SECONDS):
    """Summarize fixed windows using only adjacent observations within each window."""
    if window_seconds <= 0:
        raise ValueError('時間窗長度必須大於0')

    elapsed = max(0.0, float(a.sec.iloc[-1]))
    window_count = max(1, math.ceil(elapsed / window_seconds))
    accumulators = []
    for index in range(window_count):
        start_sec = index * window_seconds
        end_sec = min((index + 1) * window_seconds, elapsed)
        accumulators.append(
            {
                'index': index,
                'start_sec': start_sec,
                'end_sec': end_sec,
                'observation_indices': [],
                'reasons': set(),
                'excluded_intervals': {},
                'valid_coverage_sec': 0.0,
                'fuel_coverage_sec': 0.0,
                'odo_coverage_sec': 0.0,
                'speed_sum': 0.0,
                'speed_coverage_sec': 0.0,
                'rpm_sum': 0.0,
                'rpm_coverage_sec': 0.0,
                'fuel_l': 0.0,
                'fuel_intervals': 0,
                'fuel_updates': 0,
                'odo_km': 0.0,
                'odo_intervals': 0,
                'cross_boundary_intervals': 0,
            }
        )

    def window_indices_for_interval(start_sec, end_sec):
        first = max(0, int(start_sec // window_seconds))
        last = min(window_count - 1, int(end_sec // window_seconds))
        return [
            index
            for index in range(first, last + 1)
            if min(end_sec, accumulators[index]['end_sec'])
            > max(start_sec, accumulators[index]['start_sec'])
        ]

    def add_exclusion(reason, indices, totals):
        totals[reason] = totals.get(reason, 0) + 1
        for index in indices:
            window = accumulators[index]
            window['reasons'].add(reason)
            window['excluded_intervals'][reason] = (
                window['excluded_intervals'].get(reason, 0) + 1
            )

    for index, second in enumerate(a.sec):
        window_index = min(window_count - 1, int(float(second) // window_seconds))
        accumulators[window_index]['observation_indices'].append(index)

    excluded_intervals = {}
    unassigned_intervals = []
    unassigned_fuel_l = 0.0
    unassigned_fuel_count = 0
    unassigned_odo_km = 0.0
    unassigned_odo_count = 0
    excluded_long_gap_fuel_l = 0.0
    excluded_long_gap_fuel_count = 0
    observed_fuel_intervals = 0
    allocated_fuel_intervals = 0
    observed_odo_intervals = 0

    for index in range(len(a) - 1):
        start_sec = float(a.sec.iloc[index])
        end_sec = float(a.sec.iloc[index + 1])
        interval_sec = end_sec - start_sec
        if interval_sec <= 0:
            bucket = min(window_count - 1, int(start_sec // window_seconds))
            add_exclusion('non_positive_interval', [bucket], excluded_intervals)
            continue

        window_indices = window_indices_for_interval(start_sec, end_sec)
        if interval_sec > gap_limit:
            add_exclusion('long_gap', window_indices, excluded_intervals)
            if valid_can.iloc[index] and valid_can.iloc[index + 1]:
                fuel_start, fuel_end = a.fuel.iloc[index : index + 2]
                fuel_delta = fuel_end - fuel_start
                if pd.notna(fuel_delta) and fuel_delta >= -1e-8:
                    excluded_long_gap_fuel_l += max(0.0, float(fuel_delta))
                    excluded_long_gap_fuel_count += 1
                elif pd.notna(fuel_delta):
                    add_exclusion(
                        'fuel_counter_reset', window_indices, excluded_intervals
                    )
                odo_start, odo_end = a.odo.iloc[index : index + 2]
                if pd.notna(odo_start) and pd.notna(odo_end) and odo_end < odo_start - 1e-8:
                    add_exclusion(
                        'odo_counter_reset', window_indices, excluded_intervals
                    )
            continue

        if not (valid_can.iloc[index] and valid_can.iloc[index + 1]):
            add_exclusion('invalid_can', window_indices, excluded_intervals)
            continue

        for window_index in window_indices:
            window = accumulators[window_index]
            overlap_sec = min(end_sec, window['end_sec']) - max(
                start_sec, window['start_sec']
            )
            window['valid_coverage_sec'] += overlap_sec
            speed = a.speed.iloc[index]
            rpm = a.rpm.iloc[index]
            if pd.notna(speed):
                window['speed_sum'] += float(speed) * overlap_sec
                window['speed_coverage_sec'] += overlap_sec
            else:
                window['reasons'].add('missing_speed')
            if pd.notna(rpm):
                window['rpm_sum'] += float(rpm) * overlap_sec
                window['rpm_coverage_sec'] += overlap_sec
            else:
                window['reasons'].add('missing_rpm')

        start_window = min(window_count - 1, int(start_sec // window_seconds))
        end_window = min(window_count - 1, int(end_sec // window_seconds))
        same_window = start_window == end_window
        boundary_fuel_delta = None
        boundary_odo_delta = None

        fuel_start, fuel_end = a.fuel.iloc[index : index + 2]
        if pd.isna(fuel_start) or pd.isna(fuel_end):
            add_exclusion('missing_fuel', window_indices, excluded_intervals)
        else:
            fuel_delta = float(fuel_end - fuel_start)
            if fuel_delta < -1e-8:
                add_exclusion('fuel_counter_reset', window_indices, excluded_intervals)
            else:
                observed_fuel_intervals += 1
                if same_window:
                    window = accumulators[start_window]
                    window['fuel_l'] += max(0.0, fuel_delta)
                    window['fuel_intervals'] += 1
                    window['fuel_coverage_sec'] += interval_sec
                    allocated_fuel_intervals += 1
                    if fuel_delta > 1e-8:
                        window['fuel_updates'] += 1
                else:
                    boundary_fuel_delta = max(0.0, fuel_delta)
                    unassigned_fuel_l += max(0.0, fuel_delta)
                    unassigned_fuel_count += 1
                    for window_index in window_indices:
                        window = accumulators[window_index]
                        window['cross_boundary_intervals'] += 1
                        window['reasons'].add('cross_window_fuel')

        odo_start, odo_end = a.odo.iloc[index : index + 2]
        if pd.isna(odo_start) or pd.isna(odo_end):
            add_exclusion('missing_odometer', window_indices, excluded_intervals)
        else:
            odo_delta = float(odo_end - odo_start)
            if odo_delta < -1e-8:
                add_exclusion('odo_counter_reset', window_indices, excluded_intervals)
            else:
                observed_odo_intervals += 1
                if same_window:
                    window = accumulators[start_window]
                    window['odo_km'] += max(0.0, odo_delta)
                    window['odo_intervals'] += 1
                    window['odo_coverage_sec'] += interval_sec
                else:
                    boundary_odo_delta = max(0.0, odo_delta)
                    unassigned_odo_km += max(0.0, odo_delta)
                    unassigned_odo_count += 1
        if not same_window and (
            boundary_fuel_delta is not None or boundary_odo_delta is not None
        ):
            unassigned_intervals.append(
                {
                    'from_time': str(a.time.iloc[index]),
                    'to_time': str(a.time.iloc[index + 1]),
                    'from_window_index': start_window,
                    'to_window_index': end_window,
                    'fuel_l': boundary_fuel_delta,
                    'distance_km': boundary_odo_delta,
                }
            )

    windows = []
    for window in accumulators:
        indices = window['observation_indices']
        duration = window['end_sec'] - window['start_sec']
        support_sec = window['valid_coverage_sec']
        fuel_coverage = window['fuel_coverage_sec']
        coverage_pct = min(100.0, 100 * support_sec / duration) if duration else None
        fuel_coverage_pct = (
            min(100.0, 100 * fuel_coverage / duration) if duration else None
        )
        if not indices:
            observed_start = observed_end = None
            valid_observations = 0
        else:
            observed_start = str(a.time.iloc[indices[0]])
            observed_end = str(a.time.iloc[indices[-1]])
            valid_observations = sum(bool(valid_can.iloc[item]) for item in indices)

        if window['fuel_intervals'] == 0 and window['cross_boundary_intervals']:
            quality_status = '解析度不足'
            window['reasons'].add('cross_window_fuel')
        elif window['fuel_intervals'] == 0:
            quality_status = '資料不足'
        elif window['fuel_updates'] == 0:
            quality_status = '未觀測到增量'
        elif window['fuel_updates'] < MIN_SEGMENT_FUEL_UPDATES:
            quality_status = '解析度不足'
        elif coverage_pct is not None and coverage_pct < 100 - 1e-6:
            quality_status = '部分涵蓋'
        else:
            quality_status = '可計算（部分觀測）'

        quality_reasons = sorted(
            QUALITY_REASON_LABELS[reason] for reason in window['reasons']
        )
        if 0 < window['fuel_updates'] < MIN_SEGMENT_FUEL_UPDATES:
            quality_reasons.append(
                f'正向燃油計數器更新 {window["fuel_updates"]}/{MIN_SEGMENT_FUEL_UPDATES} 次'
            )
        windows.append(
            {
                'index': window['index'],
                'window_start_sec': window['start_sec'],
                'window_end_sec': window['end_sec'],
                'window_start': str(a.time.iloc[0] + pd.to_timedelta(window['start_sec'], unit='s')),
                'window_end': str(a.time.iloc[0] + pd.to_timedelta(window['end_sec'], unit='s')),
                'last_partial_window': duration < window_seconds,
                'observed_start': observed_start,
                'observed_end': observed_end,
                'observation_count': len(indices),
                'valid_can_observations': valid_observations,
                'fuel_l': window['fuel_l'] if window['fuel_intervals'] else None,
                'fuel_status': (
                    'no_increment_observed'
                    if window['fuel_intervals'] and window['fuel_updates'] == 0
                    else 'observed'
                    if window['fuel_intervals']
                    else 'unavailable'
                ),
                'fuel_interval_count': window['fuel_intervals'],
                'fuel_positive_update_count': window['fuel_updates'],
                'distance_km': window['odo_km'] if window['odo_intervals'] else None,
                'distance_interval_count': window['odo_intervals'],
                'avg_speed_kmh': (
                    window['speed_sum'] / window['speed_coverage_sec']
                    if window['speed_coverage_sec']
                    else None
                ),
                'speed_coverage_sec': window['speed_coverage_sec'],
                'avg_rpm': (
                    window['rpm_sum'] / window['rpm_coverage_sec']
                    if window['rpm_coverage_sec']
                    else None
                ),
                'rpm_coverage_sec': window['rpm_coverage_sec'],
                'valid_coverage_sec': support_sec,
                'coverage_pct': coverage_pct,
                'fuel_coverage_pct': fuel_coverage_pct,
                'cross_boundary_interval_count': window['cross_boundary_intervals'],
                'excluded_intervals': window['excluded_intervals'],
                'quality_status': quality_status,
                'quality_reasons': quality_reasons,
            }
        )

    return {
        'window_seconds': window_seconds,
        'minimum_fuel_updates_per_window': MIN_SEGMENT_FUEL_UPDATES,
        'window_count': window_count,
        'windows': windows,
        'summary': {
            'allocated_fuel_l': (
                sum(window['fuel_l'] or 0.0 for window in accumulators)
                if allocated_fuel_intervals
                else None
            ),
            'unassigned_cross_window_fuel_l': (
                unassigned_fuel_l if unassigned_fuel_count else None
            ),
            'unassigned_cross_window_fuel_interval_count': unassigned_fuel_count,
            'excluded_long_gap_fuel_l': (
                excluded_long_gap_fuel_l if excluded_long_gap_fuel_count else None
            ),
            'excluded_long_gap_fuel_interval_count': excluded_long_gap_fuel_count,
            'unassigned_cross_window_distance_km': (
                unassigned_odo_km if unassigned_odo_count else None
            ),
            'unassigned_cross_window_distance_interval_count': unassigned_odo_count,
            'observed_fuel_interval_count': observed_fuel_intervals,
            'observed_distance_interval_count': observed_odo_intervals,
            'excluded_intervals': excluded_intervals,
            'measurement_note': (
                '分段耗油僅加總兩端點位於同一窗且通過品質檢查的相鄰計數器差；'
                '跨窗增量不插值，長缺口、無效CAN狀態與計數器倒退區間不計入。'
            ),
        },
        'unassigned_cross_window_intervals': unassigned_intervals,
    }


def analyze(records, detail=True, gap_limit=120, feature_only=False):
    d = pd.DataFrame(records)
    d['time'] = pd.to_datetime(d['time'], errors='coerce')
    invalid_time = int(d.time.isna().sum())
    d = d[d.time.notna()].sort_values('time', kind='stable').reset_index(drop=True)
    if d.empty:
        raise ValueError('沒有有效時間')

    a = pd.DataFrame(
        {
            key: pd.to_numeric(
                d.get(column, pd.Series(np.nan, index=d.index)), errors='coerce'
            )
            for key, column in FIELDS.items()
        }
    )
    a['time'] = d.time
    a['sec'] = (d.time - d.time.iloc[0]).dt.total_seconds()
    delta = a.sec.shift(-1) - a.sec
    dt = delta.where((delta > 0) & (delta <= gap_limit), 0).fillna(0)
    valid_can = (
        a.can_status.eq(0)
        if 'can.canStatus' in d
        else pd.Series(True, index=d.index)
    )

    # Unknown/abnormal CAN readings stay in raw table but are not trusted metrics.
    for k in ['speed','odo','fuel','rpm','load','temp']:
        a.loc[~valid_can, k] = np.nan

    ranges = {
        'speed': (0, 255),
        'gps_speed': (0, 255.9),
        'rpm': (0, 8031.875),
        'load': (0, 250),
        'temp': (-40, 210),
        'battery': (0, 30000),
        'fuel_level': (0, 100),
    }
    out_of_range = {}
    for k, (lo, hi) in ranges.items():
        bad = a[k].notna() & ~a[k].between(lo, hi)
        out_of_range[k] = int(bad.sum())
        a.loc[bad, k] = np.nan

    a['battery'] = a.battery / 1000  # source dictionary unit: millivolts
    a.loc[a.limit <= 0, 'limit'] = np.nan
    pos = (
        a.lat.between(-90, 90)
        & a.lon.between(-180, 180)
        & ~((a.lat == 0) & (a.lon == 0))
    )
    a.loc[~pos, ['lat', 'lon']] = np.nan
    a['gps_step_m'] = hav(a.lat.shift(), a.lon.shift(), a.lat, a.lon)
    prev_dt = a.sec.diff()
    jump = (a.gps_step_m / (prev_dt / 3600) / 1000 > 160) & (prev_dt > 0)
    a['break_before'] = (
        jump | ~pos | ~pos.shift(fill_value=False) | (prev_dt > gap_limit)
    ).fillna(True)
    quality = {
        'invalid_time_rows': invalid_time,
        'duplicate_timestamps': int(a.sec.duplicated().sum()),
        'missing_gps_rows': int((~pos).sum()),
        'gps_jump_segments': int(jump.sum()),
        'abnormal_can_rows': int((~valid_can).sum()),
        'can_status_available': 'can.canStatus' in d,
        'gap_seconds': float(delta[delta > gap_limit].sum()),
        'max_gap_sec': float(delta.max()) if len(a) > 1 else 0,
        'excluded_out_of_range': out_of_range,
        'gap_threshold_sec': gap_limit,
    }
    for k, derived in [('odo', 'distance'), ('fuel', 'used')]:
        reset = bool((a[k].diff().dropna() < -1e-8).any())
        quality[k + '_reset'] = reset
        # Do not silently bridge missing endpoint counters.
        okay = not reset and pd.notna(a[k].iloc[0]) and pd.notna(a[k].iloc[-1])
        a[derived] = a[k] - a[k].iloc[0] if okay else np.nan

    a['idle'] = a.speed.eq(0) & (a.rpm > 0)
    a['accel'] = a.speed.diff() / 3.6 / prev_dt.where(
        (prev_dt > 0) & (prev_dt <= 30)
    )
    elapsed = float(a.sec.iloc[-1])
    distance = a.distance.iloc[-1]
    fuel = a.used.iloc[-1]
    speed_known = float(dt[a.speed.notna()].sum())
    idle_known = float(dt[a.speed.notna() & a.rpm.notna()].sum())
    idle = float(dt[a.idle].sum())
    over = a.speed.notna() & a.limit.notna()
    overspeed = float(dt[over & (a.speed > a.limit)].sum())

    def weighted(key):
        known_seconds = dt[a[key].notna()].sum()
        if known_seconds > 0:
            return float((a[key] * dt).sum() / known_seconds)
        return np.nan

    gpsgood = (
        (prev_dt > 0)
        & (prev_dt <= gap_limit)
        & ~jump
        & a.gps_step_m.notna()
    )
    summary = {
        'start': str(a.time.iloc[0]),
        'end': str(a.time.iloc[-1]),
        'rows': len(a),
        'duration_sec': elapsed,
        'distance_km': distance,
        'fuel_l': fuel,
        'l100': 100 * fuel / distance
        if distance > 0 and pd.notna(fuel)
        else np.nan,
        'km_l': distance / fuel
        if fuel > 0 and pd.notna(distance)
        else np.nan,
        'idle_sec': idle,
        'idle_pct': 100 * idle / idle_known if idle_known else np.nan,
        'idle_known_sec': idle_known,
        'speed_mean': weighted('speed'),
        'speed_max': a.speed.max(),
        'rpm_mean': weighted('rpm'),
        'rpm_max': a.rpm.max(),
        'temp_max': a.temp.max(),
        'load_mean': weighted('load'),
        'overspeed_sec': overspeed,
        'speed_limit_known_sec': float(dt[over].sum()),
        'speed_known_sec': speed_known,
        'gps_distance_km': float(a.loc[gpsgood, 'gps_step_m'].sum() / 1000),
        'start_lat': a.lat.iloc[0],
        'start_lon': a.lon.iloc[0],
        'end_lat': a.lat.iloc[-1],
        'end_lon': a.lon.iloc[-1],
        'quality_flags': sum(
            [
                quality['duplicate_timestamps'] > 0,
                quality['gap_seconds'] > 0,
                quality['gps_jump_segments'] > 0,
                quality['abnormal_can_rows'] > 0,
                quality['odo_reset'],
                quality['fuel_reset'],
            ]
        ),
    }
    if not detail:
        return clean(summary)

    if feature_only:
        quality['missing_per_column'] = {
            column: int(d[column].isna().sum()) for column in d.columns
        }
        quality['fuel_min_positive_step_l'] = (
            a.fuel.diff().where(lambda values: values > 0).min()
        )
        quality['odo_min_positive_step_km'] = (
            a.odo.diff().where(lambda values: values > 0).min()
        )
        a['index'] = np.arange(len(a))
        a['dt'] = dt
        return clean(
            {
                'summary': summary,
                'points': a.to_dict('records'),
                'quality': quality,
            }
        )

    segments = analyze_segments(a, valid_can, gap_limit)
    stops = []
    i = 0
    while i < len(a) - 1:
        if not a.idle.iloc[i] or dt.iloc[i] <= 0:
            i += 1
            continue

        start = i
        seconds = 0
        while i < len(a) - 1 and a.idle.iloc[i] and dt.iloc[i] > 0:
            seconds += float(dt.iloc[i])
            i += 1
        if seconds >= 60:
            subset = a.iloc[start:i]
            stops.append(
                {
                    'index': start,
                    'end_index': i,
                    'start': str(a.time.iloc[start]),
                    'end': str(a.time.iloc[i]),
                    'seconds': seconds,
                    'lat': subset.lat.median(),
                    'lon': subset.lon.median(),
                    'type': '停車引擎運轉 ≥60秒',
                    'source': 'CAN推估',
                }
            )

    events = []
    seen = set()
    slots = sorted(set(re.findall(r'event\[(\d+)\]\.type', ' '.join(d.columns))))
    for idx,row in d.iterrows():
        for slot in slots:
            prefix = f'event[{slot}].'
            typ = row.get(prefix + 'type')
            if pd.isna(typ) or str(typ).strip() in ('', '0', '0.0'):
                continue

            raw = {
                c[len(prefix):]: clean(row[c])
                for c in d.columns
                if c.startswith(prefix) and pd.notna(row[c])
            }
            stamp = raw.get('startTime', str(row.time))
            end = raw.get('endTime')
            signature = (
                str(typ),
                str(stamp),
                str(end),
                json.dumps(raw, ensure_ascii=False, sort_keys=True),
            )
            if signature in seen:
                continue
            seen.add(signature)

            try:
                num = int(float(str(typ).split('.')[0]))
                name = EVENT_NAMES.get(num, str(typ))
            except ValueError:
                name = str(typ)

            et = pd.to_datetime(stamp, errors='coerce')
            nearest = int((a.time - et).abs().argmin()) if pd.notna(et) else idx
            difference = (
                abs((a.time.iloc[nearest] - et).total_seconds())
                if pd.notna(et)
                else np.nan
            )
            located = pd.notna(difference) and difference <= gap_limit
            events.append(
                {
                    'type': name,
                    'code': str(typ),
                    'start': str(stamp),
                    'end': end,
                    'index': nearest,
                    'lat': a.lat.iloc[nearest] if located else None,
                    'lon': a.lon.iloc[nearest] if located else None,
                    'location_offset_sec': difference,
                    'source': '原始事件（按欄位內容去重）',
                    'info': raw,
                }
            )

    # Speed bands are time weighted, with a separate coverage denominator.
    bands = []
    for lo, hi, label in [
        (-1, 0, '停止'),
        (0, 20, '0–20'),
        (20, 40, '20–40'),
        (40, 60, '40–60'),
        (60, 80, '60–80'),
        (80, 100, '80–100'),
        (100, float('inf'), '>100'),
    ]:
        seconds = float(dt[(a.speed > lo) & (a.speed <= hi)].sum())
        bands.append(
            {
                'band': label,
                'minutes': seconds / 60,
                'pct': 100 * seconds / speed_known if speed_known else None,
            }
        )

    quality['missing_per_column'] = {
        c: int(d[c].isna().sum()) for c in d.columns
    }
    quality['fuel_min_positive_step_l'] = a.fuel.diff().where(lambda v: v > 0).min()
    quality['odo_min_positive_step_km'] = a.odo.diff().where(lambda v: v > 0).min()
    a['index'] = np.arange(len(a))
    a['dt'] = dt
    notes = [
        f'本趟 {len(a):,} 筆紀錄，歷時 {elapsed/60:.1f} 分鐘。',
        f'可判讀區間內停車引擎運轉 {idle/60:.1f} 分鐘。',
    ]
    #if pd.notna(fuel) and pd.notna(distance):
    #    notes.append(
    #        f'累積計數器差：{distance:.2f} km、{fuel:.2f} L。'
    #    )
    if stops:
        notes.append(
            f'最長連續停車引擎運轉 '
            f'{max(s["seconds"] for s in stops)/60:.1f} 分鐘，可於地圖定位。'
        )
    return clean(
        {
            'summary': summary,
            'points': a.to_dict('records'),
            'segments': segments,
            'events': events,
            'stops': stops,
            'bands': bands,
            'quality': quality,
            'notes': notes,
            'raw': d.assign(time=d.time.astype(str))
            .where(pd.notna(d), None)
            .to_dict('records'),
        }
    )
