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


def analyze(records, detail=True, gap_limit=120):
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
