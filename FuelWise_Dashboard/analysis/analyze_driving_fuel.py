"""Compare repeated harsh acceleration/braking with per-trip fuel economy."""

import argparse
import json
import math
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analytics import analyze
from importer import prepare

BASE = Path(__file__).resolve().parent.parent
DEFAULT_THRESHOLD_KMH_S = 3.0
DEFAULT_MIN_CONSECUTIVE = 3
DEFAULT_MAX_INTERVAL_SECONDS = 30.0
DEFAULT_MIN_DISTANCE_KM = 1.0
EVENT_LABELS = {
    'acceleration': '急加速',
    'braking': '急減速',
}
EVENT_GROUPS = ('0 次', '1 次', '2 次以上')


def count_harsh_events(
    points,
    threshold_kmh_s=DEFAULT_THRESHOLD_KMH_S,
    min_consecutive=DEFAULT_MIN_CONSECUTIVE,
    max_interval_seconds=DEFAULT_MAX_INTERVAL_SECONDS,
):
    """Count runs of threshold-exceeding acceleration across valid observations."""
    events = []
    direction = 0
    streak = 0
    active_event = None

    for index in range(1, len(points)):
        point = points[index]
        previous = points[index - 1]
        elapsed = point['sec'] - previous['sec']
        speed = point.get('speed')
        previous_speed = previous.get('speed')
        valid = (
            not point.get('break_before', False)
            and speed is not None
            and previous_speed is not None
            and math.isfinite(float(speed))
            and math.isfinite(float(previous_speed))
            and math.isfinite(float(elapsed))
            and 0 < elapsed <= max_interval_seconds
        )
        acceleration = (
            (float(speed) - float(previous_speed)) / elapsed
            if valid
            else None
        )
        next_direction = (
            1
            if acceleration is not None and acceleration > threshold_kmh_s
            else -1
            if acceleration is not None and acceleration < -threshold_kmh_s
            else 0
        )

        if next_direction != direction:
            direction = next_direction
            streak = 0
            active_event = None

        if not next_direction:
            continue

        streak += 1
        magnitude = abs(acceleration)
        if streak == min_consecutive:
            active_event = {
                'type': 'acceleration' if direction > 0 else 'braking',
                'start_sec': points[index - min_consecutive]['sec'],
                'end_sec': point['sec'],
                'max_kmh_s': magnitude,
            }
            events.append(active_event)
        elif streak > min_consecutive and active_event is not None:
            active_event['end_sec'] = point['sec']
            active_event['max_kmh_s'] = max(active_event['max_kmh_s'], magnitude)

    return events


def spearman_correlation(x, y):
    """Calculate Spearman correlation using average ranks, without SciPy."""
    pairs = pd.DataFrame({'x': x, 'y': y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pairs) < 3 or pairs['x'].nunique() < 2 or pairs['y'].nunique() < 2:
        return None
    corr = pairs['x'].rank(method='average').corr(
        pairs['y'].rank(method='average')
    )
    return float(corr) if pd.notna(corr) else None


def event_group(count):
    if count == 0:
        return EVENT_GROUPS[0]
    if count == 1:
        return EVENT_GROUPS[1]
    return EVENT_GROUPS[2]


def event_points_from_records(records):
    data = pd.DataFrame(records)
    if data.empty or 'time' not in data:
        return []
    data['time'] = pd.to_datetime(data['time'], format='mixed', errors='coerce')
    data = data[data['time'].notna()].sort_values('time', kind='stable').reset_index(drop=True)
    if data.empty:
        return []

    speed = pd.to_numeric(
        data.get('can.canSpeed', pd.Series(np.nan, index=data.index)),
        errors='coerce',
    )
    valid_can = (
        pd.to_numeric(data['can.canStatus'], errors='coerce').eq(0)
        if 'can.canStatus' in data
        else pd.Series(True, index=data.index)
    )
    speed = speed.where(speed.between(0, 255) & valid_can)
    seconds = (data['time'] - data['time'].iloc[0]).dt.total_seconds()
    return [
        {
            'sec': float(second),
            'speed': None if pd.isna(value) else float(value),
            'break_before': False,
        }
        for second, value in zip(seconds, speed)
    ]


def make_trip_result(
    records,
    trip_number,
    minimum_distance_km,
    summary=None,
):
    if summary is None:
        summary = analyze(records, detail=False)
    distance = summary['distance_km']
    fuel = summary['fuel_l']
    l100 = summary['l100']
    if (
        distance is None
        or fuel is None
        or l100 is None
        or not math.isfinite(float(distance))
        or not math.isfinite(float(fuel))
        or not math.isfinite(float(l100))
        or distance < minimum_distance_km
        or distance <= 0
        or fuel < 0
    ):
        return None

    events = count_harsh_events(event_points_from_records(records))
    acceleration_count = sum(e['type'] == 'acceleration' for e in events)
    braking_count = sum(e['type'] == 'braking' for e in events)
    total_events = acceleration_count + braking_count
    return {
        'trip_number': trip_number,
        'start_date': summary['start'][:10],
        'distance_km': float(distance),
        'fuel_l': float(fuel),
        'fuel_l_per_100km': float(l100),
        'acceleration_events': acceleration_count,
        'braking_events': braking_count,
        'total_harsh_events': total_events,
        'acceleration_events_per_100km': 100 * acceleration_count / distance,
        'braking_events_per_100km': 100 * braking_count / distance,
        'total_events_per_100km': 100 * total_events / distance,
        'acceleration_group': event_group(acceleration_count),
        'braking_group': event_group(braking_count),
        'both_repeated': acceleration_count >= 2 and braking_count >= 2,
    }


def _font_setup():
    plt.rcParams['font.sans-serif'] = [
        'Microsoft JhengHei',
        'Microsoft YaHei',
        'Noto Sans CJK TC',
        'DejaVu Sans',
    ]
    plt.rcParams['axes.unicode_minus'] = False


def save_scatter_chart(data, output_path):
    _font_setup()
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    colors = {'acceleration': '#268779', 'braking': '#d18a43'}
    markers = {'acceleration': 'o', 'braking': '^'}
    for key, label in EVENT_LABELS.items():
        x = data[f'{key}_events_per_100km']
        y = data['fuel_l_per_100km']
        ax.scatter(
            x,
            y,
            s=22,
            alpha=0.35,
            color=colors[key],
            marker=markers[key],
            label=f'{label}（Spearman ρ={_format_corr(spearman_correlation(x, y))}）',
            edgecolors='none',
        )
    event_rates = pd.concat(
        [data['acceleration_events_per_100km'], data['braking_events_per_100km']],
        ignore_index=True,
    )
    x_limit = max(1.0, float(event_rates.quantile(0.99)))
    y_limit = max(1.0, float(data['fuel_l_per_100km'].quantile(0.99)))
    ax.set_xlim(left=0, right=x_limit)
    ax.set_ylim(bottom=0, top=y_limit)
    ax.set(
        title='急加減速頻率與行程油耗',
        xlabel='事件次數 / 100 km',
        ylabel='行程油耗（L/100 km）',
    )
    ax.text(
        0.01,
        0.99,
        '座標顯示至第 99 百分位；相關係數使用全部行程',
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=9,
        color='#56616a',
    )
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_group_chart(data, output_path):
    _font_setup()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True, constrained_layout=True)
    for ax, key, title in zip(
        axes,
        ('acceleration_group', 'braking_group'),
        ('急加速事件數', '急減速事件數'),
    ):
        groups = []
        labels = []
        for group in EVENT_GROUPS:
            values = data.loc[data[key] == group, 'fuel_l_per_100km'].dropna()
            if values.empty:
                continue
            groups.append(values.to_numpy())
            labels.append(f'{group}\n(n={len(values)})')
        if groups:
            box = ax.boxplot(
                groups,
                tick_labels=labels,
                patch_artist=True,
                showfliers=False,
            )
            for patch in box['boxes']:
                patch.set_facecolor('#dcebe8')
                patch.set_edgecolor('#6c8f89')
            for median in box['medians']:
                median.set_color('#245f56')
                median.set_linewidth(2)
        else:
            ax.text(0.5, 0.5, '沒有可比較資料', ha='center', va='center')
        ax.set_title(title)
        ax.set_xlabel('每趟事件分類')
        ax.grid(axis='y', alpha=0.2)
    axes[0].set_ylabel('行程油耗（L/100 km）')
    fig.suptitle('不同急加減速次數組別的油耗分布')
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _format_corr(value):
    return '資料不足' if value is None else f'{value:.3f}'


def write_report(data, output_path, excluded_trips, min_distance_km):
    rows = []
    for event_key, group_key in (
        ('acceleration_events', 'acceleration_group'),
        ('braking_events', 'braking_group'),
    ):
        corr = spearman_correlation(
            data[f'{event_key}_per_100km'], data['fuel_l_per_100km']
        )
        rows.append(
            f'- {EVENT_LABELS["acceleration" if event_key.startswith("acceleration") else "braking"]}頻率與油耗的 Spearman 相關：**{_format_corr(corr)}**'
        )
        rows.append(f'  - 此相關使用 {len(data)} 趟有效行程；不能解讀為因果關係。')
        for group in EVENT_GROUPS:
            values = data.loc[data[group_key] == group, 'fuel_l_per_100km']
            if not values.empty:
                rows.append(
                    f'  - {group}：{len(values)} 趟，中位油耗 {values.median():.2f} L/100km（IQR {values.quantile(0.25):.2f}–{values.quantile(0.75):.2f}）。'
                )

    report = [
        '# 反覆急加速／急減速與油耗分析',
        '',
        f'- 有效行程：{len(data):,} 趟',
        f'- 排除行程：{excluded_trips:,} 趟（油耗／里程計數器無效或距離小於 {min_distance_km:g} km）',
        '- 急加減速規則：速度變化率嚴格大於 3 km/h/s，且至少連續 3 個有效觀測；相鄰間隔須不超過 30 秒。',
        '- 油耗採每趟累積燃油量差 ÷ 累積里程差 × 100；不跨行程相減。',
        '- 觀察性分析，不代表急加減速造成油耗變化；載重、路況、駕駛路線、怠速等因素未控制。',
        '',
        '## 相關與分組摘要',
        '',
        *rows,
        '',
        '## 圖表',
        '',
        '- `event_rate_vs_fuel.png`：急加速／急減速每 100 km 次數與行程油耗散佈圖。',
        '  - 為避免少數極端值壓縮圖表座標，散佈圖只顯示至第 99 百分位；相關係數仍使用全部有效行程。',
        '- `event_groups_fuel_boxplot.png`：每趟 0 次、1 次、2 次以上事件組的油耗分布。',
        '- `trip_analysis.csv`：可供進一步篩選的逐趟彙總資料。',
        '',
    ]
    output_path.write_text('\n'.join(report), encoding='utf-8')


def find_source(path):
    if path is not None:
        return path
    return next((candidate for candidate in (BASE / 'data.xlsx', BASE / 'data.xls') if candidate.exists()), None)


def run_analysis(
    source,
    output_dir,
    minimum_distance_km=DEFAULT_MIN_DISTANCE_KM,
):
    db_path = prepare(source, BASE / '.cache')
    results = []
    excluded = 0
    with sqlite3.connect(f'file:{db_path.as_posix()}?mode=ro', uri=True) as connection:
        trips = connection.execute(
            'SELECT id, vehicle, journey, summary FROM trips ORDER BY id'
        ).fetchall()
        for trip_number, (trip_id, vehicle, journey, summary_json) in enumerate(trips, 1):
            records = [
                {
                    'time': row[0],
                    'can.canSpeed': row[1],
                    'can.canStatus': row[2],
                }
                for row in connection.execute(
                    """SELECT
                           json_extract(raw, '$.time'),
                           json_extract(raw, '$."can.canSpeed"'),
                           json_extract(raw, '$."can.canStatus"')
                       FROM points
                       WHERE vehicle=? AND journey=?
                       ORDER BY seq""",
                    (vehicle, journey),
                )
            ]
            try:
                trip_result = make_trip_result(
                    records,
                    trip_number,
                    minimum_distance_km,
                    summary=json.loads(summary_json),
                )
            except (ValueError, KeyError, IndexError) as exc:
                print(f'略過第 {trip_number} 趟（行程分析失敗）：{exc}', flush=True)
                excluded += 1
                continue
            if trip_result is None:
                excluded += 1
                continue
            results.append(trip_result)
            if trip_number % 250 == 0:
                print(f'已分析 {trip_number:,}/{len(trips):,} 趟', flush=True)

    if not results:
        raise ValueError('沒有符合條件的有效行程，請確認來源檔或降低最小里程門檻。')

    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(results)
    data.to_csv(output_dir / 'trip_analysis.csv', index=False, encoding='utf-8-sig')
    save_scatter_chart(data, output_dir / 'event_rate_vs_fuel.png')
    save_group_chart(data, output_dir / 'event_groups_fuel_boxplot.png')
    write_report(
        data,
        output_dir / 'analysis_summary.md',
        excluded,
        minimum_distance_km,
    )
    return len(data), excluded


def main():
    parser = argparse.ArgumentParser(
        description='分析反覆急加速／急減速與每趟油耗的關係，輸出 CSV 與 PNG 圖表。'
    )
    parser.add_argument('--file', type=Path, help='來源 Excel 路徑；預設使用 data.xlsx/data.xls')
    parser.add_argument(
        '--output',
        type=Path,
        default=BASE / 'analysis_output',
        help='輸出資料夾（預設：analysis_output）',
    )
    parser.add_argument(
        '--min-distance-km',
        type=float,
        default=DEFAULT_MIN_DISTANCE_KM,
        help='納入分析的最小單趟里程 km（預設：1）',
    )
    args = parser.parse_args()
    source = find_source(args.file)
    if source is None or not source.is_file():
        parser.error('找不到來源 Excel，請使用 --file 指定檔案。')
    if not math.isfinite(args.min_distance_km) or args.min_distance_km < 0:
        parser.error('--min-distance-km 必須是大於或等於 0 的有限數值。')

    try:
        included, excluded = run_analysis(
            source.resolve(),
            args.output.resolve(),
            args.min_distance_km,
        )
    except (OSError, sqlite3.Error, ValueError) as exc:
        parser.error(str(exc))
    print(
        f'完成：納入 {included:,} 趟，排除 {excluded:,} 趟。\n'
        f'輸出位置：{args.output.resolve()}',
        flush=True,
    )


if __name__ == '__main__':
    main()
