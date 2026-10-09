"""Shared applicability rules for the general-transport L/100 km model.

One definition used by training, the dashboard API and (through the API
payload) the front end.  The thresholds are a product scope chosen for this
round; they are not validated physical limits and must not be tuned on model
error.  Raw trips are never removed; this only decides whether the model
applies.
"""

import math

from analysis.train_fuel_model import GAP_LIMIT_SECONDS, _target_exclusion_reasons

RULE_VERSION = 'applicability-v1'

APPLICABILITY_CONFIG = {
    'rule_version': RULE_VERSION,
    'min_distance_km': 5.0,
    'stationary_min_observable_s': 20 * 60,
    'stationary_ratio_threshold': 0.80,
    'min_observation_coverage': 0.70,
    'gap_limit_s': GAP_LIMIT_SECONDS,
    'stationary_event_min_s': 60,
}

LABEL_TEXTS = {
    'short_distance': '短距離行程',
    'stationary_dominant': '可觀測區間以停車引擎運轉為主',
    'insufficient_observation': '觀測不足',
    'invalid_target': '燃油或里程目標無效',
    'eligible_transport': '可進行一般運輸油耗分析',
}
EXCLUSION_CODES = (
    'short_distance',
    'stationary_dominant',
    'insufficient_observation',
    'invalid_target',
)

_MISSING_KEY_REASONS = {
    'invalid_fuel_counter_endpoint',
    'invalid_odometer_endpoint',
    'missing_or_nonfinite_distance',
    'missing_or_nonfinite_fuel',
    'missing_or_nonfinite_target',
}


def _num(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _known(value):
    return _num(value) is not None


def time_accounting(points, trip_duration_s, gap_limit_s=GAP_LIMIT_SECONDS):
    """Partition the trip time axis into disjoint, non-overlapping categories.

    Each row holds its state until the next row only when the step is in
    (0, gap_limit_s]; longer steps are 'long_gap' and are never extended.
    """
    observable = invalid_can = other_unknown = long_gap = 0.0
    stationary = 0.0
    for index, point in enumerate(points):
        dt = _num(point.get('dt')) or 0.0
        if index + 1 < len(points):
            step = (_num(points[index + 1].get('sec')) or 0.0) - (
                _num(point.get('sec')) or 0.0
            )
            if step > gap_limit_s:
                long_gap += step
        if dt <= 0:
            continue
        speed, rpm = _num(point.get('speed')), _num(point.get('rpm'))
        if speed is not None and rpm is not None:
            observable += dt
            if speed == 0 and rpm > 0:
                stationary += dt
        elif _num(point.get('can_status')) not in (None, 0.0):
            invalid_can += dt
        else:
            other_unknown += dt
    covered = observable + invalid_can + other_unknown + long_gap
    return {
        'observable_s': observable,
        'invalid_can_s': invalid_can,
        'other_unknown_s': other_unknown,
        'long_gap_s': long_gap,
        'uncovered_s': max(0.0, (trip_duration_s or 0.0) - covered),
        'stationary_engine_on_s': stationary,
    }


def stationary_events(points, min_seconds=60):
    """Contiguous stationary-engine-on runs lasting at least min_seconds."""
    events, run = [], 0.0
    for point in points:
        dt = _num(point.get('dt')) or 0.0
        speed, rpm = _num(point.get('speed')), _num(point.get('rpm'))
        if dt > 0 and speed == 0 and rpm is not None and rpm > 0:
            run += dt
            continue
        if run >= min_seconds:
            events.append(run)
        run = 0.0
    if run >= min_seconds:
        events.append(run)
    return events


def evaluate_applicability(analysis, config=None):
    """Return the applicability decision for one analysed trip."""
    cfg = config or APPLICABILITY_CONFIG
    summary = analysis.get('summary', {})
    points = analysis.get('points') or []
    target_reasons = _target_exclusion_reasons(
        {
            'summary': summary,
            'quality': analysis.get('quality', {}),
            'points': points,
        }
    )
    distance = _num(summary.get('distance_km'))
    duration = _num(summary.get('duration_sec'))
    if duration is None and points:
        duration = _num(points[-1].get('sec'))

    accounting = time_accounting(
        points, duration, cfg['gap_limit_s']
    )
    observable = accounting['observable_s']
    coverage = (
        min(1.0, observable / duration) if duration and duration > 0 else None
    )
    stationary = accounting['stationary_engine_on_s']
    ratio = stationary / observable if observable > 0 else None
    events = stationary_events(points, cfg['stationary_event_min_s'])

    missing_key = [r for r in target_reasons if r in _MISSING_KEY_REASONS]
    if duration is None or duration <= 0:
        missing_key.append('trip_duration_unavailable')

    labels = []
    if distance is not None and distance < cfg['min_distance_km']:
        labels.append('short_distance')
    if (
        observable >= cfg['stationary_min_observable_s']
        and ratio is not None
        and ratio >= cfg['stationary_ratio_threshold']
    ):
        labels.append('stationary_dominant')
    if coverage is None or coverage < cfg['min_observation_coverage']:
        labels.append('insufficient_observation')
    if target_reasons:
        labels.append('invalid_target')
    exclusion_reasons = list(labels)
    if not exclusion_reasons:
        labels.append('eligible_transport')

    return {
        'model_eligible': not exclusion_reasons,
        'labels': labels,
        'exclusion_reasons': exclusion_reasons,
        'rule_version': cfg['rule_version'],
        'distance_km': distance,
        'trip_duration_s': duration,
        'observable_duration_s': observable,
        'observation_coverage': coverage,
        'stationary_engine_on_duration_s': stationary,
        'stationary_engine_on_ratio_observed': ratio,
        'stationary_event_duration_s': float(sum(events)),
        'longest_stationary_event_s': float(max(events)) if events else 0.0,
        'time_accounting': accounting,
        'target_issues': target_reasons,
        'missing_key_fields': missing_key,
        'label_texts': {code: LABEL_TEXTS[code] for code in labels},
        'definitions': {
            'observable_duration_s': '速度與轉速皆有效（CAN 正常）的區間聯集；不跨越超過門檻的長缺口。',
            'observation_coverage': 'observable_duration_s ÷ trip_duration_s',
            'stationary_engine_on_ratio_observed': (
                'stationary_engine_on_duration_s ÷ observable_duration_s；'
                '分母只含可判讀時間，不是整趟時間。'
            ),
        },
        'thresholds': {
            key: cfg[key]
            for key in (
                'min_distance_km',
                'stationary_min_observable_s',
                'stationary_ratio_threshold',
                'min_observation_coverage',
            )
        },
    }


def flatten_applicability(result, prefix='applic_'):
    """Scalar columns for tabular storage of one decision."""
    return {
        f'{prefix}model_eligible': bool(result['model_eligible']),
        f'{prefix}labels': ';'.join(result['labels']),
        f'{prefix}exclusion_reasons': ';'.join(result['exclusion_reasons']),
        f'{prefix}rule_version': result['rule_version'],
        f'{prefix}trip_duration_s': result['trip_duration_s'],
        f'{prefix}observable_duration_s': result['observable_duration_s'],
        f'{prefix}observation_coverage': result['observation_coverage'],
        f'{prefix}stationary_engine_on_duration_s': result[
            'stationary_engine_on_duration_s'
        ],
        f'{prefix}stationary_engine_on_ratio_observed': result[
            'stationary_engine_on_ratio_observed'
        ],
    }
