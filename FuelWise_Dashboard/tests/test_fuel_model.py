import unittest
from datetime import datetime, timedelta

from analysis.train_fuel_model import (
    FEATURE_COLUMNS,
    FORBIDDEN_FEATURE_TOKENS,
    _assert_no_leakage,
    extract_features,
    temporal_split,
)
from analytics import analyze


def sample_trip(start, end_offset=10, vehicle='vehicle-1', journey=None):
    begin = datetime(2025, 1, 1) + timedelta(days=start)
    return {
        'vehicle': vehicle,
        'journey': journey or f'journey-{start}',
        'start': begin.isoformat(sep=' '),
        'end': (begin + timedelta(seconds=end_offset)).isoformat(sep=' '),
        '_start_dt': begin,
        '_end_dt': begin + timedelta(seconds=end_offset),
    }


def raw_point(seconds, speed, rpm=1000, fuel=100, odo=10):
    stamp = datetime(2025, 1, 1) + timedelta(seconds=seconds)
    return {
        'time': stamp.isoformat(sep=' '),
        'can.canStatus': 0,
        'can.canSpeed': speed,
        'can.engine.rpm': rpm,
        'can.engine.totalFuelUsed': fuel,
        'can.totalMileage': odo,
        'can.engine.engineLoad': 50,
        'can.engine.engineCoolantTemp': 80,
    }


class FuelModelTests(unittest.TestCase):
    def test_feature_inputs_exclude_target_derived_and_identifiers(self):
        _assert_no_leakage()
        for feature in FEATURE_COLUMNS:
            self.assertFalse(
                any(token in feature.lower() for token in FORBIDDEN_FEATURE_TOKENS),
                feature,
            )

    def test_temporal_split_drops_trips_crossing_boundaries(self):
        rows = [sample_trip(day, end_offset=10) for day in range(100)]
        crossing = sample_trip(69, end_offset=2 * 24 * 60 * 60, journey='crossing')
        rows.append(crossing)
        splits, excluded, first, second = temporal_split(rows)
        self.assertTrue(any(key[1] == 'crossing' for key, _ in excluded))
        sets = {
            name: {(row['vehicle'], row['journey']) for row in values}
            for name, values in splits.items()
        }
        self.assertFalse(sets['train'] & sets['validation'])
        self.assertFalse(sets['train'] & sets['test'])
        self.assertFalse(sets['validation'] & sets['test'])
        self.assertLess(
            max(row['_end_dt'] for row in splits['train']),
            min(row['_start_dt'] for row in splits['validation']),
        )
        self.assertLess(
            max(row['_end_dt'] for row in splits['validation']),
            min(row['_start_dt'] for row in splits['test']),
        )
        self.assertLess(first, second)

    def test_features_include_coverage_and_kinetic_proxy_without_fuel(self):
        result = analyze(
            [
                raw_point(0, 0),
                raw_point(10, 36, fuel=100.5, odo=10.1),
                raw_point(20, 0, fuel=101, odo=10.2),
            ],
            detail=True,
        )
        features, coverage, quality = extract_features(result)
        self.assertEqual(set(features), set(FEATURE_COLUMNS))
        self.assertEqual(set(coverage), {f'coverage_pct__{name}' for name in FEATURE_COLUMNS})
        self.assertEqual(features['kinetic_increase_j_per_kg'], 50.0)
        self.assertIsNotNone(features['kinetic_increase_j_per_kg_km'])
        self.assertFalse(any('fuel' in name.lower() for name in features))
        self.assertFalse(quality['fuel_reset'])

    def test_feature_only_analysis_preserves_existing_summary_and_quality(self):
        rows = [
            raw_point(0, 0),
            raw_point(10, 10, fuel=100.5, odo=10.1),
        ]
        full = analyze(rows, detail=True)
        compact = analyze(rows, detail=True, feature_only=True)
        self.assertEqual(compact['summary'], full['summary'])
        self.assertEqual(compact['quality'], full['quality'])
        self.assertEqual(compact['points'], full['points'])
        self.assertNotIn('segments', compact)
        self.assertNotIn('raw', compact)


if __name__ == '__main__':
    unittest.main()
