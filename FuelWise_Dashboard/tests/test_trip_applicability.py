import unittest

from analysis.trip_applicability import (
    APPLICABILITY_CONFIG,
    RULE_VERSION,
    evaluate_applicability,
)
from analysis.vehicle_history_experiment import (
    HISTORY_FEATURES,
    build_vehicle_history_features,
)


def make_trip(distance=10.0, fuel=1.0, steps=None, duration=None):
    """steps: list of (dt_seconds, speed, rpm, can_status)."""
    steps = steps if steps is not None else [(60, 30.0, 1500.0, 0)] * 60
    points, sec = [], 0.0
    for dt, speed, rpm, can in steps:
        points.append(
            {
                'sec': sec, 'dt': dt, 'speed': speed, 'rpm': rpm,
                'can_status': can, 'fuel': 100.0, 'odo': 1000.0,
            }
        )
        sec += dt
    points.append(
        {'sec': sec, 'dt': 0, 'speed': 0.0, 'rpm': 0.0, 'can_status': 0,
         'fuel': 101.0, 'odo': 1010.0}
    )
    total = duration if duration is not None else sec
    l100 = fuel / distance * 100 if distance else None
    return {
        'summary': {
            'distance_km': distance, 'fuel_l': fuel, 'l100': l100,
            'duration_sec': total,
        },
        'quality': {},
        'points': points,
    }


class ApplicabilityRuleTests(unittest.TestCase):
    def test_rule_version_and_eligible_trip(self):
        result = evaluate_applicability(make_trip())
        self.assertEqual(result['rule_version'], RULE_VERSION)
        self.assertTrue(result['model_eligible'])
        self.assertEqual(result['labels'], ['eligible_transport'])
        self.assertAlmostEqual(result['observation_coverage'], 1.0)

    def test_distance_boundary_is_strictly_below_five(self):
        self.assertIn('short_distance', evaluate_applicability(make_trip(4.99))['labels'])
        self.assertNotIn('short_distance', evaluate_applicability(make_trip(5.0))['labels'])

    def test_coverage_boundary_70_percent(self):
        def trip(observable_steps):
            steps = [(60, 30.0, 1500.0, 0)] * observable_steps
            steps += [(60, None, None, 9)] * (100 - observable_steps)
            return make_trip(steps=steps)

        at = evaluate_applicability(trip(70))
        below = evaluate_applicability(trip(69))
        self.assertAlmostEqual(at['observation_coverage'], 0.70)
        self.assertNotIn('insufficient_observation', at['labels'])
        self.assertIn('insufficient_observation', below['labels'])

    def test_stationary_ratio_boundary_and_denominator(self):
        def trip(idle_steps, moving_steps):
            steps = [(60, 0.0, 800.0, 0)] * idle_steps
            steps += [(60, 30.0, 1500.0, 0)] * moving_steps
            return make_trip(steps=steps)

        at = evaluate_applicability(trip(24, 6))
        below = evaluate_applicability(trip(23, 7))
        self.assertAlmostEqual(at['stationary_engine_on_ratio_observed'], 0.8)
        self.assertIn('stationary_dominant', at['labels'])
        self.assertNotIn('stationary_dominant', below['labels'])
        self.assertAlmostEqual(at['stationary_engine_on_duration_s'], 24 * 60)
        self.assertAlmostEqual(at['observable_duration_s'], 30 * 60)

    def test_stationary_needs_twenty_observable_minutes(self):
        short = make_trip(steps=[(60, 0.0, 800.0, 0)] * 19)
        enough = make_trip(steps=[(60, 0.0, 800.0, 0)] * 20)
        self.assertNotIn('stationary_dominant', evaluate_applicability(short)['labels'])
        self.assertIn('stationary_dominant', evaluate_applicability(enough)['labels'])

    def test_ratio_denominator_excludes_invalid_can_and_gaps(self):
        steps = [(60, 0.0, 800.0, 0)] * 30 + [(60, None, None, 9)] * 120
        result = evaluate_applicability(make_trip(steps=steps))
        self.assertAlmostEqual(result['stationary_engine_on_ratio_observed'], 1.0)
        self.assertAlmostEqual(result['observation_coverage'], 30 / 150)
        self.assertEqual(result['time_accounting']['invalid_can_s'], 120 * 60)
        self.assertIn('insufficient_observation', result['labels'])

    def test_long_gap_not_extended_and_not_counted_observable(self):
        steps = [(60, 30.0, 1500.0, 0)] * 10 + [(600, 30.0, 1500.0, 0)]
        trip = make_trip(steps=steps)
        # the long step is not held as state (dt=0 by the analysis rule).
        trip['points'][10]['dt'] = 0
        result = evaluate_applicability(trip)
        self.assertAlmostEqual(result['observable_duration_s'], 600)
        self.assertEqual(result['time_accounting']['long_gap_s'], 600)
        self.assertAlmostEqual(result['observation_coverage'], 600 / 1200)

    def test_unknown_time_is_not_idle_or_zero(self):
        steps = [(60, None, None, 0)] * 30
        result = evaluate_applicability(make_trip(steps=steps))
        self.assertEqual(result['observable_duration_s'], 0)
        self.assertIsNone(result['stationary_engine_on_ratio_observed'])
        self.assertEqual(result['time_accounting']['other_unknown_s'], 1800)

    def test_zero_duration_and_missing_values(self):
        trip = make_trip(steps=[])
        trip['points'] = []
        trip['summary'].update(duration_sec=0, distance_km=None, fuel_l=None, l100=None)
        result = evaluate_applicability(trip)
        self.assertFalse(result['model_eligible'])
        self.assertIsNone(result['observation_coverage'])
        self.assertIn('insufficient_observation', result['labels'])
        self.assertIn('invalid_target', result['labels'])
        self.assertIn('trip_duration_unavailable', result['missing_key_fields'])

    def test_multi_label_is_not_double_counted(self):
        steps = [(60, 0.0, 800.0, 0)] * 25 + [(60, None, None, 9)] * 100
        result = evaluate_applicability(make_trip(distance=1.0, steps=steps))
        self.assertGreaterEqual(len(result['exclusion_reasons']), 3)
        self.assertFalse(result['model_eligible'])
        decisions = [result, evaluate_applicability(make_trip())]
        unique_excluded = sum(not d['model_eligible'] for d in decisions)
        overlapping = sum(len(d['exclusion_reasons']) for d in decisions)
        self.assertEqual(unique_excluded, 1)
        self.assertGreater(overlapping, unique_excluded)

    def test_high_fuel_use_alone_does_not_exclude(self):
        result = evaluate_applicability(make_trip(distance=10.0, fuel=9.0))
        self.assertTrue(result['model_eligible'])

    def test_thresholds_come_from_one_config(self):
        self.assertEqual(APPLICABILITY_CONFIG['min_distance_km'], 5.0)
        self.assertEqual(APPLICABILITY_CONFIG['min_observation_coverage'], 0.70)
        self.assertEqual(APPLICABILITY_CONFIG['stationary_ratio_threshold'], 0.80)


class EligibleHistoryTests(unittest.TestCase):
    @staticmethod
    def row(vehicle, journey, start, end, target):
        return {
            'vehicle': vehicle, 'journey': journey, 'start': start,
            'end': end, 'target_l_per_100km': target,
        }

    def test_history_uses_only_supplied_eligible_past_trips(self):
        eligible = [
            self.row('V', 'a', '2025-01-01 08:00:00', '2025-01-01 09:00:00', 10.0),
            self.row('V', 'b', '2025-01-02 08:00:00', '2025-01-02 09:00:00', 14.0),
            self.row('V', 'c', '2025-01-03 08:00:00', '2025-01-03 09:00:00', 100.0),
        ]
        # an ineligible 150 L/100 km parking trip is never passed in.
        features = build_vehicle_history_features(eligible, eligible)
        self.assertEqual(features[0][HISTORY_FEATURES[1]], 0)
        self.assertEqual(features[1][HISTORY_FEATURES[0]], 10.0)
        self.assertEqual(features[2][HISTORY_FEATURES[0]], 12.0)
        self.assertEqual(features[2][HISTORY_FEATURES[1]], 2)

    def test_frozen_history_ignores_future_and_validation_targets(self):
        train = [
            self.row('V', 'a', '2025-01-01 08:00:00', '2025-01-01 09:00:00', 10.0),
            self.row('V', 'b', '2025-01-02 08:00:00', '2025-01-02 09:00:00', 20.0),
        ]
        target = [self.row('V', 'z', '2025-02-01 08:00:00', '2025-02-01 09:00:00', 500.0)]
        frozen = build_vehicle_history_features(train, target, frozen=True)
        self.assertEqual(frozen[0][HISTORY_FEATURES[0]], 15.0)
        self.assertEqual(frozen[0][HISTORY_FEATURES[1]], 2)


if __name__ == '__main__':
    unittest.main()
