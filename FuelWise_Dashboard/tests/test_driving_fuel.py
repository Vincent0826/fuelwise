import unittest

from analysis.analyze_driving_fuel import (
    count_harsh_events,
    event_group,
    make_trip_result,
    spearman_correlation,
)


def point(sec, speed, break_before=False):
    return {'sec': sec, 'speed': speed, 'break_before': break_before}


def row(second, mileage, fuel, speed):
    return {
        'time': f'2025-01-01 00:00:{second:02d}',
        'can.totalMileage': mileage,
        'can.engine.totalFuelUsed': fuel,
        'can.canSpeed': speed,
        'can.engine.rpm': 1000,
        'can.canStatus': 0,
        'gps.latitude': 23,
        'gps.longitude': 120,
    }


class DrivingFuelTests(unittest.TestCase):
    def test_counts_each_sustained_acceleration_once(self):
        points = [
            point(0, 0),
            point(10, 40),
            point(20, 80),
            point(30, 120),
            point(40, 160),
        ]
        events = count_harsh_events(points)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['type'], 'acceleration')

    def test_counts_repeated_acceleration_and_braking_separately(self):
        points = [
            point(0, 0),
            point(10, 40),
            point(20, 80),
            point(30, 120),
            point(40, 120),
            point(50, 80),
            point(60, 40),
            point(70, 0),
            point(80, 0),
            point(90, 40),
            point(100, 80),
            point(110, 120),
        ]
        events = count_harsh_events(points)
        self.assertEqual(
            [event['type'] for event in events],
            ['acceleration', 'braking', 'acceleration'],
        )

    def test_long_gap_and_break_do_not_join_event_streaks(self):
        points = [
            point(0, 0),
            point(10, 40),
            point(50, 80),
            point(60, 120),
            point(70, 160, break_before=True),
            point(80, 200),
            point(90, 240),
        ]
        self.assertEqual(count_harsh_events(points), [])

    def test_trip_result_uses_trip_counter_difference_and_event_count(self):
        records = [
            row(0, 100, 20, 0),
            row(10, 100.5, 21, 40),
            row(20, 101, 22, 80),
            row(30, 101.5, 23, 120),
        ]
        result = make_trip_result(records, 1, 0.1)
        self.assertAlmostEqual(result['fuel_l_per_100km'], 200.0)
        self.assertEqual(result['acceleration_events'], 1)
        self.assertEqual(result['braking_events'], 0)

    def test_trip_below_minimum_distance_is_excluded(self):
        records = [row(0, 100, 20, 0), row(10, 100.05, 21, 1)]
        self.assertIsNone(make_trip_result(records, 1, 1.0))

    def test_event_group_and_spearman(self):
        self.assertEqual([event_group(i) for i in range(3)], ['0 次', '1 次', '2 次以上'])
        self.assertAlmostEqual(spearman_correlation([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertIsNone(spearman_correlation([1, 1, 1], [3, 2, 1]))


if __name__ == '__main__':
    unittest.main()
