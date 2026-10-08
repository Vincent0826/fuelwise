import unittest
from datetime import datetime, timedelta

from analytics import analyze

def row(t,odo=10,fuel=100,speed=0,rpm=600,status=0):
 return {'time':'2025-01-01 00:'+t,'can.totalMileage':odo,'can.engine.totalFuelUsed':fuel,'can.canSpeed':speed,'can.engine.rpm':rpm,'can.canStatus':status,'gps.latitude':23,'gps.longitude':120}

def segment_row(seconds,odo=10,fuel=100,speed=0,rpm=600,status=0):
 time=(datetime(2025,1,1)+timedelta(seconds=seconds)).isoformat(sep=' ')
 return {'time':time,'can.totalMileage':odo,'can.engine.totalFuelUsed':fuel,'can.canSpeed':speed,'can.engine.rpm':rpm,'can.canStatus':status,'gps.latitude':23,'gps.longitude':120}

class AnalyticsTests(unittest.TestCase):
 def test_counters_time_and_idle(self):
  d=analyze([row('00:00'),row('01:00',11,101),row('02:00',12,102,40)])
  self.assertEqual(d['summary']['fuel_l'],2);self.assertEqual(d['summary']['distance_km'],2);self.assertEqual(d['summary']['idle_sec'],120)
 def test_no_bridge_gap(self):
  d=analyze([row('00:00'),row('10:00'),row('11:00')]);self.assertEqual(d['summary']['idle_sec'],60);self.assertEqual(d['quality']['gap_seconds'],600)
 def test_reset_not_repaired(self):
  d=analyze([row('00:00'),row('01:00',11,1)]);self.assertIsNone(d['summary']['fuel_l']);self.assertTrue(d['quality']['fuel_reset'])
 def test_bad_endpoint(self):
  d=analyze([row('00:00',status=9),row('01:00',11,101)]);self.assertIsNone(d['summary']['fuel_l']);self.assertEqual(d['summary']['idle_sec'],0)
 def test_event_dedup(self):
  r=[row('00:00'),row('01:00')]
  for x in r:x.update({'event[0].type':6,'event[0].startTime':'2025-01-01 00:00:00'})
  self.assertEqual(len(analyze(r)['events']),1)
 def test_segments_weight_signals_by_irregular_intervals(self):
  result=analyze([
   segment_row(0,speed=10,rpm=1000),
   segment_row(10,speed=20,rpm=2000),
   segment_row(40,speed=20,rpm=2000),
  ])
  segment=result['segments']['windows'][0]
  self.assertEqual(segment['avg_speed_kmh'],17.5)
  self.assertEqual(segment['avg_rpm'],1750)
  self.assertEqual(segment['valid_coverage_sec'],40)
  self.assertTrue(segment['last_partial_window'])
 def test_segments_do_not_allocate_cross_window_deltas_twice(self):
  result=analyze([
   segment_row(0,odo=10,fuel=100),
   segment_row(840,odo=10,fuel=100),
   segment_row(899,odo=11,fuel=101),
   segment_row(901,odo=12,fuel=102),
  ])
  windows=result['segments']['windows']
  summary=result['segments']['summary']
  self.assertEqual(windows[0]['fuel_l'],1)
  self.assertIsNone(windows[1]['fuel_l'])
  self.assertEqual(summary['allocated_fuel_l'],1)
  self.assertEqual(summary['unassigned_cross_window_fuel_l'],1)
  self.assertEqual(summary['unassigned_cross_window_fuel_interval_count'],1)
  self.assertEqual(result['segments']['unassigned_cross_window_intervals'][0]['fuel_l'],1)
  self.assertIn('有效燃油增量跨越時間窗邊界，未分配', windows[0]['quality_reasons'])
  self.assertIn('有效燃油增量跨越時間窗邊界，未分配', windows[1]['quality_reasons'])
  self.assertEqual(windows[0]['cross_boundary_interval_count'],1)
  self.assertEqual(windows[1]['cross_boundary_interval_count'],1)
 def test_segments_exclude_long_gaps_and_record_counter_resets(self):
  long_gap=analyze([
   segment_row(0,fuel=100),
   segment_row(121,fuel=101),
   segment_row(130,fuel=101.5),
  ])['segments']
  self.assertEqual(long_gap['windows'][0]['fuel_l'],0.5)
  self.assertEqual(long_gap['summary']['excluded_long_gap_fuel_l'],1)
  self.assertEqual(long_gap['summary']['excluded_intervals']['long_gap'],1)
  reset=analyze([
   segment_row(0,fuel=100),
   segment_row(1,fuel=101),
   segment_row(2,fuel=50),
  ])
  self.assertIsNone(reset['summary']['fuel_l'])
  self.assertEqual(reset['segments']['windows'][0]['fuel_l'],1)
  self.assertEqual(
   reset['segments']['summary']['excluded_intervals']['fuel_counter_reset'],1
  )
 def test_segments_exclude_invalid_can_and_missing_counters(self):
  invalid_can=analyze([
   segment_row(0,fuel=100),
   segment_row(10,fuel=101,status=9),
   segment_row(20,fuel=102),
  ])['segments']
  self.assertIsNone(invalid_can['windows'][0]['fuel_l'])
  self.assertEqual(invalid_can['summary']['excluded_intervals']['invalid_can'],2)
  missing_fuel=analyze([
   segment_row(0,fuel=None),
   segment_row(10,fuel=101),
  ])['segments']
  self.assertIsNone(missing_fuel['windows'][0]['fuel_l'])
  self.assertEqual(missing_fuel['summary']['excluded_intervals']['missing_fuel'],1)
 def test_segments_label_unchanged_fuel_counter_without_claiming_zero_use(self):
  segment=analyze([
   segment_row(0,fuel=100),
   segment_row(10,fuel=100),
   segment_row(20,fuel=100),
  ])['segments']['windows'][0]
  self.assertEqual(segment['fuel_l'],0)
  self.assertEqual(segment['fuel_status'],'no_increment_observed')
  self.assertEqual(segment['quality_status'],'未觀測到增量')
if __name__=='__main__':unittest.main()
