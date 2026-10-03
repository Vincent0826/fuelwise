import unittest
from analytics import analyze

def row(t,odo=10,fuel=100,speed=0,rpm=600,status=0):
 return {'time':'2025-01-01 00:'+t,'can.totalMileage':odo,'can.engine.totalFuelUsed':fuel,'can.canSpeed':speed,'can.engine.rpm':rpm,'can.canStatus':status,'gps.latitude':23,'gps.longitude':120}
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
if __name__=='__main__':unittest.main()
