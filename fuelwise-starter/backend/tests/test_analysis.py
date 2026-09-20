import unittest
from datetime import datetime,timedelta,timezone
from app.services.analysis import Sample,analyze
class AnalysisTest(unittest.TestCase):
    def test_irregular_intervals(self):
        t=datetime(2025,1,1,tzinfo=timezone.utc)
        s=[Sample(t,100,10,0,600),Sample(t+timedelta(seconds=20),100,10,36,1000),Sample(t+timedelta(seconds=30),100.1,10.5,0,0)]
        r=analyze(s)
        self.assertEqual(r['engine_on_stop_estimate_s'],20)
        self.assertEqual(r['duration_s'],30)
        self.assertAlmostEqual(r['fuel_used_l'],.5)
    def test_counter_reset_rejected(self):
        t=datetime(2025,1,1,tzinfo=timezone.utc)
        with self.assertRaises(ValueError):
            analyze([Sample(t,100,10,0,0),Sample(t+timedelta(seconds=5),90,10,0,0)])
if __name__=='__main__':unittest.main()
