import json
import unittest
from pathlib import Path
import numpy as np
from morefromless.trajectory.pose_io import load_pose_trajectory
from morefromless.trajectory.reproduce_appendix import reconstruct,diagnostics


class AppendixTests(unittest.TestCase):
    def check_task(self,task,expected_reduction):
        source=Path(__file__).resolve().parents[1]/'examples/trajectories'/task
        g,r,saved=[load_pose_trajectory(source/name) for name in
                  ('generated_pose_xyzrpy.csv','real_retargeted_pose_xyzrpy.csv','fused_pose_xyzrpy.csv')]
        meta=json.loads((source/'fused_pose_trajectory_xyzrpy.json').read_text(encoding='utf-8'))
        actual=reconstruct(g,r,base_weight=meta['real_weight_base'],middle_bonus=meta['mid_real_bonus'],
                           smooth_window=meta['smooth_window'],fixed_median_depth=task=='pour')
        self.assertEqual(actual,saved)
        self.assertAlmostEqual(diagnostics(g,r,actual)['relative_reduction_percent'],expected_reduction,places=1)
    def test_pour_all_saved_pose_components(self):self.check_task('pour',61.2)
    def test_wipe_all_saved_pose_components(self):self.check_task('wipe',62.0)
    def test_time_grid_mismatch_rejected(self):
        from morefromless.trajectory.pose_io import PoseSample
        with self.assertRaises(ValueError):
            reconstruct([PoseSample(0,0,0,0,0,0,0,0)],[PoseSample(0,1,0,0,0,0,0,0)],
                        base_weight=.55,middle_bonus=.1,smooth_window=5,fixed_median_depth=False)

if __name__=='__main__':unittest.main()
