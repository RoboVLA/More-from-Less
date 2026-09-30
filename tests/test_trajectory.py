import tempfile
import unittest
from pathlib import Path
from morefromless.trajectory.pose_io import PoseSample,load_pose_trajectory,save_pose_csv,save_pose_json
from morefromless.trajectory.fuse_trajectories import fuse_trajectories
class TrajectoryTests(unittest.TestCase):
    def s(self,f,t,x,z=3):return PoseSample(f,t,x,2,z,10,20,30)
    def test_timestamp_interpolation(self):
        fused=fuse_trajectories([self.s(4,.5,100),self.s(8,1.5,200)],[self.s(0,0,0),self.s(2,2,40)],.5,False)
        self.assertEqual([s.x for s in fused],[55,115])
        self.assertEqual([s.frame for s in fused],[4,8])
        self.assertEqual([s.time_s for s in fused],[.5,1.5])
    def test_keep_generated_z(self):
        p=fuse_trajectories([self.s(0,0,0,17)],[self.s(0,0,100,99)],1,True)[0]
        self.assertEqual((p.x,p.z),(100,17))
    def test_reject_empty_invalid_weight(self):
        with self.assertRaises(ValueError):fuse_trajectories([],[self.s(0,0,0)],.5,False)
        with self.assertRaises(ValueError):fuse_trajectories([self.s(0,0,0)],[self.s(0,0,0)],1.1,False)
    def test_csv_json_roundtrip(self):
        samples=[self.s(5,.123,12.5)]
        with tempfile.TemporaryDirectory() as d:
            for suffix,save in [('csv',save_pose_csv),('json',save_pose_json)]:
                p=Path(d)/('pose.'+suffix);save(p,samples);self.assertEqual(load_pose_trajectory(p),samples)
    def test_packaged_examples(self):
        root=Path(__file__).resolve().parents[1]
        for task in ['pour','wipe']:
            a=load_pose_trajectory(root/f'examples/trajectories/{task}/generated_pose_xyzrpy.csv')
            b=load_pose_trajectory(root/f'examples/trajectories/{task}/real_retargeted_pose_xyzrpy.csv')
            self.assertEqual(len(fuse_trajectories(a,b,.55,False)),len(a))
if __name__=='__main__':unittest.main()
