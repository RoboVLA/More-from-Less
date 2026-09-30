"""Recompute saved Appendix A.4 curve data from unchanged G/R input CSVs.

Historical algorithm: 0.55 + 0.10*sin(pi*t) real weight, centered truncated
five-frame mean, unwrapped/wrapped Euler components, and median Z for pouring.
This reproduces offline diagnostics, not calibrated 6-D tracking or online control.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .pose_io import PoseSample, load_pose_trajectory, save_pose_csv

ROOT=Path(__file__).resolve().parents[2]


def centered_mean(values, window):
    half=window//2
    return np.array([values[max(0,i-half):min(len(values),i+half+1)].mean(axis=0) for i in range(len(values))])


def reconstruct(generated, real, *, base_weight, middle_bonus, smooth_window, fixed_median_depth):
    if len(generated)!=len(real) or not generated: raise ValueError("Matched nonempty offline inputs required")
    if smooth_window<1 or smooth_window%2!=1: raise ValueError("Positive odd smoothing window required")
    if not np.allclose([s.time_s for s in generated],[s.time_s for s in real],atol=1e-6,rtol=0):
        raise ValueError("The saved retargeted input must already use the generated time grid")
    def values(samples):return np.array([[s.x,s.y,s.z,s.rx,s.ry,s.rz] for s in samples])
    g=values(generated);r=values(real)
    if not np.isfinite(g).all() or not np.isfinite(r).all():raise ValueError("Nonfinite input")
    g[:,3:]=np.rad2deg(np.unwrap(np.deg2rad(g[:,3:]),axis=0))
    r[:,3:]=np.rad2deg(np.unwrap(np.deg2rad(r[:,3:]),axis=0))
    weight=np.clip(base_weight+middle_bonus*np.sin(np.pi*np.linspace(0,1,len(g))),0,1)[:,None]
    fused=centered_mean((1-weight)*g+weight*r,smooth_window)
    fused[:,3:]=(fused[:,3:]+180)%360-180
    if fixed_median_depth:fused[:,2]=np.median(np.r_[g[:,2],r[:,2]])
    return [PoseSample(s.frame,s.time_s,*map(float,row)) for s,row in zip(generated,fused)]


def diagnostics(generated,real,fused):
    def xyz(samples):return np.array([[s.x,s.y,s.z] for s in samples])
    g,r,f=map(xyz,(generated,real,fused))
    dg=np.linalg.norm(g-r,axis=1);df=np.linalg.norm(f-r,axis=1)
    return dict(samples=len(g),mean_G_to_R_mm=float(dg.mean()),mean_F_to_R_mm=float(df.mean()),
                relative_reduction_percent=float(100*(1-df.mean()/dg.mean())),
                fused_path_length_mm=float(np.linalg.norm(np.diff(f,axis=0),axis=1).sum()),
                interpretation="Internal offline consistency only; R participates in F. Coordinates use estimated intrinsics and modeled depth.")


def reproduce(task, out_dir, tolerance=2e-5):
    source=ROOT/'examples/trajectories'/task
    meta=json.loads((source/'fused_pose_trajectory_xyzrpy.json').read_text(encoding='utf-8-sig'))
    g=load_pose_trajectory(source/'generated_pose_xyzrpy.csv')
    r=load_pose_trajectory(source/'real_retargeted_pose_xyzrpy.csv')
    saved=load_pose_trajectory(source/'fused_pose_xyzrpy.csv')
    fused=reconstruct(g,r,base_weight=meta['real_weight_base'],middle_bonus=meta['mid_real_bonus'],
                      smooth_window=meta['smooth_window'],fixed_median_depth=task=='pour')
    attrs=('x','y','z','rx','ry','rz')
    error=np.array([[getattr(a,k)-getattr(b,k) for k in attrs] for a,b in zip(fused,saved)])
    error[:,3:]=(error[:,3:]+180)%360-180
    report=diagnostics(g,r,fused)
    report.update(task=task,max_absolute_saved_difference=float(np.max(np.abs(error))),tolerance=tolerance,
                  parameters={k:meta[k] for k in ('real_weight_base','mid_real_bonus','smooth_window')})
    report['matches_saved_curve']=bool(np.max(np.abs(error))<=tolerance)
    if not report['matches_saved_curve']:raise ValueError(f"{task}: recomputation does not match saved curve: {report}")
    out=Path(out_dir).resolve()
    for protected in ('examples','static','paper','reference_code'):
        if out.is_relative_to((ROOT/protected).resolve()):raise ValueError("Output would modify preserved inputs/assets")
    out.mkdir(parents=True,exist_ok=True)
    save_pose_csv(out/'recomputed_fused_pose_xyzrpy.csv',fused)
    (out/'diagnostics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=('pour','wipe','all'),default='all')
    parser.add_argument('--out-dir',type=Path,default=ROOT/'outputs/appendix_reproduction')
    args=parser.parse_args()
    for task in ('pour','wipe') if args.task=='all' else (args.task,):
        print(json.dumps(reproduce(task,args.out_dir/task),indent=2))

if __name__=='__main__':main()
