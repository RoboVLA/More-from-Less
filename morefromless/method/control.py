"""Equation-level reference controller. No robot drivers or historical settings.

Uses absolute 12-D targets (5 joints + gripper per arm). A caller supplies
calibration, Jacobians, task masks, collision linearization and a full nonlinear
collision check. Returned 'hold' decisions are requests to the hardware adapter;
they never silently become an unchecked position command.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import numpy as np

ARM = np.array([0,1,2,3,4,6,7,8,9,10])
GRIP = np.array([5,11])


def finite_array(value, shape=None):
    a=np.asarray(value,dtype=float)
    if (shape is not None and a.shape!=shape) or not np.isfinite(a).all():
        raise ValueError("Invalid shape or nonfinite control input")
    return a


def pixel_translation(delta_uv, depth, intrinsics, camera_to_base_rotation):
    uv=finite_array(delta_uv,(2,)); K=finite_array(intrinsics,(3,3))
    R=finite_array(camera_to_base_rotation,(3,3))
    if not np.isfinite(depth) or depth<=0 or K[0,0]<=0 or K[1,1]<=0:
        raise ValueError("Current measured depth and valid calibration are required")
    if not np.allclose(R.T@R,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(R),1,atol=1e-6):
        raise ValueError("Invalid camera rotation")
    return R@np.array([depth*uv[0]/K[0,0],depth*uv[1]/K[1,1],0.])


def task_masked_reference(current, base, jacobians, selections, targets, *, damping, max_residual, max_increment):
    current=finite_array(current,(12,));base=finite_array(base,(12,))
    if not np.isfinite([damping,max_residual,max_increment]).all() or damping<=0 or min(max_residual,max_increment)<=0:
        raise ValueError("Explicit positive DLS and validity limits required")
    if not len(jacobians)==len(selections)==len(targets)==2:
        raise ValueError("Both active and supporting arm tasks must be specified")
    rows=[];values=[]
    for arm,(J,S,x) in enumerate(zip(jacobians,selections,targets)):
        J=finite_array(J,(6,10));S=finite_array(S);x=finite_array(x,(6,))
        if S.ndim!=2 or S.shape[1]!=6 or not 1<=len(S)<=5 or np.linalg.matrix_rank(S)!=len(S):
            raise ValueError("Each arm requires 1..5 independent explicit task components")
        other=slice(5,10) if arm==0 else slice(0,5)
        if not np.allclose(J[:,other],0): raise ValueError("Jacobian arm-channel order mismatch")
        masked=S@J
        if np.linalg.matrix_rank(masked)<len(S): raise ValueError("Task is locally rank deficient")
        rows.append(masked);values.append(S@x)
    J=np.vstack(rows);x=np.concatenate(values)
    dq=np.linalg.solve(J.T@J+damping*damping*np.eye(10),J.T@x)
    if np.linalg.norm(J@dq-x)>max_residual or np.max(np.abs(dq))>max_increment:
        raise ValueError("Reference is unreachable under configured local limits")
    reference=current.copy();reference[ARM]+=dq;reference[GRIP]=base[GRIP]
    return reference


@dataclass(frozen=True)
class ImageReference:
    observation_time: float
    received_time: float
    pixels: np.ndarray
    relative_times: np.ndarray


class AsyncReferences:
    """One in-flight generator request. poll never waits for generation."""
    def __init__(self, generator):
        self.generator=generator;self.executor=ThreadPoolExecutor(max_workers=1)
        self.future=None;self.origin=None;self.latest=None;self.last_error=None

    def request(self, rgb, language, observation_time):
        if self.future is not None: return False
        if not np.isfinite(observation_time): raise ValueError("Finite capture time required")
        self.origin=float(observation_time)
        self.future=self.executor.submit(self.generator,np.array(rgb,copy=True),language)
        return True

    @property
    def pending(self):
        return self.future is not None

    def poll(self, now):
        if self.future is not None and self.future.done():
            try:
                pixels,times=self.future.result()
                pixels=finite_array(pixels);times=finite_array(times)
                if pixels.ndim!=2 or pixels.shape[1]!=2 or times.shape!=(len(pixels),) or len(times)<2 or np.any(np.diff(times)<=0) or times[0]<0:
                    raise ValueError("Generator must return a time-indexed 2-D image trajectory")
                self.latest=ImageReference(self.origin,float(now),pixels,times)
                self.last_error=None
            except Exception as exc:
                self.latest=None;self.last_error=type(exc).__name__
            finally:self.future=None
        return self.latest

    def close(self):
        self.executor.shutdown(wait=False,cancel_futures=True)


def gate_weight(risks, thresholds, scales, weights, bias, valid):
    risks=finite_array(risks,(4,));thresholds=finite_array(thresholds,(4,))
    scales=finite_array(scales,(4,));weights=finite_array(weights,(4,))
    if np.any(scales<=0) or not np.isfinite(bias): raise ValueError("Invalid frozen gate settings")
    if not valid or not np.any(risks>thresholds): return 0.
    score=np.clip(np.dot(weights,risks/scales)-bias,-60,60)
    return float(1/(1+np.exp(-score)))


@dataclass(frozen=True)
class QPResult:
    command: np.ndarray | None
    slack: np.ndarray
    reason: str


def shared_qp(raw, current, previous, lower, upper, step_limit, change_limit,
              hard_G, hard_h, soft_G, soft_h, slack_max, *, weight, slack_penalty,
              tolerance=1e-8, max_sweeps=10000):
    """Solve the strictly convex diagonal QP by dual coordinate ascent.

    hard_G @ delta >= hard_h; soft_G @ delta + slack >= soft_h.
    Joint/step/rate limits are hard. Never accept a nonconverged solution.
    This portable solver is not a validated high-frequency robot runtime.
    """
    raw,current,previous,lower,upper,step_limit,change_limit,weight=[
        finite_array(x,(12,)) for x in (raw,current,previous,lower,upper,step_limit,change_limit,weight)]
    H=finite_array(hard_G);h=finite_array(hard_h);S=finite_array(soft_G);s=finite_array(soft_h)
    caps=finite_array(slack_max)
    if H.shape!=(len(h),12) or S.shape!=(len(s),12) or caps.shape!=s.shape:
        raise ValueError("Constraint dimension mismatch")
    if not np.isfinite([slack_penalty,tolerance]).all() or np.any(weight<=0) or np.any(caps<0) or np.any(step_limit<0) or np.any(change_limit<0) or slack_penalty<=0 or tolerance<=0 or max_sweeps<1:
        raise ValueError("Invalid QP configuration")
    lo=np.maximum.reduce([lower,current-step_limit,previous-change_limit])
    hi=np.minimum.reduce([upper,current+step_limit,previous+change_limit])
    if np.any(lo>hi): return QPResult(None,np.zeros(len(s)),"infeasible_bounds")
    n=len(s);I=np.eye(12);Z=np.zeros((12,n))
    A=np.vstack([np.hstack([I,Z]),np.hstack([-I,Z]),
                 np.hstack([H,np.zeros((len(h),n))]),
                 np.hstack([S,np.eye(n)]),np.hstack([np.zeros((n,12)),np.eye(n)]),
                 np.hstack([np.zeros((n,12)),-np.eye(n)])])
    b=np.concatenate([lo-raw,raw-hi,h,s,np.zeros(n),-caps])
    inv=1/np.r_[weight,np.full(n,slack_penalty)]
    diagonal=(A*A*inv).sum(axis=1)
    if np.any((diagonal==0)&(b>tolerance)):return QPResult(None,np.zeros(n),"infeasible_constraint")
    keep=diagonal>0;A=A[keep];b=b[keep];diagonal=diagonal[keep]
    dual=np.zeros(len(b));z=np.zeros(12+n)
    for _ in range(max_sweeps):
        for i in range(len(b)):
            updated=max(0.,dual[i]+(b[i]-A[i]@z)/diagonal[i])
            z+=(updated-dual[i])*inv*A[i];dual[i]=updated
        residual=b-A@z
        if np.max(residual,initial=0)<=tolerance and np.max(np.abs(np.where(dual>tolerance,residual,np.maximum(residual,0))),initial=0)<=tolerance:
            return QPResult(raw+z[:12],z[12:],"accepted")
    return QPResult(None,z[12:],"qp_not_converged")


@dataclass(frozen=True)
class Decision:
    command: np.ndarray | None
    alpha: float
    mode: str
    reason: str
    reference_age: float | None


class OnlineController:
    """Single output chain: VLA -> gate -> shared QP -> nonlinear check.

    map_reference(image_reference, observation, base) must perform current-state
    alignment, depth/calibration, task-masked IK and reference validity checks.
    It returns an absolute reference or raises ValueError. build_constraints is
    called after candidate smoothing, so linearization uses the actual raw target.
    nonlinear_check must cover joint limits AND complete robot/environment geometry.
    """
    def __init__(self, policy, references, map_reference, build_constraints, nonlinear_check,
                 *, thresholds, scales, weights, bias, max_reference_age, gate_smoothing=1., command_smoothing=1.):
        if not np.isfinite(max_reference_age) or max_reference_age<=0 or not 0<gate_smoothing<=1 or not 0<command_smoothing<=1:
            raise ValueError("Explicit age and smoothing configuration required")
        self.policy=policy;self.references=references;self.map_reference=map_reference
        self.build_constraints=build_constraints;self.nonlinear_check=nonlinear_check
        self.thresholds=finite_array(thresholds,(4,));self.scales=finite_array(scales,(4,))
        self.weights=finite_array(weights,(4,));self.bias=bias;self.max_age=max_reference_age
        self.gs=gate_smoothing;self.cs=command_smoothing;self.previous_alpha=0.

    def tick(self, observation, current, previous, risks, now, request_reference=False):
        if not np.isfinite([now,observation['timestamp']]).all() or observation['timestamp']>now:
            raise ValueError("Current observation requires a valid capture timestamp")
        inference={k:observation[k] for k in ('rgb_global','rgb_left','rgb_right','language','joints')}
        base=finite_array(self.policy(inference),(32,12))[0]
        current=finite_array(current,(12,));previous=finite_array(previous,(12,))
        risks=finite_array(risks,(4,))
        image=self.references.poll(now);reference=None;age=None
        # Do not re-request on every high-risk tick while a fresh reference exists.
        stale=image is None or not 0<=now-image.observation_time<=self.max_age
        if request_reference or (np.any(risks>self.thresholds) and stale):
            self.references.request(observation['rgb_global'],observation['language'],observation['timestamp'])
        if image is not None and not self.references.pending:
            age=now-image.observation_time
            if 0<=age<=self.max_age and image.received_time<=now:
                try:
                    reference=finite_array(self.map_reference(image,observation,base),(12,)).copy()
                    reference[GRIP]=base[GRIP]
                except ValueError:reference=None
        proposed=gate_weight(risks,self.thresholds,self.scales,self.weights,self.bias,reference is not None)
        smoothed=self.gs*proposed+(1-self.gs)*self.previous_alpha
        # Hard gate AFTER smoothing: invalid/waiting/low-risk cannot leak old alpha.
        alpha=smoothed if reference is not None and np.any(risks>self.thresholds) else 0.
        self.previous_alpha=alpha
        raw=base if alpha==0 else (1-alpha)*base+alpha*reference
        raw=self.cs*raw+(1-self.cs)*previous
        result=shared_qp(raw,current,previous,**self.build_constraints(raw,current,previous,observation))
        if result.command is None:return Decision(None,alpha,"hold",result.reason,age)
        if not self.nonlinear_check(result.command,current,observation):
            return Decision(None,alpha,"hold","nonlinear_reject",age)
        return Decision(result.command,alpha,"execute","accepted",age)
