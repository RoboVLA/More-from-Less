"""Failure discovery -> local candidates -> joint acceptance -> retraining.

This is newly implemented orchestration. Real simulator, feasibility predicates,
policy training and state representations are supplied by the caller.
"""
from dataclasses import dataclass
from typing import Any, Callable
import numpy as np


def local_states(failed_state, radius, lower, upper, count, rng):
    """Bounded uniform local proposals; caller still applies all feasibility checks."""
    center=np.asarray(failed_state,dtype=float)
    radius=np.broadcast_to(np.asarray(radius,dtype=float),center.shape)
    lower=np.broadcast_to(np.asarray(lower,dtype=float),center.shape)
    upper=np.broadcast_to(np.asarray(upper,dtype=float),center.shape)
    if center.ndim!=1 or not all(np.isfinite(v).all() for v in (center,radius,lower,upper)) or np.any(radius<0) or np.any(lower>upper) or count<1:
        raise ValueError("Invalid local sampling bounds")
    lo=np.maximum(lower,center-radius);hi=np.minimum(upper,center+radius)
    if np.any(lo>hi):return []
    return [rng.uniform(lo,hi) for _ in range(count)]


@dataclass(frozen=True)
class Rollout:
    identifier: str
    states: tuple
    actions: tuple
    complete: bool
    task_success: bool
    physics_valid: bool
    safety_valid: bool
    failure_state: Any = None
    source: str = "simulation"
    split: str = "adaptation"
    quality: float = 1.0


def acceptance(trajectory):
    return (trajectory.source=="simulation" and trajectory.split=="adaptation"
            and trajectory.complete and trajectory.task_success
            and trajectory.physics_valid and trajectory.safety_valid
            and len(trajectory.actions)>0 and len(trajectory.states)==len(trajectory.actions)+1
            and np.isfinite(trajectory.quality) and trajectory.quality>0)


def real_dominant_weights(real_count, accepted, real_mass):
    """Normalize group masses independently; quality cannot dilute real dominance."""
    if real_count<=0 or not 0.5<real_mass<=1:
        raise ValueError("Require real demonstrations and explicit real mass > 0.5")
    if any(not acceptance(t) for t in accepted): raise ValueError("Unaccepted virtual data")
    if not accepted: return np.full(real_count,1/real_count),np.empty(0)
    quality=np.array([t.quality for t in accepted],dtype=float)
    return np.full(real_count,real_mass/real_count),(1-real_mass)*quality/quality.sum()


def expand(policy, real_data, adaptation_initial_states, *, rollout: Callable,
           sample_near: Callable, feasible: Callable, retrain: Callable,
           max_rounds: int, real_mass: float, residual_explorer=None):
    if max_rounds<1 or not adaptation_initial_states: raise ValueError("Explicit adaptation states and round budget required")
    real_dominant_weights(len(real_data),[],real_mass)
    accepted=[]; seen=set(); records=[]
    for iteration in range(1,max_rounds+1):
        # The first failure neighborhood comes from this initial-policy rollout.
        probes=[rollout(policy,state,None) for state in adaptation_initial_states]
        if any(t.source!="simulation" or t.split!="adaptation" for t in probes):
            raise ValueError("Failure discovery must exclude held-out evaluation data")
        failures=[t.failure_state for t in probes if not t.task_success and t.failure_state is not None]
        candidates=feasible_count=accepted_count=0
        for failed in failures:
            for state in sample_near(failed):
                if not feasible(state): continue
                feasible_count+=1
                # residual_explorer is visible only to simulation rollout.
                candidate=rollout(policy,state,residual_explorer)
                candidates+=1
                if acceptance(candidate) and candidate.identifier not in seen:
                    accepted.append(candidate);seen.add(candidate.identifier);accepted_count+=1
        reason="no_failures" if not failures else "no_new_accepted" if not accepted_count else "round_limit" if iteration==max_rounds else "continue"
        records.append(dict(round=iteration,discovery_rollouts=len(probes),failures=len(failures),
                            feasible_initial_states=feasible_count,candidates=candidates,
                            accepted=accepted_count,cumulative_accepted=len(accepted),stop_reason=reason))
        if not accepted_count: break
        rw,vw=real_dominant_weights(len(real_data),accepted,real_mass)
        policy=retrain(policy,tuple(real_data),tuple(accepted),rw,vw)
    return policy,tuple(accepted),records
