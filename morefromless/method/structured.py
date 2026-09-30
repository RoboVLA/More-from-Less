"""Differentiable objectives for Eqs. (4)-(7); plug in a real VLA and geometry.

All normalization/loss settings are explicit inputs, not historical defaults.
Only RGB, language and current joints are sent to the policy. Geometry callbacks
must return differentiable transforms and signed clearances in common units.
"""
from dataclasses import dataclass
import torch


def rotation_log(rotation):
    """SO(3) local log, differentiable near identity; reject the pi branch cut."""
    cosine = ((rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    if torch.any(cosine.detach() < -0.9999):
        raise ValueError("Relative rotation is outside the supported local log branch.")
    skew = torch.stack((rotation[..., 2, 1]-rotation[..., 1, 2],
                        rotation[..., 0, 2]-rotation[..., 2, 0],
                        rotation[..., 1, 0]-rotation[..., 0, 1]), -1)
    # Clamp protects the inactive acos branch from an infinite derivative at I.
    eps = 10 * torch.finfo(rotation.dtype).eps
    theta = torch.acos(cosine.clamp(-1+eps, 1-eps))
    regular = theta / (2 * torch.sin(theta))
    small = 0.5 + (1-cosine) / 6
    return skew * torch.where(cosine > 0.9999, small, regular)[..., None]


def geometry_relation(absolute_actions, fixed_context, geometry):
    """geometry(actions, context) -> left/right 4x4, self/env signed distances."""
    left, right, d_self, d_env = geometry(absolute_actions, fixed_context)
    inverse_rotation = left[..., :3, :3].transpose(-1, -2)
    translation = (inverse_rotation @ (right[..., :3, 3]-left[..., :3, 3])[..., None]).squeeze(-1)
    relative_rotation = inverse_rotation @ right[..., :3, :3]
    rho = torch.cat((translation, rotation_log(relative_rotation), d_self[..., None]), -1)
    return rho, d_self, d_env


def detached(value):
    if isinstance(value, torch.Tensor): return value.detach()
    if isinstance(value, dict): return {k: detached(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return type(value)(detached(v) for v in value)
    return value


@dataclass
class LossSettings:
    action_mean: torch.Tensor
    action_scale: torch.Tensor
    relation_mean: torch.Tensor
    relation_scale: torch.Tensor
    relation_diagonal: torch.Tensor
    lambda_relation: float
    lambda_safety: float
    minimum_self_distance: float
    minimum_environment_distance: float

    def validate(self):
        for name, count in (("action_mean",12),("action_scale",12),
                            ("relation_mean",7),("relation_scale",7),("relation_diagonal",7)):
            value=getattr(self,name)
            if value.shape != (count,) or not torch.isfinite(value).all():
                raise ValueError(f"Invalid {name}: expected {count} finite fixed values")
            if value.requires_grad: raise ValueError(f"{name} must be frozen")
        if torch.any(self.action_scale <= 0) or torch.any(self.relation_scale <= 0):
            raise ValueError("Normalization scales must be strictly positive")
        if not torch.isfinite(torch.tensor([self.lambda_relation,self.lambda_safety,self.minimum_self_distance,self.minimum_environment_distance])).all():
            raise ValueError("Loss configuration must be finite")
        if min(self.lambda_relation,self.lambda_safety,self.minimum_self_distance,
               self.minimum_environment_distance) < 0:
            raise ValueError("Loss weights and clearance margins must be nonnegative")


def structured_loss(predicted, demonstrated, valid_steps, context, geometry, settings):
    """Inputs may be normalized; losses/geometry use denormalized absolute targets."""
    settings.validate()
    if predicted.ndim != 3 or predicted.shape[1:] != (32,12) or demonstrated.shape != predicted.shape:
        raise ValueError("Expected aligned B x 32 x 12 action chunks")
    if valid_steps.shape != predicted.shape[:2] or valid_steps.dtype != torch.bool or not valid_steps.any():
        raise ValueError("A boolean mask with at least one valid action step is required")
    # Index BEFORE geometry evaluation so padding cannot create invalid rotations.
    pred=predicted[valid_steps]; target=demonstrated.detach()[valid_steps]
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise ValueError("Valid action steps must be finite")
    # Context must use the same masked-step indexing as actions.
    fixed=detached(context)
    if isinstance(fixed,dict):
        fixed={k:v[valid_steps] if isinstance(v,torch.Tensor) and v.shape[:2]==valid_steps.shape else v for k,v in fixed.items()}
    pred_abs=pred*settings.action_scale+settings.action_mean
    target_abs=target*settings.action_scale+settings.action_mean
    rho_p,d_self,d_env=geometry_relation(pred_abs,fixed,geometry)
    with torch.no_grad(): rho_t,_,_=geometry_relation(target_abs,fixed,geometry)
    if not all(torch.isfinite(v).all() for v in (rho_p,rho_t,d_self,d_env)):
        raise ValueError("Geometry returned nonfinite values")
    # BC sums channels of both arms; expectation is over valid steps.
    bc=(pred_abs-target_abs).abs().sum(-1).mean()
    p=(rho_p-settings.relation_mean)/settings.relation_scale
    t=(rho_t-settings.relation_mean)/settings.relation_scale
    rel=((p-t)*settings.relation_diagonal).square().sum(-1).mean()
    safe=(torch.relu(settings.minimum_self_distance-d_self).square()+
          torch.relu(settings.minimum_environment_distance-d_env).square()).mean()
    total=bc+settings.lambda_relation*rel+settings.lambda_safety*safe
    return {"loss":total,"bc":bc,"relation":rel,"safety":safe}


def train_step(policy, optimizer, observation, demonstrated, valid_steps, context, geometry, settings):
    required={"rgb_global","rgb_left","rgb_right","language","joints"}
    if set(observation)!=required: raise ValueError("Inference inputs must be exactly RGB x3, language and joints")
    optimizer.zero_grad(set_to_none=True)
    predicted=policy(**observation)
    losses=structured_loss(predicted,demonstrated,valid_steps,context,geometry,settings)
    losses["loss"].backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in policy.parameters()):
        optimizer.zero_grad(set_to_none=True)
        raise ValueError("Nonfinite policy gradient; optimizer step refused")
    optimizer.step()
    return {key:float(value.detach()) for key,value in losses.items()}
