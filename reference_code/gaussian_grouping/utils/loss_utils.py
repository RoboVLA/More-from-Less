# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def masked_l1_loss(network_output, gt, mask):
    mask = mask.float()[None, :, :].repeat(gt.shape[0], 1, 1)
    loss = torch.abs((network_output - gt)) * mask
    loss = loss.sum() / mask.sum().clamp_min(1.0)
    return loss


def weighted_l1_loss(network_output, gt, weight):
    loss = torch.abs((network_output - gt)) * weight
    return loss.mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(1).mean(1).mean(1)


def loss_cls_3d(features, predictions, k=5, lambda_val=2.0, max_points=200000, sample_size=800):
    """
    Original 3D neighborhood consistency regularizer.
    Kept for compatibility with the original project, although the new train.py
    does not use it by default.
    """
    if features.size(0) > max_points:
        indices = torch.randperm(features.size(0), device=features.device)[:max_points]
        features = features[indices]
        predictions = predictions[indices]

    if features.size(0) == 0:
        return torch.zeros([], device=predictions.device, dtype=predictions.dtype)

    sample_size = min(sample_size, features.size(0))
    indices = torch.randperm(features.size(0), device=features.device)[:sample_size]
    sample_features = features[indices]
    sample_preds = predictions[indices]

    k = min(k, features.size(0))
    dists = torch.cdist(sample_features, features)
    _, neighbor_indices_tensor = dists.topk(k, largest=False)

    neighbor_preds = predictions[neighbor_indices_tensor]

    kl = sample_preds.unsqueeze(1) * (
        torch.log(sample_preds.unsqueeze(1) + 1e-10) - torch.log(neighbor_preds + 1e-10)
    )
    loss = kl.sum(dim=-1).mean()

    num_classes = predictions.size(1)
    normalized_loss = loss / max(num_classes, 1)

    return lambda_val * normalized_loss


def contrastive_identity_loss(
    features,
    labels,
    temperature=0.1,
    ignore_index=-1,
    max_samples=1024,
    max_anchors=256,
    eps=1e-8,
):
    """
    Low-memory supervised contrastive / InfoNCE-style identity loss.

    Compared with full NxN contrastive learning, this version:
    1. downsamples the full sample pool to max_samples
    2. downsamples anchors to max_anchors
    3. computes [A, N] similarities instead of [N, N]

    Args:
        features: [N, D]
        labels:   [N]
        temperature: contrastive temperature
        ignore_index: ignored label id
        max_samples: maximum number of total samples kept in the comparison pool
        max_anchors: maximum number of anchor samples
    """
    device = features.device
    dtype = features.dtype

    if features.dim() != 2:
        raise ValueError(f"features must be [N, D], got shape {tuple(features.shape)}")
    if labels.dim() != 1:
        raise ValueError(f"labels must be [N], got shape {tuple(labels.shape)}")
    if features.size(0) != labels.size(0):
        raise ValueError("features and labels must have the same first dimension")

    # remove ignored labels
    valid = labels != ignore_index
    features = features[valid]
    labels = labels[valid]

    if features.size(0) < 2:
        return torch.zeros([], device=device, dtype=dtype)

    # global pool downsample
    if features.size(0) > max_samples:
        perm = torch.randperm(features.size(0), device=device)[:max_samples]
        features = features[perm]
        labels = labels[perm]

    # only keep classes that have at least two samples
    unique_labels, counts = torch.unique(labels, return_counts=True)
    valid_classes = unique_labels[counts >= 2]
    if valid_classes.numel() == 0:
        return torch.zeros([], device=device, dtype=dtype)

    valid_mask = torch.zeros_like(labels, dtype=torch.bool)
    for cls_id in valid_classes:
        valid_mask |= (labels == cls_id)

    features = features[valid_mask]
    labels = labels[valid_mask]

    if features.size(0) < 2:
        return torch.zeros([], device=device, dtype=dtype)

    # anchor downsample
    if features.size(0) > max_anchors:
        anchor_idx = torch.randperm(features.size(0), device=device)[:max_anchors]
        anchor_features = features[anchor_idx]
        anchor_labels = labels[anchor_idx]
        anchor_source_idx = anchor_idx
    else:
        anchor_features = features
        anchor_labels = labels
        anchor_source_idx = torch.arange(features.size(0), device=device)

    features = F.normalize(features, dim=1)
    anchor_features = F.normalize(anchor_features, dim=1)

    # [A, N] similarity matrix
    sim = torch.matmul(anchor_features, features.t()) / temperature

    # positive mask
    pos_mask = anchor_labels.unsqueeze(1) == labels.unsqueeze(0)

    # remove exact self-pairs for anchors that originate from the feature pool
    pos_mask[torch.arange(anchor_features.size(0), device=device), anchor_source_idx] = False

    valid_anchor = pos_mask.sum(dim=1) > 0
    if valid_anchor.sum() == 0:
        return torch.zeros([], device=device, dtype=dtype)

    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
    exp_sim = torch.exp(sim)

    pos_sum = (exp_sim * pos_mask.float()).sum(dim=1)
    all_sum = exp_sim.sum(dim=1)

    loss = -torch.log((pos_sum + eps) / (all_sum + eps))
    loss = loss[valid_anchor].mean()
    return loss


def orthogonal_decoupling_loss_soft(
    features,
    probs,
    exclude_background=True,
    eps=1e-8,
):
    """
    Orthogonal decoupling loss using soft assignments.

    Args:
        features: [N, D]
        probs:    [N, C]
        exclude_background: whether to exclude class 0 as background
    """
    device = features.device
    dtype = features.dtype

    if features.dim() != 2:
        raise ValueError(f"features must be [N, D], got shape {tuple(features.shape)}")
    if probs.dim() != 2:
        raise ValueError(f"probs must be [N, C], got shape {tuple(probs.shape)}")
    if features.size(0) != probs.size(0):
        raise ValueError("features and probs must have the same first dimension")

    if exclude_background and probs.size(1) > 1:
        probs = probs[:, 1:]

    if probs.size(1) < 2:
        return torch.zeros([], device=device, dtype=dtype)

    features = F.normalize(features, dim=1)

    class_mass = probs.sum(dim=0)
    valid_cls = class_mass > eps
    if valid_cls.sum() < 2:
        return torch.zeros([], device=device, dtype=dtype)

    probs = probs[:, valid_cls]
    class_mass = class_mass[valid_cls]

    centers = (probs.t() @ features) / (class_mass.unsqueeze(1) + eps)  # [K, D]
    centers = F.normalize(centers, dim=1)

    gram = centers @ centers.t()  # cosine similarity matrix
    k = gram.size(0)
    if k < 2:
        return torch.zeros([], device=device, dtype=dtype)

    eye = torch.eye(k, device=device, dtype=gram.dtype)
    off_diag = gram - eye

    loss = off_diag.pow(2).sum() / (k * (k - 1) + eps)
    return loss
def knn_laplacian_loss(xyz, active_mask, k=8):
    pts = xyz[active_mask]
    if pts.shape[0] < k + 1:
        return torch.tensor(0.0, device=xyz.device)

    dist = torch.cdist(pts, pts)
    knn_idx = dist.topk(k=k+1, largest=False).indices[:, 1:]
    neighbors = pts[knn_idx]
    mean_neighbors = neighbors.mean(dim=1)

    loss = ((pts - mean_neighbors) ** 2).sum(dim=1).mean()
    return loss