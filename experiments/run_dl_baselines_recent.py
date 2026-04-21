"""
Additional DL baselines (2022-2024 era) for EWGB-TAD paper.
Runs on GPU server. Self-contained.

Methods:
1. Anomaly Transformer (Xu et al., ICLR 2022) - Association Discrepancy
2. DCdetector (Yang et al., KDD 2023) - Dual Attention Contrastive

Usage:
    python experiments_dl2.py --dataset all --gpu 0
"""

import numpy as np
import time
import json
import os
import sys
import argparse
import math
from collections import defaultdict
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ===================== Anomaly Transformer =====================
# Xu et al., "Anomaly Transformer: Time Series Anomaly Detection
# with Association Discrepancy", ICLR 2022

class AnomalyAttention(nn.Module):
    """Anomaly attention with learnable prior-association and series-association."""
    def __init__(self, d_model, nhead, seq_len):
        super().__init__()
        self.d_k = d_model // nhead
        self.nhead = nhead
        self.seq_len = seq_len

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Learnable prior-association (Gaussian kernel scale)
        self.sigma = nn.Parameter(torch.ones(nhead, 1, 1))

    def forward(self, x):
        B, L, D = x.shape
        H = self.nhead

        Q = self.W_Q(x).view(B, L, H, self.d_k).transpose(1, 2)  # (B, H, L, d_k)
        K = self.W_K(x).view(B, L, H, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, L, H, self.d_k).transpose(1, 2)

        # Series-association (standard attention)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        series_assoc = torch.softmax(scores, dim=-1)

        # Prior-association (Gaussian kernel based on distance)
        dist = torch.abs(torch.arange(L, device=x.device).float().unsqueeze(0) -
                         torch.arange(L, device=x.device).float().unsqueeze(1))
        sigma = torch.clamp(self.sigma, min=0.1)
        prior_assoc = torch.softmax(-dist.unsqueeze(0) / (2 * sigma ** 2), dim=-1)
        prior_assoc = prior_assoc.unsqueeze(0).expand(B, -1, -1, -1)

        # Association discrepancy (KL divergence)
        series_log = torch.log(series_assoc + 1e-8)
        prior_log = torch.log(prior_assoc + 1e-8)
        kl_sp = F.kl_div(prior_log, series_assoc, reduction='none').sum(dim=-1)
        kl_ps = F.kl_div(series_log, prior_assoc, reduction='none').sum(dim=-1)
        assoc_disc = (kl_sp + kl_ps).mean(dim=1)  # (B, L)

        # Attention output
        attn_out = torch.matmul(series_assoc, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(attn_out)

        return out, assoc_disc


class AnomalyTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, seq_len, dim_ff=128, dropout=0.1):
        super().__init__()
        self.attn = AnomalyAttention(d_model, nhead, seq_len)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, assoc_disc = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x, assoc_disc


class AnomalyTransformerModel(nn.Module):
    """Anomaly Transformer: reconstruction + association discrepancy."""
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=3, seq_len=32):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            AnomalyTransformerLayer(d_model, nhead, seq_len)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x):
        h = self.input_proj(x)
        assoc_discs = []
        for layer in self.layers:
            h, disc = layer(h)
            assoc_discs.append(disc)
        recon = self.output_proj(h)
        # Average association discrepancy across layers
        avg_disc = torch.stack(assoc_discs, dim=0).mean(dim=0)  # (B, L)
        return recon, avg_disc


# ===================== DCdetector =====================
# Yang et al., "DCdetector: Dual Attention Contrastive Representation
# Learning for Time Series Anomaly Detection", KDD 2023

class PatchEmbedding(nn.Module):
    """Split sequence into patches and embed."""
    def __init__(self, input_dim, d_model, patch_len, stride):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len * input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, L, D = x.shape
        # Create patches
        patches = []
        for i in range(0, L - self.patch_len + 1, self.stride):
            patch = x[:, i:i + self.patch_len, :].reshape(B, -1)
            patches.append(patch)
        if len(patches) == 0:
            patches.append(x.reshape(B, -1)[:, :self.patch_len * D])
        patches = torch.stack(patches, dim=1)  # (B, n_patches, patch_len*D)
        return self.norm(self.proj(patches))


class ContrastiveAttention(nn.Module):
    """Attention module for DCdetector."""
    def __init__(self, d_model, nhead):
        super().__init__()
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        B, L_q, D = query.shape
        L_k = key.shape[1]
        H = self.nhead

        Q = self.W_Q(query).view(B, L_q, H, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(B, L_k, H, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(B, L_k, H, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L_q, D)
        return self.out_proj(out), attn


class DCdetectorModel(nn.Module):
    """DCdetector with dual (patch-wise and channel-wise) attention."""
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2,
                 seq_len=32, patch_len=4, stride=2):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # Patch embedding
        self.patch_embed = PatchEmbedding(input_dim, d_model, patch_len, stride)
        n_patches = max(1, (seq_len - patch_len) // stride + 1)

        # Positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)

        # Patch-wise attention layers (self-attention on patches)
        self.patch_layers = nn.ModuleList([
            ContrastiveAttention(d_model, nhead) for _ in range(num_layers)
        ])
        self.patch_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])

        # Channel-wise attention layers (cross-attention)
        self.channel_layers = nn.ModuleList([
            ContrastiveAttention(d_model, nhead) for _ in range(num_layers)
        ])
        self.channel_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])

        # Reconstruction heads
        self.recon_patch = nn.Linear(d_model, patch_len * input_dim)
        self.recon_channel = nn.Linear(d_model, patch_len * input_dim)

        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = n_patches

    def forward(self, x):
        B, L, D = x.shape

        # Patch embedding
        patches = self.patch_embed(x)  # (B, n_patches, d_model)
        patches = patches + self.pos_enc[:, :patches.shape[1], :]

        # Patch-wise path (self-attention)
        h_patch = patches
        patch_attns = []
        for layer, norm in zip(self.patch_layers, self.patch_norms):
            h_normed = norm(h_patch)
            attn_out, attn = layer(h_normed, h_normed, h_normed)
            h_patch = h_patch + attn_out
            patch_attns.append(attn)

        # Channel-wise path (using reversed patch order as cross-attention target)
        h_channel = patches
        channel_attns = []
        reversed_patches = torch.flip(patches, dims=[1])
        for layer, norm in zip(self.channel_layers, self.channel_norms):
            h_normed = norm(h_channel)
            attn_out, attn = layer(h_normed, reversed_patches, reversed_patches)
            h_channel = h_channel + attn_out
            channel_attns.append(attn)

        # Reconstruction from both paths
        recon_p = self.recon_patch(h_patch)    # (B, n_patches, patch_len*D)
        recon_c = self.recon_channel(h_channel)

        # Contrastive discrepancy: difference between two reconstruction paths
        return recon_p, recon_c, h_patch, h_channel


# ===================== Feature/Data Utils (from experiments_dl.py) =====================

def extract_spatial_features(trajectory):
    diffs = np.diff(trajectory, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    total_length = np.sum(seg_lengths)
    od_vec = trajectory[-1] - trajectory[0]
    od_distance = np.linalg.norm(od_vec)
    detour_ratio = total_length / (od_distance + 1e-8)
    if od_distance > 1e-8:
        od_unit = od_vec / od_distance
        od_perp = np.array([-od_unit[1], od_unit[0]])
        lateral_devs = np.abs(np.dot(trajectory - trajectory[0], od_perp))
    else:
        lateral_devs = np.linalg.norm(trajectory - trajectory[0], axis=1)
    mean_lateral_dev = np.mean(lateral_devs)
    max_lateral_dev = np.max(lateral_devs)
    bbox_area = (trajectory[:, 0].max() - trajectory[:, 0].min()) * (trajectory[:, 1].max() - trajectory[:, 1].min())
    centroid = np.mean(trajectory, axis=0)
    rog = np.sqrt(np.mean(np.sum((trajectory - centroid) ** 2, axis=1)))
    return np.array([total_length, od_distance, detour_ratio,
                     mean_lateral_dev, max_lateral_dev, bbox_area, rog])


def extract_kinematic_features(trajectory, dt=1.0):
    diffs = np.diff(trajectory, axis=0)
    speeds = np.linalg.norm(diffs, axis=1) / dt
    mean_speed = np.mean(speeds)
    std_speed = np.std(speeds)
    max_speed = np.max(speeds)
    if len(speeds) > 1:
        accels = np.abs(np.diff(speeds)) / dt
        mean_accel, std_accel = np.mean(accels), np.std(accels)
    else:
        mean_accel = std_accel = 0.0
    stop_ratio = np.sum(speeds < mean_speed * 0.1 + 1e-8) / len(speeds)
    if len(diffs) > 1:
        angles = np.arctan2(diffs[:, 1], diffs[:, 0])
        turn_angles = np.abs(np.diff(angles))
        turn_angles = np.minimum(turn_angles, 2 * np.pi - turn_angles)
        mean_turn, max_turn = np.mean(turn_angles), np.max(turn_angles)
    else:
        mean_turn = max_turn = 0.0
    return np.array([mean_speed, std_speed, max_speed, mean_accel, std_accel,
                     stop_ratio, mean_turn, max_turn])


def compute_entropy(values, n_bins=16):
    if len(values) < 2:
        return 0.0
    hist, _ = np.histogram(values, bins=n_bins, density=False)
    hist = hist / (hist.sum() + 1e-8)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist + 1e-12))


def extract_entropy_features(trajectory, n_bins=16, grid_size=10):
    diffs = np.diff(trajectory, axis=0)
    speeds = np.linalg.norm(diffs, axis=1)
    speed_entropy = compute_entropy(speeds, n_bins)
    accel_entropy = compute_entropy(np.diff(speeds), n_bins) if len(speeds) > 1 else 0.0
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    heading_entropy = compute_entropy(headings, n_bins)
    if len(headings) > 1:
        turns = np.diff(headings)
        turns = np.arctan2(np.sin(turns), np.cos(turns))
        turn_entropy = compute_entropy(turns, n_bins)
    else:
        turn_entropy = 0.0
    traj_min = trajectory.min(axis=0)
    traj_range = (trajectory.max(axis=0) - trajectory.min(axis=0)) + 1e-8
    norm_traj = (trajectory - traj_min) / traj_range
    grid_indices = np.clip((norm_traj * grid_size).astype(int), 0, grid_size - 1)
    grid_cells = grid_indices[:, 0] * grid_size + grid_indices[:, 1]
    unique_cells, cell_counts = np.unique(grid_cells, return_counts=True)
    cell_probs = cell_counts / cell_counts.sum()
    grid_occ_entropy = -np.sum(cell_probs * np.log2(cell_probs + 1e-12))
    if len(grid_cells) > 1:
        transitions = grid_cells[:-1] * grid_size * grid_size + grid_cells[1:]
        unique_trans, trans_counts = np.unique(transitions, return_counts=True)
        trans_probs = trans_counts / trans_counts.sum()
        grid_trans_entropy = -np.sum(trans_probs * np.log2(trans_probs + 1e-12))
    else:
        grid_trans_entropy = 0.0
    return np.array([speed_entropy, accel_entropy, heading_entropy,
                     turn_entropy, grid_occ_entropy, grid_trans_entropy])


def extract_all_features(trajectories):
    spatial = np.array([extract_spatial_features(t) for t in trajectories])
    kinematic = np.array([extract_kinematic_features(t) for t in trajectories])
    entropy = np.array([extract_entropy_features(t) for t in trajectories])
    return np.hstack([spatial, kinematic, entropy])


def resample_trajectory(traj, target_len=32):
    n = len(traj)
    if n == target_len:
        return traj
    t_orig = np.linspace(0, 1, n)
    t_new = np.linspace(0, 1, target_len)
    return np.column_stack([np.interp(t_new, t_orig, traj[:, d]) for d in range(traj.shape[1])])


# ===================== Data Generators (same as experiments_dl.py) =====================

def generate_synthetic(n_normal=5000, contamination=0.1, seq_len=32, n_routes=6, seed=42):
    rng = np.random.RandomState(seed)
    n_anomaly = int(n_normal * contamination / (1 - contamination))
    n_per_type = max(1, n_anomaly // 4)
    route_defs = []
    for i in range(n_routes):
        angle = 2 * np.pi * i / n_routes
        start = np.array([0.5 + 0.4 * np.cos(angle), 0.5 + 0.4 * np.sin(angle)])
        end = np.array([0.5 + 0.4 * np.cos(angle + np.pi), 0.5 + 0.4 * np.sin(angle + np.pi)])
        route_defs.append((start, end))

    def make_normal(route_idx):
        start, end = route_defs[route_idx]
        t = np.linspace(0, 1, seq_len).reshape(-1, 1)
        path = start + t * (end - start)
        path += 4 * t * (1 - t) * rng.normal(0, 0.04, (1, 2))
        path += rng.normal(0, 0.02, path.shape)
        return path

    trajectories, labels, anom_types = [], [], []
    for i in range(n_normal):
        trajectories.append(make_normal(i % n_routes))
        labels.append(0); anom_types.append(-1)

    for atype in range(4):
        for _ in range(n_per_type):
            ridx = rng.randint(0, n_routes)
            traj = make_normal(ridx)
            od_dist = np.linalg.norm(route_defs[ridx][1] - route_defs[ridx][0])
            direction = route_defs[ridx][1] - route_defs[ridx][0]
            perp = np.array([-direction[1], direction[0]])
            perp = perp / (np.linalg.norm(perp) + 1e-8)
            if atype == 0:
                s, e = int(0.3*seq_len), int(0.7*seq_len)
                mag = rng.uniform(0.1, 0.25) * od_dist
                sign = rng.choice([-1, 1])
                for k in range(s, e):
                    frac = (k-s)/(e-s+1e-8)
                    traj[k] += sign * perp * mag * np.sin(np.pi * frac)
            elif atype == 1:
                ci = int(rng.uniform(0.4, 0.6)*seq_len)
                r = rng.uniform(0.05, 0.15)*od_dist
                nl = min(8, seq_len//4)
                si = max(0, ci-nl//2)
                for k in range(nl):
                    if si+k < seq_len:
                        a = 2*np.pi*k/nl
                        traj[si+k] += np.array([r*np.cos(a), r*np.sin(a)])
            elif atype == 2:
                sp = int(rng.uniform(0.2, 0.5)*seq_len)
                ns = min(6, seq_len//5)
                cp = traj[sp].copy()
                for k in range(ns):
                    if sp+k < seq_len:
                        traj[sp+k] = cp + rng.normal(0, 0.005, 2)
            elif atype == 3:
                ds = int(rng.uniform(0.5, 0.7)*seq_len)
                ad = rng.uniform(30, 60)*np.pi/180*rng.choice([-1, 1])
                for k in range(ds, seq_len):
                    if k > 0:
                        diff = traj[k]-traj[k-1]
                        ca, sa = np.cos(ad), np.sin(ad)
                        traj[k] = traj[k-1]+np.array([ca*diff[0]-sa*diff[1], sa*diff[0]+ca*diff[1]])
                    ad *= 0.95
            trajectories.append(traj); labels.append(1); anom_types.append(atype)

    perm = rng.permutation(len(trajectories))
    trajectories = [trajectories[i] for i in perm]
    return trajectories, np.array(labels)[perm], np.array(anom_types)[perm]


def generate_grid_network(n_normal=5000, contamination=0.1, seq_len=32, seed=42):
    rng = np.random.RandomState(seed)
    n_anomaly = int(n_normal * contamination / (1 - contamination))
    n_per_type = max(1, n_anomaly // 4)
    grid_size = 6
    routes = [((0,0),(5,5)),((5,0),(0,5)),((0,2),(5,2)),((2,0),(2,5)),
              ((0,0),(5,0)),((0,5),(5,5)),((3,0),(3,5)),((0,3),(5,3))]

    def make_grid_path(s, e):
        si,sj = s; ei,ej = e
        path_nodes = [(si,sj)]
        ci,cj = si,sj
        while ci!=ei or cj!=ej:
            if rng.random()<0.5 and ci!=ei:
                ci += 1 if ei>ci else -1
            elif cj!=ej:
                cj += 1 if ej>cj else -1
            elif ci!=ei:
                ci += 1 if ei>ci else -1
            path_nodes.append((ci,cj))
        points = np.array(path_nodes, dtype=float)/(grid_size-1)
        if len(points)<2:
            points = np.array([points[0], points[0]+[0.01,0.01]])
        t_orig = np.linspace(0,1,len(points))
        t_new = np.linspace(0,1,seq_len)
        traj = np.column_stack([np.interp(t_new, t_orig, points[:,0]),
                                np.interp(t_new, t_orig, points[:,1])])
        traj += rng.normal(0, 0.01, traj.shape)
        return traj

    trajectories, labels, anom_types = [], [], []
    for i in range(n_normal):
        s,e = routes[i%len(routes)]
        trajectories.append(make_grid_path(s,e))
        labels.append(0); anom_types.append(-1)

    for atype in range(4):
        for _ in range(n_per_type):
            ridx = rng.randint(0,len(routes))
            s,e = routes[ridx]
            traj = make_grid_path(s,e)
            od_dist = np.linalg.norm(traj[-1]-traj[0])
            direction = traj[-1]-traj[0]
            perp = np.array([-direction[1], direction[0]])
            perp = perp/(np.linalg.norm(perp)+1e-8)
            if atype==0:
                si,ei = int(0.3*seq_len), int(0.7*seq_len)
                mag = rng.uniform(0.1,0.25)*od_dist
                sign = rng.choice([-1,1])
                for k in range(si,ei):
                    frac = (k-si)/(ei-si+1e-8)
                    traj[k] += sign*perp*mag*np.sin(np.pi*frac)
            elif atype==1:
                ci = seq_len//2; r = rng.uniform(0.05,0.15)*od_dist
                nl = min(8, seq_len//4)
                for k in range(nl):
                    idx = ci-nl//2+k
                    if 0<=idx<seq_len:
                        a = 2*np.pi*k/nl
                        traj[idx] += np.array([r*np.cos(a), r*np.sin(a)])
            elif atype==2:
                sp = int(rng.uniform(0.2,0.5)*seq_len)
                ns = min(6,seq_len//5); cp = traj[sp].copy()
                for k in range(ns):
                    if sp+k<seq_len: traj[sp+k] = cp+rng.normal(0,0.005,2)
            elif atype==3:
                ds = int(rng.uniform(0.5,0.7)*seq_len)
                ad = rng.uniform(30,60)*np.pi/180*rng.choice([-1,1])
                for k in range(ds,seq_len):
                    if k>0:
                        diff = traj[k]-traj[k-1]
                        ca,sa = np.cos(ad),np.sin(ad)
                        traj[k] = traj[k-1]+np.array([ca*diff[0]-sa*diff[1],sa*diff[0]+ca*diff[1]])
                    ad *= 0.95
            trajectories.append(traj); labels.append(1); anom_types.append(atype)

    perm = rng.permutation(len(trajectories))
    trajectories = [trajectories[i] for i in perm]
    return trajectories, np.array(labels)[perm], np.array(anom_types)[perm]


def _gen_clustered(n_normal, contamination, seq_len, seed, n_clusters, lon_range, lat_range, noise_std):
    rng = np.random.RandomState(seed)
    n_anomaly = int(n_normal * contamination / (1-contamination))
    n_per_type = max(1, n_anomaly//4)

    n_od = n_normal*3
    starts = np.column_stack([rng.uniform(*lon_range, n_od), rng.uniform(*lat_range, n_od)])
    ends = np.column_stack([rng.uniform(*lon_range, n_od), rng.uniform(*lat_range, n_od)])
    dists = np.linalg.norm(ends-starts, axis=1)
    valid = (dists>0.005)&(dists<0.15 if lon_range[1]-lon_range[0]<0.5 else dists<0.3)
    starts,ends = starts[valid], ends[valid]
    od_flat = np.hstack([starts, ends])
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cl_labels = km.fit_predict(od_flat[:min(len(od_flat), n_normal*2)])
    clusters = defaultdict(list)
    for i,cl in enumerate(cl_labels): clusters[cl].append(i)

    templates = {}
    for cl_id in clusters:
        center = km.cluster_centers_[cl_id]
        cs,ce = center[:2], center[2:]
        direction = ce-cs
        perp = np.array([-direction[1], direction[0]])
        perp = perp/(np.linalg.norm(perp)+1e-8)
        t = np.linspace(0,1,seq_len)
        template = np.zeros((seq_len,2))
        for _ in range(rng.randint(1,4)):
            bt = rng.uniform(0.15,0.85); bs = rng.uniform(0.08,0.2)
            bm = rng.normal(0,0.1)*np.linalg.norm(direction)
            template += np.outer(np.exp(-0.5*((t-bt)/bs)**2), perp*bm)
        templates[cl_id] = template

    trajectories, labels, anom_types = [], [], []
    for i in range(n_normal):
        cl_id = i%n_clusters
        if clusters[cl_id]:
            idx = clusters[cl_id][i%len(clusters[cl_id])]
            s,e = starts[idx], ends[idx]
        else:
            s = np.array([rng.uniform(*lon_range), rng.uniform(*lat_range)])
            e = s+rng.uniform(-0.05,0.05,2)
        t_arr = np.linspace(0,1,seq_len).reshape(-1,1)
        path = s+t_arr*(e-s)
        if cl_id in templates: path += templates[cl_id]*rng.uniform(0.8,1.2)
        path += rng.normal(0, noise_std, path.shape)
        trajectories.append(path); labels.append(0); anom_types.append(-1)

    for atype in range(4):
        for _ in range(n_per_type):
            idx = rng.randint(0, n_normal)
            traj = trajectories[idx].copy()
            od_dist = np.linalg.norm(traj[-1]-traj[0])
            direction = traj[-1]-traj[0]
            perp = np.array([-direction[1], direction[0]])
            perp = perp/(np.linalg.norm(perp)+1e-8)
            if atype==0:
                si,ei = int(0.3*seq_len), int(0.7*seq_len)
                mag = rng.uniform(0.15,0.35)*od_dist; sign = rng.choice([-1,1])
                for k in range(si,ei):
                    frac = (k-si)/(ei-si+1e-8)
                    traj[k] += sign*perp*mag*np.sin(np.pi*frac)
            elif atype==1:
                ci = seq_len//2; r = rng.uniform(0.08,0.2)*od_dist
                nl = min(10,seq_len//3); si = max(0,ci-nl//2)
                for k in range(nl):
                    if si+k<seq_len:
                        a = 2*np.pi*k/nl
                        traj[si+k] += np.array([r*np.cos(a), r*np.sin(a)])
            elif atype==2:
                sp = int(rng.uniform(0.2,0.5)*seq_len)
                ns = min(8,seq_len//4); cp = traj[sp].copy()
                for k in range(ns):
                    if sp+k<seq_len: traj[sp+k] = cp+rng.normal(0,od_dist*0.003,2)
            elif atype==3:
                ds = int(rng.uniform(0.4,0.6)*seq_len)
                ad = rng.uniform(40,80)*np.pi/180*rng.choice([-1,1])
                for k in range(ds,seq_len):
                    if k>0:
                        diff = traj[k]-traj[k-1]
                        if np.linalg.norm(diff)<1e-10: diff = (traj[-1]-traj[0])/seq_len
                        ca,sa = np.cos(ad),np.sin(ad)
                        traj[k] = traj[k-1]+np.array([ca*diff[0]-sa*diff[1],sa*diff[0]+ca*diff[1]])*1.3
                    ad *= 0.92
            trajectories.append(traj); labels.append(1); anom_types.append(atype)

    perm = rng.permutation(len(trajectories))
    trajectories = [trajectories[i] for i in perm]
    return trajectories, np.array(labels)[perm], np.array(anom_types)[perm]


def generate_porto_like(n_normal=5000, contamination=0.1, seed=42):
    return _gen_clustered(n_normal, contamination, 32, seed, 20,
                          (-8.70,-8.54), (41.10,41.20), 0.0001)

def generate_geolife_like(n_normal=5000, contamination=0.1, seed=42):
    return _gen_clustered(n_normal, contamination, 32, seed, 15,
                          (116.1,116.65), (39.75,40.15), 0.0002)


# ===================== Training Functions =====================

def train_anomaly_transformer(seqs, device, epochs=50, batch_size=128, lr=1e-3):
    """Train Anomaly Transformer and return anomaly scores."""
    scaler = StandardScaler()
    N, L, D = seqs.shape
    seqs_flat = scaler.fit_transform(seqs.reshape(N*L, D))
    seqs_scaled = seqs_flat.reshape(N, L, D)

    X = torch.FloatTensor(seqs_scaled).to(device)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)

    model = AnomalyTransformerModel(input_dim=D, d_model=64, nhead=4,
                                     num_layers=3, seq_len=L).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training with minimax strategy (as in the paper)
    model.train()
    for epoch in range(epochs):
        for (batch,) in loader:
            # Phase 1: minimize reconstruction, maximize association discrepancy
            recon, assoc_disc = model(batch)
            recon_loss = ((recon - batch) ** 2).mean()
            disc_loss = assoc_disc.mean()
            loss = recon_loss - 0.01 * disc_loss  # lambda=0.01
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Phase 2: minimize association discrepancy (adversarial)
            recon, assoc_disc = model(batch)
            loss2 = 0.01 * assoc_disc.mean()
            optimizer.zero_grad()
            loss2.backward()
            optimizer.step()

    # Scoring: reconstruction error * association discrepancy
    model.eval()
    with torch.no_grad():
        recon, assoc_disc = model(X)
        recon_err = ((recon - X) ** 2).mean(dim=2)  # (N, L)
        # Combine: point-wise anomaly score = recon_error * softmax(assoc_disc)
        disc_weights = torch.softmax(assoc_disc, dim=-1)
        point_scores = recon_err * disc_weights
        scores = point_scores.mean(dim=1).cpu().numpy()  # trajectory-level
    return scores


def train_dcdetector(seqs, device, epochs=50, batch_size=128, lr=1e-3):
    """Train DCdetector and return anomaly scores."""
    scaler = StandardScaler()
    N, L, D = seqs.shape
    seqs_flat = scaler.fit_transform(seqs.reshape(N*L, D))
    seqs_scaled = seqs_flat.reshape(N, L, D)

    X = torch.FloatTensor(seqs_scaled).to(device)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)

    model = DCdetectorModel(input_dim=D, d_model=64, nhead=4, num_layers=2,
                             seq_len=L, patch_len=4, stride=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for (batch,) in loader:
            recon_p, recon_c, h_patch, h_channel = model(batch)

            # Reconstruction loss for both paths
            # Average over patches -> reconstruct original
            loss_recon = ((recon_p - recon_c) ** 2).mean()

            # Contrastive loss: patch vs channel representations should be similar
            # for normal data (both paths see the same normal patterns)
            h_p_norm = F.normalize(h_patch, dim=-1)
            h_c_norm = F.normalize(h_channel, dim=-1)
            contrastive_loss = (1 - (h_p_norm * h_c_norm).sum(dim=-1)).mean()

            loss = loss_recon + 0.1 * contrastive_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Scoring: discrepancy between two reconstruction paths
    model.eval()
    with torch.no_grad():
        recon_p, recon_c, h_patch, h_channel = model(X)
        # Anomaly score = disagreement between patch and channel paths
        recon_diff = ((recon_p - recon_c) ** 2).mean(dim=2)  # (N, n_patches)
        h_diff = ((h_patch - h_channel) ** 2).mean(dim=2)     # (N, n_patches)
        scores = (recon_diff + h_diff).mean(dim=1).cpu().numpy()
    return scores


# ===================== Evaluation =====================

def evaluate(labels, scores, contamination):
    auc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)
    threshold = np.percentile(scores, (1-contamination)*100)
    preds = (scores > threshold).astype(int)
    f1 = f1_score(labels, preds, zero_division=0)
    return auc, auprc, f1


# ===================== Main =====================

def run_experiment(dataset_name, dataset_gen_func, seeds, contaminations,
                   device, epochs=50, batch_size=128, lr=1e-3):
    results = {}
    dl_methods = {
        'AnomalyTrans': train_anomaly_transformer,
        'DCdetector': train_dcdetector,
    }

    for contamination in contaminations:
        print(f"\n{'='*60}")
        print(f"{dataset_name} -- Contamination: {contamination*100:.0f}%")
        print(f"{'='*60}")

        all_results = {m: {'AUC':[], 'AUPRC':[], 'F1':[], 'Time':[]} for m in dl_methods}

        for seed in seeds:
            print(f"  Seed {seed}...", end='', flush=True)
            t0 = time.time()

            trajs, labels, _ = dataset_gen_func(n_normal=5000, contamination=contamination, seed=seed)
            seqs = np.array([resample_trajectory(t, 32) for t in trajs])

            for method_name, train_func in dl_methods.items():
                mt0 = time.time()
                try:
                    scores = train_func(seqs, device, epochs=epochs,
                                        batch_size=batch_size, lr=lr)
                    auc, auprc, f1 = evaluate(labels, scores, contamination)
                    elapsed = time.time() - mt0
                    all_results[method_name]['AUC'].append(auc)
                    all_results[method_name]['AUPRC'].append(auprc)
                    all_results[method_name]['F1'].append(f1)
                    all_results[method_name]['Time'].append(elapsed)
                except Exception as e:
                    print(f"\n    ERROR {method_name}: {e}")
                    import traceback; traceback.print_exc()
                    all_results[method_name]['AUC'].append(0.5)
                    all_results[method_name]['AUPRC'].append(0.0)
                    all_results[method_name]['F1'].append(0.0)
                    all_results[method_name]['Time'].append(0.0)

            print(f" done ({time.time()-t0:.1f}s)")

        print(f"\n{'Method':<25} {'AUC':>12} {'AUPRC':>12} {'F1':>12} {'Time':>8}")
        print('-'*73)
        for mn in dl_methods:
            r = all_results[mn]
            print(f"  {mn:<23} {np.mean(r['AUC']):.4f}+/-{np.std(r['AUC']):.4f} "
                  f"{np.mean(r['AUPRC']):.4f}+/-{np.std(r['AUPRC']):.4f} "
                  f"{np.mean(r['F1']):.4f}+/-{np.std(r['F1']):.4f} {np.mean(r['Time']):.1f}s")

        key = f"{dataset_name}_{contamination}"
        results[key] = {}
        for mn in dl_methods:
            r = all_results[mn]
            results[key][mn] = {
                'AUC_mean': float(np.mean(r['AUC'])), 'AUC_std': float(np.std(r['AUC'])),
                'AUPRC_mean': float(np.mean(r['AUPRC'])), 'AUPRC_std': float(np.std(r['AUPRC'])),
                'F1_mean': float(np.mean(r['F1'])), 'F1_std': float(np.std(r['F1'])),
                'Time_mean': float(np.mean(r['Time'])),
            }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='all',
                        choices=['synthetic', 'grid', 'porto', 'geolife', 'all'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")

    seeds = [42, 123, 456, 789, 1024]
    contaminations = [0.05, 0.10, 0.15]

    datasets = {
        'synthetic': generate_synthetic,
        'grid': generate_grid_network,
        'porto': generate_porto_like,
        'geolife': generate_geolife_like,
    }

    all_results = {}
    if args.dataset == 'all':
        run_datasets = list(datasets.keys())
    else:
        run_datasets = [args.dataset]

    for ds_name in run_datasets:
        print(f"\n{'#'*60}")
        print(f"# Dataset: {ds_name}")
        print(f"{'#'*60}")
        ds_results = run_experiment(
            ds_name, datasets[ds_name], seeds, contaminations,
            device, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
        )
        all_results.update(ds_results)

    os.makedirs('results', exist_ok=True)
    out_path = 'results/dl2_baselines_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {out_path}")


if __name__ == '__main__':
    main()
