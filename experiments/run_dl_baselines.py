"""
Deep Learning baseline experiments for EWGB-TAD paper.
Runs on GPU server. Self-contained: includes data generation and feature extraction.

Tests 5 DL baselines on 4 datasets:
- Synthetic (6-route), Grid-network, Porto Taxi, GeoLife

Usage:
    python experiments_dl.py --dataset synthetic --gpu 0
    python experiments_dl.py --dataset all --gpu 0
"""

import numpy as np
import time
import json
import os
import sys
import argparse
from collections import defaultdict
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ===================== DL Models =====================

class TrajectoryLSTMAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32, num_layers=2):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True, dropout=0.1)
        self.enc_fc = nn.Linear(hidden_dim, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True, dropout=0.1)
        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        enc_out, (h, c) = self.encoder(x)
        z = self.enc_fc(h[-1])
        dec_input = self.dec_fc(z).unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(dec_input)
        return self.output(dec_out)


class TransformerAE(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, seq_len=32):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                    dim_feedforward=128,
                                                    dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                    dim_feedforward=128,
                                                    dropout=0.1, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, input_dim)

    def forward(self, x):
        h = self.input_proj(x) + self.pos_enc[:, :x.size(1), :]
        memory = self.encoder(h)
        dec_out = self.decoder(h, memory)
        return self.output(dec_out)


class DeepSVDDNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hd), nn.ReLU(), nn.BatchNorm1d(hd)])
            prev_dim = hd
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class USAD_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim), nn.ReLU()
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        return w1, w2, w3


class TranAD_Model(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, seq_len=32):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                dim_feedforward=128, dropout=0.1,
                                                batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        dec_layer1 = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                 dim_feedforward=128, dropout=0.1,
                                                 batch_first=True)
        self.decoder1 = nn.TransformerDecoder(dec_layer1, num_layers=1)
        dec_layer2 = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                 dim_feedforward=128, dropout=0.1,
                                                 batch_first=True)
        self.decoder2 = nn.TransformerDecoder(dec_layer2, num_layers=1)
        self.output1 = nn.Linear(d_model, input_dim)
        self.output2 = nn.Linear(d_model, input_dim)

    def forward(self, x):
        h = self.input_proj(x) + self.pos_enc[:, :x.size(1), :]
        memory = self.encoder(h)
        o1 = self.output1(self.decoder1(h, memory))
        o2 = self.output2(self.decoder2(h, memory))
        return o1, o2


# ===================== DL Detectors =====================

def train_lstm_ae(seqs, device, epochs=50, batch_size=128, lr=1e-3):
    """LSTM-AE: reconstruction error as anomaly score."""
    scaler = StandardScaler()
    N, L, D = seqs.shape
    seqs_flat = scaler.fit_transform(seqs.reshape(N * L, D))
    seqs_scaled = seqs_flat.reshape(N, L, D)

    X = torch.FloatTensor(seqs_scaled).to(device)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)

    model = TrajectoryLSTMAE(input_dim=D, hidden_dim=64, latent_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for (batch,) in loader:
            recon = model(batch)
            loss = ((recon - batch) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        recon = model(X)
        errors = ((recon - X) ** 2).mean(dim=(1, 2)).cpu().numpy()
    return errors


def train_transformer_ae(seqs, device, epochs=50, batch_size=128, lr=1e-3):
    """Transformer-AE: reconstruction error as anomaly score."""
    scaler = StandardScaler()
    N, L, D = seqs.shape
    seqs_flat = scaler.fit_transform(seqs.reshape(N * L, D))
    seqs_scaled = seqs_flat.reshape(N, L, D)

    X = torch.FloatTensor(seqs_scaled).to(device)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)

    model = TransformerAE(input_dim=D, d_model=64, nhead=4, num_layers=2, seq_len=L).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for (batch,) in loader:
            recon = model(batch)
            loss = ((recon - batch) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        recon = model(X)
        errors = ((recon - X) ** 2).mean(dim=(1, 2)).cpu().numpy()
    return errors


def train_deep_svdd(features, device, epochs=50, batch_size=128, lr=1e-3):
    """Deep SVDD on extracted features."""
    scaler = StandardScaler()
    X_np = scaler.fit_transform(features)
    X = torch.FloatTensor(X_np).to(device)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)

    model = DeepSVDDNet(input_dim=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Phase 1: pretrain with AE
    ae_decoder = nn.Sequential(
        nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, X.shape[1])
    ).to(device)
    ae_opt = torch.optim.Adam(list(model.parameters()) + list(ae_decoder.parameters()), lr=lr)
    model.train()
    for epoch in range(min(20, epochs)):
        for (batch,) in loader:
            z = model(batch)
            recon = ae_decoder(z)
            loss = ((recon - batch) ** 2).mean()
            ae_opt.zero_grad()
            loss.backward()
            ae_opt.step()

    model.eval()
    with torch.no_grad():
        center = model(X).mean(dim=0)

    model.train()
    for epoch in range(epochs):
        for (batch,) in loader:
            z = model(batch)
            loss = ((z - center) ** 2).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        z = model(X)
        scores = ((z - center) ** 2).sum(dim=1).cpu().numpy()
    return scores


def train_usad(features, device, epochs=50, batch_size=128, lr=1e-3):
    """USAD on extracted features."""
    scaler = StandardScaler()
    X_np = scaler.fit_transform(features)
    X = torch.FloatTensor(X_np).to(device)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)

    model = USAD_Model(input_dim=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        n = epoch + 1
        for (batch,) in loader:
            w1, w2, w3 = model(batch)
            loss1 = (1/n) * ((batch - w1)**2).mean() + (1 - 1/n) * ((batch - w3)**2).mean()
            loss2 = (1/n) * ((batch - w2)**2).mean() - (1 - 1/n) * ((batch - w3)**2).mean()
            loss = loss1 + loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        w1, w2, w3 = model(X)
        scores = (0.5 * ((X - w1)**2).mean(dim=1) +
                  0.5 * ((X - w2)**2).mean(dim=1)).cpu().numpy()
    return scores


def train_tranad(seqs, device, epochs=50, batch_size=128, lr=1e-3):
    """TranAD on trajectory sequences."""
    scaler = StandardScaler()
    N, L, D = seqs.shape
    seqs_flat = scaler.fit_transform(seqs.reshape(N * L, D))
    seqs_scaled = seqs_flat.reshape(N, L, D)

    X = torch.FloatTensor(seqs_scaled).to(device)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)

    model = TranAD_Model(input_dim=D, d_model=64, nhead=4, seq_len=L).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        n = epoch + 1
        for (batch,) in loader:
            o1, o2 = model(batch)
            loss = (1/n) * ((batch - o1)**2).mean() + (1 - 1/n) * ((batch - o2)**2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        o1, o2 = model(X)
        scores = (0.5 * ((X - o1)**2).mean(dim=(1, 2)) +
                  0.5 * ((X - o2)**2).mean(dim=(1, 2))).cpu().numpy()
    return scores


# ===================== Feature Extraction =====================

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
        mean_accel = np.mean(accels)
        std_accel = np.std(accels)
    else:
        mean_accel = std_accel = 0.0
    speed_threshold = mean_speed * 0.1 + 1e-8
    stop_ratio = np.sum(speeds < speed_threshold) / len(speeds)
    if len(diffs) > 1:
        angles = np.arctan2(diffs[:, 1], diffs[:, 0])
        turn_angles = np.abs(np.diff(angles))
        turn_angles = np.minimum(turn_angles, 2 * np.pi - turn_angles)
        mean_turn = np.mean(turn_angles)
        max_turn = np.max(turn_angles)
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
    all_feat = np.hstack([spatial, kinematic, entropy])
    return all_feat


# ===================== Data Generators =====================

def generate_synthetic(n_normal=5000, contamination=0.1, seq_len=32, n_routes=6, seed=42):
    """Generate synthetic trajectories with 4 anomaly types."""
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
        mid_offset = rng.normal(0, 0.04, (1, 2))
        path += 4 * t * (1 - t) * mid_offset
        path += rng.normal(0, 0.02, path.shape)
        return path

    trajectories = []
    labels = []
    anom_types = []

    for i in range(n_normal):
        trajectories.append(make_normal(i % n_routes))
        labels.append(0)
        anom_types.append(-1)

    for atype in range(4):
        for _ in range(n_per_type):
            ridx = rng.randint(0, n_routes)
            traj = make_normal(ridx)
            od_dist = np.linalg.norm(route_defs[ridx][1] - route_defs[ridx][0])
            direction = route_defs[ridx][1] - route_defs[ridx][0]
            perp = np.array([-direction[1], direction[0]])
            perp = perp / (np.linalg.norm(perp) + 1e-8)

            if atype == 0:  # DETOUR
                s, e = int(0.3 * seq_len), int(0.7 * seq_len)
                mag = rng.uniform(0.1, 0.25) * od_dist
                sign = rng.choice([-1, 1])
                for k in range(s, e):
                    frac = (k - s) / (e - s + 1e-8)
                    traj[k] += sign * perp * mag * np.sin(np.pi * frac)
            elif atype == 1:  # LOOP
                ci = int(rng.uniform(0.4, 0.6) * seq_len)
                r = rng.uniform(0.05, 0.15) * od_dist
                nl = min(8, seq_len // 4)
                si = max(0, ci - nl // 2)
                for k in range(nl):
                    if si + k < seq_len:
                        a = 2 * np.pi * k / nl
                        traj[si + k] += np.array([r * np.cos(a), r * np.sin(a)])
            elif atype == 2:  # SPEED
                sp = int(rng.uniform(0.2, 0.5) * seq_len)
                ns = min(6, seq_len // 5)
                cp = traj[sp].copy()
                for k in range(ns):
                    if sp + k < seq_len:
                        traj[sp + k] = cp + rng.normal(0, 0.005, 2)
            elif atype == 3:  # ROUTE DEVIATION
                ds = int(rng.uniform(0.5, 0.7) * seq_len)
                ad = rng.uniform(30, 60) * np.pi / 180 * rng.choice([-1, 1])
                for k in range(ds, seq_len):
                    if k > 0:
                        diff = traj[k] - traj[k-1]
                        ca, sa = np.cos(ad), np.sin(ad)
                        traj[k] = traj[k-1] + np.array([ca*diff[0]-sa*diff[1], sa*diff[0]+ca*diff[1]])
                    ad *= 0.95

            trajectories.append(traj)
            labels.append(1)
            anom_types.append(atype)

    perm = rng.permutation(len(trajectories))
    trajectories = [trajectories[i] for i in perm]
    labels = np.array(labels)[perm]
    anom_types = np.array(anom_types)[perm]
    return trajectories, labels, anom_types


def generate_grid_network(n_normal=5000, contamination=0.1, seq_len=32, seed=42):
    """Generate grid-network trajectories."""
    rng = np.random.RandomState(seed)
    n_anomaly = int(n_normal * contamination / (1 - contamination))
    n_per_type = max(1, n_anomaly // 4)

    grid_size = 6
    intersections = [(i, j) for i in range(grid_size) for j in range(grid_size)]

    def make_grid_path(start_node, end_node):
        si, sj = start_node
        ei, ej = end_node
        path_nodes = [(si, sj)]
        ci, cj = si, sj
        while ci != ei or cj != ej:
            if rng.random() < 0.5 and ci != ei:
                ci += 1 if ei > ci else -1
            elif cj != ej:
                cj += 1 if ej > cj else -1
            elif ci != ei:
                ci += 1 if ei > ci else -1
            path_nodes.append((ci, cj))
        points = np.array(path_nodes, dtype=float) / (grid_size - 1)
        if len(points) < 2:
            points = np.array([points[0], points[0] + [0.01, 0.01]])
        t_orig = np.linspace(0, 1, len(points))
        t_new = np.linspace(0, 1, seq_len)
        traj = np.column_stack([np.interp(t_new, t_orig, points[:, 0]),
                                np.interp(t_new, t_orig, points[:, 1])])
        traj += rng.normal(0, 0.01, traj.shape)
        return traj

    # Define routes
    routes = [
        ((0,0),(5,5)), ((5,0),(0,5)), ((0,2),(5,2)),
        ((2,0),(2,5)), ((0,0),(5,0)), ((0,5),(5,5)),
        ((3,0),(3,5)), ((0,3),(5,3)),
    ]

    trajectories, labels, anom_types = [], [], []
    for i in range(n_normal):
        s, e = routes[i % len(routes)]
        trajectories.append(make_grid_path(s, e))
        labels.append(0)
        anom_types.append(-1)

    for atype in range(4):
        for _ in range(n_per_type):
            ridx = rng.randint(0, len(routes))
            s, e = routes[ridx]
            traj = make_grid_path(s, e)
            od_dist = np.linalg.norm(traj[-1] - traj[0])
            direction = traj[-1] - traj[0]
            perp = np.array([-direction[1], direction[0]])
            perp = perp / (np.linalg.norm(perp) + 1e-8)

            if atype == 0:
                s_i, e_i = int(0.3 * seq_len), int(0.7 * seq_len)
                mag = rng.uniform(0.1, 0.25) * od_dist
                sign = rng.choice([-1, 1])
                for k in range(s_i, e_i):
                    frac = (k - s_i) / (e_i - s_i + 1e-8)
                    traj[k] += sign * perp * mag * np.sin(np.pi * frac)
            elif atype == 1:
                ci = seq_len // 2
                r = rng.uniform(0.05, 0.15) * od_dist
                nl = min(8, seq_len // 4)
                for k in range(nl):
                    idx = ci - nl//2 + k
                    if 0 <= idx < seq_len:
                        a = 2 * np.pi * k / nl
                        traj[idx] += np.array([r * np.cos(a), r * np.sin(a)])
            elif atype == 2:
                sp = int(rng.uniform(0.2, 0.5) * seq_len)
                ns = min(6, seq_len // 5)
                cp = traj[sp].copy()
                for k in range(ns):
                    if sp + k < seq_len:
                        traj[sp + k] = cp + rng.normal(0, 0.005, 2)
            elif atype == 3:
                ds = int(rng.uniform(0.5, 0.7) * seq_len)
                ad = rng.uniform(30, 60) * np.pi / 180 * rng.choice([-1, 1])
                for k in range(ds, seq_len):
                    if k > 0:
                        diff = traj[k] - traj[k-1]
                        ca, sa = np.cos(ad), np.sin(ad)
                        traj[k] = traj[k-1] + np.array([ca*diff[0]-sa*diff[1], sa*diff[0]+ca*diff[1]])
                    ad *= 0.95

            trajectories.append(traj)
            labels.append(1)
            anom_types.append(atype)

    perm = rng.permutation(len(trajectories))
    trajectories = [trajectories[i] for i in perm]
    labels = np.array(labels)[perm]
    anom_types = np.array(anom_types)[perm]
    return trajectories, labels, anom_types


def generate_porto_like(n_normal=5000, contamination=0.1, seq_len=32, n_clusters=20, seed=42):
    """Generate Porto-like trajectories from random OD pairs in Porto's bounding box."""
    rng = np.random.RandomState(seed)
    n_anomaly = int(n_normal * contamination / (1 - contamination))
    n_per_type = max(1, n_anomaly // 4)

    # Porto bounding box
    lon_range = (-8.70, -8.54)
    lat_range = (41.10, 41.20)

    from sklearn.cluster import KMeans
    # Generate random OD pairs
    n_od = n_normal * 3
    starts = np.column_stack([rng.uniform(*lon_range, n_od), rng.uniform(*lat_range, n_od)])
    ends = np.column_stack([rng.uniform(*lon_range, n_od), rng.uniform(*lat_range, n_od)])
    dists = np.linalg.norm(ends - starts, axis=1)
    valid = (dists > 0.005) & (dists < 0.15)
    starts, ends = starts[valid], ends[valid]
    od_flat = np.hstack([starts, ends])

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = km.fit_predict(od_flat[:min(len(od_flat), n_normal * 2)])

    clusters = defaultdict(list)
    for i, cl in enumerate(cluster_labels):
        clusters[cl].append(i)

    # Generate route templates per cluster
    templates = {}
    for cl_id, indices in clusters.items():
        center = km.cluster_centers_[cl_id]
        cs, ce = center[:2], center[2:]
        direction = ce - cs
        perp = np.array([-direction[1], direction[0]])
        perp = perp / (np.linalg.norm(perp) + 1e-8)
        t = np.linspace(0, 1, seq_len)
        template = np.zeros((seq_len, 2))
        for _ in range(rng.randint(1, 4)):
            bt = rng.uniform(0.15, 0.85)
            bs = rng.uniform(0.08, 0.2)
            bm = rng.normal(0, 0.1) * np.linalg.norm(direction)
            weights = np.exp(-0.5 * ((t - bt) / bs) ** 2)
            template += np.outer(weights, perp * bm)
        templates[cl_id] = template

    trajectories, labels, anom_types = [], [], []
    for i in range(n_normal):
        cl_id = i % n_clusters
        if clusters[cl_id]:
            idx = clusters[cl_id][i % len(clusters[cl_id])]
            s, e = starts[idx], ends[idx]
        else:
            s = rng.uniform(lon_range[0], lon_range[1], 2)
            s[1] = rng.uniform(*lat_range)
            e = s + rng.uniform(-0.05, 0.05, 2)

        t_arr = np.linspace(0, 1, seq_len).reshape(-1, 1)
        path = s + t_arr * (e - s)
        if cl_id in templates:
            path += templates[cl_id] * rng.uniform(0.8, 1.2)
        path += rng.normal(0, 0.0001, path.shape)
        trajectories.append(path)
        labels.append(0)
        anom_types.append(-1)

    # Inject anomalies (same as Porto)
    for atype in range(4):
        for _ in range(n_per_type):
            idx = rng.randint(0, n_normal)
            traj = trajectories[idx].copy()
            od_dist = np.linalg.norm(traj[-1] - traj[0])
            direction = traj[-1] - traj[0]
            perp = np.array([-direction[1], direction[0]])
            perp = perp / (np.linalg.norm(perp) + 1e-8)

            if atype == 0:
                s_i, e_i = int(0.3 * seq_len), int(0.7 * seq_len)
                mag = rng.uniform(0.15, 0.35) * od_dist
                sign = rng.choice([-1, 1])
                for k in range(s_i, e_i):
                    frac = (k - s_i) / (e_i - s_i + 1e-8)
                    traj[k] += sign * perp * mag * np.sin(np.pi * frac)
            elif atype == 1:
                ci = seq_len // 2
                r = rng.uniform(0.08, 0.2) * od_dist
                nl = min(10, seq_len // 3)
                si = max(0, ci - nl // 2)
                for k in range(nl):
                    if si + k < seq_len:
                        a = 2 * np.pi * k / nl
                        traj[si + k] += np.array([r * np.cos(a), r * np.sin(a)])
            elif atype == 2:
                sp = int(rng.uniform(0.2, 0.5) * seq_len)
                ns = min(8, seq_len // 4)
                cp = traj[sp].copy()
                for k in range(ns):
                    if sp + k < seq_len:
                        traj[sp + k] = cp + rng.normal(0, od_dist * 0.003, 2)
            elif atype == 3:
                ds = int(rng.uniform(0.4, 0.6) * seq_len)
                ad = rng.uniform(40, 80) * np.pi / 180 * rng.choice([-1, 1])
                for k in range(ds, seq_len):
                    if k > 0:
                        diff = traj[k] - traj[k-1]
                        if np.linalg.norm(diff) < 1e-10:
                            diff = (traj[-1] - traj[0]) / seq_len
                        ca, sa = np.cos(ad), np.sin(ad)
                        traj[k] = traj[k-1] + np.array([ca*diff[0]-sa*diff[1], sa*diff[0]+ca*diff[1]]) * 1.3
                    ad *= 0.92

            trajectories.append(traj)
            labels.append(1)
            anom_types.append(atype)

    perm = rng.permutation(len(trajectories))
    trajectories = [trajectories[i] for i in perm]
    labels = np.array(labels)[perm]
    anom_types = np.array(anom_types)[perm]
    return trajectories, labels, anom_types


def generate_geolife_like(n_normal=5000, contamination=0.1, seq_len=32, n_clusters=15, seed=42):
    """Generate GeoLife-like trajectories in Beijing bbox."""
    rng = np.random.RandomState(seed)
    n_anomaly = int(n_normal * contamination / (1 - contamination))
    n_per_type = max(1, n_anomaly // 4)

    lon_range = (116.1, 116.65)
    lat_range = (39.75, 40.15)

    from sklearn.cluster import KMeans
    n_od = n_normal * 3
    starts = np.column_stack([rng.uniform(*lon_range, n_od), rng.uniform(*lat_range, n_od)])
    ends = np.column_stack([rng.uniform(*lon_range, n_od), rng.uniform(*lat_range, n_od)])
    dists = np.linalg.norm(ends - starts, axis=1)
    valid = (dists > 0.005) & (dists < 0.3)
    starts, ends = starts[valid], ends[valid]
    od_flat = np.hstack([starts, ends])

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = km.fit_predict(od_flat[:min(len(od_flat), n_normal * 2)])

    clusters = defaultdict(list)
    for i, cl in enumerate(cluster_labels):
        clusters[cl].append(i)

    templates = {}
    for cl_id, indices in clusters.items():
        center = km.cluster_centers_[cl_id]
        cs, ce = center[:2], center[2:]
        direction = ce - cs
        perp = np.array([-direction[1], direction[0]])
        perp = perp / (np.linalg.norm(perp) + 1e-8)
        t = np.linspace(0, 1, seq_len)
        template = np.zeros((seq_len, 2))
        for _ in range(rng.randint(1, 4)):
            bt = rng.uniform(0.15, 0.85)
            bs = rng.uniform(0.08, 0.2)
            bm = rng.normal(0, 0.1) * np.linalg.norm(direction)
            weights = np.exp(-0.5 * ((t - bt) / bs) ** 2)
            template += np.outer(weights, perp * bm)
        templates[cl_id] = template

    trajectories, labels, anom_types = [], [], []
    for i in range(n_normal):
        cl_id = i % n_clusters
        if clusters[cl_id]:
            idx = clusters[cl_id][i % len(clusters[cl_id])]
            s, e = starts[idx], ends[idx]
        else:
            s = np.array([rng.uniform(*lon_range), rng.uniform(*lat_range)])
            e = s + rng.uniform(-0.05, 0.05, 2)
        t_arr = np.linspace(0, 1, seq_len).reshape(-1, 1)
        path = s + t_arr * (e - s)
        if cl_id in templates:
            path += templates[cl_id] * rng.uniform(0.8, 1.2)
        path += rng.normal(0, 0.0002, path.shape)
        trajectories.append(path)
        labels.append(0)
        anom_types.append(-1)

    for atype in range(4):
        for _ in range(n_per_type):
            idx = rng.randint(0, n_normal)
            traj = trajectories[idx].copy()
            od_dist = np.linalg.norm(traj[-1] - traj[0])
            direction = traj[-1] - traj[0]
            perp = np.array([-direction[1], direction[0]])
            perp = perp / (np.linalg.norm(perp) + 1e-8)

            if atype == 0:
                s_i, e_i = int(0.3 * seq_len), int(0.7 * seq_len)
                mag = rng.uniform(0.15, 0.35) * od_dist
                sign = rng.choice([-1, 1])
                for k in range(s_i, e_i):
                    frac = (k - s_i) / (e_i - s_i + 1e-8)
                    traj[k] += sign * perp * mag * np.sin(np.pi * frac)
            elif atype == 1:
                ci = seq_len // 2
                r = rng.uniform(0.08, 0.2) * od_dist
                nl = min(10, seq_len // 3)
                si = max(0, ci - nl // 2)
                for k in range(nl):
                    if si + k < seq_len:
                        a = 2 * np.pi * k / nl
                        traj[si + k] += np.array([r * np.cos(a), r * np.sin(a)])
            elif atype == 2:
                sp = int(rng.uniform(0.2, 0.5) * seq_len)
                ns = min(8, seq_len // 4)
                cp = traj[sp].copy()
                for k in range(ns):
                    if sp + k < seq_len:
                        traj[sp + k] = cp + rng.normal(0, od_dist * 0.003, 2)
            elif atype == 3:
                ds = int(rng.uniform(0.4, 0.6) * seq_len)
                ad = rng.uniform(40, 80) * np.pi / 180 * rng.choice([-1, 1])
                for k in range(ds, seq_len):
                    if k > 0:
                        diff = traj[k] - traj[k-1]
                        if np.linalg.norm(diff) < 1e-10:
                            diff = (traj[-1] - traj[0]) / seq_len
                        ca, sa = np.cos(ad), np.sin(ad)
                        traj[k] = traj[k-1] + np.array([ca*diff[0]-sa*diff[1], sa*diff[0]+ca*diff[1]]) * 1.3
                    ad *= 0.92

            trajectories.append(traj)
            labels.append(1)
            anom_types.append(atype)

    perm = rng.permutation(len(trajectories))
    trajectories = [trajectories[i] for i in perm]
    labels = np.array(labels)[perm]
    anom_types = np.array(anom_types)[perm]
    return trajectories, labels, anom_types


# ===================== Evaluation =====================

def evaluate(labels, scores, contamination):
    auc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)
    threshold = np.percentile(scores, (1 - contamination) * 100)
    preds = (scores > threshold).astype(int)
    f1 = f1_score(labels, preds, zero_division=0)
    return auc, auprc, f1


def resample_trajectory(traj, target_len=32):
    """Resample trajectory to fixed length."""
    n = len(traj)
    if n == target_len:
        return traj
    t_orig = np.linspace(0, 1, n)
    t_new = np.linspace(0, 1, target_len)
    return np.column_stack([np.interp(t_new, t_orig, traj[:, d]) for d in range(traj.shape[1])])


# ===================== Main Experiment =====================

def run_dl_experiment(dataset_name, dataset_gen_func, seeds, contaminations,
                      device, epochs=50, batch_size=128, lr=1e-3):
    """Run all DL baselines on one dataset."""
    results = {}

    dl_methods = {
        'Traj-LSTM-AE': ('seq', train_lstm_ae),
        'Transformer-AE': ('seq', train_transformer_ae),
        'DeepSVDD': ('feat', train_deep_svdd),
        'USAD': ('feat', train_usad),
        'TranAD': ('seq', train_tranad),
    }

    for contamination in contaminations:
        print(f"\n{'='*60}")
        print(f"{dataset_name} -- Contamination: {contamination*100:.0f}%")
        print(f"{'='*60}")

        all_results = {m: {'AUC': [], 'AUPRC': [], 'F1': [], 'Time': []} for m in dl_methods}

        for seed in seeds:
            print(f"  Seed {seed}...", end='', flush=True)
            t0 = time.time()

            trajs, labels, anom_types = dataset_gen_func(
                n_normal=5000, contamination=contamination, seed=seed
            )
            seqs = np.array([resample_trajectory(t, 32) for t in trajs])  # (N, 32, 2)
            features = extract_all_features(trajs)  # (N, 21)

            for method_name, (input_type, train_func) in dl_methods.items():
                mt0 = time.time()
                try:
                    if input_type == 'seq':
                        scores = train_func(seqs, device, epochs=epochs,
                                           batch_size=batch_size, lr=lr)
                    else:
                        scores = train_func(features, device, epochs=epochs,
                                           batch_size=batch_size, lr=lr)

                    auc, auprc, f1 = evaluate(labels, scores, contamination)
                    elapsed = time.time() - mt0
                    all_results[method_name]['AUC'].append(auc)
                    all_results[method_name]['AUPRC'].append(auprc)
                    all_results[method_name]['F1'].append(f1)
                    all_results[method_name]['Time'].append(elapsed)
                except Exception as e:
                    print(f"\n    ERROR {method_name}: {e}")
                    all_results[method_name]['AUC'].append(0.5)
                    all_results[method_name]['AUPRC'].append(0.0)
                    all_results[method_name]['F1'].append(0.0)
                    all_results[method_name]['Time'].append(0.0)

            print(f" done ({time.time()-t0:.1f}s)")

        # Print results
        print(f"\n{'Method':<30} {'AUC':>12} {'AUPRC':>12} {'F1':>12} {'Time':>8}")
        print('-' * 78)
        for method_name in dl_methods:
            r = all_results[method_name]
            auc_m, auc_s = np.mean(r['AUC']), np.std(r['AUC'])
            auprc_m, auprc_s = np.mean(r['AUPRC']), np.std(r['AUPRC'])
            f1_m, f1_s = np.mean(r['F1']), np.std(r['F1'])
            t_m = np.mean(r['Time'])
            print(f"  {method_name:<28} {auc_m:.4f}+/-{auc_s:.4f} "
                  f"{auprc_m:.4f}+/-{auprc_s:.4f} {f1_m:.4f}+/-{f1_s:.4f} {t_m:.1f}s")

        key = f"{dataset_name}_{contamination}"
        results[key] = {}
        for method_name in dl_methods:
            r = all_results[method_name]
            results[key][method_name] = {
                'AUC_mean': float(np.mean(r['AUC'])),
                'AUC_std': float(np.std(r['AUC'])),
                'AUPRC_mean': float(np.mean(r['AUPRC'])),
                'AUPRC_std': float(np.std(r['AUPRC'])),
                'F1_mean': float(np.mean(r['F1'])),
                'F1_std': float(np.std(r['F1'])),
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
        ds_results = run_dl_experiment(
            ds_name, datasets[ds_name], seeds, contaminations,
            device, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
        )
        all_results.update(ds_results)

    # Save results
    os.makedirs('results', exist_ok=True)
    out_path = f'results/dl_baselines_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {out_path}")


if __name__ == '__main__':
    main()
