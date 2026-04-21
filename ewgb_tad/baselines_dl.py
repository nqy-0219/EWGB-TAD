"""
Deep learning baselines for trajectory anomaly detection.
Requires PyTorch.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


class TrajectoryLSTMAE(nn.Module):
    """LSTM Autoencoder for trajectory reconstruction."""
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
        # x: (batch, seq_len, input_dim)
        enc_out, (h, c) = self.encoder(x)
        z = self.enc_fc(h[-1])  # (batch, latent_dim)
        dec_input = self.dec_fc(z).unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(dec_input)
        recon = self.output(dec_out)
        return recon


class TransformerAE(nn.Module):
    """Transformer Autoencoder for trajectory anomaly detection."""
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
    """Deep SVDD network for trajectory features."""
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
    """USAD: UnSupervised Anomaly Detection with two autoencoders."""
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
    """TranAD: Transformer-based adversarial anomaly detection."""
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


# ===================== Detector Wrappers =====================

class DLBaselineDetector:
    """Base class for DL anomaly detectors."""
    def __init__(self, name, epochs=50, batch_size=128, lr=1e-3, device='cuda'):
        self.name = name
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device if torch.cuda.is_available() else 'cpu'

    def fit_score(self, features, trajectories=None):
        raise NotImplementedError


class TrajLSTMAEDetector(DLBaselineDetector):
    """LSTM-AE: reconstruction error as anomaly score."""
    def __init__(self, **kwargs):
        super().__init__('Traj-LSTM-AE', **kwargs)

    def fit_score(self, features, trajectories=None):
        if trajectories is None:
            raise ValueError("TrajLSTMAE needs trajectories")

        seqs = np.array([t for t in trajectories])  # (N, seq_len, 2)
        scaler = StandardScaler()
        N, L, D = seqs.shape
        seqs_flat = seqs.reshape(N * L, D)
        seqs_flat = scaler.fit_transform(seqs_flat)
        seqs = seqs_flat.reshape(N, L, D)

        X = torch.FloatTensor(seqs).to(self.device)
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model = TrajectoryLSTMAE(input_dim=D, hidden_dim=64, latent_dim=32).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss(reduction='none')

        model.train()
        for epoch in range(self.epochs):
            for (batch,) in loader:
                recon = model(batch)
                loss = criterion(recon, batch).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            recon = model(X)
            errors = ((recon - X) ** 2).mean(dim=(1, 2)).cpu().numpy()
        return errors


class TransformerAEDetector(DLBaselineDetector):
    """Transformer-AE: reconstruction error as anomaly score."""
    def __init__(self, **kwargs):
        super().__init__('Transformer-AE', **kwargs)

    def fit_score(self, features, trajectories=None):
        if trajectories is None:
            raise ValueError("TransformerAE needs trajectories")

        seqs = np.array([t for t in trajectories])
        scaler = StandardScaler()
        N, L, D = seqs.shape
        seqs_flat = seqs.reshape(N * L, D)
        seqs_flat = scaler.fit_transform(seqs_flat)
        seqs = seqs_flat.reshape(N, L, D)

        X = torch.FloatTensor(seqs).to(self.device)
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model = TransformerAE(input_dim=D, d_model=64, nhead=4, num_layers=2, seq_len=L).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        model.train()
        for epoch in range(self.epochs):
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


class DeepSVDDDetector(DLBaselineDetector):
    """Deep SVDD on extracted features."""
    def __init__(self, **kwargs):
        super().__init__('DeepSVDD', **kwargs)

    def fit_score(self, features, trajectories=None):
        scaler = StandardScaler()
        X_np = scaler.fit_transform(features)
        X = torch.FloatTensor(X_np).to(self.device)

        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model = DeepSVDDNet(input_dim=X.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)

        # Phase 1: pretrain with AE
        ae_decoder = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, X.shape[1])
        ).to(self.device)
        ae_opt = torch.optim.Adam(list(model.parameters()) + list(ae_decoder.parameters()), lr=self.lr)
        model.train()
        for epoch in range(min(20, self.epochs)):
            for (batch,) in loader:
                z = model(batch)
                recon = ae_decoder(z)
                loss = ((recon - batch) ** 2).mean()
                ae_opt.zero_grad()
                loss.backward()
                ae_opt.step()

        # Compute center
        model.eval()
        with torch.no_grad():
            all_z = model(X)
            center = all_z.mean(dim=0)

        # Phase 2: SVDD training
        model.train()
        for epoch in range(self.epochs):
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


class USADDetector(DLBaselineDetector):
    """USAD on extracted features."""
    def __init__(self, **kwargs):
        super().__init__('USAD', **kwargs)

    def fit_score(self, features, trajectories=None):
        scaler = StandardScaler()
        X_np = scaler.fit_transform(features)
        X = torch.FloatTensor(X_np).to(self.device)

        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model = USAD_Model(input_dim=X.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        model.train()
        for epoch in range(self.epochs):
            n = epoch + 1
            for (batch,) in loader:
                w1, w2, w3 = model(batch)
                loss1 = (1 / n) * ((batch - w1) ** 2).mean() + (1 - 1 / n) * ((batch - w3) ** 2).mean()
                loss2 = (1 / n) * ((batch - w2) ** 2).mean() - (1 - 1 / n) * ((batch - w3) ** 2).mean()
                loss = loss1 + loss2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            w1, w2, w3 = model(X)
            alpha, beta = 0.5, 0.5
            scores = (alpha * ((X - w1) ** 2).mean(dim=1) +
                      beta * ((X - w2) ** 2).mean(dim=1)).cpu().numpy()
        return scores


class TranADDetector(DLBaselineDetector):
    """TranAD on trajectory sequences."""
    def __init__(self, **kwargs):
        super().__init__('TranAD', **kwargs)

    def fit_score(self, features, trajectories=None):
        if trajectories is None:
            raise ValueError("TranAD needs trajectories")

        seqs = np.array([t for t in trajectories])
        scaler = StandardScaler()
        N, L, D = seqs.shape
        seqs_flat = seqs.reshape(N * L, D)
        seqs_flat = scaler.fit_transform(seqs_flat)
        seqs = seqs_flat.reshape(N, L, D)

        X = torch.FloatTensor(seqs).to(self.device)
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model = TranAD_Model(input_dim=D, d_model=64, nhead=4, seq_len=L).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        model.train()
        for epoch in range(self.epochs):
            n = epoch + 1
            for (batch,) in loader:
                o1, o2 = model(batch)
                loss = (1 / n) * ((batch - o1) ** 2).mean() + (1 - 1 / n) * ((batch - o2) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            o1, o2 = model(X)
            scores = (0.5 * ((X - o1) ** 2).mean(dim=(1, 2)) +
                      0.5 * ((X - o2) ** 2).mean(dim=(1, 2))).cpu().numpy()
        return scores


def get_dl_baselines(epochs=50, batch_size=128, lr=1e-3, device='cuda'):
    return {
        'Traj-LSTM-AE': TrajLSTMAEDetector(epochs=epochs, batch_size=batch_size, lr=lr, device=device),
        'Transformer-AE': TransformerAEDetector(epochs=epochs, batch_size=batch_size, lr=lr, device=device),
        'DeepSVDD': DeepSVDDDetector(epochs=epochs, batch_size=batch_size, lr=lr, device=device),
        'USAD': USADDetector(epochs=epochs, batch_size=batch_size, lr=lr, device=device),
        'TranAD': TranADDetector(epochs=epochs, batch_size=batch_size, lr=lr, device=device),
    }
