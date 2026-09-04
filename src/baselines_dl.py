"""
Deep learning baselines for trajectory anomaly detection.
Requires PyTorch.
"""

import copy
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


def _synchronize_if_cuda(device) -> None:
    """Synchronize the actual CUDA device before measuring an async phase."""
    device_obj = torch.device(device)
    if device_obj.type == "cuda":
        torch.cuda.synchronize(device_obj)


class TrajectoryLSTMAE(nn.Module):
    """LSTM Autoencoder for trajectory reconstruction."""
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32, num_layers=2, dropout=0.1):
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True, dropout=recurrent_dropout)
        self.enc_fc = nn.Linear(hidden_dim, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True, dropout=recurrent_dropout)
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

    def fit_score(self, features, trajectories=None, **kwargs):
        raise NotImplementedError


class TrajLSTMAEDetector(DLBaselineDetector):
    """LSTM-AE: reconstruction error as anomaly score."""
    def __init__(self, **kwargs):
        self.hidden_dim = int(kwargs.pop("hidden_dim", 64))
        self.latent_dim = int(kwargs.pop("latent_dim", 32))
        self.num_layers = int(kwargs.pop("num_layers", 2))
        self.dropout = float(kwargs.pop("dropout", 0.1))
        self.validation_fraction = float(kwargs.pop("validation_fraction", 0.1))
        self.patience = kwargs.pop("patience", 8)
        self.min_epochs = int(kwargs.pop("min_epochs", 10))
        self.min_delta = float(kwargs.pop("min_delta", 1e-5))
        super().__init__('Traj-LSTM-AE', **kwargs)

    def fit_score(
        self,
        features,
        trajectories=None,
        seed=42,
        score_population=True,
    ):
        if trajectories is None:
            raise ValueError("TrajLSTMAE needs trajectories")

        overall_started = time.perf_counter()
        seqs = np.array([t for t in trajectories])  # (N, seq_len, 2)
        scaler = StandardScaler()
        N, L, D = seqs.shape
        rng = np.random.RandomState(seed)
        indices = rng.permutation(N)
        n_val = max(1, int(round(N * self.validation_fraction)))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        scaler.fit(seqs[train_indices].reshape(len(train_indices) * L, D))
        seqs = scaler.transform(seqs.reshape(N * L, D)).reshape(N, L, D)

        X = torch.FloatTensor(seqs).to(self.device)
        train_tensor = X[torch.as_tensor(train_indices, device=self.device)]
        val_tensor = X[torch.as_tensor(val_indices, device=self.device)]
        dataset = TensorDataset(train_tensor)
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            generator=generator,
        )

        model = TrajectoryLSTMAE(
            input_dim=D,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        _synchronize_if_cuda(self.device)
        preparation_seconds = time.perf_counter() - overall_started
        training_started = time.perf_counter()
        history = []
        best_state = None
        best_val_loss = float("inf")
        best_epoch = 0
        stale_epochs = 0
        model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_items = 0
            for (batch,) in loader:
                recon = model(batch)
                loss = criterion(recon, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu()) * len(batch)
                epoch_items += len(batch)
            model.eval()
            with torch.no_grad():
                val_recon = model(val_tensor)
                val_loss = float(criterion(val_recon, val_tensor).detach().cpu())
            model.train()
            train_loss = epoch_loss / max(1, epoch_items)
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": val_loss})
            if val_loss < best_val_loss - self.min_delta:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            if (
                self.patience is not None
                and epoch + 1 >= self.min_epochs
                and stale_epochs >= int(self.patience)
            ):
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        _synchronize_if_cuda(self.device)
        training_seconds = time.perf_counter() - training_started
        scoring_started = time.perf_counter()
        model.eval()
        with torch.no_grad():
            score_tensor = X if score_population else val_tensor
            recon = model(score_tensor)
            errors = ((recon - score_tensor) ** 2).mean(dim=(1, 2)).cpu().numpy()
        _synchronize_if_cuda(self.device)
        scoring_seconds = time.perf_counter() - scoring_started
        self.timing_ = {
            "data_preparation_seconds": preparation_seconds,
            "training_seconds": training_seconds,
            "scoring_seconds": scoring_seconds,
            "fit_score_seconds": time.perf_counter() - overall_started,
            "epochs_completed": len(history),
            "max_epochs": self.epochs,
            "best_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
            "early_stopped": len(history) < self.epochs,
            "validation_fraction": self.validation_fraction,
            "training_samples": int(len(train_indices)),
            "validation_samples": int(len(val_indices)),
            "model_selection": "minimum unlabeled validation reconstruction MSE",
            "early_stopping": {
                "patience": self.patience,
                "min_epochs": self.min_epochs,
                "min_delta": self.min_delta,
                "restore_best_weights": True,
            },
            "initialization": "PyTorch default Linear/LSTM initialization after deterministic seed",
            "input_shape": [int(N), int(L), int(D)],
            "normalization": "StandardScaler fitted on the inner training trajectories, then applied to all trajectories",
            "configuration": {
                "hidden_dim": self.hidden_dim,
                "latent_dim": self.latent_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "batch_size": self.batch_size,
                "learning_rate": self.lr,
            },
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "training_history": history,
        }
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
        self.hidden_dim = int(kwargs.pop("hidden_dim", 64))
        self.latent_dim = int(kwargs.pop("latent_dim", 16))
        self.validation_fraction = float(kwargs.pop("validation_fraction", 0.1))
        self.patience = kwargs.pop("patience", 8)
        self.min_epochs = int(kwargs.pop("min_epochs", 10))
        self.min_delta = float(kwargs.pop("min_delta", 1e-5))
        super().__init__('USAD', **kwargs)

    def fit_score(
        self,
        features,
        trajectories=None,
        seed=42,
        score_population=True,
    ):
        overall_started = time.perf_counter()
        scaler = StandardScaler()
        features = np.asarray(features, dtype=np.float32)
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(features))
        n_val = max(1, int(round(len(features) * self.validation_fraction)))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        scaler.fit(features[train_indices])
        X_np = scaler.transform(features)
        X = torch.FloatTensor(X_np).to(self.device)

        train_tensor = X[torch.as_tensor(train_indices, device=self.device)]
        val_tensor = X[torch.as_tensor(val_indices, device=self.device)]
        dataset = TensorDataset(train_tensor)
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            generator=generator,
        )

        model = USAD_Model(
            input_dim=X.shape[1],
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        _synchronize_if_cuda(self.device)
        preparation_seconds = time.perf_counter() - overall_started
        training_started = time.perf_counter()
        history = []
        best_state = None
        best_val_loss = float("inf")
        best_epoch = 0
        stale_epochs = 0
        model.train()
        for epoch in range(self.epochs):
            n = epoch + 1
            epoch_loss = 0.0
            epoch_items = 0
            for (batch,) in loader:
                w1, w2, w3 = model(batch)
                loss1 = (1 / n) * ((batch - w1) ** 2).mean() + (1 - 1 / n) * ((batch - w3) ** 2).mean()
                loss2 = (1 / n) * ((batch - w2) ** 2).mean() - (1 - 1 / n) * ((batch - w3) ** 2).mean()
                loss = loss1 + loss2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu()) * len(batch)
                epoch_items += len(batch)
            model.eval()
            with torch.no_grad():
                val_w1, val_w2, _ = model(val_tensor)
                val_loss = float(
                    (0.5 * ((val_tensor - val_w1) ** 2).mean(dim=1)
                     + 0.5 * ((val_tensor - val_w2) ** 2).mean(dim=1)).mean().cpu()
                )
            model.train()
            train_loss = epoch_loss / max(1, epoch_items)
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": val_loss})
            if val_loss < best_val_loss - self.min_delta:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            if (
                self.patience is not None
                and epoch + 1 >= self.min_epochs
                and stale_epochs >= int(self.patience)
            ):
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        _synchronize_if_cuda(self.device)
        training_seconds = time.perf_counter() - training_started
        scoring_started = time.perf_counter()
        model.eval()
        with torch.no_grad():
            score_tensor = X if score_population else val_tensor
            w1, w2, w3 = model(score_tensor)
            alpha, beta = 0.5, 0.5
            scores = (alpha * ((score_tensor - w1) ** 2).mean(dim=1) +
                      beta * ((score_tensor - w2) ** 2).mean(dim=1)).cpu().numpy()
        _synchronize_if_cuda(self.device)
        scoring_seconds = time.perf_counter() - scoring_started
        self.timing_ = {
            "data_preparation_seconds": preparation_seconds,
            "training_seconds": training_seconds,
            "scoring_seconds": scoring_seconds,
            "fit_score_seconds": time.perf_counter() - overall_started,
            "epochs_completed": len(history),
            "max_epochs": self.epochs,
            "best_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
            "early_stopped": len(history) < self.epochs,
            "validation_fraction": self.validation_fraction,
            "training_samples": int(len(train_indices)),
            "validation_samples": int(len(val_indices)),
            "model_selection": "minimum unlabeled validation reconstruction score",
            "early_stopping": {
                "patience": self.patience,
                "min_epochs": self.min_epochs,
                "min_delta": self.min_delta,
                "restore_best_weights": True,
            },
            "initialization": "PyTorch default Linear initialization after deterministic seed",
            "input_shape": [int(len(features)), int(features.shape[1])],
            "normalization": "StandardScaler fitted on the inner training feature vectors, then applied to all vectors",
            "configuration": {
                "hidden_dim": self.hidden_dim,
                "latent_dim": self.latent_dim,
                "batch_size": self.batch_size,
                "learning_rate": self.lr,
            },
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "training_history": history,
        }
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
