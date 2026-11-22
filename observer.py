"""Causal robot-only observation and model-based system identification."""

import numpy as np
from scipy.optimize import lsq_linear
from scipy.signal import savgol_filter
import torch
from pathlib import Path
from terrain_capacity101 import CapacitySupportOperator

ACTIONS = ("short", "release", "hold")
END = {"short": 5.20, "release": 7.13, "hold": 8.65}
OPERATION = {"short": 10.55, "release": 12.10, "hold": 13.75}


def source_obs(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in ("time", "feet", "foot_fz")}


class Features:
    def __init__(self, encoder_checkpoint):
        self.checkpoint = Path(encoder_checkpoint)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = None

    def load_encoder(self):
        if self.encoder is not None:
            return
        self.encoder = CapacitySupportOperator().to(self.device)
        checkpoint = self.checkpoint
        self.encoder.load_state_dict(
            torch.load(checkpoint, map_location=self.device, weights_only=False)[
                "model"
            ]
        )
        self.encoder.eval()

    def __call__(self, obs, end, kind="both"):
        if kind not in ("gru", "id", "both"):
            raise ValueError(kind)
        valid = np.asarray(obs["time"]) <= end + 1e-6
        t = np.asarray(obs["time"])[valid]
        if len(t) < 300 or abs(t[-1] - end) > 0.008:
            raise ValueError(f"Incomplete prefix for {end:.2f} s ({len(t)} samples)")
        feet = np.asarray(obs["feet"])[valid]
        fz = np.asarray(obs["foot_fz"])[valid, 0]
        rel = feet[:, 0] - np.median(feet[:, 1:], axis=1)
        rel -= np.median(rel[(t >= 2.20) & (t < 2.35)], axis=0)
        if kind == "id":
            return None, identify(t, -rel[:, 2], fz)
        self.load_encoder()
        vel = np.vstack((np.zeros((1, 3)), np.diff(rel, axis=0) / np.diff(t)[:, None]))
        ix = np.unique(np.r_[np.arange(0, len(t), 5), len(t) - 1])
        x = np.zeros((len(ix), 1, 10), "f")
        x[:, 0, 2] = fz[ix]
        x[:, 0, 3:6] = rel[ix]
        x[:, 0, 6:9] = vel[ix]
        x[:, 0, 9] = fz[ix] > 5

        def tensor(v):
            return torch.as_tensor(v, device=self.device, dtype=torch.float32)

        dt = np.r_[t[ix[0]], np.diff(t[ix])]
        with torch.no_grad():
            ctx = (
                self.encoder.encode(
                    tensor(x[None]),
                    tensor(dt[None, :, None]),
                    tensor(np.ones((1, len(ix), 1))),
                    torch.tensor([len(ix)], device=self.device),
                )
                .context[0]
                .cpu()
                .numpy()
            )
        # Seconds and millimetres are always derived from the source foot, never the hidden board.
        d = -rel[:, 2]
        sample = []
        for a, b in (
            (3.8, 4.05),
            (4.45, 4.68),
            (4.70, 4.94),
            (5.0, 5.20),
            (end - 0.50, end - 0.28),
            (end - 0.20, end),
        ):
            use = (t >= a) & (t <= b)
            sample.extend((np.median(d[use]) * 1000, np.median(fz[use]) / 10))
        return np.r_[ctx, sample], (identify(t, d, fz) if kind == "both" else None)


def identify(t, d, fz):
    # Fit equilibrium and transient spring, viscous damping and a contact bias.
    # The relaxation time is selected by measured force residual on the visible prefix.
    dt = np.median(np.diff(t))
    disp = savgol_filter(d, 25, 2)
    v = savgol_filter(d, 25, 2, deriv=1, delta=dt)
    acc = savgol_filter(d, 25, 2, deriv=2, delta=dt)
    m = (t > 3.70) & (fz > 9.0) & (abs(v) < 0.12)
    idx = np.where(m)[0][::4]
    if len(idx) < 10:
        raise ValueError("insufficient measured contacts")
    best = None
    from scipy.signal import lfilter

    for tau in (0.18, 0.40, 0.8, 1.4, 2.2, 3.2):
        alpha = -np.expm1(-dt / tau)
        memory = lfilter([alpha], [1.0, alpha - 1.0], disp)
        X = np.c_[disp[idx], disp[idx] - memory[idx], v[idx], np.ones(len(idx))]
        # D positive downward; downward acceleration = (applied load - support)/m.
        y = fz[idx] - 8 * acc[idx]
        fit = lsq_linear(
            X, y, bounds=([300.0, 0.0, 0.0, -90.0], [65000.0, 60000.0, 2500.0, 90.0])
        )
        err = float(np.mean((X @ fit.x - y) ** 2))
        if best is None or err < best[0]:
            best = (err, fit.x, tau)
    err, (k, kt, c, bias), tau = best
    return np.array(
        [k / 10000.0, kt / 10000.0, c / 300.0, tau, np.sqrt(err) / 10.0, bias / 20.0]
    )
