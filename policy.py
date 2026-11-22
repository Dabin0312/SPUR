"""Deployable contact-value head and physics control, using robot observations only."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.linalg import expm
import torch
from observer import Features, ACTIONS, END, OPERATION
from value_model import ValueHeads, decision


def scale_fit(x):
    x = np.asarray(x)
    return dict(mean=x.mean(0), std=np.maximum(x.std(0), 1e-3))


def scaled(x, fit):
    return np.clip((x - fit["mean"]) / fit["std"], -5.0, 5.0)


def logistic(x, y):
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)
    norm = scale_fit(x)
    X = np.column_stack((np.ones(len(y)), scaled(x, norm)))

    def obj(w):
        z = X @ w
        penalty = np.r_[0.1, np.full(x.shape[1], 2.5)]
        return (np.logaddexp(0.0, z) - y * z).sum() + 0.5 * np.sum(
            penalty * w * w
        ), X.T @ (expit(z) - y) + penalty * w

    fit = minimize(obj, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B")
    if not fit.success:
        raise RuntimeError(fit.message)
    return dict(norm=norm, w=fit.x)


def evaluate(fit, x):
    return expit(fit["w"][0] + scaled(x, fit["norm"]) @ fit["w"][1:])


def id_readout(x):
    k, kt, c, tau, err, bias = x
    # Predicted sinking under 335 N following a 0.65s ramp and a 4s hold.
    # A solved damping or stiffness near the fit bounds can make explicit Euler
    # unstable at 8 ms even for a passive board. Integrate each constant force
    # sample using its exact linear state transition (depth, velocity, memory).
    dt = 0.008
    dynamics = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-10000 * (k + kt) / 8.0, -300 * c / 8.0, 10000 * kt / 8.0, 1.0 / 8.0],
            [1.0 / tau, 0.0, -1.0 / tau, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    transition = expm(dynamics * dt)
    A = transition[:3, :3]
    B = transition[:3, 3]
    state = np.zeros(3)
    u = np.clip((np.arange(610) * dt - 0.15) / 0.65, 0.0, 1.0)
    peak = 0.0
    for force in 334.86 * u * u * (3 - 2 * u):
        state = A @ state + B * force
        peak = max(peak, state[0])
    return [peak * 1000, k, kt, tau, err]


def feat(row, kind):
    return row[0] if kind == "gru" else np.array(id_readout(row[1]))


def ctx(row, prob, prices):
    return np.r_[
        row[0][:32],
        prob,
        prices["bypass_extra_s"] / 300,
        prices["failed_direct_allowance_s"] / 600,
        prices["cutoff"],
    ]


def calibrated(z, temperature, offset):
    return expit(z / temperature + offset)


class Policy:
    def __init__(self, path, encoder_checkpoint):
        self.pack = torch.load(path, map_location="cpu", weights_only=False)
        self.features = Features(encoder_checkpoint)
        self.head = ValueHeads(len(self.pack["norm"]["mean"]))
        self.head.load_state_dict(self.pack["net"])
        self.head.eval()

    def readout(self, row, method, a):
        kind = "id" if method == "physics_id" else "gru"
        return float(evaluate(self.pack["readouts"][kind][a], feat(row, kind)))

    def choose(self, row, method, prices):
        prob = self.readout(row, method, "short")
        cutoff = prices["cutoff"]
        accepted = bool(prob >= cutoff)
        extra = np.array([OPERATION[a] - OPERATION["short"] for a in ACTIONS[1:]])
        if method in ("short", "always", "hold"):
            return ("short" if method == "short" else "hold"), {
                "current": prob,
                "cutoff": cutoff,
            }
        if method == "physics_id":
            # Development-calibrated nearest-neighbour VoI using identified material
            # parameters, initial pass probability and both measured switch directions.
            db = self.pack["id_events"]
            X = scaled(np.r_[feat(row, "id"), prob], db["norm"])
            distance = np.mean((db["X"] - X) ** 2, axis=1) + 4 * abs(
                db["threshold"] - cutoff
            )
            idx = np.where(db["accepted"] == accepted)[0]
            if not len(idx):
                return "short", {"current": prob, "cutoff": cutoff}
            # One vote per physical board, even when evaluated at several map prices.
            idx = sorted(idx, key=lambda j: distance[j])
            used = set()
            nearest = []
            for j in idx:
                mat = int(db["group"][j])
                if mat not in used:
                    nearest.append(j)
                    used.add(mat)
                if len(nearest) == 5:
                    break
            weights = np.exp(-np.minimum(distance[nearest], 20.0)) + 1e-6
            P = np.average(db["labels"][nearest], axis=0, weights=weights)
            wrong = np.average(db["harms"][nearest], axis=0, weights=weights)
        else:
            x = torch.tensor(
                scaled(ctx(row, prob, prices), self.pack["norm"]), dtype=torch.float32
            )
            with torch.no_grad():
                z = self.head(x).numpy()
            P = calibrated(z, self.pack["temp"], self.pack["offset"])
            wrong = self.pack["harm_rate"]
            if method == "no_loss":
                P[:, 0] = 0
            if method == "no_rescue":
                P[:, 1] = 0
            if method == "flat":
                vals = np.r_[
                    0.0,
                    [
                        (
                            self.pack["flat"][a]["w"][0]
                            + scaled(
                                ctx(row, prob, prices), self.pack["flat"][a]["norm"]
                            )
                            @ self.pack["flat"][a]["w"][1:]
                        )
                        - extra[i]
                        for i, a in enumerate(ACTIONS[1:])
                    ],
                ]
                return ACTIONS[int(vals.argmax())], {
                    "current": prob,
                    "cutoff": cutoff,
                    "utility": vals.tolist(),
                }
        a, v = decision(P, accepted, prices, extra, wrong)
        return a, dict(
            current=prob,
            cutoff=cutoff,
            benefit_event_probability=P.tolist(),
            harm_probability=wrong.tolist(),
            net_value_seconds=v,
        )
