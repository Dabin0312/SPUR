"""Two action-conditioned probabilities after a shared Go2 history encoder."""

import numpy as np
import torch
from torch import nn
from observer import ACTIONS, OPERATION


class ValueHeads(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(dim, 24), nn.SiLU(), nn.Linear(24, 16), nn.SiLU()
        )
        self.detect_loss = nn.Linear(16, 2)
        self.recover_shortcut = nn.Linear(16, 2)

    def forward(self, x):
        h = self.trunk(x)
        return torch.stack((self.detect_loss(h), self.recover_shortcut(h)), dim=-1)


def durations(costs):
    # Costs already expressed in physical seconds; failed missions reported separately.
    return float(costs["failed_direct_allowance_s"]), float(costs["bypass_extra_s"])


def decision(prob, accepted, event_prices, extra, wrong):
    # Validation harmful-switch probabilities penalize both directions.
    cfall, detour = durations(event_prices)
    good, bad = (0, 1) if accepted else (1, 0)
    gain, loss = (cfall, detour) if accepted else (detour, cfall)
    values = prob[:, good] * gain - wrong[:, good] * loss - extra
    best = int(np.argmax(np.r_[0.0, values]))
    return ACTIONS[best], values.tolist()


def event_labels(before, after, support_loss, completed):
    avoided = (before == 1) & (after == 0) & support_loss
    restored = (before == 0) & (after == 1) & completed
    # Opposite changes are recorded for calibrating the downside of information.
    harm_detour = (before == 1) & (after == 0) & completed
    harm_direct = (before == 0) & (after == 1) & support_loss
    return np.stack((avoided, restored), axis=-1), np.stack(
        (harm_detour, harm_direct), axis=-1
    )


def train_heads(x, labels, eligible, groups):
    # loss weights sum equally per material, even if several map costs share it.
    torch.manual_seed(142)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.as_tensor(x, dtype=torch.float32, device=device)
    Y = torch.as_tensor(labels, dtype=torch.float32, device=device)
    E = torch.as_tensor(eligible, dtype=torch.float32, device=device)
    group = np.array(groups)
    wt = torch.tensor(
        [1 / np.sum(group == g) for g in group], device=device, dtype=torch.float32
    )
    net = ValueHeads(x.shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=0.003, weight_decay=0.02)
    for _ in range(180):
        logits = net(X)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, Y, reduction="none"
        )
        loss = ((loss * E).sum((1, 2)) * wt).sum() / (
            (E.sum((1, 2)) * wt).sum().clamp_min(1.0)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {k: v.detach().cpu() for k, v in net.state_dict().items()}
