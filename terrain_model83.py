"""History-preserving source-conditioned support operator; development variant83.

The model learns terrain response. Target articulation, controller and collision
dynamics stay in the prediction simulator. Do not supply true material state,
test material ID, future measured loads or episode outcomes to these methods.
All features use documented, fixed physical scales, never body-weight-normalized
forces that erase the difference between source and target loading.
"""

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TerrainBelief:
    context: torch.Tensor
    memory: torch.Tensor

    def fork(self):
        """Independent memory for each counterfactual plan, same source context."""
        return TerrainBelief(self.context.clone(), self.memory.clone())


class ContactSupportOperator(nn.Module):
    # Per-foot: Fxyz, relative position xyz, relative velocity xyz, contact flag.
    contact_features = 10

    def __init__(self, width=64, context_size=32):
        super().__init__()
        self.width = width
        self.contact_mlp = nn.Sequential(
            nn.Linear(10, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU()
        )
        self.source_gru = nn.GRU(33, width, batch_first=True)
        self.history_queries = nn.Parameter(torch.randn(3, width) * 0.03)
        self.history_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.context_head = nn.Sequential(
            nn.LayerNorm(width * 4),
            nn.Linear(width * 4, width),
            nn.SiLU(),
            nn.Linear(width, context_size),
        )
        self.memory_head = nn.Linear(width * 4, width)
        self.initial_surface_head = nn.Sequential(
            nn.Linear(context_size + width, 32), nn.SiLU(), nn.Linear(32, 2)
        )
        self.memory_cell = nn.GRUCell(4 + context_size, width)
        # Base stiffness and damping, conditioned only on source observations.
        self.nominal = nn.Linear(context_size, 2)
        self.residual = nn.Sequential(
            nn.Linear(context_size + width + 2, width), nn.SiLU(), nn.Linear(width, 1)
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.register_buffer(
            "foot_scale",
            torch.tensor([400.0, 400.0, 400.0, 0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0]),
        )

    def encode(self, contacts, delta_t, foot_mask, lengths):
        """[B,T,F,10], [B,T,1], [B,T,F], [B] -> inferred terrain belief.

        Coordinates are relative to the observed contact patch. Missing feet
        are masked; temporal order is kept. No target plan is an input.
        """
        if contacts.ndim != 4 or contacts.shape[-1] != 10:
            raise ValueError("contact schema")
        if delta_t.shape != contacts.shape[:2] + (1,):
            raise ValueError("time schema")
        emb = self.contact_mlp(contacts / self.foot_scale)
        weights = foot_mask.to(emb.dtype).unsqueeze(-1)
        pooled = (emb * weights).sum(-2) / weights.sum(-2).clamp_min(1.0)
        seq = torch.cat([pooled, delta_t / 0.02], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            seq, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        encoded, last = self.source_gru(packed)
        tokens, _ = nn.utils.rnn.pad_packed_sequence(
            encoded, batch_first=True, total_length=contacts.shape[1]
        )
        padding = (
            torch.arange(tokens.shape[1], device=tokens.device)[None, :]
            >= lengths[:, None]
        )
        queries = self.history_queries[None].expand(tokens.shape[0], -1, -1)
        history, _ = self.history_attention(
            queries, tokens, tokens, key_padding_mask=padding, need_weights=False
        )
        # Retain the complete observed loading/release history even when the
        # last frames contain withdrawal. No target plan or true material input.
        combined = torch.cat([last[-1], history.flatten(1)], dim=-1)
        return TerrainBelief(
            torch.tanh(self.context_head(combined)),
            torch.tanh(self.memory_head(combined)),
        )

    def initial_surface(self, belief):
        """Infer post-source surface state from source observations, not truth."""
        raw = self.initial_surface_head(
            torch.cat([belief.context, belief.memory], dim=-1)
        )
        return 0.15 * torch.sigmoid(raw[..., 0]), 0.5 * torch.tanh(raw[..., 1])

    def support_force(self, belief, downward_depth, downward_velocity, panel_mass):
        """Gross upward support force in N.

        Query state belongs to the prediction world, not the real test world.
        With explicit gravity compensation, the physics adapter subtracts m*g
        before applying the external net force to a gravity-disabled panel.
        Positivity bounds numerical forces; this alone is NOT a passivity proof.
        """
        q = torch.stack([downward_depth / 0.05, downward_velocity / 0.5], dim=-1)
        positive = F.softplus(self.nominal(belief.context))
        k = 100.0 + 6000.0 * positive[..., 0]
        c = 10.0 + 120.0 * positive[..., 1]
        history = 200.0 * torch.tanh(
            self.residual(
                torch.cat([belief.context, belief.memory, q], dim=-1)
            ).squeeze(-1)
        )
        return (
            panel_mass * 9.81 + k * downward_depth + c * downward_velocity + history
        ).clamp(0.0, 2500.0)

    def advance(self, belief, depth, velocity, solver_contact_force, delta_t):
        """Call AFTER one physical prediction step; use solver-produced force.

        Contact force at t+1 updates memory for t+2. It never leaks into the
        force evaluation that produced that same physical step.
        """
        physical = torch.stack(
            [
                depth / 0.05,
                velocity / 0.5,
                solver_contact_force / 400.0,
                delta_t / 0.02,
            ],
            dim=-1,
        )
        hidden = self.memory_cell(
            torch.cat([belief.context, physical], dim=-1), belief.memory
        )
        return TerrainBelief(belief.context, hidden)
