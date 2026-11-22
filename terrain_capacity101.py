"""Source-conditioned irreversible-support extension of the v83 operator.

The original positive spring plus bounded residual cannot represent zero support
after a large irreversible collapse. This version carries a monotone loss state
between counterfactual contact steps and gates the entire support force.
Parameter bounds are modeling choices, not claims of universal terrain physics.
"""

from dataclasses import dataclass
import torch
from torch import nn
from terrain_model83 import ContactSupportOperator


@dataclass
class CapacityBelief:
    context: torch.Tensor
    memory: torch.Tensor
    damage: torch.Tensor

    def fork(self):
        return CapacityBelief(
            self.context.clone(), self.memory.clone(), self.damage.clone()
        )


class CapacitySupportOperator(ContactSupportOperator):
    def __init__(self):
        super().__init__()
        self.capacity_head = nn.Linear(32, 2)
        nn.init.zeros_(self.capacity_head.weight)
        # Broad stroke limits; initialized from the declared development family
        # prior, then learned with equally available force/damage supervision.
        with torch.no_grad():
            self.capacity_head.bias.copy_(torch.logit(torch.tensor([0.44, 0.40])))
        self.initial_damage_head = nn.Linear(32 + 64, 1)
        nn.init.zeros_(self.initial_damage_head.weight)
        nn.init.constant_(self.initial_damage_head.bias, -8.0)

    def encode(self, contacts, delta_t, foot_mask, lengths):
        b = super().encode(contacts, delta_t, foot_mask, lengths)
        damage = torch.sigmoid(
            self.initial_damage_head(torch.cat([b.context, b.memory], -1))
        ).squeeze(-1)
        return CapacityBelief(b.context, b.memory, damage)

    def capacity(self, belief):
        raw = torch.sigmoid(self.capacity_head(belief.context))
        yielding = 0.010 + 0.050 * raw[..., 0]
        failed = yielding + 0.004 + 0.030 * raw[..., 1]
        return yielding, failed

    def damage_at(self, belief, depth):
        yielding, failed = self.capacity(belief)
        current = ((depth - yielding) / (failed - yielding)).clamp(0.0, 1.0)
        return torch.maximum(belief.damage, current)

    def support_force(self, belief, downward_depth, downward_velocity, panel_mass):
        intact = super().support_force(
            belief, downward_depth, downward_velocity, panel_mass
        )
        return (1.0 - self.damage_at(belief, downward_depth)) * intact

    def advance(self, belief, depth, velocity, solver_contact_force, delta_t):
        updated = super().advance(
            belief, depth, velocity, solver_contact_force, delta_t
        )
        return CapacityBelief(
            updated.context, updated.memory, self.damage_at(belief, depth)
        )
