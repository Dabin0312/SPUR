"""Select G1's route from causal Go2 observations on a two-fork course."""

from observer import END
from route_cost import opportunity


def select_route(
    policy,
    costs,
    initial_probe,
    extend_probe,
    *,
    method="two_heads",
    failure_allowance_s=650.0,
):
    """Return (route, observations) in the order in which Go2 tested the planks.

    costs gives route durations in seconds for LL, LR, RL, RR and BB (bypass).
    initial_probe(panel) supplies Go2 readings through the initial load.
    extend_probe(panel, action) continues that contact and supplies its full history.
    Both callbacks return time, feet and foot_fz arrays. The simulator adapter
    carries the post-probe plank state into subsequent probes and the G1 run.
    """
    if set(costs) != {"LL", "LR", "RL", "RR", "BB"} or failure_allowance_s <= 0:
        raise ValueError(
            "Expected four shortcut costs, a bypass cost and a positive failure allowance."
        )
    methods = {
        "two_heads",
        "physics_id",
        "no_rescue",
        "no_loss",
        "flat",
        "short",
        "hold",
        "always",
        "no_probe",
    }
    if method not in methods:
        raise ValueError(f"Unknown policy: {method}")
    routes = sorted((r for r in costs if r != "BB"), key=lambda r: (costs[r], r))
    records = {}
    if method == "no_probe":
        return routes[0], records

    cutoff = 1.0 - (costs["BB"] - costs[routes[0]]) / (2 * failure_allowance_s)
    feature_kind = "id" if method == "physics_id" else "gru"

    def acquire(panel):
        extra = opportunity(panel, records, costs, policy.pack["development_prior"])
        prices = dict(
            bypass_extra_s=extra,
            failed_direct_allowance_s=failure_allowance_s,
            cutoff=cutoff,
        )
        prefix = initial_probe(panel)
        initial = policy.features(prefix, END["short"], feature_kind)
        action, _ = policy.choose(initial, method, prices)
        row = initial
        if action != "short":
            history = extend_probe(panel, action)
            row = policy.features(history, END[action], feature_kind)
        score = policy.readout(row, method, action)
        records[panel] = dict(
            action=action, probability=score, safe=bool(score >= cutoff)
        )

    if method == "always":
        for panel in ("P1L", "P1R", "P2L", "P2R"):
            acquire(panel)
    for route in routes:
        panels = ("P1" + route[0], "P2" + route[1])
        if any(p in records and not records[p]["safe"] for p in panels):
            continue
        for panel in panels:
            if panel not in records:
                acquire(panel)
            if not records[panel]["safe"]:
                break
        else:
            return route, records
    return "BB", records
