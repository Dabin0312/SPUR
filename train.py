"""Family-grouped out-of-fold action labels, cross-fitted calibration, then freeze."""

import json, hashlib
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit, logit
from argparse import ArgumentParser
from pathlib import Path
from observer import Features, source_obs, ACTIONS, END
from policy import logistic, evaluate, feat, ctx, scale_fit, scaled
from value_model import ValueHeads, event_labels, train_heads


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def audit_target(case):
    s = json.loads((case / "G1/summary.json").read_text())
    with np.load(case / "G1/terrain.private.npz") as z:
        # Confirm the direct foot actually loaded the first-left board into collapse.
        failure_samples = np.flatnonzero(
            (z["depth"][:, 0] >= 0.048) & (z["gross_support"][:, 0] < 1.0)
        )
        onset = float(z["time"][failure_samples[0]]) if len(failure_samples) else None
    with np.load(case / "G1/rollout.npz") as z:
        feet = z["feet"]
        load = z["foot_fz"]
        on_board = (
            (feet[:, :, 0] > 1.0)
            & (feet[:, :, 0] < 3.25)
            & (abs(feet[:, :, 1] - 0.55) < 0.25)
        )
        loaded = onset is not None and bool(
            ((load > 40) & on_board & (z["time"][:, None] <= onset + 0.004)).any()
        )
    lost = bool(
        loaded
        and not s["crossed"]
        and (not s["fault"] or onset <= s["fault"]["time"] + 0.004)
    )
    return dict(
        arrived=bool(s["crossed"]),
        loss=lost,
        other=not s["crossed"] and not lost,
        loss_onset_s=onset,
        loaded_before_loss=loaded,
        terminal=s["terminal_stand_s"],
    )


def cross_readouts(X, Y, groups):
    preds = np.zeros_like(Y, dtype=float)
    for f in set(groups):
        tr = groups != f
        te = ~tr
        preds[te] = evaluate(logistic(X[tr], Y[tr]), X[te])
    return preds, logistic(X, Y)


def fit(out, dev, encoder_checkpoint):
    out.mkdir(parents=True, exist_ok=False)
    enc = Features(encoder_checkpoint)
    hist = []
    labels = []
    gids = []
    audit = []
    for i in range(16):
        case = dev / f"D{i:02}"
        s = audit_target(case)
        audit.append(dict(case=case.name, **s))
        if s["other"]:
            continue
        hist.append(
            [enc(source_obs(case / a / "rollout.npz"), END[a]) for a in ACTIONS]
        )
        labels.append(s["arrived"])
        gids.append(i // 2)
    write(out / "physical_labels.json", audit)
    groups = np.asarray(gids)
    y = np.array(labels)
    if len(set(groups)) < 4 or min(sum(y), sum(~y)) < 2:
        raise ValueError("Insufficient development G1 outcome variation")
    readouts = {}
    pred = {}
    for kind in ("gru", "id"):
        readouts[kind] = {}
        pred[kind] = []
        for j, a in enumerate(ACTIONS):
            xx = np.array([feat(h[j], kind) for h in hist])
            p, final = cross_readouts(xx, y, groups)
            readouts[kind][a] = final
            pred[kind].append(p)
        pred[kind] = np.stack(pred[kind], 1)

    # Map prices vary in development to expose different decision boundaries.
    # Report the complete predeclared grid; the test uses its own geometry prices.
    def events(kind):
        X = []
        Y = []
        harm = []
        E = []
        G = []
        case_ids = []
        acc = []
        threshold = []
        utility = []
        for i, h in enumerate(hist):
            for charge in (8.0, 35.0, 90.0, 150.0, 220.0, 310.0, 420.0):
                cut = 1 - charge / 650.0
                price = dict(
                    bypass_extra_s=charge, failed_direct_allowance_s=650.0, cutoff=cut
                )
                accepted = pred[kind][i] >= cut
                before = np.repeat(accepted[0], 2)
                good, bad = event_labels(before, accepted[1:], ~y[i], y[i])
                Y.append(good)
                harm.append(bad)
                E.append(np.stack([before, ~before], 1))
                x = (
                    ctx(h[0], pred[kind][i, 0], price)
                    if kind == "gru"
                    else np.r_[feat(h[0], "id"), pred[kind][i, 0]]
                )
                X.append(x)
                G.append(groups[i])
                case_ids.append(i)
                acc.append(accepted[0])
                threshold.append(cut)
                u = (
                    good[:, 0] * 650.0
                    + good[:, 1] * charge
                    - bad[:, 0] * charge
                    - bad[:, 1] * 650.0
                )
                utility.append(u)
        return tuple(
            map(np.asarray, (X, Y, harm, E, G, case_ids, acc, threshold, utility))
        )

    X, Y, H, E, G, ids, acc, cuts, ut = events("gru")
    all_z = np.zeros(Y.shape)

    def apply(x, w):
        n = ValueHeads(x.shape[1])
        n.load_state_dict(w)
        n.eval()
        with torch.no_grad():
            return n(torch.as_tensor(x, dtype=torch.float32)).numpy()

    for group in set(G):
        tr = G != group
        te = ~tr
        norm = scale_fit(X[tr])
        weights = train_heads(scaled(X[tr], norm), Y[tr], E[tr], ids[tr])
        all_z[te] = apply(scaled(X[te], norm), weights)
    temp = np.ones((2, 2))
    offset = np.zeros((2, 2))
    for a in range(2):
        for j in range(2):
            use = E[:, a, j]
            z = all_z[use, a, j]
            truth = Y[use, a, j].astype(float)
            if not len(z):
                offset[a, j] = -10.0
                continue

            def objective(v):
                zz = z / np.exp(v[0]) + v[1]
                return np.mean(np.logaddexp(0, zz) - truth * zz) + 0.025 * (
                    v[0] ** 2 + v[1] ** 2
                )

            ans = minimize(objective, np.zeros(2), bounds=[(-2.0, 2.0), (-5.0, 5.0)])
            temp[a, j] = np.exp(ans.x[0])
            offset[a, j] = ans.x[1]
            if not truth.any():
                offset[a, j] = -20.0
    norm = scale_fit(X)
    weights = train_heads(scaled(X, norm), Y, E, ids)
    XI, YI, HI, EI, GI, iids, ai, ct, _ = events("id")
    ni = scale_fit(XI)
    # Continuous utility regression, same representation / pricing / action opportunities.
    flat_norm = scale_fit(X)
    XF = np.column_stack((np.ones(len(X)), scaled(X, flat_norm)))
    flat = {
        a: dict(
            norm=flat_norm,
            w=np.linalg.solve(XF.T @ XF + 2.5 * np.eye(XF.shape[1]), XF.T @ ut[:, j]),
        )
        for j, a in enumerate(ACTIONS[1:])
    }
    write(
        out / "events.json",
        dict(
            OOF="paired parameter groups held out together",
            group_count=len(set(groups)),
            materials=len(y),
            counts_per_action_head=Y.sum(0).tolist(),
            eligible_counts=E.sum(0).tolist(),
            harm_counts=H.sum(0).tolist(),
            distinct_materials_with_events=[
                [len(set(ids[Y[:, a, j]])) for j in range(2)] for a in range(2)
            ],
            cue="Heads trained on cross-fitted predictions and verified physical G1 outcomes; class-absent heads suppressed.",
        ),
    )
    pack = dict(
        development_prior=float((sum(y) + 1) / (len(y) + 2)),
        norm=norm,
        net=weights,
        temp=temp,
        offset=offset,
        readouts=readouts,
        flat=flat,
        harm_rate=H.sum(0) / np.maximum(E.sum(0), 1),
        id_events=dict(
            X=scaled(XI, ni),
            norm=ni,
            group=iids,
            accepted=ai,
            threshold=ct,
            labels=YI,
            harms=HI,
        ),
    )
    torch.save(pack, out / "model.pt")
    write(
        out / "frozen.json",
        dict(
            sha256=hashlib.sha256((out / "model.pt").read_bytes()).hexdigest(),
            training="16 development material conditions, family paired, outcome censoring in physical_labels.json",
        ),
    )


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development", type=Path, required=True, help="Parent of D00 through D15"
    )
    parser.add_argument(
        "--encoder",
        type=Path,
        required=True,
        help="Pretrained contact encoder checkpoint",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory for fitted models"
    )
    args = parser.parse_args()
    fit(args.output, args.development, args.encoder)
