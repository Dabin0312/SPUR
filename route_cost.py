"""Expected extra route duration if one candidate panel is unusable."""

from itertools import product


def opportunity(panel, records, costs, prior):
    boards = ("P1L", "P1R", "P2L", "P2R")
    known = {p: o["safe"] for p, o in records.items() if p != panel}
    missing = [p for p in boards if p != panel and p not in known]
    total = 0.0
    for bits in product((False, True), repeat=len(missing)):
        regular = dict(known, **dict(zip(missing, bits)))
        weight = prior ** sum(bits) * (1 - prior) ** (len(bits) - sum(bits))

        def best(ok):
            state = dict(regular, **{panel: ok})
            return min(
                t
                for r, t in costs.items()
                if r == "BB" or (state["P1" + r[0]] and state["P2" + r[1]])
            )

        total += weight * (best(False) - best(True))
    return max(0.0, float(total))
