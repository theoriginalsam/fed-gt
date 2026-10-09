#!/usr/bin/env python3
"""Exp 25: the audit game with participation, on measured detection curves.

Earlier incentive results (exp20) ask one question: what fine makes honesty a
best response? They assume every client stays in the federation. Here clients
can also leave, and some know the relabelling strategy of Proposition 2, which
the audit flags at exactly the honest false-alarm rate alpha.

Each client i has a participation value W_i, a cheating value V_i and a LoRA
rank r_i. With expected fine F = rho_a * P per audit window of T pooled rounds:

    exit        0
    honest      W - alpha * F
    shape(c)    W + V (1 - c^2) - D_r(c, T) * F       (D measured in exp20)
    relabel     W + V           - alpha * F           (sophisticated clients only)

and every client plays its best response. The relabelling payoff uses the
honest flag rate alpha, which Proposition 2 guarantees for the complete audit
transcript on average over the auditor's model P (not for each fixed dataset).
Strong-privacy shaping is not modelled as flagged at alpha: the complete
audit's pooled power against it was not measured. Worlds compared:

    matched     audit calibrated on matched adapters, no client can relabel
    s=x         a fraction x of clients knows relabelling
    sealed      noise is added inside a sealed component: no shaping or
                relabelling; bypassing the seal is visible (detection 1) and
                honest clients are never flagged (alpha = 0)

Outputs, as a function of F / median(V): participation, the share of
participants who are honest, and the noise energy actually delivered.
Also a feasibility bound from the held-out false-alarm rates of exp24.

CPU-only, seconds.
"""
import json, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE.parent / "results"

ALPHA = 0.05
RANKS = [4, 8, 16, 32]
N_CLIENTS = 20000
SEED = 25


def load_curves(T):
    d = json.load(open(RES / "exp20_incentive_frontier.json"))
    curves = {}
    for r in RANKS:
        row = next(x for x in d["ranks"][str(r)] if x["rounds"] == T)
        c = np.array([p["c"] for p in row["curve"]])
        D = np.array([p["detect"] for p in row["curve"]])
        curves[r] = (c, D)
    return curves


def _marginal(u, dist, sd):
    """Map uniforms to a value distribution with median 1."""
    from scipy import stats
    if dist == "lognormal":
        return np.exp(sd * stats.norm.ppf(u))
    if dist == "gamma2":                        # increasing hazard rate
        g = stats.gamma(2.0); return g.ppf(u) / g.median()
    if dist == "uniform":                       # increasing hazard rate
        return 2.0 * u
    if dist == "weibull0.7":                    # decreasing hazard rate
        w = stats.weibull_min(0.7); return w.ppf(u) / w.median()
    raise ValueError(dist)


def population(seed=SEED, n=N_CLIENTS, sd=1.0, dist="lognormal", rho=0.0):
    from scipy import stats
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    u = np.clip(stats.norm.cdf(z), 1e-12, 1 - 1e-12)
    W = _marginal(u[:, 0], dist, sd)
    V = _marginal(u[:, 1], dist, sd)
    rank = rng.choice(RANKS, n)
    u_soph = rng.random(n)            # client knows relabelling iff u_soph < s
    return W, V, rank, u_soph


def play(F, world, curves, W, V, rank, u_soph, s=0.0, gamma=1.0):
    """Best response of every client. Returns per-client strategy and noise kept (c^2)."""
    n = len(W)
    if world == "sealed":
        u_hon = W.copy()                         # no false alarms from attestation
        u_bypass = W + V - F                     # unsealed update is visible
        u = np.stack([np.zeros(n), u_hon, u_bypass])
        choice = np.argmax(u, 0)                 # 0 exit, 1 honest, 2 cheat
        kept = np.where(choice == 1, 1.0, np.where(choice == 2, 0.0, np.nan))
        return np.array(["exit", "honest", "cheat"])[choice], kept

    u_hon = W - ALPHA * F
    best_shape = np.full(n, -np.inf)
    best_c = np.ones(n)
    for r in RANKS:
        m = rank == r
        c, D = curves[r]
        sel = c < 1.0
        c, D = c[sel], D[sel]
        # utility of each c for clients of this rank: W + V(1-c^2) - D F
        U = W[m, None] + V[m, None] * (1 - c[None, :] ** 2) ** gamma - D[None, :] * F
        k = np.argmax(U, 1)
        best_shape[m] = U[np.arange(m.sum()), k]
        best_c[m] = c[k]
    soph = u_soph < s
    u_rel = np.where(soph, W + V - ALPHA * F, -np.inf)
    u = np.stack([np.zeros(n), u_hon, best_shape, u_rel])
    choice = np.argmax(u, 0)                     # ties go to the earlier (honest) option
    kept = np.select([choice == 1, choice == 2, choice == 3], [1.0, best_c ** 2, 0.0], np.nan)
    names = np.array(["exit", "honest", "cheat", "cheat"])
    return names[choice], kept


def summarise(strat, kept):
    part = strat != "exit"
    n_part = part.sum()
    return {
        "participation": float(part.mean()),
        "honest_share_of_participants": float((strat == "honest").sum() / n_part) if n_part else None,
        "honest_of_all": float((strat == "honest").mean()),
        "noise_kept_mean_participants": float(np.nanmean(kept)) if n_part else None,
    }


def heldout_bound():
    """Lower bound on W/V for an honest-inducing audit that honest clients accept.

    IC needs F >= V * G(c) / (D(c) - a) for every c; at c = 0, G = 1 and D <= 1,
    so F >= V / (1 - a). Participation of honest clients needs a * F <= W.
    Together: W / V >= a / (1 - a), whatever the detector.
    """
    e = json.load(open(RES / "exp24_heldout_calibration.json"))
    out = []
    for row in e["rows"]:
        for src, v in row["sources"].items():
            a = v["fa"]["decl"]
            out.append({"rank": row["rank"], "calibrated_on": src, "alpha_decl": a,
                        "min_W_over_V": a / (1 - a)})
    return out


def main():
    T = 10
    curves = load_curves(T)
    W, V, rank, u_soph = population()
    Fs = np.round(np.logspace(-1, 2, 31), 4)
    worlds = [("matched", 0.0), ("s=0.1", 0.1), ("s=0.5", 0.5), ("s=1.0", 1.0),
              ("sealed", 0.0)]
    sweep = {}
    for name, s in worlds:
        wname = "sealed" if name == "sealed" else "audit"
        sweep[name] = [dict(F=float(F), **summarise(*play(F, wname, curves, W, V, rank, u_soph, s)))
                       for F in Fs]

    # Server's best fine in each world: maximise the share of ALL clients that
    # participate honestly (protected participants).
    best = {k: max(v, key=lambda x: x["honest_of_all"]) for k, v in sweep.items()}

    # Robustness: dispersion of W and V, and pooled rounds T.
    robust = []
    for sd in (0.5, 1.0, 1.5):
        Wr, Vr, rr, ur = population(sd=sd)
        for T2 in (1, 10, 20):
            cv = load_curves(T2)
            for name, s in [("matched", 0.0), ("s=0.5", 0.5), ("sealed", 0.0)]:
                wname = "sealed" if name == "sealed" else "audit"
                rows = [summarise(*play(F, wname, cv, Wr, Vr, rr, ur, s)) for F in Fs]
                hs = [x["honest_share_of_participants"] for x in rows]
                ho = [x["honest_of_all"] for x in rows]
                robust.append({"sd": sd, "T": T2, "world": name,
                               "max_honest_of_all": max(ho),
                               "honest_share_at_max_F": hs[-1],
                               "peak_honest_share": max(h for h in hs if h is not None)})

    # Robustness to the assumed gain function and value distributions (T = 10).
    shapes = []
    for dist in ("lognormal", "gamma2", "uniform", "weibull0.7"):
        for rho in (-0.5, 0.0, 0.5):
            Wr, Vr, rr, ur = population(dist=dist, rho=rho)
            for gamma in (0.5, 1.0, 2.0):
                rec = {"dist": dist, "rho": rho, "gamma": gamma}
                for name, s in [("matched", 0.0), ("s=0.1", 0.1), ("s=0.5", 0.5), ("sealed", 0.0)]:
                    wname = "sealed" if name == "sealed" else "audit"
                    rows = [summarise(*play(F, wname, curves, Wr, Vr, rr, ur, s, gamma)) for F in Fs]
                    hs = [x["honest_share_of_participants"] for x in rows if x["honest_share_of_participants"] is not None]
                    rec[name] = {"max_honest_of_all": max(x["honest_of_all"] for x in rows),
                                 "peak_share": max(hs), "end_share": hs[-1]}
                shapes.append(rec)

    out = {"config": {"alpha": ALPHA, "T": T, "n_clients": N_CLIENTS, "seed": SEED,
                      "W_V": "independent lognormal, median 1, log-sd 1",
                      "F_grid": Fs.tolist()},
           "sweep": sweep, "server_best": best, "robustness": robust, "shapes": shapes,
           "heldout_bound": heldout_bound()}
    path = RES / "exp25_audit_game.json"
    json.dump(out, open(path, "w"), indent=1)

    print(f"T={T}, alpha={ALPHA}, {N_CLIENTS} clients")
    print(f"{'world':10s} {'F*':>6s} {'honest/all':>10s} {'particip':>9s} {'honest/part':>11s} {'noise kept':>10s}")
    for k, b in best.items():
        print(f"{k:10s} {b['F']:6.2f} {b['honest_of_all']:10.3f} {b['participation']:9.3f} "
              f"{b['honest_share_of_participants']:11.3f} {b['noise_kept_mean_participants']:10.3f}")
    print("\nhonest share of participants as the fine grows (F = 0.1, 1, 3, 10, 30, 100):")
    idx = [int(np.argmin(abs(Fs - f))) for f in (0.1, 1, 3, 10, 30, 100)]
    for k, v in sweep.items():
        print(f"  {k:10s} " + "  ".join(
            f"{v[i]['honest_share_of_participants']:.2f}/{v[i]['participation']:.2f}" for i in idx))
    print("\nrobustness (sd, T, world): max honest/all, honest share at F=100")
    for r in robust:
        print(f"  sd={r['sd']} T={r['T']:2d} {r['world']:8s} {r['max_honest_of_all']:.3f}  {r['honest_share_at_max_F']:.3f}")
    print("\nshape robustness (dist, rho, gamma): s=0.5 peak->end share, s=0.1 peak->end, sealed max honest/all")
    for r in shapes:
        print(f"  {r['dist']:10s} rho={r['rho']:+.1f} g={r['gamma']}  s=0.5 {r['s=0.5']['peak_share']:.2f}->{r['s=0.5']['end_share']:.2f}"
              f"  s=0.1 {r['s=0.1']['peak_share']:.2f}->{r['s=0.1']['end_share']:.2f}  sealed {r['sealed']['max_honest_of_all']:.3f}"
              f"  matched {r['matched']['max_honest_of_all']:.3f}")
    print("\nheld-out feasibility bound W/V >= a/(1-a):")
    for b in out["heldout_bound"]:
        print(f"  rank {b['rank']:2d} {b['calibrated_on']:8s} a={b['alpha_decl']:.3f}  W/V >= {b['min_W_over_V']:.3f}")
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
