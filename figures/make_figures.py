#!/usr/bin/env python3
"""Every plot in the paper, read from results/*.json and saved adapters. Output: figures/out/."""
import glob, json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
pathlib.Path(__file__).resolve().parent.joinpath("out").mkdir(exist_ok=True)

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = pathlib.Path(__file__).resolve().parent / "out"
J = lambda f: json.load(open(ROOT / "results" / f))

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 6.8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.1, "lines.markersize": 3.6,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COL = {4: "#0072B2", 8: "#E69F00", 16: "#009E73", 32: "#D55E00"}
MK = {4: "o", 8: "s", 16: "^", 32: "D"}
W1, W2 = 3.45, 7.0
DATA_SEALED = ROOT / "results" / "sealed_training"


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf"); plt.close(fig); print("wrote", name)


def panel(ax, tag):
    ax.text(-0.16, 1.04, tag, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


# --------------------------------------------------------------- spectra
def fig_spectra():
    import torch
    rng = np.random.default_rng(0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.25))
    for r in (4, 8, 16, 32):
        fs = sorted(glob.glob(str(ROOT / f"campaign/*_flexlora_*/adapters/round0[01][62]_*_r{r}__*q_proj.pt")))
        pick = rng.choice(len(fs), size=min(150, len(fs)), replace=False)
        prof = []
        for i in pick:
            st = torch.load(fs[i], map_location="cpu")
            A = st["lora_A"].double().numpy(); B = st["lora_B"].double().numpy()
            _, Rb = np.linalg.qr(B); _, Ra = np.linalg.qr(A.T)
            s = np.linalg.svd(Rb @ Ra.T, compute_uv=False); prof.append(s / s[0])
        prof = np.array(prof); idx = np.arange(1, r + 1)
        med, lo, hi = np.median(prof, 0), np.quantile(prof, .25, 0), np.quantile(prof, .75, 0)
        a.fill_between(idx, lo, hi, color=COL[r], alpha=.18, lw=0)
        a.plot(idx, med, marker=MK[r], color=COL[r], label=f"rank {r}", ms=2.6)
    a.set_xscale("log", base=2); a.set_yscale("log")
    a.set_xlabel("singular value index $i$"); a.set_ylabel("$s_i / s_1$ (median, IQR)")
    a.legend(loc="lower left"); panel(a, "(a)")
    rows = J("exp16_cross_dataset_spectra.json")["rows"]
    dsm = {"yelp": "o", "alpaca": "s", "gsm8k": "^"}
    for row in rows:
        x = {4: 0, 8: 1, 16: 2, 32: 3}[row["rank"]] + (-.12 if row["method"] == "hetero_spa" else .12)
        x += {"yelp": -.05, "alpaca": 0, "gsm8k": .05}[row["dataset"]]
        filled = row["method"] == "flexlora"
        b.errorbar(x, row["gamma_mean"], yerr=row["gamma_sd_over_seeds"], fmt=dsm[row["dataset"]],
                   color=COL[row["rank"]], mfc=COL[row["rank"]] if filled else "white", ms=3.4,
                   elinewidth=.6, capsize=1.2)
    b.set_xticks(range(4)); b.set_xticklabels(["rank 4", "rank 8", "rank 16", "rank 32"])
    b.set_ylabel(r"decay exponent $\gamma$ ($s_i \propto i^{-\gamma}$)")
    for lab, m in dsm.items():
        b.plot([], [], m, color="k", mfc="white", label=lab)
    b.plot([], [], "o", color="k", mfc="k", label="FlexLoRA (filled)")
    b.plot([], [], "o", color="k", mfc="white", label="SPA (open)")
    b.legend(loc="upper left", ncol=2); panel(b, "(b)")
    save(fig, "spectra")


# ---------------------------------------------------------- shaping attack
def fig_attack():
    rows = J("exp17_cross_dataset_security.json")["rows"]
    order = sorted(rows, key=lambda r: (r["rank"], ["yelp", "alpaca", "gsm8k"].index(r["dataset"]),
                                         r["method"]))
    x = np.arange(len(order)) + np.repeat(np.arange(4), 6) * 1.2
    fig, ax = plt.subplots(figsize=(W2, 2.0))
    ax.axhline(0.05, color="0.4", ls=":", lw=.8)
    ax.text(x[0] - .9, 0.075, r"honest rate $\alpha$", fontsize=6.5, color="0.35", ha="left")
    ax.plot(x, [r["attack_flag"] for r in order], "o", color="#D55E00", label="energy (tail) test vs. shaping cheater")
    ax.plot(x, [r["defence_power"] for r in order], "s", color="#0072B2", label="declared-subspace challenge vs. shaping cheater")
    ax.plot(x, [r["decoy_combined_flag"] for r in order], "^", color="#E69F00", label="declaration check vs. decoy")
    code = {"yelp": "Y", "alpaca": "A", "gsm8k": "G", "flexlora": "F", "hetero_spa": "S"}
    ax.set_xticks(x)
    ax.set_xticklabels([code[r["dataset"]] + code[r["method"]] for r in order], fontsize=5.5)
    ax.tick_params(axis="x", length=2, pad=1.5)
    for k in range(4):
        ax.text(x[k * 6:(k + 1) * 6].mean(), -0.3, f"rank {4 * 2 ** k}", ha="center", fontsize=7)
    ax.set_ylim(-0.05, 1.08); ax.set_ylabel("flag rate")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62), fontsize=6.6)
    save(fig, "attack_cells")


# ------------------------------------------------ declared vs estimated, partial
def fig_declared():
    e11 = {r["rank"]: r for r in J("exp11_commit_challenge.json")["rows"]}
    e12 = {r["rank"]: r for r in J("exp12_declared_subspace.json")["rows"]}
    R = [4, 8, 16, 32]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.15)); fig.subplots_adjust(wspace=0.3)
    a.plot(R, [e11[r]["q"]["20"]["power"] for r in R], "o-", color="#56B4E9",
           label="server estimates subspace, cheater starves core and strips")
    a.plot(R, [e12[r]["results"]["estimated|f=1.0"]["power"]["20"] for r in R], "o--", color="#D55E00",
           label="server estimates subspace, cheater starves core only")
    a.plot(R, [e12[r]["results"]["declared|f=1.0"]["power"]["20"] for r in R], "s-", color="#0072B2",
           label="client declares subspace")
    a.axhline(.05, color="0.4", ls=":", lw=.8)
    for r in R:
        a.text(r, 0.15, f"{e12[r]['alignment']:.3f}", ha="center", fontsize=6, color="0.35")
    a.text(4, 0.22, "alignment of server's estimate:", fontsize=6, color="0.35", ha="left")
    a.set_xscale("log", base=2); a.set_xticks(R); a.set_xticklabels(R); a.set_xlabel("rank $r$")
    a.set_ylabel("challenge power ($q=20$)"); a.set_ylim(-.03, 1.08)
    a.legend(loc="center left", bbox_to_anchor=(0, .52), fontsize=6.2); panel(a, "(a)")
    F = ["1.0", "0.5", "0.25"]; wdt = .2
    for i, r in enumerate(R):
        vals = [e12[r]["results"][f"declared|f={f}"]["power"]["20"] for f in F]
        b.bar(np.arange(3) + (i - 1.5) * wdt, vals, wdt, color=COL[r], label=f"rank {r}")
    b.axhline(.05, color="0.4", ls=":", lw=.8)
    b.set_xticks(range(3)); b.set_xticklabels(["all of core\n(50% RMS left)", "half of core\n(78% RMS left)",
                                                "quarter of core\n(85-91% RMS left)"])
    b.set_ylabel("challenge power ($q=20$)"); b.set_xlabel("share of the core the cheater starves")
    b.legend(loc="upper right"); panel(b, "(b)")
    save(fig, "declared_partial")


# ------------------------------------------------------------- visibility
def fig_visibility():
    rows = J("exp15_sigma_sweep.json")["rows"]
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    for r in (4, 8, 16, 32):
        rr = [x for x in rows if x["rank"] == r]
        ax.plot([x["visibility"] for x in rr], [x["alignment"] for x in rr], MK[r], color=COL[r],
                label=f"rank {r}", ms=4)
    ax.axvline(1, color="0.4", ls=":", lw=.8); ax.text(1.08, .35, "noise floor", fontsize=6.5, color="0.35")
    ax.set_xscale("log"); ax.set_xlabel(r"visibility $v = s_r / (\sigma(\sqrt{d_1}+\sqrt{d_2}))$")
    ax.set_ylabel("alignment of estimated subspace"); ax.legend(loc="lower right")
    save(fig, "visibility")


# --------------------------------------------------------------- rank check
def fig_rank():
    rows = {r["rank"]: r for r in J("exp18_two_sided_check.json")["rows"]}
    R = [4, 8, 16, 32]; V = ["decoy", "decoy_tuned", "decoy_weak", "decoy_weak_tuned"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.1))
    hon = [rows[r]["adversaries"]["honest"]["excess"] for r in R]
    dmin = [min(rows[r]["adversaries"][v]["excess"] for v in V) for r in R]
    dmax = [max(rows[r]["adversaries"][v]["excess"] for v in V) for r in R]
    xs = np.arange(4)
    a.bar(xs - .18, hon, .34, color="0.7", label="honest")
    a.bar(xs + .18, dmax, .34, color="#E69F00", label="decoy (max over 4 variants)")
    a.errorbar(xs + .18, dmax, yerr=[np.array(dmax) - np.array(dmin), np.zeros(4)], fmt="none",
               ecolor="k", elinewidth=.6, capsize=1.5)
    a.axhline(1, color="k", ls="--", lw=.7); a.text(3.45, 1.15, "MP edge", fontsize=6.5, ha="right")
    a.set_xticks(xs); a.set_xticklabels([f"rank {r}" for r in R])
    a.set_ylabel(r"excess $= s_{r+1}/(\sigma(\sqrt{d_1}+\sqrt{d_2}))$"); a.legend(loc="upper right")
    panel(a, "(a)")
    for j, v in enumerate(V):
        b.plot(R, [rows[r]["adversaries"][v]["flags"]["CH+RK"] for r in R], marker="os^D"[j],
               label=v.replace("_", " "), color=["#E69F00", "#D55E00", "#009E73", "#0072B2"][j])
    b.plot(R, [rows[r]["adversaries"]["honest"]["flags"]["CH+RK"] for r in R], "k:", label="honest (false alarm)")
    b.set_xscale("log", base=2); b.set_xticks(R); b.set_xticklabels(R); b.set_xlabel("rank $r$")
    b.set_ylabel("flag rate (challenge or rank check)"); b.set_ylim(-.03, 1.08)
    b.legend(loc="center left", bbox_to_anchor=(0, .42))
    panel(b, "(b)")
    save(fig, "rank_check")


# ---------------------------------------------------------- detection limit
def fig_limit():
    d = J("exp19_cheat_level_sweep.json"); C = d["config"]["cheats"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.1))
    for r in (4, 8, 16, 32):
        a.plot(C, d["table_a_power_q20_T1"][str(r)], marker=MK[r], color=COL[r], label=f"rank {r}")
        rows = d["table_b_rounds_to_power"][str(r)]
        b.plot([x["cheat"] for x in rows], [x["rounds"] for x in rows], marker=MK[r], color=COL[r], label=f"rank {r}")
    from scipy.stats import norm as _n
    za, zb = _n.ppf(0.95), _n.ppf(0.9)
    cc = np.linspace(.5, .95, 60)
    for r in (4, 8, 16, 32):
        meff = 20 / (1 + 19 / r ** 2)
        b.plot(cc, 2 * (za + cc ** 2 * zb) ** 2 / (meff * (1 - cc ** 2) ** 2), "--", color=COL[r], lw=.8)
    b.plot([], [], "k--", lw=.8, label=r"variance-matched approximation")
    a.axhline(.05, color="0.4", ls=":", lw=.8)
    a.set_xlabel("cheat level $c$ (fraction of noise delivered)"); a.set_ylabel("power ($q=20$, one round)")
    a.legend(); panel(a, "(a)")
    b.set_yscale("log"); b.set_xlabel("cheat level $c$"); b.set_ylabel("audited rounds for power 0.9")
    b.legend(); panel(b, "(b)")
    save(fig, "detection_limit")


# --------------------------------------------------------------- incentives
def fig_incentive():
    d = J("exp20_incentive_frontier.json")
    fig, ax = plt.subplots(figsize=(W1, 2.15))
    for r in (4, 8, 16, 32):
        rows = d["ranks"][str(r)]
        ax.plot([x["rounds"] for x in rows], [x["penalty_over_V"] for x in rows], marker=MK[r],
                color=COL[r], label=f"rank {r}")
    ax.axhline(1 / 0.95, color="k", ls="--", lw=.8)
    ax.text(100, 1.12, r"floor $1/(1-\alpha)=1.05$", fontsize=6.5, ha="right")
    ax.set_xscale("log"); ax.set_xlabel("audited rounds pooled $T$"); ax.set_ylabel("required fine $P/V$")
    ax.legend(); save(fig, "incentive")


# ------------------------------------------------------------- regime map
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def exact_eps_rows(rows, n):
    import sys; sys.path.insert(0, str(ROOT))
    from fedgt.privacy_calibration import eps_of_sigma
    K = np.sqrt(2 * np.log(1.25 / 1e-5))
    out = []
    for x in rows:
        C = x["eps_module_round"] * x["sigma"] / K          # the median norm the run used
        out.append(dict(x, eps_exact=eps_of_sigma(x["sigma"], 2 * C), n=n))
    return out


def fig_regime():
    rows = exact_eps_rows(J("exp22_realistic_noise.json")["rows"], 60)
    dense = ROOT / "results" / "exp22b_transition_sweep.json"
    if dense.exists():
        rows += exact_eps_rows(json.load(open(dense))["rows"], 150)
    e23 = J("exp23_clipping_binding.json")["rows"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.55), sharex=True)
    fig.subplots_adjust(wspace=0.28)
    for r in (4, 8, 32):
        rr = sorted([x for x in rows if x["rank"] == r], key=lambda x: -x["eps_exact"])
        eps = np.array([x["eps_exact"] for x in rr])
        for key, ls, mfc in (("decl_vs_random", "-", None), ("decl_vs_tilt15", "--", "white")):
            p = np.array([x[key] for x in rr]); n = np.array([x["n"] for x in rr])
            lo, hi = zip(*[wilson(pp * nn, nn) for pp, nn in zip(p, n)])
            err = [np.clip(p - np.array(lo), 0, None), np.clip(np.array(hi) - p, 0, None)]
            a.errorbar(eps, p, yerr=err, marker=MK[r], color=COL[r],
                       ls=ls, mfc=mfc or COL[r], elinewidth=.5, capsize=1.2, ms=3.2,
                       label=f"rank {r}" if key == "decl_vs_random" else None)
    a.axhline(.05, color="0.4", ls=":", lw=.8)
    a.plot([], [], "k-", label="random declaration")
    a.plot([], [], "k--", mfc="white", label="15° tilted declaration")
    a.set_ylabel("declaration check power"); a.set_ylim(-.04, 1.1)
    a.legend(fontsize=6, ncol=5, loc="lower left", bbox_to_anchor=(0.0, 1.06), columnspacing=0.8,
             handlelength=1.6)
    panel(a, "(a)")
    for r in (4, 8, 16, 32):
        rr = sorted([x for x in e23 if x["rank"] == r], key=lambda x: -x["epsilon"])
        b.plot([x["epsilon"] for x in rr], [x["withheld_sim"] for x in rr], marker=MK[r], color=COL[r],
               label=f"rank {r}")
        b.plot([x["epsilon"] for x in rr], [x["withheld_formula"] for x in rr], color=COL[r], ls=":", lw=.8)
    b.plot([], [], "k:", lw=.8, label=r"formula $1-\sqrt{1-2/a}$")
    b.set_ylabel("core noise withheld\n(cheater that keeps its learning)"); b.set_ylim(-.04, 1.08)
    b.set_xscale("log"); b.invert_xaxis()
    for ax in (a, b):
        ax.set_xlabel(r"exact $\varepsilon$ per module per round  (stronger privacy $\rightarrow$)")
    b.legend(fontsize=6, loc="center left"); panel(b, "(b)")
    for ax in (a, b):
        ax.axvspan(10, 0.35, color="0.92", zorder=0)
        ax.axvline(5e7, color="0.5", ls="-.", lw=.7)
    a.text(3e7, 0.5, "our test\nsetting", fontsize=6, ha="right", color="0.35")
    a.text(1.9, 0.5, "meaningful\nprivacy", fontsize=6, ha="center", color="0.35")
    save(fig, "regime_map")


# --------------------------------------------------------------- audit game (exp25)
def fig_game():
    d = J("exp25_audit_game.json")["sweep"]
    style = {"matched": ("no relabelling", "#0072B2", "-"),
             "s=0.1": ("10% can relabel", "#E69F00", "-"),
             "s=0.5": ("50% can relabel", "#D55E00", "-"),
             "s=1.0": ("all can relabel", "#777777", ":"),
             "sealed": ("sealed noise", "#009E73", "--")}
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.1))
    keys = [("honest_share_of_participants", "honest share of participants"),
            ("participation", "participation"),
            ("honest_of_all", "honest participants / all clients")]
    for ax, (k, lab), tag in zip(axs, keys, "abc"):
        for w, (name, col, ls) in style.items():
            F = [x["F"] for x in d[w]]; y = [x[k] for x in d[w]]
            ax.plot(F, y, ls, color=col, label=name)
        ax.set_xscale("log"); ax.set_ylim(-0.03, 1.03); ax.set_xlabel("expected fine $F$ / median $V$")
        ax.set_ylabel(lab); panel(ax, f"({tag})")
    axs[1].legend(loc="lower left")
    fig.tight_layout(w_pad=1.2)
    save(fig, "audit_game")


# --------------------------------------------------------------- training with enforced noise (GPU runs)
def fig_utility():
    """(a) Perplexity per round for each 20-round run (seed 42). (b) Share of clip-only training's
    one-round perplexity gain reached by denoised releases, per seed, against eps per release."""
    D = DATA_SEALED
    load = lambda f: json.load(open(f))
    full = {}
    for f in sorted((D / "yelp").glob("*/sealed_*_seed42.json")):
        d = load(f); full[d["condition"]] = d
    probe = {}
    for f in sorted((D / "probe" / "yelp").glob("*/sealed_*_seed*.json")):
        d = load(f); seed = int(f.stem.split("seed")[1])
        probe.setdefault(seed, {})[d["condition"]] = d["rounds"][0]
    base42 = full["base"]["rounds"][0]["perplexity"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.3))
    sty = {"none": ("no clip, no noise", "#000000", "-"), "clip": ("clip only", "#777777", "--"),
           "1e7": (r"$\varepsilon=10^7$, raw", "#D55E00", ":"),
           "1e7d": (r"$\varepsilon=10^7$, denoised", "#009E73", "-"),
           "1e6d": (r"$\varepsilon=10^6$, denoised", "#0072B2", "-")}
    for c, (lab, col, ls) in sty.items():
        if c in full:
            y = [r["perplexity"] for r in full[c]["rounds"]]
            a.plot(np.arange(1, len(y) + 1), y, ls, color=col, label=lab)
    a.axhline(base42, color="#bbbbbb", ls=":", lw=1, label="base model")
    a.set_xlabel("round"); a.set_ylabel("perplexity"); a.set_xticks([1, 5, 10, 15, 20])
    a.set_ylim(7, 15); a.legend(loc="upper right", fontsize=6); panel(a, "(a)")
    levels = ["1e4d", "1e5d", "1e6d", "1e7d", "1e8d"]
    eps = [float(l[:-1]) for l in levels]
    base = {42: base42}
    clip1 = {42: full["clip"]["rounds"][0]["perplexity"]}
    for sd in (43, 44):
        base[sd] = probe[sd]["1e4d"]["perplexity"]          # nothing kept: the base model
        clip1[sd] = probe[sd]["clip"]["perplexity"]
    shades = {42: "#0072B2", 43: "#56B4E9", 44: "#003f6b"}
    fr_all, kept_all = [], []
    for sd in sorted(probe):
        fr = [(base[sd] - probe[sd][l]["perplexity"]) / (base[sd] - clip1[sd]) for l in levels]
        fr_all.append(fr)
        kept_all.append([probe[sd][l]["sealed"]["denoise"]["kept"] for l in levels])
        b.plot(eps, fr, "o-", color=shades[sd], ms=3, lw=0.9, label=f"seed {sd}")
    kept_mean = np.mean(kept_all, 0)
    for e, f_, k in zip(eps, np.max(fr_all, 0), kept_mean):
        b.annotate(f"{k:.0f}", (e, f_), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=6)
    b.axhline(1, color="#777777", ls="--", lw=1)
    b.axhline(0, color="#bbbbbb", ls=":", lw=1)
    b.set_xscale("log"); b.set_xlabel(r"$\varepsilon$ per release")
    b.set_ylabel("share of clip-only gain, one round"); b.set_ylim(-0.08, 1.15)
    b.legend(loc="upper left", fontsize=6); panel(b, "(b)")
    fig.tight_layout(w_pad=1.2)
    save(fig, "utility")


if __name__ == "__main__":
    for f in (fig_spectra, fig_attack, fig_declared, fig_visibility, fig_rank, fig_limit, fig_incentive, fig_regime, fig_game, fig_utility):
        f()
