#!/usr/bin/env python3
"""Exp 23: can a clipping bound stop a client faking its noise/learning split?

The binding problem
-------------------
In commit-and-challenge the client commits to its clean update and its noise,
and answers challenges inside its declared core. The auditor checks the two
committed parts add up to what was sent. But inside the core only the client
can tell noise from learning. A cheater that halved its core noise can commit
to a different split of the same submitted matrix: a "noise" part that looks
full sized in the core, and a "learning" part that silently absorbs the
difference. Every check reads the sent matrix, which is unchanged, so every
check passes.

Differential privacy already requires each client's learning to be clipped to
a maximum size C. If the client must prove its committed learning is within
C, the fake split is constrained. This measures how much core noise a cheater
can still withhold.

The model, all inside the r x r core (exact, by rotational invariance)
----------------------------------------------------------------------
  true learning core  S        (singular values of a real adapter, clip active: ||S|| = C)
  cheater's noise     c * G    (G i.i.d. N(0, sigma^2), c < 1)
  sent core           S + c G
  committed noise     N' = (1 + C/||S + cG||) (S + c G), the largest noise claim the clip allows
  committed learning  S + c G - N'      has norm exactly C
  challenge on N'     must pass the honest-calibrated threshold

The cheater passes iff its claimed noise ||S + cG|| + C is large enough to
clear the challenge threshold, which sits near the honest size sigma r. Writing a = sigma r / C, a first-order
calculation gives the smallest feasible c as sqrt(max(0, 1 - 2/a)), so the
cheater can withhold about 1/a of its core noise. With the Gaussian mechanism
sigma = C k / epsilon (k = sqrt(2 ln(1.25/delta))), so a = k r / epsilon and
the answer depends only on the privacy level and the rank.

CPU-only. Reads FlexLoRA adapters for realistic learning spectra.
"""
import argparse
import glob
import json

import numpy as np
import torch

DELTA = 1e-5
K = float(np.sqrt(2 * np.log(1.25 / DELTA)))
RANKS = [4, 8, 16, 32]
EPS = [0.5, 1, 2, 4, 8, 16, 32, 64, 256, 1024, 23000]
Q = 20
ALPHA = 0.05
PASS = 0.9
N_TRIALS = 4000


def spectra(rank, pattern, n, rng):
    fs = sorted(glob.glob(pattern.format(r=rank)))
    pick = rng.choice(len(fs), size=min(n, len(fs)), replace=False)
    out = []
    for i in pick:
        st = torch.load(fs[i], map_location="cpu")
        A = st["lora_A"].double().numpy(); B = st["lora_B"].double().numpy()
        _, Rb = np.linalg.qr(B); _, Ra = np.linalg.qr(A.T)
        out.append(np.linalg.svd(Rb @ Ra.T, compute_uv=False))
    return np.array(out)


def chal_stat(M, sigma, rng):
    """q random unit-direction challenges on r x r cores M (batch, r, r)."""
    n, r, _ = M.shape
    a = rng.standard_normal((n, r, Q)); a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = rng.standard_normal((n, r, Q)); b /= np.linalg.norm(b, axis=1, keepdims=True)
    z = np.einsum("nrq,nrs,nsq->nq", a, M, b)
    return np.sum(z * z, axis=1) / sigma ** 2


def pass_rate(svals, eps, c, thr, rng, n):
    r = svals.shape[1]
    idx = rng.integers(len(svals), size=n)
    S = np.zeros((n, r, r)); S[:, np.arange(r), np.arange(r)] = svals[idx]
    C = np.linalg.norm(svals[idx], axis=1)                 # clip active at the true norm
    sigma = C * K / eps
    G = rng.standard_normal((n, r, r)) * sigma[:, None, None]
    sent = S + c * G
    nrm = np.linalg.norm(sent, axis=(1, 2))
    # best fake: claim as much as possible as noise while the committed
    # learning stays exactly at the clip, N' = (1 + C/||sent||) sent, so the
    # committed learning is -C sent/||sent|| with norm C. Forcing ||N'|| to the
    # honest mean instead (the first version) fails the clip on ordinary
    # fluctuations and wrongly rejects even a cheater with c = 1.
    Np = sent * (1.0 + C / nrm)[:, None, None]
    clip_ok = np.ones(n, dtype=bool)
    chal_ok = chal_stat(Np, sigma, rng) >= thr
    return float(np.mean(clip_ok & chal_ok)), float(np.mean(clip_ok)), float(np.mean(chal_ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="campaign/*_flexlora_*/adapters/round0[01][62]_*_r{r}__*q_proj.pt")
    ap.add_argument("--out", default="results/exp23_clipping_binding.json")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    # honest challenge threshold: independent of sigma and of the adapter
    r_thr = {}
    for r in RANKS:
        stats = [chal_stat(rng.standard_normal((20000, r, r)), np.ones(20000), rng)
                 for _ in range(5)]                         # 100k honest draws, chunked
        r_thr[r] = float(np.quantile(np.concatenate(stats), ALPHA))

    print("=" * 100)
    print("Exp 23: how much core noise can a cheater withhold behind a fake split, with a clip?")
    print(f"(delta={DELTA}, q={Q}, cheater must pass clip AND challenge with prob >= {PASS})")
    print("=" * 100)
    print("withheld = 1 - c_min, the share of its core noise (standard deviation) it can drop")
    out = []
    for r in RANKS:
        sv = spectra(r, args.pattern, 120, rng)
        print(f"\nrank {r}")
        print(f"  {'epsilon':>9} {'a = k r/eps':>12} {'withheld (sim)':>15} {'withheld (formula)':>19} "
              f"{'energy withheld':>16}")
        for eps in EPS:
            a = K * r / eps
            cs = np.round(np.arange(0.0, 1.0001, 0.01), 2)
            c_min = 1.0
            for c in cs:
                p, _, _ = pass_rate(sv, eps, c, r_thr[r], rng, N_TRIALS)
                if p >= PASS:
                    c_min = float(c); break
            form = float(np.sqrt(max(0.0, 1 - 2 / a)))
            row = {"rank": r, "epsilon": eps, "a": a, "c_min_sim": c_min, "c_min_formula": form,
                   "withheld_sim": 1 - c_min, "withheld_formula": 1 - form,
                   "energy_withheld_sim": 1 - c_min ** 2}
            out.append(row)
            print(f"  {eps:>9g} {a:>12.3f} {100*(1-c_min):>14.1f}% {100*(1-form):>18.1f}% "
                  f"{100*(1-c_min**2):>15.1f}%")
    with open(args.out, "w") as f:
        json.dump({"config": {"delta": DELTA, "q": Q, "pass": PASS, "k": K}, "rows": out}, f, indent=1)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
