#!/usr/bin/env python3
"""Exp 17: do the security results hold across datasets and methods?

exp16 showed the spectra generalise. This checks the three claims that
actually matter for the paper, across the full campaign (3 datasets x 2
methods x 3 seeds pooled):

  ATTACK    does the calibrated shaping adversary still evade the tail test?
            (exp7 found flag rates of 0.03 to 0.16 against honest 0.03 to 0.10)
  DEFENCE   does the declared-subspace challenge still catch it?
            (exp12 found power 1.00 at every rank with 20 challenges)
  DECOY     does the decoy still slip past both the energy check and the
            challenge? (exp13 found 0.03 to 0.08)

Seeds are pooled per dataset and method, which matters at rank 32 where only
one client is sampled per round, giving 84 adapters per cell rather than 28.
Sample sizes are reported so the thinner cells are visible.

CPU-only. A few hours. Reads ./campaign/.
"""
import argparse, glob, json, os, re, time
import numpy as np, torch
from sklearn.utils.extmath import randomized_svd
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fedgt.spectral_audit import tail_energy_dof

D, MARGIN, SIGMA, CHEAT, Q, ALPHA, N_ITER = 3584, 2, 5e-5, 0.5, 20, 0.05, 4
DATASETS = ["yelp", "alpaca", "gsm8k"]
METHODS  = ["hetero_spa", "flexlora"]
RANKS    = [4, 8, 16, 32]


def load_pool(ds, method, rank, n, rng):
    """All three seeds pooled, q_proj only."""
    files = sorted(glob.glob(f"campaign/{ds}_{method}_s*/adapters/*_r{rank}__*q_proj.pt"))
    if not files: return []
    pick = rng.choice(len(files), size=min(n, len(files)), replace=False)
    out = []
    for i in pick:
        st = torch.load(files[i], map_location="cpu")
        out.append((st["lora_A"].float().numpy().astype(np.float64),
                    st["lora_B"].float().numpy().astype(np.float64)))
    return out, len(files)


def kk(r): return min(r + MARGIN, D - 1)
def basis(W, r):
    U, s, Vt = randomized_svd(W, n_components=r, n_iter=N_ITER, random_state=0)
    return U, Vt.T, s
def proj_out(E, U, V):
    E = E - U @ (U.T @ E); return E - (E @ V) @ V.T
def tail_sigma2(W, r):
    k = kk(r)
    _, s, _ = randomized_svd(W, n_components=k, n_iter=N_ITER, random_state=None)
    return (float(np.sum(W*W)) - float(np.sum(s**2))) / tail_energy_dof(D, D, k)
def chal(C, q, rng):
    r = C.shape[0]
    A = rng.normal(size=(r,q)); A /= np.linalg.norm(A,axis=0,keepdims=True)
    B = rng.normal(size=(r,q)); B /= np.linalg.norm(B,axis=0,keepdims=True)
    return float(np.sum(np.einsum("ij,ij->j", A, C @ B)**2) / SIGMA**2)
def energy_ratio(Ws, Ud, Vd, r):
    _, s, _ = randomized_svd(Ws, n_components=r, n_iter=N_ITER, random_state=0)
    return float(np.linalg.norm(Ud.T @ Ws @ Vd)**2) / max(float(np.sum(s**2)), 1e-300)


def shaped_calibrated(W, r, rng):
    """exp7's adversary: starve the core block, pad to match an honest tail reading."""
    U, V, _ = basis(W, r)
    N = rng.normal(0.0, SIGMA, size=W.shape)
    C = U.T @ N @ V
    N = N + U @ ((CHEAT - 1.0) * C) @ V.T                      # starve the core
    k = kk(r); dof = tail_energy_dof(D, D, k)
    target = tail_sigma2(W + rng.normal(0.0, SIGMA, size=W.shape), r)
    E = proj_out(rng.normal(0.0, 1.0, size=W.shape), U, V)
    E /= (np.linalg.norm(E) + 1e-30)
    c0 = tail_sigma2(W + N, r) * dof
    a  = np.sqrt(max(target*dof - c0, 0.0))
    return W + N + a*E, N, U, V


def decoy(W, r, rng):
    """exp13's survivor: fake direction declared in place of the weakest real one."""
    U, V, sv = basis(W, r)
    N = rng.normal(0.0, SIGMA, size=W.shape)
    C = U.T @ N @ V
    d = np.where(np.arange(r) == r-1, CHEAT, 1.0)
    N = N + U @ ((d[:,None]-1.0)*C) @ V.T
    X = rng.normal(size=(D,1)); X -= U @ (U.T @ X); ud, _ = np.linalg.qr(X)
    Y = rng.normal(size=(D,1)); Y -= V @ (V.T @ Y); vd, _ = np.linalg.qr(Y)
    Wd = W + sv[0] * (ud @ vd.T)
    Ud = np.concatenate([U[:, :r-1], ud], axis=1)
    Vd = np.concatenate([V[:, :r-1], vd], axis=1)
    return Wd + N, N, Ud, Vd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--ranks", type=int, nargs="+", default=RANKS)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    print("="*104)
    print(f"Exp 17: attack and defences across datasets and methods "
          f"(sigma={SIGMA}, cheat={CHEAT}, q={Q}, alpha={ALPHA}, n={a.trials})")
    print("="*104)
    print(f"  {'dataset':<8} {'method':<11} {'rank':>4} {'pool':>5} "
          f"{'ATTACK evades':>14} {'DEFENCE catches':>16} {'DECOY evades':>13}")

    out = []
    for ds in DATASETS:
        for method in METHODS:
            for r in a.ranks:
                got = load_pool(ds, method, r, 60, rng)
                if not got: continue
                pool, navail = got
                t0 = time.time()
                h_tail, h_chal, s_tail, s_chal, d_en, d_chal, h_en = [],[],[],[],[],[],[]
                for _ in range(a.trials):
                    A,B = pool[rng.integers(len(pool))]; W = B@A
                    U,V,_ = basis(W, r)
                    # honest
                    N = rng.normal(0.0, SIGMA, size=W.shape)
                    h_tail.append(tail_sigma2(W+N, r))
                    h_chal.append(chal(U.T @ N @ V, Q, rng))
                    h_en.append(energy_ratio(W+N, U, V, r))
                    # calibrated shaping adversary
                    Ws, Ns, Us, Vs = shaped_calibrated(W, r, rng)
                    s_tail.append(tail_sigma2(Ws, r))
                    s_chal.append(chal(Us.T @ Ns @ Vs, Q, rng))
                    # decoy
                    Wd, Nd, Ud, Vd = decoy(W, r, rng)
                    d_en.append(energy_ratio(Wd, Ud, Vd, r))
                    d_chal.append(chal(Ud.T @ Nd @ Vd, Q, rng))
                h_tail,h_chal,s_tail,s_chal,d_en,d_chal,h_en = map(
                    np.array,(h_tail,h_chal,s_tail,s_chal,d_en,d_chal,h_en))

                thr_t = float(np.quantile(h_tail, ALPHA))
                thr_c = float(np.quantile(h_chal, ALPHA))
                thr_e = float(np.quantile(h_en,  ALPHA))
                atk_flag  = float(np.mean(s_tail < thr_t))     # low = evades
                hon_flag  = float(np.mean(h_tail < thr_t))
                def_power = float(np.mean(s_chal < thr_c))     # high = defence works
                dec_flag  = float(np.mean((d_en < thr_e) | (d_chal < thr_c)))

                out.append({"dataset":ds,"method":method,"rank":r,"adapters_available":navail,
                            "attack_flag":atk_flag,"honest_flag":hon_flag,
                            "defence_power":def_power,"decoy_combined_flag":dec_flag})
                print(f"  {ds:<8} {method:<11} {r:>4} {navail:>5} "
                      f"{atk_flag:>8.2f} (hon {hon_flag:.2f}) {def_power:>13.2f} "
                      f"{dec_flag:>13.2f}   [{time.time()-t0:.0f}s]")
            print()

    with open("results/exp17_cross_dataset_security.json","w") as f:
        json.dump({"config":{"sigma":SIGMA,"cheat":CHEAT,"q":Q,"alpha":ALPHA,
                             "trials":a.trials},"rows":out}, f, indent=2)
    A_ = [r["attack_flag"] for r in out]; Dp = [r["defence_power"] for r in out]
    Dc = [r["decoy_combined_flag"] for r in out]
    print(f"  attack flag rate  {min(A_):.2f} to {max(A_):.2f}   (exp7 on Yelp: 0.03 to 0.16)")
    print(f"  defence power     {min(Dp):.2f} to {max(Dp):.2f}   (exp12 on Yelp: 1.00)")
    print(f"  decoy flag rate   {min(Dc):.2f} to {max(Dc):.2f}   (exp13 on Yelp: 0.03 to 0.08)")
    print("\nSaved: results/exp17_cross_dataset_security.json")


if __name__ == "__main__":
    main()
