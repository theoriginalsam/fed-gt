#!/usr/bin/env python3
"""Exp 5, part A: characterize the spectra of REAL trained SPA adapters.

For each rank, sample real adapters, compute the exact nonzero singular
values of Delta_W = B @ A (via the cheap r x r route), and report:
  - the signal scale (per-entry RMS of Delta_W),
  - the decay depth s_r / s_1,
  - a power-law decay exponent gamma (s_i ~ i^-gamma),
  - for a few candidate contract noise levels sigma, whether the smallest
    signal singular value s_r sits above the Marchenko-Pastur noise floor
    sigma * (sqrt(d1) + sqrt(d2)). Signal below the floor is the regime
    that leaks into the tail and biases the auditor.

CPU-only. Uses the real adapters unpacked in ./adapters/.
"""
import glob
import re
import numpy as np
import torch

ADIR = "adapters"
RANKS = [4, 8, 16, 32]
N_PER_RANK = 40          # adapters sampled per rank
SIGMA_FRACS = [0.5, 1.0, 2.0]   # sigma as multiples of per-entry RMS of Delta_W


def signal_svals(A, B):
    """Exact nonzero singular values of B@A via QR on the thin factors."""
    # B: (d, r), A: (r, d)
    Qb, Rb = np.linalg.qr(B)          # Qb (d,r), Rb (r,r)
    Qa, Ra = np.linalg.qr(A.T)        # A.T (d,r) -> Qa (d,r), Ra (r,r)
    mid = Rb @ Ra.T                   # (r,r), same singular values as B@A
    return np.linalg.svd(mid, compute_uv=False)


def gamma_fit(s):
    """Power-law exponent: slope of log s_i vs log i (i=1..r)."""
    if len(s) < 2:
        return float("nan")
    i = np.arange(1, len(s) + 1)
    x, y = np.log(i), np.log(s + 1e-30)
    return float(-np.polyfit(x, y, 1)[0])


def main():
    rng = np.random.default_rng(0)
    print("=" * 78)
    print("Exp 5A: real SPA adapter spectra (Qwen2.5-7B, q/v_proj, d=3584)")
    print("=" * 78)
    for r in RANKS:
        files = sorted(glob.glob(f"{ADIR}/*_r{r}__*.pt"))
        if not files:
            print(f"rank {r}: no files"); continue
        pick = rng.choice(len(files), size=min(N_PER_RANK, len(files)),
                          replace=False)
        s1s, srs, ratios, gammas, rms, floor_ok = [], [], [], [], [], {f: 0 for f in SIGMA_FRACS}
        d1 = d2 = 3584
        for idx in pick:
            st = torch.load(files[idx], map_location="cpu")
            A = st["lora_A"].float().numpy().astype(np.float64)
            B = st["lora_B"].float().numpy().astype(np.float64)
            s = signal_svals(A, B)
            s = np.sort(s)[::-1]
            s = s[s > 1e-12]
            if len(s) < 2:
                continue
            # per-entry RMS of Delta_W: ||BA||_F / sqrt(d1 d2) = ||s||_2 / sqrt(d1 d2)
            rms_i = np.linalg.norm(s) / np.sqrt(d1 * d2)
            s1s.append(s[0]); srs.append(s[-1]); ratios.append(s[-1] / s[0])
            gammas.append(gamma_fit(s)); rms.append(rms_i)
            for frac in SIGMA_FRACS:
                sigma = frac * rms_i
                floor = sigma * (np.sqrt(d1) + np.sqrt(d2))
                floor_ok[frac] += int(s[-1] > floor)
        n = len(s1s)
        print(f"\nrank {r}  (n={n} adapters)")
        print(f"  per-entry RMS of DeltaW : {np.mean(rms):.4e}  "
              f"(so a DP sigma is naturally O(1e-3..1e-2))")
        print(f"  top singular value s_1  : {np.mean(s1s):.4e}")
        print(f"  smallest signal s_r     : {np.mean(srs):.4e}")
        print(f"  decay depth s_r / s_1   : {np.mean(ratios):.4f}  "
              f"(1.0=flat, ->0 = steep decay)")
        print(f"  power-law gamma         : {np.mean(gammas):.2f}")
        for frac in SIGMA_FRACS:
            print(f"  sigma={frac:>3}xRMS: signal s_r above noise floor in "
                  f"{floor_ok[frac]}/{n} adapters")


if __name__ == "__main__":
    main()
