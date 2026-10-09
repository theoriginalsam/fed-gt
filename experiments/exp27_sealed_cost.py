#!/usr/bin/env python3
"""Exp 27: cost of the sealed-noise component at the real model size.

Qwen2.5-7B with LoRA on q_proj and v_proj: 28 layers, q_proj 3584 x 3584 and
v_proj 512 x 3584, so 56 modules and 411M entries per client per round. One
release (validate, clip, draw noise, hash, sign, encrypt) and one server
acceptance (decrypt, hash, verify) are timed, with peak memory and bytes sent.

Runs in software on a laptop CPU; inside a TEE, encryption and noise drawing
would run in the enclave, so these are lower bounds on enclave cost only in the
sense of excluding enclave overheads (memory encryption, paging).
"""
import argparse, json, pathlib, platform, resource, sys, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fedgt.sealed_noise import BoxConfig, SealedNoiseBox, Server, Vendor, measurement_of
from fedgt.privacy_calibration import sigma_of_eps


def peak_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2 ** 30 if platform.system() == "Darwin" else r / 2 ** 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    shapes = []
    for L in range(args.layers):
        shapes += [(f"l{L}.q", (3584, 3584)), (f"l{L}.v", (512, 3584))]
    shapes = tuple(shapes)
    n = sum(a * b for _, (a, b) in shapes)
    C = 1.0
    sigma = sigma_of_eps(8.0, 2 * C)          # eps = 8 for one release
    cfg = BoxConfig(clip=C, sigma=sigma, eps_budget=1e9, shapes=shapes)
    vendor = Vendor()
    server = Server(vendor.public_key, measurement_of(SealedNoiseBox), cfg)
    box = SealedNoiseBox(cfg, 1, vendor, server.sign_pk, server.enc_pk)
    server.enrol(1, box.certificate)
    rng = np.random.default_rng(0)
    upd = {name: rng.normal(0, 1e-3, s).astype(np.float32) for name, s in shapes}
    print(f"{len(shapes)} modules, {n/1e6:.1f}M entries, input {n*4/2**30:.2f} GiB float32; "
          f"peak so far {peak_gb():.2f} GiB", flush=True)

    # component costs on one module type, scaled to the full update
    q = upd[shapes[0][0]]
    t = time.perf_counter(); float(np.sum(np.square(q, dtype=np.float64))); t_norm = time.perf_counter() - t
    g = np.random.Generator(np.random.Philox(key=1))
    t = time.perf_counter(); g.normal(0, 1.0, q.size); t_noise = time.perf_counter() - t

    rel, acc, sent = [], [], []
    for rep in range(args.reps):
        t = time.perf_counter()
        ct = box.release(server.nonce(1, rep + 1), upd)
        rel.append(time.perf_counter() - t)
        sent.append(sum(len(c) for c in ct))
        t = time.perf_counter()
        out = server.receive(1, ct)
        acc.append(time.perf_counter() - t)
        del ct, out
        print(f"  rep {rep}: release {rel[-1]:.1f}s  accept {acc[-1]:.1f}s  sent {sent[-1]/2**30:.3f} GiB  "
              f"peak {peak_gb():.2f} GiB", flush=True)
    res = {"modules": len(shapes), "entries": n, "input_gib": n * 4 / 2 ** 30,
           "release_s": rel, "accept_s": acc, "sent_bytes": sent[0],
           "norm_pass_s_full": t_norm * n / q.size,
           "noise_draw_s_full": t_noise * n / q.size,
           "peak_rss_gib": peak_gb(), "cpu": platform.processor() or platform.machine(),
           "sigma": sigma, "eps_per_release": 8.0, "lora_rank8_factor_bytes": 28 * 8 * (3584 + 3584 + 512 + 3584) * 4}
    print(json.dumps({k: v for k, v in res.items()}, indent=1))
    json.dump(res, open(ROOT / "results" / "exp27_sealed_cost.json", "w"), indent=1)


if __name__ == "__main__":
    main()
