# FedGT — Audit the Noise

Verifiable local differential privacy for federated LoRA fine-tuning:
a **spectral audit** (reads the injected noise level from the tail singular
values of a LoRA update) + an **audit game** (rewards/penalties that make
honest noise injection every client's best strategy).

**Everything in this repo is CPU-only.** No GPU, no torch, no LLM needed for
the kill test, the theory experiments, or the strategic simulations. That is
by design: the core science is statistics + game theory on small matrices.

## Setup (laptop or any server, CPU is fine)

```bash
cd FedGT
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # just numpy, scipy, pytest
```

## 1. Run the tests (sanity check, ~1–2 min)

```bash
pytest tests/ -v
```

## 2. Run the KILL TEST (the go/no-go experiment, ~2–5 min)

```bash
python run_kill_test.py          # fast: 256x256 rank-8
python run_kill_test.py --big    # closer to real LoRA: 1024x1024 rank-16
```

PASS = a client injecting 50% less noise than contracted is caught with
power ≥ 0.9 within ≤ 10 audited rounds, while honest clients are flagged
at ≤ alpha. Results land in `results/kill_test.json`.

## 3. Run the experiments

```bash
python experiments/exp1_noise_estimation.py   # estimator accuracy vs rank/noise
python experiments/exp2_detection_power.py    # power curves (Result 1 table)
python experiments/exp3_strategic_sim.py      # IC check, minimal audit rate,
                                              # best-response dynamics, end-to-end sim
```

## Layout

```
fedgt/
  config.py          # all dataclass configs
  lora_update.py     # synthetic LoRA updates + Gaussian mechanism + real-adapter loader
  spectral_audit.py  # THE DETECTOR: tail-spectrum sigma estimator + calibrated test
  mechanism.py       # THE GAME: IC condition, minimal audit prob, best-response dynamics
  simulation.py      # multi-round FL harness (clients, random audits, payoffs)
experiments/         # exp1–exp3 (paper figures come from these)
tests/               # pytest suite
run_kill_test.py     # the two-week go/no-go experiment
```

## Plugging in your real SPA / HetLoRA-M adapters (later, with GPU)

The simulator calls `fedgt.lora_update.client_round_update` to get each
round's update. To audit **real** adapters instead:

1. In your SPA training loop, save each client's per-round adapter, e.g.
   `torch.save({"delta_w": (B @ A).cpu()}, f"round{t}_client{i}.pt")`
   — or save the PEFT state dict directly (keys containing `lora_A`/`lora_B`).
2. Load them here with `fedgt.load_real_adapter(path)` (needs CPU torch:
   `pip install torch --index-url https://download.pytorch.org/whl/cpu`).
3. Feed the list of matrices to `SpectralAuditor.audit_client(...)`.

The auditor itself never needs a GPU — one SVD of a LoRA-sized matrix is
milliseconds. Only the *training* that produces real adapters needs a GPU,
and that can wait.

## Honest caveats (write these in the paper too)

- The null threshold is calibrated by Monte Carlo **including** signal
  leakage into the tail, so the false-alarm guarantee is by construction —
  but calibration assumes the auditor knows the client's rank r (true for
  LoRA, where rank is part of the protocol).
- Detection power numbers here use synthetic low-rank signal; real trained
  adapters have decaying (not exactly-rank-r) spectra. That's exactly what
  the real-adapter phase must confirm — it is the honest version of the
  kill test, not a formality.
- The gain function `gain(c) = gain_scale * (1 - c^2)` is a modeling choice;
  Phase-3 work is to estimate it from real accuracy-vs-noise curves.
