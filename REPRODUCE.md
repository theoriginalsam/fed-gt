# Reproducing the paper's numbers

Every table and figure of *Can a Server Verify a Client's Privacy Noise?* maps to one
command and one result file below. All audit experiments run on CPU from saved adapters;
only training (the last section) needs a GPU.

## Inputs

| input | what it is | where |
|---|---|---|
| `campaign/` | 15,120 query-projection adapters: 18 runs (Yelp, Alpaca, GSM8K x SPA, FlexLoRA x seeds 42/43/44), rounds 1, 6, 12, 10 clients per round, 28 layers | produced by the training code with `--save-adapters-dir` |
| `adapters/` | separate 20-round Yelp run: 3,360 query and 1,680 value adapters | same |
| `results/cache_exp26_singular_values.npz` | singular values of every campaign adapter (lets exp26 run without `campaign/`) | in this repository |

Adapters are stored per LoRA module as float32 `lora_A` / `lora_B` (fp16 would bias the tail singular values).

## Tables and figures

Tables are named by caption because their numbers depend on the build.

| paper item | command | result file |
|---|---|---|
| Fig. spectra of real adapters | `python experiments/exp16_cross_dataset_spectra.py` | `results/exp16_cross_dataset_spectra.json` (+ `campaign/`) |
| Table "Tail estimator on real adapters" | `python experiments/exp5b_real_kill_test.py` | `results/exp5_real_kill_test.json` |
| Table "Shaping adversary vs. the tail estimator" | `python experiments/exp7_calibrated_cheater.py` | `results/exp7_calibrated_cheater.json` |
| Fig. all 24 settings (tail test, challenge, decoy) | `python experiments/exp17_cross_dataset_security.py` | `results/exp17_cross_dataset_security.json` |
| "Other statistics of the update" (spike consistency, drift) | `python experiments/exp8_spike_consistency.py`, `exp9_temporal_defense.py`, `exp10_drift_trajectory.py` | `results/exp8_*.json`, `exp9_*.json`, `exp10_*.json` |
| Fig. declared vs. estimated subspace, partial cheater | `python experiments/exp11_commit_challenge.py`, `exp12_declared_subspace.py` | `results/exp11_*.json`, `exp12_*.json` |
| Fig. visibility | `python experiments/exp15_sigma_sweep.py` | `results/exp15_sigma_sweep.json` |
| Table "Lying strategies against the energy check and the challenge" | `python experiments/exp13_declaration_check.py` | `results/exp13_declaration_check.json` |
| Fig. rank check against decoys | `python experiments/exp18_two_sided_check.py` | `results/exp18_two_sided_check.json` |
| Fig. detection limit, Eq. for audited rounds | `python experiments/exp19_cheat_level_sweep.py` | `results/exp19_cheat_level_sweep.json` |
| Fig. fine needed for honesty | `python experiments/exp20_incentive_frontier.py` | `results/exp20_incentive_frontier.json` |
| Table "False alarm rates ... held-out adapters" | `python experiments/exp24_heldout_calibration.py` | `results/exp24_heldout_calibration.json` |
| Table "Effective privacy of the relabelling cheater in the Gaussian model", Fig. regime map (b) | `python experiments/exp23_clipping_binding.py` | `results/exp23_clipping_binding.json` |
| Table "An informed auditor against a relabeller built from real adapters" | `bash run_on_gpu_machine_v3.sh` (CPU; cores clipped to norm 1, train / calibrate / test split of clients, 10 distinct module adapters pooled) | `results/exp26c_practical_relabel_{main,h0,h0_same,noiseonly}.json` (logs: `results/log_exp26c_*.txt`) |
| Earlier unclipped diagnostic of the same table (superseded) | `bash run_on_gpu_machine_v2.sh` (`--no-clip --split crossfit`) | `results/exp26b_practical_relabel_*.json` |
| Table "Checks at realistic privacy noise", Fig. regime map (a) | `python experiments/exp22_realistic_noise.py` and the transition sweep | `results/exp22_realistic_noise.json`, `results/exp22b_transition_sweep.json` |
| Value-projection results (limitations) | `python experiments/exp21_vproj_security.py --module v_proj --ranks 4 8 --adapters <rounds 1-6 of the 20-round run>` | `results/exp21_v_proj_security.json` (log: `results/log_exp21_v_proj.txt`) |
| Fig. audit game with exit, robustness | `python experiments/exp25_audit_game.py` | `results/exp25_audit_game.json` |
| Relabeller detection-rate sensitivity | `python experiments/exp25_audit_game.py --relabel-sensitivity` | `results/exp25_relabel_sensitivity.json` |
| Table "Attacks on the sealed component" | `python -m pytest tests/test_sealed_noise.py` | test output (19 tests) |
| Sealed component cost | `python experiments/exp27_sealed_cost.py` | `results/exp27_sealed_cost.json` |
| Fig. training with enforced noise | training code (below) | `results/sealed_training/` |

Exact privacy levels are computed with `fedgt/privacy_calibration.py` (Balle and Wang's exact
curve, replace-one sensitivity 2C). Figures are drawn by `figures/make_figures.py` from the result
files above.

## Training with enforced noise (GPU)

Repository `fed-llm-rework`, branch `algo/ablations`; see `SEALED_GPU_RUNS.md` there.

```
python experiments/run_sealed_utility.py --dataset yelp --seed 42 --conditions base none clip
python experiments/run_sealed_utility.py --dataset yelp --seed 42 --conditions 1e4d 1e5d 1e6d 1e7d 1e8d --num-rounds 1 --results-dir results_sealed_probe
python experiments/run_sealed_utility.py --dataset yelp --seed 42 --conditions 1e6d 1e7d 1e7
for sd in 43 44; do python experiments/run_sealed_utility.py --dataset yelp --seed $sd --conditions clip 1e4d 1e5d 1e6d 1e7d 1e8d --num-rounds 1 --results-dir results_sealed_probe; done
```

Configuration: Qwen2.5-7B-Instruct (bf16), LoRA on q_proj and v_proj of all 28 layers, ranks
4/8/16/32 for 20/20/5/5 of 50 clients, scaling 2r, no dropout; Dirichlet label partition with
concentration 0.5; 5 clients per round, 20 rounds, 100 local AdamW steps (lr 2e-4, batch 4,
gradient accumulation 4), sequence length 256; FlexLoRA aggregation weighted by rank x data size.
Joint clip C = 1.5 over all 56 modules; noise calibrated per release with sensitivity 2C,
delta 1e-5. Accuracy: rating token after the prompt, 5,000 test reviews. Perplexity: 200 test
prompts without the rating.
