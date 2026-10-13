#!/bin/bash
# Second batch of FedGT CPU jobs for the GPU machine (they do not use the GPU).
# Run from the FedGT folder:  nohup bash run_on_gpu_machine_v2.sh > fedgt_jobs_v2.log 2>&1 &
# Re-evaluates the practical relabeller with client-disjoint training, threshold and scoring
# (k-fold over clients, leave-one-client-out thresholds, bootstrap intervals over clients).
cd "$(dirname "$0")"
E=(python -u experiments/exp26_practical_relabel.py --eps 1 80 24000 --no-clip --split crossfit)
"${E[@]}" --tag _main                                      > results/log_exp26b_main.txt 2>&1 &
"${E[@]}" --h 0 --tag _h0                                  > results/log_exp26b_h0.txt 2>&1 &
"${E[@]}" --h 0 --public same_seed --tag _h0_same          > results/log_exp26b_h0_same.txt 2>&1 &
"${E[@]}" --feats noise_only --tag _noiseonly              > results/log_exp26b_noiseonly.txt 2>&1 &
wait
tail -n 14 results/log_exp26b_*.txt
echo "== done"
