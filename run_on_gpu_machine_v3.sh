#!/bin/bash
# Third batch of FedGT CPU jobs for the GPU machine (they do not use the GPU).
# Run from the FedGT folder:  nohup bash run_on_gpu_machine_v3.sh > fedgt_jobs_v3.log 2>&1 &
# Re-runs the practical relabeller (Table 8) with honest cores clipped to norm 1, so the
# budgets are real, and a strict train / calibrate / test split of clients, so nothing
# that decides a test client saw its data. Pooling uses 10 distinct module adapters.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4
E=(python -u experiments/exp26_practical_relabel.py --eps 1 80 24000)
"${E[@]}" --tag _main                                      > results/log_exp26c_main.txt 2>&1 &
"${E[@]}" --h 0 --tag _h0                                  > results/log_exp26c_h0.txt 2>&1 &
"${E[@]}" --h 0 --public same_seed --tag _h0_same          > results/log_exp26c_h0_same.txt 2>&1 &
"${E[@]}" --feats noise_only --tag _noiseonly              > results/log_exp26c_noiseonly.txt 2>&1 &
wait
tail -n 14 results/log_exp26c_*.txt
echo "== done"
