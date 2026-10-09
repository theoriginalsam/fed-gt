#!/bin/bash
# FedGT jobs for the GPU machine. None of these need the GPU itself; they run on
# its CPU while the GPU trains. Run from the FedGT folder:  bash run_on_gpu_machine.sh
# Results land in results/ (send back results/exp26_*.json and results/exp27_sealed_cost.json).
set -e
cd "$(dirname "$0")"
pip install -q cryptography scikit-learn scipy numpy pytest

echo "== 1. attack tests for the sealed-noise box (about 1 minute)"
python -m pytest tests/test_sealed_noise.py -q

echo "== 2. cost of the box at full 7B size (needs about 4 GB RAM, a few minutes)"
python experiments/exp27_sealed_cost.py

echo "== 3. what gives the practical relabeller away (5 variants in parallel, uses the cached spectra)"
E=(python -u experiments/exp26_practical_relabel.py --eps 1 80 24000)
"${E[@]}" --h 0 --tag _h0                                  > results/log_exp26_h0.txt 2>&1 &
"${E[@]}" --h 0 --public same_seed --tag _h0_same          > results/log_exp26_h0_same.txt 2>&1 &
"${E[@]}" --feats no_sort --tag _nosort                    > results/log_exp26_nosort.txt 2>&1 &
"${E[@]}" --feats noise_only --tag _noiseonly              > results/log_exp26_noiseonly.txt 2>&1 &
"${E[@]}" --h 0 --public same_seed --feats no_sort --tag _h0_same_nosort > results/log_exp26_h0_same_nosort.txt 2>&1 &
wait
tail -n 13 results/log_exp26_*.txt
echo "== done"
