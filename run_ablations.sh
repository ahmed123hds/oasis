#!/bin/bash
# run_ablations.sh
#
# Usage:
#   source ~/Downloads/Documents/Work/Research/CVPR/pytorch_env/bin/activate
#   cd ~/Downloads/Documents/Work/Research/OASIS
#   bash run_ablations.sh 2>&1 | tee run_ablations_full.log
#
# The script expects to run INSIDE the activated environment.

set -e

EPOCHS=20
BS=256
BASE_ARGS="--epochs $EPOCHS --batch-size $BS"          # flags accepted by both scripts
OASIS_ARGS="$BASE_ARGS --min-numel 1024"               # OASIS-only extras

echo "=========================================================="
echo " OASIS Ablation Suite — Adaptive vs Fixed-Rank Tracking  "
echo "=========================================================="

# ── [A] Baseline (reference for all overhead calculations) ──────────────────
echo ">>> [A] Baseline"
python3 train_baseline.py $BASE_ARGS | tee log_baseline.txt

# ── [B] OASIS-Track (Adaptive, PRIMARY contribution) ────────────────────────
echo ">>> [B] OASIS-Track — Adaptive rank (energy_tau=0.97, r_max=72)"
python3 train_cifar10.py --mode track --r-max 72 --energy-tau 0.97 $OASIS_ARGS \
    | tee log_track_adaptive.txt

# ── [C] OASIS-Track — Fixed-rank ablation sweep ─────────────────────────────
# The observed adaptive mean rank is ~42. We test below, at, and above it.
# This ablation directly answers: "is adaptive rank selection necessary?"
#
# Expected outcome:
#   rank=20 → under-compress → accuracy OK, low grad savings
#   rank=42 → matched mean → similar to adaptive but less gradient savings
#   rank=72 → always max → high savings but suboptimal accuracy / cosine
#   adaptive → best balance of savings + fidelity

for FRANK in 20 42 72; do
    echo ">>> [C-${FRANK}] OASIS-Track — Fixed rank=${FRANK}"
    python3 train_cifar10.py \
        --mode track \
        --r-max 72 \
        --energy-tau 0.97 \
        --fixed-rank $FRANK \
        $OASIS_ARGS \
        | tee log_track_fixed_r${FRANK}.txt
done

# ── [D] 3-Seed robustness (Adaptive Track only) ──────────────────────────────
echo ">>> [D] 3-Seed robustness — Adaptive Track"
for SEED in 100 200 300; do
    python3 train_baseline.py     $BASE_ARGS --seed $SEED | tee log_baseline_seed${SEED}.txt
    python3 train_cifar10.py --mode track --r-max 72 --energy-tau 0.97 $OASIS_ARGS \
        --seed $SEED | tee log_track_adaptive_seed${SEED}.txt
done

# ── [E] Optimizer-aware vs energy criterion (exact mode, sanity check) ──────
echo ">>> [E] Optimizer-aware vs Energy criterion (Exact mode, K=1)"
python3 train_cifar10.py --mode exact --criterion optimizer_aware $OASIS_ARGS | tee log_exact_optaware.txt
python3 train_cifar10.py --mode exact --criterion energy          $OASIS_ARGS | tee log_exact_energy.txt

# ── [F] CIFAR-100 generalization ─────────────────────────────────────────────
echo ">>> [F] CIFAR-100 Baseline + OASIS-Track"
python3 train_baseline.py                                  $BASE_ARGS --dataset cifar100 | tee log_baseline_cifar100.txt
python3 train_cifar10.py --mode track --r-max 72 --energy-tau 0.97 $OASIS_ARGS \
    --dataset cifar100 | tee log_track_cifar100.txt

echo "=========================================================="
echo " Ablation Suite Complete! "
echo "=========================================================="

# ── Auto-generate final results table ────────────────────────────────────────
echo ""
echo ">>> Parsing results..."
python3 parse_results.py

