#!/usr/bin/env bash
# ==============================================================================
# CC-SAT-DA Benchmark Reproduction Script
# Runs evaluations across target sizes: N = 50, 100, 200, 1608
# ==============================================================================
set -e

DATA_DIR=${1:-"./data"}
BACKBONE=${2:-"efficientnet_b2"}

echo "=========================================================="
echo "Reproducing CC-SAT-DA Benchmarks with Backbone: $BACKBONE"
echo "Dataset directory: $DATA_DIR"
echo "=========================================================="

for N in 50 100 200 1608; do
  echo ""
  echo ">>> Training CC-SAT-DA (N = $N) ..."
  python3 train.py \
    --data_dir "$DATA_DIR" \
    --method cc_satda \
    --backbone "$BACKBONE" \
    --target_n "$N" \
    --epochs 30 \
    --batch_size 32 \
    --lr 1e-4 \
    --lambda_proto 0.2 \
    --lambda_disp 0.01 \
    --tau 0.35 \
    --seed 42 \
    --checkpoint_dir "./checkpoints"

  echo ">>> Evaluating on held-out test cohort (345 clinical images) ..."
  python3 evaluate.py \
    --checkpoint "./checkpoints/cc_satda_${BACKBONE}_n${N}_best.pth" \
    --data_dir "$DATA_DIR" \
    --sweep_tau
done

echo ""
echo "=========================================================="
echo "Benchmark Reproduction Complete!"
echo "=========================================================="
