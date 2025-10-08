#!/bin/bash

# Distributed training orchestrator
# Usage: ./run_distributed_training.sh <model_type> <base_experiment_name>

MODEL_TYPE=${1:-xgboost_standard}
BASE_EXPERIMENT_NAME=${2:-distributed_experiment}

echo "Starting distributed training with model: $MODEL_TYPE"
echo "Base experiment name: $BASE_EXPERIMENT_NAME"

# Change to script directory
cd "$(dirname "$0")"

# Check if batch files exist
if [ ! -f "sku_batch_01.json" ]; then
    echo "Error: Batch files not found. Run create_sku_batches.py first."
    exit 1
fi

# Create output directory
mkdir -p ../../xgb_offsite

# Launch 10 parallel training processes
pids=()

for i in {1..10}; do
    batch_file="sku_batch_${i}.json"
    experiment_name="${BASE_EXPERIMENT_NAME}_${i}"

    echo "Launching batch $i: $experiment_name"

    python train_single_batch.py "$batch_file" "$MODEL_TYPE" "$experiment_name" > "log_batch_${i}.txt" 2>&1 &

    pids+=($!)
done

echo "All processes launched. PIDs: ${pids[@]}"
echo "Waiting for completion..."

# Wait for all processes to complete
for pid in "${pids[@]}"; do
    wait $pid
    echo "Process $pid completed"
done

echo "All training jobs completed!"
echo "Results available in ../../xgb_offsite/"