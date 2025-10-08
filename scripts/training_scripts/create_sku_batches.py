import polars as pl
import json
import argparse
import subprocess
import sys

def create_batches():
    # Load clean data
    data_path = "../../data/db_snapshot_offsite/train_data/processed/train_data_features.feather"
    df_clean = pl.read_ipc(data_path)

    # Get all unique SKU tuples (same as cell 17 but without .sample(3))
    sku_tuples = [(d['productID'], d['storeID']) for d in df_clean.select(pl.col("productID"), pl.col("storeID")).unique().to_dicts()]
    sku_tuples = sku_tuples[:1000]  # Limit to first 100 for testing
    # Split into 10 equal batchesa
    total_skus = len(sku_tuples)
    batch_size = total_skus // 10
    remainder = total_skus % 10

    batches = []
    start = 0

    for i in range(10):
        # Add one extra SKU to first 'remainder' batches
        current_batch_size = batch_size + (1 if i < remainder else 0)
        end = start + current_batch_size
        batches.append(sku_tuples[start:end])
        start = end

    print(f"Total SKUs: {total_skus}")
    print(f"Batch sizes: {[len(batch) for batch in batches]}")

    # Save batches as JSON files
    for i, batch in enumerate(batches):
        filename = f"sku_batch_{i+1}.json"
        with open(filename, 'w') as f:
            json.dump(batch, f) 
        print(f"Saved {filename} with {len(batch)} SKUs")

    return batches

def main():
    parser = argparse.ArgumentParser(description="Create SKU batches and optionally trigger training")
    parser.add_argument("--trigger-training", action="store_true", help="Automatically start distributed training")
    parser.add_argument("--model-type", default="xgboost_quantile", help="Model type for training")
    parser.add_argument("--experiment-name", default="distributed_experiment", help="Base experiment name")

    args = parser.parse_args()

    # Create batches
    batches = create_batches()

    # Optionally trigger training
    if args.trigger_training:
        print(f"\nStarting distributed training with {args.model_type}...")
        try:
            subprocess.run([
                "./run_distributed_training.sh",
                args.model_type,
                args.experiment_name
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Training failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()