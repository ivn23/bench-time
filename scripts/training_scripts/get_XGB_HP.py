import polars as pl

# Load clean data
data_path = "../../data/db_snapshot_offsite/train_data/processed/train_data_features.feather"
df_clean = pl.read_ipc(data_path)

# Get all unique SKU tuples (same as cell 17 but without .sample(3))
sku_tuples = [(d['productID'], d['storeID']) for d in df_clean.select(pl.col("productID"), pl.col("storeID")).unique().to_dicts()]

# Split into 10 equal batches
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






