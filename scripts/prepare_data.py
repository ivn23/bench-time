import polars as pl
import pickle 
from data_functions import create_calendric_features, add_lag_features, add_trend_feature



# Define file paths
mapping_path = '../data/feature_mapping_train.pkl'
features_path = '../data/train_data_features.feather'
target_path = '../data/train_data_target.feather'

# Load the mapping (pickle file)
with open(mapping_path, 'rb') as f:
    mapping = pickle.load(f)

# Load train_features and train_target from Feather files
train_features = pl.read_ipc(features_path)  # Polars uses `read_ipc` for Feather files
train_target = pl.read_ipc(target_path)

# Convert the mapping (dictionary or list) to a Polars DataFrame
feature_mapping = pl.DataFrame(mapping)



df = create_calendric_features(train_features, 'date')
df = df.to_dummies(
    columns=["day_of_week", "month", "quarter", "week_of_year", "year", "is_weekend"]
)

df = add_lag_features(
    df,
    lags=range(1, 8),
    group_by_cols=["skuID", "frequency"],
    value_col="feature_0038",
    date_col="date"
)


df = df.drop('lag_target_1','feature_0038')
df = df.filter(pl.col("not_for_sale") != 1)

df = add_trend_feature(df, date_col="date")


