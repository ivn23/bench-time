import polars as pl
import typing as t


def create_calendric_features(df: pl.DataFrame, date_column: str) -> pl.DataFrame:
    """
    Create calendric features for a given Polars DataFrame with a date column.

    Parameters:
        df (pl.DataFrame): Input Polars DataFrame containing a date column.
        date_column (str): Name of the column containing dates.

    Returns:
        pl.DataFrame: Polars DataFrame with added calendric features.
    """
    # Ensure the date column is in datetime format
    df = df.with_columns(pl.col(date_column).cast(pl.Date).alias(date_column))

    # Create basic calendric features
    df = df.with_columns([
        pl.col(date_column).dt.month().alias("month"),
        (pl.col(date_column).dt.weekday()).alias("day_of_week"),  # Monday=1, ..., Sunday=7
        (pl.col(date_column).dt.strftime("%V").cast(pl.Int32)).alias("week_of_year"),  # ISO week number
        (pl.col(date_column).dt.year()).alias("year"),
        (pl.col(date_column).dt.weekday() >= 5).alias("is_weekend")  # True for Saturday/Sunday
    ])

    # Create the "quarter" column based on the "month" column
    df = df.with_columns(((pl.col("month") - 1) // 3 + 1).alias("quarter"))

    return df

def add_lag_features(
    df: pl.DataFrame,
    lags: t.Union[int, t.List[int], range] = 1,
    group_by_cols: t.List[str] = None,
    value_col: str = "value",
    date_col: str = "date",
    
) -> pl.DataFrame:
    """
    Add lag features to a DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing time series data.
    value_col : str, default "value"
        Name of the column to create lag features for.
    group_by_cols : List[str], optional
        List of columns to group by when creating lags. If None, defaults to 
        ["skuID", "frequency"] if both exist, otherwise ["skuID"].
    date_col : str, default "date"
        Name of the date column used for sorting.
    lags : Union[int, List[int], range], optional
        Lag periods to create. If int, creates lags from 1 to that number.
        If List[int] or range, creates lags for those specific values.
        If None, defaults to range(1, 8).

    Returns
    -------
    pl.DataFrame
        DataFrame with added lag columns named "{value_col}_lag_{lag}".
    """
    
    # Handle default parameters
    if group_by_cols is None:
        group_by_cols = ["skuID", "frequency"]
    
    sort_cols = group_by_cols + [date_col]
    
    # Validate that required columns exist
    missing_cols = [col for col in sort_cols + [value_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
    
    # Sort the DataFrame
    df_sorted = df.sort(sort_cols)
    
    # Create lag features
    lag_features = [
        pl.col(value_col).shift(lag).over(group_by_cols).alias(f"{value_col}_lag_{lag}")
        for lag in lags
    ]
    
    # Add lag columns to the DataFrame
    result = df_sorted.with_columns(lag_features)
    
    return result

def add_trend_feature(df: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
    """
    Adds a 'trend' feature to the DataFrame, counting days from the earliest to the latest date.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing a date column.
    date_col : str, default "date"
        Name of the date column.

    Returns
    -------
    pl.DataFrame
        DataFrame with a new 'trend' column joined on the date.
    """
    earliest = df[date_col].min()
    latest = df[date_col].max()

    # Create a date range from earliest to latest (inclusive)
    date_range = pl.date_range(earliest, latest, "1d", eager=True)

    # Create the trend column: count from 1 to N
    trend = pl.int_range(1, len(date_range) + 1, eager=True)

    # Create the new DataFrame
    result = pl.DataFrame({
        date_col: date_range,
        "trend": trend
    })

    # Join the trend column to the original DataFrame
    df = df.join(result, on=date_col, how="left")
    return df