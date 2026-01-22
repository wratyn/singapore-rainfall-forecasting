import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from src.cleaning import clean_data
from src.features2 import (
    TimeFeatures,
    StructuralWeatherImputer,
    TemperatureFeatures,
    WindRainFeatures,
    NeighborRainFeatures,
    LagRollingFeatures,
    LocationMonthTargetEncoder,
)
from src.dataset import load_raw_train_data
from src.config import PROCESSED_DIR

target = "daily_rainfall_total_mm"
OUTPUT_PATH = PROCESSED_DIR / "processed_data.csv"


def get_correlation_neighbors(df, target_col, k=5, min_corr=0.3):
    piv = df.pivot_table(index="date", columns="location", values=target_col)
    corr_matrix = piv.corr()

    neighbors = {}
    for loc in corr_matrix.columns:
        s = corr_matrix[loc].drop(index=loc).dropna()
        s = s[s > min_corr]
        neighbors[loc] = s.sort_values(ascending=False).head(k).index.tolist()

    return neighbors


def main():
    df_raw = load_raw_train_data()
    df_clean = clean_data(df_raw)

    neighbor_dict = get_correlation_neighbors(
        df_clean, target_col=target, k=5, min_corr=0.3
    )

    processing_pipeline = Pipeline(
        [
            ("time_feats", TimeFeatures(add_sigmoid=True)),
            (
                "imputer",
                StructuralWeatherImputer(
                    columns=["mean_temperature_c", "mean_wind_speed_kmh"],
                    location_col="location",
                ),
            ),
            ("temp_phys", TemperatureFeatures()),
            ("wind_phys", WindRainFeatures()),
            (
                "neighbor_lag",
                NeighborRainFeatures(
                    neighbors=neighbor_dict, feature_col=target, lags=[1, 2, 3]
                ),
            ),
            (
                "temporal_lag",
                LagRollingFeatures(
                    specs=[
                        (target, [1, 2, 3, 7, 14, 28]),  # Target Lag
                        ("mean_temperature_c", [1, 2, 3]),  # Weather Lag
                        ("mean_wind_speed_kmh", [1, 2, 3]),
                    ],
                    group_col="location",
                    date_col="date",
                ),
            ),
            ("target_enc", LocationMonthTargetEncoder(target_col=target)),
        ]
    )

    df_processed = processing_pipeline.fit_transform(df_clean)

    initial_len = len(df_processed)
    lag_col_check = f"{target}_lag28"
    if lag_col_check in df_processed.columns:
        df_processed = df_processed.dropna(subset=[lag_col_check])

    df_processed = df_processed.dropna(subset=[target])

    print(
        f"   -> Menghapus {initial_len - len(df_processed)} baris (Warm-up Lag & Empty Targets)."
    )

    print(f"Saving....")
    df_processed.to_csv(OUTPUT_PATH, index=False)

    print(f"done. Shape: {df_processed.shape}")
