import pandas as pd
import numpy as np


def remove_no_signal(df, features):
    mask_all_nan = df[features].isna().all(axis=1)
    df_clean = df.loc[~mask_all_nan].reset_index(drop=True)
    return df_clean


def clean_data(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    core_features = [
        "mean_temperature_c",
        "maximum_temperature_c",
        "minimum_temperature_c",
        "mean_wind_speed_kmh",
        "max_wind_speed_kmh",
    ]
    event_features = [
        "highest_30_min_rainfall_mm",
        "highest_60_min_rainfall_mm",
        "highest_120_min_rainfall_mm",
    ]
    target = "daily_rainfall_total_mm"

    all_features = core_features + event_features + [target]

    # hapus baris no signals
    initial_len = len(df)
    cols_to_check = [c for c in all_features if c in df.columns]
    df = remove_no_signal(df, cols_to_check)
    print(f"   [Cleaning] Menghapus {initial_len - len(df)} baris 'No Signal'")

    print("   [Cleaning] Mengisi tanggal yang hilang (Resampling per Lokasi)...")
    df = (
        df.set_index("date")
        .groupby("location")
        .resample("D")
        .asfreq()
        # .drop(columns=["location"])  # drop lokasi duplikat
        .reset_index()
    )

    print("   [Cleaning] Mengisi Event Features dengan 0.0...")
    for col in event_features:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # imputasi dengan pendekatan klimatologi

    print(
        "   [Cleaning] Mengisi Core Features (Suhu/Angin) dengan pendekatan Klimatologi..."
    )
    df["temp_month"] = df["date"].dt.month

    for col in core_features:
        if col not in df.columns:
            continue

        df[col] = df.groupby("location")[col].transform(
            lambda x: x.interpolate(method="linear", limit=3)
        )

        monthly_means = df.groupby(["location", "temp_month"])[col].transform("mean")
        df[col] = df[col].fillna(monthly_means)

        df[col] = df[col].fillna(df[col].mean())

    df = df.drop(columns=["temp_month"])

    final_cols = ["date", "location", target] + core_features + event_features
    existing_final_cols = [c for c in final_cols if c in df.columns]
    df = df[existing_final_cols]

    print(f"   [Cleaning] Done. Shape: {df.shape}")
    return df
