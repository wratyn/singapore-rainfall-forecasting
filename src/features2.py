import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# --- 1. PREPROCESSING & TIME ---


class TimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, add_sigmoid=True):
        self.add_sigmoid = add_sigmoid

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if not np.issubdtype(df["date"].dtype, np.datetime64):
            df["date"] = pd.to_datetime(df["date"])

        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day_of_week"] = df["date"].dt.dayofweek
        df["day_of_year"] = df["date"].dt.dayofyear
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

        df["season_intensity"] = 0.5 * df["doy_sin"] + 0.5 * df["doy_cos"]
        df["is_peak_rain_season"] = df["month"].isin([10, 11, 12, 1]).astype(int)

        if self.add_sigmoid:
            t = (df["date"] - df["date"].min()).dt.days
            t = t / (t.max() + 1e-6)
            df["time_sigmoid"] = 1 / (1 + np.exp(-10 * (t - 0.5)))

        return df


class StructuralWeatherImputer(BaseEstimator, TransformerMixin):
    """Mengisi nilai kosong sisa cleaning dengan Median Lokasi atau Global."""

    def __init__(self, columns, location_col="location"):
        self.location_col = location_col
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        self.loc_median_ = X.groupby(self.location_col).median(numeric_only=True)
        self.global_median_ = X.median(numeric_only=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col in self.columns:
            if col not in df.columns:
                continue
            df[col] = df[col].fillna(df[self.location_col].map(self.loc_median_[col]))
            df[col] = df[col].fillna(self.global_median_[col])
        return df


# --- 2. PHYSICS FEATURES ---


class TemperatureFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, encode_as_int=True):
        self.encode_as_int = encode_as_int

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["temp_range_c"] = df["maximum_temperature_c"] - df["minimum_temperature_c"]
        df["temp_asymmetry"] = (
            df["mean_temperature_c"] - df["minimum_temperature_c"]
        ) - (df["maximum_temperature_c"] - df["mean_temperature_c"])
        return df


class WindRainFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        gust = df["max_wind_speed_kmh"] / (df["mean_wind_speed_kmh"] + 1e-6)
        df["wind_gust_factor"] = gust.fillna(1)

        df["wind_x_rain"] = df["mean_wind_speed_kmh"] * df["highest_60_min_rainfall_mm"]
        return df


# --- 3. TIME SERIES & SPATIAL FEATURES (CRITICAL) ---


class NeighborRainFeatures(BaseEstimator, TransformerMixin):
    """Fitur Spasial: Mengambil statistik hujan dari lokasi tetangga."""

    def __init__(self, neighbors: dict, feature_col: str, lags=(1,)):
        self.neighbors = neighbors
        self.feature_col = feature_col
        self.lags = lags

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(f"   [Feature] Generating Neighbor Features for {self.feature_col}...")
        df = X.copy()
        df["date"] = pd.to_datetime(df["date"])

        piv = df.pivot_table(index="date", columns="location", values=self.feature_col)

        def get_neighbor_mean(date_idx, loc):
            neighs = self.neighbors.get(loc, [])
            valid_neighs = [n for n in neighs if n in piv.columns]
            if not valid_neighs:
                return np.nan
            return piv.loc[date_idx, valid_neighs].mean()

        neighbor_means_df = pd.DataFrame(index=piv.index, columns=piv.columns)

        for loc, neighs in self.neighbors.items():
            valid_neighs = [n for n in neighs if n in piv.columns]
            if valid_neighs:
                neighbor_means_df[loc] = piv[valid_neighs].mean(axis=1)
            else:
                neighbor_means_df[loc] = 0.0

        neighbor_means_long = neighbor_means_df.stack().reset_index()
        neighbor_means_long.columns = ["date", "location", "nbr_mean_raw"]

        df = df.merge(neighbor_means_long, on=["date", "location"], how="left")

        for lag in self.lags:
            df[f"nbr_{self.feature_col}_mean_lag{lag}"] = df.groupby("location")[
                "nbr_mean_raw"
            ].shift(lag)

        return df.drop(columns=["nbr_mean_raw"])


class LagRollingFeatures(BaseEstimator, TransformerMixin):
    """Fitur Temporal: Lag dan Rolling Window."""

    def __init__(self, specs, group_col="location", date_col="date"):
        self.specs = specs
        self.group_col = group_col
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print("   [Feature] Generating Lag & Rolling Features...")
        X = X.copy()
        X[self.date_col] = pd.to_datetime(X[self.date_col])
        X = X.sort_values([self.group_col, self.date_col])

        for col, lags in self.specs:
            if col not in X.columns:
                continue

            grp = X.groupby(self.group_col)[col]

            for lag in lags:
                # Lag Feature
                X[f"{col}_lag{lag}"] = grp.shift(lag)

            if "rainfall" in col:
                X[f"{col}_mean_7d"] = grp.shift(1).rolling(7).mean()
                X[f"{col}_mean_30d"] = grp.shift(1).rolling(30).mean()

        return X


class LocationMonthTargetEncoder(BaseEstimator, TransformerMixin):
    """Target Encoding: Rata-rata target per Lokasi & Bulan."""

    def __init__(self, target_col="daily_rainfall_total_mm"):
        self.target_col = target_col
        self.map_ = {}
        self.global_mean_ = 0

    def fit(self, X, y=None):
        df = X.copy()
        self.global_mean_ = df[self.target_col].mean()
        self.map_ = df.groupby(["location", "month"])[self.target_col].mean().to_dict()
        return self

    def transform(self, X):
        df = X.copy()

        # Map values
        def get_val(row):
            return self.map_.get((row["location"], row["month"]), self.global_mean_)

        keys = list(zip(df["location"], df["month"]))
        df["loc_month_avg_target"] = [self.map_.get(k, self.global_mean_) for k in keys]
        return df
