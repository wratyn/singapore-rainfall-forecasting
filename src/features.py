import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class StructuralWeatherImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, location_col="location"):
        self.location_col = location_col
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        self.loc_median_ = (
            X.groupby(self.location_col)
             .median(numeric_only=True)
        )

        self.global_median_ = X.median(numeric_only=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        for col in self.global_median_.index:
            if col not in df.columns:
                continue

            if col not in self.columns:
                continue

            df[col] = df[col].fillna(
                df[self.location_col].map(self.loc_median_[col])
            )

            df[col] = df[col].fillna(self.global_median_[col])

        return df


class AddExternalFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, external_df: pd.DataFrame, on="date"):
        self.external_df = external_df
        self.on = on

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()

        self.external_df[self.on] = pd.to_datetime(self.external_df[self.on])
        X_[self.on] = pd.to_datetime(X_[self.on])

        X_ = X_.merge(
            self.external_df,
            on=self.on,
            how="left",
            validate="m:1"
        )

        return X_


class LocationMedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, cols, location_col='location'):
        self.cols = cols
        self.location_col = location_col

    def fit(self, X, y=None):
        X_ = X[[self.location_col] + self.cols]
        self.medians_ = (
            X_
            .groupby(self.location_col)[self.cols]
            .median()
        )
        self.global_median_ = X_[self.cols].median()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].fillna(
                X[self.location_col].map(self.medians_[col])
            )

            X[col] = X[col].fillna(self.global_median_[col])
        return X


class TimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, add_sigmoid=True):
        self.add_sigmoid = add_sigmoid

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["date"] = pd.to_datetime(df["date"])

        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day_of_week"] = df["date"].dt.dayofweek
        df["day_of_year"] = df["date"].dt.dayofyear
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

        df["woy_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
        df["woy_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

        df["season"] = ((df["month"] % 12) // 3).astype(int)
        df["season_intensity"] = (
            0.5 * df["doy_sin"] +
            0.5 * df["doy_cos"]
        )
        df["is_peak_rain_season"] = df["month"].isin([10, 11, 12, 1]).astype(int)

        if self.add_sigmoid:
            t = (df["date"] - df["date"].min()).dt.days
            t = t / t.max()

            df["time_sigmoid"] = 1 / (1 + np.exp(-10 * (t - 0.5)))

        return df


class TemperatureFeatures(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        range_bins=(0, 5, 10, 15, 20, 100),
        mean_pos_bins=(0.0, 0.33, 0.66, 1.0),
        asy_bins=(-100, -2, 2, 100),
        encode_as_int=True,
    ):
        self.range_bins = range_bins
        self.mean_pos_bins = mean_pos_bins
        self.asy_bins = asy_bins
        self.encode_as_int = encode_as_int

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        max_t = df["maximum_temperature_c"]
        min_t = df["minimum_temperature_c"]
        mean_t = df["mean_temperature_c"]

        df["temp_range_c"] = max_t - min_t
        df["temp_mean_minus_min"] = mean_t - min_t
        df["temp_max_minus_mean"] = max_t - mean_t

        df["temp_mean_position"] = (
            (mean_t - min_t) / (df["temp_range_c"] + 1e-6)
        )

        df["temp_asymmetry"] = (
            df["temp_mean_minus_min"] - df["temp_max_minus_mean"]
        )

        df["temp_range_bin"] = pd.cut(
            df["temp_range_c"],
            bins=self.range_bins,
            labels=False,
            include_lowest=True,
        )

        df["temp_mean_position_bin"] = pd.cut(
            df["temp_mean_position"],
            bins=self.mean_pos_bins,
            labels=False,
            include_lowest=True,
        )

        df["temp_asymmetry_bin"] = pd.cut(
            df["temp_asymmetry"],
            bins=self.asy_bins,
            labels=False,
            include_lowest=True,
        )

        if self.encode_as_int:
            for c in [
                "temp_range_bin",
                "temp_mean_position_bin",
                "temp_asymmetry_bin",
            ]:
                df[c] = df[c].astype("Int64")

        return df


class WindRainFeatures(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        rain_bins=(0, 0.1, 1, 5, 10, 20, 50, 300),
        encode_as_int=True,
    ):
        self.rain_bins = rain_bins
        self.encode_as_int = encode_as_int

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        gust = df["max_wind_speed_kmh"] / df["mean_wind_speed_kmh"]
        df["wind_gust_factor"] = gust.replace(
            [np.inf, -np.inf], np.nan
        ).fillna(1)

        ratio = (
            df["highest_60_min_rainfall_mm"]
            / df["highest_30_min_rainfall_mm"]
        )
        df["rain_intensity_ratio"] = ratio.replace(
            [np.inf, -np.inf], np.nan
        ).fillna(1)

        for col in [
            "highest_30_min_rainfall_mm",
            "highest_60_min_rainfall_mm",
            "highest_120_min_rainfall_mm",
        ]:
            bin_col = f"{col}_bin"

            df[bin_col] = pd.cut(
                df[col],
                bins=self.rain_bins,
                labels=False,
                include_lowest=True,
            )

            if self.encode_as_int:
                df[bin_col] = df[bin_col].astype("Int64")

        return df


class LagFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, lag_days: int = 1):
        self.lag_days = lag_days

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        for col in [
            "mean_temperature_c",
            "highest_60_min_rainfall_mm",
            "mean_wind_speed_kmh",
        ]:
            df[f"{col}_lag{self.lag_days}"] = df[col].shift(self.lag_days)
            df[f"{col}_lag{self.lag_days}"] = df[f"{col}_lag{self.lag_days}"].fillna(df[col])

        return df


class RollingStatsFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        df["mean_temp_roll_7d"] = (
            df["mean_temperature_c"].rolling(7, min_periods=1).mean()
        )
        df["max_rain_roll_3d"] = (
            df["highest_60_min_rainfall_mm"].rolling(3, min_periods=1).max()
        )
        df["mean_wind_roll_7d"] = (
            df["mean_wind_speed_kmh"].rolling(7, min_periods=1).mean()
        )

        return df


class CyclicalInteractionFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        df["wind_x_rain"] = (
            df["mean_wind_speed_kmh"]
            * df["highest_60_min_rainfall_mm"]
        )

        try:
            df["ONI_x_temp"] = df["ONI"] * df["mean_temperature_c"]
            df["DMI_x_rainfall"] = (
                df["DMI"] * df["highest_60_min_rainfall_mm"]
            )
            df["heat_index_proxy"] = (
                df["RH"] * df["mean_temperature_c"]
            )
            df["aqi_x_temp_range"] = (
                df["AQI"] * df["temp_range_c"]
            )
        except:
            pass

        return df

class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, cols: list[str]):
        self.cols = cols

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        return df.drop(columns=self.cols, errors="ignore")


class DebugTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, name):
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(self.name, type(X), getattr(X, "shape", None))
        return X


class NeighborRainFeatures(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        neighbors: dict,
        feature_col: str,
        lags=(0, 1)
    ):
        self.neighbors = neighbors
        self.feature_col = feature_col
        self.lags = lags

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["date"] = pd.to_datetime(df["date"])

        value_map = (
            df
            .set_index(["location", "date"])[self.feature_col]
        )

        for lag in self.lags:
            means = []
            maxs = []

            for idx, row in df.iterrows():
                loc = row["location"]
                date = row["date"] - pd.Timedelta(days=lag)

                neighs = self.neighbors.get(loc, [])
                vals = []

                for n in neighs:
                    key = (n, date)
                    if key in value_map:
                        vals.append(value_map[key])

                if vals:
                    means.append(np.mean(vals))
                    maxs.append(np.max(vals))
                else:
                    means.append(np.nan)
                    maxs.append(np.nan)

            df[f"neighbor_{self.feature_col}_mean_lag{lag}"] = means
            df[f"neighbor_{self.feature_col}_max_lag{lag}"] = maxs

        return df


class LagRollingFeatures(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        specs,
        target_col=None,
        date_col="date",
        group_col=None,
        rolling_funcs=("mean",),
        min_periods=1,
        fill_method=None,
    ):
        self.specs = specs
        self.target_col = target_col
        self.date_col = date_col
        self.group_col = group_col
        self.rolling_funcs = rolling_funcs
        self.min_periods = min_periods
        self.fill_method = fill_method

    def fit(self, X, y=None):
        feature_names = []

        for col, lags in self.specs:
            for lag in lags:
                feature_names.append(f"{col}_lag_{lag}")
                for func in self.rolling_funcs:
                    feature_names.append(f"{col}_roll_{func}_{lag}")

        feature_names.append("delta_days")

        if self.target_col is not None:
            if (
                f"{self.target_col}_lag_1" in feature_names
                and f"{self.target_col}_lag_7" in feature_names
            ):
                feature_names.append("target_lag_7_minus_1")

        self.feature_names_ = feature_names
        return self

    def transform(self, X):
        X = X.copy()

        X[self.date_col] = pd.to_datetime(X[self.date_col])
        X = X.sort_values([self.group_col, self.date_col])
        X = X.set_index([self.group_col, self.date_col])

        X_reset = X.reset_index()

        X_reset["delta_days"] = (
            X_reset
            .sort_values([self.group_col, self.date_col])
            .groupby(self.group_col)[self.date_col]
            .diff()
            .dt.days
        )

        X = (
            X_reset
            .set_index([self.group_col, self.date_col])
        )


        for col, lags in self.specs:
            for lag in lags:
                X[f"{col}_lag_{lag}"] = (
                    X.groupby(level=0)[col].shift(lag)
                )

                for func in self.rolling_funcs:
                    roll_name = f"{col}_roll_{func}_{lag}"
                    X[roll_name] = (
                        X.groupby(level=0)[col]
                         .shift(1)
                         .rolling(window=lag, min_periods=self.min_periods)
                         .agg(func)
                    )

        if self.target_col is not None:
            lag_1 = f"{self.target_col}_lag_1"
            lag_7 = f"{self.target_col}_lag_7"
            if lag_1 in X.columns and lag_7 in X.columns:
                X["target_lag_7_minus_1"] = X[lag_7] - X[lag_1]

        if self.fill_method:
            for method in self.fill_method:
                X[self.feature_names_] = X[self.feature_names_].fillna(method)

        return X.drop(columns=["delta_days"]).reset_index()

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)


class LocationMonthTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        location_col: str = "location",
        month_col: str = "month",
        alpha: float = 50.0,
        output_col: str = "te_location_month",
    ):
        self.location_col = location_col
        self.month_col = month_col
        self.alpha = alpha
        self.output_col = output_col

    def fit(self, X: pd.DataFrame, y):
        X = X.copy()

        if y is None:
            raise ValueError("Target y must be provided for Target Encoding")

        df = X[[self.location_col, self.month_col]].copy()
        df["y"] = y

        self.global_mean_ = df["y"].mean()

        stats = (
            df
            .groupby([self.location_col, self.month_col])["y"]
            .agg(["mean", "count"])
            .reset_index()
        )

        stats[self.output_col] = (
            stats["mean"] * stats["count"]
            + self.global_mean_ * self.alpha
        ) / (stats["count"] + self.alpha)

        self.mapping_ = stats.set_index(
            [self.location_col, self.month_col]
        )[self.output_col]

        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()

        keys = list(zip(X[self.location_col], X[self.month_col]))
        encoded = pd.Series(keys).map(self.mapping_)

        X[self.output_col] = encoded.fillna(self.global_mean_).values
        return X