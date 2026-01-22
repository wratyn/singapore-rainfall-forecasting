import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.dataset import load_raw_train_data, load_raw_test_data
from src.config import EDA_FIGURES_DIR

df_train = load_raw_train_data()
df_test = load_raw_test_data()

# info for train data
df_train.head()
df_train.info()
df_train.shape
df_train.describe()
print((df_train.isnull().sum() / len(df_train)) * 100)
df_train["location"].nunique()

# info for test data
df_test.head()
df_test.info()
df_test.shape
df_test.describe()
print((df_test.isnull().sum() / len(df_test)) * 100)


df_eda = df_train.copy()
df_eda["date"] = pd.to_datetime(df_eda["date"])
df_eda = df_eda.set_index("date")
print("Range tanggal:", df_eda.index.min(), "sampai", df_eda.index.max())

# eda ketersediaan data dari tahun ke tahun
data_availability = df_eda.resample("YE").count()
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    data_availability.index,
    data_availability["daily_rainfall_total_mm"],
    label="Data Hujan",
)

ax.plot(
    data_availability.index,
    data_availability["mean_temperature_c"],
    label="Data Suhu",
    linestyle="--",
)

ax.set_title("Ketersediaan Data dari Tahun ke Tahun")
ax.set_ylabel("Jumlah Baris Data")
ax.legend()
ax.grid(True)

# save
output_path = EDA_FIGURES_DIR / "data_availability_per_year.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight")

plt.show()

# liat distribusi target
df_eda = df_eda.reset_index()
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df_eda["daily_rainfall_total_mm"], bins=50, kde=True)
plt.title("Distribusi Curah Hujan Harian")
plt.xlabel("Curah Hujan (mm)")
plt.yscale("log")

plt.subplot(1, 2, 2)
sns.boxplot(y=df_eda["daily_rainfall_total_mm"])
plt.title("Boxplot Curah Hujan")

plt.tight_layout()
plt.show()

# cek pola musiman
df_eda = df_eda.set_index("date")
df_eda["month"] = df_eda.index.month
monthly_rain = df_eda.groupby("month")["daily_rainfall_total_mm"].mean()
plt.figure(figsize=(10, 5))
monthly_rain.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Rata-rata Curah Hujan per Bulan di Singapura")
plt.xlabel("Bulan")
plt.ylabel("Rata-rata Hujan (mm)")
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.show()

# cek 5 lokasi terbanyak
top_locations = df_eda["location"].value_counts().head(5).index

plt.figure(figsize=(15, 6))
for loc in top_locations:
    subset = (
        df_eda[df_eda["location"] == loc]["daily_rainfall_total_mm"]
        .resample("ME")
        .sum()
    )
    plt.plot(subset.index, subset, label=loc, alpha=0.7)

plt.title("Total Hujan Bulanan di 5 Lokasi Teratas")
plt.legend()
plt.show()
