from pathlib import Path
import pandas as pd
from src.config import RAW_DIR
from src.config import PROCESSED_DIR


def load_raw_data() -> pd.DataFrame():
    return pd.read_csv(RAW_DIR / "srcsc-2025-dam-data-for-students.csv")


def load_data_flumevale() -> pd.DataFrame():
    return pd.read_csv(PROCESSED_DIR / "region_data" / "dam_data_flumevale.csv")


def load_data_lyndrassia() -> pd.DataFrame():
    return pd.read_csv(PROCESSED_DIR / "region_data" / "dam_data_lyndrassia.csv")


def load_data_navaldia() -> pd.DataFrame():
    return pd.read_csv(PROCESSED_DIR / "region_data" / "dam_data_navaldia.csv")
