from pathlib import Path
import pandas as pd
from src.config import RAW_DIR
from src.config import PROCESSED_DIR


def load_raw_train_data() -> pd.DataFrame():
    return pd.read_csv(RAW_DIR / "train.csv")


def load_raw_test_data() -> pd.DataFrame():
    return pd.read_csv(RAW_DIR / "test.csv")


def load_wss() -> pd.DataFrame():
    return pd.read_csv(RAW_DIR / "wss.csv")


def load_external_features() -> pd.DataFrame():
    return pd.read_csv(RAW_DIR / "external_features.csv")
