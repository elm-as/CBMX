"""Downloader and parser for the official NASA C-MAPSS FD001 turbofan degradation dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData"
CMAPSS_COLUMNS = (
    ["unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"s{i}" for i in range(1, 22)]
)


def download_cmapss_fd001(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Download train_FD001, test_FD001, and RUL_FD001 if missing and return parsed frames."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"):
        target = data_dir / fname
        if not target.exists():
            resp = requests.get(f"{BASE_URL}/{fname}", timeout=30)
            resp.raise_for_status()
            target.write_text(resp.text, encoding="utf-8")

    train_df = pd.read_csv(data_dir / "train_FD001.txt", sep=r"\s+", header=None, names=CMAPSS_COLUMNS)
    test_df = pd.read_csv(data_dir / "test_FD001.txt", sep=r"\s+", header=None, names=CMAPSS_COLUMNS)
    rul_series = pd.read_csv(data_dir / "RUL_FD001.txt", header=None, names=["true_rul"])["true_rul"]
    return train_df, test_df, rul_series
