"""CLI entrypoint for CBMX on official NASA C-MAPSS FD001 turbofan dataset."""

from __future__ import annotations

import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold

from src.decision.maintenance_policy import MaintenanceCostConfig, select_optimal_threshold
from src.rul.rul_calibrator import compute_nasa_asymmetric_score
from src.survival.kaplan_meier import compute_concordance_index, fit_kaplan_meier
from src.telemetry.download_cmapss import download_cmapss_fd001
from src.visualization.cbmx_plots import generate_cbmx_visuals

DATA_DIR = Path("data")
RESULTS_PATH = Path("artifacts/results.json")
DASHBOARD_PATH = Path("artifacts/dashboard.html")
IMAGES_DIR = Path("images")

ACTIVE_SENSORS = ("s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21")


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute causal rolling means and cycle-normalized features without cross-unit leakage."""
    out = df[["unit_id", "cycle", *ACTIVE_SENSORS]].copy()
    for col in ACTIVE_SENSORS:
        out[f"{col}_roll5"] = out.groupby("unit_id")[col].transform(lambda s: s.rolling(5, min_periods=1).mean())
    return out


def run_pipeline() -> dict[str, float]:
    """Download NASA C-MAPSS FD001, train GroupKFold LightGBM RUL, and evaluate survival + cost."""
    train_df, test_df, true_rul_series = download_cmapss_fd001(DATA_DIR)

    max_cycles = train_df.groupby("unit_id")["cycle"].transform("max")
    # Piece-wise linear RUL target capped at 125 cycles (standard NASA C-MAPSS protocol).
    y_train = (max_cycles - train_df["cycle"]).clip(upper=125).values
    X_train_full = build_feature_table(train_df)
    feature_cols = [c for c in X_train_full.columns if c not in ("unit_id", "cycle")]

    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(train_df))
    for tr_idx, va_idx in gkf.split(X_train_full[feature_cols], y_train, groups=train_df["unit_id"]):
        model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
        model.fit(X_train_full[feature_cols].iloc[tr_idx], y_train[tr_idx])
        oof_preds[va_idx] = np.clip(model.predict(X_train_full[feature_cols].iloc[va_idx]), 0.0, 125.0)

    final_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
    final_model.fit(X_train_full[feature_cols], y_train)

    X_test_full = build_feature_table(test_df)
    last_cycles = X_test_full.groupby("unit_id").tail(1).sort_values("unit_id")
    test_pred_rul = np.clip(final_model.predict(last_cycles[feature_cols]), 0.0, 125.0)
    true_rul = true_rul_series.values.astype(float)

    nasa_score = compute_nasa_asymmetric_score(tuple(test_pred_rul.tolist()), tuple(true_rul.tolist()))
    test_rmse = math.sqrt(mean_squared_error(true_rul, test_pred_rul))
    test_mae = mean_absolute_error(true_rul, test_pred_rul)

    # Survival analysis across combined train (uncensored) + test (right-censored) fleet.
    train_durations = tuple(train_df.groupby("unit_id")["cycle"].max().astype(int).tolist())
    test_durations = tuple(test_df.groupby("unit_id")["cycle"].max().astype(int).tolist())
    durations = train_durations + test_durations
    events = tuple([True] * len(train_durations) + [False] * len(test_durations))

    train_last = X_train_full.groupby("unit_id").tail(1).sort_values("unit_id")
    train_risks = tuple((125.0 - np.clip(final_model.predict(train_last[feature_cols]), 0.0, 125.0)).tolist())
    test_risks = tuple((125.0 - test_pred_rul).tolist())
    risks = train_risks + test_risks

    km_curve = fit_kaplan_meier(durations, events)
    c_index = compute_concordance_index(risks, durations, events)

    max_r = max(risks) if risks else 1.0
    probs = tuple(min(0.95, max(0.05, r / max(max_r, 1e-6))) for r in risks)
    policy = select_optimal_threshold(probs, events, MaintenanceCostConfig())

    generate_cbmx_visuals(durations, events, risks, true_rul, test_pred_rul, IMAGES_DIR, DASHBOARD_PATH)

    metrics = {
        "n_train_units": float(train_df["unit_id"].nunique()),
        "n_train_cycles": float(len(train_df)),
        "n_test_units": float(test_df["unit_id"].nunique()),
        "test_rmse_cycles": round(test_rmse, 4),
        "test_mae_cycles": round(test_mae, 4),
        "nasa_asymmetric_score": round(nasa_score, 4),
        "c_index": round(c_index, 4),
        "km_steps": float(len(km_curve)),
        "optimal_threshold": policy.optimal_threshold,
        "expected_cost_eur": round(policy.expected_cost_eur, 2),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print("[CBMX] Pipeline completed:", run_pipeline())
