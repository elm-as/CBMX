"""Publication-grade 300 DPI matplotlib charts and interactive Plotly dashboard for CBMX."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from src.decision.maintenance_policy import MaintenanceCostConfig, select_optimal_threshold
from src.survival.kaplan_meier import fit_kaplan_meier


def generate_cbmx_visuals(
    durations: tuple[int, ...],
    events: tuple[bool, ...],
    risks: tuple[float, ...],
    true_rul: np.ndarray,
    pred_rul: np.ndarray,
    images_dir: Path,
    dashboard_path: Path,
) -> None:
    """Render 3 PNG figures (300 DPI) and a standalone Plotly HTML5 dashboard."""
    images_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})

    _plot_kaplan_meier_strata(durations, events, risks, images_dir / "cbmx_kaplan_meier_survival.png")
    _plot_rul_predictions(true_rul, pred_rul, images_dir / "cbmx_rul_prediction_asymmetric.png")
    _plot_maintenance_cost(risks, events, images_dir / "cbmx_maintenance_cost_curve.png")
    _build_plotly_dashboard(true_rul, pred_rul, dashboard_path)


def _plot_kaplan_meier_strata(
    durations: tuple[int, ...],
    events: tuple[bool, ...],
    risks: tuple[float, ...],
    out_path: Path,
) -> None:
    arr_r = np.array(risks)
    q33, q66 = np.quantile(arr_r, [0.33, 0.66])
    strata = [
        ("Low Degradation Risk", arr_r <= q33, "#10B981"),
        ("Medium Degradation Risk", (arr_r > q33) & (arr_r <= q66), "#F59E0B"),
        ("High Degradation Risk", arr_r > q66, "#DC2626"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for label, mask, color in strata:
        d_sub = tuple(d for d, m in zip(durations, mask) if m)
        e_sub = tuple(ev for ev, m in zip(events, mask) if m)
        if not any(e_sub):
            continue
        curve = fit_kaplan_meier(d_sub, e_sub)
        ax.step([0] + [p.time for p in curve], [1.0] + [p.survival_prob for p in curve],
                where="post", label=label, color=color, linewidth=2.0)

    ax.set_title("NASA C-MAPSS FD001: Kaplan-Meier Survival by Sensor Degradation Tercile", fontweight="bold")
    ax.set_xlabel("Operational Cycles")
    ax.set_ylabel("Survival Probability S(t)")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_rul_predictions(true_rul: np.ndarray, pred_rul: np.ndarray, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.5))
    ax1.scatter(true_rul, pred_rul, alpha=0.75, color="#2563EB", edgecolors="#1E293B", s=38)
    lims = [0, max(true_rul.max(), pred_rul.max()) + 5]
    ax1.plot(lims, lims, "--", color="#DC2626", linewidth=1.4, label="Ideal Parity (y = x)")
    ax1.set_title("FD001 Test Set (100 Engines): Predicted vs True RUL", fontweight="bold")
    ax1.set_xlabel("True RUL (Cycles)")
    ax1.set_ylabel("Predicted RUL (Cycles)")
    ax1.legend(frameon=False)

    errors = pred_rul - true_rul
    ax2.hist(errors, bins=18, color="#6366F1", alpha=0.85, edgecolor="white")
    ax2.axvline(0.0, color="#DC2626", linestyle="--", linewidth=1.4)
    ax2.set_title("RUL Error Distribution (Late > 0 Penalized Heavily)", fontweight="bold")
    ax2.set_xlabel("Prediction Error (Pred - True Cycles)")
    ax2.set_ylabel("Engine Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_maintenance_cost(risks: tuple[float, ...], events: tuple[bool, ...], out_path: Path) -> None:
    max_r = max(risks) if risks else 1.0
    probs = tuple(min(0.95, max(0.05, r / max(max_r, 1e-6))) for r in risks)
    cfg = MaintenanceCostConfig()
    thresholds = [i * 0.05 for i in range(1, 20)]
    costs = []
    for thr in thresholds:
        c = sum(cfg.preventive_cost_eur if p >= thr else (cfg.unplanned_failure_cost_eur if ev else 0.0)
                for p, ev in zip(probs, events)) / len(probs)
        costs.append(c)
    best = select_optimal_threshold(probs, events, cfg)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(thresholds, costs, marker="o", color="#0F172A", linewidth=2.0)
    ax.axvline(best.optimal_threshold, color="#10B981", linestyle="--", linewidth=1.6,
               label=f"Optimal Threshold = {best.optimal_threshold:.2f} ({best.expected_cost_eur:.0f} EUR/unit)")
    ax.set_title("Expected Fleet Cost Under Asymmetric Failure Penalty", fontweight="bold")
    ax.set_xlabel("Intervention Probability Threshold")
    ax.set_ylabel("Expected Cost per Engine (EUR)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _build_plotly_dashboard(true_rul: np.ndarray, pred_rul: np.ndarray, dashboard_path: Path) -> None:
    engines = list(range(1, len(true_rul) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=engines, y=true_rul, mode="lines+markers", name="True RUL (FD001)", line=dict(color="#0F172A")))
    fig.add_trace(go.Scatter(x=engines, y=pred_rul, mode="lines+markers", name="LightGBM Predicted RUL", line=dict(color="#2563EB", dash="dash")))
    fig.update_layout(
        title="CBMX — NASA C-MAPSS FD001 Test Engines: Predicted vs Ground-Truth RUL",
        xaxis_title="Test Engine Unit ID",
        yaxis_title="Remaining Useful Life (Cycles)",
        template="plotly_white",
    )
    fig.write_html(str(dashboard_path), include_plotlyjs="cdn")
