"""Publication-quality figures for DiagnosticResult."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .core import DiagnosticResult


PALETTE = {
    "INDEPENDENT": "#029E73",
    "SUBSTITUTION": "#DE8F05",
    "REDUNDANT":    "#CC3311",
    "neutral":      "#0173B2",
    "threshold":    "#888888",
}


def plot_result(
    result: "DiagnosticResult",
    title: Optional[str] = None,
    figsize: tuple = (14, 9),
    dpi: int = 150,
):
    """
    Generate a four-panel publication-quality figure summarising the
    diagnostic result.

    Panels
    ------
    Top-left  : SHAP beeswarm (requires shap to be installed)
    Top-right : Three-method rank bar chart with LIME error bar
    Bottom-left : ΔAUC bar chart (permutation + ablation) vs. threshold
    Bottom-right: Diagnostic summary box with classification verdict

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    cls = result.classification
    color = PALETTE[cls]

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.36)

    # ── Panel 1: SHAP beeswarm ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    if result.shap_values is not None and result.X is not None:
        try:
            import shap
            shap.summary_plot(
                result.shap_values, result.X,
                feature_names=list(result.X.columns),
                show=False, plot_type="dot", max_display=len(result.X.columns),
                plot_size=None, color_bar=True,
            )
            plt.sca(ax1)
        except Exception:
            ax1.text(0.5, 0.5, "SHAP plot unavailable", ha="center", va="center",
                     transform=ax1.transAxes, fontsize=10)
    else:
        ax1.text(0.5, 0.5, "SHAP values not stored\n(run with store_shap=True)",
                 ha="center", va="center", transform=ax1.transAxes, fontsize=10)

    ax1.set_title(
        f"SHAP Beeswarm\n{result.feature} → {result.outcome}",
        fontsize=11, fontweight="bold",
    )

    # ── Panel 2: Three-method rank bars ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    n_features = result.n_features
    methods = ["SHAP\nRank", "Perm.\nRank", f"LIME\nRank\n(mean±SD)"]
    values = [result.shap_rank, result.perm_rank, result.lime_rank_mean]
    errors = [0, 0, result.lime_rank_sd or 0]
    bar_colors = [
        PALETTE["neutral"] if v <= max(1, n_features // 3) else
        PALETTE["SUBSTITUTION"] if v <= n_features // 2 else
        PALETTE["REDUNDANT"]
        for v in values
    ]
    bars = ax2.bar(methods, values, color=bar_colors, alpha=0.85,
                   edgecolor="white", linewidth=1.2, width=0.45)
    ax2.errorbar([2], [result.lime_rank_mean], yerr=[result.lime_rank_sd or 0],
                 fmt="none", color="black", capsize=6, linewidth=2)
    ax2.set_ylim(0, n_features + 0.8)
    ax2.set_yticks(range(1, n_features + 1))
    ax2.axhline(n_features // 3, color=PALETTE["threshold"], lw=1,
                ls="--", alpha=0.5, label=f"Top-{n_features // 3} threshold")
    ax2.set_ylabel("Feature Rank (lower = more important)", fontsize=9)
    ax2.set_title(f"Three-Method Consensus\n{result.feature}", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.12,
                 f"{val:.1f}", ha="center", fontsize=10, fontweight="bold")

    # ── Panel 3: ΔAUC bars ────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    cats = ["Perm. ΔAUC\n(30 repeats)", "Ablation ΔAUC\n(5-fold CV)"]
    dvals = [result.perm_delta_auc or 0, result.delta_auc_ablation]
    dcols = [PALETTE["INDEPENDENT"] if v >= result.threshold else PALETTE["REDUNDANT"]
             for v in dvals]
    b3 = ax3.bar(cats, dvals, color=dcols, alpha=0.85, edgecolor="white",
                 linewidth=1.2, width=0.38)
    if result.delta_auc_se:
        ax3.errorbar([0], [result.perm_delta_auc or 0],
                     yerr=[result.delta_auc_se * 1.96],
                     fmt="none", color="black", capsize=6, linewidth=2)
    ax3.axhline(result.threshold, color=PALETTE["threshold"], lw=1.5,
                ls="--", alpha=0.7, label=f"Threshold ({result.threshold})")
    ax3.axhline(0, color="black", lw=0.8, alpha=0.4)
    ax3.set_ylabel(f"ΔAUC ({result.feature} contribution)", fontsize=9)
    ax3.set_title(f"ΔAUC Analysis\n{result.feature} → {result.outcome}", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=8)
    for bar, val in zip(b3, dvals):
        ypos = val - 0.003 if val < 0 else val + 0.001
        ax3.text(bar.get_x() + bar.get_width() / 2, ypos,
                 f"{val:+.4f}", ha="center", fontsize=10, fontweight="bold")

    # ── Panel 4: Summary box ──────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    or_str = (f"{result.or_value:.3f} [{result.or_ci_lower:.3f}–{result.or_ci_upper:.3f}], "
              f"p={result.p_value:.3f}")
    lime_str = (f"{result.lime_rank_mean:.1f}±{result.lime_rank_sd:.1f}/{n_features}"
                if result.lime_rank_mean else "N/A")

    summary = (
        f"DIAGNOSTIC RESULT\n\n"
        f"Feature:   {result.feature}\n"
        f"Outcome:   {result.outcome}\n"
        f"Dataset:   n={result.n_samples}, events={result.n_events} "
        f"({result.n_events / result.n_samples * 100:.1f}%)\n\n"
        f"Model AUC (5-fold CV): {result.auc_full:.3f}\n\n"
        f"Metrics:\n"
        f"  SHAP rank:     {result.shap_rank}/{n_features} (|SHAP|={result.shap_value:.3f})\n"
        f"  Perm. rank:    {result.perm_rank}/{n_features}\n"
        f"  LIME rank:     {lime_str}\n"
        f"  Ablation ΔAUC: {result.delta_auc_ablation:+.4f} "
        f"({'≥' if result.delta_auc_ablation >= result.threshold else '<'}{result.threshold})\n"
        f"  OR:            {or_str}\n\n"
        f"Rules:\n"
        f"  OR significant:  {'✓' if result.or_significant else '✗'} (p={result.p_value:.3f})\n"
        f"  ΔAUC ≥ {result.threshold}: {'✓' if result.delta_auc_ablation >= result.threshold else '✗'} "
        f"(ΔAUC={result.delta_auc_ablation:+.4f})\n"
    )
    ax4.text(0.05, 0.97, summary, transform=ax4.transAxes, fontsize=8.5,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#f5f5f5", alpha=0.85))
    ax4.text(0.5, 0.06, f"★  {cls}  ★",
             transform=ax4.transAxes, fontsize=16, fontweight="bold",
             color=color, ha="center",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.12))

    fig.suptitle(
        title or f"Three-Metric Diagnostic Framework\n{result.feature} → {result.outcome}",
        fontsize=13, fontweight="bold", y=0.99,
    )
    return fig
