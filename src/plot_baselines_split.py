"""
Re-plot the synthetic baselines comparison as separate figures:
  - Main paper: corr=0.5 only (1x3)
  - Appendix:   corr=0.0 (1x3) and corr=1.0 (1x3)

No per-panel titles, no suptitle. Legend at top.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

DATA_FILE = os.path.join(os.path.dirname(__file__), "..",
                         "experiments", "baselines_synthetic", "results.jsonl")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "..",
                        "Neurips2026_Fair_Bubble_Sort", "Plots")


def load_results():
    records = []
    with open(DATA_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


BASELINE_STYLES = {
    "det_const_sort": {"marker": "s", "color": "C1", "label": "DetConstSort"},
    "fair_star":      {"marker": "^", "color": "C2", "label": "FA*IR"},
    "median_split":   {"marker": "D", "color": "C3", "label": "Median-split"},
}

CLIP_FACTOR = 1.5


def plot_row(records, corr, ns, out_prefix):
    fig, axes = plt.subplots(1, len(ns), figsize=(5 * len(ns), 4.2), squeeze=False)

    for col, n in enumerate(ns):
        ax = axes[0][col]
        W = n * (n * n - 1) / 6.0
        non_blend_costs = []

        # FBS
        fbs_recs = [r for r in records
                    if r["method"] == "fbs" and r["n"] == n
                    and r["correlation"] == corr]
        if fbs_recs:
            tau_groups = {}
            for r in fbs_recs:
                tau_groups.setdefault(r["tau"], []).append(r)
            fbs_taus = sorted(tau_groups.keys())
            fbs_phis, fbs_costs = [], []
            for t in fbs_taus:
                phis = [r["phi_achieved"] for r in tau_groups[t]]
                costs = [r["weighted_kemeny"] / W for r in tau_groups[t]]
                fbs_phis.append(np.median(phis))
                fbs_costs.append(np.median(costs))
            ax.plot(fbs_phis, fbs_costs, "o-", color="C0", linewidth=2,
                    markersize=4, label="FBS", zorder=10)
            non_blend_costs += fbs_costs

        # Blend
        blend_recs = [r for r in records
                      if r["method"] == "blend" and r["n"] == n
                      and r["correlation"] == corr]
        if blend_recs:
            alpha_groups = {}
            for r in blend_recs:
                alpha_groups.setdefault(r["alpha"], []).append(r)
            blend_alphas = sorted(alpha_groups.keys())
            blend_phis, blend_costs = [], []
            for a in blend_alphas:
                phis = [r["phi_achieved"] for r in alpha_groups[a]]
                costs = [r["weighted_kemeny"] / W for r in alpha_groups[a]]
                blend_phis.append(np.median(phis))
                blend_costs.append(np.median(costs))
            ax.plot(blend_phis, blend_costs, "x--", color="C4", linewidth=1.5,
                    markersize=5, label="Blend", zorder=8)

        # Baselines
        for bl_name, style in BASELINE_STYLES.items():
            bl_recs = [r for r in records
                       if r["method"] == bl_name and r["n"] == n
                       and r["correlation"] == corr]
            if not bl_recs:
                continue
            g_groups = {}
            for r in bl_recs:
                g_groups.setdefault(r["G"], []).append(r)
            for G in sorted(g_groups.keys()):
                phis = [r["phi_achieved"] for r in g_groups[G]]
                costs = [r["weighted_kemeny"] / W for r in g_groups[G]]
                phi_med = np.median(phis)
                cost_med = np.median(costs)
                non_blend_costs.append(cost_med)
                ax.scatter(phi_med, cost_med, marker=style["marker"],
                           color=style["color"], s=80, zorder=5,
                           edgecolors="black", linewidths=0.5)
                ax.annotate(f"G={G}", (phi_med, cost_med),
                            textcoords="offset points", xytext=(5, 5),
                            fontsize=7, alpha=0.7)

        if non_blend_costs:
            y_max = max(non_blend_costs) * CLIP_FACTOR
            if y_max > 0:
                ax.set_ylim(bottom=0, top=y_max)

        ax.set_xlabel(r"Achieved fairness $\phi$", fontsize=10)
        if col == 0:
            ax.set_ylabel("Weighted Kemeny distance", fontsize=10)
        else:
            ax.set_ylabel("")
        ax.grid(True, alpha=0.3)

    handles = [
        Line2D([0], [0], color="C0", marker="o", linewidth=2,
               markersize=5, label="FBS"),
        Line2D([0], [0], color="C4", marker="x", linewidth=1.5,
               markersize=6, label="Blend", linestyle="--"),
    ]
    for style in BASELINE_STYLES.values():
        handles.append(Line2D([0], [0], color=style["color"],
                              marker=style["marker"], linewidth=0,
                              markersize=8, label=style["label"],
                              markeredgecolor="black", markeredgewidth=0.5))

    fig.legend(handles=handles, loc="upper center", ncol=5,
               fontsize=10, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = os.path.join(PLOT_DIR, f"{out_prefix}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_prefix}.{{pdf,png}}")
    plt.close(fig)


if __name__ == "__main__":
    records = load_results()
    ns = sorted(set(r["n"] for r in records))

    plot_row(records, 0.5, ns, "fig_baselines_synthetic_main")
    plot_row(records, 0.0, ns, "fig_baselines_synthetic_corr0")
    plot_row(records, 1.0, ns, "fig_baselines_synthetic_corr1")
