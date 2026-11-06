"""
Experiment: Compare FBS against discretization-based baselines on synthetic data.

Entry points:
  python experiment_baselines_synthetic.py run   -- run all experiments
  python experiment_baselines_synthetic.py plot  -- produce comparison figures
  python experiment_baselines_synthetic.py all   -- run then plot

Results saved to experiments/baselines_synthetic/results.jsonl
"""

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from experiment_scalability import (
    generate_instance, greedy_heap,
    count_concordant_merge, count_inversions_merge,
)
from baselines import run_baseline, compute_phi_continuous, weighted_kemeny_naive

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "baselines_synthetic")
RESULTS_FILE = os.path.join(RESULTS_DIR, "results.jsonl")


def append_result(record):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_results():
    records = []
    if not os.path.exists(RESULTS_FILE):
        return records
    with open(RESULTS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ===================================================================
# Helpers
# ===================================================================

def compute_fairness(n_up, n_down, n):
    n_pairs = n * (n - 1) // 2
    if n_pairs == 0:
        return 1.0
    tau_val = (n_up - n_down) / n_pairs
    return 1.0 - abs(tau_val)


def run_fbs_sweep(scores, attrs, taus):
    """Run FBS for each tau, return list of result dicts."""
    results = []
    n = len(scores)
    for tau in taus:
        s = scores[:]
        a = attrs[:]
        t0 = time.perf_counter()
        cost, k_star = greedy_heap(s, a, tau)
        t1 = time.perf_counter()
        _, n_up = count_concordant_merge(a)
        _, n_down = count_inversions_merge(a)
        phi = compute_fairness(n_up, n_down, n)

        # Compute weighted Kemeny via the reordered list
        # FBS returns cost directly as sum of |s_i - s_j| per swap
        results.append({
            "method": "fbs",
            "tau": tau,
            "G": None,
            "phi_achieved": phi,
            "weighted_kemeny": cost,
            "k_star": k_star,
            "time": t1 - t0,
        })
    return results


# ===================================================================
# Runner
# ===================================================================

def run_all():
    ns = [100, 1000, 5000]
    correlations = [0.0, 0.5, 1.0]
    baselines = ["det_const_sort", "fair_star", "median_split"]
    G_values = [2, 3, 5, 10]
    taus = [round(t * 0.05, 2) for t in range(21)]
    repeats = 1

    # Clear previous results
    if os.path.exists(RESULTS_FILE):
        os.rename(RESULTS_FILE, RESULTS_FILE + ".bak")

    for n in ns:
        for corr in correlations:
            print(f"\n{'='*60}")
            print(f"n={n}, correlation={corr}")
            print(f"{'='*60}")

            for rep in range(repeats):
                scores, attrs = generate_instance(n, correlation=corr)

                # --- FBS sweep ---
                fbs_results = run_fbs_sweep(scores, attrs, taus)
                for r in fbs_results:
                    r.update({"n": n, "correlation": corr, "repeat": rep,
                              "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
                    append_result(r)

                if rep == 0:
                    # Print FBS summary for first repeat
                    for r in fbs_results:
                        if r["tau"] in [0.0, 0.5, 0.8, 1.0]:
                            print(f"  FBS tau={r['tau']:.2f}: "
                                  f"phi={r['phi_achieved']:.4f} "
                                  f"cost={r['weighted_kemeny']:.2f}")

                # --- Baselines ---
                for bl_name in baselines:
                    # FA*IR is binary: only G=2. Median-split is always G=2.
                    if bl_name in ("fair_star", "median_split"):
                        gs = [2]
                    else:
                        gs = G_values
                    for G in gs:
                        t0 = time.perf_counter()
                        res = run_baseline(bl_name, scores, attrs, G=G)
                        t1 = time.perf_counter()

                        rec = {
                            "method": bl_name,
                            "n": n,
                            "correlation": corr,
                            "G": G,
                            "tau": None,
                            "phi_achieved": res["phi_achieved"],
                            "weighted_kemeny": res["weighted_kemeny"],
                            "k_star": None,
                            "time": t1 - t0,
                            "repeat": rep,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        append_result(rec)

                        if rep == 0:
                            print(f"  {bl_name} G={G}: "
                                  f"phi={res['phi_achieved']:.4f} "
                                  f"cost={res['weighted_kemeny']:.2f} "
                                  f"time={t1-t0:.4f}s")

                # --- Blend baseline (continuous, sweep alpha) ---
                alphas = [round(a * 0.05, 2) for a in range(21)]
                for alpha in alphas:
                    t0 = time.perf_counter()
                    res = run_baseline("blend", scores, attrs, alpha=alpha)
                    t1 = time.perf_counter()

                    rec = {
                        "method": "blend",
                        "n": n,
                        "correlation": corr,
                        "G": None,
                        "tau": None,
                        "alpha": alpha,
                        "phi_achieved": res["phi_achieved"],
                        "weighted_kemeny": res["weighted_kemeny"],
                        "k_star": None,
                        "time": t1 - t0,
                        "repeat": rep,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    append_result(rec)

                    if rep == 0 and alpha in [0.0, 0.5, 0.8, 1.0]:
                        print(f"  blend alpha={alpha:.2f}: "
                              f"phi={res['phi_achieved']:.4f} "
                              f"cost={res['weighted_kemeny']:.2f} "
                              f"time={t1-t0:.4f}s")

            print(f"  ({repeats} repeats done)")

    print(f"\nResults saved to {RESULTS_FILE}")


# ===================================================================
# Plotter
# ===================================================================

def plot_comparison():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    records = load_results()
    if not records:
        print("No results found. Run experiments first.")
        return

    ns = sorted(set(r["n"] for r in records))
    correlations = sorted(set(r["correlation"] for r in records))

    fig, axes = plt.subplots(len(correlations), len(ns),
                             figsize=(5 * len(ns), 4.5 * len(correlations)),
                             squeeze=False)

    baseline_styles = {
        "det_const_sort": {"marker": "s", "color": "C1", "label": "DetConstSort"},
        "fair_star": {"marker": "^", "color": "C2", "label": "FA*IR"},
        "median_split": {"marker": "D", "color": "C3", "label": "Median-split"},
    }

    CLIP_FACTOR = 1.5  # y-axis upper limit = max non-Blend cost * factor

    for row, corr in enumerate(correlations):
        for col, n in enumerate(ns):
            ax = axes[row][col]

            # Normalization: W = sum_{i<j} |s(i)-s(j)| for scores [n,...,1]
            W = n * (n * n - 1) / 6.0

            non_blend_costs = []  # collected to set per-panel y-axis clip

            # --- FBS Pareto front (median over repeats) ---
            fbs_recs = [r for r in records
                        if r["method"] == "fbs" and r["n"] == n
                        and r["correlation"] == corr]
            if fbs_recs:
                # Group by tau, take median
                tau_groups = {}
                for r in fbs_recs:
                    tau_groups.setdefault(r["tau"], []).append(r)

                fbs_taus = sorted(tau_groups.keys())
                fbs_phis = []
                fbs_costs = []
                for t in fbs_taus:
                    phis = [r["phi_achieved"] for r in tau_groups[t]]
                    costs = [r["weighted_kemeny"] / W for r in tau_groups[t]]
                    fbs_phis.append(np.median(phis))
                    fbs_costs.append(np.median(costs))

                ax.plot(fbs_phis, fbs_costs, "o-", color="C0", linewidth=2,
                        markersize=4, label="FBS", zorder=10)
                non_blend_costs += fbs_costs

            # --- Blend baseline (continuous curve) ---
            blend_recs = [r for r in records
                          if r["method"] == "blend" and r["n"] == n
                          and r["correlation"] == corr]
            if blend_recs:
                alpha_groups = {}
                for r in blend_recs:
                    alpha_groups.setdefault(r["alpha"], []).append(r)
                blend_alphas = sorted(alpha_groups.keys())
                blend_phis = []
                blend_costs = []
                for a in blend_alphas:
                    phis = [r["phi_achieved"] for r in alpha_groups[a]]
                    costs = [r["weighted_kemeny"] / W for r in alpha_groups[a]]
                    blend_phis.append(np.median(phis))
                    blend_costs.append(np.median(costs))
                ax.plot(blend_phis, blend_costs, "x--", color="C4", linewidth=1.5,
                        markersize=5, label="Blend", zorder=8)

            # --- Baselines ---
            for bl_name, style in baseline_styles.items():
                bl_recs = [r for r in records
                           if r["method"] == bl_name and r["n"] == n
                           and r["correlation"] == corr]
                if not bl_recs:
                    continue

                # Group by G, take median
                g_groups = {}
                for r in bl_recs:
                    g_groups.setdefault(r["G"], []).append(r)

                for G in sorted(g_groups.keys()):
                    phis = [r["phi_achieved"] for r in g_groups[G]]
                    costs = [r["weighted_kemeny"] / W for r in g_groups[G]]
                    phi_med = np.median(phis)
                    cost_med = np.median(costs)
                    non_blend_costs.append(cost_med)

                    label = f"{style['label']} G={G}" if col == 0 and row == 0 else None
                    ax.scatter(phi_med, cost_med, marker=style["marker"],
                               color=style["color"], s=80, zorder=5,
                               edgecolors="black", linewidths=0.5)
                    ax.annotate(f"G={G}", (phi_med, cost_med),
                                textcoords="offset points", xytext=(5, 5),
                                fontsize=7, alpha=0.7)

            # Clip y-axis to 1.5 * max non-Blend cost (lets Blend exit the panel)
            if non_blend_costs:
                y_max = max(non_blend_costs) * CLIP_FACTOR
                if y_max > 0:
                    ax.set_ylim(bottom=0, top=y_max)

            ax.set_xlabel(r"Achieved fairness $\phi$", fontsize=10)
            ax.set_ylabel("Weighted Kemeny distance", fontsize=10)
            ax.set_title(f"n={n}, corr={corr}", fontsize=11)
            ax.grid(True, alpha=0.3)

    # Build legend from first subplot
    # Collect handles manually
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([0], [0], color="C0", marker="o", linewidth=2,
                       markersize=5, label="FBS")]
    handles.append(Line2D([0], [0], color="C4", marker="x", linewidth=1.5,
                          markersize=6, label="Blend", linestyle="--"))
    for bl_name, style in baseline_styles.items():
        handles.append(Line2D([0], [0], color=style["color"],
                              marker=style["marker"], linewidth=0,
                              markersize=8, label=style["label"],
                              markeredgecolor="black", markeredgewidth=0.5))

    fig.legend(handles=handles, loc="upper center", ncol=5,
               fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("FBS vs. baselines (synthetic data)",
                 fontsize=14, y=1.06)
    fig.tight_layout()

    out_pdf = os.path.join(RESULTS_DIR, "fig_baselines_synthetic.pdf")
    out_png = os.path.join(RESULTS_DIR, "fig_baselines_synthetic.png")
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved {out_pdf}")
    plt.close(fig)


# ===================================================================
# Main
# ===================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python experiment_baselines_synthetic.py [run|plot|all]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd in ("run", "all"):
        run_all()
    if cmd in ("plot", "all"):
        plot_comparison()


if __name__ == "__main__":
    main()
