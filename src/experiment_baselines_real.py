"""
Experiment: Compare FBS against discretization-based baselines on real datasets.

  python experiment_baselines_real.py run
  python experiment_baselines_real.py plot
  python experiment_baselines_real.py all
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from experiment_real_datasets import DATASETS, compute_fairness, load_results, append_result
from experiment_scalability import greedy_heap, count_concordant_merge, count_inversions_merge
from baselines import run_baseline

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "baselines_real")
RESULTS_FILE = os.path.join(RESULTS_DIR, "results_full.jsonl")


def save_result(record):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_all():
    records = []
    if not os.path.exists(RESULTS_FILE):
        return records
    with open(RESULTS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_all():
    taus = [round(t * 0.05, 2) for t in range(21)]
    baselines = ["det_const_sort", "fair_star", "median_split"]
    G_values = [2, 3, 5, 10]

    if os.path.exists(RESULTS_FILE):
        os.rename(RESULTS_FILE, RESULTS_FILE + ".bak")

    for dataset_key in ["german_credit", "compas", "law_school", "communities"]:
        loader = DATASETS[dataset_key]
        scores_raw, attrs_raw, name, score_label, attr_label = loader()
        n = len(scores_raw)

        # Sort by score descending
        paired = sorted(zip(scores_raw, attrs_raw), key=lambda x: -x[0])
        scores_sorted = [p[0] for p in paired]
        attrs_sorted = [p[1] for p in paired]

        print(f"\n{'='*60}")
        print(f"Dataset: {name}  (n={n})")
        print(f"{'='*60}")

        # --- FBS sweep ---
        for tau in taus:
            s = scores_sorted[:]
            a = attrs_sorted[:]
            t0 = time.perf_counter()
            cost, k_star = greedy_heap(s, a, tau)
            t1 = time.perf_counter()
            _, n_up = count_concordant_merge(a)
            _, n_down = count_inversions_merge(a)
            n_pairs = n * (n - 1) // 2
            phi = compute_fairness(n_up, n_down, n)

            save_result({
                "dataset": dataset_key, "method": "fbs", "n": n,
                "tau": tau, "G": None,
                "phi_achieved": phi, "weighted_kemeny": cost,
                "k_star": k_star, "time": t1 - t0,
            })
            if tau in [0.0, 0.5, 0.8, 1.0]:
                print(f"  FBS tau={tau:.2f}: phi={phi:.4f} cost={cost:.2f} time={t1-t0:.4f}s")

        # --- Baselines ---
        for bl_name in baselines:
            if bl_name in ("fair_star", "median_split"):
                gs = [2]
            else:
                gs = G_values

            for G in gs:
                t0 = time.perf_counter()
                res = run_baseline(bl_name, scores_sorted, attrs_sorted, G=G)
                t1 = time.perf_counter()

                save_result({
                    "dataset": dataset_key, "method": bl_name, "n": n,
                    "tau": None, "G": G,
                    "phi_achieved": res["phi_achieved"],
                    "weighted_kemeny": res["weighted_kemeny"],
                    "k_star": None, "time": t1 - t0,
                })
                print(f"  {bl_name} G={G}: phi={res['phi_achieved']:.4f} "
                      f"cost={res['weighted_kemeny']:.2f} time={t1-t0:.4f}s")

        # --- Blend baseline (continuous, sweep alpha) ---
        alphas = [round(a * 0.05, 2) for a in range(21)]
        for alpha in alphas:
            t0 = time.perf_counter()
            res = run_baseline("blend", scores_sorted, attrs_sorted, alpha=alpha)
            t1 = time.perf_counter()

            save_result({
                "dataset": dataset_key, "method": "blend", "n": n,
                "tau": None, "G": None, "alpha": alpha,
                "phi_achieved": res["phi_achieved"],
                "weighted_kemeny": res["weighted_kemeny"],
                "k_star": None, "time": t1 - t0,
            })
            if alpha in [0.0, 0.5, 0.8, 1.0]:
                print(f"  blend alpha={alpha:.2f}: phi={res['phi_achieved']:.4f} "
                      f"cost={res['weighted_kemeny']:.2f} time={t1-t0:.4f}s")

    print(f"\nResults saved to {RESULTS_FILE}")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    records = load_all()
    if not records:
        print("No results. Run first.")
        return

    datasets = ["compas", "german_credit", "law_school", "communities"]
    titles = {"compas": "COMPAS", "german_credit": "German Credit",
              "law_school": "Law School", "communities": "Communities"}

    # Precompute W = sum_{i<j} |s(i)-s(j)| for each dataset
    norm_W = {}
    for ds in datasets:
        loader = DATASETS[ds]
        scores_raw, _, _, _, _ = loader()
        s_sorted = sorted(scores_raw)
        n = len(s_sorted)
        norm_W[ds] = sum(s_sorted[k] * (2 * k - n + 1) for k in range(n))

    fig, axes = plt.subplots(1, 4, figsize=(24, 4.5), sharey=False)

    baseline_styles = {
        "det_const_sort": {"marker": "s", "color": "C1", "label": "DetConstSort"},
        "fair_star": {"marker": "^", "color": "C2", "label": r"FA$^*$IR"},
        "median_split": {"marker": "D", "color": "C3", "label": "Median-split"},
    }

    for ax, ds in zip(axes, datasets):
        W = norm_W[ds]
        non_blend_costs = []

        # FBS Pareto front
        fbs_recs = sorted(
            [r for r in records if r["dataset"] == ds and r["method"] == "fbs"],
            key=lambda r: r["tau"]
        )
        if fbs_recs:
            phis = [r["phi_achieved"] for r in fbs_recs]
            costs = [r["weighted_kemeny"] / W for r in fbs_recs]
            ax.plot(phis, costs, "o-", color="C0", linewidth=2.5, markersize=7,
                    label="FBS", zorder=10)
            non_blend_costs += costs

        # Blend baseline (continuous curve)
        blend_recs = sorted(
            [r for r in records if r["dataset"] == ds and r["method"] == "blend"],
            key=lambda r: r.get("alpha", 0)
        )
        if blend_recs:
            phis = [r["phi_achieved"] for r in blend_recs]
            costs = [r["weighted_kemeny"] / W for r in blend_recs]
            ax.plot(phis, costs, "x--", color="C4", linewidth=2, markersize=8,
                    label="Blend", zorder=8)

        # Baselines
        for bl_name, style in baseline_styles.items():
            bl_recs = [r for r in records if r["dataset"] == ds and r["method"] == bl_name]
            for r in bl_recs:
                c = r["weighted_kemeny"] / W
                non_blend_costs.append(c)
                ax.scatter(r["phi_achieved"], c,
                           marker=style["marker"], color=style["color"],
                           s=200, zorder=15, edgecolors="black", linewidths=1.0)
                ax.annotate(f"G={r['G']}", (r["phi_achieved"], c),
                            textcoords="offset points", xytext=(-6, 6),
                            fontsize=11, alpha=0.8, fontweight="bold",
                            ha="right")

        if non_blend_costs:
            y_max = max(non_blend_costs) * 1.5
            if y_max > 0:
                ax.set_ylim(bottom=-y_max * 0.03, top=y_max)

        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(11)
        ax.set_xlabel(r"Achieved fairness $\phi$", fontsize=14)
        if ax == axes[0]:
            ax.set_ylabel("Weighted Kemeny distance", fontsize=14)
        ax.set_title(titles[ds], fontsize=16, fontweight="bold")
        ax.tick_params(axis="both", labelsize=12)
        if ds == "german_credit":
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
        ax.grid(True, alpha=0.3)

    # Legend inside the third subplot (Law School)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color="C0", marker="o", linewidth=2.5, markersize=8, label="FBS")]
    handles.append(Line2D([0], [0], color="C4", marker="x", linewidth=2,
                          markersize=9, label="Blend", linestyle="--"))
    for bl_name, style in baseline_styles.items():
        handles.append(Line2D([0], [0], color=style["color"], marker=style["marker"],
                              linewidth=0, markersize=10, label=style["label"],
                              markeredgecolor="black", markeredgewidth=1.0))
    axes[2].legend(handles=handles, loc="upper center", fontsize=12,
                   framealpha=0.9, edgecolor="gray")
    fig.tight_layout(w_pad=0.8)

    out_pdf = os.path.join(RESULTS_DIR, "fig_baselines_real.pdf")
    out_png = os.path.join(RESULTS_DIR, "fig_baselines_real.png")
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved {out_pdf}")
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("Usage: python experiment_baselines_real.py [run|plot|all]")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd in ("run", "all"):
        run_all()
    if cmd in ("plot", "all"):
        plot()


if __name__ == "__main__":
    main()
