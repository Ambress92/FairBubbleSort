"""
Experiment B: Real-dataset evaluation.

Two entry points:
  python experiment_real_datasets.py run     -- run greedy on all datasets, save results
  python experiment_real_datasets.py plot    -- read results and produce Pareto front figures

Results are saved incrementally to experiments/{dataset}/pareto.jsonl.
"""

import csv
import json
import math
import os
import sys
import time

# Reuse greedy implementation from scalability experiment
sys.path.insert(0, os.path.dirname(__file__))
from experiment_scalability import greedy_heap, count_concordant_merge, count_inversions_merge

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "experiments")


# ===================================================================
# Dataset loaders
# ===================================================================

def load_compas():
    """COMPAS: score = decile_score, attribute = age."""
    path = os.path.join(os.path.dirname(__file__), "datasets", "compas.csv")
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    scores, attrs = [], []
    for r in rows:
        try:
            s = float(r["decile_score"])
            a = float(r["age"])
            scores.append(s)
            attrs.append(a)
        except (ValueError, KeyError):
            continue

    return scores, attrs, "COMPAS", "Recidivism risk (decile)", "Age"


def load_german_credit():
    """German Credit: score = credit_amount, attribute = age."""
    path = os.path.join(os.path.dirname(__file__), "datasets", "german_credit.csv")
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    scores = [float(r["credit_amount"]) for r in rows]
    attrs = [float(r["age"]) for r in rows]

    return scores, attrs, "German Credit", "Credit amount", "Age"


def load_law_school():
    """Law School: score = lsat, attribute = ugpa."""
    path = os.path.join(os.path.dirname(__file__), "datasets", "law_school.csv")
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    scores, attrs = [], []
    for r in rows:
        try:
            s = float(r["lsat"])
            a = float(r["ugpa"])
            scores.append(s)
            attrs.append(a)
        except (ValueError, KeyError):
            continue

    return scores, attrs, "Law School", "LSAT", "UGPA"


def load_communities():
    """Communities and Crime: score = ViolentCrimesPerPop, attribute = PctPopUnderPov."""
    path = os.path.join(os.path.dirname(__file__), "datasets", "communities.data")
    names_path = os.path.join(os.path.dirname(__file__), "datasets", "communities.names")

    # Parse attribute names
    col_names = []
    with open(names_path) as f:
        for line in f:
            if line.startswith("@attribute"):
                col_names.append(line.strip().split()[1])

    # Column indices
    score_idx = col_names.index("ViolentCrimesPerPop")  # 127
    attr_idx = col_names.index("PctPopUnderPov")         # 33

    scores, attrs = [], []
    with open(path) as f:
        for line in f:
            vals = line.strip().split(",")
            if len(vals) != len(col_names):
                continue
            s_val, a_val = vals[score_idx], vals[attr_idx]
            if s_val == "?" or a_val == "?":
                continue
            scores.append(float(s_val))
            attrs.append(float(a_val))

    return scores, attrs, "Communities", "Violent crime rate", "Poverty rate"


DATASETS = {
    "compas": load_compas,
    "german_credit": load_german_credit,
    "law_school": load_law_school,
    "communities": load_communities,
}


# ===================================================================
# Helpers
# ===================================================================

def append_result(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_results(path):
    records = []
    if not os.path.exists(path):
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_fairness(n_up, n_down, n):
    """Compute phi = 1 - |tau| where tau = (N_up - N_down) / C(n,2)."""
    n_pairs = n * (n - 1) // 2
    if n_pairs == 0:
        return 1.0
    tau_val = (n_up - n_down) / n_pairs
    return 1.0 - abs(tau_val)


def normalize(vals):
    """Normalize values to [0, 1]."""
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return [0.5] * len(vals)
    return [(v - lo) / (hi - lo) for v in vals]


# ===================================================================
# Runner
# ===================================================================

def run_dataset(dataset_key):
    loader = DATASETS[dataset_key]
    scores_raw, attrs_raw, name, score_label, attr_label = loader()

    n = len(scores_raw)
    print(f"\n{'='*60}")
    print(f"Dataset: {name}  (n={n})")
    print(f"  Score: {score_label},  Attribute: {attr_label}")
    print(f"{'='*60}")

    # Sort by score descending to get initial ranking r_s
    paired = list(zip(scores_raw, attrs_raw))
    paired.sort(key=lambda x: -x[0])
    scores_sorted = [p[0] for p in paired]
    attrs_sorted = [p[1] for p in paired]

    # Compute initial fairness (before any reranking)
    _, n_up_init = count_concordant_merge(attrs_sorted)
    _, n_down_init = count_inversions_merge(attrs_sorted)
    phi_init = compute_fairness(n_up_init, n_down_init, n)
    d0 = abs(n_up_init - n_down_init)
    print(f"  Initial: N_up={n_up_init}, N_down={n_down_init}, "
          f"D0={d0}, phi={phi_init:.4f}")

    # Output path
    out_path = os.path.join(EXPERIMENTS_DIR, dataset_key, "pareto.jsonl")
    # Clear previous results
    if os.path.exists(out_path):
        os.rename(out_path, out_path + ".bak")

    # Sweep tau
    taus = [round(t * 0.05, 2) for t in range(21)]

    for tau in taus:
        # Work on copies
        s = scores_sorted[:]
        a = attrs_sorted[:]

        t0 = time.perf_counter()
        cost, k_star = greedy_heap(s, a, tau)
        t1 = time.perf_counter()

        # Compute achieved fairness on the reranked list
        _, n_up_final = count_concordant_merge(a)
        _, n_down_final = count_inversions_merge(a)
        phi_achieved = compute_fairness(n_up_final, n_down_final, n)

        rec = {
            "dataset": dataset_key,
            "n": n,
            "tau": tau,
            "phi_achieved": phi_achieved,
            "cost": cost,
            "k_star": k_star,
            "time": t1 - t0,
            "n_up_init": n_up_init,
            "n_down_init": n_down_init,
            "n_up_final": n_up_final,
            "n_down_final": n_down_final,
            "phi_init": phi_init,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        append_result(out_path, rec)
        print(f"  tau={tau:.2f}  phi={phi_achieved:.4f}  "
              f"cost={cost:.2f}  K*={k_star:>7d}  "
              f"time={t1-t0:.4f}s")

    print(f"\n  Results saved to {out_path}")


def run_all():
    for key in ["german_credit", "compas", "law_school"]:
        run_dataset(key)


# ===================================================================
# Plotter
# ===================================================================

def plot_pareto():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    dataset_configs = [
        ("compas", "COMPAS\n(score: risk, attr: age)", "C0"),
        ("german_credit", "German Credit\n(score: amount, attr: age)", "C1"),
        ("law_school", "Law School\n(score: LSAT, attr: UGPA)", "C2"),
    ]

    # Precompute W = sum_{i<j} |s(i)-s(j)| for each dataset
    norm_W = {}
    for key, _, _ in dataset_configs:
        loader = DATASETS[key]
        scores_raw, _, _, _, _ = loader()
        s_sorted = sorted(scores_raw)
        n = len(s_sorted)
        norm_W[key] = sum(s_sorted[k] * (2 * k - n + 1) for k in range(n))

    for ax, (key, title, color) in zip(axes, dataset_configs):
        path = os.path.join(EXPERIMENTS_DIR, key, "pareto.jsonl")
        records = load_results(path)
        if not records:
            ax.set_title(f"{title}\n(no data)")
            continue

        W = norm_W[key]
        taus = [r["tau"] for r in records]
        phis = [r["phi_achieved"] for r in records]
        costs = [r["cost"] / W for r in records]
        k_stars = [r["k_star"] for r in records]

        # Plot 1: Pareto front (phi vs cost)
        ax.plot(phis, costs, "o-", color=color, linewidth=1.5, markersize=4)

        # Mark a few tau values
        for r in records:
            if r["tau"] in [0.0, 0.5, 0.8, 1.0]:
                ax.annotate(f'$\\tau$={r["tau"]:.1f}',
                            (r["phi_achieved"], r["cost"] / W),
                            textcoords="offset points", xytext=(5, 5),
                            fontsize=7, alpha=0.7)

        ax.set_xlabel(r"Achieved fairness $\phi(r)$", fontsize=10)
        ax.set_ylabel("Weighted Kemeny distance", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Fairness--accuracy trade-off on real datasets", fontsize=13, y=1.02)
    fig.tight_layout()

    out_pdf = os.path.join(EXPERIMENTS_DIR, "fig_pareto_real.pdf")
    out_png = os.path.join(EXPERIMENTS_DIR, "fig_pareto_real.png")
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved {out_pdf}")
    plt.close(fig)


def plot_summary_table():
    """Print a summary table of all datasets."""
    print(f"\n{'Dataset':<16} {'n':>6} {'phi_init':>8} {'D0':>10} "
          f"{'K*(0.8)':>8} {'cost(0.8)':>10} {'time(0.8)':>9}")
    print("-" * 75)

    for key in ["compas", "german_credit", "law_school"]:
        path = os.path.join(EXPERIMENTS_DIR, key, "pareto.jsonl")
        records = load_results(path)
        if not records:
            continue

        r0 = records[0]
        n = r0["n"]
        phi_init = r0["phi_init"]
        d0 = abs(r0["n_up_init"] - r0["n_down_init"])

        loader = DATASETS[key]
        scores_raw, _, _, _, _ = loader()
        s_sorted = sorted(scores_raw)
        W = sum(s_sorted[k] * (2 * k - n + 1) for k in range(n))

        # Find tau=0.8 record
        r08 = next((r for r in records if abs(r["tau"] - 0.8) < 0.01), None)
        if r08:
            print(f"{key:<16} {n:>6} {phi_init:>8.4f} {d0:>10} "
                  f"{r08['k_star']:>8} {r08['cost']/W:>10.6f} {r08['time']:>9.4f}s")


def plot_tau_sweep():
    """Plot tau vs achieved fairness, cost, and K* for each dataset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    dataset_configs = [
        ("compas", "COMPAS", "C0"),
        ("german_credit", "German Credit", "C1"),
        ("law_school", "Law School", "C2"),
    ]

    # Precompute W = sum_{i<j} |s(i)-s(j)| for each dataset
    norm_W = {}
    for key, _, _ in dataset_configs:
        loader = DATASETS[key]
        scores_raw, _, _, _, _ = loader()
        s_sorted = sorted(scores_raw)
        n = len(s_sorted)
        norm_W[key] = sum(s_sorted[k] * (2 * k - n + 1) for k in range(n))

    for ax, (key, title, color) in zip(axes, dataset_configs):
        path = os.path.join(EXPERIMENTS_DIR, key, "pareto.jsonl")
        records = load_results(path)
        if not records:
            continue

        W = norm_W[key]
        taus = [r["tau"] for r in records]
        costs = [r["cost"] / W for r in records]
        k_stars = [r["k_star"] for r in records]

        ax.plot(taus, costs, "o-", color=color, linewidth=1.5, markersize=4,
                label="Weighted Kemeny dist.")
        ax.set_xlabel(r"Fairness threshold $\tau$", fontsize=10)
        ax.set_ylabel("Weighted Kemeny distance", fontsize=10, color=color)
        ax.tick_params(axis="y", labelcolor=color)

        ax2 = ax.twinx()
        ax2.plot(taus, k_stars, "s--", color="C3", linewidth=1, markersize=3,
                 alpha=0.7, label=r"$K^*$")
        ax2.set_ylabel(r"Number of swaps $K^*$", fontsize=10, color="C3")
        ax2.tick_params(axis="y", labelcolor="C3")

        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle(r"Cost and swaps vs fairness threshold $\tau$", fontsize=13, y=1.02)
    fig.tight_layout()

    out_pdf = os.path.join(EXPERIMENTS_DIR, "fig_tau_sweep_real.pdf")
    out_png = os.path.join(EXPERIMENTS_DIR, "fig_tau_sweep_real.png")
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved {out_pdf}")
    plt.close(fig)


# ===================================================================
# Main
# ===================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python experiment_real_datasets.py [run|plot|all]")
        print("  run   -- run greedy on all datasets")
        print("  plot  -- produce figures from saved results")
        print("  all   -- run then plot")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd in ("run", "all"):
        run_all()

    if cmd in ("plot", "all"):
        plot_pareto()
        plot_tau_sweep()
        plot_summary_table()


if __name__ == "__main__":
    main()
