"""
Scalability comparison: FBS vs baselines (wall-clock time of reranking only).

  python experiment_scalability_baselines.py run
  python experiment_scalability_baselines.py plot
  python experiment_scalability_baselines.py all
"""

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from experiment_scalability import generate_instance, greedy_heap
from baselines import (
    discretize_quantile, det_const_sort, fair_star,
    median_split_interleave, compute_phi_continuous,
)
from collections import Counter

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "baselines_scalability")
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


def run_all():
    ns = [100, 300, 1000, 3000, 10000, 30000]
    correlation = 1.0  # worst case for FBS
    repeats_default = 1

    if os.path.exists(RESULTS_FILE):
        os.rename(RESULTS_FILE, RESULTS_FILE + ".bak")

    for n in ns:
        print(f"\n{'='*50}")
        print(f"n={n}")
        print(f"{'='*50}")

        repeats = repeats_default
        for rep in range(repeats):
            scores, attrs = generate_instance(n, correlation=correlation)

            # --- FBS (tau=1.0, worst case) ---
            s, a = scores[:], attrs[:]
            t0 = time.perf_counter()
            cost, k_star = greedy_heap(s, a, 1.0)
            t1 = time.perf_counter()
            append_result({"method": "FBS", "n": n, "time": t1 - t0,
                           "repeat": rep, "detail": "tau=1.0"})
            if rep == 0:
                print(f"  FBS:            {t1-t0:.4f}s")

            # --- DetConstSort G=2 ---
            groups = discretize_quantile(attrs, 2)
            gc = Counter(groups)
            proportions = {g: c / n for g, c in gc.items()}
            t0 = time.perf_counter()
            det_const_sort(scores, groups, proportions)
            t1 = time.perf_counter()
            append_result({"method": "DetConstSort", "n": n, "time": t1 - t0,
                           "repeat": rep, "detail": "G=2"})
            if rep == 0:
                print(f"  DetConstSort G=2: {t1-t0:.4f}s")

            # --- DetConstSort G=10 ---
            groups10 = discretize_quantile(attrs, 10)
            gc10 = Counter(groups10)
            proportions10 = {g: c / n for g, c in gc10.items()}
            t0 = time.perf_counter()
            det_const_sort(scores, groups10, proportions10)
            t1 = time.perf_counter()
            append_result({"method": "DetConstSort G=10", "n": n, "time": t1 - t0,
                           "repeat": rep, "detail": "G=10"})
            if rep == 0:
                print(f"  DetConstSort G=10: {t1-t0:.4f}s")

            # --- FA*IR G=2 (capped at k=400) ---
            if n <= 10000:  # skip for very large n, it's constant time anyway
                groups2 = discretize_quantile(attrs, 2)
                gc2 = Counter(groups2)
                proportions2 = {g: c / n for g, c in gc2.items()}
                t0 = time.perf_counter()
                fair_star(scores, groups2, proportions2)
                t1 = time.perf_counter()
                append_result({"method": "FA*IR", "n": n, "time": t1 - t0,
                               "repeat": rep, "detail": "G=2, k=min(n,400)"})
                if rep == 0:
                    print(f"  FA*IR:          {t1-t0:.4f}s")

            # --- Median-split ---
            t0 = time.perf_counter()
            median_split_interleave(scores, attrs)
            t1 = time.perf_counter()
            append_result({"method": "Median-split", "n": n, "time": t1 - t0,
                           "repeat": rep, "detail": "G=2"})
            if rep == 0:
                print(f"  Median-split:   {t1-t0:.4f}s")

    print(f"\nResults saved to {RESULTS_FILE}")


def _load_numba_fbs(corr=1.0):
    """Load optimized FBS (heap) timings from the numba scalability experiment."""
    numba_path = os.path.join(
        os.path.dirname(__file__), "..", "experiments", "scalability_numba", "scalability_vs_n.jsonl"
    )
    by_n = {}
    if os.path.exists(numba_path):
        with open(numba_path) as f:
            for line in f:
                r = json.loads(line)
                if r["method"] == "heap" and r["correlation"] == corr:
                    by_n[r["n"]] = r["time_median"]
    return by_n


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    records = load_results()
    if not records:
        print("No results. Run first.")
        return

    numba_fbs = _load_numba_fbs(corr=1.0)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    baseline_ns = sorted({r["n"] for r in records})

    styles = {
        "FBS":                {"color": "C0", "marker": "o", "ls": "-",  "lw": 2, "label": "FBS weighted"},
        "DetConstSort":       {"color": "C1", "marker": "s", "ls": "--", "lw": 1.5},
        "DetConstSort G=10":  {"color": "C4", "marker": "p", "ls": "--", "lw": 1.5},
        "Median-split":       {"color": "C3", "marker": "D", "ls": "--", "lw": 1.5},
        "Blend":              {"color": "C5", "marker": "x", "ls": "--", "lw": 1.5},
    }

    for method, style in styles.items():
        if method == "FBS":
            ns = sorted(n for n in numba_fbs if n in baseline_ns)
            times = [numba_fbs[n] for n in ns]
        else:
            mrecs = [r for r in records if r["method"] == method]
            if not mrecs:
                continue
            n_groups = {}
            for r in mrecs:
                n_groups.setdefault(r["n"], []).append(r["time"])
            ns = sorted(n_groups.keys())
            times = [np.median(n_groups[n]) for n in ns]

        ax.loglog(ns, times, label=style.get("label", method), color=style["color"],
                  marker=style["marker"], linestyle=style["ls"],
                  linewidth=style["lw"], markersize=6)

    # FA*IR: merge all variants into one line
    from collections import defaultdict as _dd
    fair_by_n = _dd(list)
    for r in records:
        if r["method"] in ("FA*IR", "FA*IR k=1000"):
            fair_by_n[r["n"]].append(r["time"])
    if fair_by_n:
        ns_f = sorted(fair_by_n.keys())
        ts_f = [np.median(fair_by_n[n]) for n in ns_f]
        ax.loglog(ns_f, ts_f, color="C2", marker="^", ls="--", lw=1.5,
                  markersize=6, label=r"FA*IR")


    ax.set_xlabel("Input size $n$", fontsize=12)
    ax.set_ylabel("Wall-clock time (seconds)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out_pdf = os.path.join(RESULTS_DIR, "fig_scalability_baselines.pdf")
    out_png = os.path.join(RESULTS_DIR, "fig_scalability_baselines.png")
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved {out_pdf}")
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("Usage: python experiment_scalability_baselines.py [run|plot|all]")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd in ("run", "all"):
        run_all()
    if cmd in ("plot", "all"):
        plot()


if __name__ == "__main__":
    main()
