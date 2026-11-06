"""
Experiment: Scalability of the Numba-accelerated FBS (Figure 4 redo).

Entry points:
  python experiment_scalability_numba.py run   -- run all measurements
  python experiment_scalability_numba.py plot  -- produce figure from saved data
  python experiment_scalability_numba.py all   -- run then plot

Results saved to experiments/scalability_numba/
"""

import json
import os
import random
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from fbs_numba import (greedy_heap, greedy_naive, greedy_unweighted,
                        warm_up, scramble_adjacent)


def generate_instance(n, correlation=1.0):
    """Fast instance generator (Numba-accelerated scrambling)."""
    s = [float(n - i) for i in range(n)]
    if correlation >= 0.99:
        a = s[:]
    elif correlation <= 0.01:
        a = list(range(1, n + 1))
        random.shuffle(a)
        a = [float(x) for x in a]
    else:
        a_arr = np.array(s)
        n_scramble = int((1 - correlation) * n * n / 4)
        scramble_adjacent(a_arr, n_scramble, n)
        a = a_arr.tolist()
    return s, a

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..",
                           "experiments", "scalability_numba")
RESULTS_FILE = os.path.join(RESULTS_DIR, "scalability_vs_n.jsonl")


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


def measure_one(greedy_fn, n, tau, correlation, repeats, method_name):
    times = []
    k_stars = []
    costs = []
    for _ in range(repeats):
        s, a = generate_instance(n, correlation)
        t0 = time.perf_counter()
        cost, k_star = greedy_fn(s, a, tau)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        k_stars.append(k_star)
        costs.append(cost)
    return {
        "method": method_name,
        "n": n,
        "tau": tau,
        "correlation": correlation,
        "repeats": repeats,
        "time_median": statistics.median(times),
        "time_min": min(times),
        "time_max": max(times),
        "k_star_median": int(statistics.median(k_stars)),
        "cost_median": statistics.median(costs),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ===================================================================
# Runner
# ===================================================================

def _already_done(records, method, n, correlation):
    """Check if a (method, n, correlation) combo already exists in records."""
    for r in records:
        if (r["method"] == method and r["n"] == n
                and abs(r.get("correlation", 1.0) - correlation) < 0.01):
            return True
    return False


def run_all():
    random.seed(42)

    print("Warming up Numba JIT...", flush=True)
    warm_up()
    print("Done.\n")

    existing = load_results()

    # --- Weighted heap, a = s (corr=1.0), psi=1.0 ---
    ns_corr = [100, 300, 1000, 3000, 10000, 30000, 100000, 300000]

    print("=" * 60)
    print("Weighted heap, a=s (corr=1.0), psi=1.0")
    print("=" * 60)
    for n in ns_corr:
        if _already_done(existing, "heap", n, 1.0):
            print(f"  heap: n={n:>7d} ... SKIP (already done)")
            continue
        print(f"  heap: n={n:>7d} ...", end="", flush=True)
        rec = measure_one(greedy_heap, n, 1.0, 1.0, 1, "heap")
        append_result(rec)
        print(f" {rec['time_median']:.4f}s, K*={rec['k_star_median']}")

    # --- Weighted heap, corr=0.8, psi=1.0 ---
    print("\n" + "=" * 60)
    print("Weighted heap, corr=0.8, psi=1.0")
    print("=" * 60)
    for n in ns_corr:
        if _already_done(existing, "heap", n, 0.8):
            print(f"  heap: n={n:>7d} ... SKIP (already done)")
            continue
        print(f"  heap: n={n:>7d} ...", end="", flush=True)
        rec = measure_one(greedy_heap, n, 1.0, 0.8, 1, "heap")
        append_result(rec)
        print(f" {rec['time_median']:.4f}s, K*={rec['k_star_median']}")

    # --- Naive greedy, a=s (corr=1.0), psi=1.0 ---
    ns_naive = [100, 300, 1000, 3000, 10000]

    print("\n--- Naive O(nK*), a=s, psi=1.0 ---")
    for n in ns_naive:
        if _already_done(existing, "naive", n, 1.0):
            print(f"  naive: n={n:>7d} ... SKIP (already done)")
            continue
        print(f"  naive: n={n:>7d} ...", end="", flush=True)
        rec = measure_one(greedy_naive, n, 1.0, 1.0, 1, "naive")
        append_result(rec)
        print(f" {rec['time_median']:.4f}s, K*={rec['k_star_median']}")

    # --- Unweighted linked-list, a=s (corr=1.0), psi=1.0 ---
    ns_unw = [100, 300, 1000, 3000, 10000, 30000, 100000, 300000]

    print("\n--- Unweighted O(K*), a=s, psi=1.0 ---")
    for n in ns_unw:
        if _already_done(existing, "unweighted", n, 1.0):
            print(f"  unweighted: n={n:>7d} ... SKIP (already done)")
            continue
        print(f"  unweighted: n={n:>7d} ...", end="", flush=True)
        rec = measure_one(greedy_unweighted, n, 1.0, 1.0, 1, "unweighted")
        append_result(rec)
        print(f" {rec['time_median']:.4f}s, K*={rec['k_star_median']}")

    # --- Random regime: a random, psi=1.0 ---
    ns_rand = [100, 300, 1000, 3000, 10000, 30000, 100000,
               300000, 1000000, 3000000]

    print("\n--- Random regime (a random), psi=1.0, heap ---")
    for n in ns_rand:
        if _already_done(existing, "heap", n, 0.0):
            print(f"  heap: n={n:>7d} ... SKIP (already done)")
            continue
        print(f"  heap: n={n:>7d} ...", end="", flush=True)
        rec = measure_one(greedy_heap, n, 1.0, 0.0, 1, "heap")
        append_result(rec)
        print(f" {rec['time_median']:.4f}s, K*={rec['k_star_median']}")

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

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    # --- Heap, a=s (corr=1.0) ---
    pts_corr = [(r["n"], r["time_median"])
                for r in records if r["method"] == "heap"
                and abs(r["tau"] - 1.0) < 0.001
                and abs(r.get("correlation", 1.0) - 1.0) < 0.01]
    if pts_corr:
        pts_corr.sort()
        ns_c, ts_c = zip(*pts_corr)
        ax.plot(ns_c, ts_c, "o-", color="C3", linewidth=2.5, markersize=6,
                label=r"Weighted, $a=s$ (heap)")

    # --- Heap, corr=0.8 ---
    pts_corr08 = [(r["n"], r["time_median"])
                  for r in records if r["method"] == "heap"
                  and abs(r["tau"] - 1.0) < 0.001
                  and abs(r.get("correlation", 1.0) - 0.8) < 0.01]
    if pts_corr08:
        pts_corr08.sort()
        ns_08, ts_08 = zip(*pts_corr08)
        ax.plot(ns_08, ts_08, "^-", color="C0", linewidth=2.5, markersize=6,
                label=r"Weighted, $\mathrm{corr}=0.8$ (heap)")

    # --- Unweighted linked-list, a=s ---
    pts_unw = [(r["n"], r["time_median"])
               for r in records if r["method"] == "unweighted"]
    if pts_unw:
        pts_unw.sort()
        ns_u, ts_u = zip(*pts_unw)
        ax.plot(ns_u, ts_u, "v-", color="C2", linewidth=2.5, markersize=6,
                label=r"Unweighted, $a=s$ (linked list)")

    # --- Naive, a=s ---
    pts_naive = [(r["n"], r["time_median"])
                 for r in records if r["method"] == "naive"]
    if pts_naive:
        pts_naive.sort()
        ns_n, ts_n = zip(*pts_naive)
        ax.plot(ns_n, ts_n, "s--", color="C1", linewidth=2.5, markersize=6,
                label=r"Weighted, $a=s$ (naive)")

    # --- Heap, random (with error bars if per-run data available) ---
    rand_recs = [r for r in records if r["method"] == "heap"
                 and r.get("correlation", 1.0) < 0.01]
    per_run = [r for r in rand_recs if "run" in r]
    if per_run:
        from collections import defaultdict
        by_n = defaultdict(list)
        for r in per_run:
            by_n[r["n"]].append(r["time"])
        ns_r = sorted(by_n.keys())
        means = [np.mean(by_n[n]) for n in ns_r]
        stds = [np.std(by_n[n]) for n in ns_r]
        ax.errorbar(ns_r, means, yerr=stds, fmt="D-", color="C4",
                    linewidth=2.5, markersize=6, capsize=4, capthick=1.5,
                    label=r"Weighted, $a$ random (heap)")
        pts_rand = list(zip(ns_r, means))
    else:
        pts_rand = [(r["n"], r["time_median"]) for r in rand_recs]
        if pts_rand:
            pts_rand.sort()
            ns_r, ts_r = zip(*pts_rand)
            ax.plot(ns_r, ts_r, "D-", color="C4", linewidth=2.5, markersize=6,
                    label=r"Weighted, $a$ random (heap)")

    # --- Reference slopes ---
    ns_ref = np.array([50, 10_000_000], dtype=np.float64)

    # n log n
    if pts_rand and len(pts_rand) >= 3:
        n0, t0 = pts_rand[len(pts_rand) // 2]
        ref = ns_ref * np.log2(ns_ref)
        ref = ref / (n0 * np.log2(n0)) * t0
        ax.plot(ns_ref, ref, ":", color="gray", linewidth=1.5,
                label=r"$\propto n\log n$")

    # n^2
    if pts_unw and len(pts_unw) >= 3:
        n0, t0 = pts_unw[len(pts_unw) // 2]
        ref_n2 = ns_ref**2
        ref_n2 = ref_n2 / (float(n0)**2) * t0
        ax.plot(ns_ref, ref_n2, linestyle=(0, (5, 3)), color="gray", linewidth=1.5,
                label=r"$\propto n^2$")

    # n^2 log n
    if pts_corr and len(pts_corr) >= 3:
        n0, t0 = pts_corr[len(pts_corr) // 2]
        ref2 = ns_ref**2 * np.log2(ns_ref)
        ref2 = ref2 / (n0**2 * np.log2(n0)) * t0
        ax.plot(ns_ref, ref2, "-.", color="gray", linewidth=1.5,
                label=r"$\propto n^2\log n$")

    # n^3
    if pts_naive and len(pts_naive) >= 3:
        n0_n, t0_n = pts_naive[len(pts_naive) // 2]
        ref3 = ns_ref**3
        ref3 = ref3 / (float(n0_n)**3) * t0_n
        ax.plot(ns_ref, ref3, "--", color="gray", linewidth=1.5, alpha=0.5,
                label=r"$\propto n^3$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of items $n$", fontsize=15)
    ax.set_ylabel("Wall-clock time (seconds)", fontsize=15)
    ax.tick_params(axis="both", labelsize=13)
    handles, labels = ax.get_legend_handles_labels()
    method_h, method_l, ref_h, ref_l = [], [], [], []
    for h, l in zip(handles, labels):
        if l.startswith(r"$\propto"):
            ref_h.append(h); ref_l.append(l)
        else:
            method_h.append(h); method_l.append(l)
    leg1 = ax.legend(method_h, method_l,
                     fontsize=11, loc="upper left", ncol=1)
    ax.add_artist(leg1)
    ax.legend(ref_h, ref_l,
              fontsize=10, loc="lower right", ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    out_pdf = os.path.join(RESULTS_DIR, "fig_scalability_vs_n.pdf")
    out_png = os.path.join(RESULTS_DIR, "fig_scalability_vs_n.png")
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    plt.close(fig)


# ===================================================================
# Main
# ===================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python experiment_scalability_numba.py [run|plot|all]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd in ("run", "all"):
        run_all()
    if cmd in ("plot", "all"):
        plot_comparison()


if __name__ == "__main__":
    main()
