"""
Experiment A: Scalability of the fair-ranking greedy.

Two entry points:
  python experiment_scalability.py run     -- measure and save to experiments/
  python experiment_scalability.py plot    -- read experiments/ and produce figures

Every single measurement is appended to a JSONL file immediately,
so partial results are never lost.
"""

import heapq
import json
import math
import os
import random
import sys
import time

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "experiments")
RESULTS_VS_N = os.path.join(RESULTS_DIR, "scalability_vs_n.jsonl")
RESULTS_VS_TAU = os.path.join(RESULTS_DIR, "scalability_vs_tau.jsonl")


# ===================================================================
# Greedy implementations
# ===================================================================

def count_inversions_merge(arr):
    """Count inversions (strict: a[i] > a[j] for i < j) via merge sort.
    Returns (sorted_arr, count)."""
    if len(arr) <= 1:
        return arr[:], 0
    mid = len(arr) // 2
    left, inv_left = count_inversions_merge(arr[:mid])
    right, inv_right = count_inversions_merge(arr[mid:])
    merged = []
    inversions = inv_left + inv_right
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            inversions += len(left) - i
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged, inversions


def count_concordant_merge(arr):
    """Count concordant pairs (strict: a[i] < a[j] for i < j) via merge sort.
    Returns (sorted_arr, count)."""
    if len(arr) <= 1:
        return arr[:], 0
    mid = len(arr) // 2
    left, conc_left = count_concordant_merge(arr[:mid])
    right, conc_right = count_concordant_merge(arr[mid:])
    merged = []
    concordant = conc_left + conc_right
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            merged.append(left[i])
            concordant += len(right) - j
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged, concordant


def _sign(x, y):
    """Return +1 if y > x, -1 if y < x, 0 if y == x."""
    if y > x:
        return 1
    elif y < x:
        return -1
    return 0


def greedy_heap(s, a, tau):
    """Heap-based greedy (Algorithm 1). Handles ties in attributes.
    Returns (total_cost, k_star_swaps)."""
    n = len(s)
    c = [abs(s[k] - s[k + 1]) for k in range(n - 1)]
    sigma = [_sign(a[k], a[k + 1]) for k in range(n - 1)]

    # Compute N_up and N_down independently (ties are excluded from both)
    _, n_up = count_concordant_merge(a)
    _, n_down = count_inversions_merge(a)

    delta_star = int(math.floor((1 - tau) * n * (n - 1) / 2))
    d0 = abs(n_up - n_down)
    # Parity fix: D changes by ±2 per swap, so D ≡ D0 (mod 2) always.
    # Adjust delta_star to be reachable.
    if delta_star % 2 != d0 % 2:
        delta_star += 1
    if d0 <= delta_star:
        return 0.0, 0

    mu = 1 if n_up > n_down else -1

    # Lazy-deletion min-heap keyed by (cost, index)
    entry_map = {}
    heap = []

    def heap_push(k):
        entry = [c[k], k, False]
        entry_map[k] = entry
        heapq.heappush(heap, entry)

    def heap_remove(k):
        if k in entry_map:
            entry_map[k][2] = True
            del entry_map[k]

    def heap_pop():
        while heap:
            cost, k, removed = heapq.heappop(heap)
            if not removed:
                del entry_map[k]
                return k, cost
        return None, None

    def heap_update(k):
        heap_remove(k)
        heap_push(k)

    for k in range(n - 1):
        if sigma[k] == mu:
            heap_push(k)

    total_cost = 0.0
    swaps = 0

    while abs(n_up - n_down) > delta_star:
        k_star, cost_star = heap_pop()
        if k_star is None:
            break

        s[k_star], s[k_star + 1] = s[k_star + 1], s[k_star]
        a[k_star], a[k_star + 1] = a[k_star + 1], a[k_star]
        total_cost += cost_star
        swaps += 1

        for ell in (k_star - 1, k_star, k_star + 1):
            if ell < 0 or ell >= n - 1:
                continue
            old_sigma = sigma[ell]
            c[ell] = abs(s[ell] - s[ell + 1])
            if ell == k_star:
                # The swapped pair: strict inequality reverses
                sigma[ell] = -sigma[ell]
            else:
                # Neighbours: recompute (may become 0 = tied)
                sigma[ell] = _sign(a[ell], a[ell + 1])

            was_in = (old_sigma == mu)
            now_in = (sigma[ell] == mu)
            if was_in and not now_in:
                heap_remove(ell)
            elif not was_in and now_in:
                heap_push(ell)
            elif now_in:
                heap_update(ell)

        n_up -= mu
        n_down += mu

    return total_cost, swaps


def greedy_naive(s, a, tau):
    """Naive linear-scan greedy. Handles ties. Returns (total_cost, k_star_swaps)."""
    n = len(s)
    c = [abs(s[k] - s[k + 1]) for k in range(n - 1)]
    sigma = [_sign(a[k], a[k + 1]) for k in range(n - 1)]

    _, n_up = count_concordant_merge(a)
    _, n_down = count_inversions_merge(a)

    delta_star = int(math.floor((1 - tau) * n * (n - 1) / 2))
    d0 = abs(n_up - n_down)
    if delta_star % 2 != d0 % 2:
        delta_star += 1
    if d0 <= delta_star:
        return 0.0, 0

    mu = 1 if n_up > n_down else -1
    total_cost = 0.0
    swaps = 0

    while abs(n_up - n_down) > delta_star:
        best_k = -1
        best_c = float('inf')
        for k in range(n - 1):
            if sigma[k] == mu and c[k] < best_c:
                best_c = c[k]
                best_k = k
        if best_k == -1:
            break

        k_star = best_k
        s[k_star], s[k_star + 1] = s[k_star + 1], s[k_star]
        a[k_star], a[k_star + 1] = a[k_star + 1], a[k_star]
        total_cost += best_c
        swaps += 1

        for ell in (k_star - 1, k_star, k_star + 1):
            if ell < 0 or ell >= n - 1:
                continue
            c[ell] = abs(s[ell] - s[ell + 1])
            if ell == k_star:
                sigma[ell] = -sigma[ell]
            else:
                sigma[ell] = _sign(a[ell], a[ell + 1])

        n_up -= mu
        n_down += mu

    return total_cost, swaps


# ===================================================================
# Instance generation
# ===================================================================

def generate_instance(n, correlation=1.0):
    """
    Generate (s, a) of length n, scores in descending order.
    correlation in [0, 1]: how correlated a is with s.
      1   -> a = s (all pairs concordant, D0 = C(n,2))
      0.5 -> moderate scramble
      0   -> fully random permutation
    """
    s = [float(n - i) for i in range(n)]
    if correlation >= 0.99:
        a = s[:]
    elif correlation <= 0.01:
        a = list(range(1, n + 1))
        random.shuffle(a)
        a = [float(x) for x in a]
    else:
        a = s[:]
        n_scramble = int((1 - correlation) * n * n / 4)
        for _ in range(n_scramble):
            k = random.randint(0, n - 2)
            a[k], a[k + 1] = a[k + 1], a[k]
    return s, a


# ===================================================================
# Measurement helpers
# ===================================================================

def append_result(path, record):
    """Append one JSON record to a JSONL file (one line per record)."""
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_results(path):
    """Load all records from a JSONL file."""
    records = []
    if not os.path.exists(path):
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def measure_one(greedy_fn, n, tau, correlation, repeats, method_name):
    """Run greedy_fn `repeats` times, return a result dict with median time."""
    times = []
    k_stars = []
    costs = []
    for rep in range(repeats):
        s, a = generate_instance(n, correlation)
        t0 = time.perf_counter()
        cost, k_star = greedy_fn(s, a, tau)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        k_stars.append(k_star)
        costs.append(cost)

    import statistics
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
# Runner: measure and save
# ===================================================================

def run_vs_n():
    """Measure runtime vs n for several tau values. Save incrementally."""
    # Clear previous results for this experiment
    if os.path.exists(RESULTS_VS_N):
        os.rename(RESULTS_VS_N, RESULTS_VS_N + ".bak")

    ns = [100, 300, 1000, 3000, 10000, 30000, 100000]
    taus = [0.5, 0.8, 0.95, 1.0]
    correlation = 1.0  # a = s, maximises D0

    print("=" * 60)
    print("Experiment A.1: Runtime vs n  (correlation=1.0)")
    print("=" * 60)

    # --- Heap-based greedy ---
    for tau in taus:
        for n in ns:
            # K* = ceil((D0 - delta*)/2) with D0 = C(n,2) for correlation=1
            d0 = n * (n - 1) // 2
            delta_star = int(math.floor((1 - tau) * n * (n - 1) / 2))
            k_est = max(0, math.ceil((d0 - delta_star) / 2))
            if k_est > 5_000_000:
                print(f"  SKIP heap n={n}, tau={tau} (K*~{k_est} too large)")
                continue
            repeats = 1
            print(f"  heap: n={n:>7d}, tau={tau} ...", end="", flush=True)
            rec = measure_one(greedy_heap, n, tau, correlation, repeats, "heap")
            append_result(RESULTS_VS_N, rec)
            print(f" {rec['time_median']:.4f}s, K*={rec['k_star_median']}")

    # --- Naive greedy (tau=0.8 only, for comparison) ---
    print("\n--- Naive (tau=0.8) ---")
    for n in ns:
        d0 = n * (n - 1) // 2
        delta_star = int(math.floor(0.2 * n * (n - 1) / 2))
        k_est = max(0, math.ceil((d0 - delta_star) / 2))
        if k_est > 500_000:
            print(f"  SKIP naive n={n} (K*~{k_est} too large)")
            continue
        repeats = 1
        print(f"  naive: n={n:>7d} ...", end="", flush=True)
        rec = measure_one(greedy_naive, n, 0.8, correlation, repeats, "naive")
        append_result(RESULTS_VS_N, rec)
        print(f" {rec['time_median']:.4f}s, K*={rec['k_star_median']}")

    print(f"\nResults saved to {RESULTS_VS_N}")


def run_vs_tau():
    """Measure runtime vs tau for fixed n. Save incrementally."""
    if os.path.exists(RESULTS_VS_TAU):
        os.rename(RESULTS_VS_TAU, RESULTS_VS_TAU + ".bak")

    n_fixed = 10000
    taus = [round(t, 2) for t in list(
        [i * 0.05 for i in range(21)]
    )]
    correlation = 1.0

    print("\n" + "=" * 60)
    print(f"Experiment A.2: Runtime vs tau  (n={n_fixed}, correlation=1.0)")
    print("=" * 60)

    for tau in taus:
        print(f"  tau={tau:.2f} ...", end="", flush=True)
        rec = measure_one(greedy_heap, n_fixed, tau, correlation, 5, "heap")
        append_result(RESULTS_VS_TAU, rec)
        print(f" {rec['time_median']:.4f}s, K*={rec['k_star_median']}")

    print(f"\nResults saved to {RESULTS_VS_TAU}")


# ===================================================================
# Plotter: read saved results and produce figures
# ===================================================================

def plot_vs_n():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    records = load_results(RESULTS_VS_N)
    if not records:
        print(f"No data in {RESULTS_VS_N}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    tau_labels = {0.5: r"$\tau=0.5$", 0.8: r"$\tau=0.8$",
                  1.0: r"$\tau=1.0$"}
    colours = {0.5: "C0", 0.8: "C1", 1.0: "C3"}

    # Plot heap results per tau (correlation=1, slow regime)
    for tau in [0.5, 0.8, 1.0]:
        pts = [(r["n"], r["time_median"], r["k_star_median"])
               for r in records if r["method"] == "heap"
               and abs(r["tau"] - tau) < 0.001
               and r.get("correlation", 1.0) > 0.5]
        if not pts:
            continue
        pts.sort()
        ns, ts, ks = zip(*pts)
        label = f"$a=s$, {tau_labels.get(tau, str(tau))}"
        ax.plot(ns, ts, "o-", color=colours.get(tau, "C4"),
                label=label, linewidth=1.5, markersize=4)

    # Plot naive results
    pts_naive = [(r["n"], r["time_median"])
                 for r in records if r["method"] == "naive"]
    if pts_naive:
        pts_naive.sort()
        ns_n, ts_n = zip(*pts_naive)
        ax.plot(ns_n, ts_n, "s--", color="C1", alpha=0.5,
                label=r"Naive $O(nK^*)$, $\tau=0.8$",
                linewidth=1.2, markersize=4)

    # Plot fast regime (correlation=0, tau=1)
    pts_fast = [(r["n"], r["time_median"], r["k_star_median"])
                for r in records if r["method"] == "heap"
                and r.get("correlation", 1.0) < 0.5]
    if pts_fast:
        pts_fast.sort()
        ns_f, ts_f, ks_f = zip(*pts_fast)
        ax.plot(ns_f, ts_f, "D-", color="C4", linewidth=1.5, markersize=4,
                label=r"$a$ random, $\tau=1$")

    # Reference slopes — extend to cover both regimes
    ns_ref = np.array([100, 1000000])
    # Anchor n log n to the fast-regime data
    if pts_fast:
        n0, t0, _ = pts_fast[0]
        ref = ns_ref * np.log2(ns_ref)
        ref = ref / (n0 * np.log2(n0)) * t0
        ax.plot(ns_ref, ref, ":", color="gray", linewidth=1,
                label=r"$\propto n\log n$")
    # Anchor n^2 log n to the slow-regime data
    heap_tau1 = [(r["n"], r["time_median"]) for r in records
                 if r["method"] == "heap" and abs(r["tau"] - 1.0) < 0.001
                 and r.get("correlation", 1.0) > 0.5]
    if heap_tau1:
        heap_tau1.sort()
        n0, t0 = heap_tau1[0]
        ref2 = ns_ref**2 * np.log2(ns_ref)
        ref2 = ref2 / (n0**2 * np.log2(n0)) * t0
        ax.plot(ns_ref, ref2, "-.", color="gray", linewidth=1,
                label=r"$\propto n^2\log n$")
    # n^3 — anchor to naive data
    if pts_naive:
        n0_n, t0_n = pts_naive[0]
        ref3 = ns_ref.astype(float)**3
        ref3 = ref3 / (float(n0_n)**3) * t0_n
        ax.plot(ns_ref, ref3, "--", color="gray", linewidth=1, alpha=0.5,
                label=r"$\propto n^3$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of items $n$", fontsize=12)
    ax.set_ylabel("Wall-clock time (seconds)", fontsize=12)
    ax.set_title("Scalability of the fair-ranking greedy", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    out_pdf = os.path.join(RESULTS_DIR, "fig_scalability_vs_n.pdf")
    out_png = os.path.join(RESULTS_DIR, "fig_scalability_vs_n.png")
    fig.savefig(out_pdf, dpi=150)
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_pdf}")
    plt.close(fig)


def plot_vs_tau():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = load_results(RESULTS_VS_TAU)
    if not records:
        print(f"No data in {RESULTS_VS_TAU}")
        return

    pts = [(r["tau"], r["time_median"], r["k_star_median"]) for r in records]
    pts.sort()
    taus, times, ks = zip(*pts)

    fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))

    color1 = "C0"
    ax1.plot(taus, times, "o-", color=color1, linewidth=1.5, markersize=4)
    ax1.set_xlabel(r"Fairness threshold $\tau$", fontsize=12)
    ax1.set_ylabel("Wall-clock time (seconds)", fontsize=12, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "C3"
    ax2.plot(taus, ks, "s--", color=color2, linewidth=1, markersize=3,
             alpha=0.7)
    ax2.set_ylabel(r"Number of swaps $K^*$", fontsize=12, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    n_fixed = records[0].get("n", "?")
    ax1.set_title(f"Runtime vs fairness threshold ($n={n_fixed:,}$, $a=s$)",
                  fontsize=13)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    out_pdf = os.path.join(RESULTS_DIR, "fig_scalability_vs_tau.pdf")
    out_png = os.path.join(RESULTS_DIR, "fig_scalability_vs_tau.png")
    fig.savefig(out_pdf, dpi=150)
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_pdf}")
    plt.close(fig)


# ===================================================================
# Main
# ===================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if len(sys.argv) < 2:
        print("Usage: python experiment_scalability.py [run|plot|all]")
        print("  run  -- run measurements, save to experiments/*.jsonl")
        print("  plot -- read experiments/*.jsonl and produce figures")
        print("  all  -- run then plot")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd in ("run", "all"):
        random.seed(42)
        run_vs_n()
        run_vs_tau()

    if cmd in ("plot", "all"):
        plot_vs_n()
        plot_vs_tau()


if __name__ == "__main__":
    main()
