"""
Experiment: Proposition 4 verification for the all-ascending case a = s.

For each n = 2^t (t = 2, ..., t_max), computes:
  - d_Gr via the FBS greedy algorithm
  - LP  = sum of K* smallest pair weights (LP relaxation lower bound)
  - ratio = d_Gr / LP

Plots ratio vs n on a log-scale x-axis with the theoretical limit
(1 + sqrt(2))/2 ≈ 1.207 as a horizontal line.

Usage:
  python experiment_sorted_case.py           # compute + plot (default t_max=17)
  python experiment_sorted_case.py plot      # re-plot from saved data
  python experiment_sorted_case.py 20        # compute up to t=20
"""

import json
import math
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..",
                           "experiments", "approx_ratio_empirical")


# ==================== Greedy (heap-optimised) ====================

def simulate_greedy(n, delta_star):
    """Run FBS greedy for the sorted case a = s = (0,...,n-1).
    Returns the weighted Kemeny cost (sum of |u-v| over swapped pairs).

    Uses a min-heap with lazy deletion for O(K* log n) total time
    instead of O(n * K*) for the naive scan.
    """
    import heapq

    a = list(range(n))
    r = list(range(n))
    D = n * (n - 1) // 2   # all pairs ascending

    # version[k] tracks the current version of position k's heap entry
    version = [0] * (n - 1)

    # Build initial heap: (cost, position_index, version) for ascending pairs
    heap = []
    target = 1  # D > 0 initially, so we flip ascending pairs
    for k in range(n - 1):
        u, v = r[k], r[k + 1]
        sign = 1 if a[u] < a[v] else (-1 if a[u] > a[v] else 0)
        if sign == target:
            cost = abs(u - v)
            heapq.heappush(heap, (cost, k, version[k]))

    total_cost = 0
    while abs(D) > delta_star:
        # Find best valid entry
        while heap:
            cost, k, ver = heapq.heappop(heap)
            if ver == version[k]:
                break
        else:
            return -1  # no valid swap found

        # Perform swap at position k
        r[k], r[k + 1] = r[k + 1], r[k]
        total_cost += cost
        D -= 2 * target

        # Invalidate and re-insert affected positions: k-1, k, k+1
        for pos in (k - 1, k, k + 1):
            if 0 <= pos < n - 1:
                version[pos] += 1
                u, v = r[pos], r[pos + 1]
                sign = 1 if a[u] < a[v] else (-1 if a[u] > a[v] else 0)
                if sign == target:
                    c = abs(u - v)
                    heapq.heappush(heap, (c, pos, version[pos]))

    return total_cost


# ==================== LP relaxation bound ====================

def lp_bound(n, K_star):
    """
    LP relaxation lower bound: sum of the K* smallest pair weights.

    For a = s = (0,...,n-1), all pairs are ascending, so the pair weights
    are w_{ij} = |j - i| for i < j. The LP relaxation (dropping triangle
    inequalities) optimally assigns x=1 to the K* cheapest pairs.

    Weights sorted: gap d appears (n-d) times, d = 1,...,n-1.
    Computed in O(n) by iterating through gaps.
    """
    total = 0
    remaining = K_star
    for d in range(1, n):
        count = n - d
        take = min(count, remaining)
        total += d * take
        remaining -= take
        if remaining == 0:
            break
    return total


# ==================== Main computation ====================

def run_computation(t_max=17):
    """Compute d_Gr and LP bound for n = 2^t, t = 2..t_max."""
    results = {}

    for t in range(2, t_max + 1):
        n = 2 ** t
        cn2 = n * (n - 1) // 2
        delta_star = 0   # full balance (parity is 0 for all n=2^t, t>=2)
        K_star = cn2 // 2

        print(f"\n{'='*60}")
        print(f"t={t}, n={n}, C(n,2)={cn2}, K*={K_star}")

        # Greedy
        t0 = time.time()
        d_gr = simulate_greedy(n, delta_star)
        t_gr = time.time() - t0
        print(f"  Greedy: d_Gr = {d_gr}  ({t_gr:.2f}s)")

        # LP bound
        t0 = time.time()
        d_lp = lp_bound(n, K_star)
        t_lp = time.time() - t0
        print(f"  LP:     LP = {d_lp}  ({t_lp:.2f}s)")

        ratio = d_gr / d_lp if d_lp > 0 else None
        print(f"  Ratio:  d_Gr/LP = {ratio:.6f}" if ratio else "  Ratio:  N/A")

        results[t] = {
            'n': n,
            't': t,
            'cn2': cn2,
            'K_star': K_star,
            'd_gr': d_gr,
            'd_lp': d_lp,
            'ratio': ratio,
            'time_greedy': t_gr,
            'time_lp': t_lp,
        }

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, "sorted_case_ratios.json")
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_path}")

    return results


# ==================== Plotting ====================

def make_plot(results):
    """Generate the sorted-case ratio plot."""
    ts = sorted(int(k) for k in results.keys())
    ns = [results[str(t)]['n'] for t in ts]
    ratios = [results[str(t)]['ratio'] for t in ts]

    theory_limit = (1 + math.sqrt(2)) / 2   # ≈ 1.2071

    fig, ax = plt.subplots(figsize=(5.5, 4))

    ax.plot(ns, ratios, 'o-', color='#2c7bb6', markersize=7, linewidth=1.5,
            label=r'$d_{\mathrm{FBS}}\,/\,\mathrm{LP}$', zorder=3)

    ax.axhline(y=theory_limit, color='#d7191c', linestyle='--', linewidth=1.2,
               label=r'$(1{+}\sqrt{2})/2 \approx 1.207$', zorder=2)

    ax.axhline(y=1.0, color='grey', linestyle=':', linewidth=0.8, alpha=0.5)

    ax.set_xscale('log', base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([f'$2^{{{t}}}$' for t in ts], fontsize=9)
    ax.set_xlabel(r'$n = 2^t$', fontsize=12)
    ax.set_ylabel(r'$d_{\mathrm{FBS}}\,/\,\mathrm{LP}$', fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim(0.95, max(max(ratios), theory_limit) + 0.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        path = os.path.join(RESULTS_DIR, f"fig_sorted_case.{ext}")
        fig.savefig(path, dpi=200)
    print(f"Plot saved to {os.path.join(RESULTS_DIR, 'fig_sorted_case.{{pdf,png}}')}")
    plt.close()


# ==================== Arbitrary-n computation ====================

def run_arbitrary_n(n_max=10000):
    """Compute d_Gr/LP for all n from 4 to n_max (a = s)."""
    # Dense for small n, sparser for large n
    ns = list(range(4, min(n_max + 1, 201)))
    for n in [250, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000,
              15000, 20000, 30000, 50000, 75000]:
        if 200 < n <= n_max and n not in ns:
            ns.append(n)
    ns.sort()

    results = {}
    for n in ns:
        cn2 = n * (n - 1) // 2
        parity = cn2 % 2
        delta_star = parity  # 0 if even, 1 if odd
        K_star = math.ceil((cn2 - delta_star) / 2)

        t0 = time.time()
        d_gr = simulate_greedy(n, delta_star)
        t_gr = time.time() - t0

        d_lp = lp_bound(n, K_star)
        ratio = d_gr / d_lp if d_lp > 0 else None

        results[n] = {
            'n': n, 'cn2': cn2, 'K_star': K_star,
            'd_gr': d_gr, 'd_lp': d_lp, 'ratio': ratio,
            'time_greedy': t_gr,
        }
        print(f"  n={n:>5d}: d_Gr/LP = {ratio:.6f}  ({t_gr:.2f}s)", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, "sorted_case_arbitrary_n.json")
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_path}")
    return results


def make_plot_arbitrary(results):
    """Plot d_Gr/LP vs n for arbitrary n with a = s."""
    ns = sorted(int(k) for k in results.keys())
    ratios = [results[str(n)]['ratio'] for n in ns]

    theory_limit = (1 + math.sqrt(2)) / 2

    fig, ax = plt.subplots(figsize=(5.5, 4))

    ax.plot(ns, ratios, '-', color='#2c7bb6', linewidth=1.2,
            label=r'$d_{\mathrm{FBS}}\,/\,\mathrm{LP}$', zorder=3)

    ax.axhline(y=theory_limit, color='#d7191c', linestyle='--', linewidth=1.2,
               label=r'$(1{+}\sqrt{2})/2 \approx 1.207$', zorder=2)

    ax.axhline(y=1.0, color='grey', linestyle=':', linewidth=0.8, alpha=0.5)

    ax.set_xscale('log')
    ax.set_xlabel(r'$n$', fontsize=12)
    ax.set_ylabel(r'$d_{\mathrm{FBS}}\,/\,\mathrm{LP}$', fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim(0.95, max(max(ratios), theory_limit) + 0.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        path = os.path.join(RESULTS_DIR, f"fig_sorted_case_arbitrary_n.{ext}")
        fig.savefig(path, dpi=200)
    print(f"Plot saved to fig_sorted_case_arbitrary_n.{{pdf,png}}")
    plt.close()


# ==================== Entry point ====================

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"

    if mode == "plot":
        load_path = os.path.join(RESULTS_DIR, "sorted_case_ratios.json")
        with open(load_path) as f:
            results = json.load(f)
        make_plot(results)
    elif mode == "arbitrary":
        n_max = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
        results = run_arbitrary_n(n_max)
        results_str = {str(k): v for k, v in results.items()}
        make_plot_arbitrary(results_str)
    elif mode == "plot_arbitrary":
        load_path = os.path.join(RESULTS_DIR, "sorted_case_arbitrary_n.json")
        with open(load_path) as f:
            results = json.load(f)
        make_plot_arbitrary(results)
    else:
        t_max = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 17
        results = run_computation(t_max)
        results_str = {str(k): v for k, v in results.items()}
        make_plot_results = results_str
        make_plot(make_plot_results)
