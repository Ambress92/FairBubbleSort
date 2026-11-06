"""
Experiment: Empirical approximation ratio of FBS.

Part 1 (exact):  For n=3..8, exhaustive enumeration of all instances.
                 Saves every d_Gr/d* ratio for histogram.

Part 2 (LP bound): For n in {20, 50, 100}, random/correlated/adversarial instances.
                   Solves LP relaxation for d*_LP lower bound.
                   Computes d_Gr/d*_LP upper bound on true ratio.

Results saved to experiments/approx_ratio_empirical/
Plots saved there too.

Usage:
  python experiment_approx_ratio.py          # run everything
  python experiment_approx_ratio.py small    # only Part 1
  python experiment_approx_ratio.py large    # only Part 2
  python experiment_approx_ratio.py plot     # only re-generate plots from saved data
"""

import itertools
import json
import math
import os
import sys
import time
from itertools import combinations

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "approx_ratio_empirical")


# ==================== Shared utilities ====================

def simulate_greedy(a, n, delta_star):
    """Run FBS greedy: at each step pick the min-cost majority-sign adjacent swap."""
    r = list(range(n))
    D = 0
    for i in range(n):
        for j in range(i + 1, n):
            if a[r[i]] < a[r[j]]:
                D += 1
            elif a[r[i]] > a[r[j]]:
                D -= 1
    total_cost = 0

    while abs(D) > delta_star:
        best_k = -1
        best_cost = float('inf')
        target = +1 if D > 0 else -1
        for k in range(n - 1):
            u, v = r[k], r[k + 1]
            sign = 1 if a[u] < a[v] else (-1 if a[u] > a[v] else 0)
            if sign == target:
                c = abs(u - v)
                if c < best_cost:
                    best_cost = c
                    best_k = k
        if best_k == -1:
            return None, float('inf')
        r[best_k], r[best_k + 1] = r[best_k + 1], r[best_k]
        total_cost += best_cost
        D -= 2 * target

    return r, total_cost


# ==================== Part 1: Exhaustive small n ====================

def run_exhaustive(n):
    """Exhaustive computation for given n. Returns dict with all ratios."""
    cn2 = n * (n - 1) // 2
    parity = cn2 % 2

    all_perms = np.array(list(itertools.permutations(range(n))), dtype=np.int16)
    positions = np.arange(n, dtype=np.int32)
    all_costs = np.sum((all_perms.astype(np.int32) - positions) ** 2, axis=1) // 2
    pairs_i, pairs_j = np.triu_indices(n, k=1)

    ratios = []
    n_optimal = 0
    n_instances = 0
    worst_ratio = 0.0
    worst_info = None

    t0 = time.time()
    attr_perms = list(itertools.permutations(range(n)))
    last_print = t0

    for ap_idx, a_perm in enumerate(attr_perms):
        a = list(a_perm)
        D0 = 0
        for i in range(n):
            for j in range(i + 1, n):
                if a[i] < a[j]:
                    D0 += 1
                elif a[i] > a[j]:
                    D0 -= 1
        if D0 < 0:  # reversal symmetry
            continue
        if D0 == 0:
            continue

        a_arr = np.array(a, dtype=np.int16)
        b = a_arr[all_perms]
        diffs = b[:, pairs_j].astype(np.int16) - b[:, pairs_i].astype(np.int16)
        D_all = np.sum(np.sign(diffs), axis=1)

        seen_K = set()
        for delta_star in range(parity, D0, 2):
            K_star = math.ceil((D0 - delta_star) / 2)
            if K_star in seen_K:
                continue
            seen_K.add(K_star)

            feasible_mask = np.abs(D_all) <= delta_star
            if not np.any(feasible_mask):
                continue
            cost_opt = int(np.min(all_costs[feasible_mask]))

            _, cost_gr = simulate_greedy(a, n, delta_star)
            if cost_gr == float('inf'):
                continue

            n_instances += 1

            if cost_opt > 0:
                ratio = cost_gr / cost_opt
                ratios.append(ratio)
                if abs(ratio - 1.0) < 1e-9:
                    n_optimal += 1
                if ratio > worst_ratio:
                    worst_ratio = ratio
                    worst_info = {
                        'a': a,
                        'delta_star': delta_star,
                        'K_star': K_star,
                        'cost_gr': int(cost_gr),
                        'cost_opt': cost_opt,
                    }
            else:
                n_optimal += 1

        now = time.time()
        if now - last_print > 30:
            last_print = now
            pct = 100 * (ap_idx + 1) / len(attr_perms)
            print(f"    n={n}: {ap_idx+1}/{len(attr_perms)} ({pct:.0f}%), "
                  f"instances={n_instances}, worst={worst_ratio:.4f}", flush=True)

    return {
        'n': n,
        'n_instances': n_instances,
        'n_with_ratio': len(ratios),
        'n_optimal': n_optimal,
        'ratios': ratios,
        'worst_ratio': worst_ratio,
        'worst_info': worst_info,
    }


def run_part1():
    """Run exhaustive computation for n=3..8."""
    print("=" * 70)
    print("PART 1: Exhaustive small-n computation")
    print("=" * 70, flush=True)

    results = {}
    for n in range(3, 9):
        t0 = time.time()
        print(f"\nn={n}...", flush=True)
        res = run_exhaustive(n)
        elapsed = time.time() - t0
        res['elapsed'] = elapsed
        results[n] = res
        print(f"  n={n}: {res['n_instances']} instances, "
              f"{res['n_optimal']} greedy-optimal, "
              f"worst ratio = {res['worst_ratio']:.6f}, "
              f"elapsed = {elapsed:.1f}s", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, "exhaustive_ratios.json")
    with open(save_path, 'w') as f:
        json.dump({str(n): r for n, r in results.items()}, f, indent=2)
    print(f"\nPart 1 results saved to {save_path}", flush=True)
    return results


# ==================== Part 2: LP lower bound for large n ====================

def pair_idx(i, j, n):
    """Map pair (i, j) with i < j to linear index in {0, ..., C(n,2)-1}."""
    return i * n - i * (i + 1) // 2 + (j - i - 1)


def solve_lp(a, n, delta_star):
    """
    Solve LP relaxation of the minimum-cost ranking problem.

    Variables x_{ij} ∈ [0,1] for i<j: fraction of inversion of pair (i,j).
    Objective: min Σ_{i<j} (j-i) x_{ij}   (weighted Kemeny distance)
    Fairness:  |D0 - 2 Σ α_{ij} x_{ij}| ≤ Δ*
    Transitivity: x_{ik} ≤ x_{ij} + x_{jk},  x_{ij} + x_{jk} ≤ 1 + x_{ik}

    Returns (lp_cost, D0) or (None, D0) if infeasible/timeout.
    """
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    m = n * (n - 1) // 2

    # Pairs and objective
    pairs = np.array(list(combinations(range(n), 2)), dtype=np.int32)
    pi, pj = pairs[:, 0], pairs[:, 1]
    c = (pj - pi).astype(np.float64)

    a_arr = np.array(a)
    alpha = np.sign(a_arr[pj] - a_arr[pi]).astype(np.float64)
    D0 = int(np.sum(alpha))

    # Triple indices for transitivity constraints
    triples = np.array(list(combinations(range(n), 3)), dtype=np.int32)
    n_tri = len(triples)
    ii, jj, kk = triples[:, 0], triples[:, 1], triples[:, 2]

    ij = ii * n - ii * (ii + 1) // 2 + jj - ii - 1
    jk = jj * n - jj * (jj + 1) // 2 + kk - jj - 1
    ik = ii * n - ii * (ii + 1) // 2 + kk - ii - 1

    n_constraints = 2 + 2 * n_tri

    # --- Build sparse constraint matrix in COO format ---

    # Fairness row 0:  -α^T x ≤ -(D0 - Δ*)/2
    f0_rows = np.zeros(m, dtype=np.int32)
    f0_cols = np.arange(m, dtype=np.int32)
    f0_vals = -alpha

    # Fairness row 1:  α^T x ≤ (D0 + Δ*)/2
    f1_rows = np.ones(m, dtype=np.int32)
    f1_cols = np.arange(m, dtype=np.int32)
    f1_vals = alpha.copy()

    # Triangle type A (rows 2,4,6,...): x_ij + x_jk - x_ik ≤ 1
    base = np.arange(n_tri, dtype=np.int32)
    tA_rows = np.repeat(2 + 2 * base, 3)
    tA_cols = np.column_stack([ij, jk, ik]).ravel()
    tA_vals = np.tile([1.0, 1.0, -1.0], n_tri)

    # Triangle type B (rows 3,5,7,...): -x_ij - x_jk + x_ik ≤ 0
    tB_rows = np.repeat(3 + 2 * base, 3)
    tB_cols = np.column_stack([ij, jk, ik]).ravel()
    tB_vals = np.tile([-1.0, -1.0, 1.0], n_tri)

    all_rows = np.concatenate([f0_rows, f1_rows, tA_rows, tB_rows])
    all_cols = np.concatenate([f0_cols, f1_cols, tA_cols, tB_cols])
    all_vals = np.concatenate([f0_vals, f1_vals, tA_vals, tB_vals])

    A = coo_matrix((all_vals, (all_rows, all_cols)),
                   shape=(n_constraints, m)).tocsc()

    b = np.zeros(n_constraints)
    b[0] = -(D0 - delta_star) / 2.0
    b[1] = (D0 + delta_star) / 2.0
    b[2::2] = 1.0

    result = linprog(c, A_ub=A, b_ub=b, bounds=[(0, 1)] * m,
                     method='highs', options={'presolve': True, 'time_limit': 600})

    if result.success:
        return result.fun, D0
    else:
        return None, D0


def generate_instances(n, n_random=30, n_corr=20, seed=42):
    """Generate test instances. Yields (a, type_name) tuples.

    Note: anti-correlated is omitted because by reversal symmetry
    (a[i] ↔ (n-1)-a[i]) it is equivalent to correlated.
    """
    rng = np.random.RandomState(seed)

    # Random permutations
    for _ in range(n_random):
        yield list(rng.permutation(n)), 'random'

    # Correlated: a ≈ scores + noise
    for _ in range(n_corr):
        noise = rng.normal(0, n * 0.2, n)
        vals = np.arange(n, dtype=float) + noise
        yield list(np.argsort(np.argsort(vals))), 'correlated'

    # Fully reversed
    yield list(range(n - 1, -1, -1)), 'reversed'


def run_part2():
    """Run LP-bounded computation for large n."""
    print("\n" + "=" * 70)
    print("PART 2: LP lower bound for large n")
    print("=" * 70, flush=True)

    configs = [
        (20,  30, 20),
        (50,  30, 20),
        (100, 30, 20),
    ]

    results = []

    for n, n_rand, n_corr in configs:
        cn2 = n * (n - 1) // 2
        parity = cn2 % 2

        instances = list(generate_instances(n, n_rand, n_corr))
        print(f"\nn={n}, {len(instances)} instances, C(n,2)={cn2}, parity={parity}",
              flush=True)

        for idx, (a, itype) in enumerate(instances):
            # Compute D0
            a_arr = np.array(a)
            pairs = np.array(list(combinations(range(n), 2)), dtype=np.int32)
            D0 = int(np.sum(np.sign(a_arr[pairs[:, 1]] - a_arr[pairs[:, 0]])))

            if D0 == 0:
                continue

            # Use tightest feasible delta_star (same parity as C(n,2))
            abs_D0 = abs(D0)
            delta_star = parity if abs_D0 % 2 == parity else parity + 1
            if delta_star > abs_D0:
                continue

            # Greedy
            t_gr = time.time()
            _, cost_gr = simulate_greedy(a, n, delta_star)
            t_gr = time.time() - t_gr
            if cost_gr == float('inf'):
                continue

            # LP lower bound
            t_lp = time.time()
            cost_lp, _ = solve_lp(a, n, delta_star)
            t_lp = time.time() - t_lp

            if cost_lp is None or cost_lp < 1e-9:
                print(f"  [{idx+1}/{len(instances)}] {itype}: LP infeasible/zero, skipping",
                      flush=True)
                continue

            ratio = cost_gr / cost_lp

            results.append({
                'n': n,
                'type': itype,
                'idx': idx,
                'D0': D0,
                'delta_star': delta_star,
                'cost_gr': int(cost_gr),
                'cost_lp': float(cost_lp),
                'ratio': float(ratio),
                'greedy_time': t_gr,
                'lp_time': t_lp,
            })

            print(f"  [{idx+1}/{len(instances)}] {itype:16s}  "
                  f"ratio={ratio:.4f}  d_gr={cost_gr}  d*_LP={cost_lp:.1f}  "
                  f"LP {t_lp:.1f}s", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, "lp_bound_ratios.json")
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nPart 2 results saved to {save_path}", flush=True)
    return results


# ==================== Plotting ====================

def plot_results():
    """Generate two-panel figure: (a) exhaustive ratios n=3..9; (b) MILP-exact + LP-bound for n=20/50/100."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    os.makedirs(RESULTS_DIR, exist_ok=True)

    exhaustive_path = os.path.join(RESULTS_DIR, "exhaustive_ratios.json")
    lp_path = os.path.join(RESULTS_DIR, "lp_bound_ratios.json")
    milp_path = os.path.join(RESULTS_DIR, "milp_optimum_ratios.json")

    has_ex = os.path.exists(exhaustive_path)
    has_lp = os.path.exists(lp_path)
    has_milp = os.path.exists(milp_path)

    if has_ex:
        with open(exhaustive_path) as f:
            exhaustive = json.load(f)
    if has_lp:
        with open(lp_path) as f:
            lp_data = json.load(f)
    if has_milp:
        with open(milp_path) as f:
            milp_data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Panel (a): Scatter of exact ratios for n=3..9 ---
    ax = axes[0]
    if has_ex:
        ns_ex = [int(k) for k in sorted(exhaustive.keys(), key=int) if exhaustive[k]['ratios']]
        rng_ex = np.random.RandomState(1)
        for i, n in enumerate(ns_ex):
            ratios_plot = np.array(exhaustive[str(n)]['ratios'])
            jitter = rng_ex.normal(0, 0.12, len(ratios_plot))
            ax.scatter(i + jitter, ratios_plot, alpha=0.5, s=10, color='C0',
                       edgecolors='none', zorder=3, rasterized=True)
        ax.axhline(y=2.0, color='red', linestyle='--', linewidth=1.5, label='2-approx bound')
        ax.set_xticks(range(len(ns_ex)))
        ax.set_xticklabels([str(n) for n in ns_ex])
        ax.set_xlabel('$n$', fontsize=12)
        ax.set_ylabel(r'$d_{\mathrm{FBS}} \,/\, d^*$', fontsize=12)
        ax.legend(fontsize=11, loc='upper left', bbox_to_anchor=(0, 0.85))

    # --- Panel (b): MILP-exact (foreground) + LP-bound (background) for n=20/50/100 ---
    ax = axes[1]
    if has_milp and has_lp:
        kept_types = {'random', 'correlated', 'reversed'}
        def merge_type(t):
            return 'correlated' if t == 'reversed' else t

        by_n_milp = defaultdict(list)
        by_n_type_milp = defaultdict(lambda: defaultdict(list))
        for r in milp_data:
            if r['type'] in kept_types:
                by_n_milp[r['n']].append(r['ratio'])
                by_n_type_milp[r['n']][merge_type(r['type'])].append(r['ratio'])
        by_n_lp = defaultdict(list)
        by_n_type_lp = defaultdict(lambda: defaultdict(list))
        for r in lp_data:
            if r['type'] in kept_types:
                by_n_lp[r['n']].append(r['ratio'])
                by_n_type_lp[r['n']][merge_type(r['type'])].append(r['ratio'])

        ns = sorted(by_n_milp.keys())
        positions = list(range(len(ns)))

        # MILP box plots (right of center)
        bp_milp = ax.boxplot([by_n_milp[n] for n in ns],
                             positions=[p + 0.2 for p in positions],
                             widths=0.3, patch_artist=True, showfliers=False, zorder=1)
        for patch in bp_milp['boxes']:
            patch.set_facecolor('white'); patch.set_edgecolor('black'); patch.set_linewidth(1.2)
        for med in bp_milp['medians']:
            med.set_color('grey'); med.set_linewidth(1.5)

        # LP box plots (left of center)
        bp_lp = ax.boxplot([by_n_lp[n] for n in ns],
                           positions=[p - 0.2 for p in positions],
                           widths=0.3, patch_artist=True, showfliers=False, zorder=1)
        for patch in bp_lp['boxes']:
            patch.set_facecolor('white'); patch.set_edgecolor('black'); patch.set_linewidth(1.2)
        for med in bp_lp['medians']:
            med.set_color('grey'); med.set_linewidth(1.5)

        # 4 categories: MILP-random, MILP-correlated, LP-random, LP-correlated
        cat_styles = {
            ('milp', 'random'):     {'marker': 'o', 'color': 'C0', 'label': r'$d_{\mathrm{FBS}}/d^{\ast}$, random'},
            ('milp', 'correlated'): {'marker': 'o', 'color': 'C1', 'label': r'$d_{\mathrm{FBS}}/d^{\ast}$, correlated'},
            ('lp', 'random'):       {'marker': 'o', 'color': 'C2', 'label': r'$d_{\mathrm{FBS}}/d^{\ast}_{\mathrm{LP}}$, random'},
            ('lp', 'correlated'):   {'marker': 'o', 'color': 'gold', 'label': r'$d_{\mathrm{FBS}}/d^{\ast}_{\mathrm{LP}}$, correlated'},
        }

        rng = np.random.RandomState(42)
        for i, n in enumerate(ns):
            for (source, itype), style in cat_styles.items():
                data = by_n_type_milp if source == 'milp' else by_n_type_lp
                offset = 0.2 if source == 'milp' else -0.2
                vals = data[n].get(itype, [])
                if not vals:
                    continue
                jitter = rng.normal(0, 0.05, len(vals))
                label = style['label'] if i == 0 else None
                ax.scatter(i + offset + jitter, vals, alpha=0.6, s=25,
                           color=style['color'], marker=style['marker'],
                           edgecolors='none', zorder=3, label=label)

        ax.axhline(y=2.0, color='red', linestyle='--', linewidth=1.5, label='2-approx bound')
        ax.set_xticks(positions)
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel('$n$', fontsize=12)
        ax.set_ylabel('Approximation ratio', fontsize=12)
        ax.legend(fontsize=11, loc='upper left', bbox_to_anchor=(0, 0.85))

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        path = os.path.join(RESULTS_DIR, f"fig_approx_ratio.{ext}")
        fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Plots saved to {RESULTS_DIR}/fig_approx_ratio.{{pdf,png}}", flush=True)


# ==================== Main ====================

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    t_start = time.time()

    if mode in ("all", "small"):
        run_part1()
    if mode in ("all", "large"):
        run_part2()
    if mode in ("all", "small", "large", "plot"):
        plot_results()

    print(f"\nTotal elapsed: {time.time() - t_start:.1f}s")
    print("Done!", flush=True)
