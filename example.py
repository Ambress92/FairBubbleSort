#!/usr/bin/env python3
"""
Toy example: Fair Bubble Sort (FBS) with Numba JIT acceleration.

Demonstrates how FBS re-ranks items to satisfy a fairness constraint (tau)
on a continuous sensitive attribute, while minimizing the weighted Kendall-tau
distance to the original ranking.

Usage:
    python example.py
    python example.py --tau 0.8 --n 10
    python example.py --input sample_ranking.csv --tau 0.95
"""

import argparse
import csv
import time

import numpy as np

from src.fbs_numba import (
    count_concordant,
    count_inversions,
    greedy_heap_numba,
    warm_up,
)


def phi_score(a):
    """Compute the fairness score phi in [0, 1] for ranking a."""
    n = len(a)
    n_pairs = n * (n - 1) // 2
    if n_pairs == 0:
        return 1.0
    n_up = int(count_concordant(np.asarray(a, dtype=np.float64)))
    n_down = int(count_inversions(np.asarray(a, dtype=np.float64)))
    return 1.0 - abs(n_up - n_down) / n_pairs


def print_ranking(scores, sensitive, label):
    """Pretty-print a ranking with scores and sensitive attributes."""
    print(f"\n  {label}")
    print(f"  {'Pos':>3}  {'Score':>7}  {'Sensitive':>9}")
    print(f"  {'---':>3}  {'-------':>7}  {'---------':>9}")
    for i, (sc, sa) in enumerate(zip(scores, sensitive)):
        print(f"  {i+1:>3}  {sc:>7.3f}  {sa:>9.3f}")


def load_csv(path):
    """Load a CSV with 'score' and 'sensitive' columns, sorted by score descending."""
    scores, sensitive = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(float(row["score"]))
            sensitive.append(float(row["sensitive"]))
    scores = np.array(scores)
    sensitive = np.array(sensitive)
    order = np.argsort(-scores)
    return scores[order], sensitive[order]


def main():
    parser = argparse.ArgumentParser(description="Fair Bubble Sort — toy example")
    parser.add_argument("--tau", type=float, default=0.9,
                        help="Fairness threshold in (0, 1]. Higher = stricter. Default: 0.9")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to a CSV file with 'score' and 'sensitive' columns")
    parser.add_argument("--n", type=int, default=8,
                        help="Number of items (ignored if --input is provided). Default: 8")
    parser.add_argument("--seed", type=int, default=4,
                        help="Random seed (ignored if --input is provided). Default: 4")
    args = parser.parse_args()

    if args.input:
        scores, sensitive = load_csv(args.input)
        n = len(scores)
        source = args.input
    else:
        np.random.seed(args.seed)
        n = args.n
        scores = np.round(np.random.uniform(0, 10, size=n), 2)
        sensitive = np.round(np.random.uniform(0, 1, size=n), 3)
        order = np.argsort(-scores)
        scores = scores[order]
        sensitive = sensitive[order]
        source = f"random (seed={args.seed})"

    phi_before = phi_score(sensitive)

    print("=" * 55)
    print("  Fair Bubble Sort (FBS) — Toy Example")
    print("=" * 55)
    print(f"\n  n = {n},  tau = {args.tau},  source = {source}")

    print_ranking(scores, sensitive, "Original ranking (sorted by score):")
    print(f"\n  Fairness (phi):  {phi_before:.4f}  (threshold tau = {args.tau})")

    if phi_before >= args.tau:
        print("\n  Already fair — no swaps needed!")
        return

    # --- JIT warm-up ---
    print("\n  Compiling Numba JIT (first run only)...", end=" ", flush=True)
    t0 = time.time()
    warm_up()
    print(f"done in {time.time() - t0:.2f}s")

    # --- Run FBS ---
    s = scores.copy()
    a = sensitive.copy()

    t0 = time.time()
    cost, swaps, n_up, n_down = greedy_heap_numba(s, a, args.tau)
    elapsed = time.time() - t0

    n_pairs = n * (n - 1) // 2
    phi_after = 1.0 - abs(n_up - n_down) / n_pairs if n_pairs > 0 else 1.0

    print_ranking(s, a, "Fair ranking (after FBS):")

    print(f"\n  Results:")
    print(f"    Swaps performed:    {int(swaps)}")
    print(f"    Weighted cost:      {cost:.4f}")
    print(f"    Fairness (phi):     {phi_before:.4f} -> {phi_after:.4f}")
    print(f"    Constraint met:     phi = {phi_after:.4f} >= tau = {args.tau}")
    print(f"    Time:               {elapsed*1e6:.0f} us")
    print()


if __name__ == "__main__":
    main()
