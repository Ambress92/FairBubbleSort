"""
Experiment: Verify Proposition 5 (five-block structure) from the appendix.

For each n = 2^t (t = 2, ..., 15), runs FBS greedy on a = s = (0, ..., n-1),
stores the resulting permutation vector, and checks that the output matches
the five-block structure:

  B1: positions 1..m/2       -> items m, m-1, ..., m/2+1
  B2: position  m/2+1        -> item  n
  B3: positions m/2+2..m+1   -> items m/2, m/2-1, ..., 1
  B4: position  m+2          -> item  n-1
  B5: positions m+3..2m      -> items n-2, n-3, ..., m+1

Also verifies the closed-form cost d_FBS = m(8m^2 + 21m - 14) / 24.

Usage:
  python experiment_five_block.py run    -- run experiment, save results
  python experiment_five_block.py plot   -- re-plot from saved data
  python experiment_five_block.py all    -- run + plot
"""

import heapq
import json
import math
import os
import sys
import time

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..",
                           "experiments", "five_block_verification")
RESULTS_FILE = os.path.join(RESULTS_DIR, "results.jsonl")


def simulate_greedy_with_permutation(n, delta_star):
    """Run FBS greedy for the sorted case a = s = (0, ..., n-1).
    Returns (cost, permutation) where permutation[j] = item at position j.
    Items are 1-indexed (1..n) to match the paper's notation.
    """
    a = list(range(n))
    r = list(range(n))  # r[j] = item at position j (0-indexed items initially)
    D = n * (n - 1) // 2

    version = [0] * (n - 1)
    heap = []
    target = 1

    for k in range(n - 1):
        u, v = r[k], r[k + 1]
        sign = 1 if a[u] < a[v] else (-1 if a[u] > a[v] else 0)
        if sign == target:
            cost = abs(u - v)
            heapq.heappush(heap, (cost, k, version[k]))

    total_cost = 0
    while abs(D) > delta_star:
        while heap:
            cost, k, ver = heapq.heappop(heap)
            if ver == version[k]:
                break
        else:
            break

        r[k], r[k + 1] = r[k + 1], r[k]
        total_cost += cost
        D -= 2 * target

        for pos in (k - 1, k, k + 1):
            if 0 <= pos < n - 1:
                version[pos] += 1
                u, v = r[pos], r[pos + 1]
                sign = 1 if a[u] < a[v] else (-1 if a[u] > a[v] else 0)
                if sign == target:
                    c = abs(u - v)
                    heapq.heappush(heap, (c, pos, version[pos]))

    perm_1indexed = [x + 1 for x in r]
    return total_cost, perm_1indexed


def build_expected_five_blocks(n):
    """Build the expected permutation from the five-block structure (1-indexed)."""
    m = n // 2
    expected = [0] * n

    # B1: positions 1..m/2 -> items m, m-1, ..., m/2+1
    for j in range(1, m // 2 + 1):
        expected[j - 1] = m + 1 - j

    # B2: position m/2+1 -> item n
    expected[m // 2] = n

    # B3: positions m/2+2..m+1 -> items m/2, m/2-1, ..., 1
    for j in range(m // 2 + 2, m + 2):
        expected[j - 1] = m // 2 - (j - (m // 2 + 2))

    # Fix: B3 items are m/2, m/2-1, ..., 1
    for idx, j in enumerate(range(m // 2 + 2, m + 2)):
        expected[j - 1] = m // 2 - idx

    # B4: position m+2 -> item n-1
    expected[m + 1] = n - 1

    # B5: positions m+3..2m -> items n-2, n-3, ..., m+1
    for idx, j in enumerate(range(m + 3, 2 * m + 1)):
        expected[j - 1] = n - 2 - idx

    return expected


def closed_form_cost(n):
    """d_FBS = m(8m^2 + 21m - 14) / 24, m = n/2."""
    m = n // 2
    return m * (8 * m * m + 21 * m - 14) // 24


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


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if os.path.exists(RESULTS_FILE):
        os.rename(RESULTS_FILE, RESULTS_FILE + ".bak")

    t_min, t_max = 2, 15
    all_pass = True

    print("=" * 70)
    print("Proposition 5 verification: five-block structure of FBS")
    print(f"n = 2^t for t = {t_min} .. {t_max}  (n up to {2**t_max})")
    print("=" * 70)

    for t in range(t_min, t_max + 1):
        n = 2 ** t
        m = n // 2
        cn2 = n * (n - 1) // 2
        delta_star = 0  # C(n,2) is even for n = 2^t, t >= 2
        K_star = cn2 // 2

        print(f"\nt={t:>2d}, n={n:>6d}, m={m:>5d}, K*={K_star:>12d}")

        t0 = time.time()
        cost, perm = simulate_greedy_with_permutation(n, delta_star)
        elapsed = time.time() - t0

        expected = build_expected_five_blocks(n)
        blocks_match = (perm == expected)

        d_formula = closed_form_cost(n)
        cost_match = (cost == d_formula)

        status = "PASS" if (blocks_match and cost_match) else "FAIL"
        if not (blocks_match and cost_match):
            all_pass = False

        print(f"  Time:        {elapsed:.2f}s")
        print(f"  Cost (FBS):  {cost}")
        print(f"  Cost (form): {d_formula}")
        print(f"  Cost match:  {cost_match}")
        print(f"  Blocks match:{blocks_match}")
        print(f"  Status:      {status}")

        if not blocks_match and n <= 64:
            print(f"  Got:      {perm}")
            print(f"  Expected: {expected}")
            first_diff = next(
                (i for i in range(n) if perm[i] != expected[i]), None)
            if first_diff is not None:
                print(f"  First diff at position {first_diff + 1}: "
                      f"got {perm[first_diff]}, expected {expected[first_diff]}")

        record = {
            "t": t,
            "n": n,
            "m": m,
            "K_star": K_star,
            "cost_fbs": cost,
            "cost_formula": d_formula,
            "cost_match": cost_match,
            "blocks_match": blocks_match,
            "status": status,
            "time_seconds": round(elapsed, 3),
            "permutation": perm if n <= 1024 else None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        append_result(record)
        sys.stdout.flush()

    print("\n" + "=" * 70)
    print(f"OVERALL: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print(f"Results saved to {RESULTS_FILE}")
    print("=" * 70)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = load_results()
    if not records:
        print(f"No data in {RESULTS_FILE}")
        return

    ts = [r["t"] for r in records]
    ns = [r["n"] for r in records]
    costs = [r["cost_fbs"] for r in records]
    formulas = [r["cost_formula"] for r in records]
    times = [r["time_seconds"] for r in records]
    statuses = [r["blocks_match"] for r in records]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: Cost comparison (FBS vs formula)
    ax = axes[0]
    ax.plot(ns, costs, "o-", color="#2c7bb6", markersize=5, linewidth=1.2,
            label=r"$d_{\mathrm{FBS}}$ (computed)")
    ax.plot(ns, formulas, "x--", color="#d7191c", markersize=6, linewidth=1,
            label=r"$m(8m^2{+}21m{-}14)/24$")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"$n = 2^t$")
    ax.set_ylabel(r"$d_{\mathrm{FBS}}$")
    ax.set_title("FBS cost vs closed form")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Block verification status
    ax = axes[1]
    colors = ["#27ae60" if s else "#e74c3c" for s in statuses]
    ax.bar(range(len(ts)), [1] * len(ts), color=colors, edgecolor="white")
    ax.set_xticks(range(len(ts)))
    ax.set_xticklabels([f"$2^{{{t}}}$" for t in ts], fontsize=8)
    ax.set_xlabel(r"$n = 2^t$")
    ax.set_yticks([])
    ax.set_title("Five-block structure verification")
    for i, (t_val, ok) in enumerate(zip(ts, statuses)):
        ax.text(i, 0.5, "PASS" if ok else "FAIL",
                ha="center", va="center", fontsize=8, fontweight="bold",
                color="white")

    # Panel 3: Runtime
    ax = axes[2]
    ax.plot(ns, times, "s-", color="#8e44ad", markersize=5, linewidth=1.2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"$n = 2^t$")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("FBS runtime")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Proposition 5: Five-block structure verification",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = os.path.join(RESULTS_DIR, f"fig_five_block.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Plot saved to {os.path.join(RESULTS_DIR, 'fig_five_block.{{pdf,png}}')}")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python experiment_five_block.py [run|plot|all]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd in ("run", "all"):
        run()
    if cmd in ("plot", "all"):
        plot()
