"""
Numba-JIT accelerated Fair Bubble Sort (FBS) greedy algorithm.

Drop-in replacement for the pure-Python greedy_heap / count_inversions_merge /
count_concordant_merge from experiment_scalability.py, typically 50-100x faster.

The binary min-heap is hand-coded with flat arrays (Numba cannot use heapq).
"""

import numpy as np
import numba as nb


# ===================================================================
# Utility: fast adjacent-swap scrambling for instance generation
# ===================================================================

@nb.njit(cache=True)
def scramble_adjacent(a, n_scramble, n):
    """Perform n_scramble random adjacent swaps on array a of length n."""
    for _ in range(n_scramble):
        k = np.random.randint(0, n - 1)
        a[k], a[k + 1] = a[k + 1], a[k]


# ===================================================================
# Numba-compiled iterative merge-sort inversion / concordance counting
# (bottom-up to avoid Numba recursive-caching segfaults)
# ===================================================================

@nb.njit(cache=True)
def count_inversions(a):
    """Return number of inversions (a[i] > a[j] for i < j). Iterative bottom-up."""
    n = len(a)
    arr = a.copy()
    tmp = np.empty(n, dtype=arr.dtype)
    inv = 0
    width = 1
    while width < n:
        for left in range(0, n, 2 * width):
            mid = min(left + width, n)
            right = min(left + 2 * width, n)
            i, j, k = left, mid, left
            while i < mid and j < right:
                if arr[i] <= arr[j]:
                    tmp[k] = arr[i]; i += 1
                else:
                    tmp[k] = arr[j]; inv += mid - i; j += 1
                k += 1
            while i < mid:
                tmp[k] = arr[i]; i += 1; k += 1
            while j < right:
                tmp[k] = arr[j]; j += 1; k += 1
        arr[:n] = tmp[:n]
        width *= 2
    return inv


@nb.njit(cache=True)
def count_concordant(a):
    """Return number of concordant pairs (a[i] < a[j] for i < j). Iterative bottom-up."""
    n = len(a)
    arr = a.copy()
    tmp = np.empty(n, dtype=arr.dtype)
    conc = 0
    width = 1
    while width < n:
        for left in range(0, n, 2 * width):
            mid = min(left + width, n)
            right = min(left + 2 * width, n)
            i, j, k = left, mid, left
            while i < mid and j < right:
                if arr[i] < arr[j]:
                    tmp[k] = arr[i]; conc += right - j; i += 1
                else:
                    tmp[k] = arr[j]; j += 1
                k += 1
            while i < mid:
                tmp[k] = arr[i]; i += 1; k += 1
            while j < right:
                tmp[k] = arr[j]; j += 1; k += 1
        arr[:n] = tmp[:n]
        width *= 2
    return conc


# ===================================================================
# Hand-coded binary min-heap for (cost, index) pairs
#
# Uses three flat arrays:
#   heap_cost[0..size-1]  — key (swap cost)
#   heap_idx[0..size-1]   — value (position index k)
#   pos[k]                — where index k lives in the heap (-1 if absent)
# ===================================================================

@nb.njit(cache=True)
def _heap_less(heap_cost, heap_idx, i, j):
    """Compare by (cost, index) — leftmost tie-breaking."""
    if heap_cost[i] < heap_cost[j]:
        return True
    if heap_cost[i] == heap_cost[j] and heap_idx[i] < heap_idx[j]:
        return True
    return False


@nb.njit(cache=True)
def _heap_swap(heap_cost, heap_idx, pos, i, j):
    heap_cost[i], heap_cost[j] = heap_cost[j], heap_cost[i]
    heap_idx[i], heap_idx[j] = heap_idx[j], heap_idx[i]
    pos[heap_idx[i]] = i
    pos[heap_idx[j]] = j


@nb.njit(cache=True)
def _heap_sift_up(heap_cost, heap_idx, pos, i):
    while i > 0:
        parent = (i - 1) // 2
        if _heap_less(heap_cost, heap_idx, i, parent):
            _heap_swap(heap_cost, heap_idx, pos, i, parent)
            i = parent
        else:
            break


@nb.njit(cache=True)
def _heap_sift_down(heap_cost, heap_idx, pos, i, size):
    while True:
        smallest = i
        left = 2 * i + 1
        right = 2 * i + 2
        if left < size and _heap_less(heap_cost, heap_idx, left, smallest):
            smallest = left
        if right < size and _heap_less(heap_cost, heap_idx, right, smallest):
            smallest = right
        if smallest != i:
            _heap_swap(heap_cost, heap_idx, pos, i, smallest)
            i = smallest
        else:
            break


@nb.njit(cache=True)
def greedy_heap_numba(s, a, tau):
    """Numba-JIT greedy FBS. Mutates s and a in-place.

    Returns (total_cost, num_swaps, n_up_final, n_down_final).
    """
    n = len(s)
    c = np.empty(n - 1)
    sigma = np.empty(n - 1, dtype=nb.int8)

    for k in range(n - 1):
        c[k] = abs(s[k] - s[k + 1])
        if a[k + 1] > a[k]:
            sigma[k] = 1
        elif a[k + 1] < a[k]:
            sigma[k] = -1
        else:
            sigma[k] = 0

    n_up = count_concordant(a)
    n_down = count_inversions(a)

    n_pairs = n * (n - 1) // 2
    delta_star = int(np.floor((1.0 - tau) * n_pairs))
    d0 = abs(n_up - n_down)
    if delta_star % 2 != d0 % 2:
        delta_star += 1
    if d0 <= delta_star:
        return 0.0, 0, n_up, n_down

    if n_up > n_down:
        mu = nb.int8(1)
    else:
        mu = nb.int8(-1)

    # Allocate heap arrays (max size = n-1)
    heap_cost = np.empty(n - 1)
    heap_idx = np.empty(n - 1, dtype=nb.int64)
    pos = np.full(n - 1, -1, dtype=nb.int64)  # pos[k] = position in heap, -1 if absent
    heap_size = 0

    for k in range(n - 1):
        if sigma[k] == mu:
            heap_cost[heap_size] = c[k]
            heap_idx[heap_size] = k
            pos[k] = heap_size
            heap_size += 1
            _heap_sift_up(heap_cost, heap_idx, pos, heap_size - 1)

    total_cost = 0.0
    swaps = 0

    while abs(n_up - n_down) > delta_star:
        if heap_size == 0:
            break

        # Pop min
        k_star = heap_idx[0]
        cost_star = heap_cost[0]
        heap_size -= 1
        if heap_size > 0:
            heap_cost[0] = heap_cost[heap_size]
            heap_idx[0] = heap_idx[heap_size]
            pos[heap_idx[0]] = 0
        pos[k_star] = -1
        if heap_size > 0:
            _heap_sift_down(heap_cost, heap_idx, pos, 0, heap_size)

        # Perform swap
        s[k_star], s[k_star + 1] = s[k_star + 1], s[k_star]
        a[k_star], a[k_star + 1] = a[k_star + 1], a[k_star]
        total_cost += cost_star
        swaps += 1

        # Update neighbours
        for offset in range(-1, 2):
            ell = k_star + offset
            if ell < 0 or ell >= n - 1:
                continue
            old_sigma = sigma[ell]
            c[ell] = abs(s[ell] - s[ell + 1])
            if ell == k_star:
                sigma[ell] = -sigma[ell]
            else:
                if a[ell + 1] > a[ell]:
                    sigma[ell] = 1
                elif a[ell + 1] < a[ell]:
                    sigma[ell] = -1
                else:
                    sigma[ell] = 0

            was_in = (old_sigma == mu)
            now_in = (sigma[ell] == mu)

            if was_in and not now_in:
                # Remove from heap
                p = pos[ell]
                if p >= 0:
                    heap_size -= 1
                    if p < heap_size:
                        heap_cost[p] = heap_cost[heap_size]
                        heap_idx[p] = heap_idx[heap_size]
                        pos[heap_idx[p]] = p
                        # May need to sift up or down
                        _heap_sift_up(heap_cost, heap_idx, pos, p)
                        _heap_sift_down(heap_cost, heap_idx, pos, p, heap_size)
                    pos[ell] = -1
            elif not was_in and now_in:
                # Insert into heap
                heap_cost[heap_size] = c[ell]
                heap_idx[heap_size] = ell
                pos[ell] = heap_size
                heap_size += 1
                _heap_sift_up(heap_cost, heap_idx, pos, heap_size - 1)
            elif now_in:
                # Update key
                p = pos[ell]
                if p >= 0:
                    heap_cost[p] = c[ell]
                    _heap_sift_up(heap_cost, heap_idx, pos, p)
                    _heap_sift_down(heap_cost, heap_idx, pos, p, heap_size)

        n_up -= mu
        n_down += mu

    return total_cost, swaps, n_up, n_down


# ===================================================================
# Numba-JIT naive greedy (linear scan instead of heap) — O(n*K*)
# ===================================================================

@nb.njit(cache=True)
def greedy_naive_numba(s, a, tau):
    """Numba-JIT naive FBS (linear scan). Mutates s and a in-place.

    Returns (total_cost, num_swaps).
    """
    n = len(s)
    c = np.empty(n - 1)
    sigma = np.empty(n - 1, dtype=nb.int8)

    for k in range(n - 1):
        c[k] = abs(s[k] - s[k + 1])
        if a[k + 1] > a[k]:
            sigma[k] = 1
        elif a[k + 1] < a[k]:
            sigma[k] = -1
        else:
            sigma[k] = 0

    n_up = count_concordant(a)
    n_down = count_inversions(a)

    n_pairs = n * (n - 1) // 2
    delta_star = int(np.floor((1.0 - tau) * n_pairs))
    d0 = abs(n_up - n_down)
    if delta_star % 2 != d0 % 2:
        delta_star += 1
    if d0 <= delta_star:
        return 0.0, 0

    if n_up > n_down:
        mu = nb.int8(1)
    else:
        mu = nb.int8(-1)

    total_cost = 0.0
    swaps = 0

    while abs(n_up - n_down) > delta_star:
        best_k = -1
        best_c = 1e300
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

        for offset in range(-1, 2):
            ell = k_star + offset
            if ell < 0 or ell >= n - 1:
                continue
            c[ell] = abs(s[ell] - s[ell + 1])
            if ell == k_star:
                sigma[ell] = -sigma[ell]
            else:
                if a[ell + 1] > a[ell]:
                    sigma[ell] = 1
                elif a[ell + 1] < a[ell]:
                    sigma[ell] = -1
                else:
                    sigma[ell] = 0

        n_up -= mu
        n_down += mu

    return total_cost, swaps


# ===================================================================
# Numba-JIT unweighted FBS (doubly-linked list) — O(K*)
# ===================================================================

@nb.njit(cache=True)
def greedy_unweighted_numba(a, tau):
    """Unweighted FBS with doubly-linked list. Mutates a in-place.

    Returns num_swaps (= unweighted Kemeny distance).
    """
    n = len(a)
    sigma = np.empty(n - 1, dtype=nb.int8)
    for k in range(n - 1):
        if a[k + 1] > a[k]:
            sigma[k] = 1
        elif a[k + 1] < a[k]:
            sigma[k] = -1
        else:
            sigma[k] = 0

    n_up = count_concordant(a)
    n_down = count_inversions(a)

    n_pairs = n * (n - 1) // 2
    delta_star = int(np.floor((1.0 - tau) * n_pairs))
    d0 = abs(n_up - n_down)
    if delta_star % 2 != d0 % 2:
        delta_star += 1
    if d0 <= delta_star:
        return 0

    if n_up > n_down:
        mu = nb.int8(1)
    else:
        mu = nb.int8(-1)

    # Doubly-linked list via arrays. Sentinel head = n-1 (one past last valid).
    SENTINEL = n - 1
    ll_prev = np.full(n, -1, dtype=nb.int64)
    ll_next = np.full(n, -1, dtype=nb.int64)
    head = SENTINEL
    ll_prev[SENTINEL] = SENTINEL
    ll_next[SENTINEL] = SENTINEL

    def ll_insert(k):
        nonlocal head
        nxt = ll_next[SENTINEL]
        ll_next[SENTINEL] = k
        ll_prev[k] = SENTINEL
        ll_next[k] = nxt
        ll_prev[nxt] = k

    def ll_remove(k):
        p = ll_prev[k]
        nx = ll_next[k]
        ll_next[p] = nx
        ll_prev[nx] = p
        ll_prev[k] = -1
        ll_next[k] = -1

    for k in range(n - 1):
        if sigma[k] == mu:
            ll_insert(k)

    swaps = 0

    while abs(n_up - n_down) > delta_star:
        k_star = ll_next[SENTINEL]
        if k_star == SENTINEL:
            break

        ll_remove(k_star)

        a[k_star], a[k_star + 1] = a[k_star + 1], a[k_star]
        swaps += 1

        for offset in range(-1, 2):
            ell = k_star + offset
            if ell < 0 or ell >= n - 1:
                continue
            old_sigma = sigma[ell]
            if ell == k_star:
                sigma[ell] = -sigma[ell]
            else:
                if a[ell + 1] > a[ell]:
                    sigma[ell] = 1
                elif a[ell + 1] < a[ell]:
                    sigma[ell] = -1
                else:
                    sigma[ell] = 0

            was_in = (old_sigma == mu)
            now_in = (sigma[ell] == mu)

            if was_in and not now_in:
                ll_remove(ell)
            elif not was_in and now_in:
                ll_insert(ell)

        n_up -= mu
        n_down += mu

    return swaps


# ===================================================================
# Python wrappers (match the original API)
# ===================================================================

def greedy_heap(s, a, tau):
    """Drop-in replacement for experiment_scalability.greedy_heap.

    Accepts Python lists or numpy arrays. Returns (total_cost, num_swaps).
    """
    s_arr = np.asarray(s, dtype=np.float64)
    a_arr = np.asarray(a, dtype=np.float64)
    cost, swaps, _, _ = greedy_heap_numba(s_arr, a_arr, tau)
    if isinstance(s, list):
        for i in range(len(s)):
            s[i] = s_arr[i]
            a[i] = a_arr[i]
    return cost, int(swaps)


def greedy_naive(s, a, tau):
    """Drop-in replacement for experiment_scalability.greedy_naive."""
    s_arr = np.asarray(s, dtype=np.float64)
    a_arr = np.asarray(a, dtype=np.float64)
    cost, swaps = greedy_naive_numba(s_arr, a_arr, tau)
    if isinstance(s, list):
        for i in range(len(s)):
            s[i] = s_arr[i]
            a[i] = a_arr[i]
    return cost, int(swaps)


def greedy_heap_with_phi(s, a, tau):
    """Like greedy_heap but also returns the final phi, avoiding recomputation."""
    s_arr = np.asarray(s, dtype=np.float64)
    a_arr = np.asarray(a, dtype=np.float64)
    cost, swaps, n_up, n_down = greedy_heap_numba(s_arr, a_arr, tau)
    if isinstance(s, list):
        for i in range(len(s)):
            s[i] = s_arr[i]
            a[i] = a_arr[i]
    n = len(s)
    n_pairs = n * (n - 1) // 2
    phi = 1.0 - abs(n_up - n_down) / n_pairs if n_pairs > 0 else 1.0
    return cost, int(swaps), phi


def greedy_unweighted(s, a, tau):
    """Unweighted FBS. Returns (0.0, num_swaps) to match greedy_heap signature."""
    a_arr = np.asarray(a, dtype=np.float64)
    swaps = greedy_unweighted_numba(a_arr, tau)
    if isinstance(a, list):
        for i in range(len(a)):
            a[i] = a_arr[i]
    return 0.0, int(swaps)


def warm_up():
    """Trigger Numba JIT compilation on a tiny instance."""
    s = np.array([3.0, 2.0, 1.0])
    a = np.array([1.0, 2.0, 3.0])
    greedy_heap_numba(s, a, 1.0)
    greedy_naive_numba(s.copy(), a.copy(), 1.0)
    greedy_unweighted_numba(np.array([1.0, 2.0, 3.0]), 1.0)
    scramble_adjacent(np.array([1.0, 2.0, 3.0]), 2, 3)
