"""
Discretization-based baselines for fair ranking.

Given a continuous protected attribute, these methods discretize it into G
quantile bins and apply a group-fair ranking algorithm. The result is then
evaluated using the continuous Kendall tau fairness measure phi.

Baselines (using original author implementations where available):
  - det_const_sort: DetConstSort from FairRankTune (Geyik et al. / Celis et al.)
  - fair_star:      FA*IR from fairsearchcore (Zehlike et al. 2017) — binary only
  - median_split:   simple 2-group interleaving at the median (ours)

Libraries:
  - FairRankTune (pip install FairRankTune) — Cachel 2024, implements DetConstSort
  - fairsearchcore (pip install fairsearchcore) — Zehlike et al., implements FA*IR
"""

import math
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from experiment_scalability import count_concordant_merge, count_inversions_merge


# ===================================================================
# Monkey-patch fairsearchcore hash bug (numpy int vs Python int)
# ===================================================================

import fairsearchcore.fail_prob as _fsc_fp
_original_hash = _fsc_fp.LegalAssignmentKey.__hash__
def _fixed_hash(self):
    return int(_original_hash(self))
_fsc_fp.LegalAssignmentKey.__hash__ = _fixed_hash


# ===================================================================
# Discretization
# ===================================================================

def discretize_quantile(attrs, G):
    """Assign each element to a quantile bin in {0, 1, ..., G-1}.

    Uses sorted percentile thresholds. Handles ties by allowing
    unequal bin sizes (some bins may be larger if many values coincide).

    Returns list of integer group labels, same length as attrs.
    """
    n = len(attrs)
    if G <= 1:
        return [0] * n
    if G >= n:
        order = sorted(range(n), key=lambda i: attrs[i])
        labels = [0] * n
        for rank, idx in enumerate(order):
            labels[idx] = min(rank * G // n, G - 1)
        return labels

    sorted_vals = sorted(attrs)
    thresholds = []
    for g in range(1, G):
        idx = int(math.ceil(g * n / G)) - 1
        idx = min(idx, n - 1)
        thresholds.append(sorted_vals[idx])

    labels = []
    for v in attrs:
        label = 0
        for t in thresholds:
            if v > t:
                label += 1
            else:
                break
        labels.append(min(label, G - 1))

    return labels


# ===================================================================
# Evaluation utilities
# ===================================================================

def compute_phi_continuous(attrs_in_ranking_order):
    """Compute phi = 1 - |tau| on continuous attribute values in ranking order."""
    n = len(attrs_in_ranking_order)
    n_pairs = n * (n - 1) // 2
    if n_pairs == 0:
        return 1.0
    _, n_up = count_concordant_merge(list(attrs_in_ranking_order))
    _, n_down = count_inversions_merge(list(attrs_in_ranking_order))
    tau_val = (n_up - n_down) / n_pairs
    return 1.0 - abs(tau_val)


def weighted_kemeny_naive(new_order, scores):
    """Compute weighted Kemeny distance between identity and new_order.

    new_order[i] = original index of the item now at position i.
    scores[j] = score of original item j.

    Cost = sum of |s(orig_i) - s(orig_j)| for all pairs (i,j) where
    i appears before j in new_order but i > j (inversion w.r.t. identity).

    Uses numpy vectorization with chunking for large n to manage memory.
    """
    n = len(new_order)
    order = np.array(new_order)
    sc = np.array(scores)

    # For large n, process in chunks to limit memory usage
    CHUNK = 4000
    cost = 0.0
    for i_start in range(0, n, CHUNK):
        i_end = min(i_start + CHUNK, n)
        for j_start in range(i_start, n, CHUNK):
            j_end = min(j_start + CHUNK, n)
            if j_start == i_start:
                # Same block: only upper triangle
                block_o = order[i_start:i_end]
                oi = block_o[:, None]
                oj = block_o[None, :]
                mask = np.triu(oi > oj, k=1)
                si = sc[block_o][:, None]
                sj = sc[block_o][None, :]
                cost += np.sum(np.abs(si - sj) * mask)
            elif j_start > i_start:
                # Cross block
                oi = order[i_start:i_end, None]
                oj = order[j_start:j_end][None, :]
                mask = oi > oj
                si = sc[order[i_start:i_end]][:, None]
                sj = sc[order[j_start:j_end]][None, :]
                cost += np.sum(np.abs(si - sj) * mask)
    return float(cost)


# ===================================================================
# Baseline 1: DetConstSort (original implementation from FairRankTune)
# ===================================================================

def det_const_sort(scores, groups, proportions):
    """DetConstSort reranking using FairRankTune library.

    Reference: Geyik et al. (2019) / Celis, Straszak & Vishnoi (2018)
    Implementation: FairRankTune (Cachel, 2024)

    Args:
        scores: list of scores (descending order, indexed 0..n-1)
        groups: list of group labels (same length as scores)
        proportions: dict {group_id: fraction}

    Returns:
        new_order: list of original indices in the new ranking order
    """
    import FairRankTune as frt

    n = len(scores)
    item_ids = list(range(n))

    # FairRankTune expects:
    #   ranking_df: DataFrame with column 0 = item ids (in ranking order)
    #   scores_df:  DataFrame with column 0 = scores (aligned with ranking_df)
    #   item_group_dict: {item_id: group_label}
    #   distribution: {group_id: proportion}
    ranking_df = pd.DataFrame({0: item_ids})
    scores_df = pd.DataFrame({0: scores})
    item_group_dict = {i: groups[i] for i in range(n)}

    reranked_df, _, _ = frt.Rankers.DETCONSTSORT(
        ranking_df, item_group_dict, scores_df, proportions, k=n
    )

    return reranked_df[0].tolist()


# ===================================================================
# Baseline 2: FA*IR (original implementation from fairsearchcore)
# ===================================================================

def fair_star(scores, groups, proportions):
    """FA*IR reranking using fairsearchcore library.

    Reference: Zehlike, Bonchi, Castillo et al. (2017)
    Implementation: fairsearchcore (Fair Search project)

    NOTE: FA*IR is binary — it requires exactly one "protected" group.
    When G > 2, we treat the smallest group as protected and merge all
    others into non-protected. When G = 2, the smaller group is protected.

    Args:
        scores: list of scores (descending order)
        groups: list of group labels
        proportions: dict {group_id: fraction}

    Returns:
        new_order: list of original indices in the new ranking order
    """
    import fairsearchcore as fsc
    from fairsearchcore.models import FairScoreDoc

    n = len(scores)

    # Identify protected group: the group most underrepresented in the
    # top half of the ranking (this ensures FA*IR actually has work to do).
    group_counts = Counter(groups)
    top_half = groups[:n // 2]
    top_counts = Counter(top_half)
    # For each group, compute representation ratio: (top share) / (overall share)
    # The most underrepresented group has the lowest ratio
    ratios = {}
    for g in group_counts:
        expected = group_counts[g] / n
        actual = top_counts.get(g, 0) / (n // 2) if n > 1 else 0
        ratios[g] = actual / expected if expected > 0 else float('inf')
    protected_group = min(ratios, key=ratios.get)
    p = group_counts[protected_group] / n

    # Clamp p to avoid degenerate cases
    p = max(0.01, min(p, 0.99))

    # Create FairScoreDoc list
    candidates = []
    for i in range(n):
        is_prot = (groups[i] == protected_group)
        candidates.append(FairScoreDoc(i, scores[i], is_prot))

    # FA*IR's mtable computation is exponential and only feasible for k <= ~400.
    # For larger n, we apply FA*IR to the top-k and leave the rest unchanged.
    k = min(n, 400)

    fair = fsc.Fair(k, p, 0.1)  # alpha = 0.1 (standard)
    reranked = fair.re_rank(candidates[:k])
    top_ids = [doc.id for doc in reranked]

    # Append remaining items in original order
    remaining = [candidates[i].id for i in range(k, n)]
    return top_ids + remaining


# ===================================================================
# Baseline 3: Median-split interleaving (ours — no library needed)
# ===================================================================

def median_split_interleave(scores, attrs):
    """Split at median attribute value into 2 groups, interleave 50/50.

    Each group is sorted by score descending. Alternate picks starting
    from the high-attribute group. When one group is exhausted, append
    the rest of the other.

    Returns:
        new_order: list of original indices in the new ranking order
    """
    n = len(scores)
    sorted_attrs = sorted(attrs)
    median = sorted_attrs[n // 2]

    group_high = []  # attr >= median
    group_low = []   # attr < median
    for i in range(n):
        if attrs[i] >= median:
            group_high.append(i)
        else:
            group_low.append(i)

    new_order = []
    ih, il = 0, 0
    turn_high = True
    while ih < len(group_high) and il < len(group_low):
        if turn_high:
            new_order.append(group_high[ih])
            ih += 1
        else:
            new_order.append(group_low[il])
            il += 1
        turn_high = not turn_high

    while ih < len(group_high):
        new_order.append(group_high[ih])
        ih += 1
    while il < len(group_low):
        new_order.append(group_low[il])
        il += 1

    return new_order


# ===================================================================
# Baseline 4: Score-attribute blend (continuous heuristic)
# ===================================================================

def score_attribute_blend(scores, attrs, alpha):
    """Blend score rank with a fair target rank to trade off accuracy vs fairness.

    For each item i, compute:
        blend_key[i] = alpha * score_position[i] + (1 - alpha) * fair_position[i]

    where fair_position is an interleaved ordering of attributes (alternating
    high/low values) that achieves near-perfect phi.

    Args:
        scores: list of scores (descending order, indexed 0..n-1)
        attrs: list of continuous attribute values
        alpha: blend parameter in [0, 1]. alpha=1 -> pure score ranking,
               alpha=0 -> pure fair ranking.

    Returns:
        new_order: list of original indices in the new ranking order
    """
    n = len(scores)

    # Score position: items are already in score-descending order, so
    # score_position[i] = i (item 0 is rank 0, item 1 is rank 1, etc.)
    score_pos = list(range(n))

    # Fair target: interleave items sorted by attribute (alternate high/low)
    # to produce a ranking where ascending and descending pairs are balanced.
    attr_order = sorted(range(n), key=lambda i: attrs[i])
    low_half = attr_order[:n // 2]          # lowest attribute values
    high_half = attr_order[n // 2:][::-1]   # highest attribute values, reversed

    fair_pos = [0] * n
    pos = 0
    ih, il = 0, 0
    while ih < len(high_half) and il < len(low_half):
        fair_pos[high_half[ih]] = pos; pos += 1; ih += 1
        fair_pos[low_half[il]] = pos; pos += 1; il += 1
    while ih < len(high_half):
        fair_pos[high_half[ih]] = pos; pos += 1; ih += 1
    while il < len(low_half):
        fair_pos[low_half[il]] = pos; pos += 1; il += 1

    # Blend and sort
    blend_key = [alpha * score_pos[i] + (1 - alpha) * fair_pos[i] for i in range(n)]
    new_order = sorted(range(n), key=lambda i: blend_key[i])

    return new_order


# ===================================================================
# Top-level runner
# ===================================================================

def run_baseline(baseline_name, scores, attrs, G=2, alpha=None):
    """Run one baseline with G discretization bins.

    Args:
        baseline_name: one of "det_const_sort", "fair_star", "median_split", "blend"
        scores: list of scores (descending order)
        attrs: list of continuous attribute values
        G: number of quantile bins (ignored for median_split and blend)
        alpha: blend parameter for "blend" baseline (0=fair, 1=score)

    Returns:
        dict with: phi_achieved, weighted_kemeny, new_order
    """
    n = len(scores)

    if baseline_name == "median_split":
        new_order = median_split_interleave(scores, attrs)
    elif baseline_name == "blend":
        new_order = score_attribute_blend(scores, attrs, alpha)
    else:
        groups = discretize_quantile(attrs, G)
        group_counts = Counter(groups)
        proportions = {g: cnt / n for g, cnt in group_counts.items()}

        if baseline_name == "det_const_sort":
            new_order = det_const_sort(scores, groups, proportions)
        elif baseline_name == "fair_star":
            new_order = fair_star(scores, groups, proportions)
        else:
            raise ValueError(f"Unknown baseline: {baseline_name}")

    # Evaluate
    attrs_reordered = [attrs[i] for i in new_order]
    phi = compute_phi_continuous(attrs_reordered)
    cost = weighted_kemeny_naive(new_order, scores)

    return {
        "phi_achieved": phi,
        "weighted_kemeny": cost,
        "new_order": new_order,
    }
