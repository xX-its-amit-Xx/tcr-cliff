# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Sequence distance metrics for neighbour/cliff detection.

Edit-distance computation is the inner loop of cliff mining, so we provide a
*bounded* Levenshtein that exits early once the distance provably exceeds ``k`` —
this is what makes mining tractable on large datasets where almost no pair is a
near-neighbour.
"""

from __future__ import annotations


def hamming(a: str, b: str) -> int | None:
    """Hamming distance, or ``None`` if the strings differ in length."""
    if len(a) != len(b):
        return None
    return sum(x != y for x, y in zip(a, b))


def levenshtein(a: str, b: str) -> int:
    """Full Levenshtein (insertion/deletion/substitution) edit distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def bounded_levenshtein(a: str, b: str, max_k: int) -> int:
    """Levenshtein distance capped at ``max_k + 1``.

    Returns the true distance when it is ``<= max_k``; otherwise returns
    ``max_k + 1`` (a sentinel meaning "farther than the threshold"). Uses the
    classic banded DP with per-row early termination.
    """
    if abs(len(a) - len(b)) > max_k:
        return max_k + 1
    if a == b:
        return 0
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i]
        row_min = i
        ca = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ca == b[j - 1] else 1
            val = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            cur.append(val)
            if val < row_min:
                row_min = val
        if row_min > max_k:  # every cell in this row already exceeds the cap
            return max_k + 1
        prev = cur
    return prev[m] if prev[m] <= max_k else max_k + 1


def within_k(a: str, b: str, k: int, metric: str = "levenshtein") -> tuple[bool, int]:
    """Return ``(is_within, distance)`` for the chosen metric.

    For Hamming on unequal lengths, returns ``(False, k + 1)``.
    """
    if metric == "hamming":
        d = hamming(a, b)
        if d is None:
            return False, k + 1
        return d <= k, d
    d = bounded_levenshtein(a, b, k)
    return d <= k, d
