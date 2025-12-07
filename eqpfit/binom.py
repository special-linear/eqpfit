"""Binomial coefficient utilities for integer arguments."""
from __future__ import annotations

from typing import Iterable, List


def binom_int(n: int, k: int) -> int:
    """Compute the integer binomial coefficient C(n, k) for integer n and k >= 0.

    Supports negative ``n`` using the generalized binomial identity::

        C(n, k) = (-1)^k * C(k - n - 1, k)
    """

    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return 1
    if n < 0:
        return ((-1) ** k) * binom_int(k - n - 1, k)
    if k > n:
        return 0
    result = 1
    for i in range(1, k + 1):
        result = result * (n - i + 1) // i
    return result


def binom_row(t: int, d: int) -> List[int]:
    """Return ``[C(t,0), ..., C(t,d)]`` computed iteratively."""

    row = [1]
    for k in range(1, d + 1):
        row.append(row[-1] * (t - k + 1) // k)
    return row


def eval_binom_poly(coeffs: Iterable[int], t: int) -> int:
    """Evaluate ``sum coeffs[k] * C(t, k)`` efficiently."""

    coeff_list = list(coeffs)
    total = 0
    row = binom_row(t, len(coeff_list) - 1)
    for c, b in zip(coeff_list, row):
        total += c * b
    return total
