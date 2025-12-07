"""PORC model representation and verification helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .binom import binom_row


@dataclass
class FitResult:
    L: int
    d: int
    success: bool
    model: Optional["PORCModel"]
    reason: Optional[str] = None
    details: Optional[dict] = None


@dataclass
class EventualPORCResult:
    """Result wrapper for eventual PORC fitting.

    Attributes
    ----------
    success: bool
        Whether a fit was found after discarding a prefix.
    start_index: int
        Number of initial points discarded to obtain the fit.
    start_x: Optional[int]
        The first x-value of the fitted suffix (None when no points remain).
    fit_result: Optional[FitResult]
        The successful fit result if found.
    reason: Optional[str]
        Failure reason when ``success`` is False.
    details: Optional[dict]
        Optional auxiliary information.
    """

    success: bool
    start_index: int
    start_x: Optional[int]
    fit_result: Optional[FitResult]
    reason: Optional[str] = None
    details: Optional[dict] = None


class PORCModel:
    def __init__(self, L: int, d: int, coeffs_by_residue: Dict[int, List[int]]):
        self.L = L
        self.d = d
        self.coeffs_by_residue = coeffs_by_residue

    def eval(self, x: int) -> int:
        r = x % self.L
        t = (x - r) // self.L
        coeffs = self.coeffs_by_residue[r]
        row = binom_row(t, len(coeffs) - 1)
        return sum(c * b for c, b in zip(coeffs, row))

    def verify(self, xs: Iterable[int], vs: Iterable[int]) -> Tuple[bool, Optional[Tuple[int, int, int]]]:
        for x, v in zip(xs, vs):
            val = self.eval(x)
            if val != v:
                return False, (x, v, val)
        return True, None
