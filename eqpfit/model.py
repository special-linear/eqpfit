"""PORC model representation and verification helpers."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb, factorial
from typing import Dict, Iterable, List, Optional, Tuple

from .binom import binom_row


def _stirling1_table(n: int) -> List[List[int]]:
    """Compute signed Stirling numbers of the first kind up to order ``n``."""

    table = [[0] * (n + 1) for _ in range(n + 1)]
    table[0][0] = 1
    for i in range(1, n + 1):
        for k in range(1, i + 1):
            table[i][k] = table[i - 1][k - 1] - (i - 1) * table[i - 1][k]
    return table


def _binom_to_monomial(coeffs: List[int]) -> List[Fraction]:
    """Convert binomial-basis coefficients to monomial coefficients."""

    degree = len(coeffs) - 1
    stirling = _stirling1_table(degree)
    monomial: List[Fraction] = [Fraction(0) for _ in range(degree + 1)]
    for k, ck in enumerate(coeffs):
        if ck == 0:
            continue
        fact = factorial(k)
        for j in range(k + 1):
            s = stirling[k][j]
            if s == 0:
                continue
            monomial[j] += Fraction(ck * s, fact)
    return monomial


def _monomial_in_x(coeffs: List[int], L: int, r: int) -> List[Fraction]:
    """Convert binomial-basis coefficients to monomial coefficients in the original ``x`` variable."""

    monomial_t = _binom_to_monomial(coeffs)
    degree = len(monomial_t) - 1
    monomial_x: List[Fraction] = [Fraction(0) for _ in range(degree + 1)]
    for power, a in enumerate(monomial_t):
        if a == 0:
            continue
        scale = Fraction(1, L ** power)
        for k in range(power + 1):
            coefficient = a * Fraction(comb(power, k)) * scale * (Fraction(-r) ** (power - k))
            monomial_x[k] += coefficient
    return monomial_x


def _format_fraction(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _fraction_to_sympy_literal(value: Fraction) -> int | str:
    """Return int for whole Fractions, otherwise \"a/b\" string."""

    frac = Fraction(value)
    return frac.numerator if frac.denominator == 1 else f"{frac.numerator}/{frac.denominator}"


def _format_monomial_poly(coeffs: List[Fraction], var: str = "t") -> str:
    """Format a monomial-basis polynomial into a human-readable string."""

    terms = []
    for power, coeff in enumerate(coeffs):
        if coeff == 0:
            continue
        coeff = Fraction(coeff)
        sign = "-" if coeff < 0 else "+"
        abs_coeff = abs(coeff)
        if power == 0:
            body = _format_fraction(abs_coeff)
        elif power == 1:
            if abs_coeff == 1:
                body = var
            else:
                body = f"{_format_fraction(abs_coeff)}*{var}"
        else:
            if abs_coeff == 1:
                body = f"{var}^{power}"
            else:
                body = f"{_format_fraction(abs_coeff)}*{var}^{power}"
        terms.append((sign, body))

    if not terms:
        return "0"

    first_sign, first_body = terms[0]
    prefix = "" if first_sign == "+" else "-"
    formatted = prefix + first_body
    for sign, body in terms[1:]:
        formatted += f" {sign} {body}"
    return formatted


@dataclass
class FitResult:
    L: int
    d: int
    success: bool
    model: Optional["PORCModel"]
    reason: Optional[str] = None
    details: Optional[dict] = None
    monomial_coeffs_by_residue: Optional[Dict[int, List[Fraction]]] = None

    def __post_init__(self) -> None:
        if self.success and self.model and self.monomial_coeffs_by_residue is None:
            self.monomial_coeffs_by_residue = self.model.monomial_coeffs_by_residue

    def __str__(self) -> str:  # pragma: no cover - trivial wrapper
        return self._format()

    __repr__ = __str__

    def _format(self) -> str:
        if not self.success:
            base = f"FitResult: FAILED (L={self.L}, d={self.d}, reason={self.reason})"
            if self.details:
                base += f" details={self.details}"
            return base

        header = f"FitResult: SUCCESS (L={self.L}, d={self.d})"
        if not self.model:
            return header

        lines = [header, self.model._format_coeffs(indent="  ")]
        return "\n".join(lines)


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

    def __str__(self) -> str:  # pragma: no cover - trivial wrapper
        return self._format()

    __repr__ = __str__

    def _format(self) -> str:
        if not self.success:
            base = (
                f"EventualPORCResult: FAILED (start_index={self.start_index}, "
                f"start_x={self.start_x}, reason={self.reason})"
            )
            if self.fit_result:
                base += f"\nLast fit attempt -> {self.fit_result._format()}"
            return base

        fit_desc = (
            "unknown period" if not self.fit_result else f"period {self.fit_result.L}"
        )
        header = (
            f"EventualPORCResult: SUCCESS (dropped={self.start_index}, start_x={self.start_x}, "
            f"{fit_desc})"
        )
        if self.fit_result:
            return "\n".join([header, self.fit_result._format()])
        return header


class PORCModel:
    def __init__(self, L: int, d: int, coeffs_by_residue: Dict[int, List[int]]):
        self.L = L
        self.d = d
        self.coeffs_by_residue = coeffs_by_residue
        self.monomial_coeffs_by_residue: Dict[int, List[Fraction]] = {
            r: _monomial_in_x(coeffs, L, r) for r, coeffs in coeffs_by_residue.items()
        }

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

    def as_coeffs_array(self) -> List[List[int | str]]:
        """Return monomial coefficients per residue in a SymPy-friendly format.

        Fractions are emitted as ``\"a/b\"`` strings that ``sympy.Rational`` can parse;
        integer values stay as ``int``.
        """

        return [
            [_fraction_to_sympy_literal(c) for c in self.monomial_coeffs_by_residue[r]]
            for r in sorted(self.monomial_coeffs_by_residue)
        ]

    def _format_coeffs(self, indent: str = "") -> str:
        lines = [f"{indent}PORCModel (L={self.L}, d={self.d})"]
        for r in sorted(self.coeffs_by_residue):
            coeffs = self.coeffs_by_residue[r]
            monomials_x = self.monomial_coeffs_by_residue[r]
            # monomials_t = _binom_to_monomial(coeffs)
            poly_str_x = _format_monomial_poly(monomials_x, var="n")
            # poly_str_t = _format_monomial_poly(monomials_t, var="t")
            mono_list_x = ", ".join(_format_fraction(c) for c in monomials_x)
            # lines.append(f"{indent}  n = {r} mod {self.L}: binom coeffs {coeffs}")
            lines.append(f"{indent}  residue {r} mod {self.L}:")
            lines.append(f"{indent}   p_r(x) = {poly_str_x},  monomial coeffs in n [{mono_list_x}]")
            # lines.append(f"{indent}    Q_r(t) = {poly_str_t}  (t = (x-{r})/{self.L})")
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - trivial wrapper
        return self._format_coeffs()

    __repr__ = __str__
