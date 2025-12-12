"""Exact PORC / quasi-polynomial fitting routines."""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .binom import binom_int, binom_row
from .model import EventualPORCResult, FitResult, PORCModel

# flint is optional; detect lazily so a runtime install (e.g., in a notebook) is picked up
_FLINT_AVAILABLE = False

def _ensure_flint_available() -> bool:
    """Try importing flint on demand and cache the result."""

    global _FLINT_AVAILABLE, fmpz_mat, fmpz  # type: ignore[name-defined]
    if _FLINT_AVAILABLE:
        return True
    try:
        from flint import fmpz_mat as _fmpz_mat, fmpz as _fmpz
    except Exception:
        return False
    fmpz_mat = _fmpz_mat  # type: ignore[assignment]
    fmpz = _fmpz          # type: ignore[assignment]
    _FLINT_AVAILABLE = True
    return True

# Attempt once at import time so existing behaviour is preserved when flint is already present
_ensure_flint_available()

Backend = str
PeriodSpec = Union[int, Iterable[int], None]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_points(xs: Iterable[int], vs: Iterable[int]) -> Tuple[List[int], List[int], Optional[FitResult]]:
    xs_list = list(xs)
    vs_list = list(vs)
    if len(xs_list) != len(vs_list):
        raise ValueError("xs and vs must have the same length")
    mapping: Dict[int, int] = {}
    for x_raw, v_raw in zip(xs_list, vs_list):
        x = int(x_raw)
        v = int(v_raw)
        if x in mapping and mapping[x] != v:
            return [], [], FitResult(L=1, d=0, success=False, model=None, reason="inconsistent_duplicate_x", details={"x": x})
        mapping[x] = v
    xs_unique = sorted(mapping.keys())
    vs_unique = [mapping[x] for x in xs_unique]
    return xs_unique, vs_unique, None


def _is_consecutive(xs: Sequence[int]) -> bool:
    return all(xs[i + 1] - xs[i] == 1 for i in range(len(xs) - 1))


def _find_consecutive_run(ts: List[int], needed: int) -> Optional[int]:
    if needed <= 0:
        return 0
    for i in range(len(ts) - needed + 1):
        ok = True
        for j in range(1, needed):
            if ts[i + j] != ts[i] + j:
                ok = False
                break
        if ok:
            return i
    return None


# ---------------------------------------------------------------------------
# Linear algebra helper (with optional flint)
# ---------------------------------------------------------------------------

def _solve_integer_system(A: List[List[int]], b: List[int]) -> Tuple[Optional[List[int]], bool]:
    """Solve ``A x = b`` over the integers.

    Returns (solution, rank_deficient). The solution is a list of ints if a unique
    integer solution exists; otherwise None.
    """

    if _ensure_flint_available():
        try:
            m = len(A)
            n = len(A[0]) if A else 0
            Amat = fmpz_mat(m, n, [fmpz(val) for row in A for val in row])
            bmat = fmpz_mat(m, 1, [fmpz(val) for val in b])
            sol = Amat.solve(bmat)
            if sol is None:
                return None, False
            solution = [int(sol[i, 0]) for i in range(n)]
            # verify consistency for overdetermined systems
            if (Amat * sol) != bmat:
                return None, False
            return solution, False
        except Exception:
            # fall back to Fraction-based solver
            pass

    m = len(A)
    n = len(A[0]) if A else 0
    aug = [list(map(Fraction, row)) + [Fraction(bi)] for row, bi in zip(A, b)]
    rank = 0
    col = 0
    while rank < m and col < n:
        pivot = None
        for r in range(rank, m):
            if aug[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            col += 1
            continue
        aug[rank], aug[pivot] = aug[pivot], aug[rank]
        pivot_val = aug[rank][col]
        aug[rank] = [v / pivot_val for v in aug[rank]]
        for r in range(m):
            if r == rank:
                continue
            factor = aug[r][col]
            if factor == 0:
                continue
            aug[r] = [v - factor * vr for v, vr in zip(aug[r], aug[rank])]
        rank += 1
        col += 1
    # Check for inconsistency
    for r in range(rank, m):
        if all(aug[r][c] == 0 for c in range(n)):
            if aug[r][n] != 0:
                return None, False
    rank_deficient = rank < n
    if rank_deficient:
        return None, True
    # back substitution (already reduced)
    solution = [Fraction(0) for _ in range(n)]
    for r in range(rank - 1, -1, -1):
        pivot_col = next(c for c in range(n) if aug[r][c] != 0)
        val = aug[r][n] - sum(aug[r][c] * solution[c] for c in range(pivot_col + 1, n))
        solution[pivot_col] = val / aug[r][pivot_col]
    ints: List[int] = []
    for q in solution:
        if q.denominator != 1:
            return None, False
        ints.append(int(q.numerator))
    return ints, False


# ---------------------------------------------------------------------------
# Differences backend
# ---------------------------------------------------------------------------

def _newton_coefficients(ts: List[int], vs: List[int], degree: int) -> Tuple[int, List[int]]:
    """Return (t0, a_list) for Newton forward differences of given degree."""

    t0 = ts[0]
    diffs = [Fraction(v) for v in vs]
    a_coeffs = [diffs[0]]
    for k in range(1, degree + 1):
        diffs = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]
        a_coeffs.append(diffs[0])
    return t0, [int(a) for a in a_coeffs]


def _convert_shifted_binom(t0: int, a_coeffs: List[int], degree: int) -> List[int]:
    coeffs = [0 for _ in range(degree + 1)]
    for k in range(degree + 1):
        for j in range(0, k + 1):
            coeffs[j] += a_coeffs[k] * binom_int(-t0, k - j)
    return coeffs


def _differences_fit_for_L(xs: List[int], vs: List[int], d: int, L: int, *, common_leading: bool, leading_coeff: Optional[int], require_all_residues: bool) -> FitResult:
    residue_points: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for x, v in zip(xs, vs):
        r = x % L
        t = (x - r) // L
        residue_points[r].append((t, v))

    coeffs_by_residue: Dict[int, List[int]] = {}
    leading_value: Optional[int] = leading_coeff

    if d == 0:
        for r in range(L):
            pts = residue_points.get(r, [])
            if not pts:
                if require_all_residues:
                    return FitResult(L, d, False, None, reason="insufficient_points_in_residue", details={"residue": r})
                continue
            values = {v for _, v in pts}
            if len(values) > 1:
                return FitResult(L, d, False, None, reason="no_integer_solution", details={"residue": r})
            c0 = values.pop()
            if leading_coeff is not None and c0 != leading_coeff:
                return FitResult(L, d, False, None, reason="no_integer_solution", details={"residue": r})
            coeffs_by_residue[r] = [c0]
            leading_value = leading_value if leading_value is not None else c0
        if common_leading:
            lead_values = {coeffs_by_residue.get(r, [leading_value])[0] for r in range(L)}
            if len(lead_values) > 1:
                return FitResult(L, d, False, None, reason="common_leading_mismatch")
            lead = lead_values.pop() if lead_values else (leading_value if leading_value is not None else 0)
            for r in range(L):
                coeffs_by_residue.setdefault(r, [lead])
        model = PORCModel(L, d, coeffs_by_residue)
        return FitResult(L, d, True, model)

    needed_run = d if leading_coeff is not None else d + 1
    for r in range(L):
        pts = residue_points.get(r, [])
        if len(pts) < needed_run:
            if require_all_residues:
                return FitResult(L, d, False, None, reason="insufficient_points_in_residue", details={"residue": r})
            else:
                return FitResult(L, d, False, None, reason="nonconsecutive_t_values", details={"residue": r})
        pts.sort()
        ts = [t for t, _ in pts]
        vs_residue = [v for _, v in pts]
        start = _find_consecutive_run(ts, needed_run)
        if start is None:
            return FitResult(L, d, False, None, reason="nonconsecutive_t_values", details={"residue": r})
        ts_run = ts[start:start + needed_run]
        vs_run = vs_residue[start:start + needed_run]
        if leading_coeff is not None:
            a = leading_coeff
            adjusted = [v - a * binom_int(t, d) for t, v in zip(ts_run, vs_run)]
            t0, a_coeffs = _newton_coefficients(ts_run, adjusted, d - 1)
            coeffs = _convert_shifted_binom(t0, a_coeffs, d - 1)
            coeffs.append(a)
        else:
            t0, a_coeffs = _newton_coefficients(ts_run, vs_run, d)
            coeffs = _convert_shifted_binom(t0, a_coeffs, d)
        coeffs_by_residue[r] = coeffs
        if leading_value is None:
            leading_value = coeffs[-1]
        elif common_leading and coeffs[-1] != leading_value:
            return FitResult(L, d, False, None, reason="common_leading_mismatch")

    if common_leading and leading_value is not None:
        for coeffs in coeffs_by_residue.values():
            if coeffs[-1] != leading_value:
                return FitResult(L, d, False, None, reason="common_leading_mismatch")

    model = PORCModel(L, d, coeffs_by_residue)
    return FitResult(L, d, True, model)


# ---------------------------------------------------------------------------
# Flint backend (with pure-Python fallback)
# ---------------------------------------------------------------------------

def _rank(A: List[List[int]]) -> int:
    if not A:
        return 0
    m = len(A)
    n = len(A[0])
    aug = [list(map(Fraction, row)) for row in A]
    rank = 0
    col = 0
    while rank < m and col < n:
        pivot = None
        for r in range(rank, m):
            if aug[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            col += 1
            continue
        aug[rank], aug[pivot] = aug[pivot], aug[rank]
        pivot_val = aug[rank][col]
        aug[rank] = [v / pivot_val for v in aug[rank]]
        for r in range(rank + 1, m):
            factor = aug[r][col]
            if factor == 0:
                continue
            aug[r] = [v - factor * vr for v, vr in zip(aug[r], aug[rank])]
        rank += 1
        col += 1
    return rank


def _flint_fit_for_L(xs: List[int], vs: List[int], d: int, L: int, *, common_leading: bool, leading_coeff: Optional[int], require_all_residues: bool) -> FitResult:
    if not _ensure_flint_available():
        return FitResult(L, d, False, None, reason="flint_not_installed")

    residue_points: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for x, v in zip(xs, vs):
        r = x % L
        t = (x - r) // L
        residue_points[r].append((t, v))

    if d == 0 and leading_coeff is not None:
        coeffs_by_residue = {r: [leading_coeff] for r in range(L)}
        for r, pts in residue_points.items():
            for _, v in pts:
                if v != leading_coeff:
                    return FitResult(L, d, False, None, reason="no_integer_solution", details={"residue": r})
        model = PORCModel(L, d, coeffs_by_residue)
        return FitResult(L, d, True, model)

    if not common_leading:
        coeffs_by_residue: Dict[int, List[int]] = {}
        unknowns = d + 1 if (leading_coeff is None or d == 0) else d
        for r in range(L):
            pts = residue_points.get(r, [])
            if len(pts) < unknowns:
                if require_all_residues:
                    return FitResult(L, d, False, None, reason="insufficient_points_in_residue", details={"residue": r})
                else:
                    return FitResult(L, d, False, None, reason="rank_deficient", details={"residue": r})
            pts.sort()
            ts = [t for t, _ in pts]
            mat_rows = []
            rhs = []
            for t, v in pts:
                row = binom_row(t, d)
                if leading_coeff is not None and d > 0:
                    v = v - leading_coeff * row[-1]
                    row = row[:-1]
                mat_rows.append(row)
                rhs.append(v)
            solution, rank_def = _solve_integer_system(mat_rows, rhs)
            if rank_def:
                return FitResult(L, d, False, None, reason="rank_deficient", details={"residue": r})
            if solution is None:
                return FitResult(L, d, False, None, reason="no_integer_solution", details={"residue": r})
            if leading_coeff is not None and d > 0:
                solution.append(leading_coeff)
            coeffs_by_residue[r] = solution
        model = PORCModel(L, d, coeffs_by_residue)
        return FitResult(L, d, True, model)

    # common leading case
    if d == 0:
        # Shared constant
        pts = [(x, v) for pairs in residue_points.values() for x, v in pairs]
        if not pts:
            return FitResult(L, d, False, None, reason="insufficient_total_points")
        values = {v for _, v in pts}
        if len(values) > 1:
            return FitResult(L, d, False, None, reason="no_integer_solution")
        const = values.pop()
        coeffs_by_residue = {r: [const] for r in range(L)}
        model = PORCModel(L, d, coeffs_by_residue)
        return FitResult(L, d, True, model)

    # Build global system
    alpha_unknown = leading_coeff is None
    unknowns = L * d + (1 if alpha_unknown else 0)
    total_points = sum(len(vs_r) for vs_r in residue_points.values())
    if total_points < unknowns:
        return FitResult(L, d, False, None, reason="insufficient_total_points")
    mat_rows: List[List[int]] = []
    rhs: List[int] = []
    for x, v in zip(xs, vs):
        r = x % L
        t = (x - r) // L
        row = [0 for _ in range(unknowns)]
        base = r * d
        coeffs = binom_row(t, d)
        for k in range(d):
            row[base + k] = coeffs[k]
        if alpha_unknown:
            row[-1] = coeffs[-1]
        else:
            v -= leading_coeff * coeffs[-1]
        mat_rows.append(row)
        rhs.append(v)
    solution, rank_def = _solve_integer_system(mat_rows, rhs)
    if rank_def:
        return FitResult(L, d, False, None, reason="rank_deficient")
    if solution is None:
        return FitResult(L, d, False, None, reason="no_integer_solution")
    coeffs_by_residue: Dict[int, List[int]] = {}
    for r in range(L):
        start = r * d
        coeffs = solution[start:start + d]
        if alpha_unknown:
            coeffs.append(solution[-1])
        else:
            coeffs.append(leading_coeff)
        coeffs_by_residue[r] = coeffs
    model = PORCModel(L, d, coeffs_by_residue)
    return FitResult(L, d, True, model)


# ---------------------------------------------------------------------------
# Period utilities
# ---------------------------------------------------------------------------

def _minimal_period(seq: List[int]) -> Optional[int]:
    if not seq:
        return None
    n = len(seq)
    for p in range(1, n + 1):
        if all(seq[i] == seq[i % p] for i in range(n)):
            return p
    return None


def _differences(sequence: List[int], order: int) -> List[int]:
    data = list(sequence)
    for _ in range(order):
        data = [data[i + 1] - data[i] for i in range(len(data) - 1)]
    return data


def _candidate_periods_from_inference(xs: List[int], vs: List[int], d: int, Lmax: int) -> List[int]:
    if not xs or not _is_consecutive(xs):
        return []
    order = d + 1
    if len(vs) <= order:
        return []
    delta = _differences(vs, order)
    p = _minimal_period(delta)
    if p is None:
        return []
    candidates = set()
    for k in range(1, p + 1):
        if p % k == 0:
            candidates.add(k)
    multiple = p
    while multiple <= Lmax:
        candidates.add(multiple)
        multiple += p
    return sorted(candidates)


def _feasible_periods(xs: List[int], d: int, Lmax: int, *, common_leading: bool, leading_coeff: Optional[int], backend: Backend) -> List[int]:
    if d == 0:
        min_per = 1
    elif leading_coeff is not None:
        min_per = d
    else:
        min_per = d + 1
    feasible = []
    for L in range(1, Lmax + 1):
        counts = defaultdict(int)
        for x in xs:
            counts[x % L] += 1
        if any(counts[r] < min_per for r in range(L)):
            continue
        if backend == "flint" and common_leading and d > 0:
            unknowns = L * d + (1 if leading_coeff is None else 0)
            if len(xs) < unknowns:
                continue
        feasible.append(L)
    return feasible


# ---------------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------------

def fit_period(
    xs: Iterable[int],
    vs: Iterable[int],
    d: int,
    L: int,
    *,
    require_all_residues: bool = True,
    common_leading: bool = False,
    leading_coeff: Optional[int] = None,
    verify: bool = True,
    backend: Backend = "auto",
) -> FitResult:
    xs_norm, vs_norm, failure = _normalize_points(xs, vs)
    if failure:
        return failure
    if L <= 0:
        raise ValueError("L must be positive")
    if d < 0:
        raise ValueError("d must be non-negative")

    def verify_model(res: FitResult) -> FitResult:
        if not res.success or not verify or res.model is None:
            return res
        ok, counter = res.model.verify(xs_norm, vs_norm)
        if not ok:
            return FitResult(L, d, False, None, reason="verification_failed", details={"counterexample": counter})
        return res

    if backend == "differences":
        return verify_model(_differences_fit_for_L(xs_norm, vs_norm, d, L, common_leading=common_leading, leading_coeff=leading_coeff, require_all_residues=require_all_residues))
    if backend == "flint":
        if not _ensure_flint_available():
            return FitResult(L, d, False, None, reason="flint_not_installed")
        return verify_model(_flint_fit_for_L(xs_norm, vs_norm, d, L, common_leading=common_leading, leading_coeff=leading_coeff, require_all_residues=require_all_residues))
    if backend != "auto":
        raise ValueError(f"Unknown backend: {backend}")

    # auto backend
    if _is_consecutive(xs_norm):
        diff_result = _differences_fit_for_L(xs_norm, vs_norm, d, L, common_leading=common_leading, leading_coeff=leading_coeff, require_all_residues=require_all_residues)
        if diff_result.success:
            return verify_model(diff_result)
        if diff_result.reason in {"insufficient_points_in_residue", "nonconsecutive_t_values"}:
            if not _ensure_flint_available():
                return FitResult(L, d, False, None, reason="flint_required_for_nonconsecutive_data")
            alt = _flint_fit_for_L(xs_norm, vs_norm, d, L, common_leading=common_leading, leading_coeff=leading_coeff, require_all_residues=require_all_residues)
            return verify_model(alt)
        return diff_result
    else:
        if not _ensure_flint_available():
            return FitResult(L, d, False, None, reason="flint_required_for_nonconsecutive_data")
        alt = _flint_fit_for_L(xs_norm, vs_norm, d, L, common_leading=common_leading, leading_coeff=leading_coeff, require_all_residues=require_all_residues)
        return verify_model(alt)


def fit_porc(
    xs: Iterable[int],
    vs: Iterable[int],
    d: int,
    period: PeriodSpec = None,
    *,
    require_all_residues: bool = True,
    common_leading: bool = False,
    leading_coeff: Optional[int] = None,
    verify: bool = True,
    return_all: bool = False,
    include_failures: bool = False,
    backend: Backend = "auto",
) -> Union[FitResult, List[FitResult]]:
    xs_norm, vs_norm, failure = _normalize_points(xs, vs)
    if failure:
        return [failure] if return_all else failure

    if isinstance(period, int):
        result = fit_period(xs_norm, vs_norm, d, period, require_all_residues=require_all_residues, common_leading=common_leading, leading_coeff=leading_coeff, verify=verify, backend=backend)
        return [result] if return_all else result

    if period is not None:
        results: List[FitResult] = []
        for L in period:
            res = fit_period(xs_norm, vs_norm, d, int(L), require_all_residues=require_all_residues, common_leading=common_leading, leading_coeff=leading_coeff, verify=verify, backend=backend)
            if return_all:
                if include_failures or res.success:
                    results.append(res)
            elif res.success:
                return res
        if return_all:
            return results
        return FitResult(L=1, d=d, success=False, model=None, reason="no_period_found")

    # period None: scan
    if not require_all_residues:
        return FitResult(L=1, d=d, success=False, model=None, reason="no_period_found")
    m = len(xs_norm)
    if d == 0:
        min_per = 1
    elif leading_coeff is not None:
        min_per = d
    else:
        min_per = d + 1
    Lmax = m // min_per
    if Lmax <= 0:
        return FitResult(L=1, d=d, success=False, model=None, reason="no_period_found")

    candidates = []
    inferred = _candidate_periods_from_inference(xs_norm, vs_norm, d, Lmax)
    candidates.extend(inferred)
    strict = _feasible_periods(xs_norm, d, Lmax, common_leading=common_leading, leading_coeff=leading_coeff, backend=backend if backend != "auto" else ("differences" if _is_consecutive(xs_norm) else "flint"))
    for L in strict:
        if L not in candidates:
            candidates.append(L)
    if not candidates:
        return FitResult(L=1, d=d, success=False, model=None, reason="no_period_found")

    results: List[FitResult] = []
    for L in candidates:
        res = fit_period(xs_norm, vs_norm, d, L, require_all_residues=require_all_residues, common_leading=common_leading, leading_coeff=leading_coeff, verify=verify, backend=backend)
        if return_all:
            if include_failures or res.success:
                results.append(res)
        elif res.success:
            return res
    if return_all:
        return results
    return FitResult(L=1, d=d, success=False, model=None, reason="no_period_found")


def fit_eventual_porc(
    xs: Iterable[int],
    vs: Iterable[int],
    d: int,
    *,
    period: PeriodSpec = None,
    require_all_residues: bool = True,
    common_leading: bool = False,
    leading_coeff: Optional[int] = None,
    verify: bool = True,
    backend: Backend = "auto",
) -> EventualPORCResult:
    """Attempt to fit a PORC model after discarding an initial prefix.

    The function repeatedly tries to fit a quasi-polynomial to the suffixes of
    the provided data, discarding the first point on each failure until a
    solution is found or no points remain. Leading-coefficient constraints are
    forwarded to the underlying ``fit_porc`` call.
    """

    xs_norm, vs_norm, failure = _normalize_points(xs, vs)
    if failure:
        return EventualPORCResult(
            success=False,
            start_index=0,
            start_x=None,
            fit_result=failure,
            reason=failure.reason,
            details=failure.details,
        )

    for start in range(len(xs_norm)):
        xs_sub = xs_norm[start:]
        vs_sub = vs_norm[start:]
        result = fit_porc(
            xs_sub,
            vs_sub,
            d,
            period=period,
            require_all_residues=require_all_residues,
            common_leading=common_leading,
            leading_coeff=leading_coeff,
            verify=verify,
            backend=backend,
        )
        if isinstance(result, list):  # defensive: fit_porc return_all=False yields FitResult
            result = result[0] if result else FitResult(L=1, d=d, success=False, model=None, reason="no_period_found")
        if result.success:
            return EventualPORCResult(
                success=True,
                start_index=start,
                start_x=xs_sub[0] if xs_sub else None,
                fit_result=result,
            )

    return EventualPORCResult(
        success=False,
        start_index=len(xs_norm),
        start_x=None,
        fit_result=None,
        reason="no_eventual_fit_found",
    )
