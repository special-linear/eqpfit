# eqpfit

Exact and eventual quasi-polynomial (PORC) fitting for integer-valued sequences.

## Overview
`eqpfit` fits **P**olynomial **O**n **R**esidue **C**lasses (PORC) models of bounded degree
against integer pairs `(x_i, v_i)`. Each residue class modulo a period `L` receives a
binomial-basis polynomial that evaluates to integers for every integer input. The library
supports:

- Known period or automatic scanning over feasible periods.
- Constraints on leading binomial coefficients (`common_leading`, `leading_coeff`).
- Difference- and flint-based integer solvers selected automatically when possible.
- Verification of exact fits, with structured failure reasons.
- Eventual fitting that discards a minimal prefix until a PORC fit is found.

## Installation
The project targets Python 3.11+. By default it depends only on the standard library
and can be installed from source with:

```bash
pip install -e .
```

To enable the optional `flint` backend, install with the `flint` extra:

```bash
pip install -e .[flint]
```

The extra pulls in `python-flint`; ensure any system requirements for that package are
satisfied on your platform.

## Key types
- **PORCModel**: Holds `coeffs_by_residue` (binomial-basis coefficients of length `d+1`
  per residue) and provides `eval(x)` and `verify(xs, vs)`.
- **FitResult**: Returned by `fit_period` and `fit_porc`, with attributes `L`, `d`,
  `success`, `model`, `reason`, and optional `details`.
- **EventualPORCResult**: Returned by `fit_eventual_porc`, with `start` (index of the
  first retained point), the `fit` result, and `dropped` count.

Readable `__str__`/`__repr__` forms summarize results, including per-residue binomial
coefficients and the corresponding monomial polynomials `Q_r(t)` (where
`t = (x - r) / L`) plus drop counts for eventual fits, so `print(fit_result)` gives a
helpful snapshot.

## Core APIs

### fit_period(xs, vs, d, L, *, require_all_residues=True, common_leading=False,
               leading_coeff=None, verify=True, backend="auto")
Fit a PORC of degree `d` with fixed period `L`.

- `xs`, `vs`: integer-like inputs; duplicates with conflicting values cause failure.
- `require_all_residues`: demand enough points for every residue class mod `L`.
- `common_leading`: enforce a shared leading binomial coefficient across residues.
- `leading_coeff`: explicitly set the leading binomial coefficient (implies
  `common_leading`).
- `backend`: "auto" (prefer differences on consecutive data, fall back to flint when
  available), "differences", or "flint".

Returns a `FitResult`. When `success` is True, `model.eval(x)` reproduces all inputs.

### fit_porc(xs, vs, d, period=None, *, require_all_residues=True,
             common_leading=False, leading_coeff=None, verify=True,
             return_all=False, include_failures=False, backend="auto")
Scan candidate periods and return the first successful fit (or all results when
`return_all=True`).

- `period=None`: enumerate feasible periods based on sample size (finite when
  `require_all_residues=True`).
- `period=int` or iterable of ints: test in the given order.
- `include_failures`: include unsuccessful `FitResult` entries when returning a list.

Failures use machine-readable reasons such as `"nonconsecutive_t_values"`,
`"common_leading_mismatch"`, or `"verification_failed"` to aid debugging.

### fit_eventual_porc(xs, vs, d, *, period=None, require_all_residues=True,
                     common_leading=False, leading_coeff=None, verify=True,
                     return_all=False, include_failures=False, backend="auto")
Find an eventual PORC: repeatedly drop the earliest point until a PORC fit of degree `d`
exists or no data remains.

- Carries the same leading-coefficient controls and backend selection as `fit_porc`.
- Returns an `EventualPORCResult` containing the offset of the first retained point,
  the successful `FitResult`, and how many points were dropped. On total failure,
  `fit` holds the final failing `FitResult`.

## Usage examples

### Fit a polynomial (period 1) with known degree
```python
from eqpfit import fit_period
xs = [0, 1, 2, 3]
vs = [0, 1, 4, 9]
res = fit_period(xs, vs, d=2, L=1)
assert res.success
print(res.model.coeffs_by_residue)  # {0: [0, 0, 1]}
print(res.model.eval(4))            # 16
```

### Scan periods with a shared leading coefficient
```python
from eqpfit import fit_porc
xs = [0, 1, 2, 3, 4, 5]
vs = [0, 2, 3, 7, 8, 14]  # period-2 behavior with common leading term
res = fit_porc(xs, vs, d=2, period=None, common_leading=True)
if res.success:
    print(res)  # Pretty printing includes residues, coefficients, and Q_r(t) monomials
    print(f"Period {res.L}, leading coeff {res.model.coeffs_by_residue[0][2]}")
```

### Find an eventual PORC by discarding a prefix
```python
from eqpfit import fit_eventual_porc
xs = [0, 1, 2, 3, 4, 5]
vs = [5, 6, 0, 1, 4, 9]  # first two points are noise, remainder is quadratic
res = fit_eventual_porc(xs, vs, d=2)
print(res)        # Pretty printing shows dropped prefix and fitted model
print(res.start)  # 2 (zero-based index of first retained point)
print(res.fit.L)  # inferred period of the remaining data
```

### Enforce a fixed leading coefficient
```python
from eqpfit import fit_period
xs = [0, 1, 2, 3]
vs = [1, 4, 11, 22]
res = fit_period(xs, vs, d=2, L=1, leading_coeff=1)
assert res.success
print(res.model.coeffs_by_residue[0])  # leading binomial coefficient equals 1
```

## Running tests
Execute the full test suite with:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

All tests are deterministic and require no external services.
