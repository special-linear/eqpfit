import unittest
from fractions import Fraction

import eqpfit.binom as binom
from eqpfit import (
    EventualPORCResult,
    FitResult,
    PORCModel,
    fit_eventual_porc,
    fit_period,
    fit_porc,
)


class BinomTests(unittest.TestCase):
    def test_binom_int_positive_and_negative(self):
        self.assertEqual(binom.binom_int(5, 2), 10)
        self.assertEqual(binom.binom_int(-3, 2), 6)  # (-1)^2 * C(4,2)
        with self.assertRaises(ValueError):
            binom.binom_int(3, -1)

    def test_binom_row_and_eval(self):
        row = binom.binom_row(4, 3)
        self.assertEqual(row, [1, 4, 6, 4])
        coeffs = [2, -1, 3]
        value = binom.eval_binom_poly(coeffs, 4)
        expected = sum(c * b for c, b in zip(coeffs, binom.binom_row(4, 2)))
        self.assertEqual(value, expected)


class ModelTests(unittest.TestCase):
    def test_model_eval_and_verify(self):
        coeffs_by_residue = {0: [0, 1, 2]}  # n^2 = C(n,1) + 2*C(n,2)
        model = PORCModel(L=1, d=2, coeffs_by_residue=coeffs_by_residue)
        self.assertEqual(model.eval(4), 16)
        ok, counter = model.verify([0, 1, 2, 3], [0, 1, 4, 9])
        self.assertTrue(ok)
        self.assertIsNone(counter)

    def test_string_shows_monomials(self):
        coeffs_by_residue = {0: [0, 1, 2]}  # translates to Q(t) = t^2
        model = PORCModel(L=1, d=2, coeffs_by_residue=coeffs_by_residue)
        formatted = model._format_coeffs()
        # self.assertIn("binom coeffs [0, 1, 2]", formatted)
        self.assertIn("monomial coeffs in n [0, 0, 1]", formatted)
        # self.assertIn("Q_r(t) = t^2", formatted)

    def test_monomial_coeffs_exposed_on_results(self):
        coeffs_by_residue = {0: [0, 1, 2]}
        model = PORCModel(L=1, d=2, coeffs_by_residue=coeffs_by_residue)
        self.assertEqual(
            model.monomial_coeffs_by_residue[0],
            [Fraction(0), Fraction(0), Fraction(1)],
        )

        fit_result = fit_period([0, 1, 2, 3], [0, 1, 4, 9], d=2, L=1, backend="auto")
        self.assertTrue(fit_result.success)
        self.assertEqual(
            fit_result.monomial_coeffs_by_residue,
            {0: [Fraction(0), Fraction(0), Fraction(1)]},
        )

    def test_monomial_coeffs_use_original_variable(self):
        xs = list(range(6))
        vs = [x // 2 for x in xs]  # period-2 behavior: floor(x/2)
        res = fit_period(xs, vs, d=1, L=2, backend="auto")
        self.assertTrue(res.success)
        self.assertEqual(
            res.monomial_coeffs_by_residue,
            {
                0: [Fraction(0), Fraction(1, 2)],
                1: [Fraction(-1, 2), Fraction(1, 2)],
            },
        )
        formatted = res._format()
        self.assertIn("monomial coeffs in n [0, 1/2]", formatted)

    def test_as_coeffs_array_sympy_format(self):
        xs = list(range(6))
        vs = [x // 2 for x in xs]  # period-2 behavior: floor(x/2)
        res = fit_period(xs, vs, d=1, L=2, backend="auto")
        self.assertTrue(res.success)
        self.assertIsNotNone(res.model)
        coeff_arrays = res.model.as_coeffs_array()
        self.assertEqual(coeff_arrays, [[0, "1/2"], ["-1/2", "1/2"]])

    def test_string_formats(self):
        fit_ok = fit_period([0, 2, 1, 3], [1, 3, 7, 11], d=1, L=2, backend="auto")
        self.assertTrue(fit_ok.success)

        formatted_fit = fit_ok._format()
        self.assertIn("SUCCESS", formatted_fit)
        self.assertIn("residue 0", formatted_fit)
        self.assertIn("residue 1", formatted_fit)
        # self.assertIn("Q_r(t)", formatted_fit)

        fail_fit = FitResult(L=2, d=1, success=False, model=None, reason="boom", details={"k": 1})
        self.assertIn("FAILED", fail_fit._format())
        self.assertIn("boom", fail_fit._format())

        eventual = EventualPORCResult(
            success=True,
            start_index=1,
            start_x=0,
            fit_result=fit_ok,
        )
        formatted_eventual = eventual._format()
        self.assertIn("dropped=1", formatted_eventual)
        self.assertIn("period 2", formatted_eventual)


class FitPeriodTests(unittest.TestCase):
    def test_fit_period_consecutive_polynomial(self):
        xs = [0, 1, 2, 3, 4]
        vs = [x * x for x in xs]
        res = fit_period(xs, vs, d=2, L=1, backend="auto")
        self.assertTrue(res.success)
        self.assertEqual(res.reason, None)
        self.assertEqual(res.model.coeffs_by_residue[0], [0, 1, 2])
        self.assertEqual(res.model.eval(5), 25)

    def test_common_leading_enforced(self):
        xs = [0, 1, 2, 3, 4, 5]
        vs = [x * x for x in xs]
        res = fit_period(xs, vs, d=2, L=2, common_leading=True, backend="auto")
        self.assertTrue(res.success)
        self.assertEqual(res.model.coeffs_by_residue[0][-1], 8)
        self.assertEqual(res.model.coeffs_by_residue[1][-1], 8)

    def test_leading_coeff_fixed(self):
        xs = [0, 1, 2, 3, 4, 5]
        vs = [x * x for x in xs]
        res = fit_period(xs, vs, d=2, L=2, leading_coeff=8, backend="auto")
        self.assertTrue(res.success)
        for coeffs in res.model.coeffs_by_residue.values():
            self.assertEqual(coeffs[-1], 8)

    def test_insufficient_points_failure(self):
        xs = [0, 2, 4]  # only residue 0 points for L=2
        vs = [0, 4, 16]
        res = fit_period(xs, vs, d=2, L=2, backend="differences")
        self.assertFalse(res.success)
        self.assertEqual(res.reason, "insufficient_points_in_residue")

    def test_verification_failure(self):
        xs = [0, 1, 2]
        vs = [0, 1, 5]  # not a quadratic with d=2 through x=0,1,2 consecutively
        res = fit_period(xs, vs, d=1, L=1, backend="auto")
        self.assertFalse(res.success)
        self.assertEqual(res.reason, "verification_failed")


class FitPorcTests(unittest.TestCase):
    def test_period_scan_prefers_success(self):
        xs = [0, 1, 2, 3]
        vs = [0, 1, 4, 9]
        res = fit_porc(xs, vs, d=2, period=None, backend="auto")
        self.assertTrue(res.success)
        self.assertEqual(res.L, 1)

    def test_inconsistent_duplicate_x(self):
        xs = [0, 0, 1]
        vs = [1, 2, 3]
        res = fit_porc(xs, vs, d=0, period=1, backend="auto")
        self.assertFalse(res.success)
        self.assertEqual(res.reason, "inconsistent_duplicate_x")


class FitEventualPorcTests(unittest.TestCase):
    def test_eventual_fit_discards_prefix(self):
        xs = [-1, 0, 1, 2, 3]
        vs = [5, 0, 1, 4, 9]  # leading point breaks global fit
        res = fit_eventual_porc(xs, vs, d=2, period=1, backend="auto")
        self.assertTrue(res.success)
        self.assertEqual(res.start_index, 1)
        self.assertEqual(res.start_x, 0)
        self.assertIsNotNone(res.fit_result)
        self.assertEqual(res.fit_result.model.eval(4), 16)

    def test_eventual_fit_with_leading_coeff_constraint(self):
        xs = [0, 1, 2, 3, 4, 5]
        vs = [10, 1, 4, 9, 16, 25]  # first point off
        res = fit_eventual_porc(xs, vs, d=2, period=2, leading_coeff=8, backend="auto")
        self.assertTrue(res.success)
        self.assertEqual(res.start_index, 1)
        self.assertEqual(res.fit_result.model.coeffs_by_residue[0][-1], 8)
        self.assertEqual(res.fit_result.model.coeffs_by_residue[1][-1], 8)

    def test_eventual_fit_failure(self):
        xs = [0, 2]
        vs = [0, 4]
        res = fit_eventual_porc(xs, vs, d=1, period=2, backend="auto")
        self.assertFalse(res.success)
        self.assertEqual(res.reason, "no_eventual_fit_found")


if __name__ == "__main__":
    unittest.main()
