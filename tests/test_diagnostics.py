"""Public diagnostics: invariant targets, separated departures, and calibration."""
import math
import unittest

import numpy as np
from numpy.testing import assert_allclose

from didcheck import diagnose_arrays, variance_test, _core


class DiagnosticChecks(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(91283)
        self.y0 = self.rng.normal(size=(64, 4))
        self.y1 = self.rng.normal(size=(71, 4))*[1, 1.1, 1.3, 1.5]

    def fit(self, y0=None, y1=None, **kw):
        return diagnose_arrays(self.y0 if y0 is None else y0,
                               self.y1 if y1 is None else y1, **kw)

    def test_default_fisher_and_sum_do_not_assign_analytic_calibration(self):
        result = self.fit()
        self.assertTrue(result.mean.valid and result.variance.valid)
        self.assertFalse(result.fisher.valid)
        self.assertTrue(math.isnan(result.fisher.pvalue))
        self.assertFalse(result.combined.valid)
        self.assertTrue(math.isnan(result.combined.pvalue))
        self.assertIsNone(result.combined.df)
        self.assertIsNone(result.combined.reject)
        self.assertAlmostEqual(result.combined.statistic,
                               result.mean.statistic+result.variance.statistic)

    def test_explicit_fisher_matches_independent_closed_form(self):
        result = self.fit(fisher=True, fisher_justification='Assumed whole-trajectory central symmetry.')
        product = result.mean.pvalue*result.variance.pvalue
        expected = product*(1-math.log(product))
        self.assertTrue(result.fisher.valid)
        self.assertEqual(result.fisher.df, 4)
        self.assertAlmostEqual(result.fisher.pvalue, expected, places=14)
        self.assertAlmostEqual(result.fisher.statistic, -2*math.log(product), places=13)

    def test_each_reference_produces_same_wald_statistics(self):
        base = self.fit()
        for reference in range(4):
            with self.subTest(reference=reference):
                result = self.fit(reference=reference)
                assert_allclose([result.mean.statistic, result.variance.statistic],
                                [base.mean.statistic, base.variance.statistic], rtol=2e-12)

    def test_unit_shifts_common_time_path_and_common_scale(self):
        base = self.fit()
        variants = [
            (self.y0+self.rng.normal(size=(len(self.y0), 1)),
             self.y1+self.rng.normal(size=(len(self.y1), 1))),
            (self.y0+[2, -1, .5, 3], self.y1+[2, -1, .5, 3]),
            (-3*self.y0, -3*self.y1),
        ]
        for i, (a, b) in enumerate(variants):
            with self.subTest(transformation=i):
                actual = self.fit(a, b)
                assert_allclose([actual.mean.statistic, actual.variance.statistic],
                                [base.mean.statistic, base.variance.statistic], rtol=2e-12, atol=2e-12)
                assert_allclose(actual.variance_ratios, base.variance_ratios, rtol=2e-12)

    def test_group_scaling_changes_ratio_level_but_not_variance_contrast(self):
        base = self.fit()
        actual = self.fit(y1=2.5*self.y1)
        assert_allclose(actual.variance_ratios, base.variance_ratios*6.25, rtol=2e-12)
        assert_allclose(actual.variance_contrast, base.variance_contrast, atol=2e-15)
        self.assertAlmostEqual(actual.variance.statistic, base.variance.statistic, places=12)

    def test_identical_samples_and_separated_departures(self):
        same = self.fit(self.y0, self.y0)
        assert_allclose([same.mean.statistic, same.variance.statistic], 0, atol=0)
        assert_allclose([same.mean.pvalue, same.variance.pvalue], 1, atol=0)
        mean_only = self.fit(self.y0, self.y0+.7*np.arange(4))
        self.assertGreater(mean_only.mean.statistic, 1)
        self.assertLess(mean_only.variance.statistic, 1e-20)
        symmetric = np.concatenate((self.y0, -self.y0))
        variance_only = self.fit(symmetric, symmetric*[1, 1, 1, 2])
        self.assertLess(variance_only.mean.statistic, 1e-20)
        self.assertGreater(variance_only.variance.statistic, 1)

    def test_batch_core_and_public_single_panel_agree(self):
        a = np.stack([self.y0, self.y0+np.arange(4)])
        b = np.stack([self.y1, self.y1*[1, 1, 1, 1.1]])
        batched = _core.analyze_batch(a, b)
        for i in range(2):
            public = self.fit(a[i], b[i])
            assert_allclose(public.mean_contrast, batched['h_mean'][i], atol=0)
            assert_allclose(public.variance_covariance, batched['cov_var'][i], atol=0)
            self.assertEqual(public.variance.statistic, batched['W_var'][i])

    def test_chi_square_survival_against_scipy_wide_grid(self):
        from scipy.stats import chi2
        points = np.r_[0, np.logspace(-8, 3, 71), np.inf]
        for df in [1, 2, 3, 4, 5, 8, 11]:
            with self.subTest(df=df):
                assert_allclose(_core.chi2_sf(points, df), chi2.sf(points, df), rtol=3e-11, atol=2e-14)

    def test_chi_square_survival_by_independent_density_integration(self):
        # Composite Simpson integration is independent of gamma-tail recurrences.
        grid = np.linspace(2.3, 120, 20001)
        for df in [3, 4, 5, 8]:
            density = np.exp((df/2-1)*np.log(grid)-grid/2-math.lgamma(df/2)-df/2*math.log(2))
            numerical = (grid[1]-grid[0])/3*(density[0]+density[-1]
                        +4*density[1:-1:2].sum()+2*density[2:-1:2].sum())
            self.assertAlmostEqual(float(_core.chi2_sf(2.3, df)), numerical, places=10)

    def test_centered_roots_vanish_for_identity_resample(self):
        original = _core.analyze(self.y0, self.y1)
        star = _core.analyze_batch(self.y0[None], self.y1[None])
        w, good = _core.centered_sum_batch(star, original)
        self.assertTrue(good[0])
        self.assertEqual(w[0], 0)
        self.assertGreater(original['W_sum'], 1)

    def test_centered_sum_is_separate_component_quadratic_forms(self):
        original = _core.analyze(self.y0, self.y1)
        indices0 = self.rng.integers(0, len(self.y0), (7, len(self.y0)))
        indices1 = self.rng.integers(0, len(self.y1), (7, len(self.y1)))
        stars = _core.analyze_batch(self.y0[indices0], self.y1[indices1])
        actual, good = _core.centered_sum_batch(stars, original)
        for j in range(7):
            dm = stars['h_mean'][j]-original['h_mean']
            dv = stars['h_var'][j]-original['h_var']
            expected = dm @ np.linalg.inv(stars['cov_mean'][j]) @ dm
            expected += dv @ np.linalg.inv(stars['cov_var'][j]) @ dv
            self.assertTrue(good[j])
            self.assertAlmostEqual(actual[j], expected, places=12)

    def test_result_table_summary_and_alpha_decision(self):
        first = self.fit()
        same_threshold = self.fit(alpha=first.variance.pvalue)
        self.assertTrue(same_threshold.variance.reject)
        self.assertEqual(first.to_frame().index.tolist(), ['mean', 'variance', 'fisher', 'combined'])
        self.assertIn('Nonrejection does not establish', first.summary())
        self.assertEqual(first.mean.df, 3)
        self.assertEqual(first.variance.df, 3)
        self.assertTrue(any('numerical validity' in s for s in first.warnings))


if __name__ == '__main__':
    unittest.main()
