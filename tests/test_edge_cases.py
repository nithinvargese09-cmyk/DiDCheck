"""Fail visibly on unsupported windows, malformed panels, and singular designs."""
import unittest
import warnings

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from didcheck import diagnose_arrays, variance_test, _core


def panel_frame(y0, y1):
    rows = []
    for group, y in enumerate((y0, y1)):
        for i, row in enumerate(y):
            for t, value in enumerate(row):
                rows.append(dict(unit=f'{group}:{i}', time=2010+t, treated=group, outcome=value))
    return pd.DataFrame(rows)


class EdgeCaseChecks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(589)
        self.y0, self.y1 = rng.normal(size=(35, 4)), rng.normal(size=(37, 4))
        self.data = panel_frame(self.y0, self.y1)

    def table_fit(self, data=None, **kw):
        args = dict(outcome='outcome', unit='unit', time='time', treatment='treated',
                    pre_periods=[2010, 2011, 2012, 2013])
        args.update(kw)
        return variance_test(self.data if data is None else data, **args)

    def test_requires_three_preperiods_and_complete_two_dimensional_arrays(self):
        pairs = [(self.y0[:, :2], self.y1[:, :2]),
                 (self.y0[:, 0], self.y1[:, 0]),
                 (self.y0[None], self.y1[None]),
                 (self.y0[:, :3], self.y1),
                 (self.y0[:1], self.y1)]
        for a, b in pairs:
            with self.subTest(shape0=a.shape, shape1=b.shape), self.assertRaises(ValueError):
                diagnose_arrays(a, b)

    def test_nonfinite_outcomes_raise_without_silent_deletion(self):
        for value in [np.nan, np.inf, -np.inf]:
            bad = self.y0.copy()
            bad[2, 1] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                diagnose_arrays(bad, self.y1)

    def test_reference_and_analysis_options_are_checked(self):
        bad_options = [dict(reference=-1), dict(reference=4), dict(reference=.5),
                       dict(reference=True), dict(alpha=0), dict(alpha=1), dict(alpha=np.nan),
                       dict(alpha=True), dict(bootstrap=-1), dict(bootstrap=1.5),
                       dict(batch_size=0), dict(random_state=-1), dict(fisher='yes'),
                       dict(fisher=True), dict(fisher=True, fisher_justification='  ')]
        for options in bad_options:
            with self.subTest(options=options), self.assertRaises(ValueError):
                diagnose_arrays(self.y0, self.y1, **options)

    def test_zero_variance_and_singular_covariance_are_unavailable(self):
        bad_pairs = [(np.zeros_like(self.y0), np.zeros_like(self.y1)),
                     (np.arange(35)[:, None]*np.arange(4)[None, :],
                      np.arange(37)[:, None]*np.arange(4)[None, :])]
        for a, b in bad_pairs:
            with self.subTest(shape=a.shape), warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                result = diagnose_arrays(a, b)
                self.assertFalse(result.variance.valid)
                self.assertTrue(np.isnan(result.variance.pvalue))
                self.assertIsNone(result.variance.reject)
                self.assertTrue(any('unavailable' in s for s in result.warnings))

    def test_no_pseudoinverse_or_reduced_degrees_of_freedom(self):
        h = np.array([[1., 0, 0]])
        singular = np.diag([1., 1, 0])[None]
        stat, valid = _core.wald_batch(h, singular)
        self.assertFalse(valid[0])
        self.assertTrue(np.isnan(stat[0]))
        for smallest, should_pass in [(1e-11, False), (1e-9, True)]:
            _, valid = _core.wald_batch(h, np.diag([1., .5, smallest])[None])
            self.assertEqual(bool(valid[0]), should_pass)

    def test_internal_two_period_degeneracy_is_exact(self):
        result = _core.analyze(self.y0[:, :2], self.y1[:, :2])
        self.assertFalse(result['valid_var'])
        assert_allclose(result['h_var'], 0, atol=1e-15)
        self.assertTrue(result['two_period_degeneracy'])

    def test_table_order_and_explicit_reference_match_arrays(self):
        periods = [2013, 2011, 2010, 2012]
        actual = self.table_fit(self.data.sample(frac=1, random_state=41),
                                pre_periods=periods, reference=2010)
        expected = diagnose_arrays(self.y0[:, [3, 1, 0, 2]], self.y1[:, [3, 1, 0, 2]], reference=2)
        assert_allclose(actual.variance_contrast, expected.variance_contrast, atol=2e-15)
        assert_allclose(actual.mean_covariance, expected.mean_covariance, atol=2e-15)
        self.assertEqual(actual.periods, tuple(periods))
        self.assertEqual(actual.reference, 2010)

    def test_duplicate_unit_time_and_time_varying_group_rejected(self):
        duplicate = pd.concat([self.data, self.data.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            self.table_fit(duplicate)
        changing = self.data.copy()
        changing.loc[0, 'treated'] = 1
        with self.assertRaises(ValueError):
            self.table_fit(changing)
        nonbinary = self.data.copy()
        nonbinary.loc[nonbinary.treated == 1, 'treated'] = 2
        with self.assertRaises(ValueError):
            self.table_fit(nonbinary)

    def test_explicit_missing_policy_reports_dropped_units(self):
        incomplete = self.data.drop(index=0)
        with self.assertRaises(ValueError):
            self.table_fit(incomplete)
        result = self.table_fit(incomplete, missing='drop_units')
        expected = diagnose_arrays(self.y0[1:], self.y1)
        assert_allclose(result.variance.statistic, expected.variance.statistic, rtol=2e-12)
        self.assertEqual(result.metadata['dropped_units'], ('0:0',))
        self.assertEqual(result.n_control, 34)
        self.assertTrue(any('removed 1 units' in s for s in result.warnings))
        nonfinite = self.data.copy()
        nonfinite.loc[0, 'outcome'] = np.nan
        with self.assertRaises(ValueError):
            self.table_fit(nonfinite)
        dropped = self.table_fit(nonfinite, missing='drop_units')
        assert_allclose(dropped.variance.statistic, expected.variance.statistic, rtol=2e-12)

    def test_missing_window_invalid_reference_and_policy_rejected(self):
        for options in [dict(pre_periods=[2010, 2011]),
                        dict(pre_periods=[2010, 2011, 2011]),
                        dict(pre_periods=[2010, 2011, 2099]),
                        dict(reference=2099), dict(missing='silently_drop')]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.table_fit(**options)

    def test_invalid_cluster_layouts_rejected(self):
        options = [dict(cluster_size=3), dict(cluster_size=0), dict(cluster_size=True),
                   dict(cluster_size=(35, 37)), dict(clusters=([0]*35, [0]*37)),
                   dict(clusters=([0]*34, [1]*37)),
                   dict(clusters=(list(range(35)), list(range(37))), cluster_size=1)]
        for kw in options:
            with self.subTest(options=kw), self.assertRaises(ValueError):
                diagnose_arrays(self.y0, self.y1, **kw)


if __name__ == '__main__':
    unittest.main()
