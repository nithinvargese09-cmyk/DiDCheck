"""Consequential covariance checks by routes independent of the vectorized core.

The loop oracle uses cross-fourth moments and explicit reference differences;
it does not call the implementation's moment, contrast, or covariance helpers.
These finite examples verify arithmetic, not asymptotic coverage.
"""
import math
import unittest

import numpy as np
from numpy.testing import assert_allclose

from didcheck import _core
from didcheck.covariance import contrast_matrix, within_unit_demean, wald_statistic


def loop_oracle(control, treated, reference=0):
    """Explicit sums, unequal group sizes, and inverse quadratic forms."""
    k = control.shape[1]
    moments = []
    for y in (control, treated):
        n = len(y)
        z = [[float(v) - math.fsum(row) / k for v in row] for row in y]
        mu = [math.fsum(row[t] for row in z) / n for t in range(k)]
        x = [[row[t] - mu[t] for t in range(k)] for row in z]
        q = [math.fsum(row[t] ** 2 for row in x) / n for t in range(k)]
        cm = [[math.fsum(row[t] * row[s] for row in x) / (n * (n - 1))
               for s in range(k)] for t in range(k)]
        cv = [[(math.fsum(row[t] ** 2 * row[s] ** 2 for row in x)
                / (n * q[t] * q[s]) - 1) / n
               for s in range(k)] for t in range(k)]
        moments.append((mu, q, cm, cv, n))
    a, b = moments
    others = [t for t in range(k) if t != reference]
    dmu = [b[0][t] - a[0][t] for t in range(k)]
    rho = [math.log(b[1][t] * b[4] / (b[4] - 1))
           - math.log(a[1][t] * a[4] / (a[4] - 1)) for t in range(k)]
    hm = np.array([dmu[t] - dmu[reference] for t in others])
    hv = np.array([rho[t] - rho[reference] for t in others])
    covariances = []
    for index in (2, 3):
        covariances.append(np.array([
            [sum(g[index][t][s] - g[index][t][reference]
                 - g[index][reference][s] + g[index][reference][reference]
                 for g in moments) for s in others] for t in others]))
    vm, vv = covariances
    return dict(h_mean=hm, h_var=hv, cov_mean=vm, cov_var=vv,
                ratios=np.exp(rho), W_mean=float(hm @ np.linalg.inv(vm) @ hm),
                W_var=float(hv @ np.linalg.inv(vv) @ hv))


def example_panels():
    rng = np.random.default_rng(610122026)
    control = rng.normal(size=(53, 4)) @ np.array(
        [[1, .2, 0, 0], [0, 1, .3, 0], [0, 0, 1, .2], [0, 0, 0, 1]])
    treated = rng.normal(size=(61, 4)) * [1, 1.2, 1.4, 1.6] + np.arange(4) * .12
    return control, treated


class CovarianceChecks(unittest.TestCase):
    def test_public_covariance_utilities_match_explicit_oracle(self):
        y0, y1 = example_panels()
        oracle = loop_oracle(y0, y1)
        expected_z = np.array([[float(v)-math.fsum(row)/len(row) for v in row] for row in y0])
        assert_allclose(within_unit_demean(y0), expected_z, atol=3e-16)
        assert_allclose(contrast_matrix(4, 2),
                        [[1, 0, -1, 0], [0, 1, -1, 0], [0, 0, -1, 1]], atol=0)
        stat, pvalue, good = wald_statistic(oracle['h_var'], oracle['cov_var'])
        self.assertTrue(good)
        self.assertAlmostEqual(stat, oracle['W_var'], places=12)
        self.assertAlmostEqual(pvalue, float(_core.chi2_sf(stat, 3)), places=14)

    def test_public_wald_rejects_singular_and_malformed_inputs(self):
        stat, pvalue, good = wald_statistic([1., 0], [[1., 0], [0, 0]])
        self.assertFalse(good)
        self.assertTrue(math.isnan(stat) and math.isnan(pvalue))
        for h, v in [([], []), ([[1., 0]], np.eye(2)), ([1., 0], np.eye(3))]:
            with self.subTest(contrast=h), self.assertRaises(ValueError):
                wald_statistic(h, v)

    def test_explicit_cross_fourth_moment_oracle(self):
        y0, y1 = example_panels()
        observed = _core.analyze(y0, y1)
        expected = loop_oracle(y0, y1)
        for key in expected:
            with self.subTest(array=key):
                assert_allclose(observed[key], expected[key], rtol=2e-12, atol=2e-12)
        self.assertAlmostEqual(observed['W_var'], 14.373711223671723, places=11)

    def test_mean_covariance_is_unbiased_trajectory_covariance_over_n(self):
        y0, y1 = example_panels()
        # Raw contrasts eliminate each unit's demeaning constant exactly.
        d0 = y0[:, 1:] - y0[:, [0]]
        d1 = y1[:, 1:] - y1[:, [0]]
        expected = np.cov(d0, rowvar=False, ddof=1) / len(y0)
        expected += np.cov(d1, rowvar=False, ddof=1) / len(y1)
        assert_allclose(_core.analyze(y0, y1)['cov_mean'], expected, atol=2e-16)

    def test_variance_covariance_keeps_cross_period_products(self):
        y0, y1 = example_panels()
        result = _core.analyze(y0, y1)
        expected = loop_oracle(y0, y1)['cov_var']
        assert_allclose(result['cov_var'], expected, atol=2e-16)
        self.assertGreater(abs(expected[0, 1]), .001)
        # A diagonal contrast covariance would materially change this statistic.
        wrong = result['h_var'] @ np.linalg.inv(np.diag(np.diag(expected))) @ result['h_var']
        self.assertGreater(abs(wrong - result['W_var']), .1)

    def test_sample_divisor_adjustment_cancels_in_contrasts(self):
        y0, y1 = example_panels()
        q = []
        for y in (y0, y1):
            z = y - y.mean(axis=1, keepdims=True)
            q.append(np.var(z, axis=0, ddof=0))
        rho = np.log(q[1]) - np.log(q[0])
        result = _core.analyze(y0, y1)
        assert_allclose(result['h_var'], rho[1:] - rho[0], atol=5e-16)
        factor = (len(y1) / (len(y1) - 1)) / (len(y0) / (len(y0) - 1))
        assert_allclose(result['ratios'], q[1] / q[0] * factor, atol=5e-16)

    def test_transformed_target_exact_finite_support(self):
        omega = np.array([[1, .5, 0], [.5, 1, 0], [0, 0, 1.]])
        atoms = np.sqrt(3) * np.linalg.cholesky(omega).T
        atoms = np.concatenate((atoms, -atoms))
        m = np.eye(3) - np.ones((3, 3)) / 3
        assert_allclose(atoms.T @ atoms / 6, omega, atol=3e-16)
        z = atoms - atoms.mean(axis=1, keepdims=True)
        assert_allclose(z.T @ z / 6, m @ omega @ m, atol=3e-16)
        assert_allclose(np.diag(z.T @ z / 6) / (2 / 3), [2/3, 2/3, 7/6], atol=3e-16)

    def test_reference_coordinates_and_am_identity(self):
        y0, y1 = example_panels()
        base = _core.analyze(y0, y1, fisher_valid=True)
        m = np.eye(4) - np.ones((4, 4)) / 4
        for r in range(4):
            with self.subTest(reference=r):
                a = _core.contrast_matrix(4, r)
                assert_allclose(a @ m, a, atol=0)
                actual = _core.analyze(y0, y1, reference=r, fisher_valid=True)
                oracle = loop_oracle(y0, y1, reference=r)
                for key in oracle:
                    assert_allclose(actual[key], oracle[key], rtol=2e-12, atol=2e-12)
                for key in ['W_mean', 'W_var', 'p_fisher']:
                    assert_allclose(actual[key], base[key], rtol=2e-12, atol=2e-12)

    def test_equal_size_clusters_against_explicit_cluster_sum(self):
        rng = np.random.default_rng(431)
        y0, y1 = rng.normal(size=(30, 4)), rng.normal(size=(40, 4))
        expected = []
        for y, size in ((y0, 3), (y1, 4)):
            n, k = y.shape
            z = y - y.mean(axis=1, keepdims=True)
            x = z - z.mean(axis=0)
            q = (x*x).mean(axis=0)
            p = x*x/q - 1
            group_count = n // size
            matrices = []
            for scores in (x, p):
                out = np.zeros((k - 1, k - 1))
                for start in range(0, n, size):
                    block = [math.fsum(scores[i, t] for i in range(start, start+size))
                             for t in range(k)]
                    s = np.array([block[t]-block[0] for t in range(1, k)]) / n
                    out += np.outer(s, s)
                matrices.append(group_count / (group_count-1) * out)
            expected.append(matrices)
        actual = _core.analyze(y0, y1, cluster_size=(3, 4))
        assert_allclose(actual['cov_mean'], expected[0][0]+expected[1][0], atol=2e-16)
        assert_allclose(actual['cov_var'], expected[0][1]+expected[1][1], atol=2e-16)

    def test_shared_clusters_sum_signed_scores_before_products(self):
        rng = np.random.default_rng(882)
        g = 12
        y0, y1 = rng.normal(size=(g*3, 4)), rng.normal(size=(g*2, 4))
        c0, c1 = np.repeat(np.arange(g), 3), np.repeat(np.arange(g), 2)
        sm, sv = np.zeros((g, 3)), np.zeros((g, 3))
        for sign, y, labels in ((-1, y0, c0), (1, y1, c1)):
            z = y - y.mean(axis=1, keepdims=True)
            x = z-z.mean(axis=0)
            q = (x*x).mean(axis=0)
            for i, label in enumerate(labels):
                for t in range(1, 4):
                    sm[label, t-1] += sign*(x[i,t]-x[i,0])/len(y)
                    sv[label, t-1] += sign*(x[i,t]**2/q[t]-x[i,0]**2/q[0])/len(y)
        actual = _core.analyze_clustered(y0, y1, c0, c1)
        assert_allclose(actual['cov_mean'], g/(g-1)*sm.T@sm, atol=2e-16)
        assert_allclose(actual['cov_var'], g/(g-1)*sv.T@sv, atol=2e-16)
        separated = _core.analyze_clustered(y0, y1, c0, c1+g)
        self.assertGreater(np.max(abs(actual['cov_var']-separated['cov_var'])), .001)

    def test_two_cluster_conventions_have_declared_finite_corrections(self):
        rng = np.random.default_rng(73)
        y0, y1 = rng.normal(size=(30, 4)), rng.normal(size=(30, 4))
        grouped = _core.analyze(y0, y1, cluster_size=3)
        pooled = _core.analyze_clustered(y0, y1, np.repeat(np.arange(10), 3),
                                        np.repeat(np.arange(10, 20), 3))
        factor = (10/9)/(20/19)
        for key in ['cov_mean', 'cov_var']:
            assert_allclose(grouped[key], pooled[key]*factor, atol=2e-16)

    def test_gaussian_four_period_contrast_covariance_by_isserlis(self):
        # Correlation after demeaning iid Gaussian coordinates is -1/3.
        # Cov(X_t²/q_t-1,X_s²/q_s-1)=2 corr(X_t,X_s)².
        c = np.full((4, 4), 2/9)
        np.fill_diagonal(c, 2)
        a = np.eye(4)[1:]-np.eye(4)[0]
        assert_allclose(a@c@a.T, (8/9)*2*(np.eye(3)+np.ones((3, 3))), atol=0)


if __name__ == '__main__':
    unittest.main()
