"""Small independent calculations, not a repetition of the published study."""
import numpy as np
import pytest

from didcheck import _core
from didcheck.bootstrap import centered_bootstrap, _resample_shared_clusters, _tail


def _sample(n0=40, n1=45, k=4):
    rng = np.random.default_rng(511)
    return rng.normal(size=(n0, k)), rng.normal(size=(n1, k)) + .12*np.arange(k)


def _manual_roots_covariance(a, b, reference=0):
    """Independent per-unit fourth-moment loops; no core estimator calls."""
    k = a.shape[1]
    contrast = np.eye(k)[[t for t in range(k) if t != reference]] - np.eye(k)[reference]
    group = []
    for y in (a, b):
        z = np.array([row - sum(row)/k for row in y])
        mu = np.mean(z, axis=0)
        x = z - mu
        q = np.mean(x*x, axis=0)
        psi = np.array([row*row/q - 1 for row in x])
        cm = sum(np.outer(row, row) for row in x) / (len(y)*(len(y)-1))
        cv = sum(np.outer(row, row) for row in psi) / len(y)**2
        group.append((mu, q*len(y)/(len(y)-1), cm, cv))
    g0, g1 = group
    return (contrast@(g1[0]-g0[0]), contrast@(np.log(g1[1])-np.log(g0[1])),
            contrast@(g0[2]+g1[2])@contrast.T, contrast@(g0[3]+g1[3])@contrast.T)


def test_unit_draws_center_both_roots_and_use_the_same_resample():
    a, b = _sample()
    fit = centered_bootstrap(a, b, repetitions=9, random_state=24, batch_size=9, reference=2)
    h0m, h0v, _, _ = _manual_roots_covariance(a, b, 2)
    rng = np.random.default_rng(24)
    i0 = rng.integers(len(a), size=(9, len(a)))
    i1 = rng.integers(len(b), size=(9, len(b)))
    manual_m, manual_v = [], []
    for ia, ib in zip(i0, i1):
        hm, hv, cm, cv = _manual_roots_covariance(a[ia], b[ib], 2)
        dm, dv = hm-h0m, hv-h0v
        manual_m.append(dm @ np.linalg.solve(cm, dm))
        manual_v.append(dv @ np.linalg.solve(cv, dv))
    np.testing.assert_allclose(fit['mean']['statistics'], manual_m, rtol=2e-12)
    np.testing.assert_allclose(fit['variance']['statistics'], manual_v, rtol=2e-12)
    np.testing.assert_allclose(fit['sum']['statistics'], np.array(manual_m)+manual_v, rtol=2e-12)
    for name in ('mean', 'variance', 'sum'):
        block = fit[name]
        assert block['valid'] and block['failed_draws'] == 0
        assert block['pvalue'] == (1 + sum(block['statistics'] >= block['statistic'])) / 10


def test_plus_one_ties_and_failed_draws():
    r = _tail(2., np.array([2., 1., 3.]), np.ones(3, dtype=bool))
    assert r['exceedances'] == 2 and r['pvalue'] == .75
    failed = _tail(2., np.array([2., np.nan, 3.]), np.array([True, False, True]))
    assert failed['exceedances'] == 2 and failed['failed_draws'] == 1
    assert not failed['valid'] and np.isnan(failed['pvalue'])
    assert len(failed['statistics']) == 3


def test_exact_replay_includes_batch_size():
    a, b = _sample()
    x = centered_bootstrap(a, b, repetitions=13, random_state=7, batch_size=5)
    y = centered_bootstrap(a, b, repetitions=13, random_state=7, batch_size=5)
    for name in ('mean', 'variance', 'sum'):
        np.testing.assert_array_equal(x[name]['statistics'], y[name]['statistics'])
    assert x['batch_size'] == 5 and x['repetitions'] == 13


def test_stratified_clusters_with_different_sizes():
    a, b = _sample(36, 40)
    output = centered_bootstrap(a, b, repetitions=7, random_state=19, batch_size=7, cluster_size=(3, 2))
    rng = np.random.default_rng(19)
    i0 = (rng.integers(12, size=(7, 12))[..., None]*3 + np.arange(3)).reshape(7, 36)
    i1 = (rng.integers(20, size=(7, 20))[..., None]*2 + np.arange(2)).reshape(7, 40)
    observed = _core.analyze(a, b, cluster_size=(3, 2))
    star = _core.analyze_batch(a[i0], b[i1], cluster_size=(3, 2))
    expected, flags = _core.centered_sum_batch(star, observed)
    np.testing.assert_array_equal(output['sum']['statistics'], expected)
    np.testing.assert_array_equal(output['sum']['valid_draws'], flags)


def test_shared_cluster_copies_are_fresh_and_groups_stay_together():
    a, b = _sample(6, 5)
    c0, c1 = ["a", "a", "b", "b", "c", "c"], ["a", "b", "b", "c", "c"]
    aa, bb, ca, cb = _resample_shared_clusters(a, b, c0, c1, np.random.default_rng(0), selected=['b', 'b', 'a'])
    np.testing.assert_array_equal(aa, a[[2, 3, 2, 3, 0, 1]])
    np.testing.assert_array_equal(bb, b[[1, 2, 1, 2, 0]])
    np.testing.assert_array_equal(ca, [0, 0, 1, 1, 2, 2])
    np.testing.assert_array_equal(cb, [0, 0, 1, 1, 2])


def test_shared_bootstrap_recomputes_denominators_and_covariances():
    a, b = _sample(24, 30)
    c0, c1 = np.repeat(np.arange(12), 2), np.repeat(np.arange(10), 3)
    out = centered_bootstrap(a, b, clusters=(c0, c1), repetitions=6, random_state=31)
    observed = _core.analyze_clustered(a, b, c0, c1)
    rng = np.random.default_rng(31)
    expected = []
    for _ in range(6):
        aa, bb, ca, cb = _resample_shared_clusters(a, b, c0, c1, rng)
        fit = _core.analyze_clustered(aa, bb, ca, cb)
        dm, dv = fit['h_mean']-observed['h_mean'], fit['h_var']-observed['h_var']
        expected.append(dm @ np.linalg.solve(fit['cov_mean'], dm) + dv @ np.linalg.solve(fit['cov_var'], dv))
    np.testing.assert_allclose(out['sum']['statistics'], expected, rtol=2e-12)
    assert out['sum']['failed_draws'] == 0


def test_two_period_variance_failure_is_retained_but_mean_can_be_available():
    a, b = _sample(40, 45, 2)
    fit = centered_bootstrap(a, b, repetitions=11)
    assert fit['mean']['valid']
    for name in ('variance', 'sum'):
        assert not fit[name]['valid'] and np.isnan(fit[name]['pvalue'])
        assert fit[name]['failed_draws'] == 11
        assert len(fit[name]['statistics']) == 11


def test_pooled_draw_losing_a_group_is_recorded_without_retry():
    a, b = _sample(12, 12)
    out = centered_bootstrap(a, b, repetitions=20, random_state=0,
                             clusters=(np.zeros(12, dtype=int), np.ones(12, dtype=int)))
    # Original G=2 covariance is rank deficient; pooled draws also sometimes
    # select only one group. All scheduled draws still have their own slot.
    assert len(out['sum']['statistics']) == 20
    assert np.isnan(out['sum']['pvalue'])
    assert out['sum']['failed_draws'] > 0


@pytest.mark.parametrize('kwargs', [dict(repetitions=0), dict(repetitions=True), dict(batch_size=1.5),
    dict(random_state=-1), dict(reference=True), dict(reference=4), dict(cluster_size=7),
    dict(cluster_size=(3,)), dict(cluster_size=True), dict(cluster_size=40),
    dict(cluster_size=2, clusters=([0]*40, [1]*45)), dict(clusters=([None]*40, [1]*45))])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        centered_bootstrap(*_sample(), **kwargs)


def test_nonfinite_observed_input_returns_unavailable_not_nonrejection():
    a, b = _sample()
    a[0, 0] = np.nan
    result = centered_bootstrap(a, b, repetitions=5, random_state=10)
    for name in ('mean', 'variance', 'sum'):
        assert not result[name]['valid']
        assert np.isnan(result[name]['pvalue'])
        assert np.isnan(result[name]['statistic'])
        assert len(result[name]['statistics']) == 5
