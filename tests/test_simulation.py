"""Generator identities and small Monte Carlo accounting checks."""
import numpy as np
import pandas as pd
import pytest

from didcheck import _core
from didcheck.simulation import simulate_panel, monte_carlo, _generate_original, standard_errors, ar_transform


def test_original_generator_matches_explicit_draw_order():
    rng = np.random.default_rng(123)
    tau = rng.normal(0, .3, (2, 1, 4))
    expected = []
    for g in [0, 1]:
        errors = rng.standard_normal((2, 8, 4))
        if g:
            errors *= np.sqrt(1 + .08*np.arange(1, 5))[None, None, :]
        panel = errors + rng.normal(0, .5, (2, 8, 1)) + tau
        if g:
            panel += .15*np.arange(1, 5)[None, None, :]
        expected.append(panel)
    actual = _generate_original(np.random.default_rng(123), 2, 8, mean_delta=.15, variance_delta=.08)
    for want, got in zip(expected, actual):
        np.testing.assert_array_equal(want, got)


def test_long_panel_is_exact_original_k4_draw_and_cluster_ids_are_distinct():
    df = simulate_panel(12, 12, mean_delta=.1, variance_delta=.08, cluster_size=3,
                        cluster_rho=.4, phi=.5, random_state=33)
    a, b = _generate_original(np.random.default_rng(33), 1, 12, mean_delta=.1, variance_delta=.08,
                              cluster_size=3, cluster_rho=.4, phi=.5)
    assert list(df.columns) == ['unit', 'time', 'treated', 'outcome', 'cluster']
    np.testing.assert_array_equal(df.outcome.to_numpy(), np.r_[a.ravel(), b.ravel()])
    assert set(df.query('treated == 0').cluster).isdisjoint(set(df.query('treated == 1').cluster))
    assert df.groupby('cluster').unit.nunique().eq(3).all()
    assert not df.duplicated(['unit', 'time']).any()


def test_unequal_groups_and_periods_are_complete_and_replayable():
    a = simulate_panel(12, 18, 5, cluster_size=3, cluster_rho=.2, random_state=9)
    b = simulate_panel(12, 18, 5, cluster_size=3, cluster_rho=.2, random_state=9)
    pd.testing.assert_frame_equal(a, b)
    assert a.shape == (150, 5)
    assert a.groupby('unit').time.nunique().eq(5).all()
    assert a.groupby('treated').unit.nunique().tolist() == [12, 18]


def test_mean_departure_is_deterministic_and_leaves_control_unchanged():
    null = simulate_panel(20, 22, 5, random_state=88)
    alt = simulate_panel(20, 22, 5, mean_delta=.15, random_state=88)
    difference = alt.outcome.to_numpy() - null.outcome.to_numpy()
    np.testing.assert_allclose(difference, .15*null.time*null.treated, atol=1e-15)


@pytest.mark.parametrize('law', ['normal', 'uniform', 't5', 'laplace', 'exponential'])
def test_innovation_laws_use_exact_documented_standardization(law):
    rng, check = np.random.default_rng(5), np.random.default_rng(5)
    expected = {'normal': lambda: check.standard_normal((3, 4)),
                'uniform': lambda: check.uniform(-np.sqrt(3), np.sqrt(3), (3, 4)),
                't5': lambda: check.standard_t(5, (3, 4))*np.sqrt(3/5),
                'laplace': lambda: check.laplace(0, 1/np.sqrt(2), (3, 4)),
                'exponential': lambda: check.exponential(1, (3, 4))-1}[law]()
    np.testing.assert_array_equal(standard_errors(rng, (3, 4), law), expected)


def test_ar_transform_uses_stationary_first_coordinate():
    innovations = np.array([[1., 2., 3.]])
    transformed = ar_transform(innovations, .5)
    expected = [1., .5 + np.sqrt(.75)*2]
    expected.append(.5*expected[1] + np.sqrt(.75)*3)
    np.testing.assert_allclose(transformed[0], expected)
    np.testing.assert_array_equal(innovations, [[1., 2., 3.]])


def test_monte_carlo_raw_and_summary_match_small_original_batch():
    out = monte_carlo(repetitions=7, n_control=30, n_treated=30, random_state=12,
                      fisher_valid=True, batch_size=7)
    a, b = _generate_original(np.random.default_rng(12), 7, 30)
    expected = _core.analyze_batch(a, b, fisher_valid=True)
    for key in ('W_mean', 'W_var', 'W_sum', 'p_mean', 'p_var', 'p_fisher'):
        np.testing.assert_array_equal(out['raw'][key], expected[key])
    for row in out['summary'].to_dict('records'):
        suffix = {'variance': 'var'}.get(row['test'], row['test'])
        count = np.count_nonzero(out['raw']['valid_'+suffix] & (out['raw']['p_'+suffix] <= .05))
        assert row['rejections'] == count
        assert row['scheduled_repetitions'] == 7
        assert row['valid_repetitions'] + row['failed_repetitions'] == 7
        assert row['rejection_rate'] == count/7
        assert row['wilson95_low'] <= row['rejection_rate'] <= row['wilson95_high']


def test_monte_carlo_disables_fisher_and_retains_failures():
    out = monte_carlo(repetitions=5, n_control=12, n_treated=13, n_periods=2, batch_size=2)
    rows = out['summary'].set_index('test')
    assert rows.loc['variance', 'failed_repetitions'] == 5
    assert rows.loc['variance', 'valid_repetitions'] == 0
    assert np.isnan(out['raw']['p_var']).all()
    assert rows.loc['fisher', 'disabled_repetitions'] == 5
    assert rows.loc['fisher', 'failed_repetitions'] == 0
    assert np.isnan(rows.loc['fisher', 'rejection_rate'])
    assert len(out['raw']['W_var']) == 5


def test_clustered_monte_carlo_uses_declared_cluster_covariance():
    out = monte_carlo(repetitions=3, n_control=30, n_treated=30, cluster_size=3,
                      cluster_rho=.4, phi=.5, random_state=11)
    a, b = _generate_original(np.random.default_rng(11), 3, 30, cluster_size=3, cluster_rho=.4, phi=.5)
    expected = _core.analyze_batch(a, b, cluster_size=3)
    np.testing.assert_array_equal(out['raw']['W_var'], expected['W_var'])


@pytest.mark.parametrize('kwargs', [dict(n_control=True), dict(n_periods=1), dict(n_treated=2.5),
    dict(variance_delta=-.25), dict(phi=1), dict(phi=np.nan), dict(law='unknown'),
    dict(law='t5', phi=.5), dict(cluster_size=3), dict(cluster_rho=-.1),
    dict(cluster_rho=.2), dict(cluster_size=2, law='uniform'), dict(random_state=-1),
    dict(mean_delta=float('inf')), dict(mean_delta=1e308), dict(variance_delta=1e308)])
def test_invalid_generator_parameters(kwargs):
    with pytest.raises(ValueError):
        simulate_panel(**kwargs)


@pytest.mark.parametrize('kwargs', [dict(repetitions=0), dict(batch_size=True), dict(alpha=0),
    dict(alpha=1), dict(alpha='invalid'), dict(reference=5), dict(fisher_valid='yes'), dict(fisher_valid=True, law='exponential')])
def test_invalid_monte_carlo_parameters(kwargs):
    with pytest.raises(ValueError):
        monte_carlo(**kwargs)


def test_monte_carlo_replay_with_partial_final_batch():
    settings = dict(repetitions=11, n_control=18, n_treated=22, n_periods=3,
                    random_state=51, batch_size=4, law='exponential')
    one, two = monte_carlo(**settings), monte_carlo(**settings)
    pd.testing.assert_frame_equal(one['summary'], two['summary'])
    for key in one['raw']:
        np.testing.assert_array_equal(one['raw'][key], two['raw'][key])
    assert len(one['raw']['p_mean']) == 11
