"""Small public-API integration checks; no published research reruns."""
import warnings

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from didcheck import diagnose_arrays, variance_test
from didcheck.bootstrap import centered_bootstrap
from didcheck.plotting import plot_bootstrap, plot_variance_ratios


def _data():
    rng = np.random.default_rng(207)
    a, b = rng.normal(size=(24, 4)), rng.normal(size=(24, 4))*[1., 1.1, 1.2, 1.3]
    rows = []
    for group, values in enumerate((a, b)):
        for unit, trajectory in enumerate(values):
            for period, outcome in zip(["2018Q4", "2019Q1", "2019Q2", "2019Q3"], trajectory):
                rows.append(dict(unit=f'{group}:{unit:03}', time=period, treated=group,
                                 outcome=outcome, cluster=f'cluster-{unit//2:02}'))
    return a, b, pd.DataFrame(rows)


def _table_fit(frame, **kwargs):
    params = dict(outcome='outcome', unit='unit', time='time', treatment='treated',
                  pre_periods=['2019Q3', '2019Q1', '2018Q4', '2019Q2'], reference='2018Q4')
    params.update(kwargs)
    return variance_test(frame, **params)


def _assert_matching_results(a, b):
    # DataFrame pivots and sliced arrays can have different memory layouts;
    # equivalent summation orders may differ by ordinary float roundoff.
    np.testing.assert_allclose(a.mean_contrast, b.mean_contrast, rtol=2e-12, atol=2e-14)
    np.testing.assert_allclose(a.variance_covariance, b.variance_covariance, rtol=2e-12, atol=2e-14)
    for name in ('mean', 'variance', 'sum'):
        np.testing.assert_allclose(a.bootstrap[name]['statistics'], b.bootstrap[name]['statistics'],
                                   rtol=2e-12, atol=2e-14)
        assert a.bootstrap[name]['pvalue'] == b.bootstrap[name]['pvalue']


def test_dataframe_period_order_and_shuffling_replay_array_bootstrap():
    a, b, df = _data()
    params = dict(bootstrap=9, random_state=21, batch_size=4)
    ordered = _table_fit(df, **params)
    shuffled = _table_fit(df.sample(frac=1, random_state=6), **params)
    arrays = diagnose_arrays(a[:, [3, 1, 0, 2]], b[:, [3, 1, 0, 2]], reference=2, **params)
    _assert_matching_results(ordered, shuffled)
    for name in ('mean', 'variance', 'sum'):
        np.testing.assert_array_equal(ordered.bootstrap[name]['statistics'],
                                      shuffled.bootstrap[name]['statistics'])
    _assert_matching_results(ordered, arrays)
    assert ordered.periods == ('2019Q3', '2019Q1', '2018Q4', '2019Q2')
    assert ordered.reference == '2018Q4'
    assert ordered.metadata['control_units'] == tuple(f'0:{i:03}' for i in range(24))


def test_shared_cluster_public_bootstrap_matches_array_interface():
    a, b, df = _data()
    labels = np.repeat([f'cluster-{i:02}' for i in range(12)], 2)
    params = dict(bootstrap=8, random_state=42, batch_size=3)
    tabular = _table_fit(df.sample(frac=1, random_state=31), cluster='cluster', cluster_design='shared', **params)
    direct = diagnose_arrays(a[:, [3, 1, 0, 2]], b[:, [3, 1, 0, 2]],
                             reference=2, clusters=(labels, labels), **params)
    _assert_matching_results(tabular, direct)
    assert 'shared' in tabular.metadata['sampling']
    assert any('cluster regularity' in message for message in tabular.warnings)
    assert tabular.bootstrap['repetitions'] == 8
    assert set(tabular.to_frame().index) == {'mean', 'variance', 'fisher', 'combined', 'bootstrap_mean', 'bootstrap_variance'}


def test_stratified_dataframe_clusters_are_reordered_as_whole_blocks():
    a, b, df = _data()
    # Randomly attach units to equal-size blocks: row-order resampling would be
    # wrong unless preparation first rebuilds contiguous blocks for each group.
    perm = np.random.default_rng(17).permutation(24)
    mapping = {f'{g}:{i:03}': f'{g}:{perm[i]//2:02}' for g in (0, 1) for i in range(24)}
    df['cluster'] = df.unit.map(mapping)
    out = _table_fit(df.sample(frac=1, random_state=1), cluster='cluster',
                     cluster_design='stratified_equal_size', bootstrap=7, random_state=9)
    ids0, ids1 = out.metadata['control_units'], out.metadata['treated_units']
    ia = [int(x.split(':')[1]) for x in ids0]
    ib = [int(x.split(':')[1]) for x in ids1]
    direct = diagnose_arrays(a[ia][:, [3, 1, 0, 2]], b[ib][:, [3, 1, 0, 2]], reference=2,
                             cluster_size=(2, 2), bootstrap=7, random_state=9)
    _assert_matching_results(out, direct)
    for ids in (ids0, ids1):
        assert all(mapping[ids[j]] == mapping[ids[j+1]] for j in range(0, 24, 2))


def test_missing_and_drop_preserve_whole_units_and_warnings():
    a, b, df = _data()
    missing = df.drop(index=0)
    with pytest.raises(ValueError, match='incomplete'):
        _table_fit(missing)
    dropped = _table_fit(missing, missing='drop_units', bootstrap=6, random_state=9)
    expected = diagnose_arrays(a[1:, [3, 1, 0, 2]], b[:, [3, 1, 0, 2]], reference=2,
                               bootstrap=6, random_state=9)
    _assert_matching_results(dropped, expected)
    assert dropped.metadata['dropped_units'] == ('0:000',)
    assert dropped.n_control == 23
    assert any('removed 1 units' in message for message in dropped.warnings)
    assert 'Nonrejection does not establish' in dropped.summary()


def test_drop_that_breaks_equal_size_cluster_design_is_rejected():
    _, _, df = _data()
    df['cluster'] = df.treated.astype(str) + ':' + df.cluster
    with pytest.raises(ValueError, match='equal-size'):
        _table_fit(df.drop(index=0), missing='drop_units', cluster='cluster', cluster_design='stratified_equal_size')


@pytest.mark.parametrize('alpha', [None, 'bad', [.05], np.array([.05]), complex(.05, 0)])
def test_invalid_alpha_datatypes_raise_value_error(alpha):
    a, b, _ = _data()
    with pytest.raises(ValueError, match='alpha'):
        diagnose_arrays(a, b, alpha=alpha)


@pytest.mark.parametrize('reference', [True, .5, 'first', None, np.array([0])])
def test_invalid_array_reference_datatypes_raise_value_error(reference):
    a, b, _ = _data()
    with pytest.raises(ValueError, match='reference'):
        diagnose_arrays(a, b, reference=reference)


@pytest.mark.parametrize('bad', [None, np.nan, pd.NA, ['unhashable']])
def test_shared_cluster_missing_or_unhashable_labels_raise_value_error(bad):
    a, b, _ = _data()
    c0 = list(range(24))
    c0[0] = bad
    with pytest.raises(ValueError):
        diagnose_arrays(a, b, clusters=(c0, list(range(24))), bootstrap=2)


@pytest.mark.parametrize('bad_clusters', [4, object()])
def test_cluster_container_type_checked(bad_clusters):
    a, b, _ = _data()
    with pytest.raises(ValueError, match='clusters'):
        diagnose_arrays(a, b, clusters=bad_clusters)


def test_bootstrap_rejects_scalar_strings_as_label_sequences():
    a, b, _ = _data()
    with pytest.raises(ValueError, match='labels'):
        centered_bootstrap(a, b, clusters=('a'*24, 'b'*24), repetitions=2)


def test_complex_outcomes_rejected_by_array_and_bootstrap_interfaces():
    a, b, _ = _data()
    for fn in (diagnose_arrays, centered_bootstrap):
        with pytest.raises(ValueError, match='real'):
            fn(a + 1j, b)
        with pytest.raises(ValueError, match='real'):
            fn(a, b + 1j)


def test_complex_dataframe_outcomes_rejected_without_discarding_imaginary_part():
    _, _, df = _data()
    df['outcome'] = df.outcome + 1j
    with pytest.raises(ValueError, match='real'):
        _table_fit(df)


def test_figures_keep_period_labels_and_report_scheduled_failure_counts():
    a, b, df = _data()
    result = _table_fit(df, bootstrap=9, random_state=1)
    ax = plot_variance_ratios(result)
    assert [t.get_text() for t in ax.get_xticklabels()] == list(result.periods)
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), result.variance_ratios)
    ax.figure.canvas.draw()
    plt.close(ax.figure)
    for component in ('mean', 'variance', 'sum'):
        ax = plot_bootstrap(result, component)
        assert '9 scheduled' in ax.get_title()
        assert f"{result.bootstrap[component]['failed_draws']} invalid" in ax.get_title()
        ax.figure.canvas.draw()
        plt.close(ax.figure)
    with pytest.raises(ValueError, match='No bootstrap'):
        plot_bootstrap(diagnose_arrays(a, b))
    with pytest.raises(ValueError, match='component'):
        plot_bootstrap(result, 'fisher')


def test_all_failed_bootstrap_plot_does_not_create_finite_histogram_or_false_rejection():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        result = diagnose_arrays(np.zeros((12, 4)), np.zeros((13, 4)), bootstrap=5)
    assert result.combined.reject is None
    assert not result.combined.valid and np.isnan(result.combined.pvalue)
    frame = result.to_frame()
    assert not frame['valid'].any()
    assert frame['reject'].isna().all()
    assert any('5 of 5 required draws failed' in message for message in result.warnings)
    ax = plot_bootstrap(result)
    assert '5 invalid' in ax.get_title() and not ax.patches
    ax.figure.canvas.draw()
    plt.close(ax.figure)
    ax = plot_variance_ratios(result)
    assert not np.isfinite(ax.lines[0].get_ydata()).any()
    ax.figure.canvas.draw()
    plt.close(ax.figure)
