"""Bounded replay against frozen paper outputs; never a full Monte Carlo run.

Default: compare with compact, unmodified saved-output prefixes in fixtures/.
Optional --reference-root: read those prefixes directly from a companion v3
directory and verify all selected source-file hashes first. No source outputs
are modified. This reproduces the original 102 replay comparisons: first 100
panels in three analytic cases, first outer panel and 199 inner draws in eight
bootstrap cases. It is a same-core reproducibility check, not independent
mathematical validation, recovery of historical analysis, or a coverage study.

The small generator below is a transcription of the frozen runner's draw order.
Independent moment/scaling/cluster calculations live in tests/test_covariance.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np

# Supports direct execution from an unpacked source tree or an installed package.
try:
    from didcheck import _core
except ModuleNotFoundError as exc:
    if exc.name != 'didcheck':
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from didcheck import _core


HERE = Path(__file__).resolve().parent
ANALYTIC_KEYS = ('W_mean', 'W_var', 'p_mean', 'p_var', 'p_fisher',
                 'ratios', 'h_mean', 'h_var', 'cov_mean', 'cov_var')
BOOTSTRAP_KEYS = ('W_mean', 'W_var', 'W_sum', 'p_boot_mean', 'p_boot_var',
                  'p_boot_sum', 'inner_W_mean_centered', 'inner_W_var_centered',
                  'inner_W_sum_centered')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(rng, count, case):
    """Exact frozen draw order, including one common time path for both groups."""
    n, k = case['N'], 4
    times = np.arange(1, k+1)
    tau = rng.normal(0, .3, (count, 1, k))
    panels = []
    phi, size = case['phi'], case['cluster_size']
    for group in (0, 1):
        if case['law'] == 'normal':
            error = rng.standard_normal((count, n, k))
        elif case['law'] == 'exponential':
            error = rng.exponential(1, (count, n, k))-1
        else:
            raise ValueError('The bounded fixture contains only normal/exponential cases')
        if phi:
            transformed = error.copy()
            for t in range(1, k):
                transformed[..., t] = phi*transformed[..., t-1]+np.sqrt(1-phi**2)*error[..., t]
            error = transformed
        if size > 1:
            cluster = rng.standard_normal((count, n//size, k))
            transformed = cluster.copy()
            for t in range(1, k):
                transformed[..., t] = phi*transformed[..., t-1]+np.sqrt(1-phi**2)*cluster[..., t]
            rho = case['cluster_rho']
            error = np.sqrt(rho)*np.repeat(transformed, size, axis=1)+np.sqrt(1-rho)*error
        if group:
            error = error*np.sqrt(1+case['variance_delta']*times)[None, None, :]
        y = error+rng.normal(0, .5, (count, n, 1))+tau
        if group:
            y = y+case['mean_delta']*times[None, None, :]
        panels.append(y)
    return panels


def fit(control, treated, case):
    kw = dict(reference=0, fisher_valid=case['fisher_valid'])
    if case['cluster_size'] > 1:
        kw['cluster_size'] = case['cluster_size']
    return _core.analyze_batch(control, treated, **kw)


def bootstrap_prefix(rng, case):
    """Only the first outer panel; retain all 199 centered inner statistics."""
    y0, y1 = generate(rng, 1, case)
    original = {key: value[0] for key, value in fit(y0, y1, case).items()}
    n, size, inner = case['N'], case['cluster_size'], case['inner_repetitions']
    groups = n//size
    i0 = rng.integers(0, groups, (inner, groups))
    i1 = rng.integers(0, groups, (inner, groups))
    if size > 1:
        i0 = (i0[..., None]*size+np.arange(size)).reshape(inner, n)
        i1 = (i1[..., None]*size+np.arange(size)).reshape(inner, n)
    star = fit(y0[0, i0], y1[0, i1], case)
    total, good_total = _core.centered_sum_batch(star, original)
    wm, good_m = _core.wald_batch(star['h_mean']-original['h_mean'], star['cov_mean'])
    wv, good_v = _core.wald_batch(star['h_var']-original['h_var'], star['cov_var'])
    good_m &= star['valid_mean']
    good_v &= star['valid_var']
    result = {key: original[key] for key in ('W_mean', 'W_var', 'W_sum')}
    for name, w, valid, obs in [('mean', wm, good_m, original['W_mean']),
                               ('var', wv, good_v, original['W_var']),
                               ('sum', total, good_total, original['W_sum'])]:
        ok = np.isfinite(obs) and valid.all()
        result['p_boot_'+name] = (1+np.count_nonzero(w >= obs))/(inner+1) if ok else np.nan
    result.update(inner_W_mean_centered=wm, inner_W_var_centered=wv,
                  inner_W_sum_centered=total)
    return result


def compare(actual, expected, case, array, rtol=1e-10, atol=1e-12):
    a, b = np.asarray(actual), np.asarray(expected)
    shape_match = a.shape == b.shape
    exact = bool(shape_match and np.array_equal(a, b, equal_nan=True))
    # Tail-count p-values are discrete and must agree exactly, not just closely.
    close = exact if array.startswith('p_boot_') else bool(
        shape_match and np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))
    delta = None
    if shape_match:
        mask = np.isfinite(a) & np.isfinite(b)
        delta = float(np.max(np.abs(a[mask]-b[mask]))) if np.any(mask) else 0.0
    return dict(case=case, array=array, shape_match=shape_match,
                exact_match=exact, tolerance_match=close, max_absolute_error=delta)


def run(reference_root=None, strict_bitwise=False):
    provenance_path = HERE/'REFERENCE_PROVENANCE.json'
    provenance = json.loads(provenance_path.read_text())
    fixture_path = HERE/provenance['fixture']['file']
    if sha256(fixture_path) != provenance['fixture']['sha256']:
        raise ValueError('Compact reference fixture hash mismatch')
    root = Path(reference_root) if reference_root is not None else None
    if root is not None:
        config_path = root/'records'/'frozen_simulation_config.json'
        if sha256(config_path) != provenance['configuration']['sha256']:
            raise ValueError('External frozen configuration does not match the selected reference')
        for item in provenance['cases']:
            if sha256(root/item['source_file']) != item['source_sha256']:
                raise ValueError('External saved output hash mismatch: '+item['source_file'])
    checks = []
    with np.load(fixture_path, allow_pickle=False) as fixture:
        for item in provenance['cases']:
            case = item['configuration']
            stage = 2 if case['family'] == 'bootstrap' else 1
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
                [provenance['base_seed'], stage, case['index']])))
            if stage == 1:
                actual = fit(*generate(rng, 100, case), case)
                keys = ANALYTIC_KEYS
            else:
                actual = bootstrap_prefix(rng, case)
                keys = BOOTSTRAP_KEYS
            external = np.load(root/item['source_file'], allow_pickle=False) if root else None
            try:
                for key in keys:
                    expected = (external[key][:100] if stage == 1 else external[key][0]) if external is not None else fixture[case['id']+'__'+key]
                    checks.append(compare(actual[key], expected, case['id'], key))
            finally:
                if external is not None:
                    external.close()
    exact_count = sum(row['exact_match'] for row in checks)
    tolerance_count = sum(row['tolerance_match'] for row in checks)
    core_hash = sha256(_core.__file__)
    core_matches = core_hash == provenance['core_sha256']
    passed = len(checks) == 102 and core_matches and tolerance_count == 102
    if strict_bitwise:
        passed = passed and exact_count == 102
    return dict(
        status='PASS' if passed else 'FAIL', check_count=len(checks),
        exact_match_count=exact_count, tolerance_match_count=tolerance_count,
        strict_bitwise=strict_bitwise, rtol=1e-10, atol=1e-12,
        source='external verified companion outputs' if root else 'bundled compact saved-output prefixes',
        scope=provenance['scope'],
        interpretation='Bounded same-core reproducibility, not independent inference validation or a full study rerun.',
        core_sha256=core_hash, core_matches_reference=core_matches,
        fixture_sha256=provenance['fixture']['sha256'],
        provenance_sha256=sha256(provenance_path),
        runtime=dict(python=platform.python_version(), numpy=np.__version__, system=platform.system()),
        checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, help='Optional external v3 companion root; read-only')
    parser.add_argument('--output', type=Path, help='Write JSON summary to this path, otherwise print it')
    parser.add_argument('--strict-bitwise', action='store_true', help='Require bit equality in addition to portable tolerances')
    args = parser.parse_args()
    result = run(args.reference_root, args.strict_bitwise)
    rendered = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.write_text(rendered)
        print(f"{result['status']}: {result['tolerance_match_count']}/102 within tolerance; "
              f"{result['exact_match_count']}/102 bit-exact")
    else:
        print(rendered, end='')
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
