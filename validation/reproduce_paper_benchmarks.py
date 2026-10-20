"""Reproduce three complete prespecified analytic benchmarks (15,000 panels).

Uses the installed package's four-period generator and frozen numerical core,
with the original PCG64 SeedSequence streams and 100-panel batching. Compares
every statistic, p-value, numerical-validity flag and rejection count with
compact frozen outputs. No empirical data or historical analysis is recovered.

This is PARTIAL_STUDY_REPRODUCTION: three of 57 analytic scenarios, without
the remaining analytic scenarios, bootstrap calibration study or post-principal
check. It establishes package reproduction of these saved numerical findings,
not independent mathematical validation or general finite-sample calibration.

Usage (from the installed package's source distribution):
    python validation/reproduce_paper_benchmarks.py --output validation/paper_benchmarks.json
    python validation/reproduce_paper_benchmarks.py --strict-bitwise --output summary.json
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import didcheck
from didcheck import _core, simulation


HERE = Path(__file__).resolve().parent
CASE_IDS = ('primary_null_N50', 'primary_null_N2000', 'primary_variance_only_N2000')
FLOAT_KEYS = ('W_mean', 'W_var', 'p_mean', 'p_var', 'p_fisher')
FLAG_KEYS = ('valid_mean', 'valid_var', 'valid_fisher')
GENERATOR_KEYS = ('N', 'mean_delta', 'variance_delta', 'law', 'phi', 'cluster_size', 'cluster_rho')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(actual, expected, case, key):
    same_shape = actual.shape == expected.shape
    exact = bool(same_shape and np.array_equal(actual, expected, equal_nan=True))
    close = exact if key in FLAG_KEYS else bool(
        same_shape and np.allclose(actual, expected, rtol=1e-10, atol=1e-12, equal_nan=True))
    error = None
    if same_shape:
        finite = np.isfinite(actual) & np.isfinite(expected)
        error = (float(np.max(np.abs(actual[finite].astype(float)-expected[finite].astype(float))))
                 if np.any(finite) else 0.0)
    return dict(case=case, array=key, shape_match=same_shape,
                exact_match=exact, tolerance_match=close, max_absolute_error=error)


def reproduce(strict_bitwise=False):
    started = time.perf_counter()
    provenance_path = HERE/'PAPER_BENCHMARKS_PROVENANCE.json'
    provenance = json.loads(provenance_path.read_text())
    fixture_path = HERE/provenance['fixture']['file']
    if sha256(fixture_path) != provenance['fixture']['sha256']:
        raise ValueError('Frozen benchmark fixture hash mismatch')
    core_hash = sha256(_core.__file__)
    if core_hash != provenance['core_sha256']:
        raise ValueError('Installed numerical core differs from the frozen reference')
    if tuple(item['configuration']['id'] for item in provenance['cases']) != CASE_IDS:
        raise ValueError('This script is limited to the three named benchmark scenarios')
    if provenance['batch_size'] != 100 or provenance['base_seed'] != 2026101207:
        raise ValueError('Reproduction requires the original batch size and seed')

    checks, case_reports = [], []
    with np.load(fixture_path, allow_pickle=False) as fixture:
        for item in provenance['cases']:
            case = item['configuration']
            count = case['repetitions']
            if count != 5000 or case['law'] != 'normal' or case['cluster_size'] != 1 or case['phi'] != 0:
                raise ValueError('Unexpected benchmark design or replication count')
            case_started = time.perf_counter()
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
                [2026101207, 1, case['index']])))
            pieces = {key: [] for key in FLOAT_KEYS+FLAG_KEYS}
            for start in range(0, count, 100):
                y0, y1 = simulation._generate_original(
                    rng, min(100, count-start), **{key: case[key] for key in GENERATOR_KEYS})
                fit = _core.analyze_batch(y0, y1, reference=provenance['reference_period'],
                                          fisher_valid=case['fisher_valid'])
                for key in pieces:
                    pieces[key].append(fit[key])
            actual = {key: np.concatenate(value) for key, value in pieces.items()}
            for key in FLOAT_KEYS+FLAG_KEYS:
                checks.append(compare(actual[key], fixture[case['id']+'__'+key], case['id'], key))

            reports = []
            for name, suffix in [('mean', 'mean'), ('variance', 'var'), ('Fisher', 'fisher')]:
                valid = actual['valid_'+suffix] & np.isfinite(actual['p_'+suffix])
                rejected = valid & (actual['p_'+suffix] <= provenance['nominal_alpha'])
                n_valid, n_rejected = int(valid.sum()), int(rejected.sum())
                expected = next(row for row in item['expected_results'] if row['test'] == name)
                matches = (n_rejected == expected['rejections']
                           and n_valid == expected['valid_repetitions']
                           and count-n_valid == expected['invalid_repetitions'])
                reports.append(dict(test=name, scheduled_repetitions=count,
                                    valid_repetitions=n_valid, invalid_repetitions=count-n_valid,
                                    rejections=n_rejected, rate_percent=100*n_rejected/count,
                                    reference_rejections=expected['rejections'],
                                    reference_rate_percent=expected['rate_percent'],
                                    counts_match=matches))
            case_reports.append(dict(case=case['id'], configuration=case,
                                     seed_sequence=item['seed_sequence'],
                                     source_file=item['source_file'], source_sha256=item['source_sha256'],
                                     elapsed_seconds=time.perf_counter()-case_started,
                                     results=reports))

    exact_count = sum(row['exact_match'] for row in checks)
    close_count = sum(row['tolerance_match'] for row in checks)
    counts_match = all(row['counts_match'] for case in case_reports for row in case['results'])
    passed = len(checks) == 24 and close_count == 24 and counts_match
    if strict_bitwise:
        passed = passed and exact_count == 24
    return dict(
        status='PASS' if passed else 'FAIL', label='PARTIAL_STUDY_REPRODUCTION',
        completed_utc=datetime.now(timezone.utc).isoformat(),
        scope=provenance['scope'], interpretation=(
            'Independent package execution reproduces selected frozen study outputs using the same '
            'scientific core and DGP. This does not independently prove the estimator or reproduce '
            'the full 198000-panel analytic study, bootstrap studies, or historical empirical analysis.'),
        total_panels=15000, case_count=3, batch_size=100, nominal_alpha=provenance['nominal_alpha'],
        rejection_rule=provenance['rejection_rule'],
        array_check_count=len(checks), exact_match_count=exact_count,
        tolerance_match_count=close_count, result_count=9, all_counts_match=counts_match,
        strict_bitwise=strict_bitwise, rtol=1e-10, atol=1e-12,
        elapsed_seconds=time.perf_counter()-started,
        provenance_sha256=sha256(provenance_path), fixture_sha256=provenance['fixture']['sha256'],
        core_sha256=core_hash, generator_module_sha256=sha256(simulation.__file__),
        reproduction_script_sha256=sha256(__file__),
        original_configuration=provenance['configuration'], original_summary=provenance['frozen_summary'],
        original_runner=provenance['frozen_runner'],
        runtime=dict(python=platform.python_version(), numpy=np.__version__,
                     didcheck=didcheck.__version__, system=platform.system(),
                     package_location='installed distribution' if 'site-packages' in didcheck.__file__ else 'source tree'),
        cases=case_reports, checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Portable JSON summary path; otherwise print JSON')
    parser.add_argument('--strict-bitwise', action='store_true', help='Require all 24 arrays to match bit-exactly')
    args = parser.parse_args()
    result = reproduce(strict_bitwise=args.strict_bitwise)
    rendered = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.write_text(rendered)
        print(f"{result['status']}: {result['total_panels']} panels; "
              f"{result['exact_match_count']}/24 bit-exact arrays; "
              f"all 9 rejection/validity counts match={result['all_counts_match']}; "
              f"{result['elapsed_seconds']:.2f}s")
        for case in result['cases']:
            rates = ', '.join(f"{row['test']}={row['rejections']}/5000 ({row['rate_percent']:.2f}%)"
                              for row in case['results'])
            print(case['case']+': '+rates)
    else:
        print(rendered, end='')
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
