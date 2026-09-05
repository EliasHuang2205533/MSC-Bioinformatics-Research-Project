from hashlib import sha256
from importlib import import_module
from pathlib import Path
import numpy as np
import pandas as pd

from msc_deterministic_model_common import MODEL_SPEC_VERSION

common = import_module('MSC-STEP3A-COMMON')
stft = import_module('MSC-STEP3A-STFT')
cwt = import_module('MSC-STEP3A-CWT')
hilbert = import_module('MSC-STEP3A-HILBERT')
step3a_validation = import_module('MSC-STEP3A-VALIDATE')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEP2B_TABLES = PROJECT_ROOT / 'results' / 'step2b' / 'tables'
STEP3A_TABLES = PROJECT_ROOT / 'results' / 'step3a' / 'tables'
OUTPUT_FOLDER = PROJECT_ROOT / 'results' / 'step3b' / 'tables'
TRUTH_FILE = STEP3A_TABLES / 'scenario_ground_truth_parameters.csv'
SOURCE_TRACE_FILE = STEP2B_TABLES / 'synthetic_traces_long_format.csv.gz'
CALIBRATION_FILE = STEP3A_TABLES / 'amplitude_calibration_factors.csv'
CHALLENGE_TRUTH_FILE = OUTPUT_FOLDER / 'step3b_challenge_ground_truth.csv.gz'
METHOD_FILES = {
    'STFT': OUTPUT_FOLDER / 'stft_challenge_estimates.csv.gz',
    'CWT_pyBOAT': OUTPUT_FOLDER / 'cwt_pyboat_challenge_estimates.csv.gz',
    'Hilbert': OUTPUT_FOLDER / 'hilbert_challenge_estimates.csv.gz',
}
ANALYSES = {
    'STFT': stft.analyse_stft,
    'CWT_pyBOAT': cwt.analyse_cwt,
    'Hilbert': hilbert.analyse_hilbert,
}
METHOD_APPLICABILITY = common.METHOD_APPLICABILITY
TARGET_FAMILIES = (
    'pre_period_hours',
    'post_period_hours',
    'pre_amplitude_trajectory',
    'post_amplitude_trajectory',
    'phase_shift_hours',
)
CHALLENGES = ('gaussian_noise', 'transient_artifact', 'missing_block_6h')
NOISE_RATIO = 0.0276
ARTIFACT_MIN_HOURS = 1.0
ARTIFACT_MAX_HOURS = 3.0
ARTIFACT_MIN_FACTOR = 1.0
ARTIFACT_MAX_FACTOR = 3.0
MISSING_HOURS = 6.0
SEGMENT_GAP_HOURS = common.SEGMENT_GAP_HOURS
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260903
DESIGN_SEED = 20260812
EPS = 1e-10
EXPECTED_SOURCES = 530
EXPECTED_ANCHORS = 53
EXPECTED_CLEAN_ROWS = 2650
EXPECTED_CHALLENGE_ROWS = 7950
ESTIMATE_COLUMNS = common.RESULT_COLUMNS
ERROR_METRICS = [
    'primary_error', 'target_segment_shape_nrmse', 'envelope_log_rate_absolute_error_per_day',
    'phase_equivalent_absolute_error_hours', 'ratio_log_error',
]
GOODNESS_METRICS = ['trajectory_pearson']
BOOTSTRAP_METRICS = {
    'pre_period_hours': ['primary_error'],
    'post_period_hours': ['primary_error'],
    'phase_shift_hours': ['primary_error', 'phase_equivalent_absolute_error_hours'],
    'pre_amplitude_trajectory': [
        'primary_error', 'target_segment_shape_nrmse', 'trajectory_pearson',
        'envelope_log_rate_absolute_error_per_day', 'ratio_log_error',
    ],
    'post_amplitude_trajectory': [
        'primary_error', 'target_segment_shape_nrmse', 'trajectory_pearson',
        'envelope_log_rate_absolute_error_per_day', 'ratio_log_error',
    ],
}

def stable_seed(*parts):
    value = '|'.join(map(str, parts)).encode('utf-8')
    return int.from_bytes(sha256(value).digest()[:8], 'little') % (2 ** 32 - 1)

def existing_table(folder, stem):
    compressed = folder / f'{stem}.csv.gz'
    return compressed if compressed.exists() else folder / f'{stem}.csv'

def segment_masks(time_hours, forskolin_hour):
    return time_hours <= forskolin_hour - SEGMENT_GAP_HOURS, time_hours >= forskolin_hour + SEGMENT_GAP_HOURS

def placement_segment(family, challenge, rng):
    if family.startswith('pre_'):
        return 'pre'
    if family.startswith('post_'):
        return 'post'
    if challenge == 'gaussian_noise':
        return 'both'
    return 'pre' if rng.integers(0, 2) == 0 else 'post'

def select_artifact_centre(time_hours, mask, half_width, rng):
    indices = np.flatnonzero(mask)
    lower = time_hours[indices[0]] + half_width
    upper = time_hours[indices[-1]] - half_width
    candidates = np.flatnonzero(mask & (time_hours >= lower) & (time_hours <= upper))
    return float(time_hours[int(rng.choice(candidates))])

def select_missing_block(mask, dt_hours, rng):
    indices = np.flatnonzero(mask)
    n_points = int(round(MISSING_HOURS / dt_hours))
    start_position = int(rng.integers(1, len(indices) - n_points))
    start_index = int(indices[start_position])
    return start_index, start_index + n_points, n_points

def challenge_row(clean, trace_info, challenge):
    family = str(clean['scenario_family'])
    seed = stable_seed(DESIGN_SEED, clean['source_synthetic_id'], family, challenge)
    rng = np.random.default_rng(seed)
    time_hours = np.asarray(trace_info['time_hours'], dtype=float)
    envelope = np.asarray(trace_info['gt_amplitude'], dtype=float)
    pre_mask, post_mask = segment_masks(time_hours, float(trace_info['forskolin_hour']))
    segment = placement_segment(family, challenge, rng)
    mask = pre_mask if segment == 'pre' else post_mask if segment == 'post' else pre_mask | post_mask
    scale = max(float(np.median(envelope[mask])), EPS)
    row = dict(clean)
    row.update({
        'clean_synthetic_id': clean['synthetic_id'],
        'synthetic_id': f"{clean['synthetic_id']}__B_{challenge}",
        'added_component': f'robustness_{challenge}',
        'challenge_type': challenge,
        'challenge_seed': seed,
        'challenge_segment': segment,
        'amplitude_scale': scale,
        'noise_ratio': np.nan,
        'noise_sigma': np.nan,
        'artifact_center_hour': np.nan,
        'artifact_fwhm_hours': np.nan,
        'artifact_amplitude_factor': np.nan,
        'artifact_signed_amplitude': np.nan,
        'missing_start_index': np.nan,
        'missing_end_index_exclusive': np.nan,
        'missing_start_hour': np.nan,
        'missing_end_hour': np.nan,
        'missing_duration_hours': np.nan,
        'missing_n_points': np.nan,
        'missing_interpolation': 'none',
    })
    if challenge == 'gaussian_noise':
        row['noise_ratio'] = NOISE_RATIO
        row['noise_sigma'] = NOISE_RATIO * scale
    elif challenge == 'transient_artifact':
        fwhm = float(rng.uniform(ARTIFACT_MIN_HOURS, ARTIFACT_MAX_HOURS))
        centre = select_artifact_centre(time_hours, mask, fwhm / 2.0, rng)
        factor = float(rng.uniform(ARTIFACT_MIN_FACTOR, ARTIFACT_MAX_FACTOR))
        sign = -1.0 if rng.integers(0, 2) == 0 else 1.0
        row.update({
            'artifact_center_hour': centre,
            'artifact_fwhm_hours': fwhm,
            'artifact_amplitude_factor': factor,
            'artifact_signed_amplitude': sign * factor * scale,
        })
    else:
        start, end, n_points = select_missing_block(mask, float(trace_info['dt_hours']), rng)
        row.update({
            'missing_start_index': start,
            'missing_end_index_exclusive': end,
            'missing_start_hour': float(time_hours[start]),
            'missing_end_hour': float(time_hours[start] + n_points * trace_info['dt_hours']),
            'missing_duration_hours': n_points * float(trace_info['dt_hours']),
            'missing_n_points': n_points,
            'missing_interpolation': 'linear_to_original_regular_grid',
        })
    return row

def source_grids(source_ids):
    table = pd.read_csv(SOURCE_TRACE_FILE, usecols=['synthetic_id', 'time_days_zero'], low_memory=False)
    table = table[table['synthetic_id'].astype(str).isin(source_ids)]
    return {str(key): group for key, group in table.groupby('synthetic_id', sort=False)}

def generate_challenge_truth():
    truth = pd.read_csv(TRUTH_FILE, low_memory=False)
    clean = truth[(truth['added_component'] == 'target_only') & truth['scenario_family'].isin(TARGET_FAMILIES)].copy()
    complete = (
        len(clean) == EXPECTED_CLEAN_ROWS
        and clean['source_synthetic_id'].nunique() == EXPECTED_SOURCES
        and clean['anchor_recording_uid'].nunique() == EXPECTED_ANCHORS
    )
    if not complete:
        raise ValueError('STEP3A target-only design does not contain 2,650 rows from 530 sources and 53 anchors.')
    grids = source_grids(set(clean['source_synthetic_id'].astype(str)))
    rows = []
    for row in clean.to_dict('records'):
        trace_info = common.prepare_generated_trace(row, grids[str(row['source_synthetic_id'])])
        rows.extend(challenge_row(row, trace_info, challenge) for challenge in CHALLENGES)
    output = pd.DataFrame(rows)
    if len(output) != EXPECTED_CHALLENGE_ROWS or output['synthetic_id'].duplicated().any():
        raise RuntimeError('STEP3B challenge design is incomplete or contains duplicate IDs.')
    missing = output[output['challenge_type'] == 'missing_block_6h']
    if (
        not missing['missing_n_points'].eq(18).all()
        or not np.allclose(missing['missing_duration_hours'], MISSING_HOURS)
    ):
        raise RuntimeError('The 6-hour missing blocks do not contain exactly 18 observations.')
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    output.to_csv(CHALLENGE_TRUTH_FILE, index=False)
    return output, grids

def apply_challenge(trace_info, row):
    signal = np.asarray(trace_info['signal'], dtype=float).copy()
    time_hours = np.asarray(trace_info['time_hours'], dtype=float)
    challenge = row['challenge_type']
    if challenge == 'gaussian_noise':
        rng = np.random.default_rng(int(row['challenge_seed']))
        signal += rng.normal(0.0, float(row['noise_sigma']), len(signal))
    elif challenge == 'transient_artifact':
        sigma = float(row['artifact_fwhm_hours']) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        distance = (time_hours - float(row['artifact_center_hour'])) / sigma
        signal += float(row['artifact_signed_amplitude']) * np.exp(-0.5 * distance ** 2)
    else:
        start = int(row['missing_start_index'])
        end = int(row['missing_end_index_exclusive'])
        keep = np.ones(len(signal), dtype=bool)
        keep[start:end] = False
        signal[start:end] = np.interp(time_hours[start:end], time_hours[keep], signal[keep])
    output = dict(trace_info)
    output['signal'] = signal
    output['synthetic_id'] = row['synthetic_id']
    return output

def empty_estimate(row, method, applicable, error=''):
    output = {column: np.nan for column in ESTIMATE_COLUMNS}
    output.update({
        'model_spec_version': MODEL_SPEC_VERSION,
        'synthetic_id': row['synthetic_id'],
        'source_synthetic_id': row['source_synthetic_id'],
        'scenario_family': row['scenario_family'],
        'method': method,
        'method_applicable': applicable,
        'processing_success': False if applicable else np.nan,
        'failure_reason': error,
        'pre_period_valid': False,
        'post_period_valid': False,
        'phase_shift_valid': False,
        'pre_amplitude_valid': False,
        'post_amplitude_valid': False,
        'n_pre_estimates': 0,
        'n_post_estimates': 0,
    })
    return output

def run_method(method, truth, grids):
    factor = common.load_amplitude_correction_factor(CALIBRATION_FILE, method)
    analysis = ANALYSES[method]
    rows = []
    for row in truth.to_dict('records'):
        applicable = row['scenario_family'] in METHOD_APPLICABILITY[method]
        if not applicable:
            rows.append(empty_estimate(row, method, False))
            continue
        try:
            trace_info = common.prepare_generated_trace(row, grids[str(row['source_synthetic_id'])])
            result = analysis(apply_challenge(trace_info, row), factor)
            result['method_applicable'] = True
            rows.append({column: result.get(column, np.nan) for column in ESTIMATE_COLUMNS})
        except Exception as error:
            rows.append(empty_estimate(row, method, True, f'{type(error).__name__}: {error}'))
    output = pd.DataFrame(rows, columns=ESTIMATE_COLUMNS)
    if len(output) != EXPECTED_CHALLENGE_ROWS or output['synthetic_id'].duplicated().any():
        raise RuntimeError(f'{method} STEP3B output is incomplete or contains duplicate IDs.')
    output.to_csv(METHOD_FILES[method], index=False)
    return output

def paired_metrics(challenge, clean):
    metrics = list(dict.fromkeys(ERROR_METRICS + GOODNESS_METRICS + ['ground_truth', 'estimate', 'signed_error']))
    clean_columns = ['method', 'synthetic_id'] + metrics
    renamed = {
        'synthetic_id': 'clean_synthetic_id',
        **{column: f'clean_{column}' for column in metrics},
    }
    reference = clean[clean_columns].rename(columns=renamed)
    paired = challenge.merge(reference, on=['method', 'clean_synthetic_id'], how='left', validate='many_to_one')
    for metric in ERROR_METRICS:
        paired[f'deterioration_{metric}'] = paired[metric] - paired[f'clean_{metric}']
    for metric in GOODNESS_METRICS:
        paired[f'deterioration_{metric}'] = paired[f'clean_{metric}'] - paired[metric]
    return paired[paired['method_applicable']].copy()

def anchor_metrics(paired):
    groups = [
        'method', 'scenario_family', 'challenge_type', 'target_status', 'tissue',
        'anchor_recording_uid', 'anchor_recording_id', 'method_applicable',
        'primary_metric_name', 'primary_metric_unit',
    ]
    numeric = ['primary_error', 'clean_primary_error', 'deterioration_primary_error']
    for metric in ERROR_METRICS[1:] + GOODNESS_METRICS:
        numeric.extend([metric, f'clean_{metric}', f'deterioration_{metric}'])
    aggregation = {column: 'median' for column in numeric}
    aggregation.update({'processing_success': 'mean', 'valid_primary': 'mean', 'source_synthetic_id': 'nunique'})
    output = paired.groupby(groups, dropna=False, sort=True).agg(aggregation).reset_index()
    return output.rename(columns={
        'source_synthetic_id': 'n_variants',
        'processing_success': 'processing_success_fraction',
        'valid_primary': 'valid_fraction',
    })

def bootstrap_triplet(group, metric, seed):
    columns = [f'clean_{metric}', metric, f'deterioration_{metric}']
    values = group[columns].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    values = values[np.isfinite(values).all(axis=1)]
    point = np.median(values, axis=0) if len(values) else np.full(3, np.nan)
    lower = np.full(3, np.nan)
    upper = np.full(3, np.nan)
    completed = 0
    if len(values) >= 2:
        rng = np.random.default_rng(seed)
        samples = values[rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))]
        estimates = np.median(samples, axis=1)
        lower, upper = np.percentile(estimates, [2.5, 97.5], axis=0)
        completed = BOOTSTRAP_REPLICATES
    names = ('clean', 'challenge', 'deterioration')
    result = {'n_anchors_valid': len(values), 'bootstrap_replicates': completed, 'bootstrap_seed': seed}
    for index, name in enumerate(names):
        result[f'{name}_point_estimate'] = point[index]
        result[f'{name}_ci_lower_95'] = lower[index]
        result[f'{name}_ci_upper_95'] = upper[index]
    return result

def bootstrap_summary(anchors):
    groups = [
        'method', 'scenario_family', 'challenge_type', 'target_status',
        'method_applicable', 'primary_metric_name', 'primary_metric_unit',
    ]
    rows = []
    for keys, group in anchors.groupby(groups, dropna=False, sort=True):
        metadata = dict(zip(groups, keys))
        group = group.sort_values('anchor_recording_uid')
        for metric in BOOTSTRAP_METRICS[metadata['scenario_family']]:
            seed_parts = [value for column, value in zip(groups, keys) if column != 'challenge_type']
            seed = stable_seed(BOOTSTRAP_SEED, *seed_parts, metric)
            rows.append({**metadata, 'metric': metric, **bootstrap_triplet(group, metric, seed)})
    return pd.DataFrame(rows)

def failure_summary(metrics):
    table = metrics.copy()
    table['n_applicable'] = table['method_applicable'].astype(int)
    table['successful_applicable'] = table['processing_success'].where(table['method_applicable'])
    table['valid_applicable'] = table['valid_primary'].astype(float).where(table['method_applicable'])
    groups = [
        'method', 'scenario_family', 'challenge_type', 'target_status',
        'method_applicable',
    ]
    return table.groupby(groups, dropna=False, sort=True).agg(
        n_rows=('synthetic_id', 'size'),
        n_applicable=('n_applicable', 'sum'),
        processing_success_fraction=('successful_applicable', 'mean'),
        target_metric_valid_fraction=('valid_applicable', 'mean'),
    ).reset_index()

def compact_paired(paired):
    metadata = [
        'model_spec_version', 'synthetic_id', 'clean_synthetic_id', 'source_synthetic_id',
        'anchor_index', 'variant_number', 'anchor_recording_uid', 'anchor_recording_id', 'tissue',
        'scenario_family', 'challenge_type', 'challenge_segment', 'target_status', 'method',
        'method_applicable', 'processing_success', 'failure_reason', 'primary_metric_name',
        'primary_metric_unit', 'valid_primary',
    ]
    metrics = [
        'ground_truth', 'estimate', 'signed_error', 'primary_error',
        'clean_primary_error', 'deterioration_primary_error',
    ]
    for metric in ERROR_METRICS[1:] + GOODNESS_METRICS:
        metrics.extend([metric, f'clean_{metric}', f'deterioration_{metric}'])
    return paired[[column for column in metadata + metrics if column in paired.columns]]

def validate(challenge_truth, challenge_estimates):
    full_truth = pd.read_csv(TRUTH_FILE, low_memory=False)
    clean_truth = full_truth[
        (full_truth['added_component'] == 'target_only')
        & full_truth['scenario_family'].isin(TARGET_FAMILIES)
    ].copy()
    clean_ids = set(clean_truth['synthetic_id'].astype(str))
    clean_tables = []
    method_stems = [
        ('STFT', 'stft_estimates'),
        ('CWT_pyBOAT', 'cwt_pyboat_estimates'),
        ('Hilbert', 'hilbert_estimates'),
    ]
    for method, stem in method_stems:
        table = pd.read_csv(existing_table(STEP3A_TABLES, stem), low_memory=False)
        clean_tables.append(table[table['synthetic_id'].astype(str).isin(clean_ids)])
    clean_metrics = step3a_validation.build_trace_metrics(
        clean_truth, pd.concat(clean_tables, ignore_index=True)
    )
    challenge_metrics = step3a_validation.build_trace_metrics(
        challenge_truth, pd.concat(challenge_estimates.values(), ignore_index=True)
    )
    paired = paired_metrics(challenge_metrics, clean_metrics)
    anchors = anchor_metrics(paired)
    if not anchors['n_variants'].eq(10).all():
        raise RuntimeError('Each STEP3B anchor cell must contain exactly 10 source variants.')
    bootstrap = bootstrap_summary(anchors)
    failures = failure_summary(challenge_metrics)
    compact_paired(paired).to_csv(OUTPUT_FOLDER / 'step3b_paired_clean_vs_challenge.csv.gz', index=False)
    anchors.to_csv(OUTPUT_FOLDER / 'step3b_anchor_level_metrics.csv', index=False)
    bootstrap.to_csv(OUTPUT_FOLDER / 'step3b_anchor_bootstrap_summary.csv', index=False)
    failures.to_csv(OUTPUT_FOLDER / 'step3b_failure_summary.csv', index=False)
    return paired, anchors, bootstrap, failures

def main():
    truth, grids = generate_challenge_truth()
    estimates = {method: run_method(method, truth, grids) for method in ANALYSES}
    validate(truth, estimates)

if __name__ == '__main__':
    main()
