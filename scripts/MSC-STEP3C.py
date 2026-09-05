from hashlib import sha256
from importlib import import_module
from pathlib import Path
import numpy as np
import pandas as pd
from msc_deterministic_model_common import PARAMETER_COLUMNS, default_reference_mask, deterministic_model

common = import_module('MSC-STEP3A-COMMON')
validation = import_module('MSC-STEP3A-VALIDATE')
stft = import_module('MSC-STEP3A-STFT')
cwt = import_module('MSC-STEP3A-CWT')
hilbert = import_module('MSC-STEP3A-HILBERT')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEP3A_TABLES = PROJECT_ROOT / 'results' / 'step3a' / 'tables'
STEP3B_TABLES = PROJECT_ROOT / 'results' / 'step3b' / 'tables'
OUTPUT_FOLDER = PROJECT_ROOT / 'results' / 'step3c' / 'tables'
STEP3A_TRUTH_FILE = STEP3A_TABLES / 'scenario_ground_truth_parameters.csv'
STEP3B_TRUTH_FILE = STEP3B_TABLES / 'step3b_challenge_ground_truth.csv.gz'
CALIBRATION_FILE = STEP3A_TABLES / 'amplitude_calibration_factors.csv'
STEP3A_ESTIMATE_FILES = {
    'STFT': STEP3A_TABLES / 'stft_estimates.csv.gz',
    'CWT_pyBOAT': STEP3A_TABLES / 'cwt_pyboat_estimates.csv.gz',
    'Hilbert': STEP3A_TABLES / 'hilbert_estimates.csv.gz',
}
STEP3B_ESTIMATE_FILES = {
    'STFT': STEP3B_TABLES / 'stft_challenge_estimates.csv.gz',
    'CWT_pyBOAT': STEP3B_TABLES / 'cwt_pyboat_challenge_estimates.csv.gz',
    'Hilbert': STEP3B_TABLES / 'hilbert_challenge_estimates.csv.gz',
}
ANALYSES = {
    'STFT': stft.analyse_stft,
    'CWT_pyBOAT': cwt.analyse_cwt,
    'Hilbert': hilbert.analyse_hilbert,
}
PLANS = [
    ('ENV_CWT_PRE_PERIOD', 'envelope', 'CWT_pyBOAT', 'pre_period_hours', 'pre_envelope'),
    ('ENV_CWT_POST_PERIOD', 'envelope', 'CWT_pyBOAT', 'post_period_hours', 'post_envelope'),
    ('ENV_HILBERT_PRE_PERIOD', 'envelope', 'Hilbert', 'pre_period_hours', 'pre_envelope'),
    ('ENV_HILBERT_POST_PERIOD', 'envelope', 'Hilbert', 'post_period_hours', 'post_envelope'),
    ('ENV_CWT_PHASE', 'envelope', 'CWT_pyBOAT', 'phase_shift_hours', 'pre_envelope'),
    ('ENV_HILBERT_PHASE', 'envelope', 'Hilbert', 'phase_shift_hours', 'pre_envelope'),
    ('ART_CWT_PRE_PERIOD', 'artifact', 'CWT_pyBOAT', 'pre_period_hours', 'transient_artifact'),
    ('ART_CWT_POST_PERIOD', 'artifact', 'CWT_pyBOAT', 'post_period_hours', 'transient_artifact'),
    ('ART_HILBERT_PRE_PERIOD', 'artifact', 'Hilbert', 'pre_period_hours', 'transient_artifact'),
    ('ART_CWT_PHASE', 'artifact', 'CWT_pyBOAT', 'phase_shift_hours', 'transient_artifact'),
]
for method in ('CWT_pyBOAT', 'Hilbert', 'STFT'):
    for epoch in ('pre', 'post'):
        PLANS.append((f'ART_{method.upper()}_{epoch.upper()}_AMPLITUDE', 'artifact', method, f'{epoch}_amplitude_trajectory', 'transient_artifact'))
PLAN = pd.DataFrame(PLANS, columns=['plan_id', 'rescue_experiment', 'method', 'scenario_family', 'source_component'])
REFERENCE_AMPLITUDE = 2.0
ARTIFACT_THRESHOLD_FRACTION = 0.05
ARTIFACT_PADDING_POINTS = 1
ARTIFACT_EDGE_FIT_POINTS = 6
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260903
EXPECTED_SOURCES = 530
EXPECTED_ANCHORS = 53
EPS = 1e-10

def stable_seed(*parts):
    return int.from_bytes(sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'little') % (2 ** 32 - 1)

def load_tables(files):
    return pd.concat([pd.read_csv(path, low_memory=False) for path in files.values()], ignore_index=True)

def source_grids(truth):
    sources = truth[['source_synthetic_id', 'duration_days']].drop_duplicates()
    dt_days = common.SAMPLING_INTERVAL_HOURS / 24.0
    return {
        str(row.source_synthetic_id): pd.DataFrame({'time_days_zero': np.arange(0.0, float(row.duration_days) + 0.5 * dt_days, dt_days)})
        for row in sources.itertuples(index=False)
    }

def select_step3a_truth(truth, experiment, condition):
    rows = []
    for plan in PLAN[PLAN['rescue_experiment'].eq(experiment)].to_dict('records'):
        component = 'target_only' if condition == 'A' else plan['source_component']
        selected = truth[
            truth['scenario_family'].eq(plan['scenario_family'])
            & truth['added_component'].eq(component)
        ].copy()
        selected['plan_id'] = plan['plan_id']
        selected['rescue_experiment'] = experiment
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)

def select_artifact_truth(truth):
    rows = []
    for plan in PLAN[PLAN['rescue_experiment'].eq('artifact')].to_dict('records'):
        selected = truth[
            truth['challenge_type'].eq('transient_artifact')
            & truth['scenario_family'].eq(plan['scenario_family'])
        ].copy()
        selected['plan_id'] = plan['plan_id']
        selected['rescue_experiment'] = 'artifact'
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)

def reconstruct_components(row, time_days):
    parameters = {column: float(row[column]) for column in PARAMETER_COLUMNS}
    return deterministic_model(
        t_days=time_days,
        forskolin_day=float(row['forskolin_day_zero']),
        duration_days=float(row['duration_days']),
        drift_reference_day=float(row['drift_reference_day_zero']),
        parameters=parameters,
        initial_phase=float(row['profiled_initial_phase_radians']),
    )

def envelope_corrected_truth(truth):
    corrected = select_step3a_truth(truth, 'envelope', 'B')
    corrected['source_condition_synthetic_id'] = corrected['synthetic_id'].astype(str)
    corrected['synthetic_id'] = corrected['synthetic_id'].astype(str) + '__C_oracle_envelope_normalised'
    corrected['correction_type'] = 'oracle_ground_truth_envelope_normalisation'
    corrected['correction_reference_amplitude'] = REFERENCE_AMPLITUDE
    return corrected

def artifact_component(time_hours, row):
    sigma = float(row['artifact_fwhm_hours']) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    distance = (np.asarray(time_hours, dtype=float) - float(row['artifact_center_hour'])) / sigma
    return float(row['artifact_signed_amplitude']) * np.exp(-0.5 * distance ** 2)

def artifact_mask(time_hours, row):
    component = artifact_component(time_hours, row)
    mask = np.abs(component) >= ARTIFACT_THRESHOLD_FRACTION * np.max(np.abs(component))
    for _ in range(ARTIFACT_PADDING_POINTS):
        mask = np.convolve(mask.astype(int), np.ones(3, dtype=int), mode='same') > 0
    indices = np.flatnonzero(mask)
    if indices[0] == 0:
        mode = 'left_boundary_linear_extrapolation'
    elif indices[-1] == len(mask) - 1:
        mode = 'right_boundary_linear_extrapolation'
    else:
        mode = 'interior_linear_interpolation'
    return mask, mode

def artifact_corrected_truth(truth, grids):
    corrected = select_artifact_truth(truth)
    metadata = []
    for row in corrected.to_dict('records'):
        info = common.prepare_generated_trace(row, grids[str(row['source_synthetic_id'])])
        mask, mode = artifact_mask(info['time_hours'], row)
        indices = np.flatnonzero(mask)
        metadata.append((
            str(row['synthetic_id']),
            str(row['synthetic_id']) + '__C_oracle_masked',
            float(info['time_hours'][indices[0]]),
            float(info['time_hours'][indices[-1]]),
            float(info['time_hours'][indices[-1]] - info['time_hours'][indices[0]] + info['dt_hours']),
            int(mask.sum()),
            mode,
        ))
    values = pd.DataFrame(metadata, columns=['source_condition_synthetic_id', 'synthetic_id', 'mask_start_hour', 'mask_end_hour', 'mask_duration_hours', 'mask_n_points', 'mask_edge_mode'])
    corrected = corrected.drop(columns='synthetic_id').reset_index(drop=True).join(values)
    corrected['correction_type'] = 'oracle_artifact_mask_linear_interpolation'
    corrected['mask_threshold_fraction'] = ARTIFACT_THRESHOLD_FRACTION
    corrected['mask_padding_points'] = ARTIFACT_PADDING_POINTS
    corrected['edge_fit_points'] = ARTIFACT_EDGE_FIT_POINTS
    return corrected

def envelope_corrected_trace(row, source_grid):
    info = common.prepare_generated_trace(row, source_grid)
    time_days = np.asarray(info['time_hours'], dtype=float) / 24.0
    components = reconstruct_components(row, time_days)
    envelope = np.asarray(components['amplitude_envelope'], dtype=float)
    corrected_raw = (
        np.asarray(components['baseline_component'], dtype=float)
        + np.asarray(components['linear_drift_component'], dtype=float)
        + np.asarray(components['acute_transient_component'], dtype=float)
        + np.asarray(components['rhythmic_component'], dtype=float) / envelope * REFERENCE_AMPLITUDE
    )
    reference = default_reference_mask(time_days, float(row['forskolin_day_zero']))
    info['signal'] = corrected_raw - np.median(corrected_raw[reference])
    info['gt_amplitude'] = np.full(len(corrected_raw), REFERENCE_AMPLITUDE)
    info['synthetic_id'] = row['synthetic_id']
    return info

def artifact_corrected_trace(row, source_grid):
    info = common.prepare_generated_trace(row, source_grid)
    time_hours = np.asarray(info['time_hours'], dtype=float)
    signal = np.asarray(info['signal'], dtype=float) + artifact_component(time_hours, row)
    mask, mode = artifact_mask(time_hours, row)
    keep = ~mask
    if mode == 'interior_linear_interpolation':
        signal[mask] = np.interp(time_hours[mask], time_hours[keep], signal[keep])
    else:
        clean_indices = np.flatnonzero(keep)
        fit_indices = clean_indices[:ARTIFACT_EDGE_FIT_POINTS] if mode.startswith('left') else clean_indices[-ARTIFACT_EDGE_FIT_POINTS:]
        slope, intercept = np.polyfit(time_hours[fit_indices], signal[fit_indices], 1)
        signal[mask] = slope * time_hours[mask] + intercept
    info['signal'] = signal
    info['synthetic_id'] = row['synthetic_id']
    return info

def run_corrected(experiment, truth, grids):
    outputs = []
    for method in PLAN.loc[PLAN['rescue_experiment'].eq(experiment), 'method'].drop_duplicates():
        plan_ids = set(PLAN.loc[PLAN['rescue_experiment'].eq(experiment) & PLAN['method'].eq(method), 'plan_id'])
        selected = truth[truth['plan_id'].isin(plan_ids)]
        factor = common.load_amplitude_correction_factor(CALIBRATION_FILE, method)
        rows = []
        for row in selected.to_dict('records'):
            info = {column: row.get(column, '') for column in common.TRACE_METADATA_COLUMNS}
            try:
                info = envelope_corrected_trace(row, grids[str(row['source_synthetic_id'])]) if experiment == 'envelope' else artifact_corrected_trace(row, grids[str(row['source_synthetic_id'])])
                result = ANALYSES[method](info, factor)
            except Exception as error:
                result = common.empty_result(info, method, True, False, f'{type(error).__name__}: {error}')
            rows.append({column: result.get(column, np.nan) for column in common.RESULT_COLUMNS})
        output = pd.DataFrame(rows, columns=common.RESULT_COLUMNS)
        stem = method.lower()
        output.to_csv(OUTPUT_FOLDER / f'step3c_{experiment}_c_{stem}_estimates.csv.gz', index=False)
        outputs.append(output)
    return pd.concat(outputs, ignore_index=True)

def condition_metrics(truth, estimates, experiment):
    outputs = []
    for plan in PLAN[PLAN['rescue_experiment'].eq(experiment)].to_dict('records'):
        plan_truth = truth[truth['plan_id'].eq(plan['plan_id'])]
        ids = set(plan_truth['synthetic_id'].astype(str))
        selected = estimates[estimates['method'].eq(plan['method']) & estimates['synthetic_id'].astype(str).isin(ids)]
        metrics = validation.build_trace_metrics(plan_truth, selected)
        metrics['plan_id'] = plan['plan_id']
        metrics['rescue_experiment'] = experiment
        outputs.append(metrics)
    return pd.concat(outputs, ignore_index=True)

def selected_metrics(metrics):
    output = metrics.copy()
    output['error'] = pd.to_numeric(output['primary_error'], errors='coerce')
    keys = ['plan_id', 'source_synthetic_id', 'rescue_experiment', 'method', 'scenario_family', 'anchor_recording_uid', 'anchor_recording_id', 'target_status', 'tissue', 'primary_metric_name', 'primary_metric_unit', 'error', 'processing_success', 'valid_primary']
    return output[keys]

def paired_metrics(a, b, c):
    keys = ['plan_id', 'source_synthetic_id']
    identity = ['rescue_experiment', 'method', 'scenario_family', 'anchor_recording_uid', 'anchor_recording_id', 'target_status', 'tissue', 'primary_metric_name', 'primary_metric_unit']
    a = selected_metrics(a)[keys + identity + ['error', 'processing_success', 'valid_primary']].rename(columns={'error': 'error_A', 'processing_success': 'processing_success_A', 'valid_primary': 'valid_primary_A'})
    b = selected_metrics(b)[keys + ['error', 'processing_success', 'valid_primary']].rename(columns={'error': 'error_B', 'processing_success': 'processing_success_B', 'valid_primary': 'valid_primary_B'})
    c = selected_metrics(c)[keys + ['error', 'processing_success', 'valid_primary']].rename(columns={'error': 'error_C', 'processing_success': 'processing_success_C', 'valid_primary': 'valid_primary_C'})
    output = a.merge(b, on=keys, validate='one_to_one').merge(c, on=keys, validate='one_to_one')
    output['delta_add'] = output['error_B'] - output['error_A']
    output['delta_rescue'] = output['error_B'] - output['error_C']
    output['delta_remaining'] = output['error_C'] - output['error_A']
    return output

def anchor_metrics(paired):
    groups = ['plan_id', 'rescue_experiment', 'method', 'scenario_family', 'target_status', 'tissue', 'anchor_recording_uid', 'anchor_recording_id', 'primary_metric_name', 'primary_metric_unit']
    values = ['error_A', 'error_B', 'error_C', 'delta_add', 'delta_rescue', 'delta_remaining']
    aggregation = {column: 'median' for column in values}
    aggregation.update({'processing_success_A': 'mean', 'processing_success_B': 'mean', 'processing_success_C': 'mean', 'valid_primary_A': 'mean', 'valid_primary_B': 'mean', 'valid_primary_C': 'mean', 'source_synthetic_id': 'nunique'})
    output = paired.groupby(groups, dropna=False, sort=True).agg(aggregation).reset_index()
    return output.rename(columns={'source_synthetic_id': 'n_variants'})

def bootstrap_summary(anchors):
    groups = ['plan_id', 'rescue_experiment', 'method', 'scenario_family', 'target_status', 'primary_metric_name', 'primary_metric_unit']
    values = ['error_A', 'error_B', 'error_C', 'delta_add', 'delta_rescue', 'delta_remaining']
    rows = []
    for keys, group in anchors.groupby(groups, dropna=False, sort=True):
        row = dict(zip(groups, keys))
        matrix = group[values].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
        matrix = matrix[np.isfinite(matrix).all(axis=1)]
        row['n_anchors_total'] = len(group)
        row['n_anchors_valid'] = len(matrix)
        row['bootstrap_replicates'] = BOOTSTRAP_REPLICATES if len(matrix) >= 2 else 0
        row['bootstrap_seed'] = stable_seed(BOOTSTRAP_SEED, *keys)
        point = np.median(matrix, axis=0) if len(matrix) else np.full(len(values), np.nan)
        lower = np.full(len(values), np.nan)
        upper = np.full(len(values), np.nan)
        if len(matrix) >= 2:
            rng = np.random.default_rng(row['bootstrap_seed'])
            samples = matrix[rng.integers(0, len(matrix), size=(BOOTSTRAP_REPLICATES, len(matrix)))]
            lower, upper = np.percentile(np.median(samples, axis=1), [2.5, 97.5], axis=0)
        for index, column in enumerate(values):
            row[f'{column}_point_estimate'] = point[index]
            row[f'{column}_ci_lower_95'] = lower[index]
            row[f'{column}_ci_upper_95'] = upper[index]
        if len(matrix) < 2:
            conclusion = 'not_estimable'
        elif row['delta_add_ci_lower_95'] <= EPS:
            conclusion = 'no_confirmed_degradation'
        elif row['delta_rescue_ci_lower_95'] <= EPS:
            conclusion = 'degradation_without_clear_rescue'
        elif row['delta_remaining_ci_lower_95'] <= EPS and row['delta_remaining_ci_upper_95'] >= -EPS:
            conclusion = 'full_rescue'
        elif row['delta_remaining_ci_upper_95'] < -EPS:
            conclusion = 'overcorrection_beyond_clean'
        else:
            conclusion = 'partial_rescue'
        row['mechanism_conclusion'] = conclusion
        row['rescue_fraction_of_median_degradation'] = row['delta_rescue_point_estimate'] / row['delta_add_point_estimate'] if np.isfinite(row['delta_add_point_estimate']) and abs(row['delta_add_point_estimate']) > EPS else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    step3a_truth = pd.read_csv(STEP3A_TRUTH_FILE, low_memory=False)
    step3b_truth = pd.read_csv(STEP3B_TRUTH_FILE, low_memory=False)
    source_ids = set(step3a_truth['source_synthetic_id'].astype(str))
    if len(source_ids) != EXPECTED_SOURCES or step3a_truth['anchor_recording_uid'].nunique() != EXPECTED_ANCHORS:
        raise ValueError('STEP3C requires 530 sources from 53 anchors.')
    grids = source_grids(step3a_truth)
    step3a_estimates = load_tables(STEP3A_ESTIMATE_FILES)
    step3b_estimates = load_tables(STEP3B_ESTIMATE_FILES)
    envelope_a_truth = select_step3a_truth(step3a_truth, 'envelope', 'A')
    envelope_b_truth = select_step3a_truth(step3a_truth, 'envelope', 'B')
    envelope_c_truth = envelope_corrected_truth(step3a_truth)
    artifact_a_truth = select_step3a_truth(step3a_truth, 'artifact', 'A')
    artifact_b_truth = select_artifact_truth(step3b_truth)
    artifact_c_truth = artifact_corrected_truth(step3b_truth, grids)
    envelope_c_truth.to_csv(OUTPUT_FOLDER / 'step3c_envelope_corrected_truth.csv.gz', index=False)
    artifact_c_truth.to_csv(OUTPUT_FOLDER / 'step3c_artifact_corrected_truth.csv.gz', index=False)
    envelope_c_estimates = run_corrected('envelope', envelope_c_truth, grids)
    artifact_c_estimates = run_corrected('artifact', artifact_c_truth, grids)
    envelope_paired = paired_metrics(
        condition_metrics(envelope_a_truth, step3a_estimates, 'envelope'),
        condition_metrics(envelope_b_truth, step3a_estimates, 'envelope'),
        condition_metrics(envelope_c_truth, envelope_c_estimates, 'envelope'),
    )
    artifact_paired = paired_metrics(
        condition_metrics(artifact_a_truth, step3a_estimates, 'artifact'),
        condition_metrics(artifact_b_truth, step3b_estimates, 'artifact'),
        condition_metrics(artifact_c_truth, artifact_c_estimates, 'artifact'),
    )
    paired = pd.concat([envelope_paired, artifact_paired], ignore_index=True)
    anchors = anchor_metrics(paired)
    if len(paired) != EXPECTED_SOURCES * len(PLAN) or not anchors['n_variants'].eq(10).all():
        raise RuntimeError('STEP3C A/B/C pairing is incomplete.')
    bootstrap = bootstrap_summary(anchors)
    paired.to_csv(OUTPUT_FOLDER / 'step3c_trace_level_abc_metrics.csv.gz', index=False)
    anchors.to_csv(OUTPUT_FOLDER / 'step3c_anchor_level_abc_metrics.csv', index=False)
    bootstrap.to_csv(OUTPUT_FOLDER / 'step3c_anchor_bootstrap_summary.csv', index=False)

if __name__ == '__main__':
    main()
