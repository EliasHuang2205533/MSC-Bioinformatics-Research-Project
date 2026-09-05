from pathlib import Path
import numpy as np
import pandas as pd
from msc_deterministic_model_common import MODEL_SPEC_VERSION, PARAMETER_COLUMNS, deterministic_model, phase_angle_to_hours

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_FOLDER = PROJECT_ROOT / 'results' / 'step3a' / 'tables'
DEFAULT_TRUTH_FILE = DEFAULT_OUTPUT_FOLDER / 'scenario_ground_truth_parameters.csv'
DEFAULT_SOURCE_TRACE_FILE = PROJECT_ROOT / 'results' / 'step2b' / 'tables' / 'synthetic_traces_long_format.csv.gz'
MIN_PERIOD_HOURS = 18.0
MAX_PERIOD_HOURS = 32.0
DETREND_CUTOFF_HOURS = 72.0
STFT_WINDOW_HOURS = 60.0
STFT_OVERLAP_FRACTION = 0.9
SEGMENT_GAP_HOURS = 30.0
SAMPLING_INTERVAL_HOURS = 20.0 / 60.0
EPS = 1e-10
METHOD_APPLICABILITY = {
    'STFT': {'pre_period_hours', 'post_period_hours', 'pre_amplitude_trajectory', 'post_amplitude_trajectory'},
    'CWT_pyBOAT': {'pre_period_hours', 'post_period_hours', 'pre_amplitude_trajectory', 'post_amplitude_trajectory', 'phase_shift_hours'},
    'Hilbert': {'pre_period_hours', 'post_period_hours', 'pre_amplitude_trajectory', 'post_amplitude_trajectory', 'phase_shift_hours'},
}
TRACE_METADATA_COLUMNS = ['model_spec_version', 'synthetic_id', 'source_synthetic_id', 'scenario_family']
RESULT_COLUMNS = [
    'model_spec_version', 'synthetic_id', 'source_synthetic_id', 'scenario_family', 'method', 'method_applicable',
    'processing_success', 'failure_reason', 'pre_period_valid', 'post_period_valid', 'phase_shift_valid',
    'pre_amplitude_valid', 'post_amplitude_valid', 'n_pre_estimates', 'n_post_estimates',
    'amplitude_correction_factor', 'estimated_pre_period_hours', 'estimated_post_period_hours',
    'estimated_pre_amplitude', 'estimated_post_amplitude', 'estimated_pre_amplitude_raw',
    'estimated_post_amplitude_raw', 'ground_truth_pre_amplitude', 'ground_truth_post_amplitude',
    'pre_amplitude_relative_error', 'post_amplitude_relative_error', 'amplitude_ratio_gt',
    'amplitude_ratio_estimated', 'ratio_log_error', 'signed_ratio_log_error',
    'estimated_pre_envelope_log_rate_per_day', 'estimated_post_envelope_log_rate_per_day',
    'estimated_phase_shift_angle_radians', 'estimated_phase_shift_hours',
    'pre_amplitude_trajectory_pearson', 'post_amplitude_trajectory_pearson',
    'pre_amplitude_trajectory_scaled_nrmse', 'post_amplitude_trajectory_scaled_nrmse',
    'pre_amplitude_absolute_nrmse', 'post_amplitude_absolute_nrmse',
]

def rolling_mean_detrend(signal, dt_hours, cutoff_hours):
    points = max(3, int(round(cutoff_hours / dt_hours)))
    if points % 2 == 0:
        points += 1
    trend = pd.Series(signal).rolling(window=points, center=True, min_periods=max(3, points // 4)).mean().interpolate(limit_direction='both').to_numpy(dtype=float)
    return np.asarray(signal, dtype=float) - trend

def make_common_time_grid(trace_info):
    time_hours = np.asarray(trace_info['time_hours'], dtype=float)
    dt_hours = float(trace_info['dt_hours'])
    nperseg = min(len(time_hours), max(8, int(round(STFT_WINDOW_HOURS / dt_hours))))
    noverlap = min(nperseg - 1, int(round(nperseg * STFT_OVERLAP_FRACTION)))
    starts = np.arange(0, len(time_hours) - nperseg + 1, nperseg - noverlap, dtype=int)
    return time_hours[0] + (starts + nperseg / 2.0) * dt_hours

def interpolate_to_common_time(source_time, values, common_time, unwrap=False):
    source_time = np.asarray(source_time, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(source_time) & np.isfinite(values)
    if np.sum(valid) < 2:
        return np.full(len(common_time), np.nan)
    x = source_time[valid]
    y = values[valid]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x, unique_index = np.unique(x, return_index=True)
    y = y[unique_index]
    if unwrap:
        y = np.unwrap(y)
    return np.interp(common_time, x, y, left=np.nan, right=np.nan)

def make_segment_masks(time_hours, forskolin_hour):
    time_hours = np.asarray(time_hours, dtype=float)
    finite = np.isfinite(time_hours)
    return finite & (time_hours <= forskolin_hour - SEGMENT_GAP_HOURS), finite & (time_hours >= forskolin_hour + SEGMENT_GAP_HOURS)

def robust_median(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else np.nan

def estimate_log_rate(time_hours, amplitude, mask, reference_hour):
    time_hours = np.asarray(time_hours, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(time_hours) & np.isfinite(amplitude) & (amplitude > EPS)
    if np.sum(mask) < 5:
        return np.nan
    x_days = (time_hours[mask] - reference_hour) / 24.0
    y = amplitude[mask]
    floor = max(float(np.nanpercentile(y, 5)) * 0.1, EPS)
    slope, _ = np.polyfit(x_days, np.log(np.maximum(y, floor)), 1)
    return float(slope)

def estimate_phase_shift(time_hours, phase_radians, pre_mask, post_mask, forskolin_hour):
    time_hours = np.asarray(time_hours, dtype=float)
    phase = np.unwrap(np.asarray(phase_radians, dtype=float))
    pre = pre_mask & np.isfinite(time_hours) & np.isfinite(phase)
    post = post_mask & np.isfinite(time_hours) & np.isfinite(phase)
    if np.sum(pre) < 5 or np.sum(post) < 5:
        return np.nan
    pre_at_f = np.polyval(np.polyfit(time_hours[pre], phase[pre], 1), forskolin_hour)
    post_at_f = np.polyval(np.polyfit(time_hours[post], phase[post], 1), forskolin_hour)
    return float(np.angle(np.exp(1j * (post_at_f - pre_at_f))))

def trajectory_metrics(estimate, truth):
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    valid = np.isfinite(estimate) & np.isfinite(truth) & (truth > EPS)
    estimate = estimate[valid]
    truth = truth[valid]
    if len(estimate) < 5:
        return np.nan, np.nan
    pearson = float(np.corrcoef(estimate, truth)[0, 1]) if np.std(estimate) > EPS and np.std(truth) > EPS else np.nan
    denominator = float(np.dot(estimate, estimate))
    scale = float(np.dot(estimate, truth) / denominator) if denominator > EPS else 1.0
    rmse = float(np.sqrt(np.mean((estimate * scale - truth) ** 2)))
    return pearson, rmse / max(float(np.mean(truth)), EPS)

def calibrated_absolute_nrmse(amplitude_raw, truth, correction_factor, mask):
    calibrated = np.asarray(amplitude_raw, dtype=float) * float(correction_factor)
    truth = np.asarray(truth, dtype=float)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(calibrated) & np.isfinite(truth) & (truth > EPS)
    if np.sum(valid) < 5:
        return np.nan
    return float(np.sqrt(np.mean((calibrated[valid] - truth[valid]) ** 2)) / max(float(np.mean(truth[valid])), EPS))

def load_amplitude_correction_factor(calibration_file, method):
    table = pd.read_csv(Path(calibration_file))
    selected = table[table['method'].astype(str).eq(str(method))]
    if len(selected) != 1:
        raise ValueError(f'Expected one calibration factor for {method}.')
    factor = float(selected['amplitude_correction_factor'].iloc[0])
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError(f'Invalid amplitude correction factor for {method}.')
    return factor

def make_result_row(trace_info, method, time_hours, period_hours, amplitude, phase_radians=None, amplitude_correction_factor=1.0):
    common_time = make_common_time_grid(trace_info)
    period_hours = interpolate_to_common_time(time_hours, period_hours, common_time)
    amplitude_raw = interpolate_to_common_time(time_hours, amplitude, common_time)
    phase_common = interpolate_to_common_time(time_hours, phase_radians, common_time, unwrap=True) if phase_radians is not None else None
    pre_mask, post_mask = make_segment_masks(common_time, trace_info['forskolin_hour'])
    period_valid = np.isfinite(period_hours) & (period_hours >= MIN_PERIOD_HOURS) & (period_hours <= MAX_PERIOD_HOURS)
    pre_period = robust_median(period_hours[pre_mask & period_valid])
    post_period = robust_median(period_hours[post_mask & period_valid])
    pre_rate = estimate_log_rate(common_time, amplitude_raw, pre_mask, float(trace_info['time_hours'][0]))
    post_rate = estimate_log_rate(common_time, amplitude_raw, post_mask, trace_info['forskolin_hour'])
    phase_shift_angle = estimate_phase_shift(common_time, phase_common, pre_mask, post_mask, trace_info['forskolin_hour']) if phase_common is not None else np.nan
    phase_shift_hours = phase_angle_to_hours(phase_shift_angle, post_period) if np.isfinite(phase_shift_angle) and np.isfinite(post_period) else np.nan
    gt_at_output = np.interp(common_time, trace_info['time_hours'], trace_info['gt_amplitude'])
    factor = float(amplitude_correction_factor)
    pre_amplitude_raw = robust_median(amplitude_raw[pre_mask])
    post_amplitude_raw = robust_median(amplitude_raw[post_mask])
    pre_amplitude = pre_amplitude_raw * factor
    post_amplitude = post_amplitude_raw * factor
    gt_pre_amplitude = robust_median(gt_at_output[pre_mask])
    gt_post_amplitude = robust_median(gt_at_output[post_mask])
    pre_r, pre_shape_nrmse = trajectory_metrics(amplitude_raw[pre_mask], gt_at_output[pre_mask])
    post_r, post_shape_nrmse = trajectory_metrics(amplitude_raw[post_mask], gt_at_output[post_mask])
    pre_absolute_nrmse = calibrated_absolute_nrmse(amplitude_raw, gt_at_output, factor, pre_mask)
    post_absolute_nrmse = calibrated_absolute_nrmse(amplitude_raw, gt_at_output, factor, post_mask)
    pre_relative_error = (pre_amplitude - gt_pre_amplitude) / gt_pre_amplitude if gt_pre_amplitude > EPS else np.nan
    post_relative_error = (post_amplitude - gt_post_amplitude) / gt_post_amplitude if gt_post_amplitude > EPS else np.nan
    ratio_gt = gt_post_amplitude / gt_pre_amplitude if gt_pre_amplitude > EPS else np.nan
    ratio_estimated = post_amplitude_raw / pre_amplitude_raw if pre_amplitude_raw > EPS else np.nan
    signed_ratio_error = float(np.log(ratio_estimated / ratio_gt)) if ratio_gt > EPS and ratio_estimated > EPS else np.nan
    row = {column: trace_info[column] for column in TRACE_METADATA_COLUMNS}
    row.update({
        'method': method, 'method_applicable': True, 'processing_success': True, 'failure_reason': '',
        'pre_period_valid': bool(np.isfinite(pre_period)), 'post_period_valid': bool(np.isfinite(post_period)),
        'phase_shift_valid': bool(np.isfinite(phase_shift_angle)), 'pre_amplitude_valid': bool(np.isfinite(pre_absolute_nrmse)),
        'post_amplitude_valid': bool(np.isfinite(post_absolute_nrmse)), 'n_pre_estimates': int(np.sum(pre_mask & period_valid)),
        'n_post_estimates': int(np.sum(post_mask & period_valid)), 'amplitude_correction_factor': factor,
        'estimated_pre_period_hours': pre_period, 'estimated_post_period_hours': post_period,
        'estimated_pre_amplitude': pre_amplitude, 'estimated_post_amplitude': post_amplitude,
        'estimated_pre_amplitude_raw': pre_amplitude_raw, 'estimated_post_amplitude_raw': post_amplitude_raw,
        'ground_truth_pre_amplitude': gt_pre_amplitude, 'ground_truth_post_amplitude': gt_post_amplitude,
        'pre_amplitude_relative_error': pre_relative_error, 'post_amplitude_relative_error': post_relative_error,
        'amplitude_ratio_gt': ratio_gt, 'amplitude_ratio_estimated': ratio_estimated,
        'ratio_log_error': abs(signed_ratio_error), 'signed_ratio_log_error': signed_ratio_error,
        'estimated_pre_envelope_log_rate_per_day': pre_rate, 'estimated_post_envelope_log_rate_per_day': post_rate,
        'estimated_phase_shift_angle_radians': phase_shift_angle, 'estimated_phase_shift_hours': phase_shift_hours,
        'pre_amplitude_trajectory_pearson': pre_r, 'post_amplitude_trajectory_pearson': post_r,
        'pre_amplitude_trajectory_scaled_nrmse': pre_shape_nrmse,
        'post_amplitude_trajectory_scaled_nrmse': post_shape_nrmse,
        'pre_amplitude_absolute_nrmse': pre_absolute_nrmse, 'post_amplitude_absolute_nrmse': post_absolute_nrmse,
    })
    return row

def empty_result(trace_info, method, applicable, success, reason):
    row = {column: np.nan for column in RESULT_COLUMNS}
    for column in TRACE_METADATA_COLUMNS:
        row[column] = trace_info.get(column, '')
    row.update({'method': method, 'method_applicable': applicable, 'processing_success': success, 'failure_reason': reason})
    if applicable:
        for column in ['pre_period_valid', 'post_period_valid', 'phase_shift_valid', 'pre_amplitude_valid', 'post_amplitude_valid']:
            row[column] = False
        row['n_pre_estimates'] = 0
        row['n_post_estimates'] = 0
    return row

def prepare_generated_trace(scenario, source_grid):
    t_zero = pd.to_numeric(source_grid['time_days_zero'], errors='coerce').to_numpy(dtype=float)
    t_zero = np.unique(np.sort(t_zero[np.isfinite(t_zero)]))
    if len(t_zero) < 20:
        raise ValueError('Fewer than 20 unique finite source time points.')
    dt_days = SAMPLING_INTERVAL_HOURS / 24.0
    uniform_t_zero = np.arange(t_zero[0], t_zero[-1] + 0.5 * dt_days, dt_days)
    parameters = {column: float(scenario[column]) for column in PARAMETER_COLUMNS}
    components = deterministic_model(t_days=uniform_t_zero, forskolin_day=float(scenario['forskolin_day_zero']), duration_days=float(scenario['duration_days']), drift_reference_day=float(scenario['drift_reference_day_zero']), parameters=parameters, initial_phase=float(scenario['profiled_initial_phase_radians']))
    info = {column: scenario[column] for column in TRACE_METADATA_COLUMNS}
    info.update({'time_hours': uniform_t_zero * 24.0, 'signal': components['synthetic_signal_normalised'], 'gt_amplitude': components['amplitude_envelope'], 'dt_hours': SAMPLING_INTERVAL_HOURS, 'forskolin_hour': float(scenario['forskolin_day_zero']) * 24.0})
    return info

def run_generated_scenario_analysis(truth_file, source_trace_file, method, analysis_function, output_file, max_traces=None):
    if method not in METHOD_APPLICABILITY:
        raise ValueError(f'Unknown method: {method}')
    truth = pd.read_csv(truth_file, low_memory=False)
    required = {'model_spec_version', 'synthetic_id', 'source_synthetic_id', 'scenario_family', 'profiled_initial_phase_radians'}
    if required - set(truth.columns):
        raise ValueError('Scenario truth is missing required columns.')
    if truth['synthetic_id'].astype(str).duplicated().any():
        raise ValueError('Scenario truth contains duplicate synthetic IDs.')
    if set(truth['model_spec_version'].dropna().astype(str)) != {MODEL_SPEC_VERSION}:
        raise ValueError('Scenario truth has an incompatible model specification.')
    if max_traces is None and len(truth) != 14840:
        raise ValueError(f'Formal Step 3A requires 14,840 scenarios; found {len(truth)}.')
    if max_traces is not None:
        truth = truth.iloc[:int(max_traces)].copy()
    source_ids = set(truth['source_synthetic_id'].astype(str))
    grids = pd.read_csv(source_trace_file, usecols=['synthetic_id', 'time_days_zero'], low_memory=False)
    grids = grids[grids['synthetic_id'].astype(str).isin(source_ids)]
    grid_lookup = {str(key): group for key, group in grids.groupby('synthetic_id', sort=False)}
    if source_ids - set(grid_lookup):
        raise ValueError('Step 2B source grids are incomplete.')
    results = []
    for scenario in truth.to_dict('records'):
        trace_info = {column: scenario.get(column, '') for column in TRACE_METADATA_COLUMNS}
        if scenario['scenario_family'] not in METHOD_APPLICABILITY[method]:
            results.append(empty_result(trace_info, method, False, np.nan, 'not_applicable'))
            continue
        try:
            trace_info = prepare_generated_trace(scenario, grid_lookup[str(scenario['source_synthetic_id'])])
            results.append(analysis_function(trace_info))
        except Exception as error:
            results.append(empty_result(trace_info, method, True, False, str(error)))
    output = pd.DataFrame(results, columns=RESULT_COLUMNS)
    if len(output) != len(truth) or output['synthetic_id'].astype(str).duplicated().any():
        raise RuntimeError('Benchmark output does not match scenario truth.')
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_file, index=False)
    return output
