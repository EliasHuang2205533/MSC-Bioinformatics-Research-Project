from hashlib import sha256
from importlib import import_module
from itertools import combinations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, hilbert, sosfiltfilt, stft

common = import_module('MSC-STEP3A-COMMON')
stft_settings = import_module('MSC-STEP3A-STFT')
cwt_settings = import_module('MSC-STEP3A-CWT')
hilbert_settings = import_module('MSC-STEP3A-HILBERT')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / 'results' / 'step2b' / 'tables' / 'synthetic_ground_truth_parameters.csv'
OUTPUT_FOLDER = PROJECT_ROOT / 'results' / 'step3d' / 'tables'
STEP3D_VERSION = '2026-09-04-dynamic-period-v3'
METHODS = ('STFT', 'CWT_pyBOAT', 'Hilbert')
SIGNAL_AMPLITUDE = 2.0
EVENT_MARGIN_HOURS = common.STFT_WINDOW_HOURS / 2.0
BOOTSTRAP_REPEATS = 2000
RANDOM_SEED = 123
EPS = 1e-10
SUMMARY_METRICS = (
    'primary_rmse_hours',
    'full_trajectory_rmse_hours',
    'primary_bias_hours',
    'primary_pearson',
    'dynamic_range_ratio',
    'timing_error_hours',
    'absolute_timing_error_hours',
    'within_1h_fraction',
)
PAIRED_METRICS = ('primary_rmse_hours', 'full_trajectory_rmse_hours')


def stable_seed(*parts):
    value = '|'.join(map(str, parts)).encode('utf-8')
    return int.from_bytes(sha256(value).digest()[:8], 'little')


def make_design():
    rows = [
        {'scenario_id': 'constant_24h', 'profile_type': 'constant', 'direction': 'none', 'duration_hours': np.nan, 'baseline_period_hours': 24.0, 'target_period_hours': 24.0, 'treatment_locked': False},
        {'scenario_id': 'linear_21to27h', 'profile_type': 'linear_chirp', 'direction': 'lengthening', 'duration_hours': np.nan, 'baseline_period_hours': 21.0, 'target_period_hours': 27.0, 'treatment_locked': False},
        {'scenario_id': 'linear_27to21h', 'profile_type': 'linear_chirp', 'direction': 'shortening', 'duration_hours': np.nan, 'baseline_period_hours': 27.0, 'target_period_hours': 21.0, 'treatment_locked': False},
    ]
    for duration in (24.0, 48.0, 72.0):
        for target, direction in ((27.0, 'lengthening'), (21.0, 'shortening')):
            rows.append({'scenario_id': f'sustained_{direction}_{int(duration)}h', 'profile_type': 'causal_sustained_transition', 'direction': direction, 'duration_hours': duration, 'baseline_period_hours': 24.0, 'target_period_hours': target, 'treatment_locked': True})
            rows.append({'scenario_id': f'transient_{direction}_{int(duration)}h', 'profile_type': 'causal_transient_pulse', 'direction': direction, 'duration_hours': duration, 'baseline_period_hours': 24.0, 'target_period_hours': target, 'treatment_locked': True})
    design = pd.DataFrame(rows)
    design.insert(0, 'scenario_number', np.arange(1, len(design) + 1))
    return design


def load_anchors():
    truth = pd.read_csv(INPUT_FILE)
    truth['variant_number'] = pd.to_numeric(truth['variant_number'], errors='raise').astype(int)
    anchors = truth.loc[truth['variant_number'].eq(0), ['model_spec_version', 'anchor_index', 'anchor_recording_uid', 'anchor_recording_id', 'tissue', 'duration_days', 'forskolin_day_zero', 'profiled_initial_phase_radians']].copy()
    anchors = anchors.sort_values('anchor_index').reset_index(drop=True)
    if len(anchors) != 53 or anchors['anchor_index'].duplicated().any():
        raise ValueError('STEP3D requires 53 unique V000 anchors.')
    if set(anchors['model_spec_version'].astype(str)) != {common.MODEL_SPEC_VERSION}:
        raise ValueError('STEP2B model specification is incompatible with STEP3D.')
    return anchors


def smoothstep(values):
    values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def period_profile(time_hours, forskolin_hour, scenario):
    profile_type = scenario['profile_type']
    baseline = float(scenario['baseline_period_hours'])
    target = float(scenario['target_period_hours'])
    if profile_type == 'constant':
        return np.full_like(time_hours, baseline)
    if profile_type == 'linear_chirp':
        fraction = (time_hours - time_hours[0]) / (time_hours[-1] - time_hours[0])
        return baseline + (target - baseline) * fraction
    duration = float(scenario['duration_hours'])
    unit_time = (time_hours - forskolin_hour) / duration
    if profile_type == 'causal_sustained_transition':
        weight = np.where(time_hours >= forskolin_hour, smoothstep(unit_time), 0.0)
        return baseline + (target - baseline) * weight
    if profile_type == 'causal_transient_pulse':
        active = (unit_time >= 0.0) & (unit_time <= 1.0)
        pulse = np.zeros_like(time_hours)
        pulse[active] = np.sin(np.pi * unit_time[active]) ** 2
        return baseline + (target - baseline) * pulse
    raise ValueError(f'Unknown profile type: {profile_type}')


def phase_from_period(time_hours, period_hours, initial_phase):
    angular_frequency = 2.0 * np.pi / period_hours
    return float(initial_phase) + cumulative_trapezoid(angular_frequency, time_hours, initial=0.0)


def analyse_stft(trace_info):
    time_hours = trace_info['time_hours']
    dt_hours = trace_info['dt_hours']
    signal = common.rolling_mean_detrend(trace_info['signal'], dt_hours, common.DETREND_CUTOFF_HOURS)
    nperseg = min(len(signal), max(8, int(round(common.STFT_WINDOW_HOURS / dt_hours))))
    noverlap = min(nperseg - 1, int(round(nperseg * common.STFT_OVERLAP_FRACTION)))
    frequencies, relative_time, spectrum = stft(signal, fs=1.0 / dt_hours, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=max(stft_settings.NFFT, nperseg), detrend=False, return_onesided=True, boundary=None, padded=False, scaling='spectrum')
    circadian = (frequencies >= 1.0 / common.MAX_PERIOD_HOURS) & (frequencies <= 1.0 / common.MIN_PERIOD_HOURS)
    circadian_spectrum = spectrum[circadian]
    circadian_frequencies = frequencies[circadian]
    ridge_index = np.argmax(np.abs(circadian_spectrum), axis=0)
    return time_hours[0] + relative_time, 1.0 / circadian_frequencies[ridge_index]


def analyse_cwt(trace_info):
    from pyboat import WAnalyzer
    time_hours = trace_info['time_hours']
    dt_hours = trace_info['dt_hours']
    analyzer = WAnalyzer(np.linspace(common.MIN_PERIOD_HOURS, common.MAX_PERIOD_HOURS, cwt_settings.N_PERIODS), dt_hours, time_unit_label='h')
    trend = analyzer.sinc_smooth(trace_info['signal'], T_c=common.DETREND_CUTOFF_HOURS)
    analyzer.compute_spectrum(trace_info['signal'] - trend, do_plot=False)
    ridge = analyzer.get_maxRidge(power_thresh=cwt_settings.RIDGE_POWER_THRESHOLD, smoothing_wsize=cwt_settings.RIDGE_SMOOTHING_POINTS)
    if ridge is None:
        ridge = analyzer.ridge_data
    if not isinstance(ridge, pd.DataFrame):
        ridge = pd.DataFrame(ridge)
    period_column = cwt_settings.find_column(ridge, ['period', 'periods', 'ridge_period'])
    time_column = cwt_settings.find_column(ridge, ['time', 'times', 'time_hours'])
    if ridge.empty or period_column is None:
        raise ValueError('CWT period ridge is unavailable.')
    if time_column is None:
        ridge_time = time_hours[0] + np.asarray(ridge.index, dtype=float) * dt_hours
    else:
        ridge_time = ridge[time_column].to_numpy(dtype=float)
        if np.nanmin(ridge_time) < time_hours[0] - dt_hours:
            ridge_time = time_hours[0] + ridge_time
    return ridge_time, ridge[period_column].to_numpy(dtype=float)


def analyse_hilbert(trace_info):
    time_hours = trace_info['time_hours']
    dt_hours = trace_info['dt_hours']
    sos = butter(hilbert_settings.FILTER_ORDER, [1.0 / common.MAX_PERIOD_HOURS, 1.0 / common.MIN_PERIOD_HOURS], btype='bandpass', fs=1.0 / dt_hours, output='sos')
    phase = np.unwrap(np.angle(hilbert(sosfiltfilt(sos, trace_info['signal']))))
    angular_frequency = np.gradient(phase, time_hours)
    period = np.full_like(angular_frequency, np.nan)
    positive = angular_frequency > 0.0
    period[positive] = 2.0 * np.pi / angular_frequency[positive]
    return time_hours, period


ANALYSERS = {'STFT': analyse_stft, 'CWT_pyBOAT': analyse_cwt, 'Hilbert': analyse_hilbert}


def metric_values(truth, estimate, mask):
    valid = np.asarray(mask, dtype=bool) & np.isfinite(truth) & np.isfinite(estimate)
    if np.sum(valid) < 8:
        raise ValueError('Fewer than eight valid period estimates.')
    truth_valid = truth[valid]
    estimate_valid = estimate[valid]
    error = estimate_valid - truth_valid
    truth_range = float(np.ptp(truth_valid))
    estimate_range = float(np.ptp(estimate_valid))
    pearson = float(np.corrcoef(truth_valid, estimate_valid)[0, 1]) if np.std(truth_valid) > EPS and np.std(estimate_valid) > EPS else np.nan
    return {
        'n_timepoints': int(np.sum(mask)),
        'n_valid_estimates': int(np.sum(valid)),
        'valid_fraction': float(np.sum(valid) / np.sum(mask)),
        'rmse_hours': float(np.sqrt(np.mean(error ** 2))),
        'bias_hours': float(np.mean(error)),
        'pearson': pearson,
        'ground_truth_dynamic_range_hours': truth_range,
        'estimated_dynamic_range_hours': estimate_range,
        'dynamic_range_ratio': estimate_range / truth_range if truth_range > EPS else np.nan,
        'within_1h_fraction': float(np.mean(np.abs(error) <= 1.0)),
    }


def crossing_time(time_hours, values, level, direction, expected_time):
    candidates = []
    for index in range(len(time_hours) - 1):
        t0, t1 = time_hours[index:index + 2]
        y0, y1 = values[index:index + 2]
        if not np.all(np.isfinite([t0, t1, y0, y1])) or y0 == y1:
            continue
        crossed = y0 <= level <= y1 if direction == 'lengthening' else y0 >= level >= y1
        if crossed:
            candidates.append(float(t0 + (level - y0) * (t1 - t0) / (y1 - y0)))
    return min(candidates, key=lambda value: abs(value - expected_time)) if candidates else np.nan


def extremum_time(time_hours, values, direction):
    if not len(time_hours):
        return np.nan
    index = int(np.nanargmax(values) if direction == 'lengthening' else np.nanargmin(values))
    observed = float(time_hours[index])
    if 0 < index < len(time_hours) - 1:
        x = time_hours[index - 1:index + 2]
        y = values[index - 1:index + 2]
        centre = x[1]
        quadratic, linear, _ = np.polyfit(x - centre, y, 2)
        correct_curvature = quadratic < 0.0 if direction == 'lengthening' else quadratic > 0.0
        if correct_curvature and abs(quadratic) > EPS:
            candidate = float(centre - linear / (2.0 * quadratic))
            if x[0] <= candidate <= x[-1]:
                observed = candidate
    return observed


def timing_metrics(time_hours, estimate, mask, forskolin_hour, scenario):
    profile_type = scenario['profile_type']
    if profile_type not in {'causal_sustained_transition', 'causal_transient_pulse'}:
        return 'not_applicable', np.nan, np.nan, np.nan, np.nan
    duration = float(scenario['duration_hours'])
    expected = forskolin_hour + duration / 2.0
    valid = mask & np.isfinite(time_hours) & np.isfinite(estimate)
    selected_time = time_hours[valid]
    selected_estimate = estimate[valid]
    if profile_type == 'causal_sustained_transition':
        level = (float(scenario['baseline_period_hours']) + float(scenario['target_period_hours'])) / 2.0
        observed = crossing_time(selected_time, selected_estimate, level, scenario['direction'], expected)
        metric = 'midpoint_crossing'
    else:
        baseline = float(scenario['baseline_period_hours'])
        deviation = selected_estimate - baseline
        if scenario['direction'] == 'lengthening' and np.nanmax(deviation) > EPS:
            observed = extremum_time(selected_time, selected_estimate, scenario['direction'])
        elif scenario['direction'] == 'shortening' and np.nanmin(deviation) < -EPS:
            observed = extremum_time(selected_time, selected_estimate, scenario['direction'])
        else:
            observed = np.nan
        metric = 'peak_deviation'
    error = observed - expected if np.isfinite(observed) else np.nan
    return metric, expected, observed, error, abs(error) if np.isfinite(error) else np.nan


def calculate_metrics(truth, estimate, common_time, forskolin_hour, scenario):
    full_mask = np.ones(len(common_time), dtype=bool)
    if bool(scenario['treatment_locked']):
        start = forskolin_hour - EVENT_MARGIN_HOURS
        end = forskolin_hour + float(scenario['duration_hours']) + EVENT_MARGIN_HOURS
        primary_mask = (common_time >= start) & (common_time <= end)
        scope = 'event_window'
    else:
        primary_mask = full_mask
        scope = 'full_common_support'
    full = metric_values(truth, estimate, full_mask)
    primary = metric_values(truth, estimate, primary_mask)
    timing_metric, expected_time, observed_time, timing_error, absolute_timing_error = timing_metrics(common_time, estimate, primary_mask, forskolin_hour, scenario)
    selected_time = common_time[primary_mask]
    return {
        'primary_evaluation_scope': scope,
        'primary_evaluation_start_hours': float(selected_time[0]),
        'primary_evaluation_end_hours': float(selected_time[-1]),
        'n_common_timepoints': full['n_timepoints'],
        'n_valid_estimates': full['n_valid_estimates'],
        'valid_estimate_fraction': full['valid_fraction'],
        'n_primary_timepoints': primary['n_timepoints'],
        'n_primary_valid_estimates': primary['n_valid_estimates'],
        'primary_valid_estimate_fraction': primary['valid_fraction'],
        'primary_rmse_hours': primary['rmse_hours'],
        'full_trajectory_rmse_hours': full['rmse_hours'],
        'primary_bias_hours': primary['bias_hours'],
        'primary_pearson': primary['pearson'],
        'ground_truth_dynamic_range_hours': primary['ground_truth_dynamic_range_hours'],
        'estimated_dynamic_range_hours': primary['estimated_dynamic_range_hours'],
        'dynamic_range_ratio': primary['dynamic_range_ratio'],
        'within_1h_fraction': primary['within_1h_fraction'],
        'timing_metric': timing_metric,
        'ground_truth_event_time_hours': expected_time,
        'estimated_event_time_hours': observed_time,
        'timing_error_hours': timing_error,
        'absolute_timing_error_hours': absolute_timing_error,
    }


def run_validation(anchors, design):
    metric_rows = []
    estimate_tables = []
    dt_hours = common.SAMPLING_INTERVAL_HOURS
    for anchor in anchors.to_dict('records'):
        duration_hours = float(anchor['duration_days']) * 24.0
        time_hours = np.arange(0.0, duration_hours + 0.5 * dt_hours, dt_hours)
        forskolin_hour = float(anchor['forskolin_day_zero']) * 24.0
        trace_template = {'time_hours': time_hours, 'dt_hours': dt_hours}
        common_time = common.make_common_time_grid(trace_template)
        for scenario in design.to_dict('records'):
            truth_period = period_profile(time_hours, forskolin_hour, scenario)
            phase = phase_from_period(time_hours, truth_period, anchor['profiled_initial_phase_radians'])
            trace_info = trace_template | {'signal': SIGNAL_AMPLITUDE * np.cos(phase)}
            truth_common = np.interp(common_time, time_hours, truth_period)
            metadata = {
                'step3d_version': STEP3D_VERSION,
                'model_spec_version': anchor['model_spec_version'],
                'anchor_index': int(anchor['anchor_index']),
                'anchor_recording_uid': anchor['anchor_recording_uid'],
                'anchor_recording_id': anchor['anchor_recording_id'],
                'tissue': anchor['tissue'],
                'scenario_number': int(scenario['scenario_number']),
                'scenario_id': scenario['scenario_id'],
                'profile_type': scenario['profile_type'],
                'direction': scenario['direction'],
                'duration_hours': scenario['duration_hours'],
            }
            for method, analyser in ANALYSERS.items():
                source_time, source_period = analyser(trace_info)
                estimate_common = common.interpolate_to_common_time(source_time, source_period, common_time)
                in_band = np.isfinite(estimate_common) & (estimate_common >= common.MIN_PERIOD_HOURS) & (estimate_common <= common.MAX_PERIOD_HOURS)
                estimate_common = np.where(in_band, estimate_common, np.nan)
                metrics = calculate_metrics(truth_common, estimate_common, common_time, forskolin_hour, scenario)
                metric_rows.append(metadata | {'method': method} | metrics)
                estimate_tables.append(pd.DataFrame({
                    'anchor_index': int(anchor['anchor_index']),
                    'scenario_id': scenario['scenario_id'],
                    'method': method,
                    'time_hours_zero': common_time,
                    'time_from_forskolin_hours': common_time - forskolin_hour,
                    'ground_truth_period_hours': truth_common,
                    'estimated_period_hours': estimate_common,
                    'in_primary_evaluation_window': (common_time >= metrics['primary_evaluation_start_hours']) & (common_time <= metrics['primary_evaluation_end_hours']),
                }))
    return pd.DataFrame(metric_rows), pd.concat(estimate_tables, ignore_index=True)


def bootstrap_median(values, seed_parts):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    centre = float(np.median(values))
    if len(values) < 2:
        return centre, np.nan, np.nan, 0
    rng = np.random.default_rng(stable_seed(RANDOM_SEED, *seed_parts))
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPEATS, len(values)))
    bootstrapped = np.median(values[indices], axis=1)
    return centre, float(np.percentile(bootstrapped, 2.5)), float(np.percentile(bootstrapped, 97.5)), BOOTSTRAP_REPEATS


def bootstrap_summary(metrics):
    rows = []
    group_columns = ['scenario_number', 'scenario_id', 'profile_type', 'direction', 'duration_hours', 'method']
    for keys, group in metrics.groupby(group_columns, dropna=False, sort=True):
        metadata = dict(zip(group_columns, keys))
        n_total = int(group['anchor_index'].nunique())
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(group[metric], errors='coerce').to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            centre, low, high, repeats = bootstrap_median(values, tuple(keys) + (metric,))
            rows.append(metadata | {'metric': metric, 'primary_metric': metric == 'primary_rmse_hours', 'n_anchors_total': n_total, 'n_anchors_valid': len(values), 'median': centre, 'ci95_low': low, 'ci95_high': high, 'bootstrap_repeats': repeats})
    return pd.DataFrame(rows)


def paired_method_summary(metrics):
    rows = []
    metadata_columns = ['scenario_number', 'scenario_id', 'profile_type', 'direction', 'duration_hours']
    for scenario_id, scenario_table in metrics.groupby('scenario_id', sort=True):
        metadata = scenario_table.iloc[0][metadata_columns].to_dict()
        for metric in PAIRED_METRICS:
            pivot = scenario_table.pivot(index='anchor_index', columns='method', values=metric)
            for method_a, method_b in combinations(METHODS, 2):
                valid = np.isfinite(pivot[method_a]) & np.isfinite(pivot[method_b])
                differences = pivot.loc[valid, method_b].to_numpy(dtype=float) - pivot.loc[valid, method_a].to_numpy(dtype=float)
                centre, low, high, repeats = bootstrap_median(differences, (scenario_id, metric, method_a, method_b))
                rows.append(metadata | {'metric': metric, 'method_a': method_a, 'method_b': method_b, 'difference_definition': 'method_b_minus_method_a', 'lower_is_better': True, 'n_anchor_pairs': len(differences), 'median_difference_hours': centre, 'ci95_low': low, 'ci95_high': high, 'bootstrap_repeats': repeats})
    return pd.DataFrame(rows)


def main():
    design = make_design()
    anchors = load_anchors()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        metrics, estimates = run_validation(anchors, design)
        summary = bootstrap_summary(metrics)
        paired = paired_method_summary(metrics)
        if len(metrics) != 53 * 15 * 3 or len(paired) != 15 * 3 * 2:
            raise RuntimeError('STEP3D output is incomplete.')
        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        design.to_csv(OUTPUT_FOLDER / 'step3d_scenario_design.csv', index=False)
        metrics.to_csv(OUTPUT_FOLDER / 'step3d_anchor_level_metrics.csv', index=False)
        summary.to_csv(OUTPUT_FOLDER / 'step3d_anchor_bootstrap_summary.csv', index=False)
        paired.to_csv(OUTPUT_FOLDER / 'step3d_paired_method_differences.csv', index=False)
        estimates.to_csv(OUTPUT_FOLDER / 'step3d_time_resolved_estimates.csv.gz', index=False)


if __name__ == '__main__':
    main()
