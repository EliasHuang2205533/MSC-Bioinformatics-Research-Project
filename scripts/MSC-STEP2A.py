from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize_scalar
from msc_deterministic_model_common import MODEL_SPEC_VERSION, PARAMETER_LOWER_BOUNDS, PARAMETER_UPPER_BOUNDS, REFERENCE_PRE_BUFFER_HOURS, circular_difference, deterministic_model, parameter_boundary_hits, phase_angle_to_hours, wrap_angle
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_INPUT_PATH = PROJECT_ROOT / 'results' / 'step1a' / 'tables' / 'all_lumicycle_analysis_long_format.csv'
FEATURE_INPUT_PATH = PROJECT_ROOT / 'results' / 'step1b' / 'tables' / 'empirical_pre_post_features_by_recording.csv'
OUTPUT_FOLDER = PROJECT_ROOT / 'results' / 'step2a'
TABLE_FOLDER = OUTPUT_FOLDER / 'tables'
FIT_PLOT_FOLDER = OUTPUT_FOLDER / 'figures' / 'fit_plots'
RESIDUAL_PLOT_FOLDER = OUTPUT_FOLDER / 'figures' / 'residual_plots'
MAX_NFEV = 3000
PHASE_GRID_SIZE = 48
PROFILE_PHASE_GRID_SIZE = 96
N_PHASE_STARTS = 3
MIN_PHASE_START_SEPARATION = np.pi / 4.0
PERIOD_STARTS_HOURS = (21.0, 24.0, 27.0, 29.0)
PROFILE_MAX_ITERATIONS = 24
PROFILE_RELATIVE_SSE_TOL = 1e-06
MAKE_PLOTS = True
EPS = 1e-08
INITIAL_EXCLUSION_HOURS = 6.0
INITIAL_EXCLUSION_DAYS = INITIAL_EXCLUSION_HOURS / 24.0
INITIAL_VALUE_DECIMALS = 8
FIT_PROCEDURE_VERSION = 'period_multistart_profile_joint_v4'

def safe_filename(value):
    return re.sub(r'[\\/:*?"<>|]+', '_', str(value))

def safe_numeric(value, fallback):
    value = pd.to_numeric(value, errors='coerce')
    if pd.isna(value) or not np.isfinite(value):
        return float(fallback)
    return float(value)

def normalise_recording(df_rec, feature_row):
    d = df_rec[['time_days', 'counts_sec']].dropna().sort_values('time_days').copy()
    split_day = safe_numeric(feature_row['forskolin_time_days'], np.nan)
    if not np.isfinite(split_day):
        raise ValueError('No valid forskolin time / 无有效 forskolin 时间。')
    recording_start_original = float(d['time_days'].min())
    exclusion_cutoff_original = recording_start_original + INITIAL_EXCLUSION_DAYS
    d = d.loc[d['time_days'] >= exclusion_cutoff_original].copy()
    if len(d) < 60:
        raise ValueError('Too few points after excluding the first 6 h. / 排除最初6小时后剩余数据点过少。')
    reference_start_original = exclusion_cutoff_original
    reference_end_original = split_day - REFERENCE_PRE_BUFFER_HOURS / 24.0
    reference_mask = (d['time_days'] >= reference_start_original) & (d['time_days'] < reference_end_original)
    if reference_mask.sum() < 30:
        raise ValueError('Too few points in the stable pre reference window / 稳定 pre 参考窗口内数据点过少。')
    reference_counts = d.loc[reference_mask, 'counts_sec'].to_numpy(float)
    pre_baseline = float(np.nanmedian(reference_counts))
    drift_reference_original = float(np.nanmedian(d.loc[reference_mask, 'time_days']))
    pre_amp_proxy = safe_numeric(feature_row['pre_amplitude_proxy'], np.nan)
    if not np.isfinite(pre_amp_proxy) or pre_amp_proxy <= 0:
        raise ValueError('Invalid pre amplitude proxy / 处理前振幅代理无效。')
    d['normalised_signal'] = (d['counts_sec'] - pre_baseline) / pre_amp_proxy
    d['reference_mask'] = reference_mask.to_numpy(dtype=bool)
    return (d, pre_baseline, pre_amp_proxy, recording_start_original, drift_reference_original, reference_start_original, reference_end_original)

def build_initial_parameter_starts(feature_row, lower, upper):
    pre_amp = 1.0
    post_amp = safe_numeric(feature_row.get('post_pre_amplitude_proxy_ratio'), 1.0)
    pre_period = safe_numeric(feature_row.get('pre_period_proxy_hours'), 24.0)
    post_period = safe_numeric(feature_row.get('post_period_proxy_hours'), 24.0)
    pre_env_log_rate = safe_numeric(feature_row.get('pre_envelope_log_rate_per_day'), -0.4)
    post_env_log_rate = safe_numeric(feature_row.get('post_envelope_log_rate_per_day'), -0.48)
    baseline_step = safe_numeric(feature_row.get('baseline_step_to_pre_amplitude'), 0.0)
    reference_scale = safe_numeric(feature_row.get('pre_amplitude_proxy'), np.nan)
    if np.isfinite(reference_scale) and reference_scale > 0:
        pre_drift_slope = safe_numeric(feature_row.get('pre_drift_slope_per_day'), 0.0) / reference_scale
        post_drift_slope = safe_numeric(feature_row.get('post_drift_slope_per_day'), 0.0) / reference_scale
    else:
        pre_drift_slope = 0.0
        post_drift_slope = 0.0
    drift_slope = np.clip(np.mean([pre_drift_slope, post_drift_slope]), -1.0, 1.0)
    empirical = np.round(np.array([pre_amp, post_amp, pre_period, post_period, pre_env_log_rate, post_env_log_rate, baseline_step, 0.0, drift_slope, 0.0], dtype=float), INITIAL_VALUE_DECIMALS)
    empirical = np.clip(empirical, lower + 1e-06, upper - 1e-06)
    starts = [('empirical', empirical)]
    conservative = empirical.copy()
    conservative[1] = np.clip(conservative[1], 0.15, 6.0)
    conservative[4] = np.clip(conservative[4], -0.7, 0.12)
    conservative[5] = np.clip(conservative[5], -0.95, 0.15)
    conservative[6] = np.clip(conservative[6], -1.5, 1.5)
    if not np.allclose(empirical, conservative, rtol=0.0, atol=1e-12):
        starts.append(('conservative', conservative))
    for pre_start in PERIOD_STARTS_HOURS:
        for post_start in PERIOD_STARTS_HOURS:
            candidate = empirical.copy()
            candidate[2] = pre_start
            candidate[3] = post_start
            if not any(np.allclose(candidate, existing, rtol=0.0, atol=1e-12) for _, existing in starts):
                starts.append((f'period_{pre_start:g}_{post_start:g}', candidate))
    return starts

def residual_vector(params, t_days, observed, forskolin_day, duration_days, drift_reference_day, initial_phase, reference_mask):
    fitted = deterministic_model(t_days, forskolin_day, duration_days, drift_reference_day, params, initial_phase, reference_mask)['fitted_signal']
    return fitted - observed

def joint_residual_vector(values, t_days, observed, forskolin_day, duration_days, drift_reference_day, reference_mask):
    values = np.asarray(values, dtype=float)
    return residual_vector(values[:10], t_days, observed, forskolin_day, duration_days, drift_reference_day, values[10], reference_mask)

def phase_objective(phase, params, t_days, observed, forskolin_day, duration_days, drift_reference_day, reference_mask):
    fitted = deterministic_model(t_days, forskolin_day, duration_days, drift_reference_day, params, wrap_angle(phase), reference_mask)['fitted_signal']
    return float(np.sum((fitted - observed) ** 2))

def select_phase_starts(t_days, observed, forskolin_day, duration_days, drift_reference_day, reference_mask, initial_params, n_starts):
    candidates = np.linspace(-np.pi, np.pi, PHASE_GRID_SIZE, endpoint=False)
    scored = sorted(((phase_objective(phase, initial_params, t_days, observed, forskolin_day, duration_days, drift_reference_day, reference_mask), float(phase)) for phase in candidates))
    selected = []
    for _, phase in scored:
        if all((abs(circular_difference(phase, kept)) >= MIN_PHASE_START_SEPARATION for kept in selected)):
            selected.append(phase)
        if len(selected) == n_starts:
            break
    if len(selected) < n_starts:
        for _, phase in scored:
            if phase not in selected:
                selected.append(phase)
            if len(selected) == n_starts:
                break
    return selected

def profile_one_start(start_phase, initial_params, lower, upper, t_days, observed, forskolin_day, duration_days, drift_reference_day, reference_mask):
    params = np.asarray(initial_params, dtype=float).copy()
    phase = wrap_angle(start_phase)
    final_result = least_squares(residual_vector, x0=params, bounds=(lower, upper), args=(t_days, observed, forskolin_day, duration_days, drift_reference_day, phase, reference_mask), loss='linear', max_nfev=MAX_NFEV)
    params = final_result.x
    params[7] = wrap_angle(params[7])
    previous_sse = phase_objective(phase, params, t_days, observed, forskolin_day, duration_days, drift_reference_day, reference_mask)
    for _ in range(PROFILE_MAX_ITERATIONS):
        accepted_phase = float(phase)
        accepted_params = params.copy()
        accepted_result = final_result
        accepted_sse = float(previous_sse)
        profile_grid = np.linspace(-np.pi, np.pi, PROFILE_PHASE_GRID_SIZE, endpoint=False)
        profile_scores = np.array([phase_objective(candidate, params, t_days, observed, forskolin_day, duration_days, drift_reference_day, reference_mask) for candidate in profile_grid])
        grid_best_phase = float(profile_grid[int(np.argmin(profile_scores))])
        grid_step = 2.0 * np.pi / PROFILE_PHASE_GRID_SIZE
        phase_fit = minimize_scalar(lambda candidate: phase_objective(wrap_angle(candidate), params, t_days, observed, forskolin_day, duration_days, drift_reference_day, reference_mask), bounds=(grid_best_phase - grid_step, grid_best_phase + grid_step), method='bounded', options={'xatol': 1e-09})
        refined_phase = wrap_angle(phase_fit.x)
        refined_phase_sse = phase_objective(refined_phase, params, t_days, observed, forskolin_day, duration_days, drift_reference_day, reference_mask)
        current_phase_sse = phase_objective(accepted_phase, params, t_days, observed, forskolin_day, duration_days, drift_reference_day, reference_mask)
        if current_phase_sse <= refined_phase_sse:
            trial_phase = accepted_phase
        else:
            trial_phase = refined_phase
        trial_result = least_squares(residual_vector, x0=params, bounds=(lower, upper), args=(t_days, observed, forskolin_day, duration_days, drift_reference_day, trial_phase, reference_mask), loss='linear', max_nfev=MAX_NFEV)
        trial_params = trial_result.x.copy()
        trial_params[7] = wrap_angle(trial_params[7])
        trial_sse = phase_objective(trial_phase, trial_params, t_days, observed, forskolin_day, duration_days, drift_reference_day, reference_mask)
        monotonic_tolerance = max(EPS, abs(accepted_sse) * 1e-08)
        if trial_sse > accepted_sse + monotonic_tolerance:
            phase = accepted_phase
            params = accepted_params
            final_result = accepted_result
            previous_sse = accepted_sse
            break
        phase = float(trial_phase)
        params = trial_params
        final_result = trial_result
        current_sse = float(trial_sse)
        relative_improvement = max(0.0, (accepted_sse - current_sse) / max(accepted_sse, EPS))
        previous_sse = current_sse
        if relative_improvement <= PROFILE_RELATIVE_SSE_TOL:
            break
    joint_polish_initial_sse = float(previous_sse)
    joint_x0 = np.concatenate([params, [float(phase)]])
    joint_lower = np.concatenate([lower, [float(phase) - np.pi]])
    joint_upper = np.concatenate([upper, [float(phase) + np.pi]])
    joint_result = least_squares(joint_residual_vector, x0=joint_x0, bounds=(joint_lower, joint_upper), args=(t_days, observed, forskolin_day, duration_days, drift_reference_day, reference_mask), loss='linear', max_nfev=MAX_NFEV)
    joint_params = joint_result.x[:10].copy()
    joint_params[7] = wrap_angle(joint_params[7])
    joint_phase = wrap_angle(joint_result.x[10])
    joint_polish_final_sse = phase_objective(joint_phase, joint_params, t_days, observed, forskolin_day, duration_days, drift_reference_day, reference_mask)
    joint_polish_accepted = bool(joint_result.success and np.isfinite(joint_polish_final_sse) and (joint_polish_final_sse <= joint_polish_initial_sse + max(EPS, abs(joint_polish_initial_sse) * 1e-08)))
    if joint_polish_accepted:
        params = joint_params
        phase = float(joint_phase)
        final_result = joint_result
    final_sse = phase_objective(phase, params, t_days, observed, forskolin_day, duration_days, drift_reference_day, reference_mask)
    return {'phase': float(phase), 'params': params, 'final_sse': float(final_sse), 'solver_success': bool(final_result.success)}

def fit_metrics(observed, fitted):
    residual = observed - fitted
    sse = float(np.sum(residual ** 2))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    signal_range = float(np.nanpercentile(observed, 95) - np.nanpercentile(observed, 5))
    nrmse = rmse / signal_range if signal_range > 0 else np.nan
    total_ss = float(np.sum((observed - np.mean(observed)) ** 2))
    r2 = 1 - sse / total_ss if total_ss > 0 else np.nan
    return {'sse': sse, 'rmse': rmse, 'mae': mae, 'nrmse_q95_q05': nrmse, 'r_squared': r2}

def fit_one_recording(recording_uid, df_rec, feature_row):
    d, raw_pre_baseline, raw_pre_amp_proxy, recording_start_original, drift_reference_original, reference_start_original, reference_end_original = normalise_recording(df_rec, feature_row)
    t_original = d['time_days'].to_numpy(dtype=float)
    observed = d['normalised_signal'].to_numpy(dtype=float)
    reference_mask = d['reference_mask'].to_numpy(dtype=bool)
    fit_start_original = float(t_original.min())
    t_zero = t_original - fit_start_original
    drift_reference_zero = drift_reference_original - fit_start_original
    forskolin_original = safe_numeric(feature_row['forskolin_time_days'], np.nan)
    forskolin_zero = forskolin_original - fit_start_original
    if forskolin_zero <= 0:
        raise ValueError('Forskolin time is not after the retained fit start. / Forskolin 时间不在屏蔽后的拟合起点之后。')
    if not 0.0 <= drift_reference_zero < forskolin_zero:
        raise ValueError('Drift reference is not inside the retained pre segment. / 漂移零点不在屏蔽后保留的 pre 段内。')
    duration_days = float(t_zero.max())
    lower = PARAMETER_LOWER_BOUNDS.copy()
    upper = PARAMETER_UPPER_BOUNDS.copy()
    parameter_starts = build_initial_parameter_starts(feature_row, lower, upper)
    profiled = []
    for parameter_start_label, initial_params in parameter_starts:
        n_phase_starts = N_PHASE_STARTS if parameter_start_label in {'empirical', 'conservative'} else 1
        phase_starts = select_phase_starts(t_zero, observed, forskolin_zero, duration_days, drift_reference_zero, reference_mask, initial_params, n_phase_starts)
        for phase in phase_starts:
            result = profile_one_start(phase, initial_params, lower, upper, t_zero, observed, forskolin_zero, duration_days, drift_reference_zero, reference_mask)
            result['parameter_start_label'] = parameter_start_label
            profiled.append(result)
    chosen = min(profiled, key=lambda item: item['final_sse'])
    if not chosen['solver_success']:
        raise RuntimeError(f'Optimisation failed for {recording_uid}')
    best = np.asarray(chosen['params'], dtype=float)
    best[7] = wrap_angle(best[7])
    initial_phase = float(chosen['phase'])
    components = deterministic_model(t_zero, forskolin_zero, duration_days, drift_reference_zero, best, initial_phase, reference_mask)
    fitted = components['fitted_signal']
    metrics = fit_metrics(observed, fitted)
    recording_id = str(df_rec['recording_id'].iloc[0])
    tissue = str(df_rec['tissue'].iloc[0])
    fitted_df = pd.DataFrame({'recording_uid': recording_uid, 'recording_id': recording_id, 'tissue': tissue, 'time_days_original': t_original, 'time_days_zero': t_zero, 'forskolin_day_zero': forskolin_zero, 'drift_reference_day_zero': drift_reference_zero, 'observed_normalised': observed, 'fitted_signal': fitted, 'residual': observed - fitted, 'baseline_component': components['baseline_component'], 'rhythmic_component': components['rhythmic_component'], 'amplitude_envelope': components['amplitude_envelope'], 'linear_drift_component': components['linear_drift_component'], 'acute_transient_component': components['acute_transient_component'], 'centering_component': components['centering_component'], 'transition_weight_post': components['transition']})
    fit_row = {'recording_uid': recording_uid, 'recording_id': recording_id, 'tissue': tissue, 'model_spec_version': MODEL_SPEC_VERSION, 'fit_procedure_version': FIT_PROCEDURE_VERSION, 'raw_pre_baseline': raw_pre_baseline, 'raw_pre_amplitude_proxy': raw_pre_amp_proxy, 'recording_start_day_original': recording_start_original, 'fit_start_day_original': fit_start_original, 'duration_days': duration_days, 'forskolin_day_original': forskolin_original, 'forskolin_day_zero': forskolin_zero, 'drift_reference_day_original': drift_reference_original, 'drift_reference_day_zero': drift_reference_zero, 'reference_window_start_day_original': reference_start_original, 'reference_window_end_day_original': reference_end_original, 'reference_window_n_points': int(np.sum(reference_mask)), 'profiled_initial_phase_radians': initial_phase, 'n_parameter_starts': len(parameter_starts), 'selected_parameter_start': chosen['parameter_start_label'], 'n_profiled_starts': len(profiled), 'fit_pre_amplitude': best[0], 'fit_post_amplitude': best[1], 'fit_pre_period_hours': best[2], 'fit_post_period_hours': best[3], 'fit_pre_envelope_log_rate_per_day': best[4], 'fit_post_envelope_log_rate_per_day': best[5], 'fit_baseline_step': best[6], 'fit_phase_shift_angle_radians': best[7], 'fit_phase_shift_hours': phase_angle_to_hours(best[7], best[3]), 'fit_linear_drift_slope_per_day': best[8], 'fit_signed_acute_transient_amplitude': best[9]}
    boundary_hits = parameter_boundary_hits(best)
    fit_row['active_bound_count'] = len(boundary_hits)
    fit_row['boundary_hit_parameters'] = ';'.join(boundary_hits) if boundary_hits else 'none'
    fit_row.update(metrics)
    return (fitted_df, fit_row)

def load_inputs():
    analysis = pd.read_csv(ANALYSIS_INPUT_PATH, low_memory=False)
    features = pd.read_csv(FEATURE_INPUT_PATH, low_memory=False)
    treated = features.loc[features['forskolin_status'].eq('administered_estimated')].copy()
    return (analysis, treated.set_index('recording_uid'))

def plot_segmented(x, y, **kwargs):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    segment = np.cumsum(np.r_[True, np.diff(x) > 21.0 / 1440.0 + EPS])
    label = kwargs.pop('label', None)
    for index, segment_id in enumerate(np.unique(segment)):
        mask = segment == segment_id
        plt.plot(x[mask], y[mask], label=label if index == 0 else '_nolegend_', **kwargs)

def make_plots(fitted_traces, fit_parameters):
    lookup = fit_parameters.set_index('recording_uid')
    for recording_uid, df_plot in fitted_traces.groupby('recording_uid'):
        df_plot = df_plot.sort_values('time_days_original')
        row = lookup.loc[recording_uid]
        recording_id = row['recording_id']
        title = f"{recording_id} | {row['tissue']} | R2={row['r_squared']:.3f} | NRMSE={row['nrmse_q95_q05']:.3f}"
        centre_line = df_plot['baseline_component'] + df_plot['linear_drift_component'] + df_plot['acute_transient_component'] + df_plot['centering_component']
        upper_envelope = centre_line + df_plot['amplitude_envelope']
        lower_envelope = centre_line - df_plot['amplitude_envelope']
        plt.figure(figsize=(12, 4))
        plot_segmented(df_plot['time_days_original'], df_plot['observed_normalised'], linewidth=0.8, alpha=0.55, label='Real recording, normalised')
        plot_segmented(df_plot['time_days_original'], df_plot['fitted_signal'], linewidth=1.5, label='Best-fit noiseless model')
        plot_segmented(df_plot['time_days_original'], upper_envelope, linestyle='--', linewidth=1.0, label='Fitted upper envelope')
        plot_segmented(df_plot['time_days_original'], lower_envelope, linestyle='--', linewidth=1.0, label='Fitted lower envelope')
        plot_segmented(df_plot['time_days_original'], centre_line, linestyle=':', linewidth=0.9, color='grey', label='Fitted centre line')
        plt.axvline(row['forskolin_day_original'], linestyle='--', linewidth=1.0, label='Forskolin time')
        plt.xlabel('Time (days)')
        plt.ylabel('Normalised signal')
        plt.title(title)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plot_key = safe_filename(recording_uid)
        plt.savefig(FIT_PLOT_FOLDER / f'{plot_key}_fit.png', dpi=300)
        plt.close()
        plt.figure(figsize=(12, 3))
        plot_segmented(df_plot['time_days_original'], df_plot['residual'], linewidth=0.8)
        plt.axhline(0.0, linewidth=1.0)
        plt.axvline(row['forskolin_day_original'], linestyle='--', linewidth=1.0)
        plt.xlabel('Time (days)')
        plt.ylabel('Residual')
        plt.title(f'{recording_id} residuals')
        plt.tight_layout()
        plt.savefig(RESIDUAL_PLOT_FOLDER / f'{plot_key}_residuals.png', dpi=300)
        plt.close()

def main():
    for folder in [TABLE_FOLDER, FIT_PLOT_FOLDER, RESIDUAL_PLOT_FOLDER]:
        folder.mkdir(parents=True, exist_ok=True)
    analysis, feature_lookup = load_inputs()
    groups = {recording_uid: recording for recording_uid, recording in analysis.groupby('recording_uid')}
    all_fitted = []
    fit_rows = []
    for recording_uid, feature_row in feature_lookup.iterrows():
        recording = groups[recording_uid]
        fitted_df, fit_row = fit_one_recording(recording_uid, recording, feature_row)
        all_fitted.append(fitted_df)
        fit_rows.append(fit_row)
    fitted_traces = pd.concat(all_fitted, ignore_index=True)
    fit_parameters = pd.DataFrame(fit_rows).sort_values('recording_uid').reset_index(drop=True)
    fitted_traces = fitted_traces.sort_values(['recording_uid', 'time_days_original']).reset_index(drop=True)
    fit_parameters.to_csv(TABLE_FOLDER / 'best_fit_parameters_by_recording.csv', index=False)
    fitted_traces.to_csv(TABLE_FOLDER / 'fitted_traces_long_format.csv', index=False)
    if MAKE_PLOTS:
        make_plots(fitted_traces, fit_parameters)
if __name__ == '__main__':
    main()
