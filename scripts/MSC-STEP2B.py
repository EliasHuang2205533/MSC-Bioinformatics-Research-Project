from pathlib import Path
import gzip
import re
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from msc_deterministic_model_common import ACUTE_TRANSIENT_SIGMA_HOURS, MODEL_SPEC_VERSION, PARAMETER_COLUMNS, TRANSITION_DURATION_DAYS, circular_difference, deterministic_model, parameter_boundary_hits, phase_angle_to_hours, wrap_angle
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FOLDER = PROJECT_ROOT / 'results' / 'step2a' / 'tables'
PARAMETER_FILE = INPUT_FOLDER / 'best_fit_parameters_by_recording.csv'
TRACE_FILE = INPUT_FOLDER / 'fitted_traces_long_format.csv'
OUTPUT_FOLDER = PROJECT_ROOT / 'results' / 'step2b'
TABLE_FOLDER = OUTPUT_FOLDER / 'tables'
FIGURE_FOLDER = OUTPUT_FOLDER / 'figures' / 'synthetic_examples'
EXCLUDED_RECORDING_IDS = {'2D-L4m2', '4A-M2SCN'}
RANDOM_SEED = 123
N_SYNTHETIC_PER_ANCHOR = 10
N_NEAREST_NEIGHBOURS = 5
MAX_INTERPOLATION_FRACTION = 0.35
EPS = 1e-08
ELIGIBLE_MIN_R_SQUARED = 0.6
INELIGIBLE_BELOW_R_SQUARED = 0.4
ELIGIBLE_MIN_CYCLES = 3.0
INELIGIBLE_BELOW_CYCLES = 2.0
ELIGIBLE_MIN_RHYTHM_TO_RESIDUAL = 0.35
INELIGIBLE_BELOW_RHYTHM_TO_RESIDUAL = 0.15
ELIGIBLE_MIN_SEGMENT_DAYS = 3.0
INELIGIBLE_BELOW_SEGMENT_DAYS = 1.5
WRITE_BATCH_SIZE = 25
N_FIGURE_TRACES = 15

def normalise_recording_id(value):
    return re.sub('\\s+', '', str(value)).casefold()

def exclude_unstable_anchors(parameters):
    recording_id = parameters['recording_id'].map(normalise_recording_id)
    excluded = {normalise_recording_id(value) for value in EXCLUDED_RECORDING_IDS}
    return parameters.loc[~recording_id.isin(excluded)].copy().reset_index(drop=True)

def build_neighbour_lookup(parameters, donor_uids):
    lookup = {}
    for tissue, table in parameters.groupby('tissue', sort=True):
        table = table.reset_index(drop=True)
        theta = table[PARAMETER_COLUMNS].to_numpy(dtype=float)
        q25 = np.nanpercentile(theta, 25, axis=0)
        q75 = np.nanpercentile(theta, 75, axis=0)
        scale = q75 - q25
        fallback = np.nanstd(theta, axis=0, ddof=0)
        scale = np.where(scale > EPS, scale, fallback)
        scale = np.where(scale > EPS, scale, 1.0)
        scale[7] = np.pi
        delta = theta[:, None, :] - theta[None, :, :]
        delta[:, :, 7] = circular_difference(theta[:, None, 7], theta[None, :, 7])
        distance = np.sqrt(np.sum((delta / scale) ** 2, axis=2))
        uid_order = table['recording_uid'].astype(str).str.casefold().to_numpy()
        donor_mask = table['recording_uid'].isin(donor_uids).to_numpy(dtype=bool)
        for i, uid in enumerate(table['recording_uid']):
            candidates = np.flatnonzero(donor_mask & (np.arange(len(table)) != i))
            n_neighbours = min(N_NEAREST_NEIGHBOURS, len(candidates))
            order = np.lexsort((uid_order[candidates], distance[i, candidates]))
            positions = candidates[order[:n_neighbours]]
            lookup[uid] = table.iloc[positions]['recording_uid'].tolist()
    return lookup

def make_local_variant(anchor_theta, neighbour_theta, fraction):
    theta = anchor_theta + fraction * (neighbour_theta - anchor_theta)
    theta[7] = wrap_angle(anchor_theta[7] + fraction * circular_difference(neighbour_theta[7], anchor_theta[7]))
    if not np.all(np.isfinite(theta)):
        raise ValueError('Generated parameters contain non-finite values.')
    if np.any(theta[:4] <= 0.0):
        raise ValueError('Generated amplitude or period is not positive.')
    return theta

def make_balanced_variant_schedule(neighbour_uids, n_local_variants, rng):
    neighbour_uids = list(neighbour_uids)
    if n_local_variants < 0:
        raise ValueError('n_local_variants must be non-negative.')
    if n_local_variants == 0:
        return []
    if not neighbour_uids:
        raise ValueError('At least one neighbour is required.')
    neighbour_order = []
    while len(neighbour_order) < n_local_variants:
        neighbour_order.extend(np.asarray(neighbour_uids, dtype=object)[rng.permutation(len(neighbour_uids))].tolist())
    neighbour_order = neighbour_order[:n_local_variants]
    fractions = np.linspace(MAX_INTERPOLATION_FRACTION / n_local_variants, MAX_INTERPOLATION_FRACTION, n_local_variants)
    fractions = fractions[rng.permutation(n_local_variants)]
    return list(zip(neighbour_order, fractions.astype(float)))

def robust_residual_scale(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    median = np.median(values)
    scale = 1.4826 * np.median(np.abs(values - median))
    if scale <= EPS:
        scale = np.sqrt(np.mean((values - median) ** 2))
    return float(max(scale, EPS))

def boundary_hit_set(row):
    return set(parameter_boundary_hits([getattr(row, column) for column in PARAMETER_COLUMNS]))

def join_reasons(reasons):
    return ';'.join(dict.fromkeys(reasons)) if reasons else 'none'

def assess_epoch_base(row, trace, epoch):
    forskolin_day = float(row.forskolin_day_zero)
    if epoch == 'pre':
        segment = trace[trace['time_days_zero'] < forskolin_day]
    else:
        segment = trace[trace['time_days_zero'] >= forskolin_day]
    if len(segment) < 2:
        return {'status': 'ineligible', 'reasons': [f'{epoch}_segment_missing'], 'duration_days': 0.0, 'n_cycles': 0.0, 'residual_scale': np.nan, 'rhythmic_rms': np.nan, 'rhythm_to_residual': np.nan, 'envelope_median': np.nan, 'envelope_p10': np.nan}
    duration_days = float(segment['time_days_zero'].max() - segment['time_days_zero'].min())
    period_hours = float(getattr(row, f'fit_{epoch}_period_hours'))
    n_cycles = duration_days * 24.0 / period_hours
    residual_scale = robust_residual_scale(segment['residual'])
    rhythmic_rms = float(np.sqrt(np.mean(segment['rhythmic_component'].to_numpy(dtype=float) ** 2)))
    rhythm_to_residual = rhythmic_rms / residual_scale
    envelope = segment['amplitude_envelope'].to_numpy(dtype=float)
    ineligible_reasons = []
    difficult_reasons = []
    r_squared = float(row.r_squared)
    required = [duration_days, period_hours, n_cycles, residual_scale, rhythmic_rms, rhythm_to_residual, r_squared]
    if not np.all(np.isfinite(required)):
        ineligible_reasons.append(f'{epoch}_nonfinite_diagnostic')
    if r_squared < INELIGIBLE_BELOW_R_SQUARED:
        ineligible_reasons.append('fit_r_squared_below_0.40')
    elif r_squared < ELIGIBLE_MIN_R_SQUARED:
        difficult_reasons.append('fit_r_squared_below_0.60')
    if duration_days < INELIGIBLE_BELOW_SEGMENT_DAYS:
        ineligible_reasons.append(f'{epoch}_segment_below_1.5_days')
    elif duration_days < ELIGIBLE_MIN_SEGMENT_DAYS:
        difficult_reasons.append(f'{epoch}_segment_below_3_days')
    if n_cycles < INELIGIBLE_BELOW_CYCLES:
        ineligible_reasons.append(f'{epoch}_fewer_than_2_cycles')
    elif n_cycles < ELIGIBLE_MIN_CYCLES:
        difficult_reasons.append(f'{epoch}_fewer_than_3_cycles')
    if rhythm_to_residual < INELIGIBLE_BELOW_RHYTHM_TO_RESIDUAL:
        ineligible_reasons.append(f'{epoch}_rhythm_to_residual_below_0.15')
    elif rhythm_to_residual < ELIGIBLE_MIN_RHYTHM_TO_RESIDUAL:
        difficult_reasons.append(f'{epoch}_rhythm_to_residual_below_0.35')
    if ineligible_reasons:
        status = 'ineligible'
        reasons = ineligible_reasons + difficult_reasons
    elif difficult_reasons:
        status = 'difficult'
        reasons = difficult_reasons
    else:
        status = 'eligible'
        reasons = []
    return {'status': status, 'reasons': reasons, 'duration_days': duration_days, 'n_cycles': n_cycles, 'residual_scale': residual_scale, 'rhythmic_rms': rhythmic_rms, 'rhythm_to_residual': rhythm_to_residual, 'envelope_median': float(np.median(envelope)), 'envelope_p10': float(np.percentile(envelope, 10))}

def add_target_boundary_rule(base_status, base_reasons, boundary_hits, columns):
    reasons = list(base_reasons)
    hit_columns = [column for column in columns if column in boundary_hits]
    if hit_columns:
        reasons.extend((f'{column}_at_bound' for column in hit_columns))
        if base_status == 'eligible':
            base_status = 'difficult'
    return (base_status, reasons, hit_columns)

def assess_anchor_eligibility(parameters, trace_groups):
    rows = []
    for row in parameters.itertuples(index=False):
        trace = trace_groups[row.recording_uid]
        pre = assess_epoch_base(row, trace, 'pre')
        post = assess_epoch_base(row, trace, 'post')
        boundary_hits = boundary_hit_set(row)
        pre_period_status, pre_period_reasons, pre_period_hits = add_target_boundary_rule(pre['status'], pre['reasons'], boundary_hits, ['fit_pre_period_hours'])
        post_period_status, post_period_reasons, post_period_hits = add_target_boundary_rule(post['status'], post['reasons'], boundary_hits, ['fit_post_period_hours'])
        pre_amp_status, pre_amp_reasons, pre_amp_hits = add_target_boundary_rule(pre['status'], pre['reasons'], boundary_hits, ['fit_pre_amplitude', 'fit_pre_envelope_log_rate_per_day'])
        post_amp_status, post_amp_reasons, post_amp_hits = add_target_boundary_rule(post['status'], post['reasons'], boundary_hits, ['fit_post_amplitude', 'fit_post_envelope_log_rate_per_day'])
        phase_reasons = list(pre['reasons']) + list(post['reasons'])
        phase_hits = []
        for column in ['fit_pre_period_hours', 'fit_post_period_hours']:
            if column in boundary_hits:
                phase_hits.append(column)
                phase_reasons.append(f'{column}_at_bound')
        if 'ineligible' in {pre['status'], post['status']}:
            phase_status = 'ineligible'
        elif 'difficult' in {pre['status'], post['status']} or phase_hits:
            phase_status = 'difficult'
        else:
            phase_status = 'eligible'
        all_hits = sorted(set(pre_period_hits + post_period_hits + pre_amp_hits + post_amp_hits + phase_hits))
        rows.append({'anchor_recording_uid': row.recording_uid, 'anchor_recording_id': row.recording_id, 'tissue': row.tissue, 'fit_r_squared': float(row.r_squared), 'fit_nrmse_q95_q05': float(row.nrmse_q95_q05), 'fit_active_bound_count': int(row.active_bound_count), 'target_boundary_hit_parameters': join_reasons(all_hits), 'pre_duration_days': pre['duration_days'], 'post_duration_days': post['duration_days'], 'pre_n_cycles': pre['n_cycles'], 'post_n_cycles': post['n_cycles'], 'pre_residual_robust_sd': pre['residual_scale'], 'post_residual_robust_sd': post['residual_scale'], 'pre_rhythmic_rms': pre['rhythmic_rms'], 'post_rhythmic_rms': post['rhythmic_rms'], 'pre_rhythm_to_residual_ratio': pre['rhythm_to_residual'], 'post_rhythm_to_residual_ratio': post['rhythm_to_residual'], 'pre_envelope_median': pre['envelope_median'], 'post_envelope_median': post['envelope_median'], 'pre_envelope_p10': pre['envelope_p10'], 'post_envelope_p10': post['envelope_p10'], 'pre_period_status': pre_period_status, 'pre_period_reason': join_reasons(pre_period_reasons), 'post_period_status': post_period_status, 'post_period_reason': join_reasons(post_period_reasons), 'pre_amplitude_envelope_status': pre_amp_status, 'pre_amplitude_envelope_reason': join_reasons(pre_amp_reasons), 'post_amplitude_envelope_status': post_amp_status, 'post_amplitude_envelope_reason': join_reasons(post_amp_reasons), 'phase_shift_status': phase_status, 'phase_shift_reason': join_reasons(phase_reasons)})
    return pd.DataFrame(rows)

def variant_diagnostics(t_days, forskolin_day, params, components, anchor_eligibility):
    t_days = np.asarray(t_days, dtype=float)
    envelope = np.asarray(components['amplitude_envelope'], dtype=float)
    pre_mask = t_days < forskolin_day
    post_mask = ~pre_mask
    diagnostics = {}
    for epoch, mask in [('pre', pre_mask), ('post', post_mask)]:
        epoch_envelope = envelope[mask]
        residual_scale = float(anchor_eligibility[f'{epoch}_residual_robust_sd'])
        ratio = epoch_envelope / max(residual_scale, EPS)
        diagnostics[f'variant_{epoch}_envelope_median'] = float(np.median(epoch_envelope))
        diagnostics[f'variant_{epoch}_envelope_p10'] = float(np.percentile(epoch_envelope, 10))
        diagnostics[f'variant_{epoch}_median_envelope_to_anchor_residual'] = float(np.median(ratio))
        diagnostics[f'variant_{epoch}_observable_fraction'] = float(np.mean(ratio >= 1.0))
    transition_mask = np.abs(t_days - forskolin_day) <= 0.25
    transition_envelope = float(np.median(envelope[transition_mask]))
    transition_residual = max(float(anchor_eligibility['pre_residual_robust_sd']), float(anchor_eligibility['post_residual_robust_sd']), EPS)
    diagnostics['variant_transition_envelope_to_anchor_residual'] = transition_envelope / transition_residual
    diagnostics['variant_acute_transient_to_transition_envelope'] = abs(float(params[9])) / max(transition_envelope, EPS)
    return diagnostics

def safe_token(value):
    value = re.sub('\\s+', '', str(value).strip())
    return re.sub('[\\\\/:*?"<>|]+', '_', value)

def save_trace_batch(trace_batch, output_path, write_header):
    pd.concat(trace_batch, ignore_index=True).to_csv(output_path, mode='w' if write_header else 'a', header=write_header, index=False)

def compress_trace_file(source_path, destination_path):
    with source_path.open('rb') as source, destination_path.open('wb') as raw_destination, gzip.GzipFile(filename='', mode='wb', compresslevel=6, fileobj=raw_destination, mtime=0) as destination:
        shutil.copyfileobj(source, destination)
    source_path.unlink()

def plot_segmented(ax, x, y, **kwargs):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    segment = np.cumsum(np.r_[True, np.diff(x) > 21.0 / 1440.0 + EPS])
    label = kwargs.pop('label', None)
    for index, segment_id in enumerate(np.unique(segment)):
        mask = segment == segment_id
        ax.plot(x[mask], y[mask], label=label if index == 0 else '_nolegend_', **kwargs)

def save_figure(fig, output_path, dpi):
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

def plot_one_trace(trace, truth, output_path):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    x = trace['time_days_zero']
    y = trace['synthetic_signal_normalised']
    centre = trace['baseline_component'] + trace['linear_drift_component'] + trace['acute_transient_component'] + trace['centering_component']
    env = trace['amplitude_envelope']
    plot_segmented(ax, x, y, color='#1f4e79', lw=1.1, label='Synthetic signal')
    plot_segmented(ax, x, centre + env, color='#d46a4c', lw=0.8, alpha=0.75)
    plot_segmented(ax, x, centre - env, color='#d46a4c', lw=0.8, alpha=0.75, label='Ground-truth envelope')
    ax.axvline(truth['forskolin_day_zero'], color='black', ls='--', lw=0.9, label='Forskolin')
    ax.set_xlabel('Time from retained fit start (days)')
    ax.set_ylabel('Normalised signal')
    ax.set_title(f"{truth['synthetic_id']} | {truth['tissue']} | anchor {truth['anchor_recording_id']}")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    save_figure(fig, output_path, 300)

def make_15_panel_figure(selected_traces, selected_truth, output_path):
    fig, axes = plt.subplots(5, 3, figsize=(12, 15), sharex=False)
    axes = axes.ravel()
    for ax, truth in zip(axes, selected_truth):
        trace = selected_traces[truth['synthetic_id']]
        x = trace['time_days_zero']
        y = trace['synthetic_signal_normalised']
        plot_segmented(ax, x, y, color='#1f4e79', lw=0.75)
        ax.axvline(truth['forskolin_day_zero'], color='#b33a3a', ls='--', lw=0.7)
        ax.set_title(f"{truth['synthetic_id']} ({truth['tissue']})", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.supxlabel('Time from retained fit start (days)', fontsize=10)
    fig.supylabel('Normalised signal', fontsize=10)
    fig.tight_layout()
    save_figure(fig, output_path, 200)

def main():
    parameters = pd.read_csv(PARAMETER_FILE)
    traces = pd.read_csv(TRACE_FILE, low_memory=False)
    parameters = exclude_unstable_anchors(parameters)
    if len(parameters) != 53:
        raise ValueError(f'Expected 53 retained anchors, found {len(parameters)}.')
    trace_groups = {uid: group.sort_values('time_days_zero').copy() for uid, group in traces.groupby('recording_uid', sort=False)}
    anchor_eligibility = assess_anchor_eligibility(parameters, trace_groups)
    eligibility_lookup = anchor_eligibility.set_index('anchor_recording_uid', drop=False)
    status_columns = ['pre_period_status', 'post_period_status', 'pre_amplitude_envelope_status', 'post_amplitude_envelope_status', 'phase_shift_status']
    donor_uids = set(anchor_eligibility.loc[~anchor_eligibility[status_columns].eq('ineligible').any(axis=1), 'anchor_recording_uid'])
    neighbour_lookup = build_neighbour_lookup(parameters, donor_uids)
    parameter_lookup = parameters.set_index('recording_uid', drop=False)
    TABLE_FOLDER.mkdir(parents=True, exist_ok=True)
    FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)
    ground_truth_path = TABLE_FOLDER / 'synthetic_ground_truth_parameters.csv'
    uncompressed_traces_path = TABLE_FOLDER / 'synthetic_traces_long_format.csv'
    traces_path = TABLE_FOLDER / 'synthetic_traces_long_format.csv.gz'
    design_path = TABLE_FOLDER / 'synthetic_design_summary.csv'
    eligibility_path = TABLE_FOLDER / 'anchor_target_eligibility.csv'
    anchor_eligibility.to_csv(eligibility_path, index=False)
    for trace_path in [uncompressed_traces_path, traces_path]:
        if trace_path.exists():
            trace_path.unlink()
    design_rng = np.random.default_rng(RANDOM_SEED)
    figure_rng = np.random.default_rng(RANDOM_SEED + 1)
    gt_rows = []
    trace_batch = []
    trace_header_needed = True
    selected_traces = {}
    total_expected = len(parameters) * N_SYNTHETIC_PER_ANCHOR
    n_select = min(N_FIGURE_TRACES, total_expected)
    figure_global_numbers = set(figure_rng.choice(np.arange(1, total_expected + 1), size=n_select, replace=False).tolist())
    global_number = 0
    for anchor_index, anchor_row in enumerate(parameters.itertuples(index=False), start=1):
        anchor_uid = anchor_row.recording_uid
        anchor_theta = np.array([getattr(anchor_row, col) for col in PARAMETER_COLUMNS], dtype=float)
        anchor_trace = trace_groups[anchor_uid]
        t_zero = anchor_trace['time_days_zero'].to_numpy(dtype=float)
        t_original = anchor_trace['time_days_original'].to_numpy(dtype=float)
        reference_start_original = float(anchor_row.reference_window_start_day_original)
        reference_end_original = float(anchor_row.reference_window_end_day_original)
        reference_mask = (t_original >= reference_start_original) & (t_original < reference_end_original)
        duration_days = float(anchor_row.duration_days)
        forskolin_zero = float(anchor_row.forskolin_day_zero)
        drift_reference_original = float(anchor_row.drift_reference_day_original)
        drift_reference_zero = float(anchor_row.drift_reference_day_zero)
        initial_phase = float(anchor_row.profiled_initial_phase_radians)
        anchor_labels = eligibility_lookup.loc[anchor_uid]
        variant_schedule = make_balanced_variant_schedule(neighbour_uids=neighbour_lookup[anchor_uid], n_local_variants=N_SYNTHETIC_PER_ANCHOR - 1, rng=design_rng)
        for variant_number in range(N_SYNTHETIC_PER_ANCHOR):
            global_number += 1
            synthetic_id = f'SYN_A{anchor_index:03d}_{safe_token(anchor_row.recording_id)}_V{variant_number:03d}'
            if variant_number == 0:
                neighbour_uid = anchor_uid
                interpolation_fraction = 0.0
                theta = anchor_theta.copy()
                variant_type = 'anchor_exact_fit'
            else:
                neighbour_uid, interpolation_fraction = variant_schedule[variant_number - 1]
                neighbour_uid = str(neighbour_uid)
                neighbour_row = parameter_lookup.loc[neighbour_uid]
                neighbour_theta = neighbour_row[PARAMETER_COLUMNS].to_numpy(dtype=float)
                interpolation_fraction = float(interpolation_fraction)
                theta = make_local_variant(anchor_theta, neighbour_theta, interpolation_fraction)
                variant_type = 'local_same_tissue_variant'
            components = deterministic_model(t_days=t_zero, forskolin_day=forskolin_zero, duration_days=duration_days, drift_reference_day=drift_reference_zero, parameters=theta, initial_phase=initial_phase, reference_mask=reference_mask)
            signal = components['synthetic_signal_normalised']
            if variant_number == 0 and (not np.allclose(signal, anchor_trace['fitted_signal'].to_numpy(dtype=float), rtol=0.0, atol=1e-10)):
                raise RuntimeError(f'V000 does not reproduce {anchor_uid}')
            truth = {'synthetic_id': synthetic_id, 'model_spec_version': MODEL_SPEC_VERSION, 'global_synthetic_number': global_number, 'anchor_index': anchor_index, 'variant_number': variant_number, 'variant_type': variant_type, 'is_exact_anchor_fit': variant_number == 0, 'anchor_recording_uid': anchor_uid, 'anchor_recording_id': anchor_row.recording_id, 'tissue': anchor_row.tissue, 'source_neighbour_recording_uid': neighbour_uid, 'interpolation_fraction': interpolation_fraction, 'random_seed': RANDOM_SEED, 'residual_noise_added': False, 'n_time_points': len(t_zero), 'duration_days': duration_days, 'fit_start_day_original': float(anchor_row.fit_start_day_original), 'forskolin_day_original': float(anchor_row.forskolin_day_original), 'forskolin_day_zero': forskolin_zero, 'drift_reference_day_original': drift_reference_original, 'drift_reference_day_zero': drift_reference_zero, 'reference_window_start_day_original': reference_start_original, 'reference_window_end_day_original': reference_end_original, 'reference_window_n_points': int(reference_mask.sum()), 'profiled_initial_phase_radians': initial_phase, 'transition_duration_days': TRANSITION_DURATION_DAYS, 'acute_transient_shape': 'causal_half_gaussian', 'acute_transient_sigma_hours': ACUTE_TRANSIENT_SIGMA_HOURS}
            truth.update(dict(zip(PARAMETER_COLUMNS, theta)))
            truth['fit_phase_shift_hours'] = phase_angle_to_hours(theta[7], theta[3])
            truth['derived_pre_envelope_final_fraction'] = float(np.exp(theta[4] * forskolin_zero))
            truth['derived_post_envelope_final_fraction'] = float(np.exp(theta[5] * (duration_days - forskolin_zero)))
            for column in ['pre_period_status', 'pre_period_reason', 'post_period_status', 'post_period_reason', 'pre_amplitude_envelope_status', 'pre_amplitude_envelope_reason', 'post_amplitude_envelope_status', 'post_amplitude_envelope_reason', 'phase_shift_status', 'phase_shift_reason']:
                truth[column] = anchor_labels[column]
            truth.update(variant_diagnostics(t_days=t_zero, forskolin_day=forskolin_zero, params=theta, components=components, anchor_eligibility=anchor_labels))
            gt_rows.append(truth)
            trace_out = pd.DataFrame({'synthetic_id': synthetic_id, 'model_spec_version': MODEL_SPEC_VERSION, 'anchor_index': anchor_index, 'variant_number': variant_number, 'anchor_recording_uid': anchor_uid, 'anchor_recording_id': anchor_row.recording_id, 'tissue': anchor_row.tissue, 'time_days_original': t_original, 'time_days_zero': t_zero, 'forskolin_day_zero': forskolin_zero, 'synthetic_signal_normalised': signal, 'amplitude_envelope': components['amplitude_envelope'], 'phase_radians': components['phase_radians']})
            trace_batch.append(trace_out)
            if global_number in figure_global_numbers:
                selected_trace = trace_out.copy()
                for column in ['baseline_component', 'linear_drift_component', 'acute_transient_component', 'centering_component']:
                    selected_trace[column] = components[column]
                selected_traces[synthetic_id] = selected_trace
            if len(trace_batch) >= WRITE_BATCH_SIZE:
                save_trace_batch(trace_batch, uncompressed_traces_path, write_header=trace_header_needed)
                trace_header_needed = False
                trace_batch = []
    if trace_batch:
        save_trace_batch(trace_batch, uncompressed_traces_path, write_header=trace_header_needed)
    compress_trace_file(uncompressed_traces_path, traces_path)
    ground_truth = pd.DataFrame(gt_rows)
    ground_truth.to_csv(ground_truth_path, index=False)
    design = ground_truth.groupby(['anchor_index', 'anchor_recording_uid', 'anchor_recording_id', 'tissue', 'pre_period_status', 'post_period_status', 'pre_amplitude_envelope_status', 'post_amplitude_envelope_status', 'phase_shift_status'], as_index=False).agg(n_synthetic=('synthetic_id', 'size'), n_exact_anchor=('is_exact_anchor_fit', 'sum'), mean_interpolation_fraction=('interpolation_fraction', 'mean'), max_interpolation_fraction=('interpolation_fraction', 'max'))
    design.to_csv(design_path, index=False)
    expected = len(parameters) * N_SYNTHETIC_PER_ANCHOR
    if expected != 530:
        raise RuntimeError(f'Formal Step 2B must generate 530 sources, not {expected}.')
    if len(ground_truth) != expected:
        raise RuntimeError(f'Expected {expected} traces, got {len(ground_truth)}.')
    selected_truth = [row for row in gt_rows if row['synthetic_id'] in selected_traces]
    for truth in selected_truth:
        plot_one_trace(selected_traces[truth['synthetic_id']], truth, FIGURE_FOLDER / f"{truth['synthetic_id']}.png")
    make_15_panel_figure(selected_traces, selected_truth, FIGURE_FOLDER / '00_all_15_synthetic_examples.png')
if __name__ == '__main__':
    main()
