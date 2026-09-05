from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from msc_deterministic_model_common import PARAMETER_COLUMNS, default_reference_mask, deterministic_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEP3A_TABLES = PROJECT_ROOT / 'results' / 'step3a' / 'tables'
STEP3B_TABLES = PROJECT_ROOT / 'results' / 'step3b' / 'tables'
STEP3C_TABLES = PROJECT_ROOT / 'results' / 'step3c' / 'tables'
STEP3A_FIGURES = PROJECT_ROOT / 'results' / 'step3a' / 'figures'
STEP3B_FIGURES = PROJECT_ROOT / 'results' / 'step3b' / 'figures'
STEP3C_FIGURES = PROJECT_ROOT / 'results' / 'step3c' / 'figures'
METHODS = ('CWT_pyBOAT', 'Hilbert', 'STFT')
METHOD_LABELS = {'CWT_pyBOAT': 'CWT', 'Hilbert': 'Hilbert', 'STFT': 'STFT'}
METHOD_COLOURS = {'CWT_pyBOAT': '#3973B7', 'Hilbert': '#E07A3F', 'STFT': '#2A9D67'}
METHOD_MARKERS = {'CWT_pyBOAT': 'o', 'Hilbert': 's', 'STFT': '^'}
METHOD_OFFSETS = {'CWT_pyBOAT': -0.18, 'Hilbert': 0.0, 'STFT': 0.18}
TARGETS = (
    'pre_period_hours',
    'post_period_hours',
    'pre_amplitude_trajectory',
    'post_amplitude_trajectory',
    'phase_shift_hours',
)
TARGET_TITLES = {
    'pre_period_hours': 'Pre-treatment period',
    'post_period_hours': 'Post-treatment period',
    'pre_amplitude_trajectory': 'Pre-treatment amplitude trajectory',
    'post_amplitude_trajectory': 'Post-treatment amplitude trajectory',
    'phase_shift_hours': 'Phase shift',
}
TARGET_SHORT = {
    'pre_period_hours': 'Pre period',
    'post_period_hours': 'Post period',
    'pre_amplitude_trajectory': 'Pre amplitude',
    'post_amplitude_trajectory': 'Post amplitude',
    'phase_shift_hours': 'Phase',
}
TARGET_YLABELS = {
    'pre_period_hours': 'Absolute error (h)',
    'post_period_hours': 'Absolute error (h)',
    'pre_amplitude_trajectory': 'Calibrated absolute NRMSE',
    'post_amplitude_trajectory': 'Calibrated absolute NRMSE',
    'phase_shift_hours': 'Circular absolute error (rad)',
}
COMPONENT_LABELS = {
    'target_only': 'Target only',
    'pre_amplitude': '+ Pre amplitude',
    'post_amplitude': '+ Post amplitude',
    'pre_period': '+ Pre period',
    'post_period': '+ Post period',
    'pre_envelope': '+ Pre envelope',
    'post_envelope': '+ Post envelope',
    'linear_drift': '+ Linear drift',
    'baseline_step': '+ Baseline step',
    'acute_treatment_transient': '+ Acute transient',
}
CHALLENGES = ('gaussian_noise', 'missing_block_6h', 'transient_artifact')
CHALLENGE_LABELS = {
    'gaussian_noise': 'Gaussian noise',
    'missing_block_6h': '6 h missing block',
    'transient_artifact': 'Transient artifact',
}
SAMPLING_INTERVAL_HOURS = 20.0 / 60.0
REFERENCE_AMPLITUDE = 2.0

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'font.weight': 'semibold',
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'legend.fontsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

def method_handles():
    return [
        Line2D([0], [0], marker=METHOD_MARKERS[method], color=METHOD_COLOURS[method], linestyle='none', markersize=8, label=METHOD_LABELS[method])
        for method in METHODS
    ]

def style_axis(axis):
    axis.set_axisbelow(True)
    axis.grid(axis='y', color='#D8D8D8', linewidth=0.9, alpha=0.8)
    axis.tick_params(width=1.2, length=4)
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontweight('semibold')
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_linewidth(1.2)
    axis.spines['bottom'].set_linewidth(1.2)

def save_figure(figure, folder, stem):
    folder.mkdir(parents=True, exist_ok=True)
    figure.savefig(folder / f'{stem}.png', dpi=600, bbox_inches='tight', facecolor='white')
    figure.savefig(folder / f'{stem}.pdf', bbox_inches='tight', facecolor='white')
    figure.savefig(folder / f'{stem}.svg', bbox_inches='tight', facecolor='white')
    plt.close(figure)

def figure_axes(figsize):
    figure = plt.figure(figsize=figsize)
    grid = figure.add_gridspec(3, 2, height_ratios=(1.0, 1.0, 1.08), hspace=0.55, wspace=0.25)
    return figure, [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
        figure.add_subplot(grid[2, :]),
    ]

def figure_axes_equal_bottom(figsize):
    figure = plt.figure(figsize=figsize)
    grid = figure.add_gridspec(3, 4, hspace=0.58, wspace=0.50)
    return figure, [
        figure.add_subplot(grid[0, 0:2]),
        figure.add_subplot(grid[0, 2:4]),
        figure.add_subplot(grid[1, 0:2]),
        figure.add_subplot(grid[1, 2:4]),
        figure.add_subplot(grid[2, 1:3]),
    ]

def errorbar(axis, positions, values, lower, upper, method):
    values = np.asarray(values, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    valid = np.isfinite(values) & np.isfinite(lower) & np.isfinite(upper)
    if not np.any(valid):
        return
    axis.errorbar(
        np.asarray(positions, dtype=float)[valid],
        values[valid],
        yerr=np.vstack((np.maximum(0.0, values[valid] - lower[valid]), np.maximum(0.0, upper[valid] - values[valid]))),
        fmt=METHOD_MARKERS[method],
        color=METHOD_COLOURS[method],
        markerfacecolor=METHOD_COLOURS[method],
        markeredgecolor='white',
        markeredgewidth=0.8,
        markersize=8.0,
        capsize=4.5,
        capthick=1.4,
        elinewidth=1.5,
        linestyle='none',
        zorder=3,
    )

def plot_step3a():
    table = pd.read_csv(STEP3A_TABLES / 'scenario_bootstrap_summary.csv')
    table = table[
        table['target_status'].eq('eligible')
        & table['method_applicable'].eq(True)
        & table['metric'].eq('primary_error')
    ].copy()
    figure, axes = figure_axes((15.5, 13.0))
    for letter, family, axis in zip('ABCDE', TARGETS, axes):
        selected = table[table['scenario_family'].eq(family)].copy()
        levels = selected[['scenario_level_number', 'added_component']].drop_duplicates().sort_values('scenario_level_number')
        x = np.arange(len(levels), dtype=float)
        level_positions = dict(zip(levels['scenario_level_number'], x))
        for method in METHODS:
            rows = selected[selected['method'].eq(method)].sort_values('scenario_level_number')
            positions = [level_positions[value] + METHOD_OFFSETS[method] for value in rows['scenario_level_number']]
            errorbar(axis, positions, rows['point_estimate'], rows['ci_lower_95'], rows['ci_upper_95'], method)
        axis.set_xticks(x, [COMPONENT_LABELS.get(value, str(value).replace('_', ' ').title()) for value in levels['added_component']], rotation=28, ha='right')
        axis.set_ylabel(TARGET_YLABELS[family])
        axis.set_title(f'{letter}  {TARGET_TITLES[family]}', loc='left', fontweight='bold')
        axis.set_ylim(bottom=0)
        style_axis(axis)
    figure.legend(handles=method_handles(), loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False)
    figure.subplots_adjust(top=0.94, bottom=0.08)
    save_figure(figure, STEP3A_FIGURES, 'STEP3A_combined_benchmark')

def plot_step3b():
    table = pd.read_csv(STEP3B_TABLES / 'step3b_anchor_bootstrap_summary.csv')
    table = table[
        table['target_status'].eq('eligible')
        & table['method_applicable'].eq(True)
        & table['metric'].eq('primary_error')
    ].copy()
    figure, axes = figure_axes_equal_bottom((19.5, 13.0))
    x = np.arange(len(CHALLENGES), dtype=float)
    for letter, family, axis in zip('ABCDE', TARGETS, axes):
        selected = table[table['scenario_family'].eq(family)].copy()
        for method in METHODS:
            rows = selected[selected['method'].eq(method)].set_index('challenge_type').reindex(CHALLENGES)
            positions = x + METHOD_OFFSETS[method]
            errorbar(axis, positions, rows['deterioration_point_estimate'], rows['deterioration_ci_lower_95'], rows['deterioration_ci_upper_95'], method)
        axis.axhline(0.0, color='#444444', linewidth=0.9, zorder=1)
        axis.set_xticks(x, [CHALLENGE_LABELS[value] for value in CHALLENGES], rotation=20, ha='right')
        axis.set_ylabel(f'Paired deterioration\n{TARGET_YLABELS[family]}')
        axis.set_title(f'{letter}  {TARGET_TITLES[family]}', loc='left', fontweight='bold')
        style_axis(axis)
    figure.legend(handles=method_handles(), loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False)
    figure.subplots_adjust(top=0.94, bottom=0.07)
    save_figure(figure, STEP3B_FIGURES, 'STEP3B_combined_robustness')

def time_grid(row):
    interval = SAMPLING_INTERVAL_HOURS / 24.0
    return np.arange(0.0, float(row['duration_days']) + 0.5 * interval, interval)

def model_components(row, time_days):
    parameters = {column: float(row[column]) for column in PARAMETER_COLUMNS}
    return deterministic_model(
        t_days=time_days,
        forskolin_day=float(row['forskolin_day_zero']),
        duration_days=float(row['duration_days']),
        drift_reference_day=float(row['drift_reference_day_zero']),
        parameters=parameters,
        initial_phase=float(row['profiled_initial_phase_radians']),
    )

def closest_to_median(table, columns):
    values = table[list(columns)].apply(pd.to_numeric, errors='coerce')
    valid = values.notna().all(axis=1)
    candidates = table.loc[valid].copy()
    values = values.loc[valid]
    centre = values.median()
    scale = values.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    distance = (((values - centre) / scale) ** 2).sum(axis=1)
    return candidates.loc[distance.idxmin()]

def representative_rows(envelope_truth, artifact_truth):
    envelope = envelope_truth[envelope_truth['target_status'].eq('eligible')].copy()
    preferred_envelope = envelope[envelope['plan_id'].eq('ENV_CWT_POST_PERIOD')]
    if not preferred_envelope.empty:
        envelope = preferred_envelope
    envelope = envelope.drop_duplicates('source_synthetic_id')
    envelope_row = closest_to_median(envelope, ('fit_post_envelope_log_rate_per_day',))
    artifact = artifact_truth[artifact_truth['target_status'].eq('eligible')].copy()
    preferred_artifact = artifact[artifact['plan_id'].eq('ART_CWT_POST_PERIOD')]
    if not preferred_artifact.empty:
        artifact = preferred_artifact
    artifact = artifact.drop_duplicates('source_synthetic_id')
    artifact_row = closest_to_median(artifact, ('artifact_fwhm_hours', 'artifact_amplitude_factor'))
    return envelope_row, artifact_row

def clean_row(truth, row):
    selected = truth[
        truth['source_synthetic_id'].astype(str).eq(str(row['source_synthetic_id']))
        & truth['scenario_family'].eq(row['scenario_family'])
        & truth['added_component'].eq('target_only')
    ]
    return selected.iloc[0]

def envelope_normalised_signal(row, time_days, components):
    envelope = np.asarray(components['amplitude_envelope'], dtype=float)
    signal = (
        np.asarray(components['baseline_component'], dtype=float)
        + np.asarray(components['linear_drift_component'], dtype=float)
        + np.asarray(components['acute_transient_component'], dtype=float)
        + np.asarray(components['rhythmic_component'], dtype=float) / envelope * REFERENCE_AMPLITUDE
    )
    reference = default_reference_mask(time_days, float(row['forskolin_day_zero']))
    return signal - np.median(signal[reference])

def artifact_signal(row, time_days, clean_signal):
    time_hours = time_days * 24.0
    sigma = float(row['artifact_fwhm_hours']) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    distance = (time_hours - float(row['artifact_center_hour'])) / sigma
    return clean_signal + float(row['artifact_signed_amplitude']) * np.exp(-0.5 * distance ** 2)

def trace_axis(axis, title, xlabel, ylabel=True):
    axis.set_title(title, loc='left', fontweight='bold')
    axis.set_xlabel(xlabel)
    if ylabel:
        axis.set_ylabel('Normalised luminescence')
    style_axis(axis)

def trace_legend(axis, legend_axis):
    handles, labels = axis.get_legend_handles_labels()
    legend_axis.legend(handles, labels, loc='upper left', frameon=False, borderaxespad=0.0, handlelength=2.2)
    legend_axis.set_axis_off()

def plot_rescue_summary(axis, table, experiment, letter, title):
    selected = table[
        table['target_status'].eq('eligible')
        & table['rescue_experiment'].eq(experiment)
    ].copy()
    selected['relative_remaining'] = selected['delta_remaining_point_estimate'] / selected['delta_add_point_estimate']
    selected = selected[np.isfinite(selected['relative_remaining'])].copy()
    selected['family_order'] = selected['scenario_family'].map({value: index for index, value in enumerate(TARGETS)})
    selected['method_order'] = selected['method'].map({value: index for index, value in enumerate(METHODS)})
    selected = selected.sort_values(['family_order', 'method_order'])
    axis.axvline(0.0, color='#222222', linewidth=1.0)
    axis.axvline(1.0, color='#777777', linewidth=1.0, linestyle='--')
    if selected.empty:
        axis.text(0.5, 0.5, 'No estimable eligible results', transform=axis.transAxes, ha='center', va='center')
        axis.set_yticks([])
    else:
        y = np.arange(len(selected), dtype=float)
        for position, row in zip(y, selected.itertuples(index=False)):
            value = float(row.relative_remaining)
            colour = METHOD_COLOURS[row.method]
            axis.hlines(position, min(0.0, value), max(0.0, value), color=colour, linewidth=1.8, alpha=0.7)
            axis.scatter(value, position, color=colour, marker=METHOD_MARKERS[row.method], s=68, edgecolor='white', linewidth=0.8, zorder=3)
        labels = [f"{METHOD_LABELS[row.method]} · {TARGET_SHORT[row.scenario_family]}" for row in selected.itertuples(index=False)]
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        values = selected['relative_remaining'].to_numpy(dtype=float)
        lower = min(-0.1, float(np.nanmin(values)) - 0.1)
        upper = max(1.1, float(np.nanmax(values)) + 0.1)
        axis.set_xlim(lower, upper)
    axis.set_xlabel('Relative remaining degradation\n(0 = clean; 1 = unrescued)')
    axis.set_title(f'{letter}  {title}', loc='left', fontweight='bold')
    style_axis(axis)

def plot_step3c():
    summary = pd.read_csv(STEP3C_TABLES / 'step3c_anchor_bootstrap_summary.csv')
    step3a_truth = pd.read_csv(STEP3A_TABLES / 'scenario_ground_truth_parameters.csv', low_memory=False)
    envelope_truth = pd.read_csv(STEP3C_TABLES / 'step3c_envelope_corrected_truth.csv.gz', low_memory=False)
    artifact_truth = pd.read_csv(STEP3C_TABLES / 'step3c_artifact_corrected_truth.csv.gz', low_memory=False)
    envelope_row, artifact_row = representative_rows(envelope_truth, artifact_truth)
    envelope_clean_row = clean_row(step3a_truth, envelope_row)
    envelope_time = time_grid(envelope_row)
    envelope_components = model_components(envelope_row, envelope_time)
    envelope_clean = model_components(envelope_clean_row, envelope_time)['synthetic_signal_normalised']
    envelope_challenge = envelope_components['synthetic_signal_normalised']
    envelope_corrected = envelope_normalised_signal(envelope_row, envelope_time, envelope_components)
    envelope_x = (envelope_time - float(envelope_row['forskolin_day_zero']))
    artifact_time = time_grid(artifact_row)
    artifact_clean = model_components(artifact_row, artifact_time)['synthetic_signal_normalised']
    artifact_challenge = artifact_signal(artifact_row, artifact_time, artifact_clean)
    artifact_x = artifact_time * 24.0 - float(artifact_row['artifact_center_hour'])
    mask_start = float(artifact_row['mask_start_hour']) - float(artifact_row['artifact_center_hour'])
    mask_end = float(artifact_row['mask_end_hour']) - float(artifact_row['artifact_center_hour'])
    artifact_masked = artifact_challenge.copy()
    artifact_masked[(artifact_x >= mask_start) & (artifact_x <= mask_end)] = np.nan
    figure = plt.figure(figsize=(22.0, 14.5))
    grid = figure.add_gridspec(3, 2, height_ratios=(1.0, 1.0, 1.25), hspace=0.44, wspace=0.38)
    axes = []
    legend_axes = []
    for row in range(2):
        for column in range(2):
            cell = grid[row, column].subgridspec(1, 2, width_ratios=(1.0, 0.48), wspace=0.05)
            axes.append(figure.add_subplot(cell[0, 0]))
            legend_axes.append(figure.add_subplot(cell[0, 1]))
    axes.extend((figure.add_subplot(grid[2, 0]), figure.add_subplot(grid[2, 1])))
    axes[0].plot(envelope_x, envelope_clean, color='#888888', linewidth=1.5, alpha=0.75, label='Clean reference (A)')
    axes[0].plot(envelope_x, envelope_challenge, color='#D95F59', linewidth=1.7, label='With envelope (B)')
    envelope_band = np.asarray(envelope_components['amplitude_envelope'], dtype=float)
    axes[0].fill_between(envelope_x, -envelope_band, envelope_band, color='#E6A15A', alpha=0.12, linewidth=0, label='Ground-truth envelope')
    axes[0].axvline(0.0, color='#555555', linewidth=0.9, linestyle='--')
    trace_legend(axes[0], legend_axes[0])
    trace_axis(axes[0], 'A  Envelope challenge', 'Time relative to forskolin (days)')
    axes[1].plot(envelope_x, envelope_clean, color='#888888', linewidth=1.5, alpha=0.75, label='Clean reference (A)')
    axes[1].plot(envelope_x, envelope_corrected, color='#3973B7', linewidth=1.7, label='Oracle-normalised (C)')
    axes[1].axvline(0.0, color='#555555', linewidth=0.9, linestyle='--')
    trace_legend(axes[1], legend_axes[1])
    trace_axis(axes[1], 'B  Known-envelope normalization', 'Time relative to forskolin (days)', False)
    local = (artifact_x >= -18.0) & (artifact_x <= 18.0)
    axes[2].plot(artifact_x[local], artifact_clean[local], color='#888888', linewidth=1.5, alpha=0.75, label='Clean reference (A)')
    axes[2].plot(artifact_x[local], artifact_challenge[local], color='#D95F59', linewidth=1.7, label='With transient (B)')
    axes[2].axvline(0.0, color='#555555', linewidth=0.9, linestyle='--')
    trace_legend(axes[2], legend_axes[2])
    trace_axis(axes[2], 'C  Transient challenge', 'Time relative to transient centre (h)')
    axes[3].plot(artifact_x[local], artifact_clean[local], color='#888888', linewidth=1.5, alpha=0.75, label='Clean reference (A)')
    axes[3].plot(artifact_x[local], artifact_masked[local], color='#3973B7', linewidth=1.7, label='Retained observations (C)')
    axes[3].axvspan(mask_start, mask_end, color='#D95F59', alpha=0.18, label='Oracle mask')
    trace_legend(axes[3], legend_axes[3])
    trace_axis(axes[3], 'D  Known-transient masking', 'Time relative to transient centre (h)', False)
    plot_rescue_summary(axes[4], summary, 'envelope', 'E', 'Envelope rescue summary')
    plot_rescue_summary(axes[5], summary, 'artifact', 'F', 'Transient rescue summary')
    figure.subplots_adjust(top=0.98, bottom=0.06)
    save_figure(figure, STEP3C_FIGURES, 'STEP3C_combined_rescue')

def main():
    plot_step3a()
    plot_step3b()
    plot_step3c()

if __name__ == '__main__':
    main()
