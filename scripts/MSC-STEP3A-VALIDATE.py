from importlib import import_module
from pathlib import Path
import numpy as np
import pandas as pd
from msc_deterministic_model_common import MODEL_SPEC_VERSION

common = import_module('MSC-STEP3A-COMMON')
DEFAULT_OUTPUT_FOLDER = common.DEFAULT_OUTPUT_FOLDER
DEFAULT_TRUTH_FILE = common.DEFAULT_TRUTH_FILE
SCALAR_TARGETS = {
    'pre_period_hours': ('fit_pre_period_hours', 'estimated_pre_period_hours'),
    'post_period_hours': ('fit_post_period_hours', 'estimated_post_period_hours'),
    'phase_shift_hours': ('fit_phase_shift_angle_radians', 'estimated_phase_shift_angle_radians'),
}
AMPLITUDE_TARGETS = {
    'pre_amplitude_trajectory': {'absolute_nrmse': 'pre_amplitude_absolute_nrmse', 'shape_nrmse': 'pre_amplitude_trajectory_scaled_nrmse', 'pearson': 'pre_amplitude_trajectory_pearson'},
    'post_amplitude_trajectory': {'absolute_nrmse': 'post_amplitude_absolute_nrmse', 'shape_nrmse': 'post_amplitude_trajectory_scaled_nrmse', 'pearson': 'post_amplitude_trajectory_pearson'},
}
EXPECTED_SCENARIOS_PER_FAMILY = {'pre_period_hours': 4, 'post_period_hours': 6, 'pre_amplitude_trajectory': 3, 'post_amplitude_trajectory': 5, 'phase_shift_hours': 10}
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260902
ANCHOR_GROUP_COLUMNS = ['method', 'scenario_family', 'scenario_level_number', 'scenario_level', 'added_component', 'target_status', 'tissue', 'anchor_recording_uid', 'anchor_recording_id', 'method_applicable', 'primary_metric_name', 'primary_metric_unit']
GROUP_COLUMNS = ['method', 'scenario_family', 'scenario_level_number', 'scenario_level', 'added_component', 'target_status', 'method_applicable', 'primary_metric_name', 'primary_metric_unit']
BOOTSTRAP_METRICS = {
    'median_primary_error': 'primary_error',
    'median_primary_error_deterioration': 'primary_error_deterioration_from_target_only',
    'median_trajectory_pearson': 'trajectory_pearson',
    'median_trajectory_pearson_deterioration': 'trajectory_pearson_deterioration_from_target_only',
    'median_target_segment_shape_nrmse': 'target_segment_shape_nrmse',
    'median_target_segment_shape_nrmse_deterioration': 'target_segment_shape_nrmse_deterioration_from_target_only',
    'median_envelope_log_rate_absolute_error_per_day': 'envelope_log_rate_absolute_error_per_day',
    'median_envelope_log_rate_absolute_error_deterioration': 'envelope_log_rate_absolute_error_deterioration_from_target_only',
    'median_phase_equivalent_absolute_error_hours': 'phase_equivalent_absolute_error_hours',
    'median_phase_equivalent_absolute_error_hours_deterioration': 'phase_equivalent_absolute_error_hours_deterioration_from_target_only',
    'median_ratio_log_error': 'ratio_log_error',
    'median_ratio_log_error_deterioration': 'ratio_log_error_deterioration_from_target_only',
    'processing_success_fraction': 'processing_success_fraction',
    'valid_fraction': 'valid_fraction',
}

def circular_error_radians(estimate, truth):
    estimate = pd.to_numeric(estimate, errors='coerce')
    truth = pd.to_numeric(truth, errors='coerce')
    return (estimate - truth + np.pi) % (2.0 * np.pi) - np.pi

def numeric_flag(values):
    return values.astype(str).str.casefold().map({'true': 1.0, 'false': 0.0})

def validate_truth_design(truth, allow_partial=False):
    required = {'model_spec_version', 'synthetic_id', 'source_synthetic_id', 'anchor_recording_uid', 'scenario_family', 'added_component'}
    if required - set(truth.columns):
        raise ValueError('Truth table is missing required columns.')
    if truth['synthetic_id'].astype(str).duplicated().any():
        raise ValueError('Truth table contains duplicate synthetic IDs.')
    if set(truth['model_spec_version'].dropna().astype(str)) != {MODEL_SPEC_VERSION}:
        raise ValueError('Truth table has an incompatible model specification.')
    counts = truth.groupby('source_synthetic_id')['synthetic_id'].size()
    if not (counts == 28).all():
        raise ValueError('Every source must have exactly 28 scenarios.')
    family_counts = truth.groupby(['source_synthetic_id', 'scenario_family'])['synthetic_id'].size().unstack(fill_value=0).reindex(columns=list(EXPECTED_SCENARIOS_PER_FAMILY), fill_value=0)
    if set(truth['scenario_family'].astype(str)) != set(EXPECTED_SCENARIOS_PER_FAMILY):
        raise ValueError('Truth table has an incorrect scenario family set.')
    if not family_counts.eq(pd.Series(EXPECTED_SCENARIOS_PER_FAMILY), axis='columns').all(axis=None):
        raise ValueError('Scenario family counts are incorrect.')
    target_counts = truth[truth['added_component'].astype(str).eq('target_only')].groupby(['source_synthetic_id', 'scenario_family']).size()
    if len(target_counts) != truth[['source_synthetic_id', 'scenario_family']].drop_duplicates().shape[0] or not (target_counts == 1).all():
        raise ValueError('Each source and family must have one target-only scenario.')
    if not allow_partial and (len(truth) != 14840 or truth['source_synthetic_id'].nunique() != 530 or truth['anchor_recording_uid'].nunique() != 53):
        raise ValueError('Formal Step 3A requires 14,840 scenarios from 530 sources and 53 anchors.')

def load_estimates(estimate_files, truth, allow_partial=False):
    outputs = []
    truth_ids = set(truth['synthetic_id'].astype(str))
    for path in estimate_files:
        table = pd.read_csv(Path(path), low_memory=False)
        required = {'model_spec_version', 'method', 'method_applicable', 'synthetic_id'}
        if table.empty or required - set(table.columns):
            raise ValueError(f'Invalid estimate file: {path}')
        methods = set(table['method'].dropna().astype(str))
        if len(methods) != 1:
            raise ValueError(f'Estimate file must contain one method: {path}')
        method = next(iter(methods))
        if method not in common.METHOD_APPLICABILITY:
            raise ValueError(f'Unknown method in estimate file: {method}')
        if set(table['model_spec_version'].dropna().astype(str)) != {MODEL_SPEC_VERSION}:
            raise ValueError(f'Incompatible model specification: {path}')
        if table['synthetic_id'].astype(str).duplicated().any() or set(table['synthetic_id'].astype(str)) != truth_ids:
            raise ValueError(f'Estimate IDs differ from scenario truth: {path}')
        expected_applicable = truth.set_index(truth['synthetic_id'].astype(str))['scenario_family'].isin(common.METHOD_APPLICABILITY[method]).reindex(table['synthetic_id'].astype(str)).to_numpy(dtype=bool)
        observed_applicable = table['method_applicable'].astype(str).str.casefold().eq('true').to_numpy(dtype=bool)
        if not np.array_equal(expected_applicable, observed_applicable):
            raise ValueError(f'Method applicability is incorrect: {path}')
        if not allow_partial and len(table) != 14840:
            raise ValueError(f'Formal estimate file must contain 14,840 rows: {path}')
        outputs.append(table)
    methods = {str(table['method'].iloc[0]) for table in outputs}
    if not allow_partial and methods != set(common.METHOD_APPLICABILITY):
        raise ValueError('Formal validation requires STFT, CWT_pyBOAT and Hilbert.')
    return pd.concat(outputs, ignore_index=True)

def build_trace_metrics(truth, estimates):
    duplicate_columns = [column for column in estimates.columns if column in truth.columns and column != 'synthetic_id']
    joined = truth.merge(estimates.drop(columns=duplicate_columns), on='synthetic_id', how='left', validate='one_to_many')
    joined['method_applicable'] = joined['method_applicable'].astype(str).str.casefold().eq('true')
    joined['processing_success'] = numeric_flag(joined['processing_success'])
    rows = []
    for family, table in joined.groupby('scenario_family', sort=False):
        selected = table.copy()
        selected['trajectory_pearson'] = np.nan
        selected['target_segment_shape_nrmse'] = np.nan
        selected['envelope_log_rate_error_per_day'] = np.nan
        selected['envelope_log_rate_absolute_error_per_day'] = np.nan
        selected['phase_signed_angle_error_radians'] = np.nan
        selected['phase_absolute_angle_error_radians'] = np.nan
        selected['phase_equivalent_signed_error_hours'] = np.nan
        selected['phase_equivalent_absolute_error_hours'] = np.nan
        if family in SCALAR_TARGETS:
            truth_column, estimate_column = SCALAR_TARGETS[family]
            selected['ground_truth'] = pd.to_numeric(selected[truth_column], errors='coerce')
            selected['estimate'] = pd.to_numeric(selected[estimate_column], errors='coerce')
            if family == 'phase_shift_hours':
                selected['phase_signed_angle_error_radians'] = circular_error_radians(selected['estimate'], selected['ground_truth'])
                selected['phase_absolute_angle_error_radians'] = selected['phase_signed_angle_error_radians'].abs()
                selected['phase_equivalent_signed_error_hours'] = selected['phase_signed_angle_error_radians'] * pd.to_numeric(selected['fit_post_period_hours'], errors='coerce') / (2.0 * np.pi)
                selected['phase_equivalent_absolute_error_hours'] = selected['phase_equivalent_signed_error_hours'].abs()
                selected['signed_error'] = selected['phase_signed_angle_error_radians']
                selected['primary_metric_name'] = 'circular_absolute_error_radians'
                selected['primary_metric_unit'] = 'radians'
            else:
                selected['signed_error'] = selected['estimate'] - selected['ground_truth']
                selected['primary_metric_name'] = 'absolute_error_hours'
                selected['primary_metric_unit'] = 'hours'
            selected['primary_error'] = selected['signed_error'].abs()
        elif family in AMPLITUDE_TARGETS:
            columns = AMPLITUDE_TARGETS[family]
            selected['ground_truth'] = np.nan
            selected['estimate'] = np.nan
            selected['signed_error'] = np.nan
            selected['primary_error'] = pd.to_numeric(selected[columns['absolute_nrmse']], errors='coerce')
            selected['trajectory_pearson'] = pd.to_numeric(selected[columns['pearson']], errors='coerce')
            selected['target_segment_shape_nrmse'] = pd.to_numeric(selected[columns['shape_nrmse']], errors='coerce')
            selected['primary_metric_name'] = 'target_segment_calibrated_absolute_nrmse'
            selected['primary_metric_unit'] = 'NRMSE'
            epoch = 'pre' if family.startswith('pre_') else 'post'
            selected['envelope_log_rate_error_per_day'] = pd.to_numeric(selected[f'estimated_{epoch}_envelope_log_rate_per_day'], errors='coerce') - pd.to_numeric(selected[f'fit_{epoch}_envelope_log_rate_per_day'], errors='coerce')
            selected['envelope_log_rate_absolute_error_per_day'] = selected['envelope_log_rate_error_per_day'].abs()
        else:
            raise ValueError(f'Unsupported scenario family: {family}')
        selected['valid_primary'] = selected['method_applicable'] & np.isfinite(selected['primary_error'])
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)

def paired_degradation(trace_metrics):
    keys = ['method', 'scenario_family', 'source_synthetic_id']
    error_metrics = ['primary_error', 'target_segment_shape_nrmse', 'envelope_log_rate_absolute_error_per_day', 'phase_absolute_angle_error_radians', 'phase_equivalent_absolute_error_hours', 'ratio_log_error']
    goodness_metrics = ['trajectory_pearson']
    signed_metrics = ['envelope_log_rate_error_per_day', 'phase_signed_angle_error_radians', 'phase_equivalent_signed_error_hours', 'signed_ratio_log_error']
    metrics = error_metrics + goodness_metrics + signed_metrics
    reference = trace_metrics[trace_metrics['added_component'].eq('target_only')][keys + metrics].rename(columns={column: f'target_only_{column}' for column in metrics})
    if reference.duplicated(keys).any():
        raise ValueError('More than one target-only reference exists for a source.')
    output = trace_metrics.merge(reference, on=keys, how='left', validate='many_to_one')
    for column in error_metrics:
        output[f'{column}_deterioration_from_target_only'] = output[column] - output[f'target_only_{column}']
    for column in goodness_metrics:
        output[f'{column}_deterioration_from_target_only'] = output[f'target_only_{column}'] - output[column]
    for column in signed_metrics:
        output[f'{column}_change_from_target_only'] = output[column] - output[f'target_only_{column}']
    return output

def failure_summary(trace_metrics):
    table = trace_metrics.copy()
    table['applicable_count'] = table['method_applicable'].astype(int)
    table['successful_applicable'] = table['processing_success'].where(table['method_applicable'])
    table['valid_applicable'] = table['valid_primary'].astype(float).where(table['method_applicable'])
    return table.groupby(['method', 'scenario_family', 'target_status', 'method_applicable'], dropna=False, sort=True).agg(n_rows=('synthetic_id', 'size'), n_applicable=('applicable_count', 'sum'), processing_success_fraction=('successful_applicable', 'mean'), target_metric_valid_fraction=('valid_applicable', 'mean')).reset_index()

def anchor_summary(paired):
    return paired.groupby(ANCHOR_GROUP_COLUMNS, dropna=False, sort=True).agg(
        n_variants=('source_synthetic_id', 'nunique'),
        processing_success_fraction=('processing_success', 'mean'),
        valid_fraction=('valid_primary', 'mean'),
        median_primary_error=('primary_error', 'median'),
        median_primary_error_deterioration=('primary_error_deterioration_from_target_only', 'median'),
        median_trajectory_pearson=('trajectory_pearson', 'median'),
        median_trajectory_pearson_deterioration=('trajectory_pearson_deterioration_from_target_only', 'median'),
        median_target_segment_shape_nrmse=('target_segment_shape_nrmse', 'median'),
        median_target_segment_shape_nrmse_deterioration=('target_segment_shape_nrmse_deterioration_from_target_only', 'median'),
        median_envelope_log_rate_error_per_day=('envelope_log_rate_error_per_day', 'median'),
        median_envelope_log_rate_absolute_error_per_day=('envelope_log_rate_absolute_error_per_day', 'median'),
        median_envelope_log_rate_absolute_error_deterioration=('envelope_log_rate_absolute_error_per_day_deterioration_from_target_only', 'median'),
        median_phase_equivalent_absolute_error_hours=('phase_equivalent_absolute_error_hours', 'median'),
        median_phase_equivalent_absolute_error_hours_deterioration=('phase_equivalent_absolute_error_hours_deterioration_from_target_only', 'median'),
        median_ratio_log_error=('ratio_log_error', 'median'),
        median_ratio_log_error_deterioration=('ratio_log_error_deterioration_from_target_only', 'median'),
    ).reset_index()

def group_summary(anchor_level):
    return anchor_level.groupby(GROUP_COLUMNS, dropna=False, sort=True).agg(
        n_anchors_total=('anchor_recording_uid', 'nunique'),
        n_anchors_valid=('median_primary_error', lambda x: int(np.isfinite(pd.to_numeric(x, errors='coerce')).sum())),
        median_processing_success_fraction=('processing_success_fraction', 'median'),
        median_valid_fraction=('valid_fraction', 'median'),
        median_primary_error=('median_primary_error', 'median'),
        q25_primary_error=('median_primary_error', lambda x: x.quantile(0.25)),
        q75_primary_error=('median_primary_error', lambda x: x.quantile(0.75)),
        median_primary_error_deterioration=('median_primary_error_deterioration', 'median'),
        median_trajectory_pearson=('median_trajectory_pearson', 'median'),
        median_trajectory_pearson_deterioration=('median_trajectory_pearson_deterioration', 'median'),
        median_target_segment_shape_nrmse=('median_target_segment_shape_nrmse', 'median'),
        median_target_segment_shape_nrmse_deterioration=('median_target_segment_shape_nrmse_deterioration', 'median'),
        median_envelope_log_rate_error_per_day=('median_envelope_log_rate_error_per_day', 'median'),
        median_envelope_log_rate_absolute_error_per_day=('median_envelope_log_rate_absolute_error_per_day', 'median'),
        q25_envelope_log_rate_absolute_error_per_day=('median_envelope_log_rate_absolute_error_per_day', lambda x: x.quantile(0.25)),
        q75_envelope_log_rate_absolute_error_per_day=('median_envelope_log_rate_absolute_error_per_day', lambda x: x.quantile(0.75)),
        median_envelope_log_rate_absolute_error_deterioration=('median_envelope_log_rate_absolute_error_deterioration', 'median'),
        median_phase_equivalent_absolute_error_hours=('median_phase_equivalent_absolute_error_hours', 'median'),
        median_phase_equivalent_absolute_error_hours_deterioration=('median_phase_equivalent_absolute_error_hours_deterioration', 'median'),
        median_ratio_log_error=('median_ratio_log_error', 'median'),
        median_ratio_log_error_deterioration=('median_ratio_log_error_deterioration', 'median'),
    ).reset_index()

def bootstrap_summary(anchor_level, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    rows = []
    applicable = anchor_level[anchor_level['method_applicable']].copy()
    for keys, group in applicable.groupby(GROUP_COLUMNS, dropna=False, sort=True):
        metadata = dict(zip(GROUP_COLUMNS, keys if isinstance(keys, tuple) else (keys,)))
        for column, metric in BOOTSTRAP_METRICS.items():
            values = pd.to_numeric(group[column], errors='coerce').to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            point = float(np.median(values))
            lower = np.nan
            upper = np.nan
            completed = 0
            if len(values) >= 2:
                samples = values[rng.integers(0, len(values), size=(int(replicates), len(values)))]
                estimates = np.median(samples, axis=1)
                lower, upper = np.percentile(estimates, [2.5, 97.5])
                lower = float(lower)
                upper = float(upper)
                completed = int(replicates)
            rows.append({**metadata, 'metric': metric, 'point_estimate': point, 'ci_lower_95': lower, 'ci_upper_95': upper, 'n_anchors_valid': len(values), 'bootstrap_replicates': completed, 'bootstrap_seed': int(seed)})
    return pd.DataFrame(rows)

def compact_paired_table(paired):
    columns = [
        'model_spec_version', 'synthetic_id', 'source_synthetic_id', 'anchor_index', 'variant_number',
        'anchor_recording_uid', 'anchor_recording_id', 'tissue', 'scenario_family', 'scenario_level_number',
        'scenario_level', 'added_component', 'target_status', 'target_status_reason', 'method', 'method_applicable',
        'processing_success', 'failure_reason', 'primary_metric_name', 'primary_metric_unit', 'ground_truth',
        'estimate', 'signed_error', 'primary_error', 'primary_error_deterioration_from_target_only',
        'trajectory_pearson', 'trajectory_pearson_deterioration_from_target_only', 'target_segment_shape_nrmse',
        'target_segment_shape_nrmse_deterioration_from_target_only', 'envelope_log_rate_error_per_day',
        'envelope_log_rate_absolute_error_per_day', 'envelope_log_rate_absolute_error_per_day_deterioration_from_target_only',
        'phase_signed_angle_error_radians', 'phase_absolute_angle_error_radians', 'phase_equivalent_signed_error_hours',
        'phase_equivalent_absolute_error_hours', 'phase_equivalent_absolute_error_hours_deterioration_from_target_only',
        'ratio_log_error', 'ratio_log_error_deterioration_from_target_only', 'valid_primary',
    ]
    return paired[[column for column in columns if column in paired.columns]]

def validate(truth_file, estimate_files, output_folder, allow_partial=False, bootstrap_replicates=BOOTSTRAP_REPLICATES):
    truth = pd.read_csv(truth_file, low_memory=False)
    validate_truth_design(truth, allow_partial=allow_partial)
    estimates = load_estimates(estimate_files, truth, allow_partial=allow_partial)
    trace_metrics = build_trace_metrics(truth, estimates)
    failures = failure_summary(trace_metrics)
    trace_metrics = trace_metrics.loc[trace_metrics['method_applicable']].copy()
    paired = paired_degradation(trace_metrics)
    anchors = anchor_summary(paired)
    if not allow_partial and not (anchors['n_variants'] == 10).all():
        raise ValueError('Formal validation requires 10 source variants per anchor and scenario cell.')
    groups = group_summary(anchors)
    bootstrap = bootstrap_summary(anchors, replicates=bootstrap_replicates)
    outputs = {
        'scenario_paired_nuisance_degradation.csv.gz': compact_paired_table(paired),
        'scenario_anchor_level_summary.csv': anchors,
        'scenario_group_summary.csv': groups,
        'scenario_bootstrap_summary.csv': bootstrap,
        'scenario_failure_summary.csv': failures,
    }
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    for filename, table in outputs.items():
        table.to_csv(output_folder / filename, index=False)
    return outputs

def main():
    files = [DEFAULT_OUTPUT_FOLDER / 'stft_estimates.csv.gz', DEFAULT_OUTPUT_FOLDER / 'cwt_pyboat_estimates.csv.gz', DEFAULT_OUTPUT_FOLDER / 'hilbert_estimates.csv.gz']
    validate(DEFAULT_TRUTH_FILE, files, DEFAULT_OUTPUT_FOLDER)

if __name__ == '__main__':
    main()
