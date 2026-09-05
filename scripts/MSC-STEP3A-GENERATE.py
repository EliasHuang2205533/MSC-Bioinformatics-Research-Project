from pathlib import Path
import numpy as np
import pandas as pd
from msc_deterministic_model_common import MODEL_SPEC_VERSION, PARAMETER_COLUMNS, phase_angle_to_hours
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEP2B_TABLE_FOLDER = PROJECT_ROOT / 'results' / 'step2b' / 'tables'
PARAMETER_FILE = STEP2B_TABLE_FOLDER / 'synthetic_ground_truth_parameters.csv'
SOURCE_TRACE_FILE = STEP2B_TABLE_FOLDER / 'synthetic_traces_long_format.csv.gz'
TABLE_FOLDER = PROJECT_ROOT / 'results' / 'step3a' / 'tables'
TRUTH_FILE = TABLE_FOLDER / 'scenario_ground_truth_parameters.csv'
DESIGN_FILE = TABLE_FOLDER / 'scenario_design_summary.csv'
EPS = 1e-10
NEUTRAL_PARAMETERS = {'fit_pre_amplitude': 2.0, 'fit_post_amplitude': 2.0, 'fit_pre_period_hours': 24.0, 'fit_post_period_hours': 24.0, 'fit_pre_envelope_log_rate_per_day': 0.0, 'fit_post_envelope_log_rate_per_day': 0.0, 'fit_baseline_step': 0.0, 'fit_phase_shift_angle_radians': 0.0, 'fit_linear_drift_slope_per_day': 0.0, 'fit_signed_acute_transient_amplitude': 0.0}
SCENARIO_DESIGNS = {'pre_period_hours': {'target_parameters': ('fit_pre_period_hours',), 'nuisances': (None, 'fit_pre_amplitude', 'fit_pre_envelope_log_rate_per_day', 'fit_linear_drift_slope_per_day')}, 'post_period_hours': {'target_parameters': ('fit_post_period_hours',), 'nuisances': (None, 'fit_post_amplitude', 'fit_post_envelope_log_rate_per_day', 'fit_linear_drift_slope_per_day', 'fit_baseline_step', 'fit_signed_acute_transient_amplitude')}, 'pre_amplitude_trajectory': {'target_parameters': ('fit_pre_amplitude', 'fit_pre_envelope_log_rate_per_day'), 'nuisances': (None, 'fit_pre_period_hours', 'fit_linear_drift_slope_per_day')}, 'post_amplitude_trajectory': {'target_parameters': ('fit_post_amplitude', 'fit_post_envelope_log_rate_per_day'), 'nuisances': (None, 'fit_post_period_hours', 'fit_linear_drift_slope_per_day', 'fit_baseline_step', 'fit_signed_acute_transient_amplitude')}, 'phase_shift_hours': {'target_parameters': ('fit_phase_shift_angle_radians',), 'nuisances': (None, 'fit_pre_amplitude', 'fit_post_amplitude', 'fit_pre_period_hours', 'fit_post_period_hours', 'fit_pre_envelope_log_rate_per_day', 'fit_post_envelope_log_rate_per_day', 'fit_linear_drift_slope_per_day', 'fit_baseline_step', 'fit_signed_acute_transient_amplitude')}}
COMPONENT_LABELS = {None: 'target_only', 'fit_pre_amplitude': 'pre_amplitude', 'fit_post_amplitude': 'post_amplitude', 'fit_pre_period_hours': 'pre_period', 'fit_post_period_hours': 'post_period', 'fit_pre_envelope_log_rate_per_day': 'pre_envelope', 'fit_post_envelope_log_rate_per_day': 'post_envelope', 'fit_baseline_step': 'baseline_step', 'fit_linear_drift_slope_per_day': 'linear_drift', 'fit_signed_acute_transient_amplitude': 'acute_treatment_transient'}
TARGET_STATUS_COLUMNS = {'pre_period_hours': 'pre_period_status', 'post_period_hours': 'post_period_status', 'pre_amplitude_trajectory': 'pre_amplitude_envelope_status', 'post_amplitude_trajectory': 'post_amplitude_envelope_status', 'phase_shift_hours': 'phase_shift_status'}
TARGET_REASON_COLUMNS = {family: column.replace('_status', '_reason') for family, column in TARGET_STATUS_COLUMNS.items()}

def validate_sources(parameters, traces, formal):
    required_parameters = {'synthetic_id', 'anchor_index', 'variant_number', 'anchor_recording_uid', 'anchor_recording_id', 'tissue', 'duration_days', 'forskolin_day_zero', 'forskolin_day_original', 'drift_reference_day_original', 'drift_reference_day_zero', 'profiled_initial_phase_radians', 'model_spec_version', *PARAMETER_COLUMNS, *TARGET_STATUS_COLUMNS.values(), *TARGET_REASON_COLUMNS.values()}
    required_traces = {'synthetic_id', 'time_days_zero'}
    missing_parameters = required_parameters - set(parameters.columns)
    missing_traces = required_traces - set(traces.columns)
    if missing_parameters:
        raise ValueError('STEP2B ground-truth table is missing: ' + ', '.join(sorted(missing_parameters)))
    if missing_traces:
        raise ValueError('STEP2B trace table is missing: ' + ', '.join(sorted(missing_traces)))
    if parameters['synthetic_id'].astype(str).duplicated().any():
        raise ValueError('Duplicate STEP2B synthetic_id values')
    if traces.duplicated(['synthetic_id', 'time_days_zero']).any():
        raise ValueError('Duplicate STEP2B source times')
    versions = set(parameters['model_spec_version'].dropna().astype(str))
    if versions != {MODEL_SPEC_VERSION}:
        raise ValueError(f'Expected model_spec_version={MODEL_SPEC_VERSION!r}, got {sorted(versions)}')
    numeric = parameters[[*PARAMETER_COLUMNS, 'duration_days', 'forskolin_day_zero', 'drift_reference_day_zero', 'profiled_initial_phase_radians']].apply(pd.to_numeric, errors='coerce')
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError('STEP2B source parameters contain non-finite values')
    parameter_ids = set(parameters['synthetic_id'].astype(str))
    trace_ids = set(traces['synthetic_id'].astype(str))
    if parameter_ids != trace_ids:
        raise ValueError(f'STEP2B parameter and trace IDs differ: missing={len(parameter_ids - trace_ids)}, extra={len(trace_ids - parameter_ids)}')
    if formal:
        if len(parameters) != 530 or parameters['anchor_recording_uid'].nunique() != 53:
            raise ValueError('Formal STEP3A requires 530 sources from 53 anchors')
        counts = parameters.groupby('anchor_recording_uid')['synthetic_id'].size()
        if not (counts == 10).all():
            raise ValueError('Every anchor must contribute exactly 10 STEP2B sources')
        variants = parameters.groupby('anchor_recording_uid')['variant_number'].agg(lambda values: set(pd.to_numeric(values, errors='raise').astype(int)))
        if not variants.map(lambda values: values == set(range(10))).all():
            raise ValueError('Every anchor must contain variants V000-V009')

def independent_scenarios(source, family):
    design = SCENARIO_DESIGNS[family]
    for number, nuisance in enumerate(design['nuisances']):
        parameters = dict(NEUTRAL_PARAMETERS)
        for target in design['target_parameters']:
            parameters[target] = float(source[target])
        if nuisance is not None:
            parameters[nuisance] = float(source[nuisance])
        yield {'scenario_level_number': number, 'scenario_level': f'S{number}_{COMPONENT_LABELS[nuisance]}', 'target_parameters': ';'.join(design['target_parameters']), 'added_parameter': 'none' if nuisance is None else nuisance, 'added_component': COMPONENT_LABELS[nuisance], 'parameters': parameters}

def generate(parameter_file=PARAMETER_FILE, source_trace_file=SOURCE_TRACE_FILE, truth_file=TRUTH_FILE, design_file=DESIGN_FILE, max_source_traces=None):
    parameter_file = Path(parameter_file)
    source_trace_file = Path(source_trace_file)
    truth_file = Path(truth_file)
    design_file = Path(design_file)
    if not parameter_file.exists():
        raise FileNotFoundError(parameter_file)
    if not source_trace_file.exists():
        raise FileNotFoundError(source_trace_file)
    parameters = pd.read_csv(parameter_file, low_memory=False)
    formal = max_source_traces is None
    if max_source_traces is not None:
        parameters = parameters.iloc[:int(max_source_traces)].copy()
    source_ids = set(parameters['synthetic_id'].astype(str))
    traces = pd.read_csv(source_trace_file, usecols=['synthetic_id', 'time_days_zero'], low_memory=False)
    traces = traces[traces['synthetic_id'].astype(str).isin(source_ids)].copy()
    validate_sources(parameters, traces, formal)
    trace_groups = {str(key): group.sort_values('time_days_zero') for key, group in traces.groupby('synthetic_id', sort=False)}
    rows = []
    for source in parameters.to_dict('records'):
        source_id = str(source['synthetic_id'])
        grid = trace_groups[source_id]
        t_zero = pd.to_numeric(grid['time_days_zero'], errors='coerce').to_numpy(dtype=float)
        t_zero = np.unique(np.sort(t_zero[np.isfinite(t_zero)]))
        if len(t_zero) < 20:
            raise ValueError(f'Fewer than 20 unique time points for {source_id}')
        forskolin_zero = float(source['forskolin_day_zero'])
        drift_reference_zero = float(source['drift_reference_day_zero'])
        if not 0.0 <= drift_reference_zero < forskolin_zero:
            raise ValueError(f'Invalid drift reference for {source_id}')
        for family in SCENARIO_DESIGNS:
            status_column = TARGET_STATUS_COLUMNS[family]
            reason_column = TARGET_REASON_COLUMNS[family]
            for scenario in independent_scenarios(source, family):
                scenario_id = f"{source_id}__{family}__S{scenario['scenario_level_number']:02d}"
                row = {'synthetic_id': scenario_id, 'model_spec_version': MODEL_SPEC_VERSION, 'source_synthetic_id': source_id, 'anchor_index': source['anchor_index'], 'variant_number': source['variant_number'], 'anchor_recording_uid': source['anchor_recording_uid'], 'anchor_recording_id': source['anchor_recording_id'], 'tissue': source['tissue'], 'scenario_family': family, 'scenario_level_number': scenario['scenario_level_number'], 'scenario_level': scenario['scenario_level'], 'target_parameters': scenario['target_parameters'], 'added_parameter': scenario['added_parameter'], 'added_component': scenario['added_component'], 'target_status': source[status_column], 'target_status_reason': source[reason_column], 'duration_days': float(source['duration_days']), 'forskolin_day_original': source['forskolin_day_original'], 'forskolin_day_zero': forskolin_zero, 'drift_reference_day_original': source['drift_reference_day_original'], 'drift_reference_day_zero': drift_reference_zero, 'profiled_initial_phase_radians': float(source['profiled_initial_phase_radians'])}
                row.update(scenario['parameters'])
                row['fit_phase_shift_hours'] = phase_angle_to_hours(row['fit_phase_shift_angle_radians'], row['fit_post_period_hours'])
                rows.append(row)
    truth = pd.DataFrame(rows)
    expected_per_source = sum((len(item['nuisances']) for item in SCENARIO_DESIGNS.values()))
    if len(truth) != len(parameters) * expected_per_source or truth['synthetic_id'].duplicated().any():
        raise RuntimeError('Invalid STEP3A scenario count or duplicate synthetic_id')
    for row in truth.to_dict('records'):
        allowed = set(str(row['target_parameters']).split(';'))
        if row['added_parameter'] != 'none':
            allowed.add(row['added_parameter'])
        active = {parameter for parameter in PARAMETER_COLUMNS if not np.isclose(float(row[parameter]), float(NEUTRAL_PARAMETERS[parameter]), atol=EPS, rtol=0.0)}
        if not active.issubset(allowed):
            raise RuntimeError(f"Cumulative component leak in {row['synthetic_id']}")
    references = truth[truth['added_component'].eq('target_only')].groupby(['source_synthetic_id', 'scenario_family']).size()
    if len(references) != len(parameters) * len(SCENARIO_DESIGNS) or not (references == 1).all():
        raise RuntimeError('Every source and family must contain one target-only scenario')
    truth_file.parent.mkdir(parents=True, exist_ok=True)
    truth.to_csv(truth_file, index=False)
    design = truth.groupby(['scenario_family', 'scenario_level_number', 'scenario_level', 'added_component', 'target_status'], dropna=False, sort=True).agg(n_scenarios=('synthetic_id', 'size'), n_source_traces=('source_synthetic_id', 'nunique'), n_anchors=('anchor_recording_uid', 'nunique')).reset_index()
    design.to_csv(design_file, index=False)
    return (truth, design)

def main():
    generate()
if __name__ == '__main__':
    main()
