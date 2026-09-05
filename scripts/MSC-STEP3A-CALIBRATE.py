from importlib import import_module
from pathlib import Path
import numpy as np
import pandas as pd
common = import_module('MSC-STEP3A-COMMON')
stft_module = import_module('MSC-STEP3A-STFT')
cwt_module = import_module('MSC-STEP3A-CWT')
hilbert_module = import_module('MSC-STEP3A-HILBERT')
CALIBRATION_AMPLITUDES = (0.5, 1.0, 2.0, 4.0)
CALIBRATION_PERIODS_HOURS = (20.0, 22.0, 24.0, 26.0, 28.0, 30.0)
CALIBRATION_PHASE_RADIANS = 0.37
MAX_CALIBRATION_GRIDS = 12
DETAIL_FILE = common.DEFAULT_OUTPUT_FOLDER / 'amplitude_calibration_details.csv'
FACTOR_FILE = common.DEFAULT_OUTPUT_FOLDER / 'amplitude_calibration_factors.csv'

def select_duration_grids(source_trace_file, max_grids=MAX_CALIBRATION_GRIDS):
    table = pd.read_csv(source_trace_file, usecols=['synthetic_id', 'variant_number', 'time_days_zero'], low_memory=False)
    table = table[pd.to_numeric(table['variant_number'], errors='coerce').eq(0)].copy()
    grids = {}
    for synthetic_id, group in table.groupby('synthetic_id', sort=False):
        time_hours = np.sort(pd.to_numeric(group['time_days_zero'], errors='coerce').to_numpy(dtype=float) * 24.0)
        time_hours = np.unique(time_hours[np.isfinite(time_hours)])
        if len(time_hours) < 20:
            continue
        duration = float(time_hours[-1] - time_hours[0])
        grids.setdefault(round(duration, 6), (str(synthetic_id), time_hours))
    groups = [grids[key] for key in sorted(grids)]
    if not groups:
        raise ValueError('No valid V000 source time grids were found for calibration')
    if len(groups) <= max_grids:
        return groups
    indices = np.unique(np.round(np.linspace(0, len(groups) - 1, max_grids)).astype(int))
    return [groups[index] for index in indices]

def pure_sine_trace_info(grid_id, source_time_hours, amplitude, period_hours, number):
    start = float(source_time_hours[0])
    end = float(source_time_hours[-1])
    uniform_time = np.arange(start, end + 0.5 * common.SAMPLING_INTERVAL_HOURS, common.SAMPLING_INTERVAL_HOURS)
    trace_info = {column: 'calibration' for column in common.TRACE_METADATA_COLUMNS}
    trace_info.update({'synthetic_id': f'cal_{number:04d}', 'source_synthetic_id': grid_id, 'scenario_family': 'pure_sine_calibration', 'time_hours': uniform_time, 'signal': amplitude * np.cos(2.0 * np.pi * uniform_time / period_hours + CALIBRATION_PHASE_RADIANS), 'gt_amplitude': np.full(len(uniform_time), amplitude, dtype=float), 'dt_hours': common.SAMPLING_INTERVAL_HOURS, 'forskolin_hour': float((start + end) / 2.0), 'n_input_points': len(uniform_time)})
    return trace_info

def ols_through_origin(estimated, truth):
    estimated = np.asarray(estimated, dtype=float)
    truth = np.asarray(truth, dtype=float)
    valid = np.isfinite(estimated) & np.isfinite(truth) & (estimated > 0)
    estimated = estimated[valid]
    truth = truth[valid]
    denominator = float(np.sum(estimated ** 2))
    if len(estimated) == 0 or denominator <= 0:
        return np.nan
    return float(np.sum(estimated * truth) / denominator)

def calibrate(source_trace_file=common.DEFAULT_SOURCE_TRACE_FILE, detail_file=DETAIL_FILE, factor_file=FACTOR_FILE, include_cwt=True, max_grids=MAX_CALIBRATION_GRIDS):
    methods = {'STFT': stft_module.analyse_stft, 'Hilbert': hilbert_module.analyse_hilbert}
    if include_cwt:
        methods['CWT_pyBOAT'] = cwt_module.analyse_cwt
    rows = []
    grids = select_duration_grids(source_trace_file, max_grids=max_grids)
    trace_number = 0
    for grid_id, time_hours in grids:
        for amplitude in CALIBRATION_AMPLITUDES:
            for period_hours in CALIBRATION_PERIODS_HOURS:
                trace_number += 1
                info = pure_sine_trace_info(grid_id, time_hours, amplitude, period_hours, trace_number)
                for method, analyse in methods.items():
                    try:
                        result = analyse(info, amplitude_correction_factor=1.0)
                        for segment in ('pre', 'post'):
                            rows.append({'calibration_id': info['synthetic_id'], 'source_grid_id': grid_id, 'method': method, 'segment': segment, 'period_hours': period_hours, 'amplitude_gt': amplitude, 'amplitude_estimated': result[f'estimated_{segment}_amplitude_raw'], 'processing_success': result['processing_success'], 'failure_reason': result['failure_reason']})
                    except Exception as error:
                        rows.append({'calibration_id': info['synthetic_id'], 'source_grid_id': grid_id, 'method': method, 'segment': 'failed', 'period_hours': period_hours, 'amplitude_gt': amplitude, 'amplitude_estimated': np.nan, 'processing_success': False, 'failure_reason': str(error)})
    details = pd.DataFrame(rows)
    summaries = []
    for method, group in details.groupby('method', sort=False):
        estimated = group['amplitude_estimated'].to_numpy(dtype=float)
        truth = group['amplitude_gt'].to_numpy(dtype=float)
        valid = np.isfinite(estimated) & np.isfinite(truth) & (estimated > 0)
        factor = ols_through_origin(estimated, truth)
        summaries.append({'method': method, 'amplitude_correction_factor': factor, 'estimation_method': 'OLS_through_origin', 'ratio_median_sensitivity_check': float(np.median(truth[valid] / estimated[valid])) if np.any(valid) else np.nan, 'n_valid_calibration_estimates': int(np.sum(valid)), 'n_failed_calibration_estimates': int(np.sum(~valid))})
    summary = pd.DataFrame(summaries)
    if summary['amplitude_correction_factor'].isna().any() or (summary['amplitude_correction_factor'] <= 0).any():
        raise RuntimeError('Amplitude calibration failed for one or more methods')
    detail_file = Path(detail_file)
    factor_file = Path(factor_file)
    detail_file.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(detail_file, index=False)
    summary.to_csv(factor_file, index=False)
    return (details, summary)

def main():
    calibrate()
if __name__ == '__main__':
    main()
