from importlib import import_module
import numpy as np
import pandas as pd
common = import_module('MSC-STEP3A-COMMON')
METHOD = 'CWT_pyBOAT'
N_PERIODS = 200
RIDGE_SMOOTHING_POINTS = 12
RIDGE_POWER_THRESHOLD = 0.0
OUTPUT_FILE = common.DEFAULT_OUTPUT_FOLDER / 'cwt_pyboat_estimates.csv.gz'

def find_column(table, candidates):
    normalised = {str(column).strip().lower().replace(' ', '_'): column for column in table.columns}
    for candidate in candidates:
        if candidate in normalised:
            return normalised[candidate]
    return None

def analyse_cwt(trace_info, amplitude_correction_factor=1.0):
    try:
        from pyboat import WAnalyzer
    except ImportError as error:
        raise ImportError('pyBOAT is not installed') from error
    time_hours = trace_info['time_hours']
    dt_hours = trace_info['dt_hours']
    analyzer = WAnalyzer(np.linspace(common.MIN_PERIOD_HOURS, common.MAX_PERIOD_HOURS, N_PERIODS), dt_hours, time_unit_label='h')
    trend = analyzer.sinc_smooth(trace_info['signal'], T_c=common.DETREND_CUTOFF_HOURS)
    analyzer.compute_spectrum(trace_info['signal'] - trend, do_plot=False)
    ridge = analyzer.get_maxRidge(power_thresh=RIDGE_POWER_THRESHOLD, smoothing_wsize=RIDGE_SMOOTHING_POINTS)
    if ridge is None:
        ridge = analyzer.ridge_data
    if not isinstance(ridge, pd.DataFrame):
        ridge = pd.DataFrame(ridge)
    if ridge.empty:
        raise ValueError('pyBOAT returned an empty ridge')
    time_column = find_column(ridge, ['time', 'times', 'time_hours'])
    period_column = find_column(ridge, ['period', 'periods', 'ridge_period'])
    amplitude_column = find_column(ridge, ['amplitude', 'ridge_amplitude', 'amp'])
    phase_column = find_column(ridge, ['phase', 'phase_radians', 'ridge_phase'])
    if period_column is None or amplitude_column is None:
        raise ValueError('Could not identify pyBOAT period/amplitude columns: ' + ', '.join(map(str, ridge.columns)))
    if time_column is None:
        ridge_time = time_hours[0] + np.asarray(ridge.index, dtype=float) * dt_hours
    else:
        ridge_time = ridge[time_column].to_numpy(dtype=float)
        if np.nanmin(ridge_time) < time_hours[0] - dt_hours:
            ridge_time = time_hours[0] + ridge_time
    phase = ridge[phase_column].to_numpy(dtype=float) if phase_column is not None else None
    return common.make_result_row(trace_info=trace_info, method=METHOD, time_hours=ridge_time, period_hours=ridge[period_column].to_numpy(dtype=float), amplitude=ridge[amplitude_column].to_numpy(dtype=float), phase_radians=phase, amplitude_correction_factor=amplitude_correction_factor)

def run(truth_file=common.DEFAULT_TRUTH_FILE, source_trace_file=common.DEFAULT_SOURCE_TRACE_FILE, output_file=OUTPUT_FILE, calibration_file=common.DEFAULT_OUTPUT_FOLDER / 'amplitude_calibration_factors.csv', max_traces=None):
    factor = common.load_amplitude_correction_factor(calibration_file, METHOD)
    return common.run_generated_scenario_analysis(truth_file, source_trace_file, METHOD, lambda trace_info: analyse_cwt(trace_info, factor), output_file, max_traces)

def main():
    run()
if __name__ == '__main__':
    main()
