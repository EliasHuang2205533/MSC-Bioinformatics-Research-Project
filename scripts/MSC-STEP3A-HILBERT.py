from importlib import import_module
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
common = import_module('MSC-STEP3A-COMMON')
METHOD = 'Hilbert'
FILTER_ORDER = 3
OUTPUT_FILE = common.DEFAULT_OUTPUT_FOLDER / 'hilbert_estimates.csv.gz'

def analyse_hilbert(trace_info, amplitude_correction_factor=1.0):
    time_hours = trace_info['time_hours']
    dt_hours = trace_info['dt_hours']
    sampling_frequency = 1.0 / dt_hours
    sos = butter(FILTER_ORDER, [1.0 / common.MAX_PERIOD_HOURS, 1.0 / common.MIN_PERIOD_HOURS], btype='bandpass', fs=sampling_frequency, output='sos')
    filtered = sosfiltfilt(sos, trace_info['signal'])
    analytic = hilbert(filtered)
    amplitude = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    angular_frequency = np.gradient(phase, time_hours)
    period = np.full_like(angular_frequency, np.nan)
    positive = angular_frequency > 0
    period[positive] = 2.0 * np.pi / angular_frequency[positive]
    return common.make_result_row(trace_info=trace_info, method=METHOD, time_hours=time_hours, period_hours=period, amplitude=amplitude, phase_radians=phase, amplitude_correction_factor=amplitude_correction_factor)

def run(truth_file=common.DEFAULT_TRUTH_FILE, source_trace_file=common.DEFAULT_SOURCE_TRACE_FILE, output_file=OUTPUT_FILE, calibration_file=common.DEFAULT_OUTPUT_FOLDER / 'amplitude_calibration_factors.csv', max_traces=None):
    factor = common.load_amplitude_correction_factor(calibration_file, METHOD)
    return common.run_generated_scenario_analysis(truth_file, source_trace_file, METHOD, lambda trace_info: analyse_hilbert(trace_info, factor), output_file, max_traces)

def main():
    run()
if __name__ == '__main__':
    main()
