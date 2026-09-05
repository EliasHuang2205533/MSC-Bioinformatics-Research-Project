from importlib import import_module
import numpy as np
from scipy.signal import stft
common = import_module('MSC-STEP3A-COMMON')
METHOD = 'STFT'
WINDOW_HOURS = common.STFT_WINDOW_HOURS
OVERLAP_FRACTION = common.STFT_OVERLAP_FRACTION
NFFT = 4096
OUTPUT_FILE = common.DEFAULT_OUTPUT_FOLDER / 'stft_estimates.csv.gz'

def analyse_stft(trace_info, amplitude_correction_factor=1.0):
    time_hours = trace_info['time_hours']
    dt_hours = trace_info['dt_hours']
    signal = common.rolling_mean_detrend(trace_info['signal'], dt_hours, common.DETREND_CUTOFF_HOURS)
    nperseg = min(len(signal), max(8, int(round(WINDOW_HOURS / dt_hours))))
    noverlap = min(nperseg - 1, int(round(nperseg * OVERLAP_FRACTION)))
    frequencies, relative_time, spectrum = stft(signal, fs=1.0 / dt_hours, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=max(NFFT, nperseg), detrend=False, return_onesided=True, boundary=None, padded=False, scaling='spectrum')
    circadian = (frequencies >= 1.0 / common.MAX_PERIOD_HOURS) & (frequencies <= 1.0 / common.MIN_PERIOD_HOURS)
    if not np.any(circadian):
        raise ValueError('No STFT bins in the circadian range')
    circadian_spectrum = spectrum[circadian]
    circadian_frequencies = frequencies[circadian]
    ridge_index = np.argmax(np.abs(circadian_spectrum), axis=0)
    columns = np.arange(circadian_spectrum.shape[1])
    ridge_complex = circadian_spectrum[ridge_index, columns]
    ridge_frequency = circadian_frequencies[ridge_index]
    ridge_time = time_hours[0] + relative_time
    return common.make_result_row(trace_info=trace_info, method=METHOD, time_hours=ridge_time, period_hours=1.0 / ridge_frequency, amplitude=2.0 * np.abs(ridge_complex), phase_radians=None, amplitude_correction_factor=amplitude_correction_factor)

def run(truth_file=common.DEFAULT_TRUTH_FILE, source_trace_file=common.DEFAULT_SOURCE_TRACE_FILE, output_file=OUTPUT_FILE, calibration_file=common.DEFAULT_OUTPUT_FOLDER / 'amplitude_calibration_factors.csv', max_traces=None):
    factor = common.load_amplitude_correction_factor(calibration_file, METHOD)
    return common.run_generated_scenario_analysis(truth_file, source_trace_file, METHOD, lambda trace_info: analyse_stft(trace_info, factor), output_file, max_traces)

def main():
    run()
if __name__ == '__main__':
    main()
