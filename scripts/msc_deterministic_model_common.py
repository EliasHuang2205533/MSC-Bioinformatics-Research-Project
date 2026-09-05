from collections.abc import Mapping
import numpy as np
MODEL_SPEC_VERSION = 'msc_x10_causal_angle_halfgaussian_v1'
TRANSITION_DURATION_DAYS = 0.2
ACUTE_TRANSIENT_SIGMA_HOURS = 3.0
REFERENCE_PRE_BUFFER_HOURS = 6.0
EPS = 1e-08
PARAMETER_COLUMNS = ['fit_pre_amplitude', 'fit_post_amplitude', 'fit_pre_period_hours', 'fit_post_period_hours', 'fit_pre_envelope_log_rate_per_day', 'fit_post_envelope_log_rate_per_day', 'fit_baseline_step', 'fit_phase_shift_angle_radians', 'fit_linear_drift_slope_per_day', 'fit_signed_acute_transient_amplitude']
PARAMETER_LOWER_BOUNDS = np.array([0.05, 0.05, 20.0, 20.0, -1.4, -1.4, -2.0, -np.pi, -1.0, -6.0])
PARAMETER_UPPER_BOUNDS = np.array([6.0, 10.0, 30.0, 30.0, 0.12, 0.15, 10.0, np.pi, 1.0, 6.0])
BOUNDARY_RELATIVE_TOLERANCE = 1e-03

def wrap_angle(angle):
    wrapped = (np.asarray(angle, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi
    return float(wrapped) if wrapped.ndim == 0 else wrapped

def circular_difference(angle, reference):
    return wrap_angle(np.asarray(angle, dtype=float) - reference)

def phase_angle_to_hours(angle_radians, post_period_hours):
    return wrap_angle(angle_radians) * float(post_period_hours) / (2.0 * np.pi)

def causal_smoothstep(t_days, forskolin_day, duration_days=TRANSITION_DURATION_DAYS):
    if not np.isfinite(duration_days) or duration_days <= 0:
        raise ValueError('Transition duration must be positive and finite.')
    u = np.clip((np.asarray(t_days, dtype=float) - float(forskolin_day)) / duration_days, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)

def exponential_envelope(t_days, amplitude_start, log_rate_per_day, segment_start, segment_end):
    duration = max(float(segment_end - segment_start), EPS)
    elapsed = np.clip(np.asarray(t_days, dtype=float) - float(segment_start), 0.0, duration)
    return float(amplitude_start) * np.exp(float(log_rate_per_day) * elapsed)

def piecewise_phase(t_hours, forskolin_hour, pre_period_hours, post_period_hours, phase_shift_angle_radians, initial_phase_radians):
    t_hours = np.asarray(t_hours, dtype=float)
    phase = np.empty_like(t_hours)
    pre = t_hours < float(forskolin_hour)
    phase[pre] = float(initial_phase_radians) + 2.0 * np.pi * t_hours[pre] / float(pre_period_hours)
    phase_at_f = float(initial_phase_radians) + 2.0 * np.pi * float(forskolin_hour) / float(pre_period_hours)
    phase[~pre] = phase_at_f + wrap_angle(phase_shift_angle_radians) + 2.0 * np.pi * (t_hours[~pre] - float(forskolin_hour)) / float(post_period_hours)
    return phase

def _parameter_vector(parameters):
    if isinstance(parameters, Mapping):
        values = [parameters[column] for column in PARAMETER_COLUMNS]
    else:
        values = parameters
    values = np.asarray(values, dtype=float)
    if values.shape != (len(PARAMETER_COLUMNS),):
        raise ValueError(f'Expected {len(PARAMETER_COLUMNS)} model parameters, got {values.shape}.')
    if not np.all(np.isfinite(values)):
        raise ValueError('Model parameters contain non-finite values.')
    values = values.copy()
    values[7] = wrap_angle(values[7])
    return values

def parameter_boundary_hits(parameters):
    values = _parameter_vector(parameters)
    tolerance = BOUNDARY_RELATIVE_TOLERANCE * (PARAMETER_UPPER_BOUNDS - PARAMETER_LOWER_BOUNDS)
    at_lower = values - PARAMETER_LOWER_BOUNDS <= tolerance
    at_upper = PARAMETER_UPPER_BOUNDS - values <= tolerance
    at_lower[7] = False
    at_upper[7] = False
    return [name for name, hit in zip(PARAMETER_COLUMNS, at_lower | at_upper) if hit]

def default_reference_mask(t_days, forskolin_day):
    t_days = np.asarray(t_days, dtype=float)
    reference_end = float(forskolin_day) - REFERENCE_PRE_BUFFER_HOURS / 24.0
    return np.isfinite(t_days) & (t_days >= float(np.nanmin(t_days))) & (t_days < reference_end)

def deterministic_model(t_days, forskolin_day, duration_days, drift_reference_day, parameters, initial_phase, reference_mask=None):
    t_days = np.asarray(t_days, dtype=float)
    if t_days.ndim != 1 or len(t_days) < 2 or (not np.all(np.isfinite(t_days))):
        raise ValueError('t_days must be a finite one-dimensional time vector.')
    p = _parameter_vector(parameters)
    pre_amplitude, post_amplitude, pre_period, post_period, pre_log_rate, post_log_rate, baseline_step, phase_shift_angle, drift_slope, acute_transient_amplitude = p
    transition = causal_smoothstep(t_days, forskolin_day)
    pre_envelope = exponential_envelope(t_days, pre_amplitude, pre_log_rate, float(t_days.min()), forskolin_day)
    post_envelope = exponential_envelope(t_days, post_amplitude, post_log_rate, forskolin_day, duration_days)
    amplitude_envelope = (1.0 - transition) * pre_envelope + transition * post_envelope
    phase = piecewise_phase(t_days * 24.0, float(forskolin_day) * 24.0, pre_period, post_period, phase_shift_angle, initial_phase)
    rhythmic_component = amplitude_envelope * np.cos(phase)
    baseline_component = transition * baseline_step
    drift_component = drift_slope * (t_days - float(drift_reference_day))
    elapsed = t_days - float(forskolin_day)
    sigma_days = ACUTE_TRANSIENT_SIGMA_HOURS / 24.0
    acute_transient_component = np.where(elapsed >= 0.0, acute_transient_amplitude * np.exp(-0.5 * (elapsed / sigma_days) ** 2), 0.0)
    model_raw = baseline_component + drift_component + acute_transient_component + rhythmic_component
    if reference_mask is None:
        reference_mask = default_reference_mask(t_days, forskolin_day)
    reference_mask = np.asarray(reference_mask, dtype=bool)
    if reference_mask.shape != t_days.shape or np.sum(reference_mask) < 5:
        raise ValueError('The model reference window contains too few points.')
    model_offset = float(np.median(model_raw[reference_mask]))
    centering_component = np.full_like(t_days, -model_offset)
    fitted_signal = model_raw + centering_component
    return {'fitted_signal': fitted_signal, 'synthetic_signal_normalised': fitted_signal, 'model_raw_signal': model_raw, 'centering_component': centering_component, 'transition': transition, 'baseline_component': baseline_component, 'linear_drift_component': drift_component, 'acute_transient_component': acute_transient_component, 'rhythmic_component': rhythmic_component, 'amplitude_envelope': amplitude_envelope, 'phase_radians': phase}
