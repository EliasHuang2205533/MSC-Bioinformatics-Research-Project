from itertools import product
from pathlib import Path
import re
import zlib

import numpy as np
import pandas as pd
from scipy.signal import stft


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "results" / "step1a" / "tables" / "all_lumicycle_analysis_long_format.csv"
CALIBRATION_FILE = PROJECT_ROOT / "results" / "step3a" / "tables" / "amplitude_calibration_factors.csv"
OUTPUT_FOLDER = PROJECT_ROOT / "results" / "step4"
MIN_PERIOD_HOURS = 18.0
MAX_PERIOD_HOURS = 32.0
DETREND_CUTOFF_HOURS = 72.0
STFT_WINDOW_HOURS = 60.0
STFT_OVERLAP_FRACTION = 0.90
STFT_NFFT = 4096
CWT_PERIOD_GRID_POINTS = 200
CWT_RIDGE_SMOOTHING_POINTS = 12
CWT_POWER_THRESHOLD = 0.0
CWT_EDGE_EXCLUSION_HOURS = MAX_PERIOD_HOURS
CWT_EVENT_EXCLUSION_HOURS = MAX_PERIOD_HOURS
STFT_EVENT_EXCLUSION_HOURS = STFT_WINDOW_HOURS / 2.0
MIN_SEGMENT_ESTIMATES = 5
BOOTSTRAP_REPEATS = 2000
RANDOM_SEED = 20260904
EPS = 1e-10

ANIMAL_RECORDINGS = {
    "20250626_M1": {
        "batch": "250625 to 250728",
        "sex": "M",
        "recordings": [
            "1C-SCN", "5C-T12", "5D-T10", "6A-T7", "6B-T5", "7C-L1",
            "7D-L2", "8A-L3",
        ],
    },
    "20250703_m1": {
        "batch": "250625 to 250728",
        "sex": "F",
        "recordings": [
            "4C-T5m1", "4D-T7m1", "5A-T10m1", "8B-L2m1", "8C-L3m1",
            "8D-L4-1m1", "6C-SCNmm1",
        ],
    },
    "20250703_m2": {
        "batch": "250625 to 250728",
        "sex": "F",
        "recordings": [
            "1B-L1m2", "1D-L2m2", "2C-L3m2", "2D-L4m2", "3A-L5m2",
            "3B-T10m2", "4B-T12m2", "5B-T12m2", "7A-SCNmm2", "7B-SCNcm2",
        ],
    },
    "20250728_M1": {
        "batch": "250728",
        "sex": "F",
        "recordings": ["6A-SCNrM1", "7A-SCNcM1"],
    },
    "20250728_M2": {
        "batch": "250728",
        "sex": "F",
        "recordings": ["6B-SCNrM2", "7B-SCNcM2"],
    },
    "20250811_1F": {
        "batch": "250811",
        "sex": "F",
        "recordings": [
            "1A-1FSCN", "2D-1FL2", "3A-1FL4", "4D-1FL4", "3D-1FT10",
            "5C-1FT7", "6A-1FL1", "7C-1FT12", "8A-1FT5",
        ],
    },
    "20250811_2M": {
        "batch": "250811",
        "sex": "M",
        "recordings": [
            "1B-2MSCN", "2C-2ML1", "4A-2MT10", "4C-2MT5", "5A-2ML2",
            "6B-2ML3", "7D-2MT12", "8C-2ML4", "8D-2MT7",
        ],
    },
    "20250811_3F": {"batch": "250811", "sex": "F", "recordings": ["1C-3FSCN"]},
    "20250811_4F": {"batch": "250811", "sex": "F", "recordings": ["1D-4FSCN"]},
    "20251003_M1": {
        "batch": "251003",
        "sex": "M",
        "recordings": ["2A-M1SCN", "2B-1MT7", "2C-M1L1", "2D-M1L4"],
    },
    "20251003_M2": {"batch": "251003", "sex": "M", "recordings": ["4A-M2SCN"]},
    "20251003_M3": {"batch": "251003", "sex": "M", "recordings": ["4B-M3SCN"]},
}

METRICS = [
    {"method": "CWT_pyBOAT", "metric": "pre_period_hours", "role": "primary", "circular": False},
    {"method": "CWT_pyBOAT", "metric": "post_period_hours", "role": "descriptive", "circular": False},
    {
        "method": "CWT_pyBOAT",
        "metric": "period_change_hours_post_minus_pre",
        "role": "primary",
        "circular": False,
    },
    {
        "method": "CWT_pyBOAT",
        "metric": "phase_shift_angle_radians",
        "role": "primary_circular",
        "circular": True,
    },
    {"method": "STFT", "metric": "pre_amplitude_counts", "role": "descriptive", "circular": False},
    {"method": "STFT", "metric": "post_amplitude_counts", "role": "descriptive", "circular": False},
    {"method": "STFT", "metric": "amplitude_ratio_post_over_pre", "role": "descriptive", "circular": False},
    {
        "method": "STFT",
        "metric": "amplitude_log_ratio_post_over_pre",
        "role": "primary",
        "circular": False,
    },
    {
        "method": "STFT",
        "metric": "pre_envelope_log_rate_per_day",
        "role": "secondary",
        "circular": False,
    },
    {
        "method": "STFT",
        "metric": "post_envelope_log_rate_per_day",
        "role": "secondary",
        "circular": False,
    },
    {
        "method": "STFT",
        "metric": "envelope_log_rate_change_post_minus_pre_per_day",
        "role": "secondary",
        "circular": False,
    },
]


def normalise_id(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def recording_key(batch, recording_id):
    return f"{str(batch).strip().casefold()}::{normalise_id(recording_id)}"


def wrap_angle(value):
    wrapped = (np.asarray(value, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi
    return float(wrapped) if wrapped.ndim == 0 else wrapped


def circular_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.angle(np.mean(np.exp(1j * values)))) if len(values) else np.nan


def scalar(group, column):
    values = group[column].dropna().unique()
    if len(values) != 1:
        raise ValueError(f"Expected one {column} value for {group['recording_uid'].iloc[0]}.")
    return values[0]


def build_animal_mapping():
    rows = []
    for animal_id, information in ANIMAL_RECORDINGS.items():
        for recording_id in information["recordings"]:
            rows.append({
                "recording_key": recording_key(information["batch"], recording_id),
                "animal_id": animal_id,
                "sex": information["sex"],
            })
    mapping = pd.DataFrame(rows)
    if len(mapping) != 55 or mapping["recording_key"].duplicated().any():
        raise RuntimeError("The experiment-sheet animal mapping is inconsistent.")
    return mapping


def load_data():
    data = pd.read_csv(INPUT_FILE, low_memory=False)
    required = {
        "recording_uid", "recording_id", "time_days", "counts_sec", "tissue",
        "forskolin_status", "forskolin_time_days",
    }
    if required - set(data.columns):
        raise ValueError("STEP1A analysis data are missing required columns.")
    selected = data["forskolin_status"].eq("administered_estimated")
    selected &= data["tissue"].isin(["SCN", "DRG"])
    data = data.loc[selected].copy()
    recording_table = data[["recording_uid", "recording_id", "batch"]].drop_duplicates().copy()
    recording_table["recording_key"] = [
        recording_key(batch, recording_id)
        for batch, recording_id in zip(
            recording_table["batch"], recording_table["recording_id"]
        )
    ]
    mapping = build_animal_mapping()
    if set(recording_table["recording_key"]) != set(mapping["recording_key"]):
        raise RuntimeError("The treated recordings do not match the experiment-sheet animal mapping.")
    recording_table = recording_table.merge(
        mapping, on="recording_key", how="left", validate="one_to_one"
    )
    data = data.merge(
        recording_table[["recording_uid", "animal_id", "sex"]],
        on="recording_uid",
        how="left",
        validate="many_to_one",
    )
    calibration = pd.read_csv(CALIBRATION_FILE)
    selected = calibration.loc[calibration["method"].eq("STFT"), "amplitude_correction_factor"]
    if len(selected) != 1 or not np.isfinite(float(selected.iloc[0])) or float(selected.iloc[0]) <= 0:
        raise ValueError("The STEP3A STFT amplitude calibration factor is invalid.")
    return data, float(selected.iloc[0])


def prepare_recording(group):
    data = group.copy()
    data["time_days"] = pd.to_numeric(data["time_days"], errors="coerce")
    data["counts_sec"] = pd.to_numeric(data["counts_sec"], errors="coerce")
    data = (
        data.dropna(subset=["time_days", "counts_sec"])
        .sort_values("time_days")
        .drop_duplicates("time_days")
    )
    time_hours = data["time_days"].to_numpy(dtype=float) * 24.0
    signal = data["counts_sec"].to_numpy(dtype=float)
    if len(time_hours) < 20:
        raise ValueError(f"Too few observations for {scalar(group, 'recording_uid')}.")
    interval = float(np.median(np.diff(time_hours)))
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError(f"Invalid sampling interval for {scalar(group, 'recording_uid')}.")
    uniform_time = np.arange(time_hours[0], time_hours[-1] + 0.5 * interval, interval)
    uniform_signal = np.interp(uniform_time, time_hours, signal)
    return uniform_time, uniform_signal, interval


def rolling_mean_detrend(signal, interval_hours):
    points = max(3, int(round(DETREND_CUTOFF_HOURS / interval_hours)))
    if points % 2 == 0:
        points += 1
    trend = (
        pd.Series(signal)
        .rolling(points, center=True, min_periods=max(3, points // 4))
        .mean()
        .interpolate(limit_direction="both")
        .to_numpy(dtype=float)
    )
    return np.asarray(signal, dtype=float) - trend


def find_column(table, candidates):
    columns = {
        str(column).strip().casefold().replace(" ", "_"): column
        for column in table.columns
    }
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    return None


def resolve_ridge_time(ridge, time_column, time_hours, interval_hours):
    if time_column is None:
        return time_hours[0] + np.asarray(ridge.index, dtype=float) * interval_hours
    candidate = ridge[time_column].to_numpy(dtype=float)
    finite = candidate[np.isfinite(candidate)]
    duration = float(time_hours[-1] - time_hours[0])
    relative = (
        len(finite)
        and np.min(finite) >= -interval_hours
        and np.max(finite) <= duration + interval_hours
    )
    absolute = (
        len(finite)
        and np.min(finite) >= time_hours[0] - interval_hours
        and np.max(finite) <= time_hours[-1] + interval_hours
    )
    if relative and not absolute:
        return time_hours[0] + candidate
    if absolute:
        return candidate
    raise ValueError("Unable to resolve pyBOAT ridge time support.")


def analyse_cwt(time_hours, signal, interval_hours):
    from pyboat import WAnalyzer
    analyzer = WAnalyzer(
        np.linspace(MIN_PERIOD_HOURS, MAX_PERIOD_HOURS, CWT_PERIOD_GRID_POINTS),
        interval_hours,
        time_unit_label="h",
    )
    trend = np.asarray(
        analyzer.sinc_smooth(signal, T_c=DETREND_CUTOFF_HOURS), dtype=float
    )
    analyzer.compute_spectrum(np.asarray(signal, dtype=float) - trend, do_plot=False)
    ridge = analyzer.get_maxRidge(
        power_thresh=CWT_POWER_THRESHOLD,
        smoothing_wsize=CWT_RIDGE_SMOOTHING_POINTS,
    )
    if ridge is None:
        ridge = analyzer.ridge_data
    if not isinstance(ridge, pd.DataFrame):
        ridge = pd.DataFrame(ridge)
    if ridge.empty:
        raise ValueError("pyBOAT returned an empty ridge.")
    time_column = find_column(ridge, ["time", "times", "time_hours"])
    period_column = find_column(ridge, ["period", "periods", "ridge_period"])
    amplitude_column = find_column(ridge, ["amplitude", "ridge_amplitude", "amp"])
    phase_column = find_column(ridge, ["phase", "phase_radians", "ridge_phase"])
    if period_column is None or amplitude_column is None or phase_column is None:
        raise ValueError("Required pyBOAT ridge columns were not returned.")
    estimates = pd.DataFrame({
        "time_hours": resolve_ridge_time(
            ridge, time_column, time_hours, interval_hours
        ),
        "period_hours": pd.to_numeric(
            ridge[period_column], errors="coerce"
        ).to_numpy(dtype=float),
        "amplitude_raw_counts": pd.to_numeric(
            ridge[amplitude_column], errors="coerce"
        ).to_numpy(dtype=float),
        "phase_radians": pd.to_numeric(
            ridge[phase_column], errors="coerce"
        ).to_numpy(dtype=float),
    })
    estimates = (
        estimates.replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values("time_hours")
    )
    if len(estimates) < 2 * MIN_SEGMENT_ESTIMATES:
        raise ValueError("Too few finite CWT ridge estimates.")
    return estimates.reset_index(drop=True)


def analyse_stft(time_hours, signal, interval_hours, correction_factor):
    detrended = rolling_mean_detrend(signal, interval_hours)
    nperseg = max(8, int(round(STFT_WINDOW_HOURS / interval_hours)))
    if len(detrended) < nperseg:
        raise ValueError("Recording is shorter than the STFT window.")
    noverlap = min(nperseg - 1, int(round(nperseg * STFT_OVERLAP_FRACTION)))
    frequencies, relative_time, spectrum = stft(
        detrended,
        fs=1.0 / interval_hours,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=max(STFT_NFFT, nperseg),
        detrend=False,
        return_onesided=True,
        boundary=None,
        padded=False,
        scaling="spectrum",
    )
    circadian = (
        (frequencies >= 1.0 / MAX_PERIOD_HOURS)
        & (frequencies <= 1.0 / MIN_PERIOD_HOURS)
    )
    circadian_spectrum = spectrum[circadian]
    circadian_frequencies = frequencies[circadian]
    ridge_index = np.argmax(np.abs(circadian_spectrum), axis=0)
    columns = np.arange(circadian_spectrum.shape[1])
    estimates = pd.DataFrame({
        "time_hours": time_hours[0] + relative_time,
        "period_hours": 1.0 / circadian_frequencies[ridge_index],
        "amplitude_calibrated_counts": (
            2.0
            * np.abs(circadian_spectrum[ridge_index, columns])
            * correction_factor
        ),
    })
    if len(estimates) < 2 * MIN_SEGMENT_ESTIMATES:
        raise ValueError("Too few finite STFT estimates.")
    return estimates


def estimate_log_rate(time_hours, amplitude, mask):
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(time_hours)
        & np.isfinite(amplitude)
        & (amplitude > EPS)
    )
    if np.sum(valid) < MIN_SEGMENT_ESTIMATES:
        return np.nan
    return float(np.polyfit(np.asarray(time_hours)[valid] / 24.0, np.log(np.asarray(amplitude)[valid]), 1)[0])


def estimate_phase_shift(time_hours, phase_radians, pre_mask, post_mask, forskolin_hour):
    phase = np.unwrap(np.asarray(phase_radians, dtype=float))
    pre = np.asarray(pre_mask, dtype=bool) & np.isfinite(phase)
    post = np.asarray(post_mask, dtype=bool) & np.isfinite(phase)
    if np.sum(pre) < MIN_SEGMENT_ESTIMATES or np.sum(post) < MIN_SEGMENT_ESTIMATES:
        return np.nan
    pre_at_f = np.polyval(
        np.polyfit(np.asarray(time_hours)[pre], phase[pre], 1), forskolin_hour
    )
    post_at_f = np.polyval(
        np.polyfit(np.asarray(time_hours)[post], phase[post], 1), forskolin_hour
    )
    return wrap_angle(post_at_f - pre_at_f)


def metadata(group):
    columns = [
        "recording_uid", "recording_id", "animal_id", "sex", "tissue", "batch",
        "sample_code", "scn_region", "level",
    ]
    return {
        column: scalar(group, column)
        if column in group and group[column].notna().any()
        else np.nan
        for column in columns
    }


def summarise_cwt(group, estimates, time_hours):
    result = metadata(group)
    forskolin_day = float(scalar(group, "forskolin_time_days"))
    forskolin_hour = forskolin_day * 24.0
    ridge_time = estimates["time_hours"].to_numpy(dtype=float)
    period = estimates["period_hours"].to_numpy(dtype=float)
    edge_distance = np.minimum(
        ridge_time - time_hours[0], time_hours[-1] - ridge_time
    )
    edge_supported = edge_distance >= CWT_EDGE_EXCLUSION_HOURS
    pre = edge_supported & (ridge_time <= forskolin_hour - CWT_EVENT_EXCLUSION_HOURS)
    post = edge_supported & (ridge_time >= forskolin_hour + CWT_EVENT_EXCLUSION_HOURS)
    if np.sum(pre) < MIN_SEGMENT_ESTIMATES or np.sum(post) < MIN_SEGMENT_ESTIMATES:
        raise ValueError(f"Too few stable CWT estimates for {result['recording_id']}.")
    pre_period = float(np.median(period[pre]))
    post_period = float(np.median(period[post]))
    phase_shift = estimate_phase_shift(
        ridge_time,
        estimates["phase_radians"].to_numpy(dtype=float),
        pre,
        post,
        forskolin_hour,
    )
    stable = pre | post
    near_boundary = (period <= MIN_PERIOD_HOURS + 0.5) | (period >= MAX_PERIOD_HOURS - 0.5)
    result.update({
        "method": "CWT_pyBOAT",
        "forskolin_time_days": forskolin_day,
        "sampling_interval_minutes": float(np.median(np.diff(time_hours)) * 60.0),
        "event_exclusion_hours": CWT_EVENT_EXCLUSION_HOURS,
        "edge_exclusion_hours": CWT_EDGE_EXCLUSION_HOURS,
        "n_pre_estimates": int(np.sum(pre)),
        "n_post_estimates": int(np.sum(post)),
        "period_boundary_fraction_stable": float(np.mean(near_boundary[stable])),
        "pre_period_hours": pre_period,
        "post_period_hours": post_period,
        "period_change_hours_post_minus_pre": post_period - pre_period,
        "phase_shift_angle_radians": phase_shift,
        "phase_shift_hours": phase_shift * post_period / (2.0 * np.pi),
    })
    trajectory = estimates.copy()
    trajectory["edge_supported"] = edge_supported
    trajectory["used_in_recording_summary"] = stable
    trajectory["segment"] = np.select(
        [pre, post], ["stable_pre", "stable_post"], default="transition_or_edge"
    )
    return result, trajectory


def summarise_stft(group, estimates, time_hours):
    result = metadata(group)
    forskolin_day = float(scalar(group, "forskolin_time_days"))
    forskolin_hour = forskolin_day * 24.0
    estimate_time = estimates["time_hours"].to_numpy(dtype=float)
    amplitude = estimates["amplitude_calibrated_counts"].to_numpy(dtype=float)
    period = estimates["period_hours"].to_numpy(dtype=float)
    pre = estimate_time <= forskolin_hour - STFT_EVENT_EXCLUSION_HOURS
    post = estimate_time >= forskolin_hour + STFT_EVENT_EXCLUSION_HOURS
    if np.sum(pre) < MIN_SEGMENT_ESTIMATES or np.sum(post) < MIN_SEGMENT_ESTIMATES:
        raise ValueError(f"Too few stable STFT estimates for {result['recording_id']}.")
    pre_amplitude = float(np.median(amplitude[pre]))
    post_amplitude = float(np.median(amplitude[post]))
    ratio = post_amplitude / pre_amplitude if pre_amplitude > EPS else np.nan
    pre_rate = estimate_log_rate(estimate_time, amplitude, pre)
    post_rate = estimate_log_rate(estimate_time, amplitude, post)
    stable = pre | post
    near_boundary = (period <= MIN_PERIOD_HOURS + 0.5) | (period >= MAX_PERIOD_HOURS - 0.5)
    result.update({
        "method": "STFT",
        "forskolin_time_days": forskolin_day,
        "sampling_interval_minutes": float(np.median(np.diff(time_hours)) * 60.0),
        "event_exclusion_hours": STFT_EVENT_EXCLUSION_HOURS,
        "n_pre_estimates": int(np.sum(pre)),
        "n_post_estimates": int(np.sum(post)),
        "period_boundary_fraction_stable": float(np.mean(near_boundary[stable])),
        "pre_amplitude_counts": pre_amplitude,
        "post_amplitude_counts": post_amplitude,
        "amplitude_ratio_post_over_pre": ratio,
        "amplitude_log_ratio_post_over_pre": (
            float(np.log(ratio)) if ratio > 0 else np.nan
        ),
        "pre_envelope_log_rate_per_day": pre_rate,
        "post_envelope_log_rate_per_day": post_rate,
        "envelope_log_rate_change_post_minus_pre_per_day": post_rate - pre_rate,
    })
    trajectory = estimates.copy()
    trajectory["used_in_recording_summary"] = stable
    trajectory["segment"] = np.select(
        [pre, post], ["stable_pre", "stable_post"], default="transition"
    )
    return result, trajectory


def attach_trajectory_metadata(trajectory, summary):
    output = trajectory.copy()
    metadata_columns = [
        "recording_uid", "recording_id", "animal_id", "sex", "tissue", "batch",
        "sample_code", "scn_region", "level",
    ]
    for column in metadata_columns:
        output[column] = summary[column]
    output["time_days"] = output["time_hours"] / 24.0
    output["hours_from_forskolin"] = (
        output["time_hours"] - summary["forskolin_time_days"] * 24.0
    )
    first = [*metadata_columns, "time_days", "time_hours", "hours_from_forskolin"]
    remaining = [column for column in output.columns if column not in set(first)]
    return output[[*first, *remaining]]


def recording_analyses(data, correction_factor):
    cwt_summaries = []
    stft_summaries = []
    cwt_trajectories = []
    stft_trajectories = []
    for _, group in data.groupby("recording_uid", sort=False):
        time_hours, signal, interval = prepare_recording(group)
        cwt_estimates = analyse_cwt(time_hours, signal, interval)
        stft_estimates = analyse_stft(
            time_hours, signal, interval, correction_factor
        )
        cwt_summary, cwt_trajectory = summarise_cwt(
            group, cwt_estimates, time_hours
        )
        stft_summary, stft_trajectory = summarise_stft(
            group, stft_estimates, time_hours
        )
        stft_summary["amplitude_correction_factor"] = correction_factor
        cwt_summaries.append(cwt_summary)
        stft_summaries.append(stft_summary)
        cwt_trajectories.append(attach_trajectory_metadata(cwt_trajectory, cwt_summary))
        stft_trajectories.append(attach_trajectory_metadata(stft_trajectory, stft_summary))
    return (
        pd.DataFrame(cwt_summaries),
        pd.DataFrame(stft_summaries),
        pd.concat(cwt_trajectories, ignore_index=True),
        pd.concat(stft_trajectories, ignore_index=True),
    )


def aggregate_animal_metrics(cwt, stft_table):
    method_tables = {"CWT_pyBOAT": cwt, "STFT": stft_table}
    rows = []
    for specification in METRICS:
        table = method_tables[specification["method"]]
        for keys, group in table.groupby(["animal_id", "sex", "tissue"], sort=False):
            group = group.copy()
            group["anatomical_unit"] = np.where(
                group["tissue"].eq("DRG"),
                group["level"].astype(str),
                group["scn_region"].fillna("whole_or_unspecified").astype(str),
            )
            unit_values = []
            for _, unit in group.groupby("anatomical_unit", sort=False):
                values = (
                    pd.to_numeric(unit[specification["metric"]], errors="coerce")
                    .dropna()
                    .to_numpy(dtype=float)
                )
                unit_values.append(
                    circular_mean(values)
                    if specification["circular"]
                    else float(np.median(values))
                )
            value = (
                circular_mean(unit_values)
                if specification["circular"]
                else float(np.median(unit_values))
            )
            rows.append({
                "method": specification["method"],
                "metric": specification["metric"],
                "analysis_role": specification["role"],
                "circular": specification["circular"],
                "animal_id": keys[0],
                "sex": keys[1],
                "tissue": keys[2],
                "value": value,
                "n_recordings_collapsed": int(group["recording_id"].nunique()),
                "n_anatomical_units_collapsed": int(group["anatomical_unit"].nunique()),
                "recording_ids": " | ".join(sorted(group["recording_id"].astype(str))),
            })
    return pd.DataFrame(rows)


def seed_for(*parts):
    checksum = zlib.crc32("|".join(map(str, parts)).encode())
    return (RANDOM_SEED + checksum) % (2**32 - 1)


def exact_sign_flip(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    observed = abs(float(np.mean(values)))
    distribution = np.array([
        abs(float(np.mean(values * np.asarray(signs))))
        for signs in product((-1.0, 1.0), repeat=len(values))
    ])
    return float(np.mean(distribution >= observed - 1e-12))


def linear_result(values, seed, perform_test):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(BOOTSTRAP_REPEATS, len(values)), replace=True)
    bootstrap = np.mean(draws, axis=1)
    return {
        "n_animals": len(values),
        "estimate": float(np.mean(values)),
        "mean": float(np.mean(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "resultant_length": np.nan,
        "bootstrap_95ci_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_95ci_high": float(np.quantile(bootstrap, 0.975)),
        "exact_sign_flip_p": exact_sign_flip(values) if perform_test else np.nan,
    }


def circular_result(values, seed):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    estimate = circular_mean(values)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPEATS, len(values)))
    bootstrap = np.angle(np.mean(np.exp(1j * values[indices]), axis=1))
    offsets = wrap_angle(bootstrap - estimate)
    return {
        "n_animals": len(values),
        "estimate": estimate,
        "mean": np.nan,
        "q25": np.nan,
        "q75": np.nan,
        "resultant_length": float(abs(np.mean(np.exp(1j * values)))),
        "bootstrap_95ci_low": float(estimate + np.quantile(offsets, 0.025)),
        "bootstrap_95ci_high": float(estimate + np.quantile(offsets, 0.975)),
        "exact_sign_flip_p": np.nan,
    }


def benjamini_hochberg(values):
    values = np.asarray(values, dtype=float)
    output = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid])]
    adjusted = values[order] * len(order) / np.arange(1, len(order) + 1)
    output[order] = np.minimum(np.minimum.accumulate(adjusted[::-1])[::-1], 1.0)
    return output


def biological_results(animal_metrics):
    rows = []
    within_metrics = [
        "period_change_hours_post_minus_pre",
        "amplitude_log_ratio_post_over_pre",
        "envelope_log_rate_change_post_minus_pre_per_day",
        "phase_shift_angle_radians",
    ]
    between_metrics = ["pre_period_hours", *within_metrics]
    for metric in within_metrics:
        data = animal_metrics.loc[animal_metrics["metric"].eq(metric)].copy()
        for tissue in ["SCN", "DRG"]:
            selected = data.loc[data["tissue"].eq(tissue)]
            circular = bool(selected["circular"].iloc[0])
            role = str(selected["analysis_role"].iloc[0])
            result = (
                circular_result(selected["value"], seed_for(metric, tissue))
                if circular
                else linear_result(selected["value"], seed_for(metric, tissue), True)
            )
            rows.append({
                "comparison": f"{tissue}_vs_zero",
                "method": selected["method"].iloc[0],
                "metric": metric,
                "analysis_role": role,
                "scale": (
                    "circular_radians_with_unwrapped_ci" if circular else "linear"
                ),
                **result,
            })
    for metric in between_metrics:
        data = animal_metrics.loc[animal_metrics["metric"].eq(metric)].copy()
        method = data["method"].iloc[0]
        role = str(data["analysis_role"].iloc[0])
        circular = bool(data["circular"].iloc[0])
        paired = data.pivot(
            index="animal_id", columns="tissue", values="value"
        ).dropna(subset=["SCN", "DRG"])
        raw_difference = (
            paired["DRG"].to_numpy(dtype=float)
            - paired["SCN"].to_numpy(dtype=float)
        )
        differences = wrap_angle(raw_difference) if circular else raw_difference
        result = (
            circular_result(differences, seed_for(metric, "paired"))
            if circular
            else linear_result(differences, seed_for(metric, "paired"), True)
        )
        rows.append({
            "comparison": "DRG_minus_SCN_paired",
            "method": method,
            "metric": metric,
            "analysis_role": role,
            "scale": (
                "circular_radians_with_unwrapped_ci" if circular else "linear"
            ),
            **result,
        })
    output = pd.DataFrame(rows)
    primary = output["analysis_role"].eq("primary") & output["exact_sign_flip_p"].notna()
    output["bh_q_primary_family"] = np.nan
    output.loc[primary, "bh_q_primary_family"] = benjamini_hochberg(
        output.loc[primary, "exact_sign_flip_p"]
    )
    return output


def main():
    data, correction_factor = load_data()
    cwt, stft_table, cwt_trajectory, stft_trajectory = recording_analyses(data, correction_factor)
    animal_metrics = aggregate_animal_metrics(cwt, stft_table)
    results = biological_results(animal_metrics)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    cwt.to_csv(OUTPUT_FOLDER / "cwt_recording_summary.csv", index=False)
    stft_table.to_csv(OUTPUT_FOLDER / "stft_recording_summary.csv", index=False)
    compression = {"method": "gzip", "compresslevel": 6, "mtime": 0}
    cwt_trajectory.to_csv(
        OUTPUT_FOLDER / "cwt_time_resolved_trajectories.csv.gz",
        index=False,
        compression=compression,
    )
    stft_trajectory.to_csv(
        OUTPUT_FOLDER / "stft_time_resolved_trajectories.csv.gz",
        index=False,
        compression=compression,
    )
    animal_metrics.to_csv(OUTPUT_FOLDER / "step4_animal_level_metrics.csv", index=False)
    results.to_csv(OUTPUT_FOLDER / "step4_biological_results.csv", index=False)


if __name__ == "__main__":
    main()
