from pathlib import Path
import re

import matplotlib
import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, filtfilt, hilbert

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "results" / "step1a" / "tables" / "all_lumicycle_analysis_long_format.csv"
OUTPUT_FOLDER = PROJECT_ROOT / "results" / "step1b"
TABLE_FOLDER = OUTPUT_FOLDER / "tables"
FIGURE_FOLDER = OUTPUT_FOLDER / "figures" / "normalised_real_traces"
OUTPUT_PATH = TABLE_FOLDER / "empirical_pre_post_features_by_recording.csv"

INITIAL_EXCLUSION_DAYS = 0.25
TREATMENT_BUFFER_DAYS = 0.25
MIN_PERIOD_HOURS = 20.0
MAX_PERIOD_HOURS = 30.0
MIN_SEGMENT_POINTS = 80
EXPECTED_INTERVAL_MIN = 20.0
INTERVAL_TOLERANCE_MIN = 1.0
TREND_24H_POINTS = 72
SMOOTH_2H_POINTS = 6

METADATA_COLUMNS = [
    "recording_uid",
    "recording_id",
    "file",
    "batch",
    "position",
    "sample_code",
    "tissue",
    "scn_region",
    "spinal_region",
    "level",
]

FEATURE_COLUMNS = [
    "post_pre_amplitude_proxy_ratio",
    "pre_period_proxy_hours",
    "post_period_proxy_hours",
    "pre_envelope_log_rate_per_day",
    "post_envelope_log_rate_per_day",
    "baseline_step_to_pre_amplitude",
    "pre_amplitude_proxy",
    "pre_drift_slope_per_day",
    "post_drift_slope_per_day",
]

REQUIRED_COLUMNS = set(METADATA_COLUMNS) | {
    "measurement_datetime",
    "time_days",
    "counts_sec",
    "forskolin_status",
    "forskolin_time_days",
}


def safe_filename(value):
    return re.sub(r'[\\/:*?"<>|]+', "_", str(value))


def scalar_value(recording, column):
    values = recording[column].dropna().unique()
    if len(values) > 1:
        raise ValueError(f"{recording['recording_uid'].iloc[0]} has multiple values in {column}")
    return values[0] if len(values) == 1 else np.nan


def prepare_segment(segment, label):
    data = segment[["measurement_datetime", "time_days", "counts_sec"]].copy()
    data["measurement_datetime"] = pd.to_datetime(data["measurement_datetime"], errors="coerce")
    data["time_days"] = pd.to_numeric(data["time_days"], errors="coerce")
    data["counts_sec"] = pd.to_numeric(data["counts_sec"], errors="coerce")
    data = data.dropna().sort_values("measurement_datetime").reset_index(drop=True)
    if len(data) > 1:
        intervals_min = data["measurement_datetime"].diff().dropna().dt.total_seconds().to_numpy() / 60.0
        invalid = (
            ~np.isfinite(intervals_min)
            | (intervals_min < EXPECTED_INTERVAL_MIN - INTERVAL_TOLERANCE_MIN)
            | (intervals_min > EXPECTED_INTERVAL_MIN + INTERVAL_TOLERANCE_MIN)
        )
        if np.any(invalid):
            raise ValueError(
                f"{label} contains intervals outside 20 +/- 1 min: "
                f"{intervals_min[invalid].tolist()}"
            )
    return data


def amplitude_proxy(segment):
    signal = segment["counts_sec"].to_numpy(dtype=float)
    if len(signal) < 10:
        return np.nan
    return float((np.percentile(signal, 95) - np.percentile(signal, 5)) / 2.0)


def rolling_baseline(segment):
    return segment["counts_sec"].rolling(
        TREND_24H_POINTS,
        center=True,
        min_periods=TREND_24H_POINTS // 2,
    ).median()


def baseline_model(segment, reference_day):
    if len(segment) < 20:
        return np.nan, np.nan
    baseline = rolling_baseline(segment)
    valid = baseline.notna()
    if valid.sum() < 20:
        return np.nan, np.nan
    relative_time = segment.loc[valid, "time_days"].to_numpy(dtype=float) - reference_day
    coefficients = np.polyfit(relative_time, baseline.loc[valid].to_numpy(dtype=float), 1)
    return float(coefficients[0]), float(coefficients[1])


def period_proxy(segment):
    signal = segment["counts_sec"].to_numpy(dtype=float)
    if len(signal) < MIN_SEGMENT_POINTS:
        return np.nan
    interval_hours = EXPECTED_INTERVAL_MIN / 60.0
    prepared_signal = detrend(signal, type="linear") * np.hanning(len(signal))
    frequencies = np.fft.rfftfreq(len(prepared_signal), d=interval_hours)
    power = np.abs(np.fft.rfft(prepared_signal)) ** 2
    positive = frequencies > 0
    frequencies = frequencies[positive]
    power = power[positive]
    if len(frequencies) == 0:
        return np.nan
    periods = 1.0 / frequencies
    circadian = (periods >= MIN_PERIOD_HOURS) & (periods <= MAX_PERIOD_HOURS)
    if not np.any(circadian):
        return np.nan
    return float(periods[circadian][np.argmax(power[circadian])])


def envelope_log_rate(segment):
    if len(segment) < MIN_SEGMENT_POINTS:
        return np.nan
    time = segment["time_days"].to_numpy(dtype=float)
    signal = segment["counts_sec"].to_numpy(dtype=float)
    baseline = rolling_baseline(segment).bfill().ffill().to_numpy(dtype=float)
    detrended_signal = signal - baseline
    sampling_frequency = 60.0 / EXPECTED_INTERVAL_MIN
    nyquist = sampling_frequency / 2.0
    low = (1.0 / MAX_PERIOD_HOURS) / nyquist
    high = (1.0 / MIN_PERIOD_HOURS) / nyquist
    try:
        numerator, denominator = butter(2, [low, high], btype="bandpass")
        envelope = np.abs(hilbert(filtfilt(numerator, denominator, detrended_signal)))
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return np.nan
    median_envelope = float(np.median(envelope))
    if not np.isfinite(median_envelope) or median_envelope <= 0:
        return np.nan
    trim = max(5, int(len(envelope) * 0.1))
    if len(envelope) > 2 * trim + 10:
        time = time[trim:-trim]
        envelope = envelope[trim:-trim]
    valid = np.isfinite(time) & np.isfinite(envelope) & (envelope > 0)
    if valid.sum() < 10:
        return np.nan
    floor = max(median_envelope * 1e-6, 1e-12)
    return float(np.polyfit(time[valid], np.log(np.maximum(envelope[valid], floor)), 1)[0])


def recording_features(recording):
    recording = recording.sort_values("measurement_datetime").copy()
    row = {column: scalar_value(recording, column) for column in METADATA_COLUMNS}
    status = scalar_value(recording, "forskolin_status")
    row["forskolin_status"] = status
    row["forskolin_time_days"] = np.nan
    row.update({column: np.nan for column in FEATURE_COLUMNS})
    if status == "not_administered":
        return row, None
    if status != "administered_estimated":
        raise ValueError(f"{row['recording_uid']} has unresolved forskolin status")
    treatment_day = float(scalar_value(recording, "forskolin_time_days"))
    if not np.isfinite(treatment_day):
        raise ValueError(f"{row['recording_uid']} has no valid forskolin time")
    recording_start = float(recording["time_days"].min())
    reference_start = recording_start + INITIAL_EXCLUSION_DAYS
    reference_end = treatment_day - TREATMENT_BUFFER_DAYS
    pre = prepare_segment(
        recording[
            (recording["time_days"] >= reference_start)
            & (recording["time_days"] < reference_end)
        ],
        f"{row['recording_uid']} stable pre",
    )
    post = prepare_segment(
        recording[recording["time_days"] > treatment_day + TREATMENT_BUFFER_DAYS],
        f"{row['recording_uid']} stable post",
    )
    pre_amplitude = amplitude_proxy(pre)
    post_amplitude = amplitude_proxy(post)
    pre_drift, pre_baseline_at_f = baseline_model(pre, treatment_day)
    post_drift, post_baseline_at_f = baseline_model(post, treatment_day)
    baseline_step = (
        post_baseline_at_f - pre_baseline_at_f
        if np.isfinite(pre_baseline_at_f) and np.isfinite(post_baseline_at_f)
        else np.nan
    )
    row.update(
        {
            "forskolin_time_days": treatment_day,
            "post_pre_amplitude_proxy_ratio": (
                post_amplitude / pre_amplitude
                if np.isfinite(pre_amplitude) and pre_amplitude > 0 and np.isfinite(post_amplitude)
                else np.nan
            ),
            "pre_period_proxy_hours": period_proxy(pre),
            "post_period_proxy_hours": period_proxy(post),
            "pre_envelope_log_rate_per_day": envelope_log_rate(pre),
            "post_envelope_log_rate_per_day": envelope_log_rate(post),
            "baseline_step_to_pre_amplitude": (
                baseline_step / pre_amplitude
                if np.isfinite(pre_amplitude) and pre_amplitude > 0 and np.isfinite(baseline_step)
                else np.nan
            ),
            "pre_amplitude_proxy": pre_amplitude,
            "pre_drift_slope_per_day": pre_drift,
            "post_drift_slope_per_day": post_drift,
        }
    )
    plot_values = {
        "treatment_day": treatment_day,
        "recording_start": recording_start,
        "reference_start": reference_start,
        "reference_end": reference_end,
        "pre_amplitude": pre_amplitude,
    }
    return row, plot_values


def make_plot(recording, feature_row, plot_values):
    if plot_values is None or not np.isfinite(plot_values["pre_amplitude"]) or plot_values["pre_amplitude"] <= 0:
        return False
    recording = recording.sort_values("measurement_datetime").reset_index(drop=True)
    reference = recording.loc[
        (recording["time_days"] >= plot_values["reference_start"])
        & (recording["time_days"] < plot_values["reference_end"]),
        "counts_sec",
    ].dropna()
    normalised = (recording["counts_sec"] - float(reference.median())) / plot_values["pre_amplitude"]
    intervals_min = recording["measurement_datetime"].diff().dt.total_seconds().div(60.0)
    run_id = intervals_min.gt(EXPECTED_INTERVAL_MIN + INTERVAL_TOLERANCE_MIN).cumsum()
    smoothed = normalised.groupby(run_id).transform(
        lambda values: values.rolling(SMOOTH_2H_POINTS, center=True, min_periods=1).mean()
    )
    figure, axis = plt.subplots(figsize=(11, 4.5))
    for index, current_run in enumerate(run_id.unique()):
        selected = run_id.eq(current_run)
        axis.plot(
            recording.loc[selected, "time_days"],
            normalised.loc[selected],
            color="0.70",
            linewidth=0.7,
            alpha=0.65,
            label="Raw normalised" if index == 0 else None,
        )
        axis.plot(
            recording.loc[selected, "time_days"],
            smoothed.loc[selected],
            color="#1f77b4",
            linewidth=1.5,
            label="2-h rolling mean" if index == 0 else None,
        )
    axis.axvspan(
        plot_values["recording_start"],
        plot_values["reference_start"],
        color="0.75",
        alpha=0.20,
        label="Excluded initial 6 h",
    )
    axis.axvspan(
        plot_values["treatment_day"] - TREATMENT_BUFFER_DAYS,
        plot_values["treatment_day"] + TREATMENT_BUFFER_DAYS,
        color="#f6c85f",
        alpha=0.24,
        label="Excluded treatment +/-6-h buffer",
    )
    axis.axvline(plot_values["treatment_day"], color="#d62728", linestyle="--", linewidth=1.2, label="Estimated forskolin time")
    axis.axhline(0, color="0.25", linewidth=0.7)
    axis.set_xlabel("Elapsed time (days)")
    axis.set_ylabel("Normalised PER2::LUC signal")
    axis.set_title(str(feature_row["recording_uid"]))
    axis.legend(frameon=False, ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURE_FOLDER / f"{safe_filename(feature_row['recording_uid'])}.png", dpi=180)
    plt.close(figure)
    return True


def main():
    TABLE_FOLDER.mkdir(parents=True, exist_ok=True)
    FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT_PATH, low_memory=False)
    missing = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    data["measurement_datetime"] = pd.to_datetime(data["measurement_datetime"], errors="coerce")
    data["time_days"] = pd.to_numeric(data["time_days"], errors="coerce")
    data["counts_sec"] = pd.to_numeric(data["counts_sec"], errors="coerce")
    if data[["measurement_datetime", "time_days", "counts_sec"]].isna().any().any():
        raise ValueError("STEP1A analysis table contains missing measurement values")
    rows = []
    plots = []
    for _, recording in data.groupby("recording_uid", sort=True, dropna=False):
        row, plot_values = recording_features(recording)
        rows.append(row)
        plots.append((recording, row, plot_values))
    features = pd.DataFrame(rows).sort_values("recording_uid").reset_index(drop=True)
    status_counts = features["forskolin_status"].value_counts().to_dict()
    if len(features) != 61 or status_counts != {"administered_estimated": 55, "not_administered": 6}:
        raise RuntimeError(f"Unexpected recording or treatment counts: {len(features)}, {status_counts}")
    treated = features["forskolin_status"].eq("administered_estimated")
    if features.loc[treated, FEATURE_COLUMNS].isna().any().any():
        missing_features = features.loc[treated, ["recording_uid", *FEATURE_COLUMNS]].set_index("recording_uid").isna()
        raise RuntimeError(f"Missing treated features: {missing_features.loc[missing_features.any(axis=1)].to_dict('index')}")
    features.to_csv(OUTPUT_PATH, index=False)
    plot_count = sum(make_plot(recording, row, plot_values) for recording, row, plot_values in plots)
    if plot_count != 55:
        raise RuntimeError(f"Unexpected number of normalised trace plots: {plot_count}")


if __name__ == "__main__":
    main()
