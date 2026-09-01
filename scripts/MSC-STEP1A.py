from pathlib import Path
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FOLDER = PROJECT_ROOT / "data" / "raw" / "All"
OUTPUT_FOLDER = PROJECT_ROOT / "results" / "step1a"
TABLE_FOLDER = OUTPUT_FOLDER / "tables"
RAW_PLOT_FOLDER = OUTPUT_FOLDER / "figures" / "raw_trace_qc"
FOURIER_PLOT_FOLDER = OUTPUT_FOLDER / "figures" / "fourier_qc"

RAW_OUTPUT_PATH = TABLE_FOLDER / "all_lumicycle_raw_long_format.csv"
ANALYSIS_OUTPUT_PATH = TABLE_FOLDER / "all_lumicycle_analysis_long_format.csv"
SUMMARY_OUTPUT_PATH = TABLE_FOLDER / "recording_summary.csv"
GAP_OUTPUT_PATH = TABLE_FOLDER / "sampling_gap_report.csv"
EXCLUSION_OUTPUT_PATH = TABLE_FOLDER / "excluded_observations.csv"

MAKE_RAW_PLOTS = True
MAKE_FOURIER_PLOTS = True
EXPECTED_INTERVAL_MIN = 20.0
INTERVAL_TOLERANCE_MIN = 1.0
GAP_THRESHOLD_MIN = EXPECTED_INTERVAL_MIN + INTERVAL_TOLERANCE_MIN
FOURIER_MIN_PERIOD_HOURS = 4.0
FOURIER_MAX_PERIOD_HOURS = 80.0
SMOOTHING_HOURS = 2.0

EXPERIMENT_SCHEDULE = {
    pd.Timestamp("2025-06-26"): pd.Timestamp("2025-07-03 17:24"),
    pd.Timestamp("2025-07-03"): pd.Timestamp("2025-07-10 17:20"),
    pd.Timestamp("2025-07-28"): pd.Timestamp("2025-08-04 17:19"),
    pd.Timestamp("2025-08-11"): pd.Timestamp("2025-08-18 16:30"),
    pd.Timestamp("2025-10-03"): pd.Timestamp("2025-10-10 17:11"),
}

EXCLUDED_OBSERVATIONS = {
    ("250728::6A-SCNrM1_Raw.csv", pd.Timestamp("2025-07-28 23:20")): (
        "competing_observation_in_20min_sequence",
        pd.Timestamp("2025-07-28 23:22"),
    ),
    ("250728::7A-SCNcM1_Raw.csv", pd.Timestamp("2025-07-28 15:42")): (
        "competing_observation_in_20min_sequence",
        pd.Timestamp("2025-07-28 15:45"),
    ),
}

METADATA_COLUMNS = [
    "recording_uid",
    "recording_id",
    "file",
    "path",
    "batch",
    "position",
    "sample_code",
    "tissue",
    "scn_region",
    "spinal_region",
    "level",
]


def safe_filename(value):
    return re.sub(r'[\\/:*?"<>|]+', "_", str(value))


def parse_sample_name(sample_name):
    value = str(sample_name).strip().upper()
    if "SCN" in value:
        if "SCNC" in value:
            region = "caudal"
        elif "SCNM" in value:
            region = "medial"
        elif "SCNR" in value:
            region = "rostral"
        else:
            region = "whole_or_unspecified"
        return {
            "tissue": "SCN",
            "scn_region": region,
            "spinal_region": None,
            "level": None,
        }
    match = re.search(r"(C\d+|T\d+|L\d+|S\d+)", value)
    if match is None:
        return {
            "tissue": "unknown",
            "scn_region": None,
            "spinal_region": None,
            "level": None,
        }
    level = match.group(1)
    return {
        "tissue": "DRG",
        "scn_region": None,
        "spinal_region": {
            "C": "cervical",
            "T": "thoracic",
            "L": "lumbar",
            "S": "sacral",
        }[level[0]],
        "level": level,
    }


def parse_filename(file_path):
    clean_name = re.sub(r"(?:\(Raw\)|_Raw|Raw)$", "", file_path.stem).strip()
    if "-" in clean_name:
        position, sample_code = clean_name.split("-", 1)
        position = position.strip()
        sample_code = sample_code.strip()
    else:
        position = None
        sample_code = clean_name
    metadata = {
        "recording_uid": f"{file_path.parent.name}::{file_path.name}",
        "recording_id": clean_name,
        "file": file_path.name,
        "path": file_path.relative_to(PROJECT_ROOT).as_posix(),
        "batch": file_path.parent.name,
        "position": position,
        "sample_code": sample_code,
    }
    metadata.update(parse_sample_name(sample_code))
    return metadata


def read_recording(file_path):
    metadata = parse_filename(file_path)
    data = pd.read_csv(file_path, header=1)
    data.columns = [str(column).strip() for column in data.columns]
    data = data.rename(columns={"Time (days)": "time_days", "counts/sec": "counts_sec"})
    required = ["Date", "Time (hr:min)", "time_days", "counts_sec"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"{metadata['path']} is missing columns: {missing}")
    data = data[required].copy()
    data.insert(0, "source_row", np.arange(len(data), dtype=int))
    data["time_days"] = pd.to_numeric(data["time_days"], errors="coerce")
    data["counts_sec"] = pd.to_numeric(data["counts_sec"], errors="coerce")
    data["measurement_datetime"] = pd.to_datetime(
        data["Date"].astype(str).str.strip()
        + " "
        + data["Time (hr:min)"].astype(str).str.strip(),
        errors="coerce",
    )
    valid = data["measurement_datetime"].notna()
    data.loc[valid, "Date"] = data.loc[valid, "measurement_datetime"].dt.strftime("%Y-%m-%d")
    for column, value in metadata.items():
        data[column] = value
    return data


def build_raw_table():
    files = sorted(DATA_FOLDER.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {DATA_FOLDER}")
    raw = pd.concat([read_recording(file_path) for file_path in files], ignore_index=True)
    if raw["measurement_datetime"].isna().any():
        bad = raw.loc[raw["measurement_datetime"].isna(), ["path", "source_row"]]
        raise ValueError(f"Missing measurement times: {bad.to_dict('records')}")
    if raw["time_days"].isna().any():
        bad = raw.loc[raw["time_days"].isna(), ["path", "source_row"]]
        raise ValueError(f"Missing elapsed times: {bad.to_dict('records')}")
    if raw["counts_sec"].isna().any():
        bad = raw.loc[raw["counts_sec"].isna(), ["path", "source_row"]]
        raise ValueError(f"Missing counts: {bad.to_dict('records')}")
    return raw


def exclude_conflicting_observations(raw):
    excluded_rows = []
    excluded_indices = []
    for (recording_uid, timestamp), (reason, retained_timestamp) in EXCLUDED_OBSERVATIONS.items():
        match = raw.index[
            raw["recording_uid"].eq(recording_uid)
            & raw["measurement_datetime"].eq(timestamp)
        ]
        retained = raw.index[
            raw["recording_uid"].eq(recording_uid)
            & raw["measurement_datetime"].eq(retained_timestamp)
        ]
        if len(match) != 1 or len(retained) != 1:
            raise RuntimeError(
                f"Unable to resolve exclusion {recording_uid} at {timestamp}: "
                f"excluded={len(match)}, retained={len(retained)}"
            )
        row = raw.loc[match[0], [
            "recording_uid",
            "recording_id",
            "file",
            "path",
            "batch",
            "source_row",
            "Date",
            "Time (hr:min)",
            "measurement_datetime",
            "time_days",
            "counts_sec",
        ]].to_dict()
        row["exclusion_reason"] = reason
        row["retained_conflicting_observation_datetime"] = retained_timestamp
        excluded_rows.append(row)
        excluded_indices.append(match[0])
    exclusions = pd.DataFrame(excluded_rows)
    analysis = raw.drop(index=excluded_indices).copy().reset_index(drop=True)
    return analysis, exclusions


def build_recording_overview(raw, analysis):
    raw_counts = raw.groupby("recording_uid", dropna=False).size().rename("n_raw_observations")
    analysis_counts = (
        analysis.groupby("recording_uid", dropna=False).size().rename("n_analysis_observations")
    )
    overview = (
        raw.groupby(METADATA_COLUMNS, dropna=False)
        .agg(
            recording_start_datetime=("measurement_datetime", "min"),
            recording_end_datetime=("measurement_datetime", "max"),
            min_time_days=("time_days", "min"),
            max_time_days=("time_days", "max"),
        )
        .reset_index()
    )
    overview["recording_duration_days"] = overview["max_time_days"] - overview["min_time_days"]
    overview = overview.merge(raw_counts, on="recording_uid", validate="one_to_one")
    overview = overview.merge(analysis_counts, on="recording_uid", validate="one_to_one")
    return overview


def assign_forskolin(overview, analysis):
    rows = []
    for _, recording in overview.iterrows():
        start_date = recording["recording_start_datetime"].normalize()
        restart = EXPERIMENT_SCHEDULE.get(start_date, pd.NaT)
        scheduled_date = restart.normalize() if pd.notna(restart) else pd.NaT
        if pd.isna(restart):
            rows.append(
                {
                    "recording_uid": recording["recording_uid"],
                    "forskolin_status": "not_administered",
                    "experiment_start_date": pd.NaT,
                    "scheduled_forskolin_date": pd.NaT,
                    "reported_restart_datetime": pd.NaT,
                    "forskolin_interval_start_datetime": pd.NaT,
                    "forskolin_interval_end_datetime": pd.NaT,
                    "forskolin_estimated_datetime": pd.NaT,
                    "forskolin_time_days": np.nan,
                    "forskolin_assignment_reason": "no_scheduled_treatment_for_recording_start_date",
                }
            )
            continue
        if recording["recording_end_datetime"] < restart:
            rows.append(
                {
                    "recording_uid": recording["recording_uid"],
                    "forskolin_status": "not_administered",
                    "experiment_start_date": start_date,
                    "scheduled_forskolin_date": scheduled_date,
                    "reported_restart_datetime": restart,
                    "forskolin_interval_start_datetime": pd.NaT,
                    "forskolin_interval_end_datetime": pd.NaT,
                    "forskolin_estimated_datetime": pd.NaT,
                    "forskolin_time_days": np.nan,
                    "forskolin_assignment_reason": "recording_ended_before_scheduled_restart",
                }
            )
            continue
        group = analysis.loc[
            analysis["recording_uid"].eq(recording["recording_uid"])
        ].sort_values("measurement_datetime")
        previous = group.loc[group["measurement_datetime"] < restart]
        following = group.loc[group["measurement_datetime"] >= restart]
        if previous.empty or following.empty:
            raise RuntimeError(f"No observations bracket restart for {recording['recording_uid']}")
        interval_start = previous.iloc[-1]
        interval_end = following.iloc[0]
        width_min = (
            interval_end["measurement_datetime"] - interval_start["measurement_datetime"]
        ).total_seconds() / 60.0
        if abs(width_min - EXPECTED_INTERVAL_MIN) > INTERVAL_TOLERANCE_MIN:
            raise RuntimeError(
                f"Treatment bracket for {recording['recording_uid']} is {width_min:g} min"
            )
        estimated_datetime = interval_start["measurement_datetime"] + (
            interval_end["measurement_datetime"] - interval_start["measurement_datetime"]
        ) / 2
        estimated_day = (
            float(interval_start["time_days"]) + float(interval_end["time_days"])
        ) / 2.0
        rows.append(
            {
                "recording_uid": recording["recording_uid"],
                "forskolin_status": "administered_estimated",
                "experiment_start_date": start_date,
                "scheduled_forskolin_date": scheduled_date,
                "reported_restart_datetime": restart,
                "forskolin_interval_start_datetime": interval_start["measurement_datetime"],
                "forskolin_interval_end_datetime": interval_end["measurement_datetime"],
                "forskolin_estimated_datetime": estimated_datetime,
                "forskolin_time_days": estimated_day,
                "forskolin_assignment_reason": "midpoint_of_actual_observations_bracketing_restart",
            }
        )
    assignments = pd.DataFrame(rows)
    overview = overview.merge(assignments, on="recording_uid", validate="one_to_one")
    counts = overview["forskolin_status"].value_counts().to_dict()
    if counts.get("administered_estimated", 0) != 55 or counts.get("not_administered", 0) != 6:
        raise RuntimeError(f"Unexpected forskolin status counts: {counts}")
    treated = overview["forskolin_status"].eq("administered_estimated")
    bracket_widths = (
        overview.loc[treated, "forskolin_interval_end_datetime"]
        - overview.loc[treated, "forskolin_interval_start_datetime"]
    ).dt.total_seconds() / 60.0
    if len(bracket_widths) != 55 or not np.allclose(bracket_widths, EXPECTED_INTERVAL_MIN):
        raise RuntimeError("Forskolin brackets are not all exactly 20 min")
    return overview, assignments


def add_assignments(table, assignments):
    return table.merge(assignments, on="recording_uid", how="left", validate="many_to_one")


def build_gap_report(analysis):
    rows = []
    for _, recording in analysis.groupby("recording_uid", sort=True, dropna=False):
        recording = recording.sort_values("measurement_datetime").reset_index(drop=True)
        intervals = recording["measurement_datetime"].diff().dt.total_seconds().div(60.0)
        for index in np.flatnonzero(intervals.gt(GAP_THRESHOLD_MIN).to_numpy()):
            previous = recording.iloc[index - 1]
            following = recording.iloc[index]
            treatment_day = following["forskolin_time_days"]
            if pd.notna(treatment_day):
                previous_relative = (float(previous["time_days"]) - float(treatment_day)) * 24.0
                following_relative = (float(following["time_days"]) - float(treatment_day)) * 24.0
            else:
                previous_relative = np.nan
                following_relative = np.nan
            rows.append(
                {
                    "recording_uid": following["recording_uid"],
                    "recording_id": following["recording_id"],
                    "file": following["file"],
                    "path": following["path"],
                    "batch": following["batch"],
                    "forskolin_status": following["forskolin_status"],
                    "previous_observation_datetime": previous["measurement_datetime"],
                    "next_observation_datetime": following["measurement_datetime"],
                    "interval_minutes": float(intervals.iloc[index]),
                    "previous_time_days": float(previous["time_days"]),
                    "next_time_days": float(following["time_days"]),
                    "previous_hours_from_forskolin": previous_relative,
                    "next_hours_from_forskolin": following_relative,
                }
            )
    return pd.DataFrame(rows)


def add_sampling_summary(overview, analysis, gaps, exclusions):
    sampling_rows = []
    for recording_uid, recording in analysis.groupby("recording_uid", sort=False, dropna=False):
        times = recording.sort_values("measurement_datetime")["measurement_datetime"]
        intervals = times.diff().dt.total_seconds().div(60.0).dropna()
        sampling_rows.append(
            {
                "recording_uid": recording_uid,
                "median_sampling_interval_min": float(intervals.median()),
                "maximum_sampling_interval_min": float(intervals.max()),
            }
        )
    sampling = pd.DataFrame(sampling_rows)
    gap_counts = gaps.groupby("recording_uid").size().rename("n_sampling_gaps")
    exclusion_counts = exclusions.groupby("recording_uid").size().rename("n_excluded_observations")
    overview = overview.merge(sampling, on="recording_uid", validate="one_to_one")
    overview = overview.merge(gap_counts, on="recording_uid", how="left")
    overview = overview.merge(exclusion_counts, on="recording_uid", how="left")
    overview["n_sampling_gaps"] = overview["n_sampling_gaps"].fillna(0).astype(int)
    overview["n_excluded_observations"] = overview["n_excluded_observations"].fillna(0).astype(int)
    return overview


def make_raw_plot(recording):
    recording = recording.sort_values("time_days")
    time = recording["time_days"].to_numpy(dtype=float)
    signal = recording["counts_sec"].to_numpy(dtype=float)
    centered = signal - np.median(signal)
    points = max(1, int(round(SMOOTHING_HOURS * 60.0 / EXPECTED_INTERVAL_MIN)))
    smoothed = pd.Series(centered).rolling(points, center=True, min_periods=1).mean()
    figure, axis = plt.subplots(figsize=(11, 4.5))
    axis.plot(time, centered, color="0.70", linewidth=0.7, alpha=0.7, label="Raw centred")
    axis.plot(time, smoothed, color="#1f77b4", linewidth=1.4, label="2-h rolling mean")
    treatment_day = recording["forskolin_time_days"].dropna()
    if len(treatment_day):
        axis.axvline(float(treatment_day.iloc[0]), color="#d62728", linestyle="--", linewidth=1.2, label="Estimated forskolin time")
    axis.axhline(0.0, color="0.25", linewidth=0.7)
    axis.set_xlabel("Elapsed time (days)")
    axis.set_ylabel("Centred counts/sec")
    axis.set_title(str(recording["recording_uid"].iloc[0]))
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(RAW_PLOT_FOLDER / f"{safe_filename(recording['recording_uid'].iloc[0])}.png", dpi=180)
    plt.close(figure)


def make_fourier_plot(recording):
    recording = recording.sort_values("time_days")
    time_hours = recording["time_days"].to_numpy(dtype=float) * 24.0
    signal = recording["counts_sec"].to_numpy(dtype=float)
    interval_hours = EXPECTED_INTERVAL_MIN / 60.0
    uniform_time = np.arange(time_hours[0], time_hours[-1] + interval_hours * 0.5, interval_hours)
    uniform_signal = np.interp(uniform_time, time_hours, signal)
    frequencies = np.fft.rfftfreq(len(uniform_signal), d=interval_hours)
    power = np.abs(np.fft.rfft(uniform_signal - np.mean(uniform_signal))) ** 2
    valid = frequencies > 0
    periods = 1.0 / frequencies[valid]
    power = power[valid]
    selected = (periods >= FOURIER_MIN_PERIOD_HOURS) & (periods <= FOURIER_MAX_PERIOD_HOURS)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(periods[selected], power[selected], color="#1f77b4", linewidth=1.3)
    axis.set_xlim(FOURIER_MIN_PERIOD_HOURS, FOURIER_MAX_PERIOD_HOURS)
    axis.set_xlabel("Period (h)")
    axis.set_ylabel("Fourier power")
    axis.set_title(str(recording["recording_uid"].iloc[0]))
    figure.tight_layout()
    figure.savefig(FOURIER_PLOT_FOLDER / f"{safe_filename(recording['recording_uid'].iloc[0])}.png", dpi=180)
    plt.close(figure)


def main():
    for folder in [TABLE_FOLDER, RAW_PLOT_FOLDER, FOURIER_PLOT_FOLDER]:
        folder.mkdir(parents=True, exist_ok=True)
    raw = build_raw_table()
    if len(raw) != 54079 or raw["recording_uid"].nunique() != 61:
        raise RuntimeError(
            f"Unexpected raw data dimensions: rows={len(raw)}, recordings={raw['recording_uid'].nunique()}"
        )
    analysis, exclusions = exclude_conflicting_observations(raw)
    if len(analysis) != 54077 or len(exclusions) != 2:
        raise RuntimeError(
            f"Unexpected analysis dimensions: rows={len(analysis)}, exclusions={len(exclusions)}"
        )
    overview = build_recording_overview(raw, analysis)
    overview, assignments = assign_forskolin(overview, analysis)
    raw = add_assignments(raw, assignments)
    analysis = add_assignments(analysis, assignments)
    gaps = build_gap_report(analysis)
    if len(gaps) != 58:
        raise RuntimeError(f"Unexpected number of sampling gaps: {len(gaps)}")
    overview = add_sampling_summary(overview, analysis, gaps, exclusions)
    raw.to_csv(RAW_OUTPUT_PATH, index=False, date_format="%Y-%m-%d %H:%M:%S")
    analysis.to_csv(ANALYSIS_OUTPUT_PATH, index=False, date_format="%Y-%m-%d %H:%M:%S")
    overview.to_csv(SUMMARY_OUTPUT_PATH, index=False, date_format="%Y-%m-%d %H:%M:%S")
    gaps.to_csv(GAP_OUTPUT_PATH, index=False, date_format="%Y-%m-%d %H:%M:%S")
    exclusions.to_csv(EXCLUSION_OUTPUT_PATH, index=False, date_format="%Y-%m-%d %H:%M:%S")
    for _, recording in analysis.groupby("recording_uid", sort=True, dropna=False):
        if MAKE_RAW_PLOTS:
            make_raw_plot(recording)
        if MAKE_FOURIER_PLOTS:
            make_fourier_plot(recording)


if __name__ == "__main__":
    main()
