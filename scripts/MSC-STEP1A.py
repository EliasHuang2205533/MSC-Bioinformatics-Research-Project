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
RAW_PLOT_FOLDER = OUTPUT_FOLDER / "figures" / "raw_traces"
FOURIER_PLOT_FOLDER = OUTPUT_FOLDER / "figures" / "fourier_spectra"

MAKE_RAW_PLOTS = True
MAKE_FOURIER_PLOTS = True
EXPECTED_SAMPLING_INTERVAL_MIN = 20
SAMPLING_TOLERANCE_MIN = 5
RECORDING_NODE_ROUND_MIN = 10
SHORT_RECORDING_DAYS = 10

NO_FORSKOLIN_SHORT_IDS = {
    "2A-SCNM4M",
    "2B-SCNM4R",
    "5C-SCNM5",
    "6D-SCNCM1",
    "7C-SCNM6",
    "8B-L4",
}

REPORTED_RESTART_GROUPS = {
    "2025-07-03 17:24": [
        "1C-SCN",
        "5C-T12",
        "5D-T10",
        "6A-T7",
        "6B-T5",
        "7C-L1",
        "7D-L2",
        "8A-L3",
    ],
    "2025-07-10 17:20": [
        "1B-L1M2",
        "1D-L2M2",
        "2C-L3M2",
        "2D-L4M2",
        "3A-L5M2",
        "3B-T10M2",
        "4B-T12M2",
        "4C-T5M1",
        "4D-T7M1",
        "5A-T10M1",
        "5B-T12M2",
        "6C-SCNMM1",
        "7A-SCNMM2",
        "7B-SCNCM2",
        "8B-L2M1",
        "8C-L3M1",
        "8D-L4-1M1",
    ],
    "2025-08-04 17:19": [
        "6A-SCNRM1",
        "6B-SCNRM2",
        "7A-SCNCM1",
        "7B-SCNCM2",
    ],
    "2025-08-18 16:30": [
        "1A-1FSCN",
        "1B-2MSCN",
        "1C-3FSCN",
        "1D-4FSCN",
        "2C-2ML1",
        "2D-1FL2",
        "3A-1FL4",
        "3D-1FT10",
        "4A-2MT10",
        "4C-2MT5",
        "4D-1FL4",
        "5A-2ML2",
        "5C-1FT7",
        "6A-1FL1",
        "6B-2ML3",
        "7C-1FT12",
        "7D-2MT12",
        "8A-1FT5",
        "8C-2ML4",
        "8D-2MT7",
    ],
    "2025-10-10 17:11": [
        "2A-M1SCN",
        "2B-1MT7",
        "2C-M1L1",
        "2D-M1L4",
        "4A-M2SCN",
        "4B-M3SCN",
    ],
}


def normalize_recording_id(value):
    return re.sub(r"\s+", "", str(value)).upper()


def parse_sample_name(sample_name):
    value = str(sample_name).strip().upper()
    if "SCN" in value:
        if "SCNC" in value:
            scn_region = "caudal"
        elif "SCNM" in value:
            scn_region = "medial"
        elif "SCNR" in value:
            scn_region = "rostral"
        else:
            scn_region = "whole_or_unspecified"
        return {
            "tissue": "SCN",
            "scn_region": scn_region,
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
    spinal_region = {
        "C": "cervical",
        "T": "thoracic",
        "L": "lumbar",
        "S": "sacral",
    }[level[0]]
    return {
        "tissue": "DRG",
        "scn_region": None,
        "spinal_region": spinal_region,
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
        "file": file_path.name,
        "path": file_path.relative_to(PROJECT_ROOT).as_posix(),
        "batch": file_path.parent.name,
        "recording_id": clean_name,
        "position": position,
        "sample_code": sample_code,
        "recording_key": normalize_recording_id(clean_name),
        "recording_uid": f"{file_path.parent.name}::{file_path.name}",
    }
    metadata.update(parse_sample_name(sample_code))
    return metadata


def build_forskolin_schedule():
    rows = []
    for restart_text, recording_ids in REPORTED_RESTART_GROUPS.items():
        restart = pd.Timestamp(restart_text)
        for recording_id in recording_ids:
            rows.append(
                {
                    "schedule_recording_id": recording_id,
                    "recording_key": normalize_recording_id(recording_id),
                    "reported_restart_datetime": restart,
                }
            )
    schedule = pd.DataFrame(rows)
    schedule.insert(0, "schedule_id", np.arange(len(schedule), dtype=int))
    return schedule


def read_recording(file_path, metadata):
    data = pd.read_csv(file_path, header=1)
    data.columns = [str(column).strip() for column in data.columns]
    data = data.rename(
        columns={
            "Time (days)": "time_days",
            "counts/sec": "counts_sec",
        }
    )
    missing = [
        column
        for column in ["Date", "Time (hr:min)", "time_days", "counts_sec"]
        if column not in data.columns
    ]
    if missing:
        raise ValueError(f"{metadata['path']}: missing required columns {missing}")
    data = data[["Date", "Time (hr:min)", "time_days", "counts_sec"]].copy()
    data["time_days"] = pd.to_numeric(data["time_days"], errors="coerce")
    data["counts_sec"] = pd.to_numeric(data["counts_sec"], errors="coerce")
    datetime_text = (
        data["Date"].astype(str).str.strip()
        + " "
        + data["Time (hr:min)"].astype(str).str.strip()
    )
    data["measurement_datetime"] = pd.to_datetime(datetime_text, errors="coerce")
    valid_datetime = data["measurement_datetime"].notna()
    data.loc[valid_datetime, "Date"] = data.loc[
        valid_datetime, "measurement_datetime"
    ].dt.strftime("%Y-%m-%d")
    for key, value in metadata.items():
        data[key] = value
    return data


def build_recording_overview(raw_all):
    overview = (
        raw_all.groupby(
            [
                "recording_uid",
                "path",
                "batch",
                "file",
                "recording_id",
                "recording_key",
            ],
            dropna=False,
        )
        .agg(
            recording_start_datetime=("measurement_datetime", "min"),
            recording_end_datetime=("measurement_datetime", "max"),
            min_time_days=("time_days", "min"),
            max_time_days=("time_days", "max"),
        )
        .reset_index()
    )
    overview["recording_duration_days"] = (
        overview["max_time_days"] - overview["min_time_days"]
    )
    normalized_short_ids = {
        normalize_recording_id(value) for value in NO_FORSKOLIN_SHORT_IDS
    }
    overview["confirmed_no_forskolin"] = (
        overview["recording_key"].isin(normalized_short_ids)
        & (overview["recording_duration_days"] < SHORT_RECORDING_DAYS)
    )
    return overview, normalized_short_ids


def assign_forskolin(row, schedule):
    if row["confirmed_no_forskolin"]:
        return pd.Series(
            {
                "forskolin_status": "not_administered",
                "matched_schedule_id": pd.NA,
                "reported_restart_datetime": pd.NaT,
                "n_schedule_candidates": 0,
            }
        )
    candidates = schedule[schedule["recording_key"] == row["recording_key"]]
    start = row["recording_start_datetime"]
    end = row["recording_end_datetime"]
    if pd.isna(start) or pd.isna(end):
        candidates = candidates.iloc[0:0]
    else:
        candidates = candidates[
            candidates["reported_restart_datetime"].between(start, end)
        ]
    if len(candidates) == 1:
        match = candidates.iloc[0]
        return pd.Series(
            {
                "forskolin_status": "administered_estimated",
                "matched_schedule_id": int(match["schedule_id"]),
                "reported_restart_datetime": match["reported_restart_datetime"],
                "n_schedule_candidates": 1,
            }
        )
    return pd.Series(
        {
            "forskolin_status": (
                "missing_schedule" if len(candidates) == 0 else "ambiguous_schedule"
            ),
            "matched_schedule_id": pd.NA,
            "reported_restart_datetime": pd.NaT,
            "n_schedule_candidates": len(candidates),
        }
    )


def empty_forskolin_interval(restart_on_node=pd.NA):
    return pd.Series(
        {
            "restart_on_file_20min_node": restart_on_node,
            "forskolin_window_start_datetime": pd.NaT,
            "forskolin_window_end_datetime": pd.NaT,
            "forskolin_estimated_datetime": pd.NaT,
        }
    )


def derive_forskolin_interval(row, raw_all):
    if row["forskolin_status"] != "administered_estimated":
        return empty_forskolin_interval()
    reported_time = row["reported_restart_datetime"]
    if pd.isna(reported_time):
        return empty_forskolin_interval()
    file_nodes = (
        raw_all.loc[
            raw_all["recording_uid"] == row["recording_uid"],
            "measurement_datetime",
        ]
        .dropna()
        .dt.round(f"{RECORDING_NODE_ROUND_MIN}min")
        .drop_duplicates()
        .sort_values()
    )
    restart_on_node = bool((file_nodes == reported_time).any())
    previous_nodes = file_nodes[file_nodes < reported_time]
    following_nodes = file_nodes[
        file_nodes == reported_time if restart_on_node else file_nodes > reported_time
    ]
    if len(previous_nodes) == 0 or len(following_nodes) == 0:
        return empty_forskolin_interval(restart_on_node)
    interval_start = previous_nodes.iloc[-1]
    interval_end = following_nodes.iloc[0]
    interval_width_min = (interval_end - interval_start).total_seconds() / 60
    lower = EXPECTED_SAMPLING_INTERVAL_MIN - SAMPLING_TOLERANCE_MIN
    upper = EXPECTED_SAMPLING_INTERVAL_MIN + SAMPLING_TOLERANCE_MIN
    if not lower <= interval_width_min <= upper:
        return empty_forskolin_interval(restart_on_node)
    estimated_time = interval_start + (interval_end - interval_start) / 2
    return pd.Series(
        {
            "restart_on_file_20min_node": restart_on_node,
            "forskolin_window_start_datetime": interval_start,
            "forskolin_window_end_datetime": interval_end,
            "forskolin_estimated_datetime": estimated_time,
        }
    )


def add_elapsed_forskolin_times(overview, raw_all):
    clock_start = (
        raw_all.assign(
            estimated_clock_start=lambda frame: frame["measurement_datetime"]
            - pd.to_timedelta(frame["time_days"], unit="D")
        )
        .groupby("recording_uid", dropna=False)["estimated_clock_start"]
        .median()
        .rename("estimated_clock_start")
        .reset_index()
    )
    overview = overview.merge(
        clock_start,
        on="recording_uid",
        how="left",
        validate="one_to_one",
    )
    for source, target in [
        ("forskolin_estimated_datetime", "forskolin_time_days"),
        ("forskolin_window_start_datetime", "forskolin_window_start_days"),
        ("forskolin_window_end_datetime", "forskolin_window_end_days"),
    ]:
        overview[target] = (
            overview[source] - overview["estimated_clock_start"]
        ).dt.total_seconds() / 86400
    return overview


def validate_forskolin_assignments(overview, schedule, normalized_short_ids):
    invalid_recordings = overview[
        ~overview["forskolin_status"].isin(
            ["administered_estimated", "not_administered"]
        )
        | (
            overview["forskolin_status"].eq("administered_estimated")
            & overview["forskolin_estimated_datetime"].isna()
        )
    ]
    found_short_ids = set(
        overview.loc[overview["confirmed_no_forskolin"], "recording_key"]
    )
    missing_short_ids = normalized_short_ids - found_short_ids
    match_counts = (
        overview["matched_schedule_id"].dropna().astype(int).value_counts()
    )
    schedule_match_counts = schedule["schedule_id"].map(match_counts).fillna(0).astype(int)
    invalid_schedule = schedule[schedule_match_counts.ne(1)]
    if len(invalid_recordings) or missing_short_ids or len(invalid_schedule):
        recording_details = invalid_recordings[
            ["path", "forskolin_status", "n_schedule_candidates"]
        ].to_dict("records")
        schedule_details = invalid_schedule[
            ["schedule_recording_id", "reported_restart_datetime"]
        ].to_dict("records")
        raise RuntimeError(
            "Forskolin assignment validation failed: "
            f"recordings={recording_details}; "
            f"missing_no_forskolin_ids={sorted(missing_short_ids)}; "
            f"schedule_entries={schedule_details}"
        )


def median_sampling_interval_min(values):
    values = pd.to_numeric(values, errors="coerce").dropna().sort_values()
    if len(values) < 2:
        return np.nan
    return float(np.median(np.diff(values)) * 24 * 60)


def count_time_gaps(values, gap_factor=1.5):
    values = pd.to_numeric(values, errors="coerce").dropna().sort_values()
    if len(values) < 3:
        return 0
    differences = np.diff(values)
    median_difference = np.median(differences)
    if not np.isfinite(median_difference) or median_difference <= 0:
        return 0
    return int(np.sum(differences > gap_factor * median_difference))


def count_duplicate_times(values):
    values = pd.to_numeric(values, errors="coerce").dropna()
    return int(values.duplicated().sum())


def build_qc_table(raw_all, overview):
    summary = (
        raw_all.groupby(
            [
                "recording_uid",
                "path",
                "batch",
                "file",
                "recording_id",
                "position",
                "sample_code",
                "tissue",
                "scn_region",
                "spinal_region",
                "level",
            ],
            dropna=False,
        )
        .agg(
            min_time_days=("time_days", "min"),
            max_time_days=("time_days", "max"),
            n_points=("time_days", "size"),
            median_sampling_interval_min=("time_days", median_sampling_interval_min),
            n_time_gaps=("time_days", count_time_gaps),
            n_duplicate_times=("time_days", count_duplicate_times),
            n_missing_time=("time_days", lambda values: int(values.isna().sum())),
            n_missing_counts=("counts_sec", lambda values: int(values.isna().sum())),
            min_counts=("counts_sec", "min"),
            max_counts=("counts_sec", "max"),
            mean_counts=("counts_sec", "mean"),
            median_counts=("counts_sec", "median"),
        )
        .reset_index()
    )
    summary["recording_duration_days"] = (
        summary["max_time_days"] - summary["min_time_days"]
    )
    assignment_columns = [
        "recording_uid",
        "forskolin_status",
        "reported_restart_datetime",
        "restart_on_file_20min_node",
        "forskolin_window_start_datetime",
        "forskolin_window_end_datetime",
        "forskolin_estimated_datetime",
        "forskolin_time_days",
        "forskolin_window_start_days",
        "forskolin_window_end_days",
    ]
    summary = summary.merge(
        overview[assignment_columns],
        on="recording_uid",
        how="left",
        validate="one_to_one",
    )
    return summary


def safe_filename(value):
    return re.sub(r'[\\/:*?"<>|]', "_", str(value))


def make_raw_trace_plots(raw_all):
    for _, data in raw_all.groupby("recording_uid", sort=False):
        data = data.sort_values("time_days").copy()
        median_interval = median_sampling_interval_min(data["time_days"])
        window_points = (
            max(1, int(round(120 / median_interval)))
            if np.isfinite(median_interval) and median_interval > 0
            else 6
        )
        data["smooth_2h_visual"] = (
            data["counts_sec"]
            .rolling(window=window_points, center=True, min_periods=1)
            .mean()
        )
        batch = data["batch"].iloc[0]
        recording_id = data["recording_id"].iloc[0]
        title = (
            f"{recording_id} | {data['tissue'].iloc[0]} | "
            f"sample={data['sample_code'].iloc[0]} | "
            f"level={data['level'].iloc[0]} | "
            f"SCN={data['scn_region'].iloc[0]} | batch={batch}"
        )
        figure, axis = plt.subplots(figsize=(12, 4))
        axis.plot(
            data["time_days"],
            data["counts_sec"],
            alpha=0.35,
            linewidth=0.8,
            label="Raw counts/sec",
        )
        axis.plot(
            data["time_days"],
            data["smooth_2h_visual"],
            linewidth=1.5,
            label="2 h rolling mean, visual only",
        )
        if data["forskolin_status"].iloc[0] == "administered_estimated":
            axis.axvspan(
                data["forskolin_window_start_days"].iloc[0],
                data["forskolin_window_end_days"].iloc[0],
                color="tab:orange",
                alpha=0.18,
                label="Forskolin timing interval",
            )
        axis.set_xlabel("Time (days)")
        axis.set_ylabel("Counts/sec")
        axis.set_title(title)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(
            RAW_PLOT_FOLDER
            / f"{safe_filename(batch)}_{safe_filename(recording_id)}_raw_trace_qc.png",
            dpi=300,
        )
        plt.close(figure)


def diagnostic_fourier_spectrum(data):
    data = (
        data[["time_days", "counts_sec"]]
        .dropna()
        .sort_values("time_days")
        .copy()
    )
    if len(data) < 10:
        return None
    time_days = data["time_days"].to_numpy(dtype=float)
    signal = data["counts_sec"].to_numpy(dtype=float)
    signal = signal - np.mean(signal)
    interval_hours = np.median(np.diff(time_days)) * 24
    if not np.isfinite(interval_hours) or interval_hours <= 0:
        return None
    frequencies = np.fft.rfftfreq(len(signal), d=interval_hours)
    power = np.abs(np.fft.rfft(signal)) ** 2
    valid = frequencies > 0
    if not np.any(valid):
        return None
    return 1 / frequencies[valid], power[valid]


def make_fourier_plots(raw_all):
    for _, data in raw_all.groupby("recording_uid", sort=False):
        result = diagnostic_fourier_spectrum(data)
        if result is None:
            continue
        period_hours, power = result
        plot_mask = (period_hours >= 4) & (period_hours <= 80)
        if not np.any(plot_mask):
            continue
        batch = data["batch"].iloc[0]
        recording_id = data["recording_id"].iloc[0]
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.plot(period_hours[plot_mask], power[plot_mask])
        axis.axvline(24, linestyle="--", linewidth=1, label="24 h")
        axis.axvspan(20, 30, alpha=0.15, label="20–30 h circadian range")
        axis.set_xlabel("Period (hours)")
        axis.set_ylabel("Fourier power")
        axis.set_title(f"Diagnostic Fourier spectrum: {recording_id}")
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(
            FOURIER_PLOT_FOLDER
            / f"{safe_filename(batch)}_{safe_filename(recording_id)}_fourier_qc.png",
            dpi=300,
        )
        plt.close(figure)


def main():
    files = sorted(
        [path for path in DATA_FOLDER.rglob("*") if path.suffix.lower() == ".csv"],
        key=lambda path: path.as_posix().lower(),
    )
    if not files:
        raise FileNotFoundError(f"No CSV files found in {DATA_FOLDER}")
    metadata_rows = [parse_filename(path) for path in files]
    if len({row["recording_uid"] for row in metadata_rows}) != len(metadata_rows):
        raise RuntimeError("Duplicate recording_uid values were found")
    raw_all = pd.concat(
        [
            read_recording(path, metadata)
            for path, metadata in zip(files, metadata_rows)
        ],
        ignore_index=True,
    )
    schedule = build_forskolin_schedule()
    overview, normalized_short_ids = build_recording_overview(raw_all)
    assignments = overview.apply(assign_forskolin, axis=1, schedule=schedule)
    overview = pd.concat([overview, assignments], axis=1)
    intervals = overview.apply(derive_forskolin_interval, axis=1, raw_all=raw_all)
    overview = pd.concat([overview, intervals], axis=1)
    overview = add_elapsed_forskolin_times(overview, raw_all)
    validate_forskolin_assignments(overview, schedule, normalized_short_ids)
    assignment_columns = [
        "recording_uid",
        "forskolin_status",
        "reported_restart_datetime",
        "restart_on_file_20min_node",
        "forskolin_window_start_datetime",
        "forskolin_window_end_datetime",
        "forskolin_estimated_datetime",
        "forskolin_time_days",
        "forskolin_window_start_days",
        "forskolin_window_end_days",
    ]
    raw_all = raw_all.merge(
        overview[assignment_columns],
        on="recording_uid",
        how="left",
        validate="many_to_one",
    )
    qc_table = build_qc_table(raw_all, overview)
    assignment_output_columns = [
        "recording_uid",
        "path",
        "batch",
        "file",
        "recording_id",
        "recording_start_datetime",
        "recording_end_datetime",
        "min_time_days",
        "max_time_days",
        "recording_duration_days",
        "forskolin_status",
        "reported_restart_datetime",
        "restart_on_file_20min_node",
        "forskolin_window_start_datetime",
        "forskolin_window_end_datetime",
        "forskolin_estimated_datetime",
        "forskolin_time_days",
        "forskolin_window_start_days",
        "forskolin_window_end_days",
    ]
    for folder in [TABLE_FOLDER, RAW_PLOT_FOLDER, FOURIER_PLOT_FOLDER]:
        folder.mkdir(parents=True, exist_ok=True)
    raw_all.drop(columns="recording_key").to_csv(
        TABLE_FOLDER / "all_lumicycle_raw_long_format.csv",
        index=False,
    )
    qc_table.to_csv(TABLE_FOLDER / "file_summary_qc.csv", index=False)
    overview[assignment_output_columns].to_csv(
        TABLE_FOLDER / "forskolin_assignment_check.csv",
        index=False,
    )
    if MAKE_RAW_PLOTS:
        make_raw_trace_plots(raw_all)
    if MAKE_FOURIER_PLOTS:
        make_fourier_plots(raw_all)


if __name__ == "__main__":
    main()
