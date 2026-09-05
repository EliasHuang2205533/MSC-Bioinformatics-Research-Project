from itertools import product
from pathlib import Path
import zlib

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEP4_FOLDER = PROJECT_ROOT / "results" / "step4"
CWT_FILE = STEP4_FOLDER / "cwt_recording_summary.csv"
STFT_FILE = STEP4_FOLDER / "stft_recording_summary.csv"
OUTPUT_FOLDER = PROJECT_ROOT / "results" / "step5"
LEVEL_ORDER = ["T5", "T7", "T10", "T12", "L1", "L2", "L3", "L4", "L5"]
FORMAL_LEVELS = ["T5", "T7", "T10", "T12", "L1", "L2", "L3", "L4"]
LEVEL_POSITION = {
    "T5": 5.0,
    "T7": 7.0,
    "T10": 10.0,
    "T12": 12.0,
    "L1": 13.0,
    "L2": 14.0,
    "L3": 15.0,
    "L4": 16.0,
    "L5": 17.0,
}
LEVEL_REGION = {
    level: "thoracic" if level.startswith("T") else "lumbar"
    for level in LEVEL_ORDER
}
BOOTSTRAP_REPEATS = 2000
PERMUTATION_REPEATS = 20000
RANDOM_SEED = 20260905

METRICS = [
    {
        "method": "CWT_pyBOAT",
        "metric": "baseline_period_hours",
        "source": "pre_period_hours",
        "role": "primary",
        "circular": False,
    },
    {
        "method": "CWT_pyBOAT",
        "metric": "period_change_hours_post_minus_pre",
        "source": "period_change_hours_post_minus_pre",
        "role": "primary",
        "circular": False,
    },
    {
        "method": "STFT",
        "metric": "amplitude_log_ratio_post_over_pre",
        "source": "amplitude_log_ratio_post_over_pre",
        "role": "primary",
        "circular": False,
    },
    {
        "method": "STFT",
        "metric": "pre_amplitude_counts",
        "source": "pre_amplitude_counts",
        "role": "descriptive",
        "circular": False,
    },
    {
        "method": "STFT",
        "metric": "post_amplitude_counts",
        "source": "post_amplitude_counts",
        "role": "descriptive",
        "circular": False,
    },
    {
        "method": "STFT",
        "metric": "pre_envelope_log_rate_per_day",
        "source": "pre_envelope_log_rate_per_day",
        "role": "secondary",
        "circular": False,
    },
    {
        "method": "STFT",
        "metric": "post_envelope_log_rate_per_day",
        "source": "post_envelope_log_rate_per_day",
        "role": "secondary",
        "circular": False,
    },
    {
        "method": "STFT",
        "metric": "envelope_log_rate_change_post_minus_pre_per_day",
        "source": "envelope_log_rate_change_post_minus_pre_per_day",
        "role": "secondary",
        "circular": False,
    },
    {
        "method": "CWT_pyBOAT",
        "metric": "phase_shift_angle_radians",
        "source": "phase_shift_angle_radians",
        "role": "secondary_circular",
        "circular": True,
    },
]


def circular_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.angle(np.mean(np.exp(1j * values)))) if len(values) else np.nan


def load_step4():
    cwt = pd.read_csv(CWT_FILE)
    stft_table = pd.read_csv(STFT_FILE)
    required = {"recording_id", "animal_id", "sex", "tissue", "level"}
    if required - set(cwt.columns) or required - set(stft_table.columns):
        raise ValueError("STEP4 recording summaries are missing required columns.")
    cwt = cwt.loc[cwt["tissue"].eq("DRG")].copy()
    stft_table = stft_table.loc[stft_table["tissue"].eq("DRG")].copy()
    complete = (
        len(cwt) == 40
        and len(stft_table) == 40
        and cwt["animal_id"].nunique() == 6
        and stft_table["animal_id"].nunique() == 6
    )
    if not complete:
        raise RuntimeError("STEP4 DRG summaries do not contain 40 recordings from six mice.")
    return {"CWT_pyBOAT": cwt, "STFT": stft_table}


def make_recording_metrics(tables):
    rows = []
    for specification in METRICS:
        table = tables[specification["method"]]
        values = pd.to_numeric(table[specification["source"]], errors="coerce")
        for index, row in table.iterrows():
            level = str(row["level"])
            rows.append({
                "method": specification["method"],
                "metric": specification["metric"],
                "analysis_role": specification["role"],
                "circular": specification["circular"],
                "recording_id": row["recording_id"],
                "animal_id": row["animal_id"],
                "sex": row["sex"],
                "level": level,
                "spinal_position": LEVEL_POSITION[level],
                "spinal_region": LEVEL_REGION[level],
                "value": float(values.loc[index]),
            })
    output = pd.DataFrame(rows)
    if not np.isfinite(output["value"]).all():
        raise RuntimeError("STEP5 input metrics contain non-finite values.")
    return output


def aggregate_animal_level(recording_metrics):
    rows = []
    keys = [
        "method", "metric", "analysis_role", "circular", "animal_id", "sex",
        "level", "spinal_position", "spinal_region",
    ]
    for key, group in recording_metrics.groupby(keys, sort=False, observed=True):
        values = group["value"].to_numpy(dtype=float)
        rows.append({
            **dict(zip(keys, key)),
            "value": (
                circular_mean(values) if bool(key[3]) else float(np.median(values))
            ),
            "n_recordings_collapsed": int(group["recording_id"].nunique()),
            "recording_ids": " | ".join(sorted(group["recording_id"].astype(str))),
        })
    return pd.DataFrame(rows)


def level_summary(animal_level):
    rows = []
    keys = [
        "method", "metric", "analysis_role", "circular", "level",
        "spinal_position",
    ]
    for key, group in animal_level.groupby(keys, sort=False, observed=True):
        values = group["value"].to_numpy(dtype=float)
        circular = bool(key[3])
        rows.append({
            **dict(zip(keys, key)),
            "n_animals": int(group["animal_id"].nunique()),
            "mean": np.nan if circular else float(np.mean(values)),
            "standard_deviation": (
                np.nan
                if circular or len(values) < 2
                else float(np.std(values, ddof=1))
            ),
            "median": np.nan if circular else float(np.median(values)),
            "q25": np.nan if circular else float(np.quantile(values, 0.25)),
            "q75": np.nan if circular else float(np.quantile(values, 0.75)),
            "circular_mean_radians": circular_mean(values) if circular else np.nan,
            "resultant_length": (
                float(abs(np.mean(np.exp(1j * values)))) if circular else np.nan
            ),
        })
    output = pd.DataFrame(rows)
    output["formal_level"] = output["level"].isin(FORMAL_LEVELS)
    return output.sort_values(["method", "metric", "spinal_position"])


def seed_for(*parts):
    return (RANDOM_SEED + zlib.crc32("|".join(map(str, parts)).encode())) % (2**32 - 1)


def quantile_interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def exact_sign_flip(values):
    values = np.asarray(values, dtype=float)
    observed = abs(float(np.mean(values)))
    distribution = np.array([
        abs(float(np.mean(values * np.asarray(signs))))
        for signs in product((-1.0, 1.0), repeat=len(values))
    ])
    return float(np.mean(distribution >= observed - 1e-12))


def region_test(data, seed):
    region = (
        data.groupby(["animal_id", "spinal_region"], observed=True)["value"]
        .median()
        .unstack("spinal_region")
        .dropna(subset=["thoracic", "lumbar"])
    )
    differences = region["lumbar"].to_numpy(dtype=float) - region["thoracic"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(BOOTSTRAP_REPEATS, len(differences)))
    bootstrap = np.mean(differences[indices], axis=1)
    low, high = quantile_interval(bootstrap)
    return {
        "n_animals": len(differences),
        "thoracic_mean": float(region["thoracic"].mean()),
        "lumbar_mean": float(region["lumbar"].mean()),
        "mean_difference_lumbar_minus_thoracic": float(np.mean(differences)),
        "median_difference_lumbar_minus_thoracic": float(np.median(differences)),
        "bootstrap_95ci_low": low,
        "bootstrap_95ci_high": high,
        "exact_sign_flip_p": exact_sign_flip(differences),
    }


def permuted_within_animals(data, repeats, rng):
    values = data["value"].to_numpy(dtype=float)
    output = np.repeat(values[:, None], repeats, axis=1)
    for indices in data.groupby("animal_id", sort=False).indices.values():
        indices = np.asarray(indices, dtype=int)
        for number in range(repeats):
            output[indices, number] = values[indices][rng.permutation(len(indices))]
    return output


def position_trend(data, seed):
    data = data.sort_values(["animal_id", "spinal_position"]).reset_index(drop=True)
    position = data["spinal_position"].to_numpy(dtype=float)
    values = data["value"].to_numpy(dtype=float)
    centred = np.empty_like(position)
    contributions = []
    for animal_id, indices in data.groupby("animal_id", sort=False).indices.items():
        indices = np.asarray(indices, dtype=int)
        x = position[indices] - np.mean(position[indices])
        centred[indices] = x
        contributions.append((animal_id, float(np.dot(x, values[indices])), float(np.dot(x, x))))
    denominator = float(np.dot(centred, centred))
    slope = float(np.dot(centred, values) / denominator)
    rng = np.random.default_rng(seed)
    permuted = permuted_within_animals(data, PERMUTATION_REPEATS, rng)
    permutation_slopes = centred @ permuted / denominator
    p_value = float((1 + np.sum(np.abs(permutation_slopes) >= abs(slope) - 1e-12)) / (PERMUTATION_REPEATS + 1))
    contributions = pd.DataFrame(contributions, columns=["animal_id", "numerator", "denominator"])
    indices = rng.integers(0, len(contributions), size=(BOOTSTRAP_REPEATS, len(contributions)))
    numerator = contributions["numerator"].to_numpy()[indices].sum(axis=1)
    bootstrap_denominator = contributions["denominator"].to_numpy()[indices].sum(axis=1)
    bootstrap = np.divide(
        numerator,
        bootstrap_denominator,
        out=np.full(BOOTSTRAP_REPEATS, np.nan),
        where=bootstrap_denominator > 0,
    )
    low, high = quantile_interval(bootstrap)
    return {
        "n_animals": int(data["animal_id"].nunique()),
        "n_animal_level_observations": len(data),
        "fixed_effect_slope_per_spinal_segment": slope,
        "bootstrap_95ci_low": low,
        "bootstrap_95ci_high": high,
        "within_animal_permutation_p": p_value,
        "n_permutations": PERMUTATION_REPEATS,
    }


def design_matrix(categories):
    categories = categories.astype(str)
    levels = list(dict.fromkeys(categories))
    indicators = [
        (categories.to_numpy() == level).astype(float)
        for level in levels[1:]
    ]
    return np.column_stack([np.ones(len(categories)), *indicators])


def level_omnibus(data, seed):
    data = data.sort_values(["animal_id", "spinal_position"]).reset_index(drop=True)
    values = data["value"].to_numpy(dtype=float)
    animal_design = design_matrix(data["animal_id"])
    level_design = pd.get_dummies(
        pd.Categorical(data["level"], categories=FORMAL_LEVELS, ordered=True),
        drop_first=True,
        dtype=float,
    ).to_numpy()
    full_design = np.column_stack([animal_design, level_design])
    rank_reduced = int(np.linalg.matrix_rank(animal_design))
    rank_full = int(np.linalg.matrix_rank(full_design))
    numerator_df = rank_full - rank_reduced
    denominator_df = len(data) - rank_full
    residual_reduced = np.eye(len(data)) - animal_design @ np.linalg.pinv(animal_design)
    residual_full = np.eye(len(data)) - full_design @ np.linalg.pinv(full_design)
    def statistic(matrix):
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        sse_reduced = np.sum((residual_reduced @ matrix) ** 2, axis=0)
        sse_full = np.sum((residual_full @ matrix) ** 2, axis=0)
        return ((sse_reduced - sse_full) / numerator_df) / (sse_full / denominator_df)
    observed = float(statistic(values)[0])
    rng = np.random.default_rng(seed)
    permuted = statistic(permuted_within_animals(data, PERMUTATION_REPEATS, rng))
    p_value = float((1 + np.sum(permuted >= observed - 1e-12)) / (PERMUTATION_REPEATS + 1))
    return {
        "n_animals": int(data["animal_id"].nunique()),
        "n_animal_level_observations": len(data),
        "partial_f_statistic": observed,
        "df_numerator": numerator_df,
        "df_denominator": denominator_df,
        "within_animal_permutation_p": p_value,
        "n_permutations": PERMUTATION_REPEATS,
    }


def benjamini_hochberg(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = values[order] * len(values) / np.arange(1, len(values) + 1)
    output = np.empty(len(values))
    output[order] = np.minimum(np.minimum.accumulate(adjusted[::-1])[::-1], 1.0)
    return output


def formal_tests(animal_level):
    primary = animal_level.loc[
        animal_level["analysis_role"].eq("primary")
        & animal_level["level"].isin(FORMAL_LEVELS)
    ].copy()
    region_rows = []
    trend_rows = []
    omnibus_rows = []
    for key, group in primary.groupby(["method", "metric", "analysis_role"], sort=False, observed=True):
        metadata = dict(zip(["method", "metric", "analysis_role"], key))
        region_rows.append({**metadata, **region_test(group, seed_for(*key, "region"))})
        trend_rows.append({**metadata, **position_trend(group, seed_for(*key, "trend"))})
        omnibus_rows.append({**metadata, **level_omnibus(group, seed_for(*key, "omnibus"))})
    region = pd.DataFrame(region_rows)
    trend = pd.DataFrame(trend_rows)
    omnibus = pd.DataFrame(omnibus_rows)
    region_p = region[["exact_sign_flip_p"]].rename(
        columns={"exact_sign_flip_p": "p"}
    ).assign(source="region", row=region.index)
    trend_p = trend[["within_animal_permutation_p"]].rename(
        columns={"within_animal_permutation_p": "p"}
    ).assign(source="trend", row=trend.index)
    combined = pd.concat([region_p, trend_p], ignore_index=True)
    combined["q"] = benjamini_hochberg(combined["p"])
    region["bh_q_primary_family"] = np.nan
    trend["bh_q_primary_family"] = np.nan
    for row in combined.itertuples(index=False):
        if row.source == "region":
            region.loc[int(row.row), "bh_q_primary_family"] = row.q
        else:
            trend.loc[int(row.row), "bh_q_primary_family"] = row.q
    return region, trend, omnibus


def main():
    tables = load_step4()
    recording_metrics = make_recording_metrics(tables)
    animal_level = aggregate_animal_level(recording_metrics)
    summary = level_summary(animal_level)
    region, trend, omnibus = formal_tests(animal_level)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    animal_level.to_csv(OUTPUT_FOLDER / "step5_animal_level_metrics.csv", index=False)
    summary.to_csv(OUTPUT_FOLDER / "step5_level_summary.csv", index=False)
    region.to_csv(OUTPUT_FOLDER / "step5_thoracic_lumbar_results.csv", index=False)
    trend.to_csv(OUTPUT_FOLDER / "step5_rostrocaudal_trend_results.csv", index=False)
    omnibus.to_csv(OUTPUT_FOLDER / "step5_level_omnibus_exploratory.csv", index=False)


if __name__ == "__main__":
    main()
