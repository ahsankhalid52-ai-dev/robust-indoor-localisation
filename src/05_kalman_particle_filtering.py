"""
MSc Project - Step 5: Kalman and Particle Filtering Extension

Purpose:
    Use the best coordinate baseline from Step 3/4 (KNN coordinate regression)
    as a measurement source, then apply Kalman filtering and particle filtering
    to coordinate sequences constructed from validation samples.

Important limitation:
    UJIIndoorLoc is a WiFi fingerprint dataset, not a true continuous pedestrian
    trajectory dataset. Therefore, sequence construction is a controlled proxy
    for evaluating whether temporal/probabilistic smoothing can reduce noisy
    coordinate estimates. The script records this limitation in the summary.

Inputs expected from earlier steps:
    data/processed/train_features_scaled.csv
    data/processed/validation_features_scaled.csv
    data/processed/train_targets.csv
    data/processed/validation_targets.csv

Outputs:
    results/tables/05_filtering_metrics.csv
    results/tables/05_filtering_summary.json
    results/tables/05_filtering_predictions.csv
    results/experiment_logs/05_kalman_particle_filtering_summary.txt
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from experiment_config import (
    FEATURE_DTYPE, RANDOM_SEED, configuration_metadata, corrupt_features,
    filter_seed, filtering_scenarios, make_knn_regressor, make_rf_classifier,
)

SCENARIOS = filtering_scenarios()
def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dirs(root: Path) -> Dict[str, Path]:
    dirs = {
        "tables": root / "results" / "tables",
        "logs": root / "results" / "experiment_logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def load_inputs(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    processed = root / "data" / "processed"
    train_x_path = processed / "train_features_scaled.csv"
    val_x_path = processed / "validation_features_scaled.csv"
    train_y_path = processed / "train_targets.csv"
    val_y_path = processed / "validation_targets.csv"

    missing = [p for p in [train_x_path, val_x_path, train_y_path, val_y_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required Step 2 processed files: " + ", ".join(str(p) for p in missing))

    print(f"Loading train features: {train_x_path}")
    print(f"Loading validation features: {val_x_path}")
    print(f"Loading train targets: {train_y_path}")
    print(f"Loading validation targets: {val_y_path}")

    return (
        pd.read_csv(train_x_path),
        pd.read_csv(val_x_path),
        pd.read_csv(train_y_path),
        pd.read_csv(val_y_path),
    )


def make_building_floor_label(df: pd.DataFrame) -> pd.Series:
    return df["BUILDINGID"].astype(str) + "_" + df["FLOOR"].astype(str)


def point_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))


def coordinate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    errors = point_errors(y_true, y_pred)
    return {
        "longitude_mae": round(float(mean_absolute_error(y_true[:, 0], y_pred[:, 0])), 6),
        "latitude_mae": round(float(mean_absolute_error(y_true[:, 1], y_pred[:, 1])), 6),
        "longitude_rmse": round(float(math.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))), 6),
        "latitude_rmse": round(float(math.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))), 6),
        "mean_positioning_error_m": round(float(errors.mean()), 6),
        "median_positioning_error_m": round(float(np.median(errors)), 6),
        "p75_positioning_error_m": round(float(np.percentile(errors, 75)), 6),
        "p90_positioning_error_m": round(float(np.percentile(errors, 90)), 6),
        "max_positioning_error_m": round(float(errors.max()), 6),
    }


def kalman_filter_sequence(measurements: np.ndarray, measurement_std: float = 6.0, process_std: float = 10.0) -> np.ndarray:
    """Constant-velocity Kalman filter over 2D coordinate measurements.

    A fixed pseudo-time step is used because UJIIndoorLoc validation rows are not
    guaranteed to be uniformly sampled trajectory points.
    """
    n = len(measurements)
    if n <= 2:
        return measurements.copy()

    dt = 1.0
    x = np.array([measurements[0, 0], measurements[0, 1], 0.0, 0.0], dtype=float)
    P = np.diag([25.0, 25.0, 100.0, 100.0])
    F = np.array(
        [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=float,
    )
    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
    Q = np.diag([process_std**2, process_std**2, (process_std * 2) ** 2, (process_std * 2) ** 2])
    R = np.diag([measurement_std**2, measurement_std**2])
    I = np.eye(4)

    filtered = np.zeros((n, 2), dtype=float)
    filtered[0] = measurements[0]

    for i in range(1, n):
        # Predict
        x = F @ x
        P = F @ P @ F.T + Q

        # Update
        z = measurements[i]
        y = z - (H @ x)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (I - K @ H) @ P
        filtered[i] = x[:2]

    return filtered


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    indexes = np.zeros(n, dtype=int)
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < n:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def particle_filter_sequence(
    measurements: np.ndarray,
    rng: np.random.Generator,
    n_particles: int = 400,
    measurement_std: float = 6.0,
    process_std: float = 10.0,
    init_std: float = 3.0,
) -> np.ndarray:
    """Lightweight 2D particle filter with constant-velocity state."""
    n = len(measurements)
    if n <= 2:
        return measurements.copy()

    particles = np.zeros((n_particles, 4), dtype=float)
    particles[:, 0] = measurements[0, 0] + rng.normal(0, init_std, n_particles)
    particles[:, 1] = measurements[0, 1] + rng.normal(0, init_std, n_particles)
    particles[:, 2] = rng.normal(0, 1.0, n_particles)
    particles[:, 3] = rng.normal(0, 1.0, n_particles)
    weights = np.ones(n_particles) / n_particles

    filtered = np.zeros((n, 2), dtype=float)
    filtered[0] = np.average(particles[:, :2], weights=weights, axis=0)

    for i in range(1, n):
        # Predict with fixed pseudo-time step.
        particles[:, 0] += particles[:, 2] + rng.normal(0, process_std, n_particles)
        particles[:, 1] += particles[:, 3] + rng.normal(0, process_std, n_particles)
        particles[:, 2] += rng.normal(0, process_std * 0.2, n_particles)
        particles[:, 3] += rng.normal(0, process_std * 0.2, n_particles)

        # Weight by closeness to measured coordinate.
        diff = particles[:, :2] - measurements[i]
        dist2 = np.sum(diff**2, axis=1)
        weights = np.exp(-0.5 * dist2 / (measurement_std**2))
        weight_sum = float(weights.sum())
        if weight_sum <= 0 or not np.isfinite(weight_sum):
            weights = np.ones(n_particles) / n_particles
        else:
            weights /= weight_sum

        particle_estimate = np.average(particles[:, :2], weights=weights, axis=0)
        # Measurement-guided correction keeps the filter stable on UJIIndoorLoc
        # proxy sequences, which are not true continuous pedestrian tracks.
        estimate = 0.75 * measurements[i] + 0.25 * particle_estimate
        filtered[i] = estimate

        # Resample if effective particle count is low.
        neff = 1.0 / np.sum(weights**2)
        if neff < n_particles * 0.5:
            idx = systematic_resample(weights, rng)
            particles = particles[idx]
            weights = np.ones(n_particles) / n_particles

    return filtered


def apply_filters_by_sequence(
    raw_xy: np.ndarray,
    val_targets: pd.DataFrame,
    predicted_building_floor: np.ndarray,
    rng: np.random.Generator,
    min_sequence_length: int = 3,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    df = val_targets[["ROW_ID", "PHONEID", "TIMESTAMP"]].copy()
    df["predicted_building_floor"] = predicted_building_floor.astype(str)
    df["sequence_key"] = df["PHONEID"].astype(str) + "__" + df["predicted_building_floor"]
    df["original_index"] = np.arange(len(df))

    kalman_xy = raw_xy.copy()
    particle_xy = raw_xy.copy()

    sequence_lengths: List[int] = []
    filtered_sequence_count = 0
    filtered_row_count = 0

    for _, group in df.groupby("sequence_key", sort=False):
        group_sorted = group.sort_values(["TIMESTAMP", "ROW_ID"])
        idx = group_sorted["original_index"].to_numpy(dtype=int)
        sequence_lengths.append(len(idx))
        if len(idx) < min_sequence_length:
            continue
        measurements = raw_xy[idx]
        kalman_xy[idx] = kalman_filter_sequence(measurements)
        particle_xy[idx] = particle_filter_sequence(measurements, rng=rng)
        filtered_sequence_count += 1
        filtered_row_count += len(idx)

    sequence_summary = {
        "sequence_key_policy": "PHONEID + predicted_BUILDING_FLOOR, sorted by TIMESTAMP and ROW_ID",
        "min_sequence_length_for_filtering": min_sequence_length,
        "total_sequences": int(len(sequence_lengths)),
        "filtered_sequences": int(filtered_sequence_count),
        "filtered_rows": int(filtered_row_count),
        "unfiltered_short_rows": int(len(raw_xy) - filtered_row_count),
        "mean_sequence_length": round(float(np.mean(sequence_lengths)), 4) if sequence_lengths else 0.0,
        "median_sequence_length": round(float(np.median(sequence_lengths)), 4) if sequence_lengths else 0.0,
        "max_sequence_length": int(max(sequence_lengths)) if sequence_lengths else 0,
    }
    return kalman_xy, particle_xy, sequence_summary


def main() -> None:
    print("MSc Project - Step 5: Kalman and Particle Filtering Extension")
    root = project_root_from_script()
    print(f"Project root: {root}")
    dirs = ensure_dirs(root)

    train_x, val_x, train_y, val_y = load_inputs(root)
    train_x = pd.DataFrame(train_x.to_numpy(dtype=FEATURE_DTYPE), columns=train_x.columns)
    val_x = pd.DataFrame(val_x.to_numpy(dtype=FEATURE_DTYPE), columns=val_x.columns)
    true_xy = val_y[["LONGITUDE", "LATITUDE"]].to_numpy(dtype=float)
    train_xy = train_y[["LONGITUDE", "LATITUDE"]].to_numpy(dtype=float)
    train_bf = make_building_floor_label(train_y)

    print(f"Training feature shape: {train_x.shape}")
    print(f"Validation feature shape: {val_x.shape}")

    print("\nTraining coordinate measurement model: KNNRegressor_5_distance...")
    start = time.perf_counter()
    coord_model = make_knn_regressor()
    coord_model.fit(train_x, train_xy)
    coord_fit_seconds = round(time.perf_counter() - start, 6)
    print(f"coordinate_model_fit_seconds={coord_fit_seconds}")

    print("Training sequence segmentation model: RandomForest_150 BUILDING_FLOOR classifier...")
    start = time.perf_counter()
    bf_model = make_rf_classifier()
    bf_model.fit(train_x, train_bf)
    bf_fit_seconds = round(time.perf_counter() - start, 6)
    print(f"building_floor_model_fit_seconds={bf_fit_seconds}")

    metrics_rows: List[Dict] = []
    prediction_frames: List[pd.DataFrame] = []
    scenario_summaries: Dict[str, Dict] = {}

    print("\nEvaluating filtering scenarios...")
    for scenario in SCENARIOS:
        name = scenario.name
        print(f"\nScenario: {name}")
        x_corrupt, corruption_summary = corrupt_features(val_x, scenario)
        filter_rng = np.random.default_rng(filter_seed(name))

        start = time.perf_counter()
        raw_xy = coord_model.predict(x_corrupt)
        raw_predict_seconds = round(time.perf_counter() - start, 6)

        predicted_bf = bf_model.predict(x_corrupt)

        start = time.perf_counter()
        kalman_xy, particle_xy, seq_summary = apply_filters_by_sequence(raw_xy, val_y, predicted_bf, filter_rng)
        filtering_seconds = round(time.perf_counter() - start, 6)

        method_outputs = {
            "raw_knn_measurement": raw_xy,
            "kalman_filtered": kalman_xy,
            "particle_filtered": particle_xy,
        }
        for method_name, pred_xy in method_outputs.items():
            row = {
                "scenario": name,
                "method": method_name,
                **coordinate_metrics(true_xy, pred_xy),
                "raw_predict_seconds": raw_predict_seconds,
                "filtering_seconds": filtering_seconds if method_name != "raw_knn_measurement" else 0.0,
                **corruption_summary,
                **seq_summary,
            }
            metrics_rows.append(row)

        pred_df = val_y[["ROW_ID", "BUILDINGID", "FLOOR", "BUILDING_FLOOR", "LONGITUDE", "LATITUDE", "PHONEID", "TIMESTAMP"]].copy()
        pred_df["scenario"] = name
        pred_df["predicted_building_floor_for_sequence"] = predicted_bf
        pred_df["raw_pred_longitude"] = raw_xy[:, 0]
        pred_df["raw_pred_latitude"] = raw_xy[:, 1]
        pred_df["kalman_longitude"] = kalman_xy[:, 0]
        pred_df["kalman_latitude"] = kalman_xy[:, 1]
        pred_df["particle_longitude"] = particle_xy[:, 0]
        pred_df["particle_latitude"] = particle_xy[:, 1]
        pred_df["raw_error_m"] = point_errors(true_xy, raw_xy)
        pred_df["kalman_error_m"] = point_errors(true_xy, kalman_xy)
        pred_df["particle_error_m"] = point_errors(true_xy, particle_xy)
        prediction_frames.append(pred_df)

        scenario_summaries[name] = {
            "corruption": corruption_summary,
            "sequence_construction": seq_summary,
            "headline_mean_errors": {
                method: coordinate_metrics(true_xy, arr)["mean_positioning_error_m"]
                for method, arr in method_outputs.items()
            },
        }
        print(
            f"raw={scenario_summaries[name]['headline_mean_errors']['raw_knn_measurement']} m | "
            f"kalman={scenario_summaries[name]['headline_mean_errors']['kalman_filtered']} m | "
            f"particle={scenario_summaries[name]['headline_mean_errors']['particle_filtered']} m"
        )

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    metrics_path = dirs["tables"] / "05_filtering_metrics.csv"
    predictions_path = dirs["tables"] / "05_filtering_predictions.csv"
    summary_json_path = dirs["tables"] / "05_filtering_summary.json"
    summary_text_path = dirs["logs"] / "05_kalman_particle_filtering_summary.txt"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    # Identify best method per scenario by mean positioning error.
    best_by_scenario = []
    for scenario_name, group in metrics_df.groupby("scenario"):
        best_row = group.sort_values("mean_positioning_error_m").iloc[0]
        raw_row = group[group["method"] == "raw_knn_measurement"].iloc[0]
        best_by_scenario.append(
            {
                "scenario": scenario_name,
                "best_method": best_row["method"],
                "best_mean_error_m": float(best_row["mean_positioning_error_m"]),
                "raw_mean_error_m": float(raw_row["mean_positioning_error_m"]),
                "change_vs_raw_m": round(float(best_row["mean_positioning_error_m"] - raw_row["mean_positioning_error_m"]), 6),
            }
        )

    summary = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "project_root": str(root),
        "step": "05_kalman_particle_filtering",
        "random_seed": RANDOM_SEED,
        "locked_configuration": configuration_metadata(),
        "input_policy": "KNN coordinate predictions are treated as noisy coordinate measurements; filters operate on validation pseudo-sequences.",
        "important_limitation": "UJIIndoorLoc validation data is not a true continuous pedestrian trajectory dataset. Sequences are constructed as a controlled proxy using PHONEID + predicted BUILDING_FLOOR sorted by TIMESTAMP.",
        "train_rows": int(len(train_x)),
        "validation_rows": int(len(val_x)),
        "feature_count": int(train_x.shape[1]),
        "models": {
            "coordinate_measurement_model": "KNNRegressor_5_distance",
            "sequence_segmentation_model": "RandomForest_150 BUILDING_FLOOR classifier",
            "filter_methods": ["raw_knn_measurement", "kalman_filtered", "particle_filtered"],
        },
        "model_fit_seconds": {
            "coordinate_model": coord_fit_seconds,
            "building_floor_model": bf_fit_seconds,
        },
        "scenarios": [s.__dict__ for s in SCENARIOS],
        "scenario_summaries": scenario_summaries,
        "best_by_scenario": best_by_scenario,
        "output_files": {
            "filtering_metrics": str(metrics_path),
            "filtering_predictions": str(predictions_path),
            "summary_json": str(summary_json_path),
            "summary_text": str(summary_text_path),
        },
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "MSc Project - Step 5 Kalman and Particle Filtering Summary",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Purpose:",
        "  Apply Kalman filtering and particle filtering to coordinate predictions generated by the KNN coordinate baseline.",
        "  This step tests whether probabilistic/temporal smoothing can reduce coordinate error under clean, noisy, missing-WAP, and combined scenarios.",
        "",
        "Important limitation:",
        "  UJIIndoorLoc is not a true continuous pedestrian trajectory dataset. Filtering sequences are constructed as a controlled proxy using PHONEID + predicted BUILDING_FLOOR sorted by TIMESTAMP.",
        "  Therefore, filtering results should be interpreted as an extension experiment, not as proof of deployment-level pedestrian tracking performance.",
        "",
        "Sequence construction:",
        "  Key: PHONEID + predicted BUILDING_FLOOR",
        "  Order: TIMESTAMP then ROW_ID",
        "  Minimum sequence length for filtering: 3 samples",
        "",
        "Headline results by scenario:",
    ]
    for scenario_name, group in metrics_df.groupby("scenario", sort=False):
        lines.append(f"  {scenario_name}:")
        for _, row in group.iterrows():
            lines.append(
                f"    {row['method']}: mean_error={row['mean_positioning_error_m']} m, "
                f"median_error={row['median_positioning_error_m']} m, p90_error={row['p90_positioning_error_m']} m"
            )
    lines.extend(
        [
            "",
            "Best method per scenario:",
        ]
    )
    for row in best_by_scenario:
        lines.append(
            f"  {row['scenario']}: {row['best_method']} | best_mean_error={row['best_mean_error_m']} m | "
            f"change_vs_raw={row['change_vs_raw_m']} m"
        )
    lines.extend(
        [
            "",
            "Interpretation guidance:",
            "  If filtering improves error, it suggests smoothing helps reduce noisy coordinate jumps in the constructed sequence.",
            "  If filtering worsens error, that is still academically useful because it shows the limitation of applying trajectory filters to a non-trajectory fingerprint dataset.",
            "  Discuss results carefully and do not claim this is a real-time tracking system unless a proper sequential dataset is later added.",
            "",
            "Next step:",
            "  Review the filtering results, then prepare visualisations and a consolidated results table for Chapter 4/5.",
        ]
    )
    summary_text_path.write_text("\n".join(lines), encoding="utf-8")

    print("\nStep 5 complete. Saved outputs:")
    for label, path in summary["output_files"].items():
        print(f"- {label}: {path}")
    print("\nSend the Step 5 summary files for review before creating charts/final results tables.")


if __name__ == "__main__":
    main()
