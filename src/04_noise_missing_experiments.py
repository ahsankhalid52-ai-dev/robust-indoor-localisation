"""
MSc Project - Step 4: Controlled RSSI Noise and Missing-WAP Experiments

Purpose:
    Evaluate how clean baseline models degrade when validation RSSI/WAP features are
    affected by controlled signal noise and missing-WAP/dropout conditions.

Important:
    - Training data stays clean.
    - Only validation features are corrupted to simulate real-world sensor noise.
    - No-signal values are represented as 0.0 after Step 2 preprocessing.
    - Gaussian RSSI noise is applied only to detected/non-zero WAP signals.
    - Missing-WAP simulation randomly masks detected/non-zero WAP signals to 0.0.

Run from project root:
    python src/04_noise_missing_experiments.py
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error


from experiment_config import (
    FEATURE_DTYPE, RANDOM_SEED, SCENARIOS, Scenario, configuration_metadata,
    corrupt_features, make_knn_classifier, make_knn_regressor,
    make_rf_classifier, make_rf_regressor,
)
def get_project_root() -> Path:
    # Script is expected at project_root/src/04_noise_missing_experiments.py
    return Path(__file__).resolve().parents[1]


def ensure_dirs(project_root: Path) -> Dict[str, Path]:
    paths = {
        "processed": project_root / "data" / "processed",
        "tables": project_root / "results" / "tables",
        "logs": project_root / "results" / "experiment_logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_processed_data(processed_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_features_path = processed_dir / "train_features_scaled.csv"
    validation_features_path = processed_dir / "validation_features_scaled.csv"
    train_targets_path = processed_dir / "train_targets.csv"
    validation_targets_path = processed_dir / "validation_targets.csv"

    required = [train_features_path, validation_features_path, train_targets_path, validation_targets_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing processed files from Step 2. Run src/02_preprocessing.py first. Missing: "
            + ", ".join(missing)
        )

    print(f"Loading train features: {train_features_path}")
    x_train = pd.read_csv(train_features_path)
    print(f"Loading validation features: {validation_features_path}")
    x_val = pd.read_csv(validation_features_path)
    print(f"Loading train targets: {train_targets_path}")
    y_train = pd.read_csv(train_targets_path)
    print(f"Loading validation targets: {validation_targets_path}")
    y_val = pd.read_csv(validation_targets_path)

    return x_train, x_val, y_train, y_val


def add_building_floor_label(targets: pd.DataFrame) -> pd.Series:
    return targets["BUILDINGID"].astype(str) + "_" + targets["FLOOR"].astype(str)


def euclidean_position_errors(y_true: pd.DataFrame, pred_coords: np.ndarray) -> np.ndarray:
    true_coords = y_true[["LONGITUDE", "LATITUDE"]].to_numpy(dtype=float)
    diff = pred_coords - true_coords
    return np.sqrt(np.sum(diff**2, axis=1))


def regression_metrics(y_true: pd.DataFrame, pred_coords: np.ndarray) -> Dict[str, float]:
    lon_true = y_true["LONGITUDE"].to_numpy(dtype=float)
    lat_true = y_true["LATITUDE"].to_numpy(dtype=float)
    lon_pred = pred_coords[:, 0]
    lat_pred = pred_coords[:, 1]
    errors = euclidean_position_errors(y_true, pred_coords)

    return {
        "longitude_mae": round(float(mean_absolute_error(lon_true, lon_pred)), 6),
        "latitude_mae": round(float(mean_absolute_error(lat_true, lat_pred)), 6),
        "longitude_rmse": round(float(np.sqrt(mean_squared_error(lon_true, lon_pred))), 6),
        "latitude_rmse": round(float(np.sqrt(mean_squared_error(lat_true, lat_pred))), 6),
        "mean_positioning_error_m": round(float(np.mean(errors)), 6),
        "median_positioning_error_m": round(float(np.median(errors)), 6),
        "p75_positioning_error_m": round(float(np.percentile(errors, 75)), 6),
        "p90_positioning_error_m": round(float(np.percentile(errors, 90)), 6),
        "max_positioning_error_m": round(float(np.max(errors)), 6),
    }


def main() -> None:
    print("MSc Project - Step 4: Controlled RSSI Noise and Missing-WAP Experiments")
    project_root = get_project_root()
    print(f"Project root: {project_root}")
    paths = ensure_dirs(project_root)

    x_train_df, x_val_df, y_train_df, y_val_df = load_processed_data(paths["processed"])
    print(f"Training feature shape: {x_train_df.shape}")
    print(f"Validation feature shape: {x_val_df.shape}")

    x_train = x_train_df.to_numpy(dtype=FEATURE_DTYPE)
    x_val_original = x_val_df

    # Prepare target labels.
    y_train_building = y_train_df["BUILDINGID"]
    y_val_building = y_val_df["BUILDINGID"]
    y_train_floor = y_train_df["FLOOR"]
    y_val_floor = y_val_df["FLOOR"]
    y_train_building_floor = add_building_floor_label(y_train_df)
    y_val_building_floor = add_building_floor_label(y_val_df)
    y_train_coords = y_train_df[["LONGITUDE", "LATITUDE"]]

    print("\nTraining clean baseline models once, then evaluating corrupted validation scenarios...")
    models_classification = {
        "KNN_5_distance": make_knn_classifier,
        "RandomForest_150": make_rf_classifier,
    }
    models_regression = {
        "KNNRegressor_5_distance": make_knn_regressor,
        "RandomForestRegressor_150": make_rf_regressor,
    }


    trained_classifiers: Dict[Tuple[str, str], object] = {}
    classification_train_targets = {
        "BUILDINGID": y_train_building,
        "FLOOR": y_train_floor,
        "BUILDING_FLOOR": y_train_building_floor,
    }

    for model_name, model_factory in models_classification.items():
        for target_name, y_train_target in classification_train_targets.items():
            print(f"Training {model_name} classifier for {target_name}...")
            start = time.perf_counter()
            model = model_factory()
            model.fit(x_train, y_train_target)
            print(f"  fit_seconds={time.perf_counter() - start:.3f}")
            trained_classifiers[(model_name, target_name)] = model

    trained_regressors: Dict[str, object] = {}
    for model_name, model_factory in models_regression.items():
        print(f"Training {model_name} for LONGITUDE/LATITUDE...")
        start = time.perf_counter()
        model = model_factory()
        model.fit(x_train, y_train_coords)
        print(f"  fit_seconds={time.perf_counter() - start:.3f}")
        trained_regressors[model_name] = model

    classification_rows: List[Dict[str, object]] = []
    coordinate_rows: List[Dict[str, object]] = []
    scenario_summaries: Dict[str, Dict[str, float]] = {}

    validation_targets_lookup = {
        "BUILDINGID": y_val_building,
        "FLOOR": y_val_floor,
        "BUILDING_FLOOR": y_val_building_floor,
    }

    for scenario_index, scenario in enumerate(SCENARIOS):
        print(f"\nEvaluating scenario: {scenario.name}")
        x_scenario_df, scenario_summary = corrupt_features(x_val_original, scenario)
        scenario_summaries[scenario.name] = scenario_summary
        x_scenario = x_scenario_df.to_numpy(dtype=FEATURE_DTYPE)

        for (model_name, target_name), model in trained_classifiers.items():
            start = time.perf_counter()
            predictions = model.predict(x_scenario)
            predict_seconds = time.perf_counter() - start
            y_true = validation_targets_lookup[target_name]
            row = {
                "scenario": scenario.name,
                "model": model_name,
                "target": target_name,
                "accuracy": round(float(accuracy_score(y_true, predictions)), 6),
                "macro_f1": round(float(f1_score(y_true, predictions, average="macro", zero_division=0)), 6),
                "weighted_f1": round(float(f1_score(y_true, predictions, average="weighted", zero_division=0)), 6),
                "predict_seconds": round(float(predict_seconds), 6),
                **scenario_summary,
            }
            classification_rows.append(row)

        for model_name, model in trained_regressors.items():
            start = time.perf_counter()
            pred_coords = model.predict(x_scenario)
            predict_seconds = time.perf_counter() - start
            row = {
                "scenario": scenario.name,
                "model": model_name,
                **regression_metrics(y_val_df, pred_coords),
                "predict_seconds": round(float(predict_seconds), 6),
                **scenario_summary,
            }
            coordinate_rows.append(row)

        # Print compact headline for the scenario.
        rf_floor = next(
            row for row in classification_rows
            if row["scenario"] == scenario.name and row["model"] == "RandomForest_150" and row["target"] == "FLOOR"
        )
        knn_coord = next(
            row for row in coordinate_rows
            if row["scenario"] == scenario.name and row["model"] == "KNNRegressor_5_distance"
        )
        print(
            f"  RF FLOOR accuracy={rf_floor['accuracy']} | "
            f"KNN coordinate mean_error={knn_coord['mean_positioning_error_m']} m"
        )

    classification_df = pd.DataFrame(classification_rows)
    coordinate_df = pd.DataFrame(coordinate_rows)

    classification_out = paths["tables"] / "04_noise_missing_classification_metrics.csv"
    coordinate_out = paths["tables"] / "04_noise_missing_coordinate_metrics.csv"
    summary_out = paths["tables"] / "04_noise_missing_summary.json"
    log_out = paths["logs"] / "04_noise_missing_experiments_summary.txt"

    classification_df.to_csv(classification_out, index=False)
    coordinate_df.to_csv(coordinate_out, index=False)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(project_root),
        "step": "04_noise_missing_experiments",
        "random_seed": RANDOM_SEED,
        "locked_configuration": configuration_metadata(),
        "train_rows": int(x_train_df.shape[0]),
        "validation_rows": int(x_val_df.shape[0]),
        "feature_count": int(x_train_df.shape[1]),
        "training_policy": "Models trained on clean Step 2 training data; validation data corrupted per scenario.",
        "scenarios": [asdict(scenario) for scenario in SCENARIOS],
        "scenario_corruption_summaries": scenario_summaries,
        "classification_metrics": classification_rows,
        "coordinate_regression_metrics": coordinate_rows,
        "output_files": {
            "classification_metrics": str(classification_out),
            "coordinate_regression_metrics": str(coordinate_out),
            "summary_json": str(summary_out),
            "summary_text": str(log_out),
        },
    }
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("MSc Project - Step 4 Noise and Missing-WAP Experiments Summary")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("Purpose:")
    lines.append("  Test robustness of clean baseline models under controlled RSSI noise and missing-WAP/dropout conditions.")
    lines.append("  Models are trained on clean Step 2 training data and evaluated on corrupted validation data.")
    lines.append("")
    lines.append("Scenarios evaluated:")
    for scenario in SCENARIOS:
        lines.append(f"  {scenario.name}: {scenario.description}")
    lines.append("")
    lines.append("Classification headline results:")
    for scenario in SCENARIOS:
        subset = classification_df[
            (classification_df["scenario"] == scenario.name)
            & (classification_df["target"].isin(["BUILDINGID", "FLOOR", "BUILDING_FLOOR"]))
        ]
        for _, row in subset.iterrows():
            lines.append(
                f"  {row['scenario']} | {row['model']} | {row['target']}: "
                f"accuracy={row['accuracy']}, macro_f1={row['macro_f1']}"
            )
    lines.append("")
    lines.append("Coordinate regression headline results:")
    for _, row in coordinate_df.iterrows():
        lines.append(
            f"  {row['scenario']} | {row['model']}: "
            f"mean_error={row['mean_positioning_error_m']} m, "
            f"median_error={row['median_positioning_error_m']} m, "
            f"p90_error={row['p90_positioning_error_m']} m"
        )
    lines.append("")
    lines.append("Interpretation guidance:")
    lines.append("  Compare each scenario against clean_validation to quantify degradation.")
    lines.append("  Strong degradation under missing-WAP scenarios means the model depends heavily on sparse detected AP signals.")
    lines.append("  These results will justify the later Kalman/particle filtering extension and the robustness discussion.")
    lines.append("")
    lines.append("Next step:")
    lines.append("  Review degradation results, then prepare Step 5: coordinate-sequence construction for Kalman and particle filtering experiments.")
    log_out.write_text("\n".join(lines), encoding="utf-8")

    print("\nStep 4 complete. Saved outputs:")
    print(f"  - classification_metrics: {classification_out}")
    print(f"  - coordinate_regression_metrics: {coordinate_out}")
    print(f"  - summary_json: {summary_out}")
    print(f"  - summary_text: {log_out}")
    print("\nSend the terminal output and generated Step 4 summary files before Kalman/particle filtering.")


if __name__ == "__main__":
    main()
