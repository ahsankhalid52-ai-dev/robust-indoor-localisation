"""
MSc Project - Step 3: Clean Baseline Models

Purpose
-------
Train and evaluate clean-data baseline localisation models using the processed
UJIIndoorLoc WiFi/RSSI features created in Step 2.

This script trains:
1. KNN classifier and Random Forest classifier for BUILDINGID, FLOOR, and BUILDING_FLOOR.
2. KNN regressor and Random Forest regressor for coordinate prediction (LONGITUDE, LATITUDE).

Inputs expected
---------------
data/processed/train_features_scaled.csv
data/processed/validation_features_scaled.csv
data/processed/train_targets.csv
data/processed/validation_targets.csv

Outputs created
---------------
results/tables/03_classification_metrics.csv
results/tables/03_coordinate_regression_metrics.csv
results/tables/03_baseline_summary.json
results/tables/03_baseline_predictions.csv
results/tables/03_confusion_matrix_<model>_<target>.csv
results/experiment_logs/03_clean_baseline_models_summary.txt

Run from project root:
python src/03_clean_baseline_models.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        mean_absolute_error,
        mean_squared_error,
    )
except ImportError as exc:  # pragma: no cover - user environment message
    print("ERROR: scikit-learn is required for Step 3.", file=sys.stderr)
    print("Install it with: pip install scikit-learn", file=sys.stderr)
    raise exc


from experiment_config import (
    FEATURE_DTYPE, RANDOM_SEED, configuration_metadata,
    make_knn_classifier, make_knn_regressor, make_rf_classifier, make_rf_regressor,
)

RANDOM_STATE = RANDOM_SEED
CLASSIFICATION_TARGETS = ["BUILDINGID", "FLOOR", "BUILDING_FLOOR"]
COORDINATE_TARGETS = ["LONGITUDE", "LATITUDE"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dirs(root: Path) -> Dict[str, Path]:
    dirs = {
        "processed": root / "data" / "processed",
        "tables": root / "results" / "tables",
        "logs": root / "results" / "experiment_logs",
        "figures": root / "results" / "figures",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Run Step 2 first: python src/02_preprocessing.py"
        )


def load_processed_data(dirs: Dict[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    files = {
        "x_train": dirs["processed"] / "train_features_scaled.csv",
        "x_valid": dirs["processed"] / "validation_features_scaled.csv",
        "y_train": dirs["processed"] / "train_targets.csv",
        "y_valid": dirs["processed"] / "validation_targets.csv",
    }
    for path in files.values():
        require_file(path)

    print(f"Loading train features: {files['x_train']}")
    x_train = pd.read_csv(files["x_train"])
    print(f"Loading validation features: {files['x_valid']}")
    x_valid = pd.read_csv(files["x_valid"])
    print(f"Loading train targets: {files['y_train']}")
    y_train = pd.read_csv(files["y_train"])
    print(f"Loading validation targets: {files['y_valid']}")
    y_valid = pd.read_csv(files["y_valid"])

    missing_targets = [col for col in CLASSIFICATION_TARGETS + COORDINATE_TARGETS if col not in y_train.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns in train_targets.csv: {missing_targets}")

    if list(x_train.columns) != list(x_valid.columns):
        raise ValueError("Training and validation feature columns do not match.")

    return x_train, x_valid, y_train, y_valid


def get_classification_models() -> Dict[str, object]:
    return {
        "KNN_5_distance": make_knn_classifier(),
        "RandomForest_150": make_rf_classifier(),
    }


def get_regression_models() -> Dict[str, object]:
    return {
        "KNNRegressor_5_distance": make_knn_regressor(),
        "RandomForestRegressor_150": make_rf_regressor(),
    }


def evaluate_classification(
    model_name: str,
    model: object,
    target_name: str,
    x_train_np: np.ndarray,
    x_valid_np: np.ndarray,
    y_train: pd.Series,
    y_valid: pd.Series,
    tables_dir: Path,
) -> Tuple[Dict[str, object], np.ndarray]:
    start = time.perf_counter()
    model.fit(x_train_np, y_train)
    fit_seconds = time.perf_counter() - start

    start = time.perf_counter()
    pred = model.predict(x_valid_np)
    predict_seconds = time.perf_counter() - start

    accuracy = accuracy_score(y_valid, pred)
    report = classification_report(y_valid, pred, output_dict=True, zero_division=0)
    macro_f1 = report.get("macro avg", {}).get("f1-score", np.nan)
    weighted_f1 = report.get("weighted avg", {}).get("f1-score", np.nan)

    labels = sorted(pd.Series(pd.concat([y_train, y_valid], axis=0)).astype(str).unique())
    # Use string labels for consistent CSV row/column names.
    cm = confusion_matrix(y_valid.astype(str), pd.Series(pred).astype(str), labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    safe_model = model_name.replace("/", "_")
    safe_target = target_name.replace("/", "_")
    cm_df.to_csv(tables_dir / f"03_confusion_matrix_{safe_model}_{safe_target}.csv")

    metrics = {
        "task": "classification",
        "model": model_name,
        "target": target_name,
        "accuracy": round(float(accuracy), 6),
        "macro_f1": round(float(macro_f1), 6),
        "weighted_f1": round(float(weighted_f1), 6),
        "fit_seconds": round(float(fit_seconds), 3),
        "predict_seconds": round(float(predict_seconds), 3),
    }
    return metrics, pred


def positioning_error_m(y_true_coords: np.ndarray, y_pred_coords: np.ndarray) -> np.ndarray:
    diff = y_pred_coords - y_true_coords
    return np.sqrt(np.sum(diff * diff, axis=1))


def evaluate_regression(
    model_name: str,
    model: object,
    x_train_np: np.ndarray,
    x_valid_np: np.ndarray,
    y_train_coords: np.ndarray,
    y_valid_coords: np.ndarray,
) -> Tuple[Dict[str, object], np.ndarray, np.ndarray]:
    start = time.perf_counter()
    model.fit(x_train_np, y_train_coords)
    fit_seconds = time.perf_counter() - start

    start = time.perf_counter()
    pred_coords = model.predict(x_valid_np)
    predict_seconds = time.perf_counter() - start

    errors = positioning_error_m(y_valid_coords, pred_coords)
    longitude_mae = mean_absolute_error(y_valid_coords[:, 0], pred_coords[:, 0])
    latitude_mae = mean_absolute_error(y_valid_coords[:, 1], pred_coords[:, 1])
    # Compute RMSE manually for compatibility with both older and newer scikit-learn versions.
    longitude_rmse = float(np.sqrt(mean_squared_error(y_valid_coords[:, 0], pred_coords[:, 0])))
    latitude_rmse = float(np.sqrt(mean_squared_error(y_valid_coords[:, 1], pred_coords[:, 1])))

    metrics = {
        "task": "coordinate_regression",
        "model": model_name,
        "longitude_mae": round(float(longitude_mae), 6),
        "latitude_mae": round(float(latitude_mae), 6),
        "longitude_rmse": round(float(longitude_rmse), 6),
        "latitude_rmse": round(float(latitude_rmse), 6),
        "mean_positioning_error_m": round(float(np.mean(errors)), 6),
        "median_positioning_error_m": round(float(np.median(errors)), 6),
        "p75_positioning_error_m": round(float(np.percentile(errors, 75)), 6),
        "p90_positioning_error_m": round(float(np.percentile(errors, 90)), 6),
        "max_positioning_error_m": round(float(np.max(errors)), 6),
        "fit_seconds": round(float(fit_seconds), 3),
        "predict_seconds": round(float(predict_seconds), 3),
    }
    return metrics, pred_coords, errors


def write_text_summary(
    path: Path,
    summary: Dict[str, object],
    classification_metrics: List[Dict[str, object]],
    regression_metrics: List[Dict[str, object]],
) -> None:
    lines: List[str] = []
    lines.append("MSc Project - Step 3 Clean Baseline Models Summary")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("Purpose:")
    lines.append("  Train and evaluate clean-data baseline models after Step 2 preprocessing.")
    lines.append("  This step does not use artificial noise, missing-WAP simulation, Kalman filtering, or particle filtering yet.")
    lines.append("")
    lines.append("Input data:")
    lines.append(f"  Training samples: {summary['train_rows']}")
    lines.append(f"  Validation samples: {summary['validation_rows']}")
    lines.append(f"  Feature count: {summary['feature_count']}")
    lines.append("  Features: WAP001-WAP520 scaled to [0, 1].")
    lines.append("")
    lines.append("Classification results:")
    for item in classification_metrics:
        lines.append(
            f"  {item['model']} | {item['target']}: "
            f"accuracy={item['accuracy']}, macro_f1={item['macro_f1']}, weighted_f1={item['weighted_f1']}"
        )
    lines.append("")
    lines.append("Coordinate regression results:")
    for item in regression_metrics:
        lines.append(
            f"  {item['model']}: mean_error={item['mean_positioning_error_m']} m, "
            f"median_error={item['median_positioning_error_m']} m, "
            f"p90_error={item['p90_positioning_error_m']} m"
        )
    lines.append("")
    lines.append("Interpretation guidance:")
    lines.append("  These are clean-data baseline results. They become the comparison point for later noise, missing-data, Kalman, and particle-filter experiments.")
    lines.append("  Do not overclaim from this step. The next valid experiment is controlled noise/missing-WAP degradation testing.")
    lines.append("")
    lines.append("Next step:")
    lines.append("  Review baseline results, then implement Step 4: controlled RSSI noise and missing-WAP simulation.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    print("MSc Project - Step 3: Clean Baseline Models")
    root = project_root()
    print(f"Project root: {root}")
    dirs = ensure_dirs(root)

    x_train, x_valid, y_train, y_valid = load_processed_data(dirs)
    print(f"Training feature shape: {x_train.shape}")
    print(f"Validation feature shape: {x_valid.shape}")

    # Use float32 to reduce memory usage while preserving enough precision for ML baselines.
    x_train_np = x_train.to_numpy(dtype=FEATURE_DTYPE)
    x_valid_np = x_valid.to_numpy(dtype=FEATURE_DTYPE)
    y_train_coords = y_train[COORDINATE_TARGETS].to_numpy(dtype=np.float64)
    y_valid_coords = y_valid[COORDINATE_TARGETS].to_numpy(dtype=np.float64)

    classification_rows: List[Dict[str, object]] = []
    regression_rows: List[Dict[str, object]] = []
    prediction_df = pd.DataFrame({
        "true_BUILDINGID": y_valid["BUILDINGID"],
        "true_FLOOR": y_valid["FLOOR"],
        "true_BUILDING_FLOOR": y_valid["BUILDING_FLOOR"],
        "true_LONGITUDE": y_valid["LONGITUDE"],
        "true_LATITUDE": y_valid["LATITUDE"],
    })

    print("\nTraining classification baselines...")
    for model_name, model in get_classification_models().items():
        for target_name in CLASSIFICATION_TARGETS:
            print(f"  Training {model_name} for {target_name}...")
            metrics, pred = evaluate_classification(
                model_name=model_name,
                model=model,
                target_name=target_name,
                x_train_np=x_train_np,
                x_valid_np=x_valid_np,
                y_train=y_train[target_name].astype(str) if target_name == "BUILDING_FLOOR" else y_train[target_name],
                y_valid=y_valid[target_name].astype(str) if target_name == "BUILDING_FLOOR" else y_valid[target_name],
                tables_dir=dirs["tables"],
            )
            classification_rows.append(metrics)
            prediction_df[f"pred_{model_name}_{target_name}"] = pred
            print(f"    accuracy={metrics['accuracy']}, macro_f1={metrics['macro_f1']}")

    print("\nTraining coordinate regression baselines...")
    for model_name, model in get_regression_models().items():
        print(f"  Training {model_name} for LONGITUDE/LATITUDE...")
        metrics, pred_coords, errors = evaluate_regression(
            model_name=model_name,
            model=model,
            x_train_np=x_train_np,
            x_valid_np=x_valid_np,
            y_train_coords=y_train_coords,
            y_valid_coords=y_valid_coords,
        )
        regression_rows.append(metrics)
        prediction_df[f"pred_{model_name}_LONGITUDE"] = pred_coords[:, 0]
        prediction_df[f"pred_{model_name}_LATITUDE"] = pred_coords[:, 1]
        prediction_df[f"error_{model_name}_m"] = errors
        print(
            f"    mean_error={metrics['mean_positioning_error_m']} m, "
            f"median_error={metrics['median_positioning_error_m']} m"
        )

    classification_metrics_path = dirs["tables"] / "03_classification_metrics.csv"
    regression_metrics_path = dirs["tables"] / "03_coordinate_regression_metrics.csv"
    predictions_path = dirs["tables"] / "03_baseline_predictions.csv"
    summary_json_path = dirs["tables"] / "03_baseline_summary.json"
    summary_txt_path = dirs["logs"] / "03_clean_baseline_models_summary.txt"

    pd.DataFrame(classification_rows).to_csv(classification_metrics_path, index=False)
    pd.DataFrame(regression_rows).to_csv(regression_metrics_path, index=False)
    prediction_df.to_csv(predictions_path, index=False)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(root),
        "step": "03_clean_baseline_models",
        "locked_configuration": configuration_metadata(),
        "train_rows": int(x_train.shape[0]),
        "validation_rows": int(x_valid.shape[0]),
        "feature_count": int(x_train.shape[1]),
        "features_used": "WAP001-WAP520 scaled features from Step 2",
        "classification_targets": CLASSIFICATION_TARGETS,
        "coordinate_targets": COORDINATE_TARGETS,
        "classification_metrics": classification_rows,
        "coordinate_regression_metrics": regression_rows,
        "output_files": {
            "classification_metrics": str(classification_metrics_path),
            "coordinate_regression_metrics": str(regression_metrics_path),
            "baseline_predictions": str(predictions_path),
            "baseline_summary_json": str(summary_json_path),
            "baseline_summary_text": str(summary_txt_path),
        },
    }
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_text_summary(summary_txt_path, summary, classification_rows, regression_rows)

    print("\nStep 3 complete. Saved outputs:")
    for label, path in summary["output_files"].items():
        print(f"  - {label}: {path}")
    print("\nSend the terminal output and generated Step 3 summary files before noise/missing-data experiments.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
