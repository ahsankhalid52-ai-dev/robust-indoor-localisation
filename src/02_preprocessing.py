"""
MSc Project - Step 2: Preprocessing Design
Project: Robust Indoor Localisation Under Sensor Noise

Purpose:
    This script prepares the UJIIndoorLoc WiFi/RSSI dataset for baseline modelling.
    It does NOT train models. It only converts the raw RSSI fingerprint data into
    clean, model-ready feature and target files.

Main preprocessing decision:
    UJIIndoorLoc uses value 100 to mean "no access point detected". This is not
    a real RSSI signal value. The script replaces 100 with -110 dBm, a weak/no-signal
    placeholder, then scales all WAP/RSSI features into the range [0, 1].

Outputs:
    data/processed/train_features_scaled.csv
    data/processed/validation_features_scaled.csv
    data/processed/train_targets.csv
    data/processed/validation_targets.csv
    results/tables/preprocessing_summary.json
    results/experiment_logs/02_preprocessing_summary.txt
"""

from __future__ import annotations

import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


NO_SIGNAL_RAW_VALUE = 100
NO_SIGNAL_REPLACEMENT_DBM = -110.0
RSSI_MIN_DBM = -110.0
RSSI_MAX_DBM = 0.0

TARGET_COLS = ["BUILDINGID", "FLOOR", "LONGITUDE", "LATITUDE"]
META_COLS = ["SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID", "TIMESTAMP"]


def project_root() -> Path:
    """Return project root based on this script location."""
    return Path(__file__).resolve().parents[1]


def ensure_dirs(root: Path) -> Dict[str, Path]:
    dirs = {
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "tables": root / "results" / "tables",
        "logs": root / "results" / "experiment_logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def extract_dataset_if_needed(raw_dir: Path) -> Path:
    """Find/extract ujiindoorloc.zip and return the folder containing trainingData.csv."""
    candidates = list(raw_dir.rglob("trainingData.csv"))
    if candidates:
        return candidates[0].parent

    zip_candidates = list(raw_dir.glob("*.zip"))
    if not zip_candidates:
        raise FileNotFoundError(
            f"No dataset zip file found in {raw_dir}. Place ujiindoorloc.zip inside data/raw/."
        )

    zip_path = zip_candidates[0]
    extract_to = raw_dir / zip_path.stem
    print(f"Extracting dataset from {zip_path} to {extract_to} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print("Extraction complete.")

    candidates = list(extract_to.rglob("trainingData.csv"))
    if not candidates:
        raise FileNotFoundError("trainingData.csv was not found after extraction.")
    return candidates[0].parent


def load_data(dataset_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = dataset_dir / "trainingData.csv"
    valid_path = dataset_dir / "validationData.csv"
    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError(
            f"Expected trainingData.csv and validationData.csv inside {dataset_dir}."
        )

    print(f"Loading training data: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"Loading validation data: {valid_path}")
    valid_df = pd.read_csv(valid_path)
    return train_df, valid_df


def detect_wap_columns(df: pd.DataFrame) -> List[str]:
    wap_cols = [c for c in df.columns if c.upper().startswith("WAP")]
    if len(wap_cols) == 0:
        raise ValueError("No WAP columns found. Expected columns like WAP001-WAP520.")
    return wap_cols


def scale_rssi_features(df: pd.DataFrame, wap_cols: List[str]) -> pd.DataFrame:
    """
    Replace UJIIndoorLoc no-signal value 100 with -110 dBm and scale to [0, 1].
    Formula: scaled = (RSSI_dBm - RSSI_MIN_DBM) / (RSSI_MAX_DBM - RSSI_MIN_DBM)
    With RSSI_MIN_DBM=-110 and RSSI_MAX_DBM=0, no signal becomes 0.
    """
    x = df[wap_cols].copy()
    x = x.replace(NO_SIGNAL_RAW_VALUE, NO_SIGNAL_REPLACEMENT_DBM)
    x = (x - RSSI_MIN_DBM) / (RSSI_MAX_DBM - RSSI_MIN_DBM)
    x = x.clip(lower=0.0, upper=1.0)
    return x


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in TARGET_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing target columns: {missing}")

    available_meta = [c for c in META_COLS if c in df.columns]
    targets = df[TARGET_COLS + available_meta].copy()
    targets["BUILDING_FLOOR"] = (
        targets["BUILDINGID"].astype(str) + "_" + targets["FLOOR"].astype(str)
    )
    targets.insert(0, "ROW_ID", range(len(targets)))
    return targets


def signal_count_stats(df: pd.DataFrame, wap_cols: List[str]) -> Dict[str, float]:
    detected_counts = (df[wap_cols] != NO_SIGNAL_RAW_VALUE).sum(axis=1)
    return {
        "min_detected_waps_per_row": int(detected_counts.min()),
        "max_detected_waps_per_row": int(detected_counts.max()),
        "mean_detected_waps_per_row": round(float(detected_counts.mean()), 4),
        "median_detected_waps_per_row": round(float(detected_counts.median()), 4),
    }


def profile_split(name: str, raw_df: pd.DataFrame, x_scaled: pd.DataFrame, targets: pd.DataFrame, wap_cols: List[str]) -> Dict:
    total_wap_values = int(raw_df[wap_cols].size)
    no_signal_count = int((raw_df[wap_cols] == NO_SIGNAL_RAW_VALUE).sum().sum())
    detected_count = total_wap_values - no_signal_count

    return {
        "split": name,
        "rows": int(raw_df.shape[0]),
        "columns_raw": int(raw_df.shape[1]),
        "wap_feature_count": int(len(wap_cols)),
        "no_signal_raw_value": NO_SIGNAL_RAW_VALUE,
        "no_signal_replacement_dbm": NO_SIGNAL_REPLACEMENT_DBM,
        "no_signal_count": no_signal_count,
        "no_signal_percentage": round(no_signal_count / total_wap_values * 100, 4),
        "detected_signal_count": detected_count,
        "detected_signal_percentage": round(detected_count / total_wap_values * 100, 4),
        "scaled_feature_min": round(float(x_scaled.min().min()), 6),
        "scaled_feature_max": round(float(x_scaled.max().max()), 6),
        "scaled_feature_mean": round(float(x_scaled.values.mean()), 6),
        "nan_count_after_scaling": int(x_scaled.isna().sum().sum()),
        "building_distribution": {str(k): int(v) for k, v in targets["BUILDINGID"].value_counts().sort_index().items()},
        "floor_distribution": {str(k): int(v) for k, v in targets["FLOOR"].value_counts().sort_index().items()},
        "building_floor_distribution": {str(k): int(v) for k, v in targets["BUILDING_FLOOR"].value_counts().sort_index().items()},
        "signal_count_stats": signal_count_stats(raw_df, wap_cols),
    }


def write_text_summary(path: Path, summary: Dict, output_files: Dict[str, str]) -> None:
    lines = []
    lines.append("MSc Project - Preprocessing Summary")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("Preprocessing decision:")
    lines.append(f"  Raw no-signal value {NO_SIGNAL_RAW_VALUE} was replaced with {NO_SIGNAL_REPLACEMENT_DBM} dBm.")
    lines.append("  WAP/RSSI values were scaled to [0, 1] using fixed RSSI bounds -110 dBm to 0 dBm.")
    lines.append("  Only WAP001-WAP520 features are used as model input features.")
    lines.append("  Metadata columns are retained in target files for analysis, but not used as model inputs.")
    lines.append("")

    for split_name in ["trainingData", "validationData"]:
        split = summary[split_name]
        lines.append(f"Profile: {split_name}")
        lines.append(f"  Rows: {split['rows']}")
        lines.append(f"  WAP features: {split['wap_feature_count']}")
        lines.append(f"  No-signal values replaced: {split['no_signal_count']} ({split['no_signal_percentage']}%)")
        lines.append(f"  Detected signal values: {split['detected_signal_count']} ({split['detected_signal_percentage']}%)")
        lines.append(f"  Scaled feature range: {split['scaled_feature_min']} to {split['scaled_feature_max']}")
        lines.append(f"  NaN count after scaling: {split['nan_count_after_scaling']}")
        lines.append(f"  Building distribution: {split['building_distribution']}")
        lines.append(f"  Floor distribution: {split['floor_distribution']}")
        lines.append(f"  Detected WAPs per row: {split['signal_count_stats']}")
        lines.append("")

    lines.append("Saved outputs:")
    for label, file_path in output_files.items():
        lines.append(f"  - {label}: {file_path}")
    lines.append("")
    lines.append("Next step:")
    lines.append("  Review preprocessing outputs, then run clean-data baseline models: KNN and Random Forest.")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    print("MSc Project - Step 2: Preprocessing")
    root = project_root()
    print(f"Project root: {root}")
    dirs = ensure_dirs(root)

    dataset_dir = extract_dataset_if_needed(dirs["raw"])
    train_df, valid_df = load_data(dataset_dir)

    wap_cols = detect_wap_columns(train_df)
    if set(wap_cols) != set(detect_wap_columns(valid_df)):
        raise ValueError("Training and validation WAP columns do not match.")

    print(f"Detected WAP features: {len(wap_cols)}")
    print("Applying preprocessing: 100 -> -110 dBm, then fixed [0, 1] scaling ...")

    x_train_scaled = scale_rssi_features(train_df, wap_cols)
    x_valid_scaled = scale_rssi_features(valid_df, wap_cols)
    y_train = build_targets(train_df)
    y_valid = build_targets(valid_df)

    output_files = {
        "train_features_scaled": str(dirs["processed"] / "train_features_scaled.csv"),
        "validation_features_scaled": str(dirs["processed"] / "validation_features_scaled.csv"),
        "train_targets": str(dirs["processed"] / "train_targets.csv"),
        "validation_targets": str(dirs["processed"] / "validation_targets.csv"),
        "preprocessing_summary_json": str(dirs["tables"] / "preprocessing_summary.json"),
        "preprocessing_summary_text": str(dirs["logs"] / "02_preprocessing_summary.txt"),
    }

    x_train_scaled.to_csv(output_files["train_features_scaled"], index=False)
    x_valid_scaled.to_csv(output_files["validation_features_scaled"], index=False)
    y_train.to_csv(output_files["train_targets"], index=False)
    y_valid.to_csv(output_files["validation_targets"], index=False)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(root),
        "preprocessing_decision": {
            "raw_no_signal_value": NO_SIGNAL_RAW_VALUE,
            "replacement_dbm": NO_SIGNAL_REPLACEMENT_DBM,
            "scaling": "fixed_minmax_from_-110_dBm_to_0_dBm",
            "model_input_features": "WAP001-WAP520 only",
            "metadata_usage": "retained for analysis only; not used as model input",
        },
        "trainingData": profile_split("trainingData", train_df, x_train_scaled, y_train, wap_cols),
        "validationData": profile_split("validationData", valid_df, x_valid_scaled, y_valid, wap_cols),
        "output_files": output_files,
    }

    Path(output_files["preprocessing_summary_json"]).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_text_summary(Path(output_files["preprocessing_summary_text"]), summary, output_files)

    print("\nPreprocessing complete. Summary:\n")
    for split_name in ["trainingData", "validationData"]:
        split = summary[split_name]
        print(split_name)
        print(f"  Rows: {split['rows']}")
        print(f"  WAP features: {split['wap_feature_count']}")
        print(f"  No-signal values replaced: {split['no_signal_count']} ({split['no_signal_percentage']}%)")
        print(f"  Scaled feature range: {split['scaled_feature_min']} to {split['scaled_feature_max']}")
        print(f"  NaN count after scaling: {split['nan_count_after_scaling']}")
        print(f"  Detected WAPs per row: {split['signal_count_stats']}")
        print()

    print("Saved outputs:")
    for label, file_path in output_files.items():
        print(f"  - {file_path}")
    print("\nStep 2 completed. Send the printed output and generated summary files before baseline modelling.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
