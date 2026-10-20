#!/usr/bin/env python3
"""
01_dataset_inspection.py

First implementation script for the MSc project:
Robust Indoor Localisation Under Sensor Noise.

Purpose:
- Extract UJIIndoorLoc dataset if ujiindoorloc.zip is placed in data/raw/
- Load trainingData.csv and validationData.csv
- Identify WAP/RSSI feature columns and target columns
- Profile no-signal values, missing values, target distributions, and metadata columns
- Save dataset inspection outputs for the progress log and methodology chapter

Run from the project root:
    python src/01_dataset_inspection.py

Expected project structure:
    MSc_Project_Code/
      data/raw/ujiindoorloc.zip
      src/01_dataset_inspection.py
      results/tables/
      results/experiment_logs/
"""

from __future__ import annotations

import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


NO_SIGNAL_VALUE = 100


def get_project_root() -> Path:
    """Return project root based on this file's location."""
    current_file = Path(__file__).resolve()
    # file is expected at <project_root>/src/01_dataset_inspection.py
    return current_file.parents[1]


def ensure_directories(project_root: Path) -> Dict[str, Path]:
    """Create expected output directories if they do not exist."""
    dirs = {
        "raw": project_root / "data" / "raw",
        "processed": project_root / "data" / "processed",
        "tables": project_root / "results" / "tables",
        "figures": project_root / "results" / "figures",
        "logs": project_root / "results" / "experiment_logs",
        "docs": project_root / "docs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def extract_dataset_if_needed(raw_dir: Path) -> Path:
    """Extract ujiindoorloc.zip if available and return the likely extraction directory."""
    zip_path = raw_dir / "ujiindoorloc.zip"
    extract_dir = raw_dir / "ujiindoorloc"

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Dataset zip not found: {zip_path}\n"
            "Place ujiindoorloc.zip inside data/raw/ and run this script again."
        )

    # Extract if CSVs are not already available.
    existing_csvs = list(extract_dir.rglob("*.csv")) if extract_dir.exists() else []
    if existing_csvs:
        print(f"Dataset already extracted at: {extract_dir}")
        return extract_dir

    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting dataset from {zip_path} to {extract_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print("Extraction complete.")
    return extract_dir


def find_dataset_csvs(search_dir: Path) -> Tuple[Path, Path]:
    """Find trainingData.csv and validationData.csv recursively."""
    csv_files = list(search_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {search_dir}")

    training_candidates = [p for p in csv_files if p.name.lower() == "trainingdata.csv"]
    validation_candidates = [p for p in csv_files if p.name.lower() == "validationdata.csv"]

    if not training_candidates or not validation_candidates:
        print("CSV files found:")
        for p in csv_files:
            print(f"  - {p}")
        raise FileNotFoundError(
            "Could not find both trainingData.csv and validationData.csv. "
            "Check the extracted dataset folder."
        )

    return training_candidates[0], validation_candidates[0]


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file with basic error handling."""
    print(f"Loading: {path}")
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def identify_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Identify WAP feature columns, target columns and metadata columns."""
    wap_cols = [c for c in df.columns if c.upper().startswith("WAP")]
    target_cols = [c for c in ["BUILDINGID", "FLOOR", "LONGITUDE", "LATITUDE"] if c in df.columns]
    known_meta = [
        "SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID", "TIMESTAMP"
    ]
    meta_cols = [c for c in known_meta if c in df.columns]
    other_cols = [c for c in df.columns if c not in wap_cols + target_cols + meta_cols]
    return {
        "wap_cols": wap_cols,
        "target_cols": target_cols,
        "meta_cols": meta_cols,
        "other_cols": other_cols,
    }


def profile_dataframe(name: str, df: pd.DataFrame, wap_cols: List[str], target_cols: List[str]) -> Dict:
    """Create a profile dictionary for one dataframe."""
    total_wap_values = int(df[wap_cols].size) if wap_cols else 0
    no_signal_count = int((df[wap_cols] == NO_SIGNAL_VALUE).sum().sum()) if wap_cols else 0
    detected_count = total_wap_values - no_signal_count
    nan_count_total = int(df.isna().sum().sum())
    nan_count_wap = int(df[wap_cols].isna().sum().sum()) if wap_cols else 0

    target_distributions = {}
    for col in ["BUILDINGID", "FLOOR"]:
        if col in df.columns:
            target_distributions[col] = {
                str(k): int(v) for k, v in df[col].value_counts(dropna=False).sort_index().items()
            }

    coordinate_summary = {}
    for col in ["LONGITUDE", "LATITUDE"]:
        if col in df.columns:
            coordinate_summary[col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
            }

    profile = {
        "name": name,
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "wap_feature_count": len(wap_cols),
        "target_columns_found": target_cols,
        "total_wap_values": total_wap_values,
        "no_signal_value_used": NO_SIGNAL_VALUE,
        "no_signal_count": no_signal_count,
        "no_signal_percentage": round((no_signal_count / total_wap_values) * 100, 4) if total_wap_values else None,
        "detected_signal_count": detected_count,
        "detected_signal_percentage": round((detected_count / total_wap_values) * 100, 4) if total_wap_values else None,
        "nan_count_total": nan_count_total,
        "nan_count_wap_features": nan_count_wap,
        "target_distributions": target_distributions,
        "coordinate_summary": coordinate_summary,
    }
    return profile


def save_outputs(
    project_root: Path,
    dirs: Dict[str, Path],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    columns: Dict[str, List[str]],
    profiles: Dict,
) -> None:
    """Save JSON, text and CSV outputs for documentation."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    json_path = dirs["tables"] / "dataset_profile.json"
    txt_path = dirs["logs"] / "01_dataset_inspection_summary.txt"
    columns_csv_path = dirs["tables"] / "dataset_columns_summary.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)

    rows = []
    for group_name, cols in columns.items():
        for col in cols:
            rows.append({"column_group": group_name, "column_name": col})
    pd.DataFrame(rows).to_csv(columns_csv_path, index=False)

    lines = []
    lines.append("MSc Project - Dataset Inspection Summary")
    lines.append(f"Generated: {timestamp}")
    lines.append(f"Project root: {project_root}")
    lines.append("")
    lines.append("Column groups:")
    for key, cols in columns.items():
        preview = ", ".join(cols[:8]) + (" ..." if len(cols) > 8 else "")
        lines.append(f"  {key}: {len(cols)} columns")
        if preview:
            lines.append(f"    {preview}")
    lines.append("")
    for profile_name, profile in profiles.items():
        lines.append(f"Profile: {profile_name}")
        lines.append(f"  Shape: {profile['shape']['rows']} rows x {profile['shape']['columns']} columns")
        lines.append(f"  WAP feature count: {profile['wap_feature_count']}")
        lines.append(f"  No-signal count: {profile['no_signal_count']} ({profile['no_signal_percentage']}%)")
        lines.append(f"  Detected signal count: {profile['detected_signal_count']} ({profile['detected_signal_percentage']}%)")
        lines.append(f"  Total NaN count: {profile['nan_count_total']}")
        lines.append(f"  WAP NaN count: {profile['nan_count_wap_features']}")
        lines.append(f"  Target columns found: {', '.join(profile['target_columns_found'])}")
        lines.append(f"  Target distributions: {profile['target_distributions']}")
        lines.append(f"  Coordinate summary: {profile['coordinate_summary']}")
        lines.append("")
    lines.append("Next step:")
    lines.append("  Review this output before writing preprocessing code. Do not train models until the no-signal handling decision is confirmed.")

    txt_path.write_text("\n".join(lines), encoding="utf-8")

    print("\nSaved outputs:")
    print(f"  - {json_path}")
    print(f"  - {txt_path}")
    print(f"  - {columns_csv_path}")


def main() -> None:
    project_root = get_project_root()
    dirs = ensure_directories(project_root)

    print("MSc Project - Step 1: Dataset Inspection")
    print(f"Project root: {project_root}")

    extract_dir = extract_dataset_if_needed(dirs["raw"])
    training_csv, validation_csv = find_dataset_csvs(extract_dir)

    train_df = load_csv(training_csv)
    valid_df = load_csv(validation_csv)

    columns = identify_columns(train_df)
    wap_cols = columns["wap_cols"]
    target_cols = columns["target_cols"]

    if len(wap_cols) == 0:
        raise ValueError("No WAP columns found. Expected columns named WAP001, WAP002, etc.")

    train_profile = profile_dataframe("trainingData", train_df, wap_cols, target_cols)
    valid_profile = profile_dataframe("validationData", valid_df, wap_cols, target_cols)
    profiles = {
        "trainingData": train_profile,
        "validationData": valid_profile,
    }

    print("\nDataset profile:")
    for name, profile in profiles.items():
        print(f"\n{name}")
        print(f"  Rows: {profile['shape']['rows']}")
        print(f"  Columns: {profile['shape']['columns']}")
        print(f"  WAP features: {profile['wap_feature_count']}")
        print(f"  No-signal values ({NO_SIGNAL_VALUE}): {profile['no_signal_count']} ({profile['no_signal_percentage']}%)")
        print(f"  Detected signal values: {profile['detected_signal_count']} ({profile['detected_signal_percentage']}%)")
        print(f"  NaN values total: {profile['nan_count_total']}")
        print(f"  Building/Floor distributions: {profile['target_distributions']}")

    print("\nDetected column groups:")
    for key, cols in columns.items():
        print(f"  {key}: {len(cols)}")

    save_outputs(project_root, dirs, train_df, valid_df, columns, profiles)
    print("\nStep 1 completed. Send the printed output and generated files for review before preprocessing.")


if __name__ == "__main__":
    main()
