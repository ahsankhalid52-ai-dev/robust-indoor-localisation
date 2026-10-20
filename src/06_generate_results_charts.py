#!/usr/bin/env python3
"""
MSc Project - Step 6: Generate Consolidated Results Tables and Charts

Purpose:
    This script consolidates outputs from Steps 3, 4, and 5 and creates
    final results tables and dissertation-ready charts.

Inputs expected:
    results/tables/03_baseline_summary.json
    results/tables/04_noise_missing_summary.json
    results/tables/05_filtering_summary.json
    results/tables/05_filtering_metrics.csv

Outputs:
    results/tables/06_consolidated_key_results.csv
    results/tables/06_filtering_comparison_table.csv
    results/tables/06_robustness_coordinate_degradation.csv
    results/tables/06_classification_degradation.csv
    results/tables/06_consolidated_results_summary.json
    results/figures/06_rf_floor_accuracy_by_scenario.png
    results/figures/06_knn_coordinate_mean_error_by_scenario.png
    results/figures/06_filtering_mean_error_comparison.png
    results/figures/06_missing_wap_effect_on_coordinate_error.png
    results/experiment_logs/06_generate_results_charts_summary.txt
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import pandas as pd


def find_project_root() -> Path:
    """Return project root assuming this script is inside src/."""
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dirs(project_root: Path) -> Dict[str, Path]:
    paths = {
        "tables": project_root / "results" / "tables",
        "figures": project_root / "results" / "figures",
        "logs": project_root / "results" / "experiment_logs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def scenario_label(name: str) -> str:
    labels = {
        "clean_validation": "Clean",
        "noise_low_2dbm": "2 dBm noise",
        "noise_medium_5dbm": "5 dBm noise",
        "noise_high_10dbm": "10 dBm noise",
        "missing_10pct_detected": "10% missing",
        "missing_20pct_detected": "20% missing",
        "missing_30pct_detected": "30% missing",
        "combined_5dbm_noise_20pct_missing": "5 dBm + 20% missing",
    }
    return labels.get(name, name)


def build_rf_floor_accuracy_table(step4: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in step4["classification_metrics"]:
        if item["model"] == "RandomForest_150" and item["target"] == "FLOOR":
            rows.append(
                {
                    "scenario": item["scenario"],
                    "scenario_label": scenario_label(item["scenario"]),
                    "rf_floor_accuracy": item["accuracy"],
                    "rf_floor_macro_f1": item["macro_f1"],
                    "missing_detected_fraction": item["missing_detected_fraction"],
                    "noise_sigma_dbm": item["noise_sigma_dbm"],
                }
            )
    return pd.DataFrame(rows)


def build_knn_coordinate_table(step4: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    clean_error = None
    for item in step4["coordinate_regression_metrics"]:
        if item["model"] == "KNNRegressor_5_distance":
            if item["scenario"] == "clean_validation":
                clean_error = item["mean_positioning_error_m"]
            rows.append(
                {
                    "scenario": item["scenario"],
                    "scenario_label": scenario_label(item["scenario"]),
                    "knn_mean_error_m": item["mean_positioning_error_m"],
                    "knn_median_error_m": item["median_positioning_error_m"],
                    "knn_p90_error_m": item["p90_positioning_error_m"],
                    "missing_detected_fraction": item["missing_detected_fraction"],
                    "noise_sigma_dbm": item["noise_sigma_dbm"],
                }
            )
    df = pd.DataFrame(rows)
    if clean_error is not None:
        df["degradation_vs_clean_m"] = df["knn_mean_error_m"] - clean_error
        df["degradation_vs_clean_pct"] = (df["degradation_vs_clean_m"] / clean_error) * 100
    return df


def build_filtering_comparison(filtering_metrics: pd.DataFrame) -> pd.DataFrame:
    keep = filtering_metrics[
        [
            "scenario",
            "method",
            "mean_positioning_error_m",
            "median_positioning_error_m",
            "p90_positioning_error_m",
            "total_sequences",
            "filtered_sequences",
            "filtered_rows",
        ]
    ].copy()
    keep["scenario_label"] = keep["scenario"].apply(scenario_label)

    # Dissertation chart order: clean first, then RSSI noise, then missing-WAP scenarios.
    scenario_order = [
        "clean_validation",
        "noise_medium_5dbm",
        "noise_high_10dbm",
        "missing_20pct_detected",
        "missing_30pct_detected",
        "combined_5dbm_noise_20pct_missing",
    ]
    method_order = ["raw_knn_measurement", "kalman_filtered", "particle_filtered"]
    keep["scenario"] = pd.Categorical(keep["scenario"], categories=scenario_order, ordered=True)
    keep["method"] = pd.Categorical(keep["method"], categories=method_order, ordered=True)
    keep = keep.sort_values(["scenario", "method"]).reset_index(drop=True)
    keep["scenario"] = keep["scenario"].astype(str)
    keep["method"] = keep["method"].astype(str)
    return keep


def build_key_results(step3: Dict[str, Any], step4: Dict[str, Any], step5: Dict[str, Any]) -> pd.DataFrame:
    rows = []

    # Step 3 key baseline results
    best_building = max(
        [x for x in step3["classification_metrics"] if x["target"] == "BUILDINGID"],
        key=lambda x: x["accuracy"],
    )
    best_floor = max(
        [x for x in step3["classification_metrics"] if x["target"] == "FLOOR"],
        key=lambda x: x["accuracy"],
    )
    best_coord = min(
        step3["coordinate_regression_metrics"],
        key=lambda x: x["mean_positioning_error_m"],
    )
    rows.append(
        {
            "finding_area": "Clean classification baseline",
            "key_result": f"{best_building['model']} achieved the best BUILDINGID accuracy",
            "value": f"{best_building['accuracy']:.4f}",
            "interpretation": "Building prediction is highly reliable under clean validation conditions.",
        }
    )
    rows.append(
        {
            "finding_area": "Clean floor baseline",
            "key_result": f"{best_floor['model']} achieved the best FLOOR accuracy",
            "value": f"{best_floor['accuracy']:.4f}",
            "interpretation": "Floor prediction is more difficult than building prediction, but Random Forest performs strongest.",
        }
    )
    rows.append(
        {
            "finding_area": "Clean coordinate baseline",
            "key_result": f"{best_coord['model']} achieved the lowest mean positioning error",
            "value": f"{best_coord['mean_positioning_error_m']:.3f} m",
            "interpretation": "KNN is the strongest coordinate baseline and becomes the measurement model for filtering.",
        }
    )

    # Step 4 robustness findings
    coord_df = build_knn_coordinate_table(step4)
    miss30 = coord_df[coord_df["scenario"] == "missing_30pct_detected"].iloc[0]
    noise10 = coord_df[coord_df["scenario"] == "noise_high_10dbm"].iloc[0]
    rows.append(
        {
            "finding_area": "RSSI noise robustness",
            "key_result": "10 dBm RSSI noise increased KNN coordinate mean error",
            "value": f"{noise10['knn_mean_error_m']:.3f} m",
            "interpretation": "Pure RSSI noise caused moderate degradation compared with missing-WAP corruption.",
        }
    )
    rows.append(
        {
            "finding_area": "Missing-WAP robustness",
            "key_result": "30% missing detected WAPs strongly increased KNN coordinate mean error",
            "value": f"{miss30['knn_mean_error_m']:.3f} m",
            "interpretation": "The model depends heavily on sparse detected access-point signals.",
        }
    )

    # Step 5 filtering findings
    best_rows = step5["best_by_scenario"]
    missing20 = [x for x in best_rows if x["scenario"] == "missing_20pct_detected"][0]
    missing30 = [x for x in best_rows if x["scenario"] == "missing_30pct_detected"][0]
    rows.append(
        {
            "finding_area": "Particle filtering under missing data",
            "key_result": "Particle filtering slightly improved 20% missing-WAP scenario",
            "value": f"{missing20['change_vs_raw_m']:.3f} m vs raw",
            "interpretation": "The improvement is small but suggests probabilistic smoothing can help severe missing-signal cases.",
        }
    )
    rows.append(
        {
            "finding_area": "Filtering limitation",
            "key_result": "Raw KNN remained best for clean and pure-noise scenarios",
            "value": "No improvement",
            "interpretation": "Filtering is limited because the dataset is not a true continuous trajectory dataset.",
        }
    )
    return pd.DataFrame(rows)


def save_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, ylabel: str, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.bar(df[x_col], df[y_col])
    plt.title(title)
    plt.xlabel("Scenario")
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_line_chart(df: pd.DataFrame, x_col: str, y_cols: List[str], title: str, ylabel: str, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    for y in y_cols:
        plt.plot(df[x_col], df[y], marker="o", label=y)
    plt.title(title)
    plt.xlabel("Scenario")
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    project_root = find_project_root()
    paths = ensure_dirs(project_root)

    print("MSc Project - Step 6: Generate Consolidated Results Tables and Charts")
    print(f"Project root: {project_root}")

    step3_path = paths["tables"] / "03_baseline_summary.json"
    step4_path = paths["tables"] / "04_noise_missing_summary.json"
    step5_path = paths["tables"] / "05_filtering_summary.json"
    filtering_metrics_path = paths["tables"] / "05_filtering_metrics.csv"

    print("Loading previous step outputs...")
    step3 = load_json(step3_path)
    step4 = load_json(step4_path)
    step5 = load_json(step5_path)
    if not filtering_metrics_path.exists():
        raise FileNotFoundError(f"Required input file not found: {filtering_metrics_path}")
    filtering_metrics = pd.read_csv(filtering_metrics_path)

    print("Building consolidated result tables...")
    rf_floor_df = build_rf_floor_accuracy_table(step4)
    knn_coord_df = build_knn_coordinate_table(step4)
    filtering_df = build_filtering_comparison(filtering_metrics)
    key_results_df = build_key_results(step3, step4, step5)

    # Wide filtering table for charts
    filtering_wide = (
        filtering_df.pivot_table(
            index=["scenario", "scenario_label"],
            columns="method",
            values="mean_positioning_error_m",
            aggfunc="first",
        )
        .reset_index()
    )
    scenario_order = [
        "clean_validation",
        "noise_medium_5dbm",
        "noise_high_10dbm",
        "missing_20pct_detected",
        "missing_30pct_detected",
        "combined_5dbm_noise_20pct_missing",
    ]
    filtering_wide["scenario"] = pd.Categorical(filtering_wide["scenario"], categories=scenario_order, ordered=True)
    filtering_wide = filtering_wide.sort_values("scenario").reset_index(drop=True)
    filtering_wide["scenario"] = filtering_wide["scenario"].astype(str)

    out_rf = paths["tables"] / "06_classification_degradation.csv"
    out_coord = paths["tables"] / "06_robustness_coordinate_degradation.csv"
    out_filter = paths["tables"] / "06_filtering_comparison_table.csv"
    out_key = paths["tables"] / "06_consolidated_key_results.csv"
    out_filter_wide = paths["tables"] / "06_filtering_mean_error_wide.csv"

    rf_floor_df.to_csv(out_rf, index=False)
    knn_coord_df.to_csv(out_coord, index=False)
    filtering_df.to_csv(out_filter, index=False)
    key_results_df.to_csv(out_key, index=False)
    filtering_wide.to_csv(out_filter_wide, index=False)

    print("Creating dissertation-ready charts...")
    fig1 = paths["figures"] / "06_rf_floor_accuracy_by_scenario.png"
    fig2 = paths["figures"] / "06_knn_coordinate_mean_error_by_scenario.png"
    fig3 = paths["figures"] / "06_filtering_mean_error_comparison.png"
    fig4 = paths["figures"] / "06_missing_wap_effect_on_coordinate_error.png"

    save_bar_chart(
        rf_floor_df,
        "scenario_label",
        "rf_floor_accuracy",
        "Random Forest Floor Accuracy by Robustness Scenario",
        "Accuracy",
        fig1,
    )
    save_bar_chart(
        knn_coord_df,
        "scenario_label",
        "knn_mean_error_m",
        "KNN Coordinate Mean Error by Robustness Scenario",
        "Mean positioning error (m)",
        fig2,
    )
    if {"raw_knn_measurement", "kalman_filtered", "particle_filtered"}.issubset(set(filtering_wide.columns)):
        save_line_chart(
            filtering_wide,
            "scenario_label",
            ["raw_knn_measurement", "kalman_filtered", "particle_filtered"],
            "Filtering Comparison: Mean Positioning Error",
            "Mean positioning error (m)",
            fig3,
        )

    missing_only = knn_coord_df[
        knn_coord_df["scenario"].isin(
            ["clean_validation", "missing_10pct_detected", "missing_20pct_detected", "missing_30pct_detected"]
        )
    ].copy()
    save_bar_chart(
        missing_only,
        "scenario_label",
        "knn_mean_error_m",
        "Effect of Missing Detected WAPs on KNN Coordinate Error",
        "Mean positioning error (m)",
        fig4,
    )

    summary = {
        "generated_at_source_steps": {
            "step3": step3.get("generated_at"),
            "step4": step4.get("generated_at"),
            "step5": step5.get("generated_at"),
        },
        "outputs": {
            "classification_degradation": str(out_rf),
            "coordinate_degradation": str(out_coord),
            "filtering_comparison": str(out_filter),
            "filtering_wide": str(out_filter_wide),
            "key_results": str(out_key),
            "figures": [str(fig1), str(fig2), str(fig3), str(fig4)],
        },
        "headline_findings": key_results_df.to_dict(orient="records"),
    }
    out_json = paths["tables"] / "06_consolidated_results_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log_path = paths["logs"] / "06_generate_results_charts_summary.txt"
    with log_path.open("w", encoding="utf-8") as f:
        f.write("MSc Project - Step 6 Consolidated Results and Charts Summary\n")
        f.write("Purpose:\n")
        f.write("  Consolidate Step 3, Step 4, and Step 5 outputs into tables and PNG charts for the dissertation.\n\n")
        f.write("Key findings:\n")
        for _, row in key_results_df.iterrows():
            f.write(f"  - {row['finding_area']}: {row['key_result']} ({row['value']}). {row['interpretation']}\n")
        f.write("\nSaved outputs:\n")
        for key, value in summary["outputs"].items():
            f.write(f"  - {key}: {value}\n")

    print("\nStep 6 complete. Saved outputs:")
    print(f"- {out_rf}")
    print(f"- {out_coord}")
    print(f"- {out_filter}")
    print(f"- {out_key}")
    print(f"- {out_json}")
    print(f"- {fig1}")
    print(f"- {fig2}")
    print(f"- {fig3}")
    print(f"- {fig4}")
    print(f"- {log_path}")
    print("\nSend the Step 6 summary text and generated charts for review before thesis chapter writing.")


if __name__ == "__main__":
    main()
