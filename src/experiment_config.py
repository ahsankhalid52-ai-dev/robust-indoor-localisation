"""Shared locked experiment configuration for MSc indoor-localisation experiments.

This module is the single source of truth for model hyperparameters, corruption
scenarios, numeric dtype, random seeds, and environment metadata used by Steps
3-5.  Keep these settings unchanged once the final experiment run is accepted.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import platform
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

CONFIG_VERSION = "2026-08-31-lock-v1"
RANDOM_SEED = 42
FEATURE_DTYPE = np.float64
RSSI_MIN_DBM = -110.0
RSSI_MAX_DBM = 0.0
RSSI_RANGE_DBM = RSSI_MAX_DBM - RSSI_MIN_DBM
NO_SIGNAL_SCALED = 0.0
RF_N_ESTIMATORS = 150
RF_CLASS_WEIGHT = "balanced_subsample"
KNN_N_NEIGHBORS = 5
KNN_WEIGHTS = "distance"
KNN_METRIC = "euclidean"
FILTER_RANDOM_SEED_OFFSET = 10_000

@dataclass(frozen=True)
class Scenario:
    name: str
    noise_sigma_dbm: float = 0.0
    missing_detected_fraction: float = 0.0
    description: str = ""

SCENARIOS: List[Scenario] = [
    Scenario("clean_validation", description="Original preprocessed validation data with no extra artificial corruption."),
    Scenario("noise_low_2dbm", 2.0, 0.0, "Gaussian RSSI noise with sigma=2 dBm applied to detected WAP signals."),
    Scenario("noise_medium_5dbm", 5.0, 0.0, "Gaussian RSSI noise with sigma=5 dBm applied to detected WAP signals."),
    Scenario("noise_high_10dbm", 10.0, 0.0, "Gaussian RSSI noise with sigma=10 dBm applied to detected WAP signals."),
    Scenario("missing_10pct_detected", 0.0, 0.10, "Randomly mask 10% of detected WAP signals to no-signal value 0.0."),
    Scenario("missing_20pct_detected", 0.0, 0.20, "Randomly mask 20% of detected WAP signals to no-signal value 0.0."),
    Scenario("missing_30pct_detected", 0.0, 0.30, "Randomly mask 30% of detected WAP signals to no-signal value 0.0."),
    Scenario("combined_5dbm_noise_20pct_missing", 5.0, 0.20, "Apply sigma=5 dBm noise and then mask 20% of detected WAP signals."),
]
FILTERING_SCENARIO_NAMES = {
    "clean_validation",
    "noise_medium_5dbm",
    "noise_high_10dbm",
    "missing_20pct_detected",
    "missing_30pct_detected",
    "combined_5dbm_noise_20pct_missing",
}

def filtering_scenarios() -> List[Scenario]:
    return [s for s in SCENARIOS if s.name in FILTERING_SCENARIO_NAMES]

def scenario_seed(scenario_name: str) -> int:
    for index, scenario in enumerate(SCENARIOS):
        if scenario.name == scenario_name:
            return RANDOM_SEED + index
    raise KeyError(f"Unknown scenario: {scenario_name}")

def filter_seed(scenario_name: str) -> int:
    return FILTER_RANDOM_SEED_OFFSET + scenario_seed(scenario_name)

def make_knn_classifier() -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors=KNN_N_NEIGHBORS, weights=KNN_WEIGHTS, metric=KNN_METRIC)

def make_knn_regressor() -> KNeighborsRegressor:
    return KNeighborsRegressor(n_neighbors=KNN_N_NEIGHBORS, weights=KNN_WEIGHTS, metric=KNN_METRIC)

def make_rf_classifier() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight=RF_CLASS_WEIGHT,
    )

def make_rf_regressor() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_SEED, n_jobs=-1)

def corrupt_features(x: pd.DataFrame, scenario: Scenario) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Create one deterministic corrupted validation set for a named scenario.

    The same named scenario therefore produces the same corrupted input in Step 4
    and Step 5. Missing-WAP masking selects exactly round(fraction * detected)
    detected values without replacement, avoiding binomial-count drift.
    """
    rng = np.random.default_rng(scenario_seed(scenario.name))
    arr = x.to_numpy(dtype=FEATURE_DTYPE, copy=True)
    detected_mask = arr > NO_SIGNAL_SCALED
    detected_before = int(detected_mask.sum())

    noise_changed = 0
    if scenario.noise_sigma_dbm > 0:
        noise = rng.normal(0.0, scenario.noise_sigma_dbm / RSSI_RANGE_DBM, size=arr.shape)
        arr[detected_mask] = arr[detected_mask] + noise[detected_mask]
        arr = np.clip(arr, 0.0, 1.0)
        noise_changed = detected_before

    detected_after_noise = arr > NO_SIGNAL_SCALED
    missing_changed = 0
    if scenario.missing_detected_fraction > 0:
        detected_indices = np.argwhere(detected_after_noise)
        missing_changed = int(round(len(detected_indices) * scenario.missing_detected_fraction))
        if missing_changed > 0:
            chosen = rng.choice(len(detected_indices), size=missing_changed, replace=False)
            rows = detected_indices[chosen, 0]
            cols = detected_indices[chosen, 1]
            arr[rows, cols] = NO_SIGNAL_SCALED

    detected_after = int((arr > NO_SIGNAL_SCALED).sum())
    summary = {
        "scenario_seed": scenario_seed(scenario.name),
        "detected_values_before": detected_before,
        "detected_values_after": detected_after,
        "noise_sigma_dbm": scenario.noise_sigma_dbm,
        "noise_values_changed": noise_changed,
        "missing_detected_fraction": scenario.missing_detected_fraction,
        "missing_values_masked": missing_changed,
        "feature_min_after": float(arr.min()),
        "feature_max_after": float(arr.max()),
        "feature_mean_after": float(arr.mean()),
    }
    return pd.DataFrame(arr, columns=x.columns), summary

def configuration_metadata() -> Dict[str, object]:
    return {
        "config_version": CONFIG_VERSION,
        "random_seed": RANDOM_SEED,
        "feature_dtype": str(np.dtype(FEATURE_DTYPE)),
        "knn": {"n_neighbors": KNN_N_NEIGHBORS, "weights": KNN_WEIGHTS, "metric": KNN_METRIC},
        "random_forest": {"n_estimators": RF_N_ESTIMATORS, "random_state": RANDOM_SEED, "class_weight": RF_CLASS_WEIGHT},
        "scenarios": [asdict(s) for s in SCENARIOS],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
