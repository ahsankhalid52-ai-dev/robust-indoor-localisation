# Robust Indoor Localisation Under Sensor Noise

**MSc Artificial Intelligence Dissertation | De Montfort University, UK**

## Project Overview

This project investigates the robustness of machine learning-based indoor localisation using Wi-Fi Received Signal Strength Indicator (RSSI) fingerprinting.

The research evaluates how sensor noise and missing Wi-Fi measurements affect indoor location predictions and explores filtering techniques to improve localisation robustness.

## Dataset

The project uses the **UJIIndoorLoc** dataset, a public benchmark for Wi-Fi fingerprint-based indoor localisation.

The dataset includes:

- Wi-Fi RSSI measurements from 520 wireless access point features.
- Building and floor labels.
- Longitude and latitude coordinates for location estimation.

## Technologies and Methods

- **Programming:** Python
- **Data Processing:** pandas, NumPy, scikit-learn
- **Machine Learning:** K-Nearest Neighbours (KNN), Random Forest
- **Filtering:** Kalman Filter and Particle Filter
- **Evaluation:** Accuracy, F1-score, MAE and RMSE

## Project Methodology

The project follows six main stages:

1. **Dataset Inspection:** Analyse the dataset structure, Wi-Fi features and missing signal values.
2. **Data Preprocessing:** Handle missing Wi-Fi signals, normalise features and prepare data for machine learning.
3. **Baseline Modelling:** Train and evaluate KNN and Random Forest models using clean Wi-Fi data.
4. **Noise and Missing Data Experiments:** Test model performance under Gaussian noise, missing Wi-Fi readings and combined disturbances.
5. **Filtering Experiments:** Evaluate Kalman and Particle filtering techniques for localisation robustness.
6. **Results and Visualisation:** Consolidate experimental results and generate comparison tables and charts.

## Repository Structure

```text
robust-indoor-localisation/
├── src/                # Python scripts
├── results/            # Experimental results and charts
├── docs/               # Project documentation
├── requirements.txt    # Python dependencies
└── README.md           # Project overview
```

## Research Objectives

- Develop a reproducible indoor localisation pipeline.
- Evaluate building and floor classification and coordinate estimation.
- Investigate the effects of sensor noise and missing Wi-Fi signals.
- Compare baseline machine learning and filtering approaches.
- Analyse performance using appropriate evaluation metrics.

## Installation

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

The UJIIndoorLoc dataset must be obtained separately and placed in the appropriate local data directory.

## Author

**Ahsan Khalid**

MSc Artificial Intelligence
De Montfort University, United Kingdom

GitHub: [https://github.com/ahsankhalid52-ai-dev](https://github.com/ahsankhalid52-ai-dev)
# robust-indoor-localisation
MSc AI dissertation on robust indoor localisation using Wi-Fi RSSI fingerprinting, KNN, Random Forest, and noise-resilience evaluation.
