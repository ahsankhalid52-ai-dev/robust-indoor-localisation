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

## Author

**Ahsan Khalid**

MSc Artificial Intelligence
De Montfort University, United Kingdom

GitHub: [https://github.com/ahsankhalid52-ai-dev](https://github.com/ahsankhalid52-ai-dev)
# robust-indoor-localisation
MSc AI dissertation on robust indoor localisation using Wi-Fi RSSI fingerprinting, KNN, Random Forest, and noise-resilience evaluation.
## How to Run the Project

1. Download the UJIIndoorLoc dataset from its original source.
2. Place the dataset ZIP file in `data/raw/` and name it `ujiindoorloc.zip`.
3. Open a terminal in the main project directory.
4. Install the dependencies:

```bash
pip install -r requirements.txt
```

5. Run the six experiments in order:

```bash
python src/01_dataset_inspection.py
python src/02_preprocessing.py
python src/03_clean_baseline_models.py
python src/04_noise_missing_experiments.py
python src/05_kalman_particle_filtering.py
python src/06_generate_results_charts.py
```

The scripts generate processed data, experiment summaries, evaluation tables and charts in the project's `data/processed/` and `results/` directories.

**Note:** The dataset is not included in this repository and must be downloaded separately. The filtering experiments use constructed validation sequences rather than recorded continuous pedestrian trajectories.
