# FP-Sentinel

**Fabry–Pérot Interferometer Simulation + AI-Based Vibration Pattern Classification**

FP-Sentinel is a browser-based, simulation-only demonstrator exploring how a simplified Fabry–Pérot interferometer (FPI) optical-intensity response could be paired with signal processing and machine learning for vibration-pattern monitoring. It is built with Python and Streamlit.

> **Important:** This is a synthetic proof of concept. No physical FPI sensor or industrial machine is connected. Signals and labels are simulated, and results are not evidence of real-world fault-detection performance.

## Features

- Interactive simulation controls for vibration pattern, noise, baseline drift, amplitude scale, and random seed.
- Simplified FPI-style optical intensity response with explicitly uncalibrated phase sensitivity.
- Time-domain plots for simulated displacement and optical intensity.
- FFT magnitude spectrum and signal-feature table.
- Random Forest classification of three **simulated patterns**: Normal, Imbalance-like, and High vibration.
- Comparison against a transparent single-feature RMS-threshold baseline.
- Synthetic test-set accuracy, macro F1, confusion matrix, and classification report.
- CSV downloads for the generated dataset and model metrics.

## Method overview

1. Generate a synthetic displacement waveform for a selected vibration pattern.
2. Map the waveform to a normalized optical intensity signal using an illustrative interference equation.
3. Add configurable noise and baseline drift.
4. Extract intensity-derived features, including RMS, standard deviation, peak deviation, crest factor, dominant frequency, spectral centroid, and high-band energy ratio.
5. Train/test a Random Forest using only the sensor-derived intensity features. The simulated displacement RMS is shown as ground truth but **excluded from model training and inference** to avoid information leakage.
6. Compare the classifier with a simple RMS-based baseline and export results.

## Run locally

### 1. Requirements

- Python 3.10 or newer recommended
- Git (optional, for cloning)

### 2. Create an environment

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, `README.md`, `LICENSE`, and `.gitignore` to its root.
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
3. Create a new app, select the repository and branch, and set the main file path to `app.py`.
4. Deploy and wait for the build to complete.
5. Open the public app URL in a private/incognito browser window to check that it loads for visitors.

Hosting plans, quotas, and availability may change. Do not include secrets or private data in the repository.

## Suggested demo flow

1. Select **Normal**, then inspect the simulated displacement and optical-intensity traces.
2. Select **Imbalance-like** and **High vibration** to compare signal patterns.
3. Increase noise and baseline drift to see how the signal and model outputs change.
4. Review the FFT, feature table, confusion matrix, and synthetic test-set metrics.
5. Download the synthetic dataset and metrics CSV files.

## Research limitations

- The optical response is a simplified illustrative model, not a calibrated FPI instrument model.
- Displacement and intensity use normalized arbitrary units; parameters are not calibrated physical settings.
- The three labels are generated patterns, not verified diagnoses of specific mechanical faults.
- Training and test samples come from the same simulator family. Reported scores therefore describe this synthetic experiment only.
- A physical validation study would require an FPI sensor and interrogator, a reference accelerometer, calibrated acquisition, and testing on independent measured data.
- The classifier's probability is a model score, not a guarantee of correctness.

## Research question

How do simulated optical noise and baseline drift affect a feature-based Random Forest's ability to distinguish synthetic vibration patterns compared with a simple RMS-threshold baseline?

## Tech stack

Python · Streamlit · NumPy · pandas · Matplotlib · scikit-learn

## Citation / attribution

If you reuse or build on this project, please credit **FP-Sentinel** and link to this repository. Add your preferred author name, institution, and project date before public release if you want them included.

## License

Released under the MIT License; see `LICENSE`.
