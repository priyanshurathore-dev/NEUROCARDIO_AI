# ECG Classification — Reproduction Project

Local Python/VS Code implementation based on:

Su et al., "An ECG Signal Acquisition and Analysis System Based on Machine Learning with Model Fusion",
Sensors 23 (2023), 7643. https://doi.org/10.3390/s23177643

## Scope and reproducibility notes

This repository implements the paper's main software pipeline:
- MIT-BIH Arrhythmia Database, 360 Hz
- wavelet denoising experiments (`bior2.6`, `db8`, `sym8`)
- beat windows of 250 samples (100 before, 150 after annotated beat location)
- traditional models: logistic regression, SVM, XGBoost
- morphological and wavelet-coefficient features
- CNN and LSTM
- five-fold out-of-fold stacking
- metrics and confusion matrices

The paper has two different classification experiments:
1. An initial four-label experiment using N/L/R/V (normal, left bundle branch block, right bundle branch block, ventricular premature beat).
2. A later AAMI-style four-class experiment using N/S/V/F, excluding Q because of limited samples.

These experiments are **not interchangeable**. This starter repo implements the N/S/V/F mapping as the main path because it matches the paper's later AAMI-style CNN experiment. The N/L/R/V experiment is not silently mixed into it.

### Details not fully specified in the paper
The paper does not fully enumerate all engineered features, every data split, all LSTM hyperparameters, or every stacking implementation detail. Where needed, this repo uses clearly marked reproducibility assumptions:
- deterministic record-level train/validation/test split (to reduce leakage)
- a documented feature set of morphology/statistical measures and wavelet coefficients
- a compact 1D CNN and LSTM rather than claiming the unpublished full layer-by-layer details
- a logistic-regression stacking meta-model trained on out-of-fold probabilities

These are reasonable implementation choices, not claims that the original authors used exactly the same choices. Do not claim exact reproduction unless the omitted settings can be recovered independently.

## 1. Install

Python 3.10 or 3.11 is recommended.

Create and activate a virtual environment in VS Code's terminal:

Windows:
```powershell
py -3.11 -m venv .venv
.venv\\Scripts\\Activate.ps1
```

macOS/Linux:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Download data

The scripts use WFDB's PhysioNet access and will download requested records as needed. An internet connection is needed on first run. The MIT-BIH database is hosted at https://physionet.org/content/mitdb/1.0.0/.

## 3. Run in order

Run commands from the repository root.

### A. Inspect and extract the dataset
```bash
python -m src.prepare_data --out data/processed --records 100 101 102 103 104 105 106 107 108 109 111 112 113 114 115 116 117 118 119 121 122 123 124 200 201 202 203 205 207 208 209 210 212 213 214 215 217 219 220 221 222 223 228 230 231 232 233 234
```

This downloads selected MIT-BIH records, uses reference annotations for segmentation, denoises the selected channel, extracts beats and saves NumPy arrays plus metadata. For a quick smoke test, use `--records 100 101 102 103 104 105`.

### B. Train/evaluate traditional models
```bash
python -m src.train_traditional --data data/processed --out results/traditional
```

### C. Train CNN and LSTM
```bash
python -m src.train_deep --data data/processed --out results/deep --epochs 30
```

### D. Train stacking fusion
```bash
python -m src.train_fusion --data data/processed --traditional results/traditional --deep results/deep --out results/fusion
```

For reliable fusion, use the full training pipeline so out-of-fold predictions are generated from training records only. The provided fusion script uses out-of-fold predictions from traditional estimators; neural-network probability outputs are included when the saved fold-prediction files are available. See comments in that script. If running a quick smoke test, first validate individual models and then run fusion on the full record list.

## 4. Recommended experiment sequence

1. Smoke test with 6 records; check all saved arrays and labels.
2. Run all records with a record-level split.
3. Compare wavelet bases and preserve the chosen setting in your report.
4. Train traditional models and deep models.
5. Run stacking only after the individual model pipeline works.
6. Repeat using the alternate N/L/R/V labels if your objective requires the paper's initial experiment.
7. Report per-class metrics, macro F1, confusion matrix, and the exact record split.

## 5. Key implementation choices

- **Segmentation:** uses annotation locations, not the custom dual-slope QRS detector described by the paper. This isolates classification performance. A detector should be evaluated separately before replacing annotations.
- **Denoising:** wavelet thresholding is an implementation of the paper's wavelet-denoising approach. The exact thresholding choices are not fully reported.
- **Filtering/data leakage:** record-level splits are recommended; any learned preprocessing or feature scaling is fit on training data only.
- **Lead selection:** channel 0 is used by default. MIT-BIH records can have different lead names; report the channel used.
- **Clinical use:** educational/research code only, not a diagnostic device.

## 6. Results to save in your report

- dataset records and split
- class counts before and after filtering
- wavelet basis comparison plots/metrics
- traditional model classification reports
- CNN/LSTM validation curves
- confusion matrices and macro F1
- fusion comparison against individual models
- deviations from the paper and limitations

## Troubleshooting

- If PhysioNet download fails, check internet access and WFDB version.
- If an XGBoost install fails on your platform, use a compatible Python version and install XGBoost separately.
- Neural-network training works on CPU; GPU is optional.
- For exact reproduction, the paper's incomplete split/features/model details need to be resolved and documented.
