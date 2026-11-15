# EUBA

EUBA is a Streamlit prototype for user-behavior anomaly detection using a
PyTorch autoencoder.

## Current functionality

- Loads a saved categorical encoder and autoencoder.
- Accepts a compatible CSV file.
- Calculates reconstruction-error anomaly scores.
- Flags records above the saved threshold.
- Displays summary metrics, charts, suspicious records, filters, and CSV exports.

The model is an anomaly detector, not proof of malicious activity. `RiskLabel`
is shown only as an exploratory comparison and is not used as a model input.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The required model artifacts are stored in `euba_artifacts/`. The uploaded CSV
must contain the feature columns used during training.

## Project status

The repository currently contains the trained inference application and model
artifacts. Training, preprocessing, evaluation, and use-case notebooks are not
included yet and can be added in later phases.
