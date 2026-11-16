# EUBA — User Behavior Anomaly Detection

> **Version 2 (branche `v2-kafka-streaming`)** : modèle réentraîné avec
> découpage temporel, seuil dynamique μ + kσ, attaques simulées et évaluation,
> tableau de bord à 5 onglets et détection en temps réel via **Kafka**.
> **Lancement en une commande :** voir [LANCEMENT.md](LANCEMENT.md)
> (`powershell -ExecutionPolicy Bypass -File run.ps1 -Kafka`).

EUBA is a cybersecurity research prototype for detecting unusual user
behavior with a PyTorch autoencoder and presenting the results in a Streamlit
SOC-style dashboard.

> **Important:** an anomaly is not proof of malicious activity. Results are
> investigation candidates that require human validation.

## Project objective

The project follows an end-to-end insider-threat analytics workflow:

1. Select and inspect an accessible cybersecurity dataset.
2. Clean the data and engineer behavioral features.
3. Train an autoencoder on normal behavior.
4. Implement supported detection use cases.
5. Evaluate the detector and compare it with a baseline.
6. Deliver a practical SOC dashboard.
7. Optionally add grounded alert explanations with generative AI.
8. Optionally add retrieval-augmented security knowledge.
9. Optionally integrate with Wazuh or another SIEM.
10. Test, document, and prepare the project for a portfolio or CV.

## Current repository status

The repository currently contains the trained inference application and model
artifacts. The training notebook, original dataset, preprocessing notebook,
evaluation notebook, and optional integrations are planned work and are not
included yet.

### Implemented now

- Streamlit dashboard for local CSV inference.
- Saved scikit-learn `OneHotEncoder`.
- Saved PyTorch autoencoder.
- Reconstruction-error anomaly scoring.
- Saved anomaly threshold.
- Detection overview metrics.
- Score distribution and status charts.
- Top 20 suspicious records.
- Status filtering and result tables.
- CSV export for all results and anomaly alerts.
- Exploratory comparison with `RiskLabel`.

### Not yet demonstrated in this repository

- Dataset acquisition and complete schema inspection.
- Reproducible ETL and feature-engineering pipeline.
- EDA and training-history visualizations.
- Time-aware or user-aware train/validation/test split evidence.
- Formal precision, recall, F1, confusion matrix, or false-positive analysis.
- Isolation Forest baseline comparison.
- Separate validated pipelines for every detection use case.
- Generative AI, RAG, and Wazuh integration.

## Dataset and feature scope

The saved metadata identifies the training dataset as:

`Zenodo - Insiders Abnormal Activities.csv`

The model expects 172 encoded input features covering behavioral and
security-related activity, including:

- Login attempts, failed attempts, location, device type, and login time.
- Sensitive-file access, file type, location, privilege, and access time.
- Unauthorized software, permissions, location, and installation time.
- Data transfers, file size, destination, and transfer time.
- Non-work websites, categories, frequency, and visit time.
- Physical access, location, badge type, and entry time.
- Social-engineering activity and response time.
- Previous incidents and incident severity.
- Public information sharing.
- Interactions with malicious accounts.
- Behavior changes.
- Network interactions, protocols, volume, and frequency.
- Poor information-security practices.
- Uploading and sending sensitive information.
- Thumb-drive insertion attempts.
- Secure printing.

Three constant columns were recorded in the metadata and excluded from the
model input:

- `PreviousIncidents_incident_date`
- `PoorInfoSecPractices_frequency`
- `SecurePrinting_print_time`

The original dataset is not committed to this repository. Users must provide a
compatible CSV containing the feature columns used during training.

## How the model works

The application loads three artifacts from `euba_artifacts/`:

| Artifact | Purpose |
|---|---|
| `euba_encoder.joblib` | Converts categorical feature values to numeric vectors |
| `euba_metadata.joblib` | Stores feature names, input size, threshold, and training metadata |
| `euba_user_autoencoder.pth` | Stores the trained PyTorch model weights |

The autoencoder architecture is:

```text
172 → 64 → 32 → 8 → 32 → 64 → 172
```

The encoder compresses each record into an eight-dimensional representation.
The decoder reconstructs the original 172-dimensional vector. The application
uses the mean binary cross-entropy reconstruction error as the anomaly score.

The saved threshold is `0.541541`:

- Score greater than the threshold: `Anomaly`
- Score less than or equal to the threshold: `Normal`

The threshold was saved with the metadata as:

`95th percentile of training errors, then selected using validation F1`

The exact training and validation procedure still needs to be documented in a
future reproducible training notebook.

## Detection use cases

The model contains features related to several proposed use cases:

| Use case | Current support | Notes |
|---|---|---|
| Abnormal login time | Partial | Login-time features are present, but a dedicated use-case detector is not yet implemented |
| Source IP anomaly | Not demonstrated | No clear source-IP feature is present in the saved metadata |
| File activity anomaly | Partial | Sensitive-file and transfer-related features are present |
| Failed login anomaly | Partial | Failed-login features are present |

“Partial” means the general model has relevant features. It does not mean that
the use case has been separately engineered, evaluated, or validated.

## Phase roadmap

### Phase 1 — Dataset selection and inspection

**Goal:** identify a suitable accessible cybersecurity dataset and verify what
the data genuinely supports.

Planned work:

- Record the dataset source, license, and access method.
- Inspect files, rows, columns, data types, timestamps, user IDs, event types,
  and labels.
- Confirm whether the data represents users, events, sessions, or aggregates.
- Identify missing values, duplicates, inconsistent formats, and label quality.
- Determine which of the four proposed use cases are actually supported.

Deliverables:

- Dataset justification and schema summary.
- Supported and unsupported use-case report.
- Initial reproducible Colab or Jupyter notebook.

### Phase 2 — Data cleaning and feature engineering

**Goal:** create a reproducible behavioral feature pipeline without leakage.

Planned work:

- Load and validate the raw data.
- Handle missing values and duplicates.
- Parse and normalize timestamps.
- Aggregate events into user and time-window records.
- Engineer numerical, categorical, count, frequency, and time-based features.
- Encode categorical values and normalize numerical values.
- Keep identifiers, labels, and post-event information out of model inputs.
- Save the cleaned dataset and transformation pipeline.

Deliverables:

- Cleaned dataset.
- Reusable feature pipeline.
- EDA visualizations.
- Feature definitions and leakage decisions.

### Phase 3 — Autoencoder development

**Goal:** train and save a reliable reconstruction-based anomaly detector.

Planned or partially completed work:

- Implement the PyTorch autoencoder.
- Split data using a defensible time-aware or user-aware strategy.
- Train primarily on normal behavior.
- Monitor training and validation loss.
- Calculate reconstruction errors.
- Select and justify an anomaly threshold.
- Save model, encoder, metadata, and inference artifacts.

Current saved architecture:

```text
Input 172
  ↓
Linear 64 + ReLU
  ↓
Linear 32 + ReLU
  ↓
Latent representation 8
  ↓
Linear 32 + ReLU
  ↓
Linear 64 + ReLU
  ↓
Linear 172 + Sigmoid
```

### Phase 4 — Four detection use cases

**Goal:** implement only the use cases supported by the inspected dataset.

For each supported use case:

- Define the relevant raw fields and engineered features.
- Create a dedicated scoring or filtering view.
- Generate interpretable alerts.
- Show evidence associated with each alert.
- Document expected false positives and limitations.

The current dashboard uses one general anomaly score. Dedicated login-time,
source-IP, file-activity, and failed-login detectors still need to be added
only after the data inspection confirms that they are valid.

### Phase 5 — Evaluation and baseline

**Goal:** measure performance and make threshold choices defensible.

Planned work:

- Calculate precision, recall, F1, and confusion matrices when trustworthy
  labels are available.
- Analyze false positives and false negatives.
- Plot normal and anomalous score distributions.
- Compare against an Isolation Forest baseline.
- Report limitations caused by proxy labels, class imbalance, and synthetic or
  incomplete data.

`RiskLabel` is currently treated as an exploratory proxy, not verified attack
ground truth.

### Phase 6 — SOC dashboard

**Goal:** provide a lightweight and practical analyst interface.

Current dashboard capabilities:

- Upload a compatible CSV.
- Display model input dimensions and threshold.
- Run local inference.
- Show total records, anomalies, normal records, and flag rate.
- Plot score and status distributions.
- Display the top 20 highest-scoring records.
- Filter records by status.
- Download all analysis results.
- Download anomaly records only.
- Compare `RiskLabel` descriptively when present.

Future dashboard improvements:

- Severity bands based on score.
- Suspicious-user aggregation.
- Event detail views.
- Time trends.
- Use-case-specific alert panels.
- Analyst feedback and case notes.

### Phase 7 — Generative AI for alert explanation (optional)

This phase should be attempted only after the core detector is stable.

Potential functionality:

- Generate concise explanations from detected evidence.
- Summarize why a record was flagged.
- Suggest investigation steps.
- Clearly separate observed facts from hypotheses.
- Never invent logs, users, actions, or attack claims.

### Phase 8 — RAG for security knowledge (optional)

Potential sources include MITRE ATT&CK and approved SOC playbooks.

The retrieval pipeline should:

- Use trusted, versioned security documentation.
- Return source references with explanations.
- Add context to investigations without changing the model score.
- Avoid presenting retrieved guidance as evidence that an attack occurred.

### Phase 9 — Wazuh integration (optional)

Future work may investigate:

- Consuming Wazuh events.
- Mapping Wazuh fields to the EUBA feature schema.
- Forwarding validated EUBA alerts to a SOC workflow.
- Testing the integration with a small, reproducible proof of concept.

No Wazuh integration is currently implemented.

### Phase 10 — Testing, documentation, and portfolio preparation

Planned deliverables:

- Organized source and notebook structure.
- Reproducible setup instructions.
- Architecture diagram.
- Dataset and feature documentation.
- Evaluation results and screenshots.
- Use-case limitations.
- CV-ready project description.
- Short presentation.

## Local setup

Create a virtual environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start the dashboard:

```powershell
streamlit run app.py
```

Then upload a CSV containing every required feature column. The required
columns are stored in `euba_metadata.joblib` under `feature_cols`.

## Repository structure

```text
EUBA/
├── app.py
├── requirements.txt
├── README.md
└── euba_artifacts/
    ├── euba_encoder.joblib
    ├── euba_metadata.joblib
    └── euba_user_autoencoder.pth
```

Virtual environments such as `.venv/` and `.venv312/` are intentionally
excluded from Git. The original dataset and raw sensitive data should also not
be committed.

## Responsible-use limitations

- Anomaly detection identifies unusual patterns, not confirmed incidents.
- A high reconstruction error can be caused by benign rare behavior.
- A normal score does not prove that behavior is safe.
- Proxy labels may not represent verified malicious activity.
- Model performance depends on the similarity between uploaded data and the
  training data.
- Thresholds must be recalibrated when the data distribution changes.
- Results should support, not replace, analyst investigation.
