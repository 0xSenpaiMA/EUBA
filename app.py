
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import plotly.express as px


# ==================================================
# CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "euba_artifacts"

MODEL_PATH = ARTIFACTS_DIR / "euba_user_autoencoder.pth"
ENCODER_PATH = ARTIFACTS_DIR / "euba_encoder.joblib"
METADATA_PATH = ARTIFACTS_DIR / "euba_metadata.joblib"

DEVICE = torch.device("cpu")


# ==================================================
# MODEL
# Must match the architecture trained in Colab
# ==================================================

class EUBAUserAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


@st.cache_resource
def load_artifacts():
    for path in [MODEL_PATH, ENCODER_PATH, METADATA_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact: {path.name}")

    metadata = joblib.load(METADATA_PATH)
    encoder = joblib.load(ENCODER_PATH)

    model = EUBAUserAutoencoder(int(metadata["input_dim"])).to(DEVICE)
    state = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()

    return model, encoder, metadata


def analyze_records(model, encoder, metadata, df):
    feature_cols = metadata["feature_cols"]

    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} required features: "
            + ", ".join(missing[:10])
        )

    # Do not pass IDs, dates, or labels to the model.
    features = df.loc[:, feature_cols].copy()
    transformed = encoder.transform(features)

    x = torch.as_tensor(transformed, dtype=torch.float32)
    criterion = nn.BCELoss(reduction="none")
    scores = []

    with torch.no_grad():
        for start in range(0, len(x), 512):
            batch = x[start:start + 512]
            reconstruction = model(batch)
            error = criterion(reconstruction, batch).mean(dim=1)
            scores.extend(error.numpy().tolist())

    threshold = float(metadata["threshold"])

    results = pd.DataFrame({
        "Anomaly Score": scores,
        "Status": [
            "Anomaly" if score > threshold else "Normal"
            for score in scores
        ],
    })

    # Metadata for analyst context only, not model inputs.
    for col in ["user_id", "date", "RiskLabel"]:
        if col in df.columns:
            results[col] = df[col].reset_index(drop=True).values

    return results, threshold


# ==================================================
# DASHBOARD UI
# ==================================================

st.set_page_config(
    page_title="EUBA | SOC Analytics",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ EUBA | SOC Analytics")
st.caption(
    "AI-powered User Behavior Anomaly Detection · "
    "Research prototype · Local inference"
)

st.info(
    "An anomaly is a record with unusually high reconstruction error. "
    "It is not proof of malicious activity."
)

try:
    model, encoder, metadata = load_artifacts()
except Exception as exc:
    st.error(f"Could not load model artifacts: {exc}")
    st.stop()

st.sidebar.header("Analysis settings")
st.sidebar.write("Model: Autoencoder")
st.sidebar.write(f"Input dimensions: {metadata['input_dim']}")
st.sidebar.write(f"Threshold: {metadata['threshold']:.6f}")
st.sidebar.caption("Threshold loaded from saved model metadata.")

uploaded_file = st.file_uploader(
    "Upload a compatible CSV dataset",
    type=["csv"],
)

if uploaded_file is None:
    st.markdown("### Getting started")
    st.write(
        "Upload the sanitized dataset used for your initial test. "
        "The CSV must contain the feature columns used during training."
    )
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

if df.empty:
    st.warning("The uploaded CSV contains no records.")
    st.stop()

if len(df) > 100_000:
    st.error("For this prototype, upload no more than 100,000 records.")
    st.stop()

st.write(f"**Dataset loaded:** {len(df):,} records")

if st.button("Run anomaly analysis", type="primary"):
    try:
        with st.spinner("Scoring records..."):
            results, threshold = analyze_records(
                model, encoder, metadata, df
            )

        # Keep results for display in this run.
        st.session_state["euba_results"] = results
        st.session_state["euba_threshold"] = threshold

    except Exception as exc:
        st.error(f"Analysis failed: {exc}")

if "euba_results" not in st.session_state:
    st.stop()

results = st.session_state["euba_results"].copy()
threshold = st.session_state["euba_threshold"]

# ==================================================
# KPI CARDS
# ==================================================

total = len(results)
anomaly_count = int((results["Status"] == "Anomaly").sum())
normal_count = total - anomaly_count
flag_rate = anomaly_count / total * 100 if total else 0

st.divider()
st.subheader("Detection overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Records analyzed", f"{total:,}")
c2.metric("Anomalies flagged", f"{anomaly_count:,}")
c3.metric("Normal records", f"{normal_count:,}")
c4.metric("Flag rate", f"{flag_rate:.2f}%")

# ==================================================
# CHARTS
# ==================================================

st.divider()
st.subheader("Anomaly score distribution")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig_hist = px.histogram(
        results,
        x="Anomaly Score",
        color="Status",
        nbins=45,
        title="Reconstruction error distribution",
        labels={"Anomaly Score": "Reconstruction error"},
    )
    fig_hist.add_vline(
        x=threshold,
        line_dash="dash",
        annotation_text="Threshold",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with chart_col2:
    status_counts = (
        results["Status"]
        .value_counts()
        .rename_axis("Status")
        .reset_index(name="Records")
    )
    fig_status = px.bar(
        status_counts,
        x="Status",
        y="Records",
        color="Status",
        title="Normal vs anomaly records",
        text="Records",
    )
    st.plotly_chart(fig_status, use_container_width=True)

# ==================================================
# TOP 20
# ==================================================

st.divider()
st.subheader("Top 20 suspicious records")
st.caption(
    "Ranked by reconstruction error, highest first. "
    "These are investigation candidates, not confirmed incidents."
)

top_n = min(20, len(results))
top_records = (
    results.sort_values("Anomaly Score", ascending=False)
    .head(top_n)
    .reset_index(drop=True)
)
top_records.insert(0, "Rank", range(1, len(top_records) + 1))

st.dataframe(
    top_records,
    use_container_width=True,
    hide_index=True,
)

# ==================================================
# ANALYST FILTER
# ==================================================

st.divider()
st.subheader("Record explorer")

filter_status = st.selectbox(
    "Status",
    ["All", "Anomaly", "Normal"],
)

filtered = results
if filter_status != "All":
    filtered = results[results["Status"] == filter_status]

st.dataframe(
    filtered,
    use_container_width=True,
    hide_index=True,
)

# ==================================================
# EXPORT
# ==================================================

st.download_button(
    "Download all analysis results",
    data=results.to_csv(index=False).encode("utf-8"),
    file_name="euba_analysis_results.csv",
    mime="text/csv",
)

alerts = results[results["Status"] == "Anomaly"]
st.download_button(
    "Download anomaly records only",
    data=alerts.to_csv(index=False).encode("utf-8"),
    file_name="euba_anomaly_alerts.csv",
    mime="text/csv",
)

# ==================================================
# LABEL COMPARISON - EXPLORATORY ONLY
# ==================================================

if "RiskLabel" in results.columns:
    st.divider()
    st.subheader("Exploratory RiskLabel comparison")
    st.warning(
        "RiskLabel is not verified attack ground truth. "
        "This comparison is descriptive only."
    )

    comparison = pd.crosstab(
        results["RiskLabel"],
        results["Status"],
        normalize="index",
    ) * 100

    st.dataframe(
        comparison.round(2).rename_axis("RiskLabel"),
        use_container_width=True,
    )

st.caption(
    f"Threshold: {threshold:.6f} · "
    "Use the results to prioritize human review, not as an automatic "
    "incident verdict."
)
