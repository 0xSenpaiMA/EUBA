"""
EUBA v2 - Tableau de bord SOC (Streamlit).

Lancement :  streamlit run v2/app.py
"""
import json
import sqlite3
import time

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from euba_model import (ARTIFACTS_DIR, TEST_DATA, V2_DIR, EUBAAutoencoder,
                        dynamic_threshold, feature_errors, risk_score,
                        reconstruction_errors)

COLORS = {"Normal": "#2a9d8f", "Usurpation": "#e63946",
          "Exfiltration": "#f4a261"}

st.set_page_config(page_title="EUBA v2 | SOC", page_icon="🛡️",
                   layout="wide")


# ==================================================
# CHARGEMENT
# ==================================================
@st.cache_resource
def load_model():
    meta = joblib.load(ARTIFACTS_DIR / "metadata.joblib")
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    model = EUBAAutoencoder(meta["input_dim"])
    model.load_state_dict(torch.load(ARTIFACTS_DIR / "autoencoder.pth",
                                     weights_only=True))
    model.eval()
    return model, scaler, meta


@st.cache_data
def score_dataset(df):
    """Calcule l'erreur de reconstruction de chaque ligne (une seule fois)."""
    model, scaler, meta = load_model()
    x = scaler.transform(df[meta["feature_cols"]])
    out = df[["user_id", "date"]].copy()
    out["scenario"] = df.get("scenario", "Inconnu")
    out["is_attack"] = df.get("is_attack", 0)
    out["error"] = reconstruction_errors(model, x)
    return out, feature_errors(model, x)


try:
    model, scaler, meta = load_model()
except FileNotFoundError:
    st.error("Modèle introuvable. Lancez d'abord : python v2/train.py")
    st.stop()

# ==================================================
# BARRE LATÉRALE : SEUIL DYNAMIQUE
# ==================================================
st.sidebar.header("⚙️ Seuil de détection")
k = st.sidebar.slider("k (nombre d'écarts-types)", 1.0, 6.0,
                      float(meta["default_k"]), 0.5)
val_errors = meta["val_errors"]
threshold = dynamic_threshold(val_errors, k)
st.sidebar.latex(r"seuil = \mu + k\,\sigma")
st.sidebar.write(f"μ = {val_errors.mean():.5f}")
st.sidebar.write(f"σ = {val_errors.std():.5f}")
st.sidebar.metric("Seuil actuel", f"{threshold:.5f}")
st.sidebar.caption("μ et σ sont calculés sur les erreurs du trafic "
                   "légitime de validation. Score de risque = 50 au seuil.")

uploaded = st.sidebar.file_uploader("Autre jeu de test (CSV)", type="csv")

if uploaded is not None:
    raw = pd.read_csv(uploaded)
elif TEST_DATA.exists():
    raw = pd.read_csv(TEST_DATA)
else:
    st.error("Jeu de test introuvable. Lancez : python v2/simulate_attacks.py")
    st.stop()

missing = [c for c in meta["feature_cols"] if c not in raw.columns]
if missing:
    st.error(f"Colonnes manquantes dans le CSV : {', '.join(missing[:8])}")
    st.stop()

results, feat_err = score_dataset(raw)
results["risk"] = risk_score(results["error"], threshold)
results["alert"] = results["error"] > threshold
has_truth = results["is_attack"].sum() > 0

# ==================================================
# EN-TÊTE
# ==================================================
st.title("🛡️ EUBA v2 · Détection d'anomalies comportementales")
st.caption("Autoencodeur entraîné uniquement sur le trafic légitime · "
           "erreur de reconstruction ‖x − x̂‖² · seuil dynamique μ + kσ")

tab_live, tab_loss, tab_eval, tab_user, tab_kafka = st.tabs(
    ["🛰️ SOC en direct", "📉 Analyse de la perte",
     "✅ Évaluation", "🔎 Détail utilisateur", "📡 Flux Kafka"])


# ==================================================
# 1. SOC EN DIRECT
# ==================================================
def render_day(day, container):
    today = results[results["date"] == day].sort_values("risk",
                                                         ascending=False)
    alerts = today[today["alert"]]
    with container.container():
        st.subheader(f"📅 Journée du {day}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Utilisateurs surveillés", len(today))
        c2.metric("🔴 Comptes en alerte", len(alerts))
        c3.metric("Score de risque max", int(today["risk"].max()))

        if len(alerts):
            st.error("🚨 Comptes suspects : " + ", ".join(
                f"**{u}** ({int(r)})"
                for u, r in zip(alerts["user_id"], alerts["risk"])))
        else:
            st.success("✅ Aucun comportement anormal ce jour.")

        table = pd.DataFrame({
            "Utilisateur": today["user_id"].astype(str),
            "Score de risque": today["risk"].astype(int),
            "Statut": np.where(today["alert"], "🔴 ALERTE", "🟢 Normal"),
        })
        if has_truth:
            table["Simulation (vérité)"] = today["scenario"].values

        def red_rows(row):
            style = ("background-color: #ffccd0; color: #7a0010; "
                     "font-weight: 600") if row["Statut"] == "🔴 ALERTE" \
                else ""
            return [style] * len(row)

        st.dataframe(
            table.head(25).style.apply(red_rows, axis=1),
            hide_index=True, use_container_width=True,
            column_config={"Score de risque": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%d")},
        )
        st.caption("Top 25 des utilisateurs du jour, triés par score de "
                   "risque. Une alerte est un signal à investiguer, pas "
                   "une preuve d'attaque.")


with tab_live:
    days = sorted(results["date"].unique())
    col_a, col_b = st.columns([3, 1])
    day = col_a.select_slider("Journée", options=days, value=days[0])
    speed = col_b.selectbox("Vitesse", [1.5, 1.0, 0.5], index=0,
                            format_func=lambda s: f"{s} s / jour")
    play = st.button("▶️ Simuler jour par jour", type="primary")
    placeholder = st.empty()
    if play:
        for d in days:
            render_day(d, placeholder)
            time.sleep(speed)
    else:
        render_day(day, placeholder)


# ==================================================
# 2. ANALYSE DE LA PERTE
# ==================================================
with tab_loss:
    st.subheader("Distribution de l'erreur de reconstruction")
    fig = px.histogram(
        results, x="error", color="scenario", nbins=120,
        barmode="overlay", opacity=0.75, log_y=True,
        color_discrete_map=COLORS,
        labels={"error": "Erreur de reconstruction ‖x − x̂‖²",
                "scenario": "Comportement"})
    fig.add_vline(x=threshold, line_dash="dash", line_color="black",
                  annotation_text=f"Seuil (k={k})")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Échelle logarithmique sur l'axe vertical pour voir les "
               "petits groupes. Bougez k dans la barre latérale : la "
               "ligne du seuil se déplace.")

    st.subheader("Courbe d'apprentissage")
    hist = meta["history"]
    fig_loss = go.Figure()
    fig_loss.add_scatter(y=hist["train_loss"], name="Entraînement")
    fig_loss.add_scatter(y=hist["val_loss"], name="Validation")
    fig_loss.update_layout(xaxis_title="Époque", yaxis_title="Perte MSE")
    st.plotly_chart(fig_loss, use_container_width=True)


# ==================================================
# 3. ÉVALUATION
# ==================================================
with tab_eval:
    if not has_truth:
        st.info("Ce jeu de données ne contient pas d'attaques simulées : "
                "pas d'évaluation possible.")
    else:
        attack = results["is_attack"] == 1
        tp = int((results["alert"] & attack).sum())
        fn = int((~results["alert"] & attack).sum())
        fp = int((results["alert"] & ~attack).sum())
        tn = int((~results["alert"] & ~attack).sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Attaques détectées (rappel)",
                  f"{100 * tp / (tp + fn):.1f} %", f"{tp}/{tp + fn}")
        c2.metric("Fausses alertes (FPR)",
                  f"{100 * fp / (fp + tn):.2f} %", f"{fp}/{fp + tn}",
                  delta_color="inverse")
        c3.metric("Précision", f"{100 * tp / max(tp + fp, 1):.1f} %")

        per_scenario = (results[attack].groupby("scenario")["alert"]
                        .agg(["sum", "count"]))
        per_scenario["Détection"] = (100 * per_scenario["sum"]
                                     / per_scenario["count"]).round(1)
        st.dataframe(per_scenario.rename(columns={
            "sum": "Détectées", "count": "Total"}),
            use_container_width=True)

        st.subheader("Matrice de confusion")
        cm = pd.DataFrame([[tn, fp], [fn, tp]],
                          index=["Réel : normal", "Réel : attaque"],
                          columns=["Prédit : normal", "Prédit : alerte"])
        st.plotly_chart(px.imshow(cm, text_auto=True,
                                  color_continuous_scale="Blues"),
                        use_container_width=True)


# ==================================================
# 4. DÉTAIL UTILISATEUR
# ==================================================
with tab_user:
    ranking = (results.groupby("user_id")["risk"].max()
               .sort_values(ascending=False))
    user = st.selectbox(
        "Utilisateur (triés du plus risqué au moins risqué)",
        ranking.index,
        format_func=lambda u: f"{u}  ·  risque max {int(ranking[u])}")

    hist_u = results[results["user_id"] == user].sort_values("date")
    fig_u = px.line(hist_u, x="date", y="risk", markers=True,
                    labels={"risk": "Score de risque", "date": "Date"})
    fig_u.add_hline(y=50, line_dash="dash", line_color="red",
                    annotation_text="Seuil d'alerte")
    fig_u.update_yaxes(range=[0, 105])
    st.plotly_chart(fig_u, use_container_width=True)

    worst = hist_u["error"].idxmax()
    st.markdown(f"**Pourquoi ?** Variables les moins bien reconstruites "
                f"le {results.loc[worst, 'date']} :")
    contrib = pd.Series(feat_err[worst], index=meta["feature_cols"])
    top = contrib.sort_values(ascending=False).head(8)
    st.plotly_chart(px.bar(top[::-1], orientation="h",
                           labels={"value": "(x − x̂)²", "index": ""}),
                    use_container_width=True)


# ==================================================
# 5. FLUX KAFKA (STREAMING TEMPS RÉEL)
# ==================================================
STREAM_DB = V2_DIR / "stream.db"


@st.fragment(run_every=2)          # seule cette partie se rafraîchit
def kafka_live():
    if not STREAM_DB.exists():
        st.info("Aucun flux reçu pour l'instant. Lancez dans 3 terminaux :")
        st.code("docker compose -f v2/docker-compose.yml up -d\n"
                "python v2/kafka_detector.py --reset\n"
                "python v2/kafka_producer.py", language="bash")
        return
    try:
        with sqlite3.connect(STREAM_DB) as con:
            live = pd.read_sql("SELECT * FROM scores", con)
    except (sqlite3.Error, pd.errors.DatabaseError):
        live = pd.DataFrame()
    if live.empty:
        st.info("Détecteur prêt, en attente des messages Kafka...")
        return

    # Le seuil suit le curseur k de la barre latérale
    live["risk"] = risk_score(live["error"], threshold)
    live["alert"] = live["error"] > threshold
    alerts = live[live["alert"]]
    lag = time.time() - live["received_at"].max()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Messages reçus", f"{len(live):,}")
    c2.metric("Journée en cours", live["date"].iloc[-1])
    c3.metric("🔴 Alertes", len(alerts))
    c4.metric("Dernier message", "en direct" if lag < 5
              else f"il y a {int(lag)} s")
    if lag > 15:
        st.warning("⏸️ Plus aucun message depuis "
                   f"{int(lag)} s : le flux est terminé ou arrêté. Si ce n'est "
                   "pas la fin, vérifiez que Kafka (Docker) et le producteur "
                   "tournent toujours.")

    per_day = (live.groupby("date")
               .agg(messages=("user_id", "size"), alertes=("alert", "sum"))
               .reset_index())
    fig = px.bar(per_day, x="date", y="alertes", text="alertes",
                 hover_data=["messages"],
                 labels={"date": "Journée", "alertes": "Alertes",
                         "messages": "Messages"},
                 color_discrete_sequence=["#e63946"])
    fig.update_xaxes(type="category")      # une barre par journée reçue
    fig.update_yaxes(rangemode="tozero", dtick=1)
    fig.update_layout(height=260, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Dernières alertes** (les plus récentes en haut)")
    if alerts.empty:
        st.success("✅ Aucune alerte pour l'instant. (Dans le jeu de test, "
                   "les attaques simulées commencent le 2024-12-21.)")
    else:
        show = alerts.sort_values("received_at", ascending=False).head(15)
        st.dataframe(pd.DataFrame({
            "Journée": show["date"],
            "Utilisateur": show["user_id"].astype(str),
            "Score de risque": show["risk"].astype(int),
            "Causes principales": show["top_features"].map(
                lambda s: ", ".join(json.loads(s))),
            "Simulation (vérité)": show["scenario"],
        }), hide_index=True, use_container_width=True,
            column_config={"Score de risque": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%d")})


with tab_kafka:
    st.subheader("📡 Détection en temps réel via Kafka")
    st.caption("Producteur → topic « euba-events » → détecteur (autoencodeur) "
               "→ alertes (topic « euba-alerts ») → ce tableau, rafraîchi "
               "toutes les 2 secondes.")
    kafka_live()
