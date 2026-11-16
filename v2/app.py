"""
EUBA v2 - Tableau de bord SOC (Streamlit).

   Flux Kafka      : détection en temps réel ; cliquer sur un utilisateur du
                     tableau des alertes affiche son détail et ses flags.
   Évaluation      : performances du modèle sur le jeu de test avec attaques.
   Dataset importé : analyse d'un CSV chargé depuis la barre latérale.

Lancement :  streamlit run v2/app.py   (ou run.ps1 à la racine du dépôt)
"""
import json
import sqlite3
import time
from functools import partial

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch

from euba_model import (ARTIFACTS_DIR, TEST_DATA, V2_DIR, EUBAAutoencoder,
                        dynamic_threshold, feature_errors, risk_score,
                        reconstruction_errors)

STREAM_DB = V2_DIR / "stream.db"

st.set_page_config(page_title="EUBA v2 | SOC", page_icon="🛡️",
                   layout="wide")


# ==================================================
# CHARGEMENT DU MODÈLE
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


try:
    model, scaler, meta = load_model()
except FileNotFoundError:
    st.error("Modèle introuvable. Lancez d'abord : python v2/train.py")
    st.stop()

cols = meta["feature_cols"]
k = float(meta["default_k"])
threshold = dynamic_threshold(meta["val_errors"], k)


# ==================================================
# BARRE LATÉRALE : IMPORT D'UN DATASET
# ==================================================
st.sidebar.header("📂 Importer un dataset")
uploaded = st.sidebar.file_uploader("Fichier CSV", type="csv")
st.sidebar.caption("")
with st.sidebar.expander(""):
    st.code("user_id\ndate\n" + "\n".join(cols), language=None)


# ==================================================
# EN-TÊTE
# ==================================================
st.title(" EUBA v2 · Détection d'anomalies comportementales")
st.caption(f" ")

tab_kafka, tab_eval, tab_upload = st.tabs(
    ["📡 Flux Kafka", " Évaluation", " Dataset importé"])


# ==================================================
# COMPOSANTS PARTAGÉS : TABLEAU DES ALERTES + DÉTAIL UTILISATEUR
# (utilisés par l'onglet Kafka et par l'onglet Dataset importé)
# ==================================================
def remember_click(table_key, users, ctx):
    """Mémorise l'utilisateur cliqué (il reste affiché même si le tableau
    se met à jour avec de nouvelles alertes)."""
    selection = st.session_state[table_key].selection
    # Clic sur la case de la ligne (rows) OU sur n'importe quelle cellule
    rows = selection.get("rows", []) or [c[0] for c in
                                          selection.get("cells", [])]
    if rows:
        st.session_state[f"picked_{ctx}"] = users[rows[0]]


def user_detail(live, user, ctx):
    """Historique de risque + flags responsables, pour un utilisateur."""
    hist = live[live["user_id"] == user].sort_values("date")
    worst = hist.loc[hist["error"].idxmax()]

    st.divider()
    head, close = st.columns([6, 1])
    head.subheader(f"🔎 Utilisateur {user}")
    if close.button("✖ Fermer", key=f"close_{ctx}"):
        st.session_state.pop(f"picked_{ctx}", None)
        st.rerun(scope="fragment" if ctx == "kafka" else "app")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risque max", int(hist["risk"].max()))
    c2.metric("Journées en alerte", int(hist["alert"].sum()))
    c3.metric("Journées reçues", len(hist))
    c4.metric("Simulation (vérité)", worst["scenario"])

    left, right = st.columns(2)
    with left:
        st.markdown("**Score de risque jour par jour**")
        fig = px.line(hist, x="date", y="risk", markers=True,
                      labels={"risk": "Score de risque", "date": "Journée"})
        fig.add_hline(y=50, line_dash="dash", line_color="red",
                      annotation_text="Seuil d'alerte")
        fig.update_xaxes(type="category")
        fig.update_yaxes(range=[0, 105])
        fig.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True,
                        key=f"curve_{ctx}")

    if "features" not in hist.columns or hist["features"].isna().all():
        right.info("Détail des flags indisponible : relancez le détecteur "
                   "avec --reset pour enregistrer les événements complets.")
        return

    # Le modèle reconstruit l'événement : on compare observé / attendu
    event = pd.DataFrame([json.loads(worst["features"])])[cols]
    x = scaler.transform(event)
    with torch.no_grad():
        x_hat = model(torch.tensor(x, dtype=torch.float32)).numpy()
    expected = scaler.inverse_transform(x_hat)[0]
    errs = feature_errors(model, x)[0]
    flags = (pd.DataFrame({
        "Variable": cols,
        "Observé": event.iloc[0].to_numpy(),
        "Attendu par le modèle": expected.round(2),
        "Erreur": errs,
    }).sort_values("Erreur", ascending=False).head(8))

    with right:
        st.markdown(f"**Pourquoi ? Flags les plus anormaux le "
                    f"{worst['date']}** (risque {int(worst['risk'])})")
        bar = px.bar(flags[::-1], x="Erreur", y="Variable", orientation="h",
                     labels={"Erreur": "(x − x̂)²", "Variable": ""},
                     color_discrete_sequence=["#1f77b4"])
        bar.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(bar, use_container_width=True, key=f"flags_{ctx}")

    st.dataframe(flags, hide_index=True, use_container_width=True,
                 column_config={"Erreur": st.column_config.ProgressColumn(
                     min_value=0, max_value=1, format="%.3f")})
    st.caption("« Observé » = valeur reçue ; « Attendu » = ce que le modèle "
               "prévoit pour un comportement normal. Un flag observé à 1 là "
               "où le modèle attend environ 0 est inhabituel.")


def alerts_section(live, ctx, empty_msg):
    """Graphique des alertes par jour + tableau cliquable + détail."""
    alerts = live[live["alert"]]

    per_day = (live.groupby("date")
               .agg(messages=("user_id", "size"), alertes=("alert", "sum"))
               .reset_index())
    fig = px.bar(per_day, x="date", y="alertes", text="alertes",
                 hover_data=["messages"],
                 labels={"date": "Journée", "alertes": "Alertes",
                         "messages": "Lignes"},
                 color_discrete_sequence=["#e63946"])
    fig.update_xaxes(type="category")
    fig.update_yaxes(rangemode="tozero", dtick=1)
    fig.update_layout(height=240, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True, key=f"perday_{ctx}")

    st.markdown("**Utilisateurs en alerte** — 👆 cliquez sur une ligne "
                "pour voir le détail et les flags")
    if alerts.empty:
        st.success(empty_msg)
    else:
        # Une ligne par utilisateur, la dernière alerte en haut
        last = alerts.sort_values("received_at").groupby("user_id").tail(1)
        per_user = (alerts.groupby("user_id")
                    .agg(n=("alert", "size"), risk=("risk", "max"),
                         at=("received_at", "max"))
                    .join(last.set_index("user_id")[
                        ["date", "top_features", "scenario"]])
                    .sort_values("at", ascending=False).reset_index())
        users = per_user["user_id"].tolist()
        table = pd.DataFrame({
            "Utilisateur": per_user["user_id"].astype(str),
            "Alertes": per_user["n"],
            "Risque max": per_user["risk"].astype(int),
            "Dernière alerte": per_user["date"],
            "Causes principales": per_user["top_features"].map(
                lambda s: ", ".join(json.loads(s))),
            "Simulation (vérité)": per_user["scenario"],
        })
        # La clé change quand une alerte arrive : la sélection est remise à
        # zéro au lieu de surligner une mauvaise ligne (les lignes bougent).
        key = f"alerts_{ctx}_{len(alerts)}"
        st.dataframe(
            table, key=key, hide_index=True, use_container_width=True,
            on_select=partial(remember_click, key, users, ctx),
            selection_mode=["single-row", "single-cell"],
            column_config={"Risque max": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%d")})

    picked = st.session_state.get(f"picked_{ctx}")
    if picked is not None and picked in set(live["user_id"]):
        user_detail(live, picked, ctx)


def evaluation_section(results, ctx):
    """Rappel, FPR, précision, détection par scénario, matrice."""
    alert = results["error"] > threshold
    attack = results["is_attack"] == 1
    tp = int((alert & attack).sum())
    fn = int((~alert & attack).sum())
    fp = int((alert & ~attack).sum())
    tn = int((~alert & ~attack).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Attaques détectées (rappel)",
              f"{100 * tp / max(tp + fn, 1):.1f} %", f"{tp}/{tp + fn}")
    c2.metric("Fausses alertes (FPR)",
              f"{100 * fp / max(fp + tn, 1):.2f} %", f"{fp}/{fp + tn}",
              delta_color="inverse")
    c3.metric("Précision", f"{100 * tp / max(tp + fp, 1):.1f} %")

    if attack.any():
        per_scenario = (pd.DataFrame({"scenario": results["scenario"],
                                      "alert": alert})[attack]
                        .groupby("scenario")["alert"].agg(["sum", "count"]))
        per_scenario["Détection (%)"] = (100 * per_scenario["sum"]
                                         / per_scenario["count"]).round(1)
        st.dataframe(per_scenario.rename(columns={
            "sum": "Détectées", "count": "Total"}), use_container_width=True)

    st.subheader("Matrice de confusion")
    cm = pd.DataFrame([[tn, fp], [fn, tp]],
                      index=["Réel : normal", "Réel : attaque"],
                      columns=["Prédit : normal", "Prédit : alerte"])
    labels = [[f"TN = {tn}<br>", f"FP = {fp}<br>"],
              [f"FN = {fn}<br>", f"TP = {tp}<br>"]]
    fig_cm = px.imshow(cm, color_continuous_scale="Blues")
    fig_cm.update_traces(text=labels, texttemplate="%{text}",
                         textfont_size=16)
    st.plotly_chart(fig_cm, use_container_width=True, key=f"cm_{ctx}")
    st.caption(" ")


# ==================================================
# 1. FLUX KAFKA
# ==================================================
def read_stream():
    """Lit les scores écrits par le détecteur Kafka."""
    if not STREAM_DB.exists():
        return None
    try:
        with sqlite3.connect(STREAM_DB) as con:
            return pd.read_sql("SELECT * FROM scores", con)
    except (sqlite3.Error, pd.errors.DatabaseError):
        return pd.DataFrame()


@st.fragment(run_every=2)          # seule cette partie se rafraîchit
def kafka_live():
    live = read_stream()
    if live is None:
        st.info("Aucun flux reçu pour l'instant. Lancez :")
        st.code("powershell -ExecutionPolicy Bypass -File run.ps1 -Kafka",
                language="bash")
        return
    if live.empty:
        st.info("Détecteur prêt, en attente des messages Kafka...")
        return

    live["risk"] = risk_score(live["error"], threshold)
    live["alert"] = live["error"] > threshold
    lag = time.time() - live["received_at"].max()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Messages reçus", f"{len(live):,}")
    c2.metric("Journée en cours", live["date"].iloc[-1])
    c3.metric("🔴 Alertes", int(live["alert"].sum()))
    c4.metric("Dernier message", "en direct" if lag < 5
              else f"il y a {int(lag)} s")
    if lag > 15:
        st.warning(f" Plus aucun message depuis {int(lag)} s : le flux est "
                   "terminé ou arrêté.")

    alerts_section(live, "kafka",
                   "✅ Aucune alerte pour l'instant. (Dans le jeu de test, "
                   "les attaques simulées commencent le 2024-12-21.)")


with tab_kafka:
    st.subheader("📡 Détection en temps réel via Kafka")
    st.caption(" ")
    kafka_live()


# ==================================================
# 2. ÉVALUATION (jeu de test complet avec attaques simulées)
# ==================================================
@st.cache_data
def score_test_set():
    df = pd.read_csv(TEST_DATA)
    out = df[["user_id", "date", "scenario", "is_attack"]].copy()
    out["error"] = reconstruction_errors(model, scaler.transform(df[cols]))
    return out


with tab_eval:
    if not TEST_DATA.exists():
        st.info("Jeu de test introuvable. Lancez : "
                "python v2/simulate_attacks.py")
    else:
        evaluation_section(score_test_set(), "test")


# ==================================================
# 3. DATASET IMPORTÉ (barre latérale)
# ==================================================
@st.cache_data(show_spinner="Analyse du dataset par l'autoencodeur...")
def score_uploaded(raw_bytes):
    """Score chaque ligne du CSV importé, comme le ferait le détecteur."""
    import io
    df = pd.read_csv(io.BytesIO(raw_bytes))
    missing = [c for c in ["user_id", "date"] + cols if c not in df.columns]
    if missing:
        return None, missing
    df = df.dropna(subset=["user_id", "date"] + cols).reset_index(drop=True)
    df["date"] = df["date"].astype(str)
    x = scaler.transform(df[cols])
    per_feature = feature_errors(model, x)
    out = pd.DataFrame({
        "received_at": np.arange(len(df), dtype=float),   # ordre du fichier
        "user_id": df["user_id"],
        "date": df["date"],
        "scenario": (df["scenario"].astype(str) if "scenario" in df.columns
                     else "Inconnu"),
        "is_attack": (df["is_attack"].astype(int)
                      if "is_attack" in df.columns else 0),
        "error": per_feature.mean(axis=1),
    })
    top_idx = np.argsort(-per_feature, axis=1)[:, :3]
    out["top_features"] = [json.dumps([cols[j] for j in row])
                           for row in top_idx]
    out["features"] = df[cols].to_json(orient="records", lines=True) \
        .strip().split("\n")
    return out, []


with tab_upload:
    st.subheader("📂 Analyse d'un dataset importé")
    if uploaded is None:
        st.info("👈 Importez un fichier CSV dans la barre latérale à gauche "
                "pour l'analyser avec le modèle.")
    else:
        result, missing = score_uploaded(uploaded.getvalue())
        if result is None:
            st.error(f"Colonnes manquantes dans « {uploaded.name} » "
                     f"({len(missing)}) : {', '.join(missing[:10])}"
                     + (" ..." if len(missing) > 10 else ""))
        elif result.empty:
            st.warning("Aucune ligne exploitable dans ce fichier.")
        else:
            result["risk"] = risk_score(result["error"], threshold)
            result["alert"] = result["error"] > threshold
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Fichier", uploaded.name)
            c2.metric("Lignes analysées", f"{len(result):,}")
            c3.metric("Utilisateurs", f"{result['user_id'].nunique():,}")
            c4.metric("🔴 Alertes", int(result["alert"].sum()))

            alerts_section(result, "upload",
                           "✅ Aucune alerte : aucun comportement ne dépasse "
                           "le seuil dans ce fichier.")

            if result["is_attack"].sum() > 0:
                st.divider()
                st.subheader("✅ Évaluation sur ce fichier")
                st.caption("La colonne is_attack permet de comparer les "
                           "alertes du modèle à la vérité.")
                evaluation_section(result, "upload")
