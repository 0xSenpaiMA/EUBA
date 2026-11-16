"""
EUBA v2 - Streaming, étape B : DÉTECTEUR (consumer Kafka).

Lit chaque message du topic « euba-events », calcule l'erreur de
reconstruction avec l'autoencodeur entraîné, puis :
  - enregistre le résultat dans v2/stream.db (lu par le tableau de bord) ;
  - publie les alertes sur le topic « euba-alerts ».

Lancement :  python v2/kafka_detector.py --reset
"""
import argparse
import json
import sqlite3
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

from euba_model import (ARTIFACTS_DIR, V2_DIR, EUBAAutoencoder,
                        dynamic_threshold, feature_errors, risk_score)

TOPIC_EVENTS = "euba-events"
TOPIC_ALERTS = "euba-alerts"
DB_PATH = V2_DIR / "stream.db"

parser = argparse.ArgumentParser()
parser.add_argument("--broker", default="localhost:9092")
parser.add_argument("--reset", action="store_true",
                    help="vide la base des scores avant de démarrer")
args = parser.parse_args()

# --- Modèle entraîné ---------------------------------------------------
meta = joblib.load(ARTIFACTS_DIR / "metadata.joblib")
scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
model = EUBAAutoencoder(meta["input_dim"])
model.load_state_dict(torch.load(ARTIFACTS_DIR / "autoencoder.pth",
                                 weights_only=True))
model.eval()
cols = meta["feature_cols"]
threshold = dynamic_threshold(meta["val_errors"], meta["default_k"])

# --- Base partagée avec le tableau de bord ------------------------------
db = sqlite3.connect(DB_PATH)
db.execute("PRAGMA journal_mode=WAL")      # lecture/écriture simultanées
if args.reset:
    db.execute("DROP TABLE IF EXISTS scores")
db.execute("""CREATE TABLE IF NOT EXISTS scores (
    received_at REAL, user_id INTEGER, date TEXT, scenario TEXT,
    is_attack INTEGER, error REAL, risk REAL, alert INTEGER,
    top_features TEXT)""")
db.commit()

# --- Kafka ---------------------------------------------------------------
try:
    # Le producteur d'alertes échoue en 5 s si Kafka est absent :
    # on le crée en premier pour vérifier que le broker répond.
    alerts_out = KafkaProducer(
        bootstrap_servers=args.broker,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        max_block_ms=5000,
    )
    consumer = KafkaConsumer(
        TOPIC_EVENTS,
        bootstrap_servers=args.broker,
        group_id="euba-detector",
        auto_offset_reset="earliest",
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
    )
except KafkaError:
    sys.exit(f"\n❌ Kafka est injoignable sur {args.broker}.\n"
             "   Vérifiez que Docker Desktop est ouvert, puis lancez :\n"
             "   docker compose -f v2/docker-compose.yml up -d")

print(f"Détecteur prêt : seuil = {threshold:.5f} (k={meta['default_k']}). "
      f"En attente de messages sur '{TOPIC_EVENTS}'...")

def process(event):
    x = scaler.transform(pd.DataFrame([event])[cols])
    per_feature = feature_errors(model, x)[0]
    error = float(per_feature.mean())          # = erreur de reconstruction
    risk = float(risk_score([error], threshold)[0])
    alert = error > threshold
    top = pd.Series(per_feature, index=cols).nlargest(3).index.tolist()

    db.execute("INSERT INTO scores VALUES (?,?,?,?,?,?,?,?,?)", (
        time.time(), int(event["user_id"]), event["date"],
        event.get("scenario", "Inconnu"), int(event.get("is_attack", 0)),
        error, risk, int(alert), json.dumps(top)))

    if alert:
        alerts_out.send(TOPIC_ALERTS, {
            "user_id": int(event["user_id"]), "date": event["date"],
            "risk": risk, "error": round(error, 5), "top_features": top})
        print(f"🚨 ALERTE  {event['date']}  utilisateur {event['user_id']}"
              f"  risque={int(risk)}  causes={', '.join(top)}")


# Boucle principale : on lit les messages par lots, et on enregistre
# après chaque lot (le tableau de bord voit donc tout, même le dernier).
while True:
    batches = consumer.poll(timeout_ms=500)
    for records in batches.values():
        for msg in records:
            process(msg.value)
    if batches:
        db.commit()
