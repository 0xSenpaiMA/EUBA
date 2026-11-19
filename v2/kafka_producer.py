"""
EUBA v2 - Streaming, étape A : PRODUCTEUR Kafka.

Simule les journaux qui arrivent du SI de l'entreprise : chaque message
est le résumé du comportement d'un utilisateur sur une journée, envoyé
sur le topic Kafka « euba-events », jour après jour.

Lancement :  python v2/kafka_producer.py --delay 0.02
"""
import argparse
import json
import sys
import time

import pandas as pd
from kafka import KafkaProducer
from kafka.errors import KafkaError

from euba_model import TEST_DATA

TOPIC_EVENTS = "euba-events"

parser = argparse.ArgumentParser()
parser.add_argument("--broker", default="localhost:9092")
parser.add_argument("--delay", type=float, default=0.02,
                    help="pause entre deux messages (secondes)")
args = parser.parse_args()

KAFKA_DOWN = ("\n❌ Kafka est injoignable sur {}.\n"
              "   Vérifiez que Docker Desktop est ouvert, puis lancez :\n"
              "   docker compose -f v2/docker-compose.yml up -d")

try:
    producer = KafkaProducer(
        bootstrap_servers=args.broker,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: str(k).encode("utf-8"),
        max_block_ms=5000, request_timeout_ms=5000,
    )
except KafkaError:
    sys.exit(KAFKA_DOWN.format(args.broker))

df = pd.read_csv(TEST_DATA).sort_values(["date", "user_id"])
print(f"Envoi de {len(df):,} messages sur '{TOPIC_EVENTS}' "
      f"({df['date'].nunique()} jours)...")
print("   (les premières attaques simulées commencent le "
      f"{df.loc[df['is_attack'] == 1, 'date'].min()})")

current_day = None
for i, row in enumerate(df.to_dict(orient="records"), 1):
    if row["date"] != current_day:
        current_day = row["date"]
        print(f"  -> journée {current_day}")
    # Clé = user_id : tous les messages d'un utilisateur restent ordonnés.
    # .get() attend l'accusé de réception : si Kafka tombe, on le voit tout
    # de suite au lieu de rester bloqué en silence.
    try:
        producer.send(TOPIC_EVENTS, key=row["user_id"],
                      value=row).get(timeout=10)
    except KafkaError:
        sys.exit(KAFKA_DOWN.format(args.broker)
                 + f"\n   (arrêt au message {i:,}, journée {current_day})")
    time.sleep(args.delay)

producer.flush()
print(f"Terminé : {i:,} messages envoyés.")
