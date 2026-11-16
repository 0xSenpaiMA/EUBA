# EUBA v2 — Lancer le projet

## En une seule commande (Windows)

Depuis la racine du dépôt, dans PowerShell :

```powershell
powershell -ExecutionPolicy Bypass -File run.ps1
```

Le script fait tout, dans l'ordre :

1. crée l'environnement Python `.venv` s'il n'existe pas ;
2. installe les dépendances de `v2/requirements.txt` (seulement la première fois, ou si le fichier change) ;
3. entraîne l'autoencodeur s'il n'est pas déjà dans `v2/artifacts/` ;
4. génère le jeu de test avec attaques simulées (`v2/euba_v2_test_with_attacks.csv`) ;
5. ouvre le tableau de bord Streamlit sur <http://localhost:8501>.

### Avec le streaming Kafka

Prérequis : **Docker Desktop** ouvert.

```powershell
powershell -ExecutionPolicy Bypass -File run.ps1 -Kafka
```

En plus des étapes ci-dessus : démarre Kafka dans Docker, ouvre une fenêtre
pour le **détecteur** et une pour le **producteur**. Ouvrez ensuite l'onglet
**« 📡 Flux Kafka »** du tableau de bord. Les premières alertes arrivent à la
journée 2024-12-21.

### Options

| Option | Effet |
|---|---|
| `-Kafka` | lance aussi Kafka, le détecteur et le producteur |
| `-Delay 0.005` | flux Kafka plus rapide (défaut : `0.02` s entre deux messages) |
| `-Retrain` | réentraîne le modèle et régénère les attaques |
| `-Port 8502` | change le port du tableau de bord |

Exemple, démo rapide : `powershell -ExecutionPolicy Bypass -File run.ps1 -Kafka -Delay 0.005`

Après la démo, arrêter Kafka :

```powershell
docker compose -f v2/docker-compose.yml down
```

## Étape par étape (toutes plateformes)

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux / macOS : source .venv/bin/activate
pip install -r v2/requirements.txt

python v2/train.py                # 1. entraînement de l'autoencodeur
python v2/simulate_attacks.py     # 2. attaques simulées + évaluation
streamlit run v2/app.py           # 3. tableau de bord

# Streamlit + Kafka (3 terminaux)
docker compose -f v2/docker-compose.yml up -d
python v2/kafka_detector.py --reset
python v2/kafka_producer.py --delay 0.02
```

## Structure de la v2

| Fichier | Rôle |
|---|---|
| `euba_test_data.csv` | données sources (une ligne = un utilisateur sur une journée) |
| `v2/euba_model.py` | code partagé : modèle, chargement des données, erreur, seuil, score |
| `v2/train.py` | prétraitement + entraînement de l'autoencodeur |
| `v2/simulate_attacks.py` | injection d'attaques simulées + évaluation |
| `v2/app.py` | tableau de bord Streamlit (5 onglets) |
| `v2/kafka_producer.py` | envoie les événements sur le topic `euba-events` |
| `v2/kafka_detector.py` | consomme le flux, calcule les scores, publie sur `euba-alerts` |
| `v2/docker-compose.yml` | Kafka 3.8 (un nœud, mode KRaft) |
| `v2/demo_kafka.ps1` | lance Kafka + détecteur + producteur |
| `v2/artifacts/` | modèle entraîné (autoencodeur, scaler, métadonnées) |

> Note Windows : clonez le dépôt dans un dossier au chemin **court**
> (ex. `C:\EUBA`). Sinon l'installation de torch peut échouer à cause de la
> limite de 260 caractères des chemins Windows.
>
> Note Windows : `v2/requirements.txt` fixe les **versions exactes** testées
> (Python 3.14), car des versions plus récentes de numpy, pandas et pyarrow ont
> été bloquées par Windows Smart App Control lors des tests.
