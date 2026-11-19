"""
EUBA v2 - Code partagé : modèle, préparation des données, score d'anomalie.

Utilisé par train.py, simulate_attacks.py et app.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

V2_DIR = Path(__file__).resolve().parent
PROJECT_DIR = V2_DIR.parent

RAW_DATA = PROJECT_DIR / "euba_test_data.csv"      # données sources
ARTIFACTS_DIR = V2_DIR / "artifacts"               # modèle v2 sauvegardé
TEST_DATA = V2_DIR / "euba_v2_test_with_attacks.csv"  # jeu de test + attaques

# Colonnes qui ne sont PAS des comportements (jamais données au modèle)
NON_FEATURES = ["user_id", "date", "RiskScore", "RiskLabel",
                "is_attack", "scenario"]

# Découpage temporel : on apprend sur le passé, on teste sur le futur
SPLIT_DATE = "2024-12-17"   # train : < 17/12   |   test : >= 17/12


class EUBAAutoencoder(nn.Module):
    """Autoencodeur : 72 -> 64 -> 32 -> 8 (goulot) -> 32 -> 64 -> 72."""

    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def load_raw():
    """Charge le CSV source et garde uniquement le trafic légitime.

    Les journées RiskLabel = High sont déjà suspectes : on les retire
    pour avoir une base 'normale' propre.
    """
    df = pd.read_csv(RAW_DATA)
    return df[df["RiskLabel"] != "High"].reset_index(drop=True)


def reconstruction_errors(model, x_scaled):
    """Erreur de reconstruction par ligne : moyenne de (x - x_hat)^2."""
    x = torch.as_tensor(np.asarray(x_scaled), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        x_hat = model(x)
        errors = ((x - x_hat) ** 2).mean(dim=1)
    return errors.numpy()


def feature_errors(model, x_scaled):
    """Erreur par variable (x - x_hat)^2 : sert à expliquer une alerte."""
    x = torch.as_tensor(np.asarray(x_scaled), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        return ((x - model(x)) ** 2).numpy()


def dynamic_threshold(normal_errors, k):
    """Seuil dynamique = moyenne + k * écart-type des erreurs normales."""
    return float(np.mean(normal_errors) + k * np.std(normal_errors))


def risk_score(errors, threshold):
    """Score de risque 0-100. Le seuil correspond à 50 (= alerte)."""
    return np.clip(50 * np.asarray(errors) / threshold, 0, 100).round(0)
