"""
EUBA v2 - Étape 2 : attaques simulées + évaluation.

On prend la période de test (à partir du 17/12/2024) et on modifie
quelques comptes pour simuler deux menaces internes :

  - Usurpation de compte : connexion inhabituelle (lieu, appareil, heure),
    échecs de connexion, accès aux fichiers sensibles, logiciel non autorisé.
  - Exfiltration : gros transfert vers l'extérieur, envoi/upload de données
    sensibles, insertion de clé USB, volume réseau élevé.

Le modèle n'a jamais vu ces attaques : on mesure s'il les détecte.

Lancement :  python v2/simulate_attacks.py   (après train.py)
"""
import joblib
import numpy as np
import torch

from euba_model import (ARTIFACTS_DIR, SPLIT_DATE, TEST_DATA,
                        EUBAAutoencoder, dynamic_threshold, load_raw,
                        reconstruction_errors)

SEED = 7
USERS_PER_SCENARIO = 5     # 5 comptes usurpés + 5 comptes exfiltrants
ATTACK_DAYS = 2            # chaque attaque dure 2 jours

# Valeurs imposées pendant l'attaque (valeur codée la plus "extrême")
SCENARIOS = {
    "Usurpation": {
        "LoginAttempts_flag": 1,
        "LoginAttempts_failed_attempts": 2,
        "LoginAttempts_login_location": 1,
        "LoginAttempts_device_type": 2,
        "LoginAttempts_login_time": 1,
        "SensitiveFilesAccess_flag": 1,
        "SensitiveFilesAccess_access_location": 1,
        "SensitiveFilesAccess_access_time": 1,
        "UnauthorizedSoftware_flag": 1,
        "BehaviorChange_flag": 1,
        "NetworkInteraction_flag": 1,
    },
    "Exfiltration": {
        "DataTransfer_flag": 1,
        "DataTransfer_file_size": 2,
        "DataTransfer_destination": 2,
        "UploadSensitiveInformation_flag": 1,
        "SendSensitiveInformation_flag": 1,
        "AttemptedThumbDriveInsertion_flag": 1,
        "SensitiveFilesAccess_flag": 1,
        "NetworkInteraction_flag": 1,
        "NetworkInteraction_data_volume": 2,
    },
}

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------
# 1. Période de test (trafic légitime) + injection des attaques
# ---------------------------------------------------------------
df = load_raw()
test = df[df["date"] >= SPLIT_DATE].copy().reset_index(drop=True)
test["is_attack"] = 0
test["scenario"] = "Normal"

days = sorted(test["date"].unique())
users = rng.choice(test["user_id"].unique(),
                   size=USERS_PER_SCENARIO * len(SCENARIOS), replace=False)

for i, (name, changes) in enumerate(SCENARIOS.items()):
    for user in users[i * USERS_PER_SCENARIO:(i + 1) * USERS_PER_SCENARIO]:
        # L'attaque commence un jour au hasard (pas le 1er, pour la démo live)
        start = rng.integers(2, len(days) - ATTACK_DAYS + 1)
        attack_days = days[start:start + ATTACK_DAYS]
        mask = (test["user_id"] == user) & test["date"].isin(attack_days)
        for col, value in changes.items():
            test.loc[mask, col] = value
        test.loc[mask, "is_attack"] = 1
        test.loc[mask, "scenario"] = name

test.to_csv(TEST_DATA, index=False)
print(f"Jeu de test : {len(test):,} lignes, "
      f"{test['is_attack'].sum()} journées attaquées "
      f"sur {test.loc[test.is_attack == 1, 'user_id'].nunique()} comptes")
print(f"Sauvegardé : {TEST_DATA.name}\n")

# ---------------------------------------------------------------
# 2. Évaluation avec le modèle entraîné
# ---------------------------------------------------------------
meta = joblib.load(ARTIFACTS_DIR / "metadata.joblib")
scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
model = EUBAAutoencoder(meta["input_dim"])
model.load_state_dict(torch.load(ARTIFACTS_DIR / "autoencoder.pth",
                                 weights_only=True))

errors = reconstruction_errors(
    model, scaler.transform(test[meta["feature_cols"]]))
threshold = dynamic_threshold(meta["val_errors"], meta["default_k"])
alert = errors > threshold
attack = test["is_attack"].to_numpy() == 1

tp = int((alert & attack).sum())
fp = int((alert & ~attack).sum())
fn = int((~alert & attack).sum())
tn = int((~alert & ~attack).sum())

print(f"Seuil (k={meta['default_k']}) = {threshold:.5f}")
print(f"Détection des attaques (rappel) : {tp}/{tp + fn} "
      f"= {100 * tp / (tp + fn):.1f} %")
for name in SCENARIOS:
    m = test["scenario"].to_numpy() == name
    print(f"   - {name:13s}: {int((alert & m).sum())}/{int(m.sum())}")
print(f"Fausses alertes (FPR)           : {fp}/{fp + tn} "
      f"= {100 * fp / (fp + tn):.2f} %")
print(f"Précision                       : "
      f"{100 * tp / max(tp + fp, 1):.1f} %")
print(f"Erreur min attaque = {errors[attack].min():.5f}  |  "
      f"erreur max normale = {errors[~attack].max():.5f}")
