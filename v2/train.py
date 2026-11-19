"""
EUBA v2 - Étape 1 : entraînement de l'autoencodeur.

  1. On garde uniquement le trafic légitime (RiskLabel != High).
  2. On apprend seulement sur la période passée (avant le 17/12/2024).
  3. Perte = MSE = moyenne de (x - x_hat)^2.
  4. Seuil dynamique = moyenne + k * écart-type des erreurs de validation.

Lancement :  python v2/train.py
"""
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from euba_model import (ARTIFACTS_DIR, NON_FEATURES, SPLIT_DATE,
                        EUBAAutoencoder, dynamic_threshold, load_raw,
                        reconstruction_errors)

SEED = 42
EPOCHS = 40
BATCH_SIZE = 256
DEFAULT_K = 3.0

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------
# 1. Données légitimes de la période d'entraînement
# ---------------------------------------------------------------
df = load_raw()
train_df = df[df["date"] < SPLIT_DATE]

feature_cols = [c for c in df.columns if c not in NON_FEATURES]
# Une colonne toujours égale à la même valeur n'apporte rien : on l'enlève
feature_cols = [c for c in feature_cols if train_df[c].nunique() > 1]

x_train, x_val = train_test_split(
    train_df[feature_cols], test_size=0.15, random_state=SEED)

scaler = MinMaxScaler().fit(x_train)          # appris sur le train seulement
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)

print(f"Lignes d'entraînement : {len(x_train):,}")
print(f"Lignes de validation  : {len(x_val):,}")
print(f"Variables             : {len(feature_cols)}")

# ---------------------------------------------------------------
# 2. Entraînement (non supervisé : aucun label utilisé)
# ---------------------------------------------------------------
model = EUBAAutoencoder(len(feature_cols))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

xt = torch.tensor(x_train, dtype=torch.float32)
xv = torch.tensor(x_val, dtype=torch.float32)
history = {"train_loss": [], "val_loss": []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    order = torch.randperm(len(xt))
    total = 0.0
    for i in range(0, len(xt), BATCH_SIZE):
        batch = xt[order[i:i + BATCH_SIZE]]
        optimizer.zero_grad()
        loss = loss_fn(model(batch), batch)    # l'entrée est aussi la cible
        loss.backward()
        optimizer.step()
        total += loss.item() * len(batch)

    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(xv), xv).item()
    history["train_loss"].append(total / len(xt))
    history["val_loss"].append(val_loss)
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:2d}  train={total / len(xt):.5f}  "
              f"val={val_loss:.5f}")

# ---------------------------------------------------------------
# 3. Seuil dynamique sur les erreurs de validation (normales)
# ---------------------------------------------------------------
val_errors = reconstruction_errors(model, x_val)
threshold = dynamic_threshold(val_errors, DEFAULT_K)
print(f"\nErreur normale : moyenne={val_errors.mean():.5f}  "
      f"écart-type={val_errors.std():.5f}")
print(f"Seuil (k={DEFAULT_K}) = {threshold:.5f}")

# ---------------------------------------------------------------
# 4. Sauvegarde
# ---------------------------------------------------------------
ARTIFACTS_DIR.mkdir(exist_ok=True)
torch.save(model.state_dict(), ARTIFACTS_DIR / "autoencoder.pth")
joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")
joblib.dump({
    "feature_cols": feature_cols,
    "input_dim": len(feature_cols),
    "val_errors": val_errors,
    "default_k": DEFAULT_K,
    "history": history,
    "split_date": SPLIT_DATE,
    "loss": "MSE = mean((x - x_hat)^2)",
    "threshold_rule": "mean + k * std des erreurs de validation normales",
}, ARTIFACTS_DIR / "metadata.joblib")
print(f"\nModèle sauvegardé dans {ARTIFACTS_DIR}")
