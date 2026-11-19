# =====================================================================
#  EUBA v2 - Lancement complet en UNE commande (Windows PowerShell)
#
#    powershell -ExecutionPolicy Bypass -File run.ps1            # tableau de bord
#    powershell -ExecutionPolicy Bypass -File run.ps1 -Kafka     # + streaming Kafka
#
#  Options :
#    -Kafka         démarre aussi Kafka (Docker), le détecteur et le producteur
#    -Delay 0.005   vitesse du flux Kafka (secondes entre deux messages)
#    -Retrain       réentraîne le modèle même s'il existe déjà
#    -Port 8501     port du tableau de bord
# =====================================================================
param(
    [switch]$Kafka,
    [double]$Delay = 0.02,
    [switch]$Retrain,
    [int]$Port = 8501
)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"

function Step($n, $msg) { Write-Host "[$n/5] $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "ERREUR : $msg" -ForegroundColor Red; exit 1 }

# --- 1. Environnement Python -------------------------------------------
Step 1 "Environnement Python (.venv)"
if ($root.Length -gt 70) {
    Write-Host "      ATTENTION : le chemin du projet est long ($($root.Length) caractères)." -ForegroundColor Yellow
    Write-Host "      torch contient des fichiers très profonds : si l'installation échoue avec" -ForegroundColor Yellow
    Write-Host "      'No such file or directory', clonez le dépôt dans un dossier court (ex. C:\EUBA)." -ForegroundColor Yellow
}
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "      création de .venv..."
    if (Get-Command py -ErrorAction SilentlyContinue) { & py -3 -m venv .venv }
    elseif (Get-Command python -ErrorAction SilentlyContinue) { & python -m venv .venv }
    else { Fail "Python 3 est introuvable. Installez-le depuis https://www.python.org" }
    if (-not (Test-Path $py)) { Fail "impossible de créer .venv" }
}

# --- 2. Dépendances (réinstallées seulement si requirements.txt change) --
Step 2 "Dépendances"
$req = Join-Path $root "v2\requirements.txt"
$stamp = Join-Path $root ".venv\.euba-requirements"
$hash = (Get-FileHash $req).Hash
if (-not (Test-Path $stamp) -or (Get-Content $stamp) -ne $hash) {
    Write-Host "      installation (la première fois, torch prend quelques minutes)..."
    & $py -m pip install --disable-pip-version-check -q -r $req
    if ($LASTEXITCODE -ne 0) { Fail "pip install a échoué" }
    Set-Content $stamp $hash
} else { Write-Host "      déjà à jour" }

# --- 3. Modèle ------------------------------------------------------------
Step 3 "Modèle (autoencodeur)"
if ($Retrain -or -not (Test-Path "v2\artifacts\autoencoder.pth")) {
    & $py -W ignore v2\train.py
    if ($LASTEXITCODE -ne 0) { Fail "l'entraînement a échoué" }
} else { Write-Host "      modèle déjà entraîné (v2\artifacts)" }

# --- 4. Jeu de test avec attaques simulées --------------------------------
Step 4 "Jeu de test avec attaques simulées"
if ($Retrain -or -not (Test-Path "v2\euba_v2_test_with_attacks.csv")) {
    & $py -W ignore v2\simulate_attacks.py
    if ($LASTEXITCODE -ne 0) { Fail "la simulation des attaques a échoué" }
} else { Write-Host "      déjà généré" }

# --- 5. Tableau de bord (+ Kafka) -----------------------------------------
Step 5 "Tableau de bord Streamlit sur http://localhost:$Port"
Start-Process powershell -WorkingDirectory $root -ArgumentList "-NoExit", "-Command",
    "`$host.UI.RawUI.WindowTitle='EUBA - Tableau de bord'; & '$py' -m streamlit run v2\app.py --server.port $Port"

$ok = $false
for ($i = 0; $i -lt 40; $i++) {
    try { Invoke-WebRequest "http://localhost:$Port" -UseBasicParsing -TimeoutSec 2 | Out-Null; $ok = $true; break }
    catch { Start-Sleep -Seconds 1 }
}
if (-not $ok) { Fail "le tableau de bord ne répond pas (voir la fenêtre 'EUBA - Tableau de bord')" }
Write-Host "      tableau de bord prêt" -ForegroundColor Green

if ($Kafka) {
    Write-Host ""
    Write-Host "Streaming Kafka :" -ForegroundColor Cyan
    & (Join-Path $root "v2\demo_kafka.ps1") -Delay $Delay
}

Write-Host ""
Write-Host "Tout est lancé. Fermez les fenêtres 'EUBA - ...' pour arrêter." -ForegroundColor Green
if ($Kafka) { Write-Host "Après la démo, arrêtez Kafka : docker compose -f v2/docker-compose.yml down" }
