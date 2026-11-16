# EUBA v2 - Démo streaming Kafka en une seule commande.
#   powershell -ExecutionPolicy Bypass -File v2\demo_kafka.ps1
# Ouvre 2 fenêtres : le détecteur, puis le producteur.
# Option : -Delay 0.005 pour une démo plus rapide.
param([double]$Delay = 0.02)

$root = Split-Path $PSScriptRoot -Parent
$py   = Join-Path $root ".venv\Scripts\python.exe"
Set-Location $root

Write-Host "1/4  Vérification de Docker..." -ForegroundColor Cyan
docker info --format '{{.ServerVersion}}' *> $null
if ($LASTEXITCODE -ne 0) { Write-Host "Docker Desktop n'est pas démarré. Ouvrez-le puis relancez." -ForegroundColor Red; exit 1 }

Write-Host "2/4  Démarrage de Kafka..." -ForegroundColor Cyan
docker compose -f v2/docker-compose.yml up -d
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    docker exec euba-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list *> $null
    if ($LASTEXITCODE -eq 0) { $ready = $true; break }
    Start-Sleep -Seconds 2
}
if (-not $ready) { Write-Host "Kafka ne répond pas." -ForegroundColor Red; exit 1 }
foreach ($t in "euba-events", "euba-alerts") {
    docker exec euba-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --if-not-exists --topic $t *> $null
}

Write-Host "3/4  Lancement du détecteur (nouvelle fenêtre)..." -ForegroundColor Cyan
$env:PYTHONIOENCODING = "utf-8"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle='EUBA - Détecteur'; & '$py' -W ignore v2\kafka_detector.py --reset" -WorkingDirectory $root
Start-Sleep -Seconds 6

Write-Host "4/4  Lancement du producteur (nouvelle fenêtre)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle='EUBA - Producteur'; & '$py' -W ignore v2\kafka_producer.py --delay $Delay" -WorkingDirectory $root

Write-Host "`nC'est parti ! Ouvrez l'onglet « Flux Kafka » du tableau de bord." -ForegroundColor Green
Write-Host "Les premières alertes arrivent à la journée 2024-12-21."
Write-Host "Pour arrêter Kafka APRÈS la démo : docker compose -f v2/docker-compose.yml down"
