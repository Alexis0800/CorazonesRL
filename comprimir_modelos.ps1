# comprimir_modelos.ps1
# Comprime los modelos necesarios para transferir entre PCs.
# Solo incluye los últimos snapshots (no todos) para reducir tamaño.
#
# Uso:
#   .\comprimir_modelos.ps1
#
# Output:
#   corazones_modelos.zip (~60-80 MB)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$destino = Join-Path $root "corazones_modelos.zip"
$tempDir = Join-Path $env:TEMP "corazones_modelos_temp"

# Limpiar temp anterior
if (Test-Path $tempDir) { Remove-Item $tempDir -Recurse -Force }
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

Write-Host "📦 Comprimiendo modelos para transferir..." -ForegroundColor Cyan

# --- v5: últimos 10 snapshots + vecnorm ---
$v5_snaps = Join-Path $tempDir "modelos_historicos\v5"
New-Item -ItemType Directory -Path $v5_snaps -Force | Out-Null

Write-Host "  v5: últimos 10 snapshots..."
Get-ChildItem "$root\modelos_historicos\v5\snapshot_*.zip" `
    | Sort-Object Name `
    | Select-Object -Last 10 `
    | Copy-Item -Destination $v5_snaps

# v5 vecnorm files
Get-ChildItem "$root\modelos_historicos\v5\*vecnorm*" `
    | Sort-Object Name `
    | Select-Object -Last 10 `
    | Copy-Item -Destination $v5_snaps -ErrorAction SilentlyContinue

# eval_log
Copy-Item "$root\modelos_historicos\v5\eval_log.jsonl" $v5_snaps -ErrorAction SilentlyContinue

# --- v6: últimos 5 snapshots (solo referencia) ---
$v6_snaps = Join-Path $tempDir "modelos_historicos\v6"
New-Item -ItemType Directory -Path $v6_snaps -Force | Out-Null

Write-Host "  v6: últimos 5 snapshots (referencia)..."
Get-ChildItem "$root\modelos_historicos\v6\snapshot_*.zip" `
    | Sort-Object Name `
    | Select-Object -Last 5 `
    | Copy-Item -Destination $v6_snaps -ErrorAction SilentlyContinue

# --- VecNormalize completo ---
Write-Host "  VecNormalize..."
$vecnorm_dst = Join-Path $tempDir "vecnormalize"
Copy-Item "$root\vecnormalize" $vecnorm_dst -Recurse -Force -ErrorAction SilentlyContinue

# --- Comprimir ---
Write-Host "  Comprimiendo..." -NoNewline
Compress-Archive -Path "$tempDir\*" -DestinationPath $destino -Force
Write-Host " OK"

# --- Tamaño ---
$size = (Get-Item $destino).Length / 1MB
Write-Host ""
Write-Host "✅ Listo: $destino" -ForegroundColor Green
Write-Host "   Tamaño: $([math]::Round($size, 1)) MB" -ForegroundColor Green
Write-Host ""
Write-Host "Para usar en la otra PC:" -ForegroundColor Yellow
Write-Host "  1. Copiar corazones_modelos.zip a la raíz del proyecto"
Write-Host "  2. Expand-Archive corazones_modelos.zip -DestinationPath . -Force"
Write-Host "  3. python train_self_play.py --self-play --steps 5000000 --eval-every 5"

# Limpiar temp
Remove-Item $tempDir -Recurse -Force -ErrorAction SilentlyContinue
