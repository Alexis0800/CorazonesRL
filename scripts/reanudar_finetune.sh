#!/usr/bin/env bash
# Auto-reanudación del entrenamiento v11 "desde cero humanizado": init BC-PIMC
# (no el campeón convergido — el fine-tune quedó refutado 2 veces, ver
# docs/auditoria_moon_2026-07-20.md) + pool con 50% mesas humanas + curriculum
# completo desde progress 0 + rail de elite por win_rate_vs_humano.
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/.."
n=0
while [ $n -lt 60 ]; do
  if ls models/v11_humano_scratch/snapshots/snapshot_* >/dev/null 2>&1; then
    EXTRA="--resume auto"
  else
    EXTRA="--bc-weights models/produccion/bc_base_con_pase.pkl"
  fi
  echo "=== lanzamiento $n ($EXTRA) $(date) ===" >> data/train_v11.log
  .venv/Scripts/python.exe scripts/train_rllib.py --total-steps 30000000 \
    --workers 10 --gpus 0 --obs-dim 228 --con-pase \
    --humano-bc models/humano_bc/pesos.npz --prob-humano 0.5 \
    --mesa-humana 0.5 --temp-humano 1.0 \
    --lr 1e-4 --lr-end 5e-5 --entropy-coeff 0.03 \
    --moon-dir models/moon_realfull --pool-diverso --ancla-experto \
    $EXTRA --output-dir models/v11_humano_scratch --snapshot-interval 200000 \
    >> data/train_v11.log 2>&1
  code=$?
  echo "=== exit $code (intento $n) ===" >> data/train_v11.log
  [ $code -eq 0 ] && break
  n=$((n+1)); sleep 12
done
echo "=== loop terminado (intentos usados: $n) ===" >> data/train_v11.log
