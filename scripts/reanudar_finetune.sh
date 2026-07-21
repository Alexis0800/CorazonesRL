#!/usr/bin/env bash
# Auto-reanudación del fine-tune v10e (2M pasos, mesas mayoritariamente humanas).
# Ray en Windows lanza access violations intermitentes (exit 139); este loop
# reanuda desde el último snapshot hasta completar --total-steps o agotar reintentos.
# Primera pasada: resume del campeón; siguientes: resume auto del propio run.
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/.."
n=0
while [ $n -lt 40 ]; do
  if ls models/v10e_humano2M/snapshots/snapshot_* >/dev/null 2>&1; then
    RESUME=auto
  else
    RESUME=models/produccion/v10c_campeon
  fi
  echo "=== lanzamiento $n (resume=$RESUME) $(date) ===" >> data/train_v10e.log
  .venv/Scripts/python.exe scripts/train_rllib.py --total-steps 27000000 \
    --workers 10 --gpus 0 --obs-dim 228 --con-pase \
    --humano-bc models/humano_bc/pesos.npz --prob-humano 0.5 \
    --mesa-humana 0.7 --temp-humano 1.0 --progress-fino \
    --lr 5e-5 --lr-end 5e-5 --entropy-coeff 0.01 \
    --moon-dir models/moon_realfull --pool-diverso --ancla-experto \
    --resume "$RESUME" --output-dir models/v10e_humano2M --snapshot-interval 100000 \
    >> data/train_v10e.log 2>&1
  code=$?
  echo "=== exit $code (intento $n) ===" >> data/train_v10e.log
  [ $code -eq 0 ] && break
  n=$((n+1)); sleep 12
done
echo "=== loop terminado (intentos usados: $n) ===" >> data/train_v10e.log
