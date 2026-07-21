"""
Genera un dataset BC dirigido a partir de decisiones REALES ya etiquetadas por
el oráculo PIMC de `pimc_regret_real.py --volcar-json` (evita recorrer rollouts
otra vez). La observación se construye EXACTAMENTE como en producción
(`Recomendador._obs`, que ya envuelve `EstimadorMoonProb` + `ObservacionBuilder`).

Dos modos (`--filtro`):
  - pozo:  SOLO decisiones de manos con pozo (sum(puntuacion_mano) == 78).
  - todos: TODAS las decisiones con etiqueta disponible -- cada ejemplo se
           marca con `es_lider` (guardado en el .npz) para poder sobre-pesar
           las decisiones de liderazgo en el entrenamiento (ver
           finetune_bc_pozo.py --peso-lider) SIN excluir el resto. Analisis
           2026-07-06: liderar tiene 22x más regret alto en manos reales que
           en estados simulados (vs bots) -- reaccionar también tiene brecha
           (9x) aunque menor, así que filtrar a SOLO liderazgo repite el
           mismo error de sobre-especialización que ya vimos con --filtro pozo.

Uso:
    python scripts/generar_dataset_pozo.py \
        --partidas data/partidas_bridge_train.jsonl \
        --volcado /tmp/moon_analysis/campeon.jsonl \
        --modelo-para-obs models/v10c_finetune_pozo/snapshots/snapshot_000025001984 \
        --filtro todos \
        --out datasets/bc_correccion_train.npz
"""
from __future__ import annotations

# --- bootstrap path: permite `python scripts/<x>.py` desde la raiz del repo ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
# --- fin bootstrap ---

import argparse
import json
from typing import Dict, List, Tuple

import numpy as np

from src.captura.escritor import cargar_partidas
from src.captura.modelos import str_a_carta_id
from src.captura.replay import mano_reconstruible, reconstruir_manos, _preparar_motor
from src.dominio.carta import Carta
from src.dominio.motor import hubo_pozo
from scripts.recomendador import Recomendador


def cargar_etiquetas(ruta_volcado: str, partidas, filtro: str = "todos") -> Dict[Tuple[str, int, int], int]:
    """(partida_id, mano, baza) -> id de la mejor carta según el oráculo.

    `filtro="pozo"` restringe a manos con pozo (sum(puntuacion_mano) == 78,
    ver regla de Pleno en motor.py); `filtro="todos"` no filtra por mano.
    """
    es_luna = {}
    for p in partidas:
        for m in p.manos:
            es_luna[(p.partida_id, m.numero_mano)] = bool(
                m.puntuacion_mano and hubo_pozo(m.puntuacion_mano))

    etiquetas = {}
    with open(ruta_volcado, encoding="utf-8") as f:
        for linea in f:
            r = json.loads(linea)
            if filtro == "pozo" and not es_luna.get((r["partida_id"], r["mano"]), False):
                continue
            mejor = min(r["scores"], key=r["scores"].get)
            etiquetas[(r["partida_id"], r["mano"], r["baza"])] = str_a_carta_id(mejor)
    return etiquetas


def generar(partidas, etiquetas: Dict[Tuple[str, int, int], int], ckpt_para_obs: str):
    rec = Recomendador(ckpt_para_obs)
    obs_l: List[np.ndarray] = []
    mask_l: List[np.ndarray] = []
    action_l: List[int] = []
    lider_l: List[bool] = []

    for p in partidas:
        rec.scores = [0, 0, 0, 0]
        for mano in p.manos:
            tiene_etiqueta = any(
                (p.partida_id, mano.numero_mano, b) in etiquetas for b in range(1, 14))
            if not mano_reconstruible(mano) or not tiene_etiqueta:
                if mano.puntuacion_mano:
                    rec.scores = [rec.scores[i] + mano.puntuacion_mano[i] for i in range(4)]
                rec.reset_mano([])
                continue

            manos_reales = reconstruir_manos(mano)
            motor = _preparar_motor(mano)
            rec.reset_mano(list(manos_reales[p.asiento_agente]))
            mesa_actual: list = []
            for j in mano.jugadas:
                actual = motor.obtener_jugador_actual()
                if actual != j.asiento:
                    break  # datos reales inconsistentes a medio camino: se descarta el resto
                carta = Carta._TODAS[j.carta_id]

                if actual == p.asiento_agente:
                    legales = motor.obtener_jugadas_legales(p.asiento_agente)
                    clave = (p.partida_id, mano.numero_mano, j.baza)
                    if len(legales) > 1 and clave in etiquetas:
                        m = rec._motor(mesa=mesa_actual)
                        obs = rec._obs(m)
                        mask = np.zeros(52, dtype=np.float32)
                        for c in legales:
                            mask[c.id] = 1.0
                        obs_l.append(obs)
                        mask_l.append(mask)
                        action_l.append(etiquetas[clave])
                        lider_l.append(len(mesa_actual) == 0)

                motor.jugar_carta(actual, carta)
                mesa_actual.append((actual, carta))
                if len(mesa_actual) == 4:
                    ganador = motor.resolver_baza()
                    rec.registrar_baza(mesa_actual, ganador)
                    mesa_actual = []

    return (np.array(obs_l, dtype=np.float32),
            np.array(mask_l, dtype=np.float32),
            np.array(action_l, dtype=np.int64),
            np.array(lider_l, dtype=np.bool_))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--partidas", required=True)
    p.add_argument("--volcado", required=True, help="jsonl de pimc_regret_real.py --volcar-json")
    p.add_argument("--modelo-para-obs", required=True,
                   help="Checkpoint usado solo para construir la observación (obs_dim/con_pase)")
    p.add_argument("--filtro", choices=["pozo", "todos"], default="todos")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    partidas = cargar_partidas(args.partidas)
    etiquetas = cargar_etiquetas(args.volcado, partidas, filtro=args.filtro)
    print(f"{len(etiquetas)} decisiones etiquetadas por el oráculo (filtro={args.filtro})")

    obs, mask, action, es_lider = generar(partidas, etiquetas, args.modelo_para_obs)
    n_lider = int(es_lider.sum()) if len(es_lider) else 0
    print(f"Dataset generado: {len(action)} ejemplos, obs_dim={obs.shape[1] if len(obs) else 0}, "
          f"{n_lider} liderando ({100*n_lider/len(action):.1f}%)" if len(action) else "Dataset vacío")

    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, obs=obs, mask=mask, action=action, es_lider=es_lider)
    print(f"Guardado en {args.out}")


if __name__ == "__main__":
    main()
