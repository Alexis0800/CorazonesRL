#!/usr/bin/env python
"""
Comparador rápido: modelos RL vs BotExperto.

Juega N partidas de cada modelo contra 3 copias de BotExperto,
reportando posición final, puntuación media y win rate.

Uso:
    python scripts/comparar_bots.py --partidas 100
    python scripts/comparar_bots.py --modelos v5_golden,v7_golden,v8 --partidas 200
"""

from __future__ import annotations
from src.agentes.bot_experto import BotExperto
from src.agentes.heuristicos import bot_conservador, bot_agresivo, bot_evasivo
from src.entorno.single_agent import CorazonesEnv

import argparse
import os
import pickle
import sys
from typing import Dict, List, Optional

import numpy as np

# Asegurar que src/ está en el path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ──────────────────────────────────────────────────────────────
# Modelos disponibles
# ──────────────────────────────────────────────────────────────

MODELOS = {
    "v5_golden": {
        "modelo": "models/v5_golden/snapshots/snapshot_0010000000.zip",
        "vecnorm": "models/v5_golden/snapshots/snapshot_0010000000_vecnorm.pkl",
        "desc": "v5_golden 10M (~1500 Elo)",
    },
    "v7_golden": {
        "modelo": "models/v7_golden/snapshots/snapshot_0014900000.zip",
        "vecnorm": "models/v7_golden/snapshots/snapshot_0014900000_vecnorm.pkl",
        "desc": "v7_golden 14.9M (1723 Elo)",
    },
    "v8": {
        "modelo": "models/v8/elite/snapshot_0017900000.zip",
        "vecnorm": None,  # Se buscará automáticamente
        "desc": "v8 elite 1.79M",
    },
}


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def normalizar_obs(obs: np.ndarray, vecnorm_path: Optional[str]) -> np.ndarray:
    """Normaliza observación con VecNormalize stats."""
    if not vecnorm_path or not os.path.exists(vecnorm_path):
        return obs
    try:
        with open(vecnorm_path, "rb") as f:
            vn = pickle.load(f)
        obs_rms = vn.obs_rms
        if obs_rms is None or obs_rms.count < 1:
            return obs
        mean = np.array(obs_rms.mean)
        var = np.array(obs_rms.var)
        return np.clip(
            (obs - mean) / np.sqrt(var + 1e-8), -10.0, 10.0
        ).astype(np.float32)
    except Exception:
        return obs


def _auto_vecnorm(modelo_path: str) -> Optional[str]:
    """Busca el VecNormalize asociado al modelo."""
    base = modelo_path.replace(".zip", "")
    candidato = base + "_vecnorm.pkl"
    if os.path.exists(candidato):
        return candidato

    # Buscar en vecnorm/ del mismo directorio
    vecnorm_dir = os.path.join(os.path.dirname(
        os.path.dirname(modelo_path)), "vecnorm")
    nombre = os.path.basename(base)
    candidato2 = os.path.join(vecnorm_dir, nombre + "_vecnorm.pkl")
    if os.path.exists(candidato2):
        return candidato2
    return None


def evaluar_vs_experto(
    modelo_path: str,
    vecnorm_path: Optional[str],
    num_partidas: int,
    etiqueta: str,
) -> Dict:
    """Evalúa un modelo contra 3 copias independientes de BotExperto.

    El modelo juega como jugador 0.
    Jugadores 1, 2 y 3 son instancias separadas de BotExperto.
    """
    from sb3_contrib import MaskablePPO

    if not os.path.exists(modelo_path):
        print(f"  [ERROR] No se encuentra {modelo_path}")
        return {"error": f"Archivo no encontrado: {modelo_path}", "total": 0}

    print(f"  Cargando {modelo_path}...")
    try:
        modelo = MaskablePPO.load(modelo_path, device="cpu")
    except Exception as e:
        print(f"  [ERROR] al cargar modelo: {e}")
        return {"error": str(e), "total": 0}

    # VecNorm auto-detección
    if vecnorm_path is None:
        vecnorm_path = _auto_vecnorm(modelo_path)
    if vecnorm_path:
        print(f"  VecNorm: {vecnorm_path}")

    # Crear una instancia de BotExperto por entorno (mantiene estado por mano)
    posiciones: List[int] = []
    puntuaciones: List[float] = []
    puntos_experto: List[float] = []

    for seed in range(num_partidas):
        # 3 copias independientes de BotExperto (cada una con su propio estado)
        bot1, bot2, bot3 = BotExperto(), BotExperto(), BotExperto()
        politicas = {
            1: bot1,
            2: bot2,
            3: bot3,
        }
        env = CorazonesEnv(agente_idx=0, politicas_oponentes=politicas)
        obs_raw, _ = env.reset(seed=seed)
        obs = normalizar_obs(obs_raw, vecnorm_path)
        done = False

        while not done:
            mask = env.action_masks()
            action, _ = modelo.predict(
                obs, action_masks=mask, deterministic=True)
            obs_raw, _reward, terminated, truncated, _ = env.step(int(action))
            obs = normalizar_obs(obs_raw, vecnorm_path)
            done = terminated or truncated

        punt_agente = env._puntuacion_historica[0]
        punt_rivales = [env._puntuacion_historica[i] for i in range(1, 4)]
        punt_experto = env._puntuacion_historica[1]

        todas = [punt_agente] + list(punt_rivales)
        ranking = sorted(range(4), key=lambda i: todas[i])
        posicion = ranking.index(0)

        posiciones.append(posicion)
        puntuaciones.append(float(punt_agente))
        puntos_experto.append(float(punt_experto))

        env.close()

        if (seed + 1) % 50 == 0:
            v = sum(1 for p in posiciones if p == 0)
            print(f"    {seed + 1}/{num_partidas} — {v} victorias")

    total = num_partidas
    return {
        "modelo": etiqueta,
        "total": total,
        "pct_1": sum(1 for p in posiciones if p == 0) / total,
        "pct_2": sum(1 for p in posiciones if p == 1) / total,
        "pct_3": sum(1 for p in posiciones if p == 2) / total,
        "pct_4": sum(1 for p in posiciones if p == 3) / total,
        "top2": sum(1 for p in posiciones if p in (0, 1)) / total,
        "punt_media": float(np.mean(puntuaciones)),
        "punt_mediana": float(np.median(puntuaciones)),
        "punt_min": float(np.min(puntuaciones)),
        "punt_max": float(np.max(puntuaciones)),
        "vs_experto_media": float(np.mean(puntos_experto)),
    }


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Comparar modelos RL vs BotExperto")
    parser.add_argument(
        "--modelos", type=str, default="v5_golden,v7_golden,v8",
        help="Modelos a evaluar (separados por coma): v5_golden,v7_golden,v8"
    )
    parser.add_argument(
        "--partidas", type=int, default=100,
        help="Partidas por modelo (default: 100)"
    )
    args = parser.parse_args()

    nombres = [n.strip() for n in args.modelos.split(",")]

    print("=" * 72)
    print("  COMPARADORA: Modelos RL vs BotExperto")
    print(f"  Partidas por modelo: {args.partidas}")
    print(f"  Oponentes: 3× BotExperto (copias independientes)")
    print("=" * 72)

    resultados = []
    for nombre in nombres:
        if nombre not in MODELOS:
            print(
                f"\n[!] Modelo desconocido: {nombre}. Disponibles: {list(MODELOS.keys())}")
            continue

        info = MODELOS[nombre]
        print(f"\n--- {info['desc']} ---")
        res = evaluar_vs_experto(
            info["modelo"],
            info.get("vecnorm"),
            args.partidas,
            nombre,
        )
        resultados.append(res)

    # Tabla resumen
    print("\n" + "=" * 72)
    print("  RESULTADOS")
    print("=" * 72)
    print(f"{'Modelo':<16} {'1º':>6} {'2º':>6} {'3º':>6} {'4º':>6} {'Top2':>6} {'Punt Media':>11} {'vsExperto':>10}")
    print("-" * 72)

    for r in resultados:
        if "error" in r:
            print(f"{r.get('modelo', '?'):<16} [ERROR] {r['error']}")
            continue
        print(
            f"{r['modelo']:<16} "
            f"{r['pct_1']:>5.0%} "
            f"{r['pct_2']:>5.0%} "
            f"{r['pct_3']:>5.0%} "
            f"{r['pct_4']:>5.0%} "
            f"{r['top2']:>5.0%} "
            f"{r['punt_media']:>10.1f} "
            f"{r['vs_experto_media']:>9.1f}"
        )

    print("-" * 72)
    print("Nota: 1º = ganó la partida (menos puntos). vsExperto = puntuación media del BotExperto.")
    print()


if __name__ == "__main__":
    main()
