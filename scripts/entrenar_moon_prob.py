"""
Genera datasets y entrena los 2 modelos aprendidos de moon_prob
(src/entorno/moon_model.py) a partir de las manos reales reconstruibles de
data/partidas_bridge.jsonl. Ver spec:
docs/superpowers/specs/2026-07-06-moon-prob-modelo-aprendido-design.md

Limitación de datos conocida: la memoria del pase (cartas dadas/recibidas)
SOLO se conoce con certeza para el asiento realmente logueado por el bridge
en cada partida (`partida.asiento_agente`) -- el bridge no registra el
intercambio de los otros 3 asientos. Al generar ejemplos desde las 4
perspectivas por mano, esas features quedan en 0 (sin dato) salvo cuando la
perspectiva evaluada ES ese asiento real. No es un bug: es la limitación
real de los datos disponibles, y coincide con el caso legítimo de "sin
información de pase" que también ocurre en producción (el 4º jugador nunca
tiene relación de pase conmigo).

Uso:
    python scripts/entrenar_moon_prob.py --partidas data/partidas_bridge.jsonl \
        --out-dir models/moon --epocas 300
"""
from __future__ import annotations

# --- bootstrap path: permite `python scripts/<x>.py` desde la raiz del repo ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
# --- fin bootstrap ---

import argparse
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.captura.escritor import cargar_partidas
from src.captura.modelos import RegistroMano, RegistroPartida
from src.captura.replay import _preparar_motor, mano_reconstruible
from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones
from src.entorno.moon_model import (
    DIM_PROPIO,
    DIM_RIVAL,
    EntradaBaza,
    _RedMoonMLP,
    _alguien_mas_tiene_puntos,
    features_propio,
    features_rival,
)

_DIRECCION_A_NUMERO_MANO = {"izquierda": 1, "derecha": 2, "enfrente": 3}


def _receptor_y_dador(direccion: Optional[str], seat: int) -> Tuple[Optional[int], Optional[int]]:
    """Índice de asiento receptor/dador del pase de `seat`, o (None, None) sin pase."""
    if direccion is None or direccion not in _DIRECCION_A_NUMERO_MANO:
        return None, None
    m = MotorCorazones()
    m.numero_mano = _DIRECCION_A_NUMERO_MANO[direccion]
    receptor = m.receptor_pase(seat)
    dador = next(d for d in range(4) if m.receptor_pase(d) == seat)
    return receptor, dador


def _lunaseat_de(mano: RegistroMano) -> Optional[int]:
    """Asiento que hizo el pozo (Pleno) en esta mano, o None si no hubo pozo.

    `puntuacion_mano` ya viene con la regla de Pleno aplicada (ver
    `MotorCorazones.calcular_puntuacion_mano` / `manual.py:_aplicar_pleno`):
    el tirador queda en 0 y los otros 3 en 26 -> la suma es 78, NUNCA 26
    (una mano normal siempre suma 26, sin importar cómo se reparten los
    puntos, así que ese caso nunca garantiza que exista un asiento en 0).
    """
    if mano.puntuacion_mano and sum(mano.puntuacion_mano) == 78:
        return mano.puntuacion_mano.index(0)
    return None


def ejemplos_de_mano(mano: RegistroMano, asiento_agente_real: int):
    """(features, label) para el modelo propio y para el rival, por cada
    baza resuelta, desde las 4 perspectivas posibles."""
    motor = _preparar_motor(mano)
    luna_seat = _lunaseat_de(mano)
    receptor_agente, dador_agente = _receptor_y_dador(mano.direccion_pase, asiento_agente_real)

    historial: List[EntradaBaza] = []
    ejemplos_propio = []
    ejemplos_rival = []
    vacios: List[set] = [set() for _ in range(4)]
    mesa_actual: list = []

    for j in mano.jugadas:
        actual = motor.obtener_jugador_actual()
        if actual != j.asiento:
            break  # datos reales inconsistentes (ver pimc_regret_real.py), se descarta el resto
        carta = Carta._TODAS[j.carta_id]
        palo_salida = motor.palo_de_salida

        if not mesa_actual:
            puntos_mano_actual = [jg.contar_puntos_bazas() for jg in motor.jugadores]
            for seat in range(4):
                if _alguien_mas_tiene_puntos(motor, seat):
                    continue
                dadas = mano.pase_dado if seat == asiento_agente_real else []
                recibidas = mano.pase_recibido if seat == asiento_agente_real else []
                feats = features_propio(
                    motor, seat, vacios, historial, dadas, recibidas,
                    [0, 0, 0, 0], puntos_mano_actual, None,
                )
                ejemplos_propio.append((feats, 1.0 if seat == luna_seat else 0.0))

                for rival in range(4):
                    if rival == seat:
                        continue
                    if seat == asiento_agente_real:
                        dadas_r = mano.pase_dado if rival == receptor_agente else []
                        recibidas_r = mano.pase_recibido if rival == dador_agente else []
                    else:
                        dadas_r, recibidas_r = [], []
                    feats_r = features_rival(
                        motor, rival, seat, vacios, historial,
                        dadas_r, recibidas_r, motor.corazones_rotos,
                    )
                    ejemplos_rival.append((feats_r, 1.0 if rival == luna_seat else 0.0))

        if mesa_actual and palo_salida is not None and carta.palo != palo_salida:
            vacios[actual].add(palo_salida)
        motor.jugar_carta(actual, carta)
        mesa_actual.append((actual, carta))
        if len(mesa_actual) == 4:
            ganador = motor.resolver_baza()
            historial.append(EntradaBaza(
                lider=mesa_actual[0][0],
                ganador=ganador,
                tenia_puntos=any(c.puntos > 0 for _, c in mesa_actual),
                lidero_corazon_o_dama=(
                    mesa_actual[0][1].es_corazon or mesa_actual[0][1].es_dama_de_picas
                ),
            ))
            mesa_actual = []

    return ejemplos_propio, ejemplos_rival


def construir_dataset(ruta_partidas: str):
    """Devuelve 2 dicts partida_id -> [(features, label), ...] (uno por modelo)
    y un dict partida_id -> timestamp (para el corte cronológico de validación)."""
    partidas = cargar_partidas(ruta_partidas)
    propio_por_partida: Dict[str, list] = {}
    rival_por_partida: Dict[str, list] = {}
    timestamp_por_partida: Dict[str, str] = {}
    avisos: List[str] = []

    for p in partidas:
        ep: list = []
        er: list = []
        for mano in p.manos:
            if not mano_reconstruible(mano):
                continue
            try:
                e1, e2 = ejemplos_de_mano(mano, p.asiento_agente)
            except ValueError as e:
                avisos.append(f"{p.partida_id}: mano {mano.numero_mano} descartada ({e})")
                continue
            ep.extend(e1)
            er.extend(e2)
        if ep or er:
            propio_por_partida[p.partida_id] = ep
            rival_por_partida[p.partida_id] = er
            timestamp_por_partida[p.partida_id] = p.timestamp

    if avisos:
        print(f"{len(avisos)} manos descartadas por datos inconsistentes:")
        for a in avisos:
            print(f"  - {a}")

    return propio_por_partida, rival_por_partida, timestamp_por_partida


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC-ROC manual (evita sumar scikit-learn por una sola métrica):
    probabilidad de que un positivo al azar tenga score mayor que un
    negativo al azar."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gana = pos[:, None] > neg[None, :]
    empata = pos[:, None] == neg[None, :]
    return float(np.mean(gana) + 0.5 * np.mean(empata))


def _brier(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(np.mean((y_score - y_true) ** 2))


def _entrenar(red, X_train, y_train, X_val, y_val, epocas: int, lr: float = 1e-3,
              paciencia: int = 15):
    opt = torch.optim.Adam(red.parameters(), lr=lr)
    perdida = nn.BCELoss()
    Xt = torch.from_numpy(X_train)
    yt = torch.from_numpy(y_train)
    Xv = torch.from_numpy(X_val)

    mejor_auc = -1.0
    mejor_brier = float("inf")
    mejor_estado = {k: v.clone() for k, v in red.state_dict().items()}
    sin_mejora = 0

    for _ in range(epocas):
        red.train()
        opt.zero_grad()
        loss = perdida(red(Xt), yt)
        loss.backward()
        opt.step()

        red.eval()
        with torch.no_grad():
            pred_val = red(Xv).numpy()
        auc = _auc(y_val, pred_val)
        brier = _brier(y_val, pred_val)
        # ponytail: el AUC solo mide ranking, no calibración -- con datos bien
        # separados llega a 1.0 en la primera época y se queda ahí, así que
        # usamos el brier como desempate para seguir mejorando la calibración
        # mientras el AUC no empeore (si no, el early stopping se "congela"
        # en la primera época que toca el AUC máximo).
        mejora = not np.isnan(auc) and (
            auc > mejor_auc or (auc == mejor_auc and brier < mejor_brier)
        )
        if mejora:
            mejor_auc = auc
            mejor_brier = brier
            mejor_estado = {k: v.clone() for k, v in red.state_dict().items()}
            sin_mejora = 0
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break

    red.load_state_dict(mejor_estado)
    red.eval()
    with torch.no_grad():
        pred_final = red(Xv).numpy()
    return red, _auc(y_val, pred_final), _brier(y_val, pred_final)


def _apilar(ejemplos):
    """[(features, label), ...] -> (X, y) arrays float32."""
    X = np.stack([f for f, _ in ejemplos]).astype(np.float32)
    y = np.array([l for _, l in ejemplos], dtype=np.float32)
    return X, y


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--partidas", required=True)
    p.add_argument("--out-dir", default="models/moon")
    p.add_argument("--epocas", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.2)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    print("Generando ejemplos desde manos reales reconstruibles...", flush=True)
    propio_por_partida, rival_por_partida, timestamp_por_partida = construir_dataset(args.partidas)
    ids = sorted(propio_por_partida.keys())
    print(f"{len(ids)} partidas con al menos una mano reconstruible", flush=True)

    rng = np.random.default_rng(args.seed)
    orden = rng.permutation(len(ids))
    corte = int(len(ids) * (1 - args.val_frac))
    train_ids = {ids[i] for i in orden[:corte]}
    val_ids = {ids[i] for i in orden[corte:]}

    # Corte cronológico ADICIONAL (solo diagnóstico, no se usa para entrenar ni
    # para early stopping): las sesiones más recientes por timestamp, para
    # detectar sobreajuste a patrones de oponentes de esos días específicos en
    # vez de generalización real.
    ids_por_fecha = sorted(ids, key=lambda pid: timestamp_por_partida[pid])
    corte_fecha = int(len(ids_por_fecha) * (1 - args.val_frac))
    # Intersecar con val_ids: si no, la mayoría de las "recientes" ya estarían
    # en train (vistas en entrenamiento) y el diagnóstico de sobreajuste
    # mediría datos que el modelo ya conoce, no datos realmente no vistos.
    val_ids_recientes = set(ids_por_fecha[corte_fecha:]) & val_ids

    out = _Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for nombre, ejemplos_por_partida, dim in (
        ("propio", propio_por_partida, DIM_PROPIO),
        ("rival", rival_por_partida, DIM_RIVAL),
    ):
        train = [e for pid in train_ids for e in ejemplos_por_partida[pid]]
        val = [e for pid in val_ids for e in ejemplos_por_partida[pid]]
        if not train or not val:
            print(f"\n=== modelo {nombre}: datos insuficientes, se omite ===")
            continue
        pct_pos = 100 * sum(l for _, l in train) / len(train)
        print(f"\n=== modelo {nombre}: {len(train)} train / {len(val)} val "
              f"({pct_pos:.1f}% positivos train) ===", flush=True)

        X_train, y_train = _apilar(train)
        X_val, y_val = _apilar(val)

        red = _RedMoonMLP(dim)
        red, auc, brier = _entrenar(red, X_train, y_train, X_val, y_val, args.epocas)
        print(f"  AUC val: {auc:.3f}  |  Brier val: {brier:.4f}")

        recientes = [e for pid in val_ids_recientes for e in ejemplos_por_partida[pid]]
        if recientes:
            X_r, y_r = _apilar(recientes)
            with torch.no_grad():
                pred_r = red(torch.from_numpy(X_r)).numpy()
            print(f"  AUC en sesiones más recientes: {_auc(y_r, pred_r):.3f}  "
                  f"|  Brier: {_brier(y_r, pred_r):.4f}  ({len(recientes)} ejemplos)"
                  "  -- si es mucho peor que el AUC de val de arriba, hay sobreajuste "
                  "a patrones de oponentes específicos de esas sesiones.")

        ruta = out / f"{nombre}.pt"
        torch.save(red.state_dict(), ruta)
        print(f"  guardado en {ruta}")


if __name__ == "__main__":
    main()
