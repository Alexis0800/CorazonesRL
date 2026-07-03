"""
Encoder offline: re-juega una partida capturada en `MotorCorazones` y emite
pares `(obs, acción)` desde la perspectiva del agente, listos para BC/fine-tune.

Clave: cada asiento juega exactamente sus 13 cartas por mano, así que las manos
(post-pase) se RECONSTRUYEN desde las jugadas registradas — no hace falta haber
visto las manos rivales. La observación se construye con el MISMO
`ObservacionBuilder` del entorno (consistencia sim↔real).

Capa pura: sin ADB, sin visión. Solo `dominio` + `entorno`.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones
from src.entorno.dimensiones import DIM_ENTORNO
from src.entorno.observacion import ObservacionBuilder
from src.captura.modelos import RegistroMano, RegistroPartida

Ejemplo = Tuple[np.ndarray, int]  # (obs, carta_id elegida)


def reconstruir_manos(mano: RegistroMano) -> List[List[Carta]]:
    """Deriva las 4 manos (post-pase) a partir de las jugadas registradas.

    Si la mano terminó por concesión ("se llevará el resto"), completa cada mano
    con las `manos_restantes` reveladas, de modo que cada asiento sume 13 cartas.
    """
    manos: List[List[Carta]] = [[], [], [], []]
    for j in mano.jugadas:
        manos[j.asiento].append(Carta._TODAS[j.carta_id])
    if mano.manos_restantes:
        if len(mano.manos_restantes) != 4:
            raise ValueError(
                f"Mano {mano.numero_mano}: manos_restantes debe tener 4 listas."
            )
        for asiento, restantes in enumerate(mano.manos_restantes):
            manos[asiento].extend(Carta._TODAS[c] for c in restantes)
    for asiento, m in enumerate(manos):
        if len(m) != 13:
            raise ValueError(
                f"Mano {mano.numero_mano}: el asiento {asiento} tiene {len(m)} "
                f"cartas (esperaba 13). Jugadas/restantes incompletas o corruptas."
            )
    return manos


def mano_reconstruible(mano: RegistroMano) -> bool:
    """True si la mano puede re-jugarse (4 manos de 13 cartas coherentes)."""
    try:
        reconstruir_manos(mano)
        return True
    except Exception:
        return False


def _preparar_motor(mano: RegistroMano) -> MotorCorazones:
    motor = MotorCorazones()
    motor.numero_mano = mano.numero_mano
    for i, cartas in enumerate(reconstruir_manos(mano)):
        motor.jugadores[i].recibir_mano(cartas)
        motor.jugadores[i].bazas_ganadas = []
    motor.corazones_rotos = False
    motor.numero_baza = 1
    motor.mesa = []
    motor.palo_de_salida = None
    motor._mano_activa = True
    motor._fijar_jugador_inicial()
    return motor


def ejemplos_de_mano(
    mano: RegistroMano, asiento_agente: int, obs_builder: ObservacionBuilder,
) -> List[Ejemplo]:
    """Re-juega una mano y devuelve `(obs, carta_id)` por cada turno del agente."""
    motor = _preparar_motor(mano)
    ejemplos: List[Ejemplo] = []
    for j in mano.jugadas:
        actual = motor.obtener_jugador_actual()
        if actual != j.asiento:
            raise ValueError(
                f"Mano {mano.numero_mano}, baza {j.baza}: orden inconsistente "
                f"(motor espera asiento {actual}, registro dice {j.asiento})."
            )
        carta = Carta._TODAS[j.carta_id]
        if actual == asiento_agente:
            obs = obs_builder.construir_desde_motor(motor, asiento_agente)
            ejemplos.append((obs, carta.id))
        motor.jugar_carta(actual, carta)
        if len(motor.mesa) == 4:
            motor.resolver_baza()
    return ejemplos


def ejemplos_de_partida(
    partida: RegistroPartida, obs_builder: ObservacionBuilder | None = None,
    dim: int = DIM_ENTORNO,
) -> List[Ejemplo]:
    """Concatena los ejemplos de todas las manos reconstruibles de una partida.

    Manos no reconstruibles (p.ej. una concesión sin `manos_restantes`, como las
    que produce el puente SFS2X para un "remate" que la app resuelve sola — ver
    `scripts/importar_sesiones_bridge.py`) se saltan en vez de abortar TODA la
    partida: dentro de una partida real es normal mezclar manos completas con
    manos truncadas, y una sola mano mala no debe tirar las demás.
    """
    builder = obs_builder or ObservacionBuilder(dim=dim)
    out: List[Ejemplo] = []
    for mano in partida.manos:
        if not mano_reconstruible(mano):
            continue
        out.extend(ejemplos_de_mano(mano, partida.asiento_agente, builder))
    return out


def partidas_a_arrays(
    partidas, dim: int = DIM_ENTORNO,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convierte un iterable de `RegistroPartida` en `(X[n,dim], y[n])`."""
    builder = ObservacionBuilder(dim=dim)
    obs_list: List[np.ndarray] = []
    acc_list: List[int] = []
    for p in partidas:
        for obs, accion in ejemplos_de_partida(p, builder, dim):
            obs_list.append(obs)
            acc_list.append(accion)
    if not obs_list:
        return np.zeros((0, dim), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.asarray(obs_list, dtype=np.float32), np.asarray(acc_list, dtype=np.int64)


__all__ = [
    "Ejemplo", "reconstruir_manos", "mano_reconstruible", "ejemplos_de_mano",
    "ejemplos_de_partida", "partidas_a_arrays",
]
