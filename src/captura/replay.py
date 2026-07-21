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
from src.entorno.moon_model import RUTA_MOON, EntradaBaza, EstimadorMoonProb
from src.entorno.observacion import ObservacionBuilder, puede_alimentar
from src.captura.modelos import RegistroMano, RegistroPartida

Ejemplo = Tuple[np.ndarray, int]  # (obs, carta_id elegida)


class _TrackerTactico:
    """Replica el bookkeeping táctico de `CorazonesEnvRLlib` durante el replay de
    una mano, para alimentar `ObservacionBuilder.construir` con el MISMO estado
    que el env construye en entrenamiento (voids, puntos de la mano, Dama de
    Picas, historial de bazas). Sin esto, la obs de datos humanos quedaba en la
    versión mínima (features estratégicas en cero) y NO alineada con el modelo.

    Espejo exacto de `env._actualizar_vacios` y `env._resolver_baza`.
    """

    def __init__(self) -> None:
        self.vacios: List[set] = [set() for _ in range(4)]
        self.puntos_mano_actual: List[int] = [0, 0, 0, 0]
        self.dama_picas_en = None
        self.historial: List[EntradaBaza] = []

    def actualizar_vacios(self, motor: MotorCorazones, jugador_idx: int, carta: Carta) -> None:
        if motor.mesa and motor.palo_de_salida is not None:
            if carta.palo != motor.palo_de_salida:
                self.vacios[jugador_idx].add(motor.palo_de_salida)

    def resolver_baza(self, motor: MotorCorazones) -> int:
        lider_idx, carta_lider = motor.mesa[0]  # capturar ANTES: resolver_baza() vacía mesa
        cartas_en_mesa = [c for _, c in motor.mesa]
        tenia_puntos = any(c.puntos > 0 for c in cartas_en_mesa)
        ganador = motor.resolver_baza()
        if any(c.es_dama_de_picas for c in cartas_en_mesa):
            self.dama_picas_en = ganador
        self.historial.append(EntradaBaza(
            lider=lider_idx, ganador=ganador, tenia_puntos=tenia_puntos,
            lidero_corazon_o_dama=carta_lider.es_corazon or carta_lider.es_dama_de_picas,
        ))
        for i, jug in enumerate(motor.jugadores):
            self.puntos_mano_actual[i] = jug.contar_puntos_bazas()
        return ganador


def _obs_para_asiento(
    builder: ObservacionBuilder, motor: MotorCorazones, asiento: int,
    tracker: _TrackerTactico, puntuacion_historica: List[int],
    estimador: EstimadorMoonProb, mano: RegistroMano, asiento_agente: int,
) -> np.ndarray:
    """Obs completa desde la perspectiva de `asiento`, con el MISMO cableado que
    `env._build_obs` (misma llamada a `construir`, mismos moon_prob del estimador).

    La memoria del pase solo se conoce para `asiento_agente` (los rivales humanos
    del puente SFS nunca revelan su pase); para los demás asientos va vacía, igual
    que el env cuando el rival no es el agente.
    """
    if asiento == asiento_agente:
        dadas = list(mano.pase_dado or [])
        recibidas = list(mano.pase_recibido or [])
    else:
        dadas, recibidas = [], []

    moon_prob_agente = estimador.propio(
        motor=motor, agente_idx=asiento, vacios=tracker.vacios,
        historial=tracker.historial, cartas_dadas=dadas, cartas_recibidas=recibidas,
        puntuacion_historica=puntuacion_historica,
        puntos_mano_actual=tracker.puntos_mano_actual, dama_picas_en=tracker.dama_picas_en,
    )
    receptor = motor.receptor_pase(asiento)
    dador = next((i for i in range(4) if motor.receptor_pase(i) == asiento), None)
    moon_prob_rival = max(
        estimador.rival(
            motor=motor, rival_idx=i, agente_idx=asiento, vacios=tracker.vacios,
            historial=tracker.historial, receptor=receptor, dador=dador,
            cartas_dadas=dadas, cartas_recibidas=recibidas,
            corazones_rotos=motor.corazones_rotos,
        )
        for i in range(4) if i != asiento
    )
    puedo_alimentar = puede_alimentar(puntuacion_historica, asiento)
    return builder.construir(
        motor=motor, agente_idx=asiento, vacios=tracker.vacios,
        puntuacion_historica=puntuacion_historica,
        puntos_mano_actual=tracker.puntos_mano_actual, dama_picas_en=tracker.dama_picas_en,
        moon_prob_agente=moon_prob_agente, moon_prob_rival=moon_prob_rival,
        puedo_alimentar=puedo_alimentar,
        fase_pase=0.0, direccion_pase=0.0, n_pase_seleccionadas=0.0,
        cartas_pasadas=dadas or None, cartas_recibidas=recibidas or None,
    )


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
    estimador: EstimadorMoonProb | None = None,
    puntuacion_historica: List[int] | None = None,
    seats: List[int] | None = None,
    con_mask: bool = False,
) -> List[Ejemplo]:
    """Re-juega una mano y devuelve `(obs, carta_id)` por cada turno de los
    asientos en `seats` (por defecto solo `asiento_agente`).

    La obs es la COMPLETA (idéntica a `env._build_obs`), gracias a `_TrackerTactico`
    que replica el bookkeeping del env. `puntuacion_historica` es el marcador
    acumulado ANTES de esta mano (lo hila `ejemplos_de_partida`); `estimador` se
    reusa entre manos para no recargar los pesos moon en cada una.
    """
    estimador = estimador if estimador is not None else EstimadorMoonProb()
    scoreboard = puntuacion_historica if puntuacion_historica is not None else [0, 0, 0, 0]
    objetivo = {asiento_agente} if seats is None else set(seats)

    motor = _preparar_motor(mano)
    tracker = _TrackerTactico()
    ejemplos: List[Ejemplo] = []
    for j in mano.jugadas:
        actual = motor.obtener_jugador_actual()
        if actual != j.asiento:
            raise ValueError(
                f"Mano {mano.numero_mano}, baza {j.baza}: orden inconsistente "
                f"(motor espera asiento {actual}, registro dice {j.asiento})."
            )
        carta = Carta._TODAS[j.carta_id]
        if actual in objetivo:
            obs = _obs_para_asiento(
                obs_builder, motor, actual, tracker, scoreboard, estimador,
                mano, asiento_agente,
            )
            if con_mask:
                # Máscara legal del turno (para BC alineado con la inferencia
                # enmascarada del env; la carta humana jugada siempre es legal).
                mask = np.zeros(52, dtype=np.float32)
                for c in motor.obtener_jugadas_legales(actual):
                    mask[c.id] = 1.0
                ejemplos.append((obs, carta.id, mask))
            else:
                ejemplos.append((obs, carta.id))
        motor.jugar_carta(actual, carta)
        tracker.actualizar_vacios(motor, actual, carta)
        if len(motor.mesa) == 4:
            tracker.resolver_baza(motor)
    return ejemplos


def ejemplos_de_partida(
    partida: RegistroPartida, obs_builder: ObservacionBuilder | None = None,
    dim: int = DIM_ENTORNO, seats_de: str = "agente",
    moon_dir: str = RUTA_MOON, con_mask: bool = False,
) -> List[Ejemplo]:
    """Concatena los ejemplos de todas las manos reconstruibles de una partida,
    hilando el marcador acumulado entre manos (obs alineada con el env).

    `seats_de`: "agente" (solo la perspectiva del agente, para BC del agente) o
    "rivales" (los 3 asientos NO-agente = los humanos, para clonar su estilo).

    `moon_dir`: pesos del `EstimadorMoonProb` para las features [187:188]. DEBE
    coincidir con el `moon_dir` del env donde el modelo se usará después (si no,
    esas 2 features quedan fuera de distribución en inferencia).

    Manos no reconstruibles se saltan, pero su `puntuacion_mano` SÍ avanza el
    marcador (para que las manos siguientes tengan la puntuación histórica correcta,
    igual que `pimc_regret_real.procesar_partida`).
    """
    builder = obs_builder or ObservacionBuilder(dim=dim)
    estimador = EstimadorMoonProb(dir_modelos=moon_dir)
    ag = partida.asiento_agente
    if seats_de == "rivales":
        seats = [i for i in range(4) if i != ag]
    else:
        seats = [ag]

    scoreboard = [0, 0, 0, 0]
    out: List[Ejemplo] = []
    for mano in partida.manos:
        if mano_reconstruible(mano):
            out.extend(ejemplos_de_mano(
                mano, ag, builder, estimador, list(scoreboard), seats, con_mask,
            ))
        for i in range(4):
            scoreboard[i] += mano.puntuacion_mano[i]
    return out


def partidas_a_arrays(
    partidas, dim: int = DIM_ENTORNO, seats_de: str = "agente",
    moon_dir: str = RUTA_MOON, con_mask: bool = False,
):
    """Convierte un iterable de `RegistroPartida` en `(X[n,dim], y[n])`, o en
    `(X, y, M[n,52])` con `con_mask=True` (M = máscara legal de cada decisión).

    `seats_de="rivales"` extrae la perspectiva de los 3 humanos (para el opponent
    de imitación humana); "agente" la del agente (BC clásico). `moon_dir` debe
    coincidir con el env de inferencia (ver `ejemplos_de_partida`).
    """
    builder = ObservacionBuilder(dim=dim)
    obs_list: List[np.ndarray] = []
    acc_list: List[int] = []
    mask_list: List[np.ndarray] = []
    for p in partidas:
        for ej in ejemplos_de_partida(p, builder, dim, seats_de, moon_dir, con_mask):
            obs_list.append(ej[0])
            acc_list.append(ej[1])
            if con_mask:
                mask_list.append(ej[2])
    if not obs_list:
        vacios = (np.zeros((0, dim), dtype=np.float32), np.zeros((0,), dtype=np.int64))
        return (*vacios, np.zeros((0, 52), dtype=np.float32)) if con_mask else vacios
    X = np.asarray(obs_list, dtype=np.float32)
    y = np.asarray(acc_list, dtype=np.int64)
    if con_mask:
        return X, y, np.asarray(mask_list, dtype=np.float32)
    return X, y


__all__ = [
    "Ejemplo", "reconstruir_manos", "mano_reconstruible", "ejemplos_de_mano",
    "ejemplos_de_partida", "partidas_a_arrays",
]
