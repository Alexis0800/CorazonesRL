"""
Genera partidas SIMULADAS (self-play headless con `MotorCorazones` + bots de
`src/agentes`) en el mismo formato `RegistroPartida` que las partidas reales
del bridge, para entrenar los modelos moon_prob (`src/entorno/moon_model.py`)
con suficientes ejemplos positivos.

Motivación (2026-07-10): sobre 1712 manos reales del bridge (589 del dataset
original + 1123 recién reconstruidas), el agente solo hizo pozo 11 veces
(1 cada ~156 manos) -- ningún volumen realista de captura humana llega a un
centenar de positivos para el modelo "propio" en un plazo razonable (haría
falta ~15000 manos más). Eso, no la arquitectura ni las features, es la razón
de que su AUC de validación (0.44-0.46) esté cerca del azar (ver
docs/superpowers/plans/2026-07-06-moon-prob-modelo-aprendido.md, Task 11).

El self-play no tiene ese límite: `BotLunatico` SÍ intenta el pozo de verdad
(a diferencia de un rival real promedio, que rara vez lo logra), así que
forzarlo en un asiento al azar produce muchos más positivos por hora de
cómputo que capturar partidas humanas.

Reutiliza el pipeline de features EXACTO de `entrenar_moon_prob.py`
(`ejemplos_de_mano` / `features_propio` / `features_rival`) sin duplicar
nada: este script solo genera jugadas y las serializa como `RegistroPartida`
-- el archivo de salida se entrena exactamente igual que uno real:

    python scripts/generar_dataset_moon_simulado.py \
        --n-partidas 800 --manos-por-partida 4 --out data/partidas_simuladas_moon.jsonl
    cat data/partidas_bridge_full.jsonl \
        data/partidas_simuladas_moon.jsonl > data/partidas_moon_entrenamiento.jsonl
    python scripts/entrenar_moon_prob.py --partidas data/partidas_moon_entrenamiento.jsonl

ACTUALIZACIÓN (2026-07-20/21): la premisa "el modelo propio no puede entrenarse
con datos reales" quedó superada — `ejemplos_de_mano` extrae las 4 perspectivas
por mano, así que las lunas RIVALES también son positivos "propio" (desde el
asiento del lunador): con `data/partidas_bridge_full.jsonl` (372 lunas) el
propio llegó a AUC 0.945 val SIN simulados. Este script queda como aumento
opcional de positivos, ya no como requisito.
"""
from __future__ import annotations

import argparse
import random
from typing import Callable, List, Union

# --- bootstrap path: permite `python scripts/<x>.py` desde la raiz del repo ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
# --- fin bootstrap ---

from src.agentes.bot_atacante_lider import BotAtacanteLider
from src.agentes.bot_castigador import BotCastigador
from src.agentes.bot_experto import BotExperto
from src.agentes.bot_lunatico import BotLunatico
from src.agentes.heuristicos import bot_agresivo, bot_conservador, bot_evasivo
from src.agentes.pase import pase_heuristico
from src.captura.escritor import EscritorJsonl
from src.captura.modelos import Jugada, RegistroMano, RegistroPartida
from src.dominio.motor import MotorCorazones, hubo_pozo

PolicyFn = Callable  # (motor, idx, legales) -> Carta; opcionalmente .pasar()/.reset()

# Arquetipos "clonables" (clase) + funciones sin estado, igual que
# opponent_pool._bot_dificil: BotLunatico con más peso, es el único que
# persigue el pozo de verdad.
_POOL_CLASES = [BotExperto, BotCastigador, BotAtacanteLider, BotLunatico]
_POOL_PESOS = [1, 1, 1, 2]
_POOL_FUNCIONES = [bot_conservador, bot_agresivo, bot_evasivo]


def _elegir_policias(rng: random.Random, forzar_lunatico: bool) -> List[PolicyFn]:
    """4 políticas (una por asiento). Con `forzar_lunatico`, un asiento al
    azar es SIEMPRE BotLunatico -- garantiza intentos de pozo reales en vez
    de depender solo de que salga elegido al azar."""
    elegidas: List[Union[type, PolicyFn]] = []
    for _ in range(4):
        if rng.random() < 0.5:
            elegidas.append(rng.choices(_POOL_CLASES, weights=_POOL_PESOS, k=1)[0])
        else:
            elegidas.append(rng.choice(_POOL_FUNCIONES))
    if forzar_lunatico:
        elegidas[rng.randrange(4)] = BotLunatico
    return [p() if isinstance(p, type) else p for p in elegidas]


def _resetear_policias(policies: List[PolicyFn]) -> None:
    """Espejo de `corazones_rllib._reset_oponentes_por_mano`: resetea el
    estado por-mano de los bots stateful (BotCastigador, BotLunatico, ...)."""
    for p in policies:
        reset = getattr(p, "reset", None)
        if callable(reset):
            reset()


def simular_mano(motor: MotorCorazones, policies: List[PolicyFn]) -> RegistroMano:
    """Juega una mano completa (reparto -> pase -> 13 bazas) sobre `motor`
    (ya compartido entre manos de la misma partida, para que `numero_mano`
    -- y por tanto la rotación del pase -- avance con naturalidad)."""
    _resetear_policias(policies)
    motor.repartir()
    numero_mano = motor.numero_mano

    mano_inicial_agente = sorted(c.id for c in motor.jugadores[0].mano)
    direccion = motor.direccion_pase()
    pase_dado: List[int] = []
    pase_recibido: List[int] = []

    if direccion is not None:
        selecciones = {}
        for idx in range(4):
            pasar = getattr(policies[idx], "pasar", None)
            cartas = pasar(motor, idx) if callable(pasar) else pase_heuristico(motor, idx)
            selecciones[idx] = cartas
        pase_dado = [c.id for c in selecciones[0]]
        dador_de_0 = next(i for i in range(4) if motor.receptor_pase(i) == 0)
        pase_recibido = [c.id for c in selecciones[dador_de_0]]
        motor.ejecutar_pase(selecciones)

    jugadas: List[Jugada] = []
    for baza in range(1, 14):
        for _ in range(4):
            idx = motor.obtener_jugador_actual()
            legales = motor.obtener_jugadas_legales(idx)
            carta = policies[idx](motor, idx, legales)
            motor.jugar_carta(idx, carta)
            jugadas.append(Jugada(asiento=idx, carta_id=carta.id, baza=baza))
        motor.resolver_baza()

    return RegistroMano(
        numero_mano=numero_mano,
        direccion_pase=direccion,
        mano_inicial_agente=mano_inicial_agente,
        pase_dado=pase_dado,
        pase_recibido=pase_recibido,
        jugadas=jugadas,
        puntuacion_mano=motor.calcular_puntuacion_mano(),
    )


def simular_partida(
    rng: random.Random, partida_id: str, manos_por_partida: int, prob_forzar_lunatico: float,
) -> RegistroPartida:
    motor = MotorCorazones()
    policies = _elegir_policias(rng, forzar_lunatico=rng.random() < prob_forzar_lunatico)
    manos: List[RegistroMano] = []
    marcador = [0, 0, 0, 0]

    for _ in range(manos_por_partida):
        mano = simular_mano(motor, policies)
        manos.append(mano)
        for i in range(4):
            marcador[i] += mano.puntuacion_mano[i]

    ranking_peor_a_mejor = sorted(range(4), key=lambda s: -marcador[s])
    return RegistroPartida(
        partida_id=partida_id,
        timestamp="2026-07-10T00:00:00.000Z",  # constante: partidas sintéticas, no tienen fecha real
        asiento_agente=0,
        fuente="self-play-simulado",
        manos=manos,
        marcador_final=marcador,
        ranking_final=ranking_peor_a_mejor,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-partidas", type=int, default=800)
    p.add_argument("--manos-por-partida", type=int, default=4)
    p.add_argument("--prob-forzar-lunatico", type=float, default=0.5,
                   help="Fracción de partidas donde un asiento al azar es SIEMPRE BotLunatico.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rng = random.Random(args.seed)
    escritor = EscritorJsonl(args.out)
    moon_total = 0
    moon_asiento0 = 0
    n_manos = 0

    for i in range(args.n_partidas):
        partida = simular_partida(
            rng, partida_id=f"sim_{i:06d}",
            manos_por_partida=args.manos_por_partida,
            prob_forzar_lunatico=args.prob_forzar_lunatico,
        )
        escritor.escribir(partida)
        for m in partida.manos:
            n_manos += 1
            if m.puntuacion_mano and hubo_pozo(m.puntuacion_mano):
                moon_total += 1
                if m.puntuacion_mano[0] == 0:
                    moon_asiento0 += 1
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{args.n_partidas} partidas, {n_manos} manos, "
                  f"{moon_total} con pozo ({moon_asiento0} del asiento 0)", flush=True)

    print(f"\nGuardado en {args.out}")
    print(f"Partidas: {args.n_partidas}  |  Manos: {n_manos}")
    print(f"Manos con pozo: {moon_total} ({100 * moon_total / n_manos:.1f}%)  "
          f"|  del asiento 0: {moon_asiento0} ({100 * moon_asiento0 / n_manos:.1f}%)")


if __name__ == "__main__":
    main()
