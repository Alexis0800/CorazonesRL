"""
PIMC-regret sobre PARTIDAS REALES (importadas del bridge SFS2X con
`importar_sesiones_bridge.py`), para responder: ¿el campeón juega peor en
partidas reales que en la evaluación simulada, o solo enfrenta rivales/suerte
distintos?

Diferencia clave con `pimc_regret.py` (que genera estados por self-play, con
manos rivales DESCONOCIDAS -> determinizar): acá `replay.py` reconstruye la
mano REAL completa (las 52 jugadas grabadas determinan las 4 manos exactas),
así que el oráculo de referencia no necesita adivinar nada -- conoce las
manos verdaderas y solo promedia sobre la política de rollout (sin ruido de
información oculta).

En cada decisión de JUEGO real del asiento del agente (con >1 carta legal):
  1. Se le pregunta al `Recomendador` (mismo camino que producción: SOLO su
     propia mano + estado público inferido en vivo) qué jugaría.
  2. Se calcula, con la mano REAL de los 4 jugadores + rollout de arquetipos
     mixtos, la puntuación esperada de cada carta legal.
  3. regret = esperado(carta del modelo) - min_c esperado(c).

Solo evalúa decisiones de JUEGO (igual que pimc_regret.py, no evalúa el pase).

Uso:
    python scripts/pimc_regret_real.py --modelo models/produccion/v10c_campeon \
        --partidas data/partidas_bridge.jsonl --max-decisiones 500 --rollouts 20
"""
from __future__ import annotations

# --- bootstrap path: permite `python scripts/<x>.py` desde la raiz del repo ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
# --- fin bootstrap ---

import argparse
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.dominio.carta import Carta
from src.captura.escritor import cargar_partidas
from src.captura.modelos import RegistroMano, RegistroPartida
from src.captura.replay import _preparar_motor, mano_reconstruible, reconstruir_manos
from src.mcts.pimc import _clonar_motor, _crear_bots_mixto, simular_resto_mano
from scripts.recomendador import Recomendador


def _oraculo_info_completa(motor_real, agente_idx, legales, rng, rollouts):
    """Puntaje esperado por carta legal, CON las manos verdaderas (sin determinizar)."""
    acumulado = {c.id: 0.0 for c in legales}
    for _ in range(rollouts):
        bots = _crear_bots_mixto(rng)
        for carta in legales:
            clon = _clonar_motor(motor_real)
            acumulado[carta.id] += simular_resto_mano(clon, agente_idx, carta, bots)
    return {cid: total / rollouts for cid, total in acumulado.items()}


def decisiones_de_mano(mano: RegistroMano, asiento_agente: int, rec: Recomendador,
                        rng, rollouts: int):
    """Re-juega una mano real en paralelo: el motor de info-completa (verdad) y el
    Recomendador (solo info pública, igual que producción). Devuelve regrets."""
    manos_reales = reconstruir_manos(mano)
    motor_real = _preparar_motor(mano)
    rec.reset_mano(list(manos_reales[asiento_agente]))

    regrets = []
    mesa_actual = []
    for j in mano.jugadas:
        actual = motor_real.obtener_jugador_actual()
        if actual != j.asiento:
            break  # orden inconsistente a medio camino: se descarta el resto de esta mano
        carta = Carta._TODAS[j.carta_id]

        if actual == asiento_agente:
            legales = motor_real.obtener_jugadas_legales(asiento_agente)
            if len(legales) > 1:
                carta_modelo = rec.recomendar_jugada(mesa_actual)
                scores = _oraculo_info_completa(motor_real, asiento_agente, legales, rng, rollouts)
                mejor = min(scores.values())
                reg = scores.get(carta_modelo.id, max(scores.values())) - mejor
                regrets.append(reg)

        motor_real.jugar_carta(actual, carta)
        mesa_actual.append((actual, carta))
        if len(mesa_actual) == 4:
            ganador = motor_real.resolver_baza()
            rec.registrar_baza(mesa_actual, ganador)
            mesa_actual = []

    return regrets


def procesar_partida(partida: RegistroPartida, rec: Recomendador, rng, rollouts: int,
                      avisos: list) -> list:
    regrets = []
    for mano in partida.manos:
        if not mano_reconstruible(mano):
            # No se puede re-jugar carta a carta, pero el marcador real SÍ se conoce
            # (viene de handFinal) -- se aplica para no desincronizar el resto de la
            # partida (features de marcador del Recomendador para las manos siguientes).
            if mano.puntuacion_mano:
                rec.scores = [rec.scores[i] + mano.puntuacion_mano[i] for i in range(4)]
            rec.reset_mano([])
            continue
        try:
            regrets.extend(decisiones_de_mano(mano, partida.asiento_agente, rec, rng, rollouts))
        except ValueError as e:
            # Datos reales imperfectos (p.ej. un eco/duplicado del bridge que la
            # dedup de importar_sesiones_bridge.py no atrapó del todo): se descarta
            # SOLO esta mano, no toda la partida -- mismo espíritu que
            # auditar_logs_servidor.py con líneas corruptas.
            avisos.append(f"{partida.partida_id}: mano {mano.numero_mano} descartada ({e})")
            if mano.puntuacion_mano:
                rec.scores = [rec.scores[i] + mano.puntuacion_mano[i] for i in range(4)]
            rec.reset_mano([])
    return regrets


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelo", required=True)
    p.add_argument("--partidas", required=True, help="RegistroPartida .jsonl (ver importar_sesiones_bridge.py)")
    p.add_argument("--max-decisiones", type=int, default=500)
    p.add_argument("--rollouts", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    partidas = cargar_partidas(args.partidas)
    print(f"{len(partidas)} partidas cargadas de {args.partidas}", flush=True)

    rec = Recomendador(args.modelo)
    print(f"Modelo: {args.modelo} (obs {rec.obs_dim}, "
          f"{'CON' if rec.con_pase else 'SIN'} pase)", flush=True)

    regrets = []
    avisos: list = []
    for k, partida in enumerate(partidas):
        regrets.extend(procesar_partida(partida, rec, rng, args.rollouts, avisos))
        if (k + 1) % 10 == 0 or len(regrets) >= args.max_decisiones:
            print(f"  {k+1}/{len(partidas)} partidas, {len(regrets)} decisiones evaluadas", flush=True)
        if len(regrets) >= args.max_decisiones:
            break

    if avisos:
        print(f"\n{len(avisos)} manos descartadas por datos inconsistentes:")
        for a in avisos:
            print(f"  - {a}")

    if not regrets:
        print("Sin decisiones evaluables (¿--partidas sin manos reconstruibles?)")
        return

    regrets = np.array(regrets[:args.max_decisiones])
    optimo = float(np.mean(regrets <= 1e-9)) * 100
    print(f"\n=== Regret en {len(regrets)} decisiones REALES ({args.modelo}) ===")
    print(f"  regret medio:   {regrets.mean():.3f} pts/jugada")
    print(f"  regret mediana: {np.median(regrets):.3f}")
    print(f"  % óptimo:       {optimo:.1f}%")
    print(f"  p90 / p99 / max: {np.percentile(regrets, 90):.2f} / "
          f"{np.percentile(regrets, 99):.2f} / {regrets.max():.2f}")
    for umbral in (3, 5, 10):
        pct = 100 * float(np.mean(regrets >= umbral))
        print(f"  decisiones con regret >= {umbral}: {pct:.1f}%")
    print("\n(regret = puntos esperados que la jugada elegida 'regala' vs. la óptima, "
          "según un oráculo de INFO COMPLETA + rollout de arquetipos mixtos sobre "
          "las manos REALES. Comparar con el regret medido en pimc_regret.py sobre "
          "estados simulados -- si acá es mucho mayor, el modelo decide peor en "
          "partidas reales; si es similar, el problema no es la calidad de decisión.)")


if __name__ == "__main__":
    main()
