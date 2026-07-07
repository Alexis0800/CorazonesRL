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
import json
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.dominio.carta import Carta
from src.captura.escritor import cargar_partidas
from src.captura.modelos import RegistroMano, RegistroPartida, carta_a_str
from src.captura.replay import _preparar_motor, mano_reconstruible, reconstruir_manos
from src.mcts.pimc import _clonar_motor, _crear_bots_mixto, simular_resto_mano
from src.agentes.bot_experto import BotExperto
from scripts.recomendador import Recomendador


class _PoliticaRecomendador:
    """Adapta `Recomendador` a la interfaz uniforme que usa este script."""

    def __init__(self, rec: Recomendador):
        self.rec = rec

    def reset_mano(self, mano_inicial) -> None:
        self.rec.reset_mano(mano_inicial)

    def elegir(self, motor, asiento_agente, mesa_actual, legales) -> Carta:
        return self.rec.recomendar_jugada(mesa_actual)

    def on_baza(self, jugadas, ganador) -> None:
        self.rec.registrar_baza(jugadas, ganador)

    def acumular_puntuacion(self, puntuacion_mano) -> None:
        self.rec.scores = [self.rec.scores[i] + puntuacion_mano[i] for i in range(4)]


class _PoliticaBotHeuristico:
    """Adapta un bot heurístico `(motor, idx, legales) -> Carta` (estrategia stateful
    intra-mano, ver `BotExperto`) a la misma interfaz, para correr el mismo backtest
    de regret con un bot en vez del campeón (referencia de techo/piso)."""

    def __init__(self, fabrica_bot):
        self._fabrica_bot = fabrica_bot
        self._bot = None

    def reset_mano(self, mano_inicial) -> None:
        self._bot = self._fabrica_bot()

    def elegir(self, motor, asiento_agente, mesa_actual, legales) -> Carta:
        return self._bot(motor, asiento_agente, legales)

    def on_baza(self, jugadas, ganador) -> None:
        pass  # el bot infiere su propio estado (voids, etc.) leyendo `motor` directamente

    def acumular_puntuacion(self, puntuacion_mano) -> None:
        pass  # los bots heurísticos de este repo no usan marcador acumulado


def _oraculo_info_completa(motor_real, agente_idx, legales, rng, rollouts):
    """Puntaje esperado por carta legal, CON las manos verdaderas (sin determinizar)."""
    acumulado = {c.id: 0.0 for c in legales}
    for _ in range(rollouts):
        bots = _crear_bots_mixto(rng)
        for carta in legales:
            clon = _clonar_motor(motor_real)
            acumulado[carta.id] += simular_resto_mano(clon, agente_idx, carta, bots)
    return {cid: total / rollouts for cid, total in acumulado.items()}


def decisiones_de_mano(mano: RegistroMano, asiento_agente: int, politica,
                        rng, rollouts: int, partida_id: str = ""):
    """Re-juega una mano real en paralelo: el motor de info-completa (verdad) y la
    política bajo prueba (solo info pública, igual que producción). Devuelve un
    registro dict por decisión (incluye `regret` y contexto legible para
    diagnóstico -- ver `--volcar-json`)."""
    manos_reales = reconstruir_manos(mano)
    motor_real = _preparar_motor(mano)
    politica.reset_mano(list(manos_reales[asiento_agente]))

    registros = []
    mesa_actual = []
    for j in mano.jugadas:
        actual = motor_real.obtener_jugador_actual()
        if actual != j.asiento:
            break  # orden inconsistente a medio camino: se descarta el resto de esta mano
        carta = Carta._TODAS[j.carta_id]

        if actual == asiento_agente:
            legales = motor_real.obtener_jugadas_legales(asiento_agente)
            if len(legales) > 1:
                carta_modelo = politica.elegir(motor_real, asiento_agente, mesa_actual, legales)
                scores = _oraculo_info_completa(motor_real, asiento_agente, legales, rng, rollouts)
                mejor = min(scores.values())
                reg = scores.get(carta_modelo.id, max(scores.values())) - mejor
                registros.append({
                    "partida_id": partida_id,
                    "mano": mano.numero_mano,
                    "baza": j.baza,
                    "carta_elegida": carta_a_str(carta_modelo.id),
                    "legales": [carta_a_str(c.id) for c in legales],
                    "scores": {carta_a_str(cid): round(s, 3) for cid, s in scores.items()},
                    "regret": reg,
                })

        motor_real.jugar_carta(actual, carta)
        mesa_actual.append((actual, carta))
        if len(mesa_actual) == 4:
            ganador = motor_real.resolver_baza()
            politica.on_baza(mesa_actual, ganador)
            mesa_actual = []

    return registros


def procesar_partida(partida: RegistroPartida, politica, rng, rollouts: int,
                      avisos: list) -> list:
    registros = []
    for mano in partida.manos:
        if not mano_reconstruible(mano):
            # No se puede re-jugar carta a carta, pero el marcador real SÍ se conoce
            # (viene de handFinal) -- se aplica para no desincronizar el resto de la
            # partida (features de marcador del Recomendador para las manos siguientes).
            if mano.puntuacion_mano:
                politica.acumular_puntuacion(mano.puntuacion_mano)
            politica.reset_mano([])
            continue
        try:
            registros.extend(decisiones_de_mano(
                mano, partida.asiento_agente, politica, rng, rollouts, partida.partida_id))
        except ValueError as e:
            # Datos reales imperfectos (p.ej. un eco/duplicado del bridge que la
            # dedup de importar_sesiones_bridge.py no atrapó del todo): se descarta
            # SOLO esta mano, no toda la partida -- mismo espíritu que
            # auditar_logs_servidor.py con líneas corruptas.
            avisos.append(f"{partida.partida_id}: mano {mano.numero_mano} descartada ({e})")
            if mano.puntuacion_mano:
                politica.acumular_puntuacion(mano.puntuacion_mano)
            politica.reset_mano([])
    return registros


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelo", help="Checkpoint RLlib a evaluar (mutuamente exclusivo con --bot)")
    p.add_argument("--bot", choices=["experto"],
                    help="Evaluar un bot heurístico en vez de un checkpoint (referencia de techo/piso)")
    p.add_argument("--partidas", required=True, help="RegistroPartida .jsonl (ver importar_sesiones_bridge.py)")
    p.add_argument("--max-decisiones", type=int, default=500)
    p.add_argument("--rollouts", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--volcar-json", help="Ruta .jsonl: vuelca TODAS las decisiones evaluadas (contexto + regret) para diagnóstico")
    args = p.parse_args()
    if not args.modelo and not args.bot:
        p.error("pasa --modelo <checkpoint> o --bot experto")

    rng = np.random.default_rng(args.seed)
    partidas = cargar_partidas(args.partidas)
    print(f"{len(partidas)} partidas cargadas de {args.partidas}", flush=True)

    if args.bot:
        politica = _PoliticaBotHeuristico(BotExperto)
        etiqueta = f"bot:{args.bot}"
        print(f"Política: bot heurístico '{args.bot}'", flush=True)
    else:
        rec = Recomendador(args.modelo)
        politica = _PoliticaRecomendador(rec)
        etiqueta = args.modelo
        print(f"Modelo: {args.modelo} (obs {rec.obs_dim}, "
              f"{'CON' if rec.con_pase else 'SIN'} pase)", flush=True)

    registros = []
    avisos: list = []
    for k, partida in enumerate(partidas):
        registros.extend(procesar_partida(partida, politica, rng, args.rollouts, avisos))
        if (k + 1) % 10 == 0 or len(registros) >= args.max_decisiones:
            print(f"  {k+1}/{len(partidas)} partidas, {len(registros)} decisiones evaluadas", flush=True)
        if len(registros) >= args.max_decisiones:
            break

    if avisos:
        print(f"\n{len(avisos)} manos descartadas por datos inconsistentes:")
        for a in avisos:
            print(f"  - {a}")

    if not registros:
        print("Sin decisiones evaluables (¿--partidas sin manos reconstruibles?)")
        return

    registros = registros[:args.max_decisiones]
    if args.volcar_json:
        with open(args.volcar_json, "w", encoding="utf-8") as f:
            for r in registros:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nVolcado {len(registros)} decisiones a {args.volcar_json}")

    regrets = np.array([r["regret"] for r in registros])
    optimo = float(np.mean(regrets <= 1e-9)) * 100
    print(f"\n=== Regret en {len(regrets)} decisiones REALES ({etiqueta}) ===")
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
