"""
Métricas de gate offline (Fase 1c del plan docs/plan_mejora_vs_humanos_2026-07-25.md).

Las 3 métricas que sirven de gate barato para futuros entrenamientos, medibles
sobre (a) datos reales del bridge por replay y (b) partidas simuladas en el env
con una política dada.

DEFINICIONES OPERATIVAS (exactas, tal como se implementan):

1. duck_innecesario_temprano
   Universo: decisiones de juego en bazas 1-8 donde TODAS estas condiciones:
     - NO lideramos (hay cartas en mesa);
     - la Q♠ ya fue CAPTURADA en una baza resuelta (cero riesgo de comerla);
     - la baza en mesa no tiene puntos (ni corazones ni Q♠);
     - tenemos con qué descargar una alta:
         * siguiendo palo: alguna legal con valor>=11 que GANA la baza actual
           (supera a la mejor carta del palo de salida en mesa), o
         * void (descartando): alguna legal con valor>=11.
   Duck: NO nos deshicimos de la alta —
     * siguiendo palo: la carta elegida no es una alta (>=11) ganadora;
     * void: la carta elegida tiene valor < 11.
   Tasa = ducks / oportunidades.

   ⚠ VARAS: las citadas en el plan (bot 36%/33.8% vs humano 22%/19.6%) NO se
   reprodujeron con esta ni con ~25 variantes barridas (solo-sigue / último en
   jugar / valor>=12 / bazas 1-4/1-6 / Q♠ viva / sin picas / duck="no ganó la
   baza" / liderando). El plan es además internamente inconsistente sobre el
   instrumento original ("Q♠ fuera" en §Fase 0 vs "Q♠ sin jugar" en §1c) y el
   análisis original no está en el repo. Varas CANÓNICAS bajo ESTE instrumento
   (ambos jsonl del bridge, 2026-07-25): bot 48.8% (n=3319) vs humanos 41.2%
   (n=9022) — brecha +7.6pp, z≈7.4, misma dirección y estable por corpus
   (48.0/41.4 en full, 50.7/40.9 en modolunar). Para gates usar SIEMPRE este
   script en ambas fuentes: lo que importa es que el bot se mueva hacia la
   vara humana medida con el MISMO instrumento.

2. muerte_como_4to
   Universo: manos donde al EMPEZAR el asiento tiene el peor marcador acumulado
   (empatado cuenta) y ese marcador es >=74 ("entramos 4tos con alguien >=74";
   si somos el peor, el >=74 es necesariamente nuestro).
   Muerte: la partida termina EN ESA MANO (alguien cruza 100) y quedamos con el
   peor marcador final (empate al peor cuenta como 4to).
   Se reporta también pts/mano comidos en esas manos.
   Varas del plan: bot 12.0% vs humanos 5.3%. Con este instrumento: bot 11.0%
   (n=409) vs humanos 2.9% (n=1114) — el bot calza, el humano queda algo bajo
   (variantes probadas: peor-estricto 11.4/3.0, muerte-eventual 25.4/7.6,
   por-partida 94/46; ninguna reproduce 5.3 exacto). Vara canónica: 11.0/2.9.

3. conversion_luna_rival
   Lunas logradas por los rivales, POR MANO y POR RIVAL:
   #{manos con pozo de un rival} / (manos * 3). Vara real: ~2.5%/mano
   (cada humano corona ~2.5% de las manos que juega contra el bot).
   Se reporta también la tasa de luna propia por mano de cada grupo.
   REPRODUCIDA EXACTA con este instrumento: luna_propia humanos 2.52%/mano
   (= la cifra de docs/auditoria_moon_2026-07-20.md), bot 0.36%.

Uso:
    # Varas sobre datos reales (bot = asiento_agente; humanos = los otros 3):
    python scripts/metricas_gate.py --fuente real

    # Sobre el env, con un checkpoint, vs mesa mixta (experto+castigador+lunatico):
    python scripts/metricas_gate.py --fuente env --modelo models/produccion/v10c_campeon --partidas 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# --- bootstrap path ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# --- fin bootstrap ---

from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones, PUNTOS_POZO, hubo_pozo, asiento_pozo

RUTAS_BRIDGE = ["data/partidas_bridge_full.jsonl", "data/partidas_bridge_modolunar.jsonl"]
VALOR_ALTA = 11  # J o mayor
BAZA_TEMPRANA_MAX = 8
UMBRAL_BORDE = 74


# ---------------------------------------------------------------------------
# Núcleo compartido (real y env): clasificación de una decisión / una mano
# ---------------------------------------------------------------------------

def dama_capturada(motor: MotorCorazones) -> bool:
    """True si la Q♠ ya está en las bazas ganadas de alguien (baza resuelta)."""
    return any(
        any(c.es_dama_de_picas for c in j.bazas_ganadas) for j in motor.jugadores
    )


def clasificar_duck(motor: MotorCorazones, carta: Carta,
                    legales: List[Carta]) -> Optional[bool]:
    """Clasifica una decisión según la métrica 1 (ver docstring del módulo).

    Returns:
        None  -> no es una oportunidad de descarga (no cuenta).
        True  -> oportunidad + duck (no se deshizo de la alta).
        False -> oportunidad + descargó la alta.
    """
    if motor.numero_baza > BAZA_TEMPRANA_MAX:
        return None
    if not motor.mesa:                      # liderando: no hay baza que ganar
        return None
    if not dama_capturada(motor):           # Q♠ viva (o en mesa) = hay riesgo
        return None
    cartas_mesa = [c for _, c in motor.mesa]
    if any(c.puntos > 0 for c in cartas_mesa):
        return None

    palo_salida = motor.palo_de_salida
    sigue_palo = any(c.palo == palo_salida for c in legales)
    if sigue_palo:
        # legales = solo cartas del palo de salida (filtro 2 del motor)
        mejor_mesa = max(c.valor for c in cartas_mesa if c.palo == palo_salida)
        ganadoras_altas = [c for c in legales
                           if c.valor >= VALOR_ALTA and c.valor > mejor_mesa]
        if not ganadoras_altas:
            return None
        return not (carta.valor >= VALOR_ALTA and carta.valor > mejor_mesa)
    # Void en el palo de salida: cualquier alta se puede descartar gratis.
    altas = [c for c in legales if c.valor >= VALOR_ALTA]
    if not altas:
        return None
    return carta.valor < VALOR_ALTA


class Acumulador:
    """Contadores de las 3 métricas para un grupo de asientos (bot/humanos/agente)."""

    def __init__(self) -> None:
        self.duck_oportunidades = 0
        self.duck_ducks = 0
        self.m4_manos = 0          # manos que entramos 4tos con >=74
        self.m4_muertes = 0        # ...y la partida terminó ahí con nosotros 4tos
        self.m4_pts: List[int] = []  # pts comidos en esas manos
        self.manos = 0             # manos-asiento observadas (para tasas de luna)
        self.lunas_propias = 0
        self.lunas_rivales = 0

    def decision(self, resultado: Optional[bool]) -> None:
        if resultado is None:
            return
        self.duck_oportunidades += 1
        if resultado:
            self.duck_ducks += 1

    def mano(self, seat: int, scores_pre: List[int], pts_mano: List[int],
             termina: bool) -> None:
        """Registra una mano cerrada desde la perspectiva de `seat`."""
        self.manos += 1
        pozo = asiento_pozo(pts_mano)
        if pozo is not None:
            if pozo == seat:
                self.lunas_propias += 1
            else:
                self.lunas_rivales += 1
        if scores_pre[seat] == max(scores_pre) and scores_pre[seat] >= UMBRAL_BORDE:
            self.m4_manos += 1
            self.m4_pts.append(pts_mano[seat])
            if termina:
                scores_fin = [s + p for s, p in zip(scores_pre, pts_mano)]
                if scores_fin[seat] == max(scores_fin):
                    self.m4_muertes += 1

    def resumen(self) -> dict:
        return {
            "duck_innecesario_temprano": (
                self.duck_ducks / self.duck_oportunidades
                if self.duck_oportunidades else None),
            "duck_oportunidades": self.duck_oportunidades,
            "muerte_como_4to": (
                self.m4_muertes / self.m4_manos if self.m4_manos else None),
            "manos_4to_borde": self.m4_manos,
            "pts_por_mano_4to_borde": (
                sum(self.m4_pts) / len(self.m4_pts) if self.m4_pts else None),
            "luna_propia_por_mano": (
                self.lunas_propias / self.manos if self.manos else None),
            "conversion_luna_rival": (
                self.lunas_rivales / (self.manos * 3) if self.manos else None),
            "manos": self.manos,
        }


# ---------------------------------------------------------------------------
# Fuente REAL: replay de los JSONL del bridge
# ---------------------------------------------------------------------------

def metricas_reales(rutas: List[str]) -> dict:
    """Replay de las partidas reales: bot = asiento_agente, humanos = los otros 3."""
    from src.captura.escritor import cargar_partidas
    from src.captura.replay import _preparar_motor, mano_reconstruible

    acc = {"bot": Acumulador(), "humanos": Acumulador()}
    manos_saltadas = 0

    for ruta in rutas:
        for partida in cargar_partidas(ruta):
            ag = partida.asiento_agente
            scores = [0, 0, 0, 0]
            for mano in partida.manos:
                # --- métrica 1: replay de las jugadas registradas ---
                if mano_reconstruible(mano):
                    motor = _preparar_motor(mano)
                    for j in mano.jugadas:
                        actual = motor.obtener_jugador_actual()
                        if actual != j.asiento:
                            break  # registro inconsistente: abandonar la mano
                        carta = Carta._TODAS[j.carta_id]
                        legales = motor.obtener_jugadas_legales(actual)
                        grupo = "bot" if actual == ag else "humanos"
                        acc[grupo].decision(clasificar_duck(motor, carta, legales))
                        motor.jugar_carta(actual, carta)
                        # REGLA CRÍTICA de replay manual: resolver cada baza.
                        if len(motor.mesa) == 4:
                            motor.resolver_baza()
                else:
                    manos_saltadas += 1

                # --- métricas 2 y 3: con la puntuación registrada ---
                pts = list(mano.puntuacion_mano or [])
                if len(pts) != 4 or sum(pts) not in (26, PUNTOS_POZO):
                    manos_saltadas += 1
                    continue
                scores_post = [s + p for s, p in zip(scores, pts)]
                termina = max(scores_post) >= MotorCorazones.LIMITE_PARTIDA
                for seat in range(4):
                    grupo = "bot" if seat == ag else "humanos"
                    acc[grupo].mano(seat, scores, pts, termina)
                scores = scores_post

    return {
        "bot": acc["bot"].resumen(),
        "humanos": acc["humanos"].resumen(),
        "manos_saltadas": manos_saltadas,
    }


# ---------------------------------------------------------------------------
# Fuente ENV: partidas simuladas con un checkpoint vs mesa mixta fija
# ---------------------------------------------------------------------------

def metricas_env(ruta_modelo: str, n_partidas: int, obs_dim: int,
                 moon_dir: str, seed_offset: int = 0) -> dict:
    """Juega n partidas (checkpoint como agente, mesa mixta fija con lunero) y
    computa las mismas 3 métricas para el agente, mirando el motor del env en
    cada decisión (mismo `clasificar_duck` que el modo real)."""
    import torch

    from src.agentes.bot_experto import BotExperto
    from src.agentes.bot_castigador import BotCastigador
    from src.agentes.bot_lunatico import BotLunatico
    from src.entorno.corazones_rllib import CorazonesEnvRLlib
    from src.entorno.dimensiones import con_pase_de_obs
    from src.rllib.eval_bots import _obtener_modelo
    from src.rllib.utils import cargar_policy_desde_checkpoint

    policy = cargar_policy_desde_checkpoint(ruta_modelo)
    model = _obtener_modelo(policy, obs_dim)

    def factory(ai=0):
        # Mesa mixta con lunero garantizado (necesario para la métrica 3).
        clases = [BotExperto, BotCastigador, BotLunatico]
        idxs = [i for i in range(4) if i != ai]
        return {i: cls() for i, cls in zip(idxs, clases)}

    env = CorazonesEnvRLlib({
        "obs_dim": obs_dim, "agente_idx": 0, "random_position": False,
        "opponent_factory": factory, "gamma": 0.999,
        "con_pase": con_pase_de_obs(obs_dim), "moon_dir": moon_dir,
    })

    acum = Acumulador()
    puestos: List[int] = []

    for i in range(n_partidas):
        torch.manual_seed(seed_offset + i)
        obs, _ = env.reset(seed=seed_offset + i)
        ag = env._agente_idx
        prev_scores = list(env._motor.puntuaciones_historicas())
        prev_manos = 0
        done = False
        info: dict = {}
        while not done:
            with torch.no_grad():
                o = torch.as_tensor(obs["obs"], dtype=torch.float32).unsqueeze(0)
                m = torch.as_tensor(obs["action_mask"], dtype=torch.float32).unsqueeze(0)
                logits, _ = model.forward({"obs": {"obs": o, "action_mask": m}}, [], None)
                accion = int(logits.argmax(dim=1).item())
            if not env._fase_pase:
                legales = env._motor.obtener_jugadas_legales(ag)
                acum.decision(clasificar_duck(
                    env._motor, Carta._TODAS[accion], legales))
            obs, _, done, _, info = env.step(accion)
            if env._manos_jugadas > prev_manos or done:
                scores = list(env._motor.puntuaciones_historicas())
                pts = [s - p for s, p in zip(scores, prev_scores)]
                if sum(pts) in (26, PUNTOS_POZO):
                    acum.mano(ag, prev_scores, pts, done)
                prev_scores = scores
                prev_manos = env._manos_jugadas
        puestos.append(info.get("puesto", 0))

    res = acum.resumen()
    res["puesto_medio"] = sum(puestos) / len(puestos) if puestos else None
    res["partidas"] = n_partidas
    return {"agente": res}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _pct(x) -> str:
    return f"{100 * x:5.1f}%" if x is not None else "  n/a "


def _imprimir_tabla(res: dict) -> None:
    grupos = [g for g in ("bot", "humanos", "agente") if g in res]
    ancho = 34
    print(f"\n{'metrica':<{ancho}}" + "".join(f"{g:>14}" for g in grupos))
    filas = [
        ("duck_innecesario_temprano", "duck_innecesario_temprano", _pct),
        ("  (oportunidades)", "duck_oportunidades", lambda v: f"{v}"),
        ("muerte_como_4to", "muerte_como_4to", _pct),
        ("  (manos 4to con >=74)", "manos_4to_borde", lambda v: f"{v}"),
        ("  pts/mano en esas manos", "pts_por_mano_4to_borde",
         lambda v: f"{v:.2f}" if v is not None else "n/a"),
        ("conversion_luna_rival (/mano/rival)", "conversion_luna_rival", _pct),
        ("luna_propia_por_mano", "luna_propia_por_mano", _pct),
        ("  (manos)", "manos", lambda v: f"{v}"),
    ]
    for etiqueta, clave, fmt in filas:
        print(f"{etiqueta:<{ancho}}" + "".join(
            f"{fmt(res[g].get(clave)):>14}" for g in grupos))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fuente", choices=["real", "env"], required=True)
    p.add_argument("--bridge", nargs="+", default=RUTAS_BRIDGE,
                   help="JSONL de partidas reales (--fuente real)")
    p.add_argument("--modelo", help="Checkpoint RLlib (--fuente env)")
    p.add_argument("--partidas", type=int, default=200)
    p.add_argument("--obs-dim", type=int, default=None)
    p.add_argument("--moon-dir", default=None)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--json", action="store_true", help="Volcar resultado en JSON")
    args = p.parse_args()

    if args.fuente == "real":
        res = metricas_reales(args.bridge)
        print(f"=== VARAS REALES ({', '.join(args.bridge)}) ===")
        print(f"(manos saltadas por registro incompleto: {res.pop('manos_saltadas')})")
    else:
        if not args.modelo:
            p.error("--fuente env requiere --modelo")
        from src.entorno.dimensiones import DIM_V12
        from src.entorno.moon_model import RUTA_MOON
        obs_dim = args.obs_dim if args.obs_dim else DIM_V12
        moon_dir = args.moon_dir if args.moon_dir else RUTA_MOON
        res = metricas_env(args.modelo, args.partidas, obs_dim, moon_dir,
                           args.seed_offset)
        print(f"=== METRICAS EN ENV ({args.modelo}, {args.partidas} partidas "
              f"vs experto+castigador+lunatico) ===")
        print(f"puesto_medio: {res['agente']['puesto_medio']:.3f}")

    _imprimir_tabla(res)
    if args.json:
        print("\n" + json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
