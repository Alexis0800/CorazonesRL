"""
Partida interactiva de Corazones: humano vs modelo RL + 2 bots.

Jugás contra el mejor modelo entrenado. En tu turno, ves tu mano,
la mesa, y elegís qué carta jugar. El modelo y los bots juegan
automáticamente.

Uso:
    python jugar_contra_modelo.py
    python jugar_contra_modelo.py --modelo modelos_historicos/v2/modelo_final
    python jugar_contra_modelo.py --modelo-mi-idx 2  # Modelo en posición 2
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.dominio.carta import Carta, _PALOS, _VALORES
from src.entorno.single_agent import CorazonesEnv
from src.agentes.heuristicos import bot_conservador, bot_agresivo, bot_evasivo

# ----------------------------------------------------------------
# Helpers de visualización
# ----------------------------------------------------------------

_PALO_SIMBOLO = {0: "♣", 1: "♦", 2: "♠", 3: "♥"}
_VALOR_SIMBOLO = {2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8",
                  9: "9", 10: "10", 11: "J", 12: "Q", 13: "K", 14: "A"}
NOMBRES_PALOS = {"♣": 0, "♦": 1, "♠": 2, "♥": 3}
_NOMBRE_VALOR = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
                 "9": 9, "10": 10, "j": 11, "J": 11, "q": 12, "Q": 12,
                 "k": 13, "K": 13, "a": 14, "A": 14}


def carta_a_str(c: Carta) -> str:
    """Convierte una Carta a string legible."""
    return f"{_VALOR_SIMBOLO[c.valor]}{_PALO_SIMBOLO[c.palo]}"


def parsear_carta(texto: str) -> int:
    """Convierte "A♥" a ID 0-51."""
    texto = texto.strip()
    if not texto:
        raise ValueError("Texto vacío")
    palo_char = texto[-1]
    if palo_char not in NOMBRES_PALOS:
        raise ValueError(f"Palo desconocido: '{palo_char}'")
    palo = NOMBRES_PALOS[palo_char]
    valor_str = texto[:-1]
    if valor_str not in _NOMBRE_VALOR:
        raise ValueError(f"Valor desconocido: '{valor_str}'")
    valor = _NOMBRE_VALOR[valor_str]
    return palo * 13 + (valor - 2)


# ----------------------------------------------------------------
# Política oponente para el modelo RL (con VecNormalize)
# ----------------------------------------------------------------

def _crear_politica_modelo(
    modelo,
    vecnorm,
    agente_idx_modelo: int,
) -> callable:
    """Crea una función política compatible con CorazonesEnv que usa el modelo RL.

    La función recibe (motor, idx, legales) y retorna una Carta, usando el
    modelo MaskablePPO con VecNormalize y action masking.

    Args:
        modelo: Modelo MaskablePPO cargado.
        vecnorm: Objeto VecNormalize cargado (o None).
        agente_idx_modelo: Índice del jugador que controla el modelo.

    Returns:
        Función política (motor, idx, legales) -> Carta.
    """

    def politica(motor, idx: int, legales: List[Carta]) -> Carta:
        # Construir observación manualmente (mismo formato que CorazonesEnv)
        obs = np.zeros(187, dtype=np.float32)

        # [0:52] Mano del modelo
        for c in motor.jugadores[idx].mano:
            obs[c.id] = 1.0

        # [52:104] Mesa
        for _, c in motor.mesa:
            obs[52 + c.id] = 1.0

        # [104:156] Cementerio (todas las bazas ganadas por todos)
        for j in motor.jugadores:
            for c in j.bazas_ganadas:
                obs[104 + c.id] = 1.0

        # Puntajes históricos — los tomamos del entorno si están disponibles
        # (no accesibles desde el motor directamente; usamos 0 como fallback)
        # [172:176] y [176:180] se quedan en 0

        # [180] Corazones rotos
        obs[180] = 1.0 if motor.corazones_rotos else 0.0

        # [181] Posición en la baza
        pos_map = {0: 0.0, 1: 0.33, 2: 0.66, 3: 1.0}
        obs[181] = pos_map.get(len(motor.mesa), 0.0)

        # Normalizar
        if vecnorm is not None:
            try:
                obs_rms = vecnorm.obs_rms
                if obs_rms is not None and obs_rms.count > 0:
                    mean = np.array(obs_rms.mean)
                    var = np.array(obs_rms.var)
                    obs = np.clip(
                        (obs - mean) / np.sqrt(var + 1e-8), -10.0, 10.0
                    ).astype(np.float32)
            except Exception:
                pass

        # Action mask
        mask = np.zeros(52, dtype=np.bool_)
        for c in legales:
            mask[c.id] = True

        # Predecir
        action, _ = modelo.predict(obs, action_masks=mask, deterministic=False)
        return Carta._TODAS[int(action.item()) if hasattr(action, 'item') else int(action)]

    return politica


# ----------------------------------------------------------------
# Mostrar estado del juego
# ----------------------------------------------------------------

def _mostrar_estado(env: CorazonesEnv, humano_idx: int, modelo_idx: int) -> None:
    """Muestra el estado actual del juego al humano."""
    motor = env.motor
    punt_hist = env._puntuacion_historica
    puntos_mano = env._puntos_mano_actual

    print("\n" + "═" * 60)
    print("  🃏  CORAZONES — Partida Interactiva  🃏")
    print("═" * 60)

    # Puntajes históricos
    print(f"\n  📊 Puntajes acumulados:")
    for i in range(4):
        rol = "👤 VOS" if i == humano_idx else (
            "🤖 MODELO" if i == modelo_idx else "🤖 Bot")
        print(f"     Jugador {i} [{rol}]: {punt_hist[i]} pts")

    # Puntos en mano actual
    print(f"\n  🎯 Puntos esta mano:")
    for i in range(4):
        print(f"     Jugador {i}: {puntos_mano[i]} pts")

    # Mesa
    if motor.mesa:
        print(f"\n  🃏 Mesa (baza actual):")
        for j_idx, carta in motor.mesa:
            rol = "VOS" if j_idx == humano_idx else (
                "MODELO" if j_idx == modelo_idx else "Bot")
            print(f"     [{rol}] {carta_a_str(carta)}")
        print(f"     Palo de salida: {_PALO_SIMBOLO[motor.palo_de_salida]}")
    else:
        print(f"\n  🃏 Mesa: (vacía — liderás vos)")

    # Corazones rotos
    print(f"\n  💔 Corazones rotos: {'SÍ' if motor.corazones_rotos else 'NO'}")
    print(f"  🔢 Baza n.º {motor.numero_baza} de 13")

    # Mano del humano
    mano = motor.jugadores[humano_idx].mano
    print(f"\n  🎴 TU MANO ({len(mano)} cartas):")
    # Agrupar por palo
    por_palo: Dict[int, List[Carta]] = {0: [], 1: [], 2: [], 3: []}
    for c in mano:
        por_palo[c.palo].append(c)
    for palo in range(4):
        if por_palo[palo]:
            cartas_str = "  ".join(
                f"{carta_a_str(c):>4}" for c in sorted(por_palo[palo], key=lambda x: x.valor)
            )
            print(f"     {_PALO_SIMBOLO[palo]}: {cartas_str}")

    print()


# ----------------------------------------------------------------
# Loop principal
# ----------------------------------------------------------------

def jugar_partida(
    modelo_path: str,
    vecnorm_path: Optional[str],
    humano_idx: int = 0,
    modelo_idx: int = 1,
) -> None:
    """Ejecuta una partida interactiva: humano vs modelo + 2 bots.

    Args:
        modelo_path: Ruta al modelo .zip.
        vecnorm_path: Ruta al VecNormalize .pkl.
        humano_idx: Índice del jugador humano (0-3).
        modelo_idx: Índice del jugador controlado por el modelo (0-3, ≠ humano).
    """
    # Validar índices
    if humano_idx == modelo_idx:
        print("❌ El humano y el modelo deben ocupar posiciones diferentes.")
        return
    if not (0 <= humano_idx <= 3 and 0 <= modelo_idx <= 3):
        print("❌ Los índices deben estar entre 0 y 3.")
        return

    # Cargar modelo
    if not modelo_path.endswith(".zip"):
        modelo_path += ".zip"
    if not os.path.exists(modelo_path):
        print(f"❌ Modelo no encontrado: {modelo_path}")
        return

    from sb3_contrib import MaskablePPO
    modelo = MaskablePPO.load(modelo_path, device="cpu")
    print(f"📦 Modelo cargado: {modelo_path}")

    # Cargar VecNormalize
    vecnorm = None
    if vecnorm_path and os.path.exists(vecnorm_path):
        with open(vecnorm_path, "rb") as f:
            vecnorm = pickle.load(f)
        print(f"📊 VecNormalize cargado: {vecnorm_path}")
    else:
        print("⚠️  Sin VecNormalize — el modelo puede jugar peor.")

    # Crear políticas
    politica_rl = _crear_politica_modelo(modelo, vecnorm, modelo_idx)

    # Asignar bots a los slots restantes
    bots_disponibles = [bot_conservador, bot_agresivo, bot_evasivo]
    politicas: Dict[int, object] = {}
    bot_idx = 0
    for i in range(4):
        if i == humano_idx or i == modelo_idx:
            continue
        politicas[i] = bots_disponibles[bot_idx]
        bot_idx += 1

    # Agregar el modelo como oponente
    politicas[modelo_idx] = politica_rl

    # Crear entorno con el humano como agente
    env = CorazonesEnv(
        agente_idx=humano_idx,
        politicas_oponentes=politicas,
    )

    print(f"\n{'═' * 60}")
    print(f"  👤 Vos: Jugador {humano_idx}")
    print(f"  🤖 Modelo RL: Jugador {modelo_idx}")
    print(
        f"  🤖 Bots: Jugadores {', '.join(str(i) for i in range(4) if i not in (humano_idx, modelo_idx))}")
    print(f"{'═' * 60}")

    obs, _ = env.reset(seed=None)

    partida_terminada = False
    while not partida_terminada:
        # Mostrar estado
        _mostrar_estado(env, humano_idx, modelo_idx)

        # Obtener jugadas legales
        legales = env.motor.obtener_jugadas_legales(humano_idx)
        if not legales:
            print("⚠️  No hay jugadas legales. Continuando...")
            # Esto no debería pasar si es nuestro turno
            break

        # Mostrar legales
        print("  ✅ Jugadas legales:")
        legales_ordenadas = sorted(legales, key=lambda c: (c.palo, c.valor))
        # Agrupar por palo
        palo_actual = None
        for c in legales_ordenadas:
            if c.palo != palo_actual:
                palo_actual = c.palo
                print(f"     {_PALO_SIMBOLO[c.palo]}:", end="")
            print(f" {carta_a_str(c)}", end="")
        print()
        print()

        # Pedir input al humano
        while True:
            try:
                eleccion = input(
                    "  🎯 ¿Qué carta jugás? (ej: 3♣ o 'ayuda'): ").strip()

                if eleccion.lower() in ("ayuda", "help", "?"):
                    print("  📝 Formato: valor + palo. Ej: A♥, 2♣, K♠, Q♦, 10♥")
                    print("  📝 También: 'salir' para terminar la partida")
                    continue

                if eleccion.lower() in ("salir", "exit", "quit", "q"):
                    print("  👋 Partida cancelada.")
                    env.close()
                    return

                carta_id = parsear_carta(eleccion)
                carta_elegida = Carta._TODAS[carta_id]

                if carta_elegida not in legales:
                    print(
                        f"  ⚠️  {carta_a_str(carta_elegida)} no es una jugada legal.")
                    continue

                break
            except ValueError as e:
                print(f"  ⚠️  Error: {e}")
                continue

        # Ejecutar la jugada
        obs_raw, reward, terminated, truncated, info = env.step(carta_id)

        if reward != 0:
            print(f"\n  💰 Recompensa: {reward:+.1f}")

        if terminated or truncated:
            partida_terminada = True

    # --- Fin de la partida ---
    print("\n" + "═" * 60)
    print("  🏁  PARTIDA TERMINADA  🏁")
    print("═" * 60)

    punt_final = env._puntuacion_historica
    ranking = sorted(range(4), key=lambda i: punt_final[i])

    print("\n  📊 Clasificación final:")
    posiciones = ["🥇 1º", "🥈 2º", "🥉 3º", "💀 4º"]
    for pos, jug_idx in enumerate(ranking):
        rol = "👤 VOS" if jug_idx == humano_idx else (
            "🤖 MODELO" if jug_idx == modelo_idx else "🤖 Bot")
        print(
            f"     {posiciones[pos]}: Jugador {jug_idx} [{rol}] — {punt_final[jug_idx]} pts")

    if ranking[0] == humano_idx:
        print("\n  🎉 ¡GANASTE! ¡Felicidades!")
    elif ranking[0] == modelo_idx:
        print("\n  🤖 El modelo RL ganó. ¡Seguí entrenando!")
    else:
        print(
            f"\n  🤖 Un bot ganó. El modelo quedó en posición {ranking.index(modelo_idx) + 1}º.")

    env.close()


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Partida interactiva de Corazones: humano vs modelo RL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python jugar_contra_modelo.py
  python jugar_contra_modelo.py --modelo modelos_historicos/v2/modelo_final
  python jugar_contra_modelo.py --humano-idx 0 --modelo-idx 2
        """,
    )
    parser.add_argument(
        "--modelo", type=str,
        default="modelos_historicos/v2/modelo_final",
        help="Ruta al modelo .zip"
    )
    parser.add_argument(
        "--vecnorm", type=str, default=None,
        help="Ruta al VecNormalize .pkl (auto-detecta)"
    )
    parser.add_argument(
        "--humano-idx", type=int, default=0,
        help="Tu posición en la mesa (0-3, default=0)"
    )
    parser.add_argument(
        "--modelo-idx", type=int, default=1,
        help="Posición del modelo RL (0-3, default=1)"
    )

    args = parser.parse_args()

    # Auto-detectar VecNormalize
    vecnorm_path = args.vecnorm
    if vecnorm_path is None:
        for candidato in [
            "vecnormalize/v2_vecnorm_final.pkl",
            "vecnormalize/v2_vecnorm.pkl",
            "vecnormalize/vecnorm.pkl",
        ]:
            if os.path.exists(candidato):
                vecnorm_path = candidato
                break

    jugar_partida(
        modelo_path=args.modelo,
        vecnorm_path=vecnorm_path,
        humano_idx=args.humano_idx,
        modelo_idx=args.modelo_idx,
    )


if __name__ == "__main__":
    main()
