"""
Asesor interactivo de cartas para Corazones.

Carga el modelo RL entrenado (MaskablePPO + VecNormalize) y recomienda
qué carta jugar en tu turno. Útil para recibir consejos durante una
partida real en otra aplicación de Corazones.

Modos de uso:
  1. Modo interactivo (línea de comandos):
     python asesor_carta.py
     → Te pregunta tus cartas, la mesa, y te recomienda la mejor jugada.

  2. Modo rápido (flags):
     python asesor_carta.py --mano "A♥ 2♣ 3♣ ..." --mesa "K♠"
     → Recomendación inmediata.

Formato de cartas:
  Valor + símbolo de palo. Ejemplos: "A♥" "2♣" "K♠" "Q♦" "10♥" "J♣"
  Palos: ♣ Tréboles  ♦ Diamantes  ♠ Picas  ♥ Corazones
  Valores: 2-10, J, Q, K, A
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# Mapeos de palos y valores para parseo de texto
NOMBRES_PALOS: Dict[str, int] = {
    "♣": 0, "♦": 1, "♠": 2, "♥": 3,
    # Alternativas sin símbolo Unicode (teclado común)
    "c": 0, "d": 1, "s": 2, "h": 3,
    "C": 0, "D": 1, "S": 2, "H": 3,
    "t": 0, "T": 0,  # Tréboles
    "p": 2, "P": 2,  # Picas
}

# Display: usamos letras (c/d/s/h) para que el usuario pueda escribirlas
_PALO_NOMBRE: Dict[int, str] = {0: "c", 1: "d", 2: "s", 3: "h"}

_VALOR_NOMBRE: Dict[int, str] = {
    2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8",
    9: "9", 10: "10", 11: "J", 12: "Q", 13: "K", 14: "A",
}

# Mapeo inverso para parseo
_NOMBRE_VALOR: Dict[str, int] = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
    "9": 9, "10": 10,
    "j": 11, "J": 11, "q": 12, "Q": 12, "k": 13, "K": 13,
    "a": 14, "A": 14,
}


# ----------------------------------------------------------------
# Parseo de cartas
# ----------------------------------------------------------------

def parsear_carta(texto: str) -> int:
    """Convierte un nombre de carta (ej. "A♥", "2♣") a su ID (0-51).

    Args:
        texto: Representación de la carta (valor + palo).

    Returns:
        Índice de la carta (0-51) donde id = palo*13 + (valor-2).

    Raises:
        ValueError: Si el formato no es reconocible.
    """
    texto = texto.strip()
    if not texto:
        raise ValueError("Cadena de carta vacía")

    # Detectar palo (último carácter, puede ser Unicode)
    palo_char = texto[-1]
    if palo_char not in NOMBRES_PALOS:
        raise ValueError(
            f"Símbolo de palo desconocido: '{palo_char}'. "
            f"Usar ♣♦♠♥ o c/d/s/h."
        )
    palo = NOMBRES_PALOS[palo_char]

    valor_str = texto[:-1]
    if valor_str not in _NOMBRE_VALOR:
        raise ValueError(
            f"Valor desconocido: '{valor_str}'. "
            f"Usar 2-10, J, Q, K, A."
        )
    valor = _NOMBRE_VALOR[valor_str]

    return palo * 13 + (valor - 2)


def parsear_mano(
    texto: str,
    sep: Optional[str] = None,
    esperadas: Optional[int] = 13,
) -> List[int]:
    """Convierte una cadena de cartas separadas por espacios/comas a IDs.

    Args:
        texto: Lista de cartas, p.ej. "2♣ 3♣ A♥ K♠".
        sep: Separador opcional (por defecto, split por whitespace).
        esperadas: Cantidad esperada de cartas (None = no validar).

    Returns:
        Lista de IDs de carta.

    Raises:
        ValueError: Si esperadas no es None y la cantidad no coincide.
    """
    partes = texto.split(sep) if sep else texto.split()
    ids = [parsear_carta(p) for p in partes if p.strip()]

    if esperadas is not None and len(ids) != esperadas:
        raise ValueError(
            f"Se esperaban {esperadas} cartas, se recibieron {len(ids)}"
        )

    return ids


def cartas_a_ids(nombres: List[str]) -> List[int]:
    """Convierte una lista de nombres de carta a lista de IDs."""
    return [parsear_carta(n) for n in nombres]


def ids_a_nombres(ids: List[int]) -> List[str]:
    """Convierte una lista de IDs de carta a nombres legibles."""
    resultado: List[str] = []
    for cid in ids:
        palo = cid // 13
        valor = (cid % 13) + 2
        resultado.append(f"{_VALOR_NOMBRE[valor]}{_PALO_NOMBRE[palo]}")
    return resultado


# ----------------------------------------------------------------
# Cómputo de jugadas legales (sin dependencia del motor)
# ----------------------------------------------------------------

# Arrays precomputados para O(1) acceso a palo/valor/puntos
_PALOS: List[int] = [cid // 13 for cid in range(52)]
_VALORES: List[int] = [(cid % 13) + 2 for cid in range(52)]


def compute_legales(
    mano: List[int],
    mesa_ids: List[int],
    corazones_rotos: bool,
    es_primera_baza: bool,
) -> List[int]:
    """Calcula las jugadas legales dadas las cartas en mano y el estado.

    Implementa las 4 reglas de Corazones:
      1. Seguir el palo si es posible.
      2. Si no se puede seguir el palo, cualquier carta es legal.
      3. No se puede liderar corazones si no están rotos (salvo solo corazones).
      4. En la primera baza no se pueden jugar corazones ni la Dama de Picas.

    Args:
        mano: Lista de IDs de cartas en la mano del jugador.
        mesa_ids: IDs de cartas ya jugadas en esta baza (vacío si lidera).
        corazones_rotos: True si ya se han roto corazones.
        es_primera_baza: True si es la primera baza de la mano.

    Returns:
        Lista de IDs de cartas legales.
    """
    if not mano:
        return []

    # Caso 1: Liderar (mesa vacía)
    if not mesa_ids:
        legales: List[int] = []
        solo_corazones = all(_PALOS[cid] == 3 for cid in mano)

        # Regla: 2♣ obligatorio en primera baza si está en mano
        if es_primera_baza and 0 in mano:
            return [0]

        for cid in mano:
            palo = _PALOS[cid]
            valor = _VALORES[cid]

            # Primera baza: no corazones, no dama de picas
            if es_primera_baza:
                if palo == 3 or (palo == 2 and valor == 12):
                    continue

            # Sin corazones rotos: no se puede liderar corazones
            if not corazones_rotos and not solo_corazones:
                if palo == 3:
                    continue

            legales.append(cid)

        return legales if legales else mano  # fallback

    # Caso 2: Seguir el palo
    palo_salida = _PALOS[mesa_ids[0]]
    cartas_del_palo = [cid for cid in mano if _PALOS[cid] == palo_salida]

    if cartas_del_palo:
        # Restricción adicional: primera baza, sin corazones ni dama de picas
        if es_primera_baza:
            return [
                cid for cid in cartas_del_palo
                if not (_PALOS[cid] == 3 or (_PALOS[cid] == 2 and _VALORES[cid] == 12))
            ] or cartas_del_palo
        return cartas_del_palo

    # Void del palo de salida: puede jugar cualquier carta
    legales = list(mano)
    if es_primera_baza:
        legales = [
            cid for cid in legales
            if not (_PALOS[cid] == 3 or (_PALOS[cid] == 2 and _VALORES[cid] == 12))
        ]
    return legales if legales else list(mano)


# ----------------------------------------------------------------
# Construcción de la observación (190 dimensiones, v5)
# ----------------------------------------------------------------

def _pozo_viable(
    mano_ids: List[int],
    corazones_rotos: bool,
    puntaje_agente: int,
) -> bool:
    """Determina si es viable intentar shooting the moon.

    Args:
        mano_ids: IDs de cartas en mano.
        corazones_rotos: Si los corazones están rotos.
        puntaje_agente: Puntaje histórico del agente.

    Returns:
        True si el pozo es viable.
    """
    if corazones_rotos:
        return False

    corazones_en_mano = sum(1 for cid in mano_ids if cid // 13 == 3)
    if corazones_en_mano < 6:
        return False

    corazones_altos = sum(
        1 for cid in mano_ids
        if cid // 13 == 3 and (cid % 13) + 2 >= 11
    )
    if corazones_altos < 3:
        return False

    if puntaje_agente >= 80:
        return False

    return True


def _debo_arriesgar(
    puntajes_historicos: List[int],
    agente_idx: int,
) -> bool:
    """Determina si el agente está tan atrás que debe arriesgarse."""
    if puntajes_historicos[agente_idx] <= 75:
        return False
    for i in range(4):
        if i != agente_idx and puntajes_historicos[i] < 30:
            return True
    return False


def _puedo_alimentar(
    puntajes_historicos: List[int],
    agente_idx: int,
) -> bool:
    """Determina si conviene darle puntos a un rival para que pierda."""
    if puntajes_historicos[agente_idx] >= 70:
        return False
    for i in range(4):
        if i != agente_idx and puntajes_historicos[i] > 85:
            return True
    return False


def construir_observacion_parcial(
    mano_ids: List[int],
    mesa_ids: List[int],
    cementerio_ids: List[int],
    vacios_por_jugador: List[set],
    puntajes_historicos: List[int],
    puntos_mano_actual: List[int],
    corazones_rotos: bool,
    dama_picas_en: Optional[int],
    agente_idx: int = 0,
) -> np.ndarray:
    """Construye el vector de observación de 190 dimensiones (v5).

    Bloques:
        [0:52]    Mano del agente (one-hot)
        [52:104]  Mesa actual (one-hot)
        [104:156] Cementerio (one-hot)
        [156:172] Vacíos conocidos (4 jugadores × 4 palos)
        [172:176] Puntajes históricos (normalizados /100)
        [176:180] Puntos de la mano actual (normalizados /26)
        [180]     Corazones rotos (0.0 o 1.0)
        [181]     Posición en la baza actual
        [182:187] Rastreador de la Dama de Picas (one-hot, 5 estados)
        [187]     pozo_viable
        [188]     debo_arriesgar
        [189]     puedo_alimentar

    Args:
        mano_ids: IDs de cartas en mano del agente.
        mesa_ids: IDs de cartas en la mesa (baza actual).
        cementerio_ids: IDs de cartas ya jugadas en bazas anteriores.
        vacios_por_jugador: Conjuntos de palos donde cada jugador está void.
        puntajes_historicos: Puntuación acumulada de cada jugador.
        puntos_mano_actual: Puntos acumulados en la mano actual por jugador.
        corazones_rotos: Si los corazones están rotos.
        dama_picas_en: Índice del jugador que tiene la Dama de Picas (None=oculta).
        agente_idx: Índice del jugador humano (0 por defecto).

    Returns:
        Array np.float32 de shape (190,).
    """
    obs = np.zeros(190, dtype=np.float32)
    a = agente_idx

    # [0:52] Mano
    for cid in mano_ids:
        obs[cid] = 1.0

    # [52:104] Mesa
    for cid in mesa_ids:
        obs[52 + cid] = 1.0

    # [104:156] Cementerio
    for cid in cementerio_ids:
        obs[104 + cid] = 1.0

    # [156:172] Vacíos (relativos al agente)
    for jug_idx in range(4):
        rel = (jug_idx - a) % 4
        for palo in vacios_por_jugador[jug_idx]:
            obs[156 + rel * 4 + palo] = 1.0

    # [172:176] Puntajes históricos (normalizados /100)
    for jug_idx in range(4):
        rel = (jug_idx - a) % 4
        obs[172 + rel] = min(puntajes_historicos[jug_idx] / 100.0, 1.0)

    # [176:180] Puntos de la mano actual (normalizados /26)
    for jug_idx in range(4):
        rel = (jug_idx - a) % 4
        obs[176 + rel] = min(puntos_mano_actual[jug_idx] / 26.0, 1.0)

    # [180] Corazones rotos
    obs[180] = 1.0 if corazones_rotos else 0.0

    # [181] Posición en la baza
    posiciones = {0: 0.0, 1: 0.33, 2: 0.66, 3: 1.0}
    obs[181] = posiciones.get(len(mesa_ids), 0.0)

    # [182:187] Dama de Picas
    if dama_picas_en is None:
        obs[182] = 1.0  # Oculta
    else:
        rel = (dama_picas_en - a) % 4
        obs[183 + rel] = 1.0

    # --- [187:190] Features estratégicas v5 ---
    obs[187] = 1.0 if _pozo_viable(
        mano_ids, corazones_rotos, puntajes_historicos[a]) else 0.0
    obs[188] = 1.0 if _debo_arriesgar(puntajes_historicos, a) else 0.0
    obs[189] = 1.0 if _puedo_alimentar(puntajes_historicos, a) else 0.0

    return obs


# ----------------------------------------------------------------
# Normalización con VecNormalize
# ----------------------------------------------------------------

def normalizar_observacion(obs: np.ndarray, vecnorm_path: Optional[str]) -> np.ndarray:
    """Aplica normalización VecNormalize a la observación.

    Args:
        obs: Vector crudo de shape (187,).
        vecnorm_path: Ruta al archivo .pkl de VecNormalize.

    Returns:
        Observación normalizada (o cruda si no hay VecNormalize).
    """
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


# ----------------------------------------------------------------
# Carga del modelo
# ----------------------------------------------------------------

def cargar_modelo(
    ruta_modelo: str,
    vecnorm_path: Optional[str] = None,
) -> Tuple[object, Optional[str]]:
    """Carga el modelo MaskablePPO y retorna (modelo, vecnorm_path).

    Args:
        ruta_modelo: Ruta al modelo .zip.
        vecnorm_path: Ruta al VecNormalize .pkl (auto-detecta si es None).

    Returns:
        Tupla (modelo, vecnorm_path).
    """
    if not ruta_modelo.endswith(".zip"):
        ruta_modelo += ".zip"

    if not os.path.exists(ruta_modelo):
        print(f"❌ ERROR: No se encuentra el modelo en {ruta_modelo}")
        sys.exit(1)

    # Auto-detectar VecNormalize
    if vecnorm_path is None:
        for candidato in [
            "vecnormalize/v2_vecnorm_final.pkl",
            "vecnormalize/v2_vecnorm.pkl",
            "vecnormalize/vecnorm.pkl",
        ]:
            if os.path.exists(candidato):
                vecnorm_path = candidato
                break

    from sb3_contrib import MaskablePPO

    print(f"📦 Cargando modelo: {ruta_modelo}")
    modelo = MaskablePPO.load(ruta_modelo, device="cpu")

    if vecnorm_path and os.path.exists(vecnorm_path):
        print(f"📊 VecNormalize: {vecnorm_path}")
    else:
        print("⚠️  Sin VecNormalize (la recomendación puede ser menos precisa)")

    return modelo, vecnorm_path


# ----------------------------------------------------------------
# Modo interactivo
# ----------------------------------------------------------------

def _preguntar_si_no(texto: str) -> bool:
    """Pregunta sí/no y retorna bool."""
    resp = input(f"{texto} [s/n]: ").strip().lower()
    return resp in ("s", "si", "sí", "y", "yes")


def _pedir_entero(texto: str, minimo: int = 0, maximo: int = 200) -> int:
    """Pide un entero al usuario."""
    while True:
        try:
            val = int(input(f"{texto}: ").strip())
            if minimo <= val <= maximo:
                return val
            print(f"  ⚠️  Valor fuera de rango [{minimo}-{maximo}]")
        except ValueError:
            print("  ⚠️  Debe ser un número entero.")


def _pedir_cartas(mensaje: str, esperadas: Optional[int] = None) -> List[int]:
    """Pide cartas al usuario y retorna IDs.

    Args:
        mensaje: Texto a mostrar.
        esperadas: Cantidad esperada (None = cualquier cantidad).

    Returns:
        Lista de IDs de carta.
    """
    while True:
        try:
            texto = input(f"{mensaje}: ").strip()
            if not texto and esperadas != 13:
                return []
            ids = [parsear_carta(p) for p in texto.split() if p.strip()]
            if esperadas is not None and len(ids) != esperadas:
                print(
                    f"  ⚠️  Se esperaban {esperadas} cartas, se recibieron {len(ids)}")
                continue
            return ids
        except ValueError as e:
            print(f"  ⚠️  {e}")


def _pedir_estado_juego() -> dict:
    """Solicita interactivamente el estado completo del juego.

    Returns:
        Diccionario con todos los campos necesarios para la observación.
    """
    print("\n" + "=" * 60)
    print("  🃏 ASESOR DE CARTAS — CORAZONES")
    print("=" * 60)
    print("Formato: valor + palo. Ej: A♥ 2♣ K♠ Q♦ 10♥ J♣")
    print("Palos: ♣=Tréboles ♦=Diamantes ♠=Picas ♥=Corazones")
    print("También podés usar c/d/s/h (ej: Ah = A♥, 2c = 2♣)")
    print()

    # 1. Mano
    mano = _pedir_cartas("🎴 Tu mano (13 cartas)", esperadas=13)

    # 2. Mesa
    mesa = _pedir_cartas("🃏 Cartas en la mesa (baza actual, vacío si liderás)")

    # 3. Posición del jugador
    print("\n📌 Tu posición en la mesa:")
    print("  0 = primer jugador en jugar | 1 | 2 | 3 = último")
    agente_idx = _pedir_entero("Tu posición (0-3)", 0, 3)

    # 4. Corazones rotos
    corazones_rotos = _preguntar_si_no("💔 ¿Corazones rotos?")

    # 5. Primera baza
    es_primera_baza = _preguntar_si_no("🔢 ¿Es la primera baza de la mano?")

    # 6. Puntajes históricos
    print("\n📊 Puntajes acumulados en la partida:")
    puntajes = []
    for i in range(4):
        puntajes.append(_pedir_entero(f"  Jugador {i}", 0, 200))
    # Normalizar: el humano es agente_idx
    puntajes_reordenados = [puntajes[agente_idx]]
    for i in range(1, 4):
        puntajes_reordenados.append(puntajes[(agente_idx + i) % 4])
    # Pero necesitamos los absolutos, no relativos. Guardamos como vienen.
    # La función construir_observacion_parcial los ordena internamente.

    # 7. Puntos de la mano actual
    print("\n🎯 Puntos acumulados en ESTA mano:")
    puntos_mano = []
    for i in range(4):
        puntos_mano.append(_pedir_entero(f"  Jugador {i}", 0, 26))

    # 8. Dama de Picas
    print("\n👑 Dama de Picas (Q♠):")
    print("  -1 = aún no apareció / desconocido")
    for i in range(4):
        print(f"  {i} = la tiene el Jugador {i}")
    dama = _pedir_entero("¿Quién tiene la Q♠?", -1, 3)

    # 9. Cementerio
    if _preguntar_si_no("\n🪦 ¿Querés ingresar cartas del cementerio (bazas anteriores)?"):
        cementerio = _pedir_cartas("Cartas del cementerio")
    else:
        cementerio = []

    # 10. Vacíos
    vacios: List[set] = [set(), set(), set(), set()]
    if _preguntar_si_no("\n📭 ¿Querés ingresar vacíos conocidos (palos donde jugadores no siguieron)?"):
        for jug in range(4):
            entrada = input(
                f"  Vacíos Jugador {jug} (♣♦♠♥ o c/d/s/h, vacío si ninguno): ").strip()
            if entrada:
                for char in entrada:
                    if char in NOMBRES_PALOS:
                        vacios[jug].add(NOMBRES_PALOS[char])

    return {
        "mano_ids": mano,
        "mesa_ids": mesa,
        "cementerio_ids": cementerio,
        "vacios_por_jugador": vacios,
        "puntajes_historicos": puntajes,
        "puntos_mano_actual": puntos_mano,
        "corazones_rotos": corazones_rotos,
        "es_primera_baza": es_primera_baza,
        "dama_picas_en": dama if dama >= 0 else None,
        "agente_idx": agente_idx,
    }


# ----------------------------------------------------------------
# Recomendación
# ----------------------------------------------------------------

def recomendar_carta(
    modelo,
    estado: dict,
    vecnorm_path: Optional[str],
    mostrar_top: int = 3,
) -> int:
    """Recomienda la mejor carta para jugar.

    Args:
        modelo: Modelo MaskablePPO cargado.
        estado: Diccionario con el estado del juego.
        vecnorm_path: Ruta al VecNormalize.
        mostrar_top: Cuántas opciones mostrar.

    Returns:
        ID de la carta recomendada.
    """
    # Calcular legales
    legales = compute_legales(
        mano=estado["mano_ids"],
        mesa_ids=estado["mesa_ids"],
        corazones_rotos=estado["corazones_rotos"],
        es_primera_baza=estado["es_primera_baza"],
    )

    if not legales:
        print("⚠️  No hay jugadas legales detectadas.")
        return -1

    # Construir observación
    obs = construir_observacion_parcial(
        mano_ids=estado["mano_ids"],
        mesa_ids=estado["mesa_ids"],
        cementerio_ids=estado["cementerio_ids"],
        vacios_por_jugador=estado["vacios_por_jugador"],
        puntajes_historicos=estado["puntajes_historicos"],
        puntos_mano_actual=estado["puntos_mano_actual"],
        corazones_rotos=estado["corazones_rotos"],
        dama_picas_en=estado["dama_picas_en"],
        agente_idx=estado["agente_idx"],
    )

    # Normalizar
    obs_norm = normalizar_observacion(obs, vecnorm_path)

    # Action mask
    mask = np.zeros(52, dtype=np.bool_)
    for lid in legales:
        mask[lid] = True

    # Predecir
    action, _states = modelo.predict(
        obs_norm, action_masks=mask, deterministic=True)
    best_action = int(action)

    # Mostrar top-N recomendaciones
    print("\n" + "─" * 50)
    print("  🎯 RECOMENDACIÓN DEL MODELO")
    print("─" * 50)

    nombres_legales = ids_a_nombres(legales)
    mejor_nombre = ids_a_nombres([best_action])[0]

    print(f"  ⭐ Mejor jugada: {mejor_nombre}  (ID={best_action})")
    print()
    print(f"  📋 Jugadas legales ({len(legales)}):")
    # Mostrar legales ordenadas por palo
    legales_ordenadas = sorted(
        legales, key=lambda cid: (_PALOS[cid], _VALORES[cid]))
    for cid in legales_ordenadas:
        nombre = ids_a_nombres([cid])[0]
        marcador = " ← RECOMENDADA" if cid == best_action else ""
        print(f"     {nombre}{marcador}")
    print()

    return best_action


# ----------------------------------------------------------------
# CLI principal
# ----------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Asesor de cartas para Corazones — Recomendación RL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python asesor_carta.py                           # Modo interactivo
  python asesor_carta.py --mano "A♥ 2♣ 3♣ ..."     # Recomendación rápida
  python asesor_carta.py --modelo modelos_historicos/v2/modelo_final
        """,
    )
    parser.add_argument(
        "--modelo", type=str,
        default="modelos_historicos/v2/modelo_final",
        help="Ruta al modelo .zip (sin extensión)"
    )
    parser.add_argument(
        "--vecnorm", type=str, default=None,
        help="Ruta al VecNormalize .pkl (auto-detecta si no se especifica)"
    )
    parser.add_argument(
        "--mano", type=str, default=None,
        help="Tus 13 cartas (ej: 'A♥ 2♣ 3♣ K♠ ...')"
    )
    parser.add_argument(
        "--mesa", type=str, default=None,
        help="Cartas en la mesa (vacío si liderás)"
    )
    parser.add_argument(
        "--corazones-rotos", action="store_true", default=False,
        help="Si los corazones ya están rotos"
    )
    parser.add_argument(
        "--primera-baza", action="store_true", default=False,
        help="Si es la primera baza de la mano"
    )

    args = parser.parse_args()

    # Cargar modelo
    modelo, vecnorm_path = cargar_modelo(args.modelo, args.vecnorm)

    if args.mano:
        # Modo rápido por flags
        mano_ids = parsear_mano(args.mano)
        mesa_ids = parsear_mano(args.mesa) if args.mesa else []

        estado = {
            "mano_ids": mano_ids,
            "mesa_ids": mesa_ids,
            "cementerio_ids": [],
            "vacios_por_jugador": [set(), set(), set(), set()],
            "puntajes_historicos": [0, 0, 0, 0],
            "puntos_mano_actual": [0, 0, 0, 0],
            "corazones_rotos": args.corazones_rotos,
            "es_primera_baza": args.primera_baza,
            "dama_picas_en": None,
            "agente_idx": 0,
        }
        recomendar_carta(modelo, estado, vecnorm_path)
    else:
        # Modo interactivo completo
        estado = _pedir_estado_juego()
        recomendar_carta(modelo, estado, vecnorm_path)


if __name__ == "__main__":
    main()
