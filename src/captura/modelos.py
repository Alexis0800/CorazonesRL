"""
DTOs persistibles del módulo de captura y helpers de (de)serialización.

Una partida real capturada se guarda como UNA línea JSON (`RegistroPartida`)
en un fichero `.jsonl`. El formato es la **fuente de verdad** de lo observado:
guarda, por mano, la mano inicial del agente, su pase y TODAS las jugadas en
orden global (de los 4 asientos). Como cada asiento juega exactamente sus 13
cartas, las manos de los rivales se pueden reconstruir desde las jugadas
(ver `replay.py`), sin necesidad de espiarlas.

Las cartas se persisten como `carta.id` (0-51) por compacidad y para evitar
ambigüedad. Los helpers `carta_a_str`/`str_a_carta_id` son solo para I/O humana.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional

from src.dominio.carta import Carta

# --- Helpers carta <-> texto (para entrada/salida humana) ---
_PALO_A_LETRA = {0: "T", 1: "D", 2: "P", 3: "C"}  # Trébol, Diamante, Pica, Corazón
_VALOR_A_STR = {11: "J", 12: "Q", 13: "K", 14: "A"}
_LETRA_A_PALO = {
    "T": 0, "♣": 0,
    "D": 1, "♦": 1,
    "P": 2, "S": 2, "♠": 2,
    "C": 3, "H": 3, "♥": 3,
}
_STR_A_VALOR = {"J": 11, "Q": 12, "K": 13, "A": 14}


def carta_a_str(carta_id: int) -> str:
    """`23` -> `'QP'` (Q de picas). Inverso de `str_a_carta_id`."""
    c = Carta._TODAS[carta_id]
    v = _VALOR_A_STR.get(c.valor, str(c.valor))
    return f"{v}{_PALO_A_LETRA[c.palo]}"


def str_a_carta_id(texto: str) -> int:
    """`'10C'`, `'AP'`, `'Q♠'` -> id 0-51. Acepta símbolos y letras EN/ES."""
    s = texto.strip().upper()
    if not s:
        raise ValueError("Carta vacía.")
    palo_ch = s[-1]
    if palo_ch not in _LETRA_A_PALO:
        raise ValueError(f"Palo desconocido en '{texto}'.")
    palo = _LETRA_A_PALO[palo_ch]
    val_str = s[:-1]
    if val_str in _STR_A_VALOR:
        valor = _STR_A_VALOR[val_str]
    else:
        try:
            valor = int(val_str)
        except ValueError:
            raise ValueError(f"Valor inválido en '{texto}'.")
        if not (2 <= valor <= 10):
            raise ValueError(f"Valor fuera de rango en '{texto}'.")
    return Carta(palo, valor).id


@dataclass
class Jugada:
    """Una carta jugada por un asiento, en orden global dentro de la mano."""
    asiento: int       # 0-3 (absoluto)
    carta_id: int      # 0-51
    baza: int          # 1-13


@dataclass
class RegistroMano:
    numero_mano: int                       # número de mano dentro de la partida
    direccion_pase: Optional[str]          # izquierda/derecha/enfrente/None
    mano_inicial_agente: List[int]         # 13 ids (PRE-pase)
    pase_dado: List[int] = field(default_factory=list)      # 3 ids (o vacío)
    pase_recibido: List[int] = field(default_factory=list)  # 3 ids (opcional)
    jugadas: List[Jugada] = field(default_factory=list)     # jugadas REALES en orden
    puntuacion_mano: List[int] = field(default_factory=list)  # 4 (por asiento)
    # --- "se llevará el resto" (concesión): la ronda termina antes de las 13 bazas ---
    # `remate_asiento` se lleva TODAS las bazas restantes; `manos_restantes` son las
    # cartas reveladas de cada asiento en ese momento (4 listas). None/[] = mano normal.
    remate_asiento: Optional[int] = None
    manos_restantes: List[List[int]] = field(default_factory=list)


@dataclass
class RegistroPartida:
    partida_id: str
    timestamp: str
    asiento_agente: int                    # 0-3
    fuente: str                            # "manual" | "adb:<app>" | "test"
    manos: List[RegistroMano] = field(default_factory=list)
    marcador_final: List[int] = field(default_factory=list)   # 4
    ranking_final: List[int] = field(default_factory=list)     # asientos peor->mejor

    # --- (de)serialización ---
    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "RegistroPartida":
        manos = [
            RegistroMano(
                numero_mano=m["numero_mano"],
                direccion_pase=m["direccion_pase"],
                mano_inicial_agente=list(m["mano_inicial_agente"]),
                pase_dado=list(m.get("pase_dado", [])),
                pase_recibido=list(m.get("pase_recibido", [])),
                jugadas=[Jugada(**j) for j in m.get("jugadas", [])],
                puntuacion_mano=list(m.get("puntuacion_mano", [])),
                remate_asiento=m.get("remate_asiento"),
                manos_restantes=[list(h) for h in m.get("manos_restantes", [])],
            )
            for m in d.get("manos", [])
        ]
        return RegistroPartida(
            partida_id=d["partida_id"],
            timestamp=d["timestamp"],
            asiento_agente=d["asiento_agente"],
            fuente=d.get("fuente", "desconocida"),
            manos=manos,
            marcador_final=list(d.get("marcador_final", [])),
            ranking_final=list(d.get("ranking_final", [])),
        )


__all__ = [
    "Jugada", "RegistroMano", "RegistroPartida",
    "carta_a_str", "str_a_carta_id",
]
