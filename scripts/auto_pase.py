"""
Auto-pase por ADB: el modelo elige las 3 cartas del pase y se TOCAN solas en la
pantalla del teléfono; luego se confirma y se leen las cartas recibidas.

Cierra el lazo visión→decisión→acción para la fase de pase. Para el resto (jugar
baza a baza) ver el copiloto `scripts/recomendador.py` (por ahora manual).

Uso:
  # en vivo por ADB (requiere `adb` en PATH y depuración USB):
  python scripts/auto_pase.py --modelo models/produccion --serial <SERIAL>

  # prueba en seco sobre un frame fijo, SIN tocar nada (imprime qué tocaría):
  python scripts/auto_pase.py --modelo models/produccion --frame calibracion/hearts_app/captura_real.png --seco

⚠ Auto-tocar una app puede violar sus términos de servicio. Úsalo bajo tu
responsabilidad, para fines personales/de investigación.
"""
from __future__ import annotations

# --- bootstrap path ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
# --- fin bootstrap ---
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse
from typing import List

# banner dato -> dirección que entiende el Recomendador
_DIR_BANNER = {"izquierda": "izquierda", "derecha": "derecha", "enfrente": "frente"}


def _construir_recomendar(modelo: str):
    """Devuelve un callback (mano_ids, direccion) -> [3 ids] usando el modelo."""
    from src.dominio.carta import Carta
    from scripts.recomendador import Recomendador

    rec = Recomendador(modelo)
    if not rec.con_pase:
        raise SystemExit("El modelo no tiene fase de pase (obs<228). Usa un v12+/v13.")

    def recomendar(mano_ids: List[int], direccion: str) -> List[int]:
        rec.reset_mano([Carta._TODAS[i] for i in mano_ids])
        cartas = rec.recomendar_pase(direccion)
        return [c.id for c in cartas]

    return recomendar


class _ClienteSeco:
    """ClienteADB de mentira: lee un frame fijo y SOLO imprime los taps."""

    def __init__(self, frame: str):
        import cv2
        self._img = cv2.imread(frame)
        if self._img is None:
            raise SystemExit(f"No se pudo leer el frame: {frame}")

    def captura(self):
        return self._img

    def tap(self, x: int, y: int) -> None:
        print(f"   [SECO] tap ({x}, {y})")


def main() -> None:
    p = argparse.ArgumentParser(description="Auto-pase por ADB con el modelo.")
    p.add_argument("--modelo", required=True)
    p.add_argument("--serial", default=None)
    p.add_argument("--adb", default="adb")
    p.add_argument("--frame", default=None,
                   help="Frame fijo para prueba en seco (sin ADB).")
    p.add_argument("--seco", action="store_true",
                   help="No toca nada: solo imprime qué tocaría (requiere --frame).")
    p.add_argument("--regiones", default="calibracion/hearts_app/regiones.json")
    p.add_argument("--banners", default="calibracion/hearts_app/banners")
    p.add_argument("--completas", default="calibracion/hearts_app/cartas_completas")
    p.add_argument("--confirmar-tpl", default=None,
                   help="PNG del botón círculo-check para localizarlo por correlación.")
    args = p.parse_args()

    from src.captura.adb import ClienteADB
    from src.captura.auto_pase import ConfigAutoPase, ControladorPase
    from src.captura.vision_cartas import ReconocedorPlantilla
    from src.captura.vision_hearts import BannerClasificador, Regiones

    reg = Regiones.cargar(args.regiones)
    clf = BannerClasificador(args.banners)
    rec_mano = ReconocedorPlantilla(args.completas)
    recomendar = _construir_recomendar(args.modelo)

    if args.seco or args.frame:
        if not args.frame:
            raise SystemExit("--seco requiere --frame.")
        cliente = _ClienteSeco(args.frame)
    else:
        cliente = ClienteADB(serial=args.serial, adb=args.adb)
        if not cliente.dispositivos():
            raise SystemExit("No hay dispositivos ADB (revisa `adb devices`).")

    ctrl = ControladorPase(
        cliente=cliente, regiones=reg, banner_clf=clf, reconocedor_mano=rec_mano,
        recomendar=recomendar, config=ConfigAutoPase(),
        plantilla_confirmar=args.confirmar_tpl,
    )
    res = ctrl.ejecutar()
    print("\n== RESULTADO ==")
    print(f"  dirección : {res.direccion}")
    print(f"  confirmado: {res.confirmado}")
    if res.nota:
        print(f"  nota      : {res.nota}")


if __name__ == "__main__":
    main()
