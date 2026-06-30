"""
Auto-pase por ADB: el modelo elige las 3 cartas del pase y se TOCAN solas en la
pantalla del teléfono; luego se confirma y se leen las cartas recibidas.

Cierra el lazo visión→decisión→acción para la fase de pase. Para el resto (jugar
baza a baza) ver el copiloto `scripts/recomendador.py` (por ahora manual).

Uso:
  # en vivo por ADB — flujo COMPLETO (selecciona + confirma):
  python scripts/auto_pase.py --modelo models/produccion --serial <SERIAL>

  # en vivo por ADB — SIN confirmar (selecciona las 3 cartas y para):
  python scripts/auto_pase.py --modelo models/produccion --serial <SERIAL> --sin-confirmar

  # prueba en seco sobre un frame fijo, SIN tocar nada (imprime qué tocaría):
  python scripts/auto_pase.py --modelo models/produccion --frame calibracion/hearts_app/captura_real.png --seco

⚠ Auto-tocar una app puede violar sus términos de servicio. Úsalo bajo tu
responsabilidad, para fines personales/de investigación.
"""
from __future__ import annotations
from typing import List
import argparse

# --- bootstrap path ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
# --- fin bootstrap ---
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# banner dato -> dirección que entiende el Recomendador
_DIR_BANNER = {"izquierda": "izquierda",
               "derecha": "derecha", "enfrente": "frente"}


def _construir_recomendar(modelo: str):
    """Devuelve un callback (mano_ids, direccion) -> [3 ids] usando el modelo."""
    from src.dominio.carta import Carta
    from scripts.recomendador import Recomendador

    rec = Recomendador(modelo)
    if not rec.con_pase:
        raise SystemExit(
            "El modelo no tiene fase de pase (obs<228). Usa un v12+/v13.")

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
    p.add_argument("--sin-confirmar", action="store_true",
                   help="Selecciona las 3 cartas y verifica la zona de pases, "
                        "pero NO toca el botón de confirmar. "
                        "Útil para probar la selección sin comprometerse. "
                        "(Requiere --serial, usa ADB real.)")
    p.add_argument(
        "--regiones", default="calibracion/hearts_app/regiones.json")
    p.add_argument("--banners", default="calibracion/hearts_app/banners")
    p.add_argument("--umbral-banner", type=float, default=None,
                   help="Umbral de distancia euclidea para clasificar el banner "
                        "(default: 1.5, reduce a 0.8 si hay muchos 'desconocido').")
    p.add_argument(
        "--completas", default="calibracion/hearts_app/cartas_completas")
    p.add_argument("--confirmar-tpl", default=None,
                   help="PNG del botón círculo-check para localizarlo por correlación. "
                        "Por defecto busca calibracion/hearts_app/confirmar.png.")
    p.add_argument("--debug", default=None,
                   help="Guarda screenshots de cada paso en el directorio indicado. "
                        "También guarda recortes del banner en 'banners/' dentro del "
                        "directorio de debug para crear plantillas especificas del "
                        "dispositivo.")
    args = p.parse_args()

    # --- default inteligente: prefiere confirmar_check.png (check blanco) ---
    _confirmar_tpl = args.confirmar_tpl
    if _confirmar_tpl is None:
        for cand in ["calibracion/hearts_app/confirmar_check.png",
                     "calibracion/hearts_app/confirmar.png"]:
            if _Path(cand).is_file():
                _confirmar_tpl = str(_Path(cand))
                break

    from src.captura.adb import ClienteADB
    from src.captura.auto_pase import ConfigAutoPase, ControladorPase
    from src.captura.vision_cartas import ReconocedorPlantilla
    from src.captura.vision_hearts import BannerClasificador, Regiones

    reg = Regiones.cargar(args.regiones)
    umbral_banner = args.umbral_banner or 1.5
    clf = BannerClasificador(args.banners, umbral=umbral_banner)
    rec_mano = ReconocedorPlantilla(args.completas)
    recomendar = _construir_recomendar(args.modelo)

    if args.seco or args.frame:
        if not args.frame:
            raise SystemExit("--seco requiere --frame.")
        if args.sin_confirmar:
            raise SystemExit(
                "--sin-confirmar requiere ADB real (--serial). "
                "Usa --seco para prueba offline en seco.")
        cliente = _ClienteSeco(args.frame)
    elif args.sin_confirmar:
        # Modo ADB real, pero sin confirmar
        cliente = ClienteADB(serial=args.serial, adb=args.adb)
        if not cliente.dispositivos():
            raise SystemExit("No hay dispositivos ADB (revisa `adb devices`).")
    else:
        cliente = ClienteADB(serial=args.serial, adb=args.adb)
        if not cliente.dispositivos():
            raise SystemExit("No hay dispositivos ADB (revisa `adb devices`).")

    cfg = ConfigAutoPase()
    if args.debug:
        cfg.debug_dir = args.debug
    ctrl = ControladorPase(
        cliente=cliente, regiones=reg, banner_clf=clf, reconocedor_mano=rec_mano,
        recomendar=recomendar, config=cfg,
        plantilla_confirmar=_confirmar_tpl,
    )
    res = ctrl.ejecutar(confirmar=not args.sin_confirmar)
    print("\n== RESULTADO ==")
    print(f"  dirección : {res.direccion}")
    print(f"  confirmado: {res.confirmado}")
    if res.nota:
        print(f"  nota      : {res.nota}")


if __name__ == "__main__":
    main()
