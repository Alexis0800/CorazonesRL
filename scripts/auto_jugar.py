"""
Auto-juego de las BAZAS directo (sin pasar por la fase de pase). Útil para
ITERAR rápido la parte de juego: lo lanzas con la partida YA en la baza 1
(pase terminado, overlay de recibidas descartado, tablero visible).

Lee tu mano de la pantalla (o se la pasas con --mano) para sembrar el estado del
`Recomendador`, y luego juega las 13 bazas tocando las cartas que el modelo
recomienda ENTRE las que la app deja claras (jugables). Ver
`src/captura/auto_juego.py` para la lógica.

Uso:
  # leer la mano de la pantalla y jugar:
  python scripts/auto_jugar.py --modelo models/produccion/v10c_campeon --serial R52W90431FP --debug debug

  # sembrar la mano a mano (si el read de la mano falla), formato del recomendador:
  python scripts/auto_jugar.py --modelo ... --serial ... --mano "8T JP QP KP 6D 6C 9D 7C QD 4C KC AC 2T"

⚠ Auto-tocar una app puede violar sus términos de servicio. Úsalo bajo tu
responsabilidad, para fines personales/de investigación.
"""
from __future__ import annotations
import argparse
import time

# --- bootstrap path ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
# --- fin bootstrap ---
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _leer_mano_pantalla(cliente, reg, rec_mano, intentos: int = 6):
    """Lee la mano de la pantalla, reintentando hasta tener 13 cartas (o lo
    mejor tras `intentos`). Devuelve lista de carta_id (sin None)."""
    from src.captura.vision_hearts import leer_mano_posiciones
    mejor: list = []
    for _ in range(intentos):
        cartas = leer_mano_posiciones(cliente.captura(), reg, rec_mano)
        ids = [c.carta_id for c in cartas if c.carta_id is not None]
        # dedup conservando orden
        vistos, limpio = set(), []
        for i in ids:
            if i not in vistos:
                vistos.add(i)
                limpio.append(i)
        if len(limpio) > len(mejor):
            mejor = limpio
        if len(mejor) >= 13:
            return mejor[:13]
        time.sleep(0.2)
    return mejor


def main() -> None:
    p = argparse.ArgumentParser(
        description="Auto-juego de bazas directo (desde la baza 1).")
    p.add_argument("--modelo", required=True)
    p.add_argument("--serial", default=None)
    p.add_argument("--adb", default="adb")
    p.add_argument("--mano", default=None,
                   help="Siembra la mano a mano (formato 'AP 10C QT ...'); "
                        "si se omite, se lee de la pantalla.")
    p.add_argument("--confirmar-jugada", action="store_true",
                   help="Toca el botón de confirmar tras tocar la carta (si la "
                        "app lo exige también al jugar, no solo al pasar).")
    p.add_argument("--regiones", default="calibracion/hearts_app/regiones.json")
    p.add_argument("--banners", default="calibracion/hearts_app/banners")
    p.add_argument("--completas",
                   default="calibracion/hearts_app/cartas_completas")
    p.add_argument("--umbral-banner", type=float, default=1.5)
    p.add_argument("--umbral-brillo", type=float, default=0.85,
                   help="Factor (× el blanco más alto de la mano) del umbral de "
                        "'blanco real' para detectar cartas jugables (claras). "
                        "Súbelo si marca de más; bájalo si marca de menos.")
    p.add_argument("--debug", default=None)
    args = p.parse_args()

    from src.dominio.carta import Carta
    from src.captura.adb import ClienteADB
    from src.captura.auto_juego import ConfigAutoJuego, ControladorBazas
    from src.captura.modelos import carta_a_str
    from src.captura.vision_cartas import ReconocedorPlantilla
    from src.captura.vision_hearts import BannerClasificador, Regiones
    from scripts.recomendador import Recomendador, parse_cartas

    reg = Regiones.cargar(args.regiones)
    clf = BannerClasificador(args.banners, umbral=args.umbral_banner)
    rec_mano = ReconocedorPlantilla(args.completas)

    recomendador = Recomendador(args.modelo)
    print(f"Modelo: {args.modelo}  (obs {recomendador.obs_dim})")

    cliente = ClienteADB(serial=args.serial, adb=args.adb)
    if not cliente.dispositivos():
        raise SystemExit("No hay dispositivos ADB (revisa `adb devices`).")

    # ── sembrar la mano ──
    if args.mano:
        mano_ids = [c.id for c in parse_cartas(args.mano)]
        print(f"Mano (manual, {len(mano_ids)}): "
              + " ".join(carta_a_str(i) for i in mano_ids))
    else:
        print("Leyendo tu mano de la pantalla...")
        mano_ids = _leer_mano_pantalla(cliente, reg, rec_mano)
        print(f"Mano leída ({len(mano_ids)}/13): "
              + " ".join(carta_a_str(i) for i in mano_ids))
        if len(mano_ids) < 13:
            print("⚠ No leí 13 cartas. Puedes sembrar la mano con --mano "
                  "'<13 cartas>' para no depender del read.")

    recomendador.reset_mano([Carta._TODAS[i] for i in mano_ids])

    # ── confirmar-jugada opcional: reusar el _confirmar de ControladorPase ──
    confirmar_fn = None
    if args.confirmar_jugada:
        from src.captura.auto_pase import ConfigAutoPase, ControladorPase
        cfg_pase = ConfigAutoPase()
        if args.debug:
            cfg_pase.debug_dir = args.debug
        _tpl = None
        for cand in ["calibracion/hearts_app/confirmar_check.png",
                     "calibracion/hearts_app/confirmar.png"]:
            if _Path(cand).is_file():
                _tpl = cand
                break
        ctrl_pase = ControladorPase(
            cliente=cliente, regiones=reg, banner_clf=clf,
            reconocedor_mano=rec_mano, recomendar=lambda *_: [],
            config=cfg_pase, plantilla_confirmar=_tpl)
        confirmar_fn = ctrl_pase._confirmar

    cfg = ConfigAutoJuego(debug_dir=args.debug,
                          confirmar_jugada=args.confirmar_jugada,
                          umbral_brillo_rel=args.umbral_brillo)
    ctrl = ControladorBazas(
        cliente=cliente, regiones=reg, banner_clf=clf,
        reconocedor_mano=rec_mano, recomendador=recomendador,
        config=cfg, confirmar_jugada_fn=confirmar_fn)
    ctrl.jugar_mano()


if __name__ == "__main__":
    main()
