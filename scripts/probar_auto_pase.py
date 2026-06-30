"""Diagnostico offline del auto-pase: vision + modelo sin ADB.

Lee un frame (o varios) de la app en fase de pase, clasifica el banner,
reconoce la mano, consulta al modelo que cartas pasaria, y anota visualmente
el resultado. NO toca nada — es solo diagnostico.

Ideal para validar:
  - Que el banner se clasifica bien (direccion del pase)
  - Que las 13 cartas de la mano se leen correctamente
  - Que el modelo recomienda un pase razonable
  - Donde tocaria cada carta (punto de toque)

Uso:
    # Un frame en fase de pase:
    python scripts/probar_auto_pase.py --frame ruta/al/frame.png

    # Varios frames (resumen):
    python scripts/probar_auto_pase.py --carpeta videos/fotogramas

    # Guardar imagenes anotadas:
    python scripts/probar_auto_pase.py --frame img.png --salida debug/

    # Con modelo y plantillas personalizadas:
    python scripts/probar_auto_pase.py --frame img.png \
        --modelo models/produccion/v10c_campeon \
        --banners calibracion/hearts_app/banners
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Bootstrap para correr desde cualquier CWD
_RAIZ = Path(__file__).resolve().parent.parent
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

import cv2
import numpy as np

from src.captura.vision_hearts import (
    BannerClasificador, CartaMano, Regiones, leer_mano_posiciones,
)
from src.captura.vision_cartas import ReconocedorPlantilla
from src.captura.modelos import carta_a_str


# ── helpers ────────────────────────────────────────────────────────────────

_SEMANTICA = {
    "pase_izquierda": "Pasar 3 a la IZQUIERDA",
    "pase_derecha": "Pasar 3 a la DERECHA",
    "pase_enfrente": "Pasar 3 al FRENTE",
    "pase_recibido": "Cartas pasadas para ti",
}
_DIR_BANNER = {"izquierda": "izquierda",
               "derecha": "derecha", "enfrente": "frente"}


def _construir_recomendar(modelo_dir: str):
    """Devuelve (mano_ids, direccion) -> [3 ids] usando el modelo."""
    from src.dominio.carta import Carta
    from scripts.recomendador import Recomendador

    rec = Recomendador(modelo_dir)
    if not rec.con_pase:
        raise SystemExit(
            "El modelo no tiene fase de pase (obs<228). Usa un v12+/v13.")
    rec._cargar_modelo()

    def recomendar(mano_ids: List[int], direccion: str) -> List[int]:
        rec.reset_mano([Carta._TODAS[i] for i in mano_ids])
        cartas = rec.recomendar_pase(direccion)
        return [c.id for c in cartas]

    return recomendar


def _rect_color(color: Tuple[int, int, int], alpha: float = 0.35
                ) -> Tuple[int, int, int]:
    """Mezcla un color con blanco para hacerlo mas claro (overlay)."""
    return tuple(int(c + (255 - c) * (1 - alpha)) for c in color)


# ── procesamiento de un frame ──────────────────────────────────────────────


def diagnosticar_frame(
    img: np.ndarray,
    regiones: Regiones,
    banner_clf: BannerClasificador,
    rec_mano: ReconocedorPlantilla,
    recomendar,
) -> dict:
    """Analiza un frame y devuelve un dict con los resultados."""
    H, W = img.shape[:2]
    info: dict = {"w": W, "h": H}

    # 1) Banner
    roi = regiones.recortar(img, regiones.banner)
    res_b, dists = banner_clf.clasificar_verbose(roi)
    info["banner_tag"] = res_b.tag
    info["banner_cat"] = res_b.categoria
    info["banner_dato"] = res_b.dato
    info["banner_dist"] = res_b.distancia
    info["banner_top3"] = [(t, round(d, 3)) for t, d in dists[:3]]
    info["banner_texto"] = _SEMANTICA.get(res_b.tag, res_b.tag)

    # 2) Mano
    cartas = leer_mano_posiciones(img, regiones, rec_mano)
    info["mano_total"] = len(cartas)
    info["mano_ok"] = len([c for c in cartas if c.carta_id is not None])
    info["cartas"] = cartas
    mano_ids = [c.carta_id for c in cartas if c.carta_id is not None]
    info["mano_ids"] = mano_ids
    info["mano_str"] = "  ".join(
        carta_a_str(cid) if cid is not None else "??" for cid in [
            c.carta_id for c in cartas])

    # 3) Recomendacion (solo si es fase de pase y hay 13 cartas)
    info["direccion"] = None
    info["pasadas"] = []
    info["pasadas_str"] = ""
    if res_b.categoria == "pase" and res_b.dato in _DIR_BANNER and len(mano_ids) >= 13:
        direccion = _DIR_BANNER[res_b.dato]
        info["direccion"] = direccion
        pasadas = recomendar(mano_ids, direccion)[:3]
        info["pasadas"] = pasadas
        info["pasadas_str"] = "  ".join(carta_a_str(c) for c in pasadas)

    return info


# ── visualizacion ──────────────────────────────────────────────────────────


def anotar_frame(img: np.ndarray, regiones: Regiones,
                 info: dict, salida: Optional[Path] = None) -> np.ndarray:
    """Dibuja el diagnostico sobre la imagen y opcionalmente la guarda."""
    out = img.copy()
    H, W = out.shape[:2]
    PALOS_COLORES = {"P": (0, 0, 255), "C": (0, 180, 0),
                     "D": (255, 0, 0), "T": (200, 150, 0)}

    # ── rectangulos de regiones ──
    for nombre, caja in [("banner", regiones.banner), ("mano", regiones.mano)]:
        x, y, cw, ch = caja
        x0, y0 = int(x * W), int(y * H)
        x1, y1 = int((x + cw) * W), int((y + ch) * H)
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 0), 2)
        cv2.putText(out, nombre, (x0 + 5, y0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

    # ── cartas de la mano ──
    for i, c in enumerate(info.get("cartas", [])):
        x, y = int(c.centro[0]), int(c.centro[1])
        if c.carta_id is not None:
            s = carta_a_str(c.carta_id)
            color = PALOS_COLORES.get(s[-1], (0, 255, 0))
            cv2.circle(out, (x, y), 16, color, 2)
            cv2.putText(out, f"{i}:{s}", (x - 22, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        else:
            cv2.circle(out, (x, y), 16, (0, 0, 255), 2)
            cv2.putText(out, f"{i}:??", (x - 18, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # ── cartas a pasar (doble circulo + numero) ──
    pasadas = info.get("pasadas", [])
    cartas = info.get("cartas", [])
    for pid in pasadas:
        for c in cartas:
            if c.carta_id == pid:
                x, y = int(c.centro[0]), int(c.centro[1])
                s = carta_a_str(pid)
                color = PALOS_COLORES.get(s[-1], (0, 255, 255))
                # Doble circulo
                cv2.circle(out, (x, y), 22, color, 3)
                cv2.circle(out, (x, y), 26, color, 1)
                # Numero de orden
                idx = pasadas.index(pid) + 1
                cv2.putText(out, str(idx), (x + 16, y - 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 3)
                break

    # ── panel de info (esquina superior izquierda) ──
    lineas = []
    lineas.append(f"Banner: {info.get('banner_texto', '?')}"
                  f"  (tag={info.get('banner_tag', '?')}"
                  f" dist={info.get('banner_dist', 0):.3f})")
    top3 = info.get("banner_top3", [])
    if top3:
        lineas.append("  top3: " +
                      " | ".join(f"{t}={d:.3f}" for t, d in top3))
    lineas.append(f"Mano: {info.get('mano_ok', 0)}/{info.get('mano_total', 0)}"
                  f" cartas reconocidas")
    mano_str = info.get("mano_str", "")
    if mano_str:
        lineas.append(f"  {mano_str}")
    if info.get("direccion"):
        lineas.append(f"Pase: {info['direccion']}")
        lineas.append(f"  Recomienda: {info.get('pasadas_str', '')}")

    # Fondo semi-transparente para el panel
    panel_h = 20 * len(lineas) + 15
    overlay = out.copy()
    cv2.rectangle(overlay, (8, 8), (W - 20, 8 + panel_h),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)

    for i, linea in enumerate(lineas):
        y = 28 + i * 20
        cv2.putText(out, linea, (14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

    # ── guardar ──
    if salida:
        salida.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(salida), out)

    return out


# ── main ────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Diagnostico offline del auto-pase (vision + modelo)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--frame", type=str, help="Un frame PNG en fase de pase")
    src.add_argument("--carpeta", type=str, help="Carpeta con frames PNG")
    ap.add_argument("--modelo", type=str,
                    default="models/produccion/v10c_campeon",
                    help="Ruta al modelo con soporte de pase")
    ap.add_argument("--regiones", type=str,
                    default="calibracion/hearts_app/regiones.json")
    ap.add_argument("--banners", type=str,
                    default="calibracion/hearts_app/banners")
    ap.add_argument("--completas", type=str,
                    default="calibracion/hearts_app/cartas_completas")
    ap.add_argument("--umbral-banner", type=float, default=1.5)
    ap.add_argument("--salida", type=str, default=None,
                    help="Directorio donde guardar las imagenes anotadas")
    ap.add_argument("--mostrar", action="store_true",
                    help="Abre cada imagen anotada en una ventana (requiere GUI)")
    args = ap.parse_args()

    # ── cargar ──
    regiones = Regiones.cargar(args.regiones)
    banner_clf = BannerClasificador(args.banners, umbral=args.umbral_banner)
    rec_mano = ReconocedorPlantilla(args.completas)
    print(f"Modelo: {args.modelo}")
    recomendar = _construir_recomendar(args.modelo)
    print(f"Banners: {banner_clf.num_plantillas} plantillas")
    print()

    salida_dir = Path(args.salida) if args.salida else None

    # ── procesar ──
    if args.frame:
        img = cv2.imread(args.frame)
        if img is None:
            print(f"ERROR: No se pudo leer {args.frame}")
            sys.exit(1)

        info = diagnosticar_frame(img, regiones, banner_clf, rec_mano,
                                  recomendar)
        _imprimir_diagnostico(info)

        salida_path = None
        if salida_dir:
            nombre = Path(args.frame).stem
            salida_path = salida_dir / f"{nombre}_diagnostico.png"
        anotada = anotar_frame(img, regiones, info, salida_path)

        if args.mostrar:
            cv2.imshow("Diagnostico auto-pase", anotada)
            print("\nPresiona cualquier tecla para cerrar...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    else:
        carpeta = Path(args.carpeta)
        pngs = sorted(carpeta.glob("*.png"))
        if not pngs:
            print(f"ERROR: Sin PNGs en {carpeta}")
            sys.exit(1)

        n, n_pase, n_13 = 0, 0, 0
        for p in pngs:
            img = cv2.imread(str(p))
            if img is None:
                continue
            n += 1
            info = diagnosticar_frame(img, regiones, banner_clf, rec_mano,
                                      recomendar)
            es_pase = info["banner_cat"] == "pase"
            tiene_13 = info["mano_ok"] >= 13
            if es_pase:
                n_pase += 1
            if tiene_13:
                n_13 += 1

            estado = "PASE" if es_pase else info["banner_tag"]
            pasadas = info.get("pasadas_str", "")
            print(f"  {p.name:<55}  banner={estado:<20}"
                  f"  mano={info['mano_ok']}/13"
                  + (f"  → {pasadas}" if pasadas else ""))

            if salida_dir:
                salida_path = salida_dir / f"{p.stem}_diagnostico.png"
                anotar_frame(img, regiones, info, salida_path)

        print(f"\n{'='*60}")
        print(f"RESUMEN: {n} frames")
        print(f"  Fase de pase: {n_pase}/{n} ({100*n_pase//n}%)" if n else "")
        print(f"  Mano 13/13:   {n_13}/{n} ({100*n_13//n}%)" if n else "")
        if salida_dir:
            print(f"  Imagenes guardadas en: {salida_dir}/")


def _imprimir_diagnostico(info: dict) -> None:
    """Imprime el diagnostico en consola."""
    print(f"Frame: {info['w']}x{info['h']}")
    print(f"Banner: {info['banner_texto']}")
    print(f"  tag={info['banner_tag']}  cat={info['banner_cat']}"
          f"  dato={info['banner_dato']}  dist={info['banner_dist']:.3f}")
    top3 = info.get("banner_top3", [])
    if top3:
        for t, d in top3:
            print(f"    {t:<22} dist={d:.3f}")
    print(f"Mano: {info['mano_ok']}/{info['mano_total']} cartas")
    print(f"  {info['mano_str']}")
    if info.get("direccion"):
        print(f"Pase: {info['direccion']}  →  {info['pasadas_str']}")


if __name__ == "__main__":
    main()
