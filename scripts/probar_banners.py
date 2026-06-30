"""Prueba el reconocimiento de banners contra las plantillas calibradas.

Uso:
    # Un solo frame (imprime clasificacion + top-5 de cada metodo)
    python scripts/probar_banners.py --frame ruta/al/frame.png

    # Carpeta de frames (resumen de aciertos/fallos)
    python scripts/probar_banners.py --carpeta videos/fotogramas

    # Con plantillas y regiones personalizadas
    python scripts/probar_banners.py --frame img.png \
        --plantillas mi_calibracion/banners \
        --regiones mi_calibracion/regiones.json

Las regiones estan en fracciones [0..1] → funciona a cualquier resolucion.
"""
from src.captura.banner import (
    BannerClasificador, BannerClasificadorTexto, _extraer_mascara_texto,
)
from src.captura.vision_hearts import Regiones, EstadoVisual
import numpy as np
import cv2
import argparse
import sys
from pathlib import Path
from collections import Counter
from typing import Optional

# Bootstrap para correr desde cualquier CWD
_RAIZ = Path(__file__).resolve().parent.parent
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))


# ── helpers ────────────────────────────────────────────────────────────────

def _bold(s: str) -> str:
    return s  # sin ANSI


def _barra(confianza: float, ancho: int = 20) -> str:
    """Barra ASCII de confianza."""
    n = max(0, min(ancho, int(confianza * ancho)))
    c = "█" if confianza > 0.8 else "▓" if confianza > 0.5 else "▒"
    return c * n + "░" * (ancho - n)


def _formato_confianza(sim: float, dist: Optional[float] = None) -> str:
    """Muestra similitud coseno y (opcional) distancia euclidea."""
    if dist is not None:
        return f"cos={sim:.3f} dist={dist:.3f}"
    return f"cos={sim:.3f}"


def clasificar_banner(img_bgr: np.ndarray, regiones: Regiones,
                      clf_texto: BannerClasificadorTexto,
                      clf_gray: Optional[BannerClasificador] = None,
                      ) -> None:
    """Clasifica un frame con ambos metodos e imprime resultados."""
    roi = regiones.recortar(img_bgr, regiones.banner)
    h_img, w_img = img_bgr.shape[:2]
    h_roi, w_roi = roi.shape[:2]
    print(f"  Frame: {w_img}×{h_img}  |  Banner ROI: {w_roi}×{h_roi}")

    # ── Mascara de texto (debug) ──
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mask = _extraer_mascara_texto(gray)
    pct_texto = np.sum(mask > 0) / mask.size
    print(f"  Texto en banner: {pct_texto:.1%}")

    # ── BannerClasificadorTexto ──
    print(f"\n  ── BannerClasificadorTexto (umbral={clf_texto.umbral}) ──")
    res_t, sims_t = clf_texto.clasificar_verbose(roi)
    print(f"  Clasificado: {res_t.tag}  |  categoria={res_t.categoria}"
          f"  dato={res_t.dato}  |  distancia={res_t.distancia:.3f}")
    if sims_t:
        print(f"  Top-5:")
        for tag, sim in sims_t[:5]:
            marker = "◄" if tag == res_t.tag else " "
            print(f"    {marker} {tag:<20}  {_barra(sim)} {sim:.3f}")

    # ── BannerClasificador (grayscale) ──
    if clf_gray is not None:
        print(
            f"\n  ── BannerClasificador grayscale (umbral={clf_gray.umbral}) ──")
        res_g, sims_g = clf_gray.clasificar_verbose(roi)
        print(f"  Clasificado: {res_g.tag}  |  categoria={res_g.categoria}"
              f"  dato={res_g.dato}  |  distancia={res_g.distancia:.3f}")
        if sims_g:
            print(f"  Top-3:")
            for tag, dist in sims_g[:3]:
                marker = "◄" if tag == res_g.tag else " "
                conf = max(0, 1 - dist / clf_gray.umbral)
                print(f"    {marker} {tag:<20}  {_barra(conf)} dist={dist:.3f}")


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prueba el reconocimiento de banners")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--frame", type=str, help="Un solo frame PNG/JPG")
    src.add_argument("--carpeta", type=str, help="Carpeta con frames PNG/JPG")
    ap.add_argument("--plantillas", type=str,
                    default="calibracion/hearts_app/banners",
                    help="Directorio de plantillas de banner (default: "
                         "calibracion/hearts_app/banners)")
    ap.add_argument("--regiones", type=str,
                    default="calibracion/hearts_app/regiones.json",
                    help="Archivo de regiones en fracciones (default: "
                         "calibracion/hearts_app/regiones.json)")
    ap.add_argument("--umbral-texto", type=float, default=0.75,
                    help="Umbral cosine similarity para BannerClasificadorTexto"
                         " (default: 0.75)")
    ap.add_argument("--umbral-gray", type=float, default=1.5,
                    help="Umbral distancia euclidea para BannerClasificador"
                         " (default: 1.5)")
    ap.add_argument("--solo-texto", action="store_true",
                    help="Usar solo BannerClasificadorTexto (no el grayscale)")
    ap.add_argument("--resumen", action="store_true",
                    help="Solo tabla resumen (con --carpeta, util para validar)")
    args = ap.parse_args()

    # ── Cargar regiones ──
    regiones = Regiones.cargar(args.regiones)

    # ── Cargar clasificadores ──
    dir_plantillas = Path(args.plantillas)
    if not dir_plantillas.is_dir():
        print(f"ERROR: No existe {dir_plantillas}")
        sys.exit(1)

    clf_texto = BannerClasificadorTexto(
        dir_plantillas, umbral=args.umbral_texto)
    print(f"BannerClasificadorTexto: {clf_texto.num_plantillas} plantillas")

    clf_gray: Optional[BannerClasificador] = None
    if not args.solo_texto:
        clf_gray = BannerClasificador(dir_plantillas, umbral=args.umbral_gray)
        print(
            f"BannerClasificador (gray): {clf_gray.num_plantillas} plantillas")

    # ── Procesar ──
    if args.frame:
        img = cv2.imread(args.frame)
        if img is None:
            print(f"ERROR: No se pudo leer {args.frame}")
            sys.exit(1)
        print(f"\n{'='*60}")
        clasificar_banner(img, regiones, clf_texto, clf_gray)
        print(f"{'='*60}")

    else:
        carpeta = Path(args.carpeta)
        pngs = sorted(carpeta.glob("*.png")) + sorted(carpeta.glob("*.jpg"))
        if not pngs:
            print(f"ERROR: Sin PNGs en {carpeta}")
            sys.exit(1)

        resumen_t: Counter = Counter()
        resumen_g: Counter = Counter()
        n, n_desconocido_t, n_desconocido_g = 0, 0, 0

        for p in pngs:
            img = cv2.imread(str(p))
            if img is None:
                continue
            n += 1

            roi = regiones.recortar(img, regiones.banner)
            res_t = clf_texto.clasificar(roi)
            resumen_t[res_t.tag] += 1
            if res_t.tag == "desconocido":
                n_desconocido_t += 1

            res_g_tag = "—"
            if clf_gray is not None:
                res_g = clf_gray.clasificar(roi)
                resumen_g[res_g.tag] += 1
                res_g_tag = res_g.tag
                if res_g.tag == "desconocido":
                    n_desconocido_g += 1

            if not args.resumen:
                print(f"  {p.name:<55}  texto→{res_t.tag:<18}"
                      f"  gray→{res_g_tag}")

        # ── Resumen ──
        print(f"\n{'='*60}")
        print(f"RESUMEN: {n} frames procesados")
        print(f"  BannerClasificadorTexto: {n - n_desconocido_t}/{n}"
              f" clasificados ({100*(n - n_desconocido_t)/n:.0f}%)")
        if resumen_t:
            for tag, count in resumen_t.most_common():
                print(f"    {tag:<20} {count:>4}")
        if clf_gray is not None:
            print(f"  BannerClasificador (gray): {n - n_desconocido_g}/{n}"
                  f" clasificados ({100*(n - n_desconocido_g)/n:.0f}%)")
            if resumen_g:
                for tag, count in resumen_g.most_common():
                    print(f"    {tag:<20} {count:>4}")


if __name__ == "__main__":
    main()
