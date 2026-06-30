"""
Segunda pasada de deduplicación sobre banners ya extraídos.
Compara solo la región del TEXTO (máscara por diferencia del fondo) para
ignorar el color y los bordes decorativos. Usa IoU de máscaras + agrupamiento
por representante (sin efecto cadena).

Uso:
  python scripts/dedup_banners.py
  python scripts/dedup_banners.py --umbral 0.85   # más laxo (agrupa más)
  python scripts/dedup_banners.py --umbral 0.95   # más estricto
  python scripts/dedup_banners.py --dry-run        # solo mostrar grupos
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

_sys = sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import cv2
import numpy as np

_ENTRADA = "calibracion/hearts_app/banners_crudos"
_SALIDA = "calibracion/hearts_app/banners_unicos"


def _cargar_banners(entrada: Path) -> list[tuple[str, np.ndarray]]:
    """Carga todos los PNG, devuelve (nombre, gray)."""
    banners: list[tuple[str, np.ndarray]] = []
    for fp in sorted(entrada.glob("*.png")):
        img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            banners.append((fp.name, img))
    return banners


def _extraer_texto(gray: np.ndarray) -> np.ndarray:
    """Extrae solo la región con texto del banner, ignorando bordes y fondo.

    1. Detecta el color de fondo (mediana de los bordes izquierdo/derecho)
    2. Umbraliza por diferencia → máscara de texto
    3. Recorta al bounding box horizontal del texto (con padding)
    4. Devuelve la máscara binaria recortada"""
    h, w = gray.shape
    # Color de fondo: mediana de 10% izquierda y 10% derecha
    margin = max(1, w // 10)
    bg_left = np.median(gray[:, :margin])
    bg_right = np.median(gray[:, -margin:])
    bg = (bg_left + bg_right) / 2.0

    # Máscara: píxeles significativamente más claros que el fondo (texto blanco)
    _, text_mask = cv2.threshold(
        gray, int(bg) + 15, 255, cv2.THRESH_BINARY)

    # Proyección horizontal para encontrar columnas con texto
    col_proj = np.sum(text_mask, axis=0)
    text_cols = np.where(col_proj > h * 0.05)[0]  # al menos 5% de la altura
    if len(text_cols) < 10:
        # Sin texto detectable, devolver máscara completa
        return text_mask

    x0 = max(0, text_cols[0] - 5)
    x1 = min(w, text_cols[-1] + 5)

    # También recortar verticalmente (opcional, el texto suele centrado)
    row_proj = np.sum(text_mask, axis=1)
    text_rows = np.where(row_proj > w * 0.01)[0]
    if len(text_rows) >= 5:
        y0 = max(0, text_rows[0] - 3)
        y1 = min(h, text_rows[-1] + 3)
        return text_mask[y0:y1, x0:x1]

    return text_mask[:, x0:x1]


def _similitud_texto(a_gray: np.ndarray, b_gray: np.ndarray) -> float:
    """Similitud basada en IoU de las máscaras de texto recortadas.

    Ignora fondo, bordes, y decoración — solo importa qué texto aparece."""
    ta = _extraer_texto(a_gray)
    tb = _extraer_texto(b_gray)

    # Redimensionar a la misma altura para comparar
    target_h = max(ta.shape[0], tb.shape[0])
    if target_h == 0:
        return 1.0

    def _resize(mask, h_target):
        if mask.shape[0] == 0 or mask.shape[1] == 0:
            return np.zeros((h_target, 1), dtype=np.uint8)
        scale = h_target / mask.shape[0]
        new_w = max(1, int(mask.shape[1] * scale))
        return cv2.resize(mask, (new_w, h_target), interpolation=cv2.INTER_NEAREST)

    ta_r = _resize(ta, target_h)
    tb_r = _resize(tb, target_h)

    # Pad al ancho máximo
    max_w = max(ta_r.shape[1], tb_r.shape[1])
    ta_p = np.zeros((target_h, max_w), dtype=np.uint8)
    tb_p = np.zeros((target_h, max_w), dtype=np.uint8)
    ta_p[:, :ta_r.shape[1]] = ta_r
    tb_p[:, :tb_r.shape[1]] = tb_r

    intersection = np.count_nonzero(ta_p & tb_p)
    union = np.count_nonzero(ta_p | tb_p)
    if union == 0:
        return 1.0
    return intersection / union


def _matriz_distancias(banners: list[tuple[str, np.ndarray]]) -> np.ndarray:
    """Matriz N×N de distancias = 1 − IoU_bordes (0 = idénticas, 1 = distintas)."""
    n = len(banners)
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            sim = _similitud_texto(banners[i][1], banners[j][1])
            d = 1.0 - sim
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _agrupar_por_representante(
    dist: np.ndarray, umbral: float, banners: list[tuple[str, np.ndarray]]
) -> list[list[int]]:
    """Agrupamiento greedy sin encadenamiento (no single-linkage).

    Ordena por nitidez (mejor primero). Cada banner se asigna al primer grupo
    cuyo REPRESENTANTE (el más nítido del grupo) tenga distancia ≤ umbral.
    Si no coincide con ningún representante, crea un nuevo grupo.

    Esto evita el efecto cadena de single-linkage donde A≈B y B≈C fusiona
    A con C aunque A≉C."""
    n = len(dist)
    # Ordenar por nitidez descendente
    orden = sorted(range(n), key=lambda i: _sharpness(banners[i][1]), reverse=True)
    
    grupos: list[list[int]] = []
    representantes: list[int] = []  # índice del representante de cada grupo
    
    for idx in orden:
        asignado = False
        for gi, rep in enumerate(representantes):
            if dist[idx, rep] <= umbral:
                grupos[gi].append(idx)
                asignado = True
                break
        if not asignado:
            grupos.append([idx])
            representantes.append(idx)
    
    return [sorted(g) for g in grupos]


def _sharpness(img: np.ndarray) -> float:
    """Varianza del Laplaciano → métrica de nitidez."""
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return float(lap.var())


def _mejor_de_grupo(
    grupo: list[int], banners: list[tuple[str, np.ndarray]]
) -> int:
    """Elige la imagen más nítida del grupo."""
    if len(grupo) == 1:
        return grupo[0]
    scores = [(i, _sharpness(banners[i][1])) for i in grupo]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0][0]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Deduplicación robusta de banners por IoU de máscaras de texto.")
    p.add_argument("--entrada", default=_ENTRADA)
    p.add_argument("--salida", default=_SALIDA)
    p.add_argument("--umbral", type=float, default=0.80,
                   help="Umbral IoU-bordes para considerar dos banners idénticos "
                        "(default: 0.80). Mayor = más estricto.")
    p.add_argument("--dry-run", action="store_true",
                   help="Solo mostrar los grupos, no guardar.")
    args = p.parse_args()

    entrada = Path(args.entrada)
    salida = Path(args.salida)

    print(f"Cargando banners de {entrada} ...")
    banners = _cargar_banners(entrada)
    print(f"  {len(banners)} banners cargados")

    if len(banners) < 2:
        print("Nada que deduplicar.")
        return

    print(f"Calculando matriz de distancias {len(banners)}×{len(banners)} ...")
    dist = _matriz_distancias(banners)

    umbral_dist = 1.0 - args.umbral
    print(f"Agrupando (umbral IoU={args.umbral:.2f} → dist≤{umbral_dist:.4f}) ...")
    grupos = _agrupar_por_representante(dist, umbral_dist, banners)

    singletons = sum(1 for g in grupos if len(g) == 1)
    multi = sum(1 for g in grupos if len(g) > 1)
    dups_eliminados = len(banners) - len(grupos)
    print(f"  {len(grupos)} grupos: {singletons} únicos, {multi} con duplicados")
    print(f"  Duplicados eliminados: {dups_eliminados}")

    # Mostrar grupos con duplicados
    if multi > 0:
        print("\nGrupos con duplicados:")
        for g in grupos:
            if len(g) > 1:
                best = _mejor_de_grupo(g, banners)
                nombres = [banners[i][0] for i in g]
                print(f"  ─ {len(g)} banners ─")
                for name in nombres:
                    marker = " ★" if name == banners[best][0] else ""
                    print(f"      {name}{marker}")

    if args.dry_run:
        print("\n[Dry run] No se guardó nada.")
        return

    salida.mkdir(parents=True, exist_ok=True)
    guardados = 0
    for gi, g in enumerate(grupos):
        best = _mejor_de_grupo(g, banners)
        name = banners[best][0]
        img_color = cv2.imread(str(entrada / name), cv2.IMREAD_COLOR)
        out_name = salida / f"banner_{gi:04d}__{Path(name).stem}.png"
        cv2.imwrite(str(out_name), img_color)
        guardados += 1

        # También guardar info del grupo
        if len(g) > 1:
            others = [banners[i][0] for i in g if i != best]
            txt = out_name.with_suffix(".txt")
            txt.write_text(
                f"Representante: {name}\n"
                f"Duplicados ({len(others)}):\n" +
                "\n".join(f"  - {o}" for o in others),
                encoding="utf-8")

    print(f"\n✓ {guardados} banners únicos guardados en {salida.resolve()}")
    print("  Copiá los que necesites a calibracion/hearts_app/banners/")


if __name__ == "__main__":
    main()
