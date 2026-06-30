"""Quick self-test del BannerClasificadorTexto con los banners unicos."""
from src.captura.banner import (
    BannerClasificadorTexto, _extraer_mascara_texto, _mascara_a_firma,
    BannerClasificador,
)
from pathlib import Path
import numpy as np
import cv2
import sys
import os
sys.path.insert(0, os.getcwd())


DIR_UNICOS = Path("calibracion/hearts_app/banners_unicos")
DIR_PLANTILLAS = Path("calibracion/hearts_app/banners")  # plantillas clasicas


def cargar_banners(dir_path: Path):
    """Carga todas las PNGs de un directorio como (path, gray_image)."""
    banners = []
    for f in sorted(dir_path.glob("*.png")):
        gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if gray is not None:
            banners.append((f.name, gray))
    return banners


def test_mascara_vacio():
    """La mascara de texto de un banner vacio deberia ser casi toda 0."""
    banners = cargar_banners(DIR_UNICOS)
    if not banners:
        print("⚠ No hay banners en", DIR_UNICOS)
        return

    # Encontrar uno que parece vacio (pocos pixeles de texto)
    for name, gray in banners:
        mask = _extraer_mascara_texto(gray)
        pct_texto = np.sum(mask > 0) / mask.size
        if pct_texto < 0.05:
            print(f"  [OK] '{name}' es vacio (texto={pct_texto:.1%})")
            return
    print("  ⚠ No se encontro banner vacio claro")


def test_similaridad_mismo_tipo():
    """Dos banners del mismo tipo deberian tener alta similitud de texto."""
    banners = cargar_banners(DIR_UNICOS)
    if len(banners) < 2:
        return

    # Calcular mascaras de texto para todos
    firmas = [(name, _mascara_a_firma(_extraer_mascara_texto(gray)))
              for name, gray in banners]

    # Comparar todos contra todos y mostrar los pares mas similares
    print(f"\n--- Top 20 pares mas similares (de {len(firmas)} banners) ---")
    pares = []
    for i in range(len(firmas)):
        for j in range(i + 1, len(firmas)):
            fi, fj = firmas[i][1], firmas[j][1]
            min_len = min(len(fi), len(fj))
            sim = float(np.dot(fi[:min_len], fj[:min_len]))
            pares.append((sim, firmas[i][0], firmas[j][0]))

    pares.sort(reverse=True)
    for sim, n1, n2 in pares[:20]:
        print(f"  {sim:.3f} | {n1[:50]:<50} | {n2[:50]}")


def test_clasificador_texto():
    """Prueba el clasificador de texto completo."""
    print("\n--- BannerClasificadorTexto ---")

    # Usar los banners UNICOS como plantillas
    unicos = cargar_banners(DIR_UNICOS)
    if len(unicos) < 3:
        print("  ⚠ Pocos banners para probar")
        return

    # Crear directorio temporal de plantillas con los primeros 10 unicos
    tmp_dir = Path("calibracion/hearts_app/_tmp_plantillas")
    tmp_dir.mkdir(exist_ok=True)
    # Limpiar
    for f in tmp_dir.glob("*.png"):
        f.unlink()

    # Copiar primeros 10 como plantillas (sin etiquetar)
    for i, (name, gray) in enumerate(unicos[:10]):
        tag = f"banner_{i:03d}"
        cv2.imwrite(str(tmp_dir / f"{tag}__0.png"), gray)

    try:
        clasificador = BannerClasificadorTexto(tmp_dir, umbral=0.60)
        print(f"  Cargadas {clasificador.num_plantillas} plantillas")

        # Clasificar cada banner contra las plantillas
        for name, gray in unicos:
            res, sims = clasificador.clasificar_verbose(
                cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
            if res.tag != "desconocido":
                print(
                    f"  {name[:50]:<50} → {res.tag} (dist={res.distancia:.3f})")
    finally:
        # Limpiar
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_banners_con_nombre():
    """Simula banners con nombres de jugador variables."""
    print("\n--- Test con nombres variables ---")

    # Crear imagenes sinteticas simulando "Turno de Alex" / "Turno de Pedro"
    plantillas_dir = Path("calibracion/hearts_app/_tmp_nombres")
    plantillas_dir.mkdir(exist_ok=True)

    try:
        # Crear plantillas: "Turno de Alex", "Recoge la baza", vacio
        for tag, texto in [
            ("turno_rival", "Turno de"),          # solo parte fija
            ("baza_rival", "recoge la baza"),       # solo parte fija
            ("vacio", ""),
        ]:
            img = np.full((60, 400), 60, dtype=np.uint8)  # fondo oscuro
            if texto:
                cv2.putText(img, texto, (15, 40), cv2.FONT_HERSHEY_DUPLEX,
                            1.0, 240, 2)  # texto claro grueso
            cv2.imwrite(str(plantillas_dir / f"{tag}__0.png"), img)

        # Crear consultas con diferentes nombres
        consultas = {
            "Turno de Maria": "turno_rival",
            "Turno de Juan Carlos": "turno_rival",
            "Maria recoge la baza": "baza_rival",
            "Juan Carlos recoge la baza": "baza_rival",
            "": "vacio",
        }

        clasificador = BannerClasificadorTexto(plantillas_dir, umbral=0.35)

        for texto, esperado in consultas.items():
            img = np.full((60, 400), 60, dtype=np.uint8)
            if texto:
                cv2.putText(img, texto, (15, 40), cv2.FONT_HERSHEY_DUPLEX,
                            1.0, 240, 2)
            res = clasificador.clasificar(img)
            ok = "OK" if res.tag == esperado else "FALLO"
            print(f"  [{ok}] '{texto}' → {res.tag} (esperado={esperado})")

    finally:
        import shutil
        shutil.rmtree(plantillas_dir, ignore_errors=True)


def test_clasificador_original():
    """Comparar con el clasificador original (grayscale)."""
    print("\n--- BannerClasificador (original) ---")
    if not DIR_PLANTILLAS.is_dir():
        print(f"  ⚠ No existe {DIR_PLANTILLAS}")
        return

    clasificador = BannerClasificador(DIR_PLANTILLAS)
    print(f"  Cargadas {len(clasificador._tpl)} plantillas")

    # Probar con los banners unicos
    unicos = cargar_banners(DIR_UNICOS)
    clasificados = 0
    for name, gray in unicos[:20]:
        res = clasificador.clasificar(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        if res.tag != "desconocido":
            clasificados += 1
    print(
        f"  Clasificados: {clasificados}/{min(20, len(unicos))} (umbral fijo)")


if __name__ == "__main__":
    print("=== SELF-TEST BannerClasificadorTexto ===\n")
    test_mascara_vacio()
    test_similaridad_mismo_tipo()
    test_clasificador_texto()
    test_banners_con_nombre()
    test_clasificador_original()
    print("\n=== FIN ===")
