"""
Prueba de Estrés (Fuzzing) para el Módulo 1: Motor del Juego de Corazones.

Enfrenta a 4 bots que eligen cartas legales de forma 100% aleatoria
durante 10,000 manos consecutivas, utilizando todos los núcleos del CPU
disponibles para máxima velocidad.

Métrica de éxito:
    - Cero excepciones de "carta inválida".
    - Cero bucles infinitos.
    - Ejecución completa en menos de 3 segundos (arquitectura multinúcleo).
"""

import multiprocessing
import time
import sys
import os
from src.motor import MotorCorazones


def ejecutar_bloque_manos(num_manos: int, worker_id: int) -> int:
    """Ejecuta un bloque de manos con bots aleatorios.

    Args:
        num_manos: Número de manos a ejecutar en este worker.
        worker_id: Identificador del worker para reporte.

    Returns:
        Número de excepciones encontradas (0 = éxito).
    """
    motor = MotorCorazones()
    excepciones = 0
    for i in range(num_manos):
        try:
            motor.jugar_mano()  # selector_cartas=None → random.choice
        except Exception as e:
            excepciones += 1
            # No imprimimos desde workers para no saturar stdout
    return excepciones


def main() -> None:
    NUM_MANOS = 10_000
    num_workers = min(multiprocessing.cpu_count(), NUM_MANOS)
    # Cada worker maneja al menos 100 manos
    num_workers = min(num_workers, NUM_MANOS // 100)
    num_workers = max(num_workers, 1)

    print(f"Iniciando prueba de estrés: {NUM_MANOS:,} manos consecutivas...")
    print(
        f"Workers paralelos: {num_workers} (CPU cores: {multiprocessing.cpu_count()})")
    print("Bots: selección 100% aleatoria de jugadas legales.")
    print("-" * 60)

    # Distribuir manos entre workers
    manos_por_worker = NUM_MANOS // num_workers
    manos_restantes = NUM_MANOS % num_workers

    distribucion = []
    for w in range(num_workers):
        extra = 1 if w < manos_restantes else 0
        distribucion.append((manos_por_worker + extra, w))

    inicio = time.perf_counter()

    # Ejecutar en paralelo con multiprocessing
    with multiprocessing.Pool(processes=num_workers) as pool:
        resultados = pool.starmap(
            ejecutar_bloque_manos,
            distribucion,
        )

    fin = time.perf_counter()
    duracion = fin - inicio
    total_excepciones = sum(resultados)

    print("-" * 60)
    print(f"RESULTADO: {NUM_MANOS:,} manos completadas en {duracion:.3f}s")

    if total_excepciones > 0:
        print(f"  FALLIDO: {total_excepciones} excepciones detectadas.")
        sys.exit(1)
    else:
        print("  Cero excepciones detectadas.")

    if duracion < 3.0:
        print(f"  ÉXITO: Ejecución en {duracion:.3f}s < 3.0s (límite).")
        print("Prueba de estrés superada exitosamente.")
        sys.exit(0)
    else:
        print(f"  FALLIDO: {duracion:.3f}s >= 3.0s (límite excedido).")
        sys.exit(1)


if __name__ == "__main__":
    # Necesario en Windows para multiprocessing
    multiprocessing.freeze_support()
    main()
