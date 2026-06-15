"""
Pruebas unitarias para el Módulo 1: Motor del Juego de Corazones.

Cubre casos límite, filtros de jugadas legales, puntuación,
gatillos de estado (corazones_rotos) y detección de Pleno (Shooting the Moon).
"""

import pytest
from src.dominio.carta import Carta
from src.dominio.baraja import Baraja
from src.dominio.jugador import Jugador
from src.dominio.motor import MotorCorazones


# ============================================================
# Pruebas de la clase Carta
# ============================================================

class TestCarta:
    def test_creacion_carta_valida(self):
        c = Carta(palo=0, valor=2)
        assert c.palo == 0
        assert c.valor == 2

    def test_creacion_carta_palo_invalido_bajo(self):
        with pytest.raises(ValueError):
            Carta(palo=-1, valor=5)

    def test_creacion_carta_palo_invalido_alto(self):
        with pytest.raises(ValueError):
            Carta(palo=4, valor=5)

    def test_creacion_carta_valor_invalido_bajo(self):
        with pytest.raises(ValueError):
            Carta(palo=0, valor=1)

    def test_creacion_carta_valor_invalido_alto(self):
        with pytest.raises(ValueError):
            Carta(palo=0, valor=15)

    def test_igualdad_cartas(self):
        a = Carta(palo=2, valor=12)
        b = Carta(palo=2, valor=12)
        c = Carta(palo=2, valor=13)
        assert a == b
        assert a != c

    def test_hash_carta(self):
        a = Carta(palo=2, valor=12)
        b = Carta(palo=2, valor=12)
        assert hash(a) == hash(b)
        s = {a, b}
        assert len(s) == 1

    def test_es_corazon(self):
        assert Carta(palo=3, valor=2).es_corazon is True
        assert Carta(palo=3, valor=14).es_corazon is True
        assert Carta(palo=0, valor=2).es_corazon is False
        assert Carta(palo=1, valor=10).es_corazon is False
        assert Carta(palo=2, valor=14).es_corazon is False

    def test_es_dama_de_picas(self):
        assert Carta(palo=2, valor=12).es_dama_de_picas is True
        assert Carta(palo=2, valor=11).es_dama_de_picas is False
        assert Carta(palo=3, valor=12).es_dama_de_picas is False

    def test_puntos_corazon(self):
        for valor in range(2, 15):
            c = Carta(palo=3, valor=valor)
            assert c.puntos == 1

    def test_puntos_dama_de_picas(self):
        c = Carta(palo=2, valor=12)
        assert c.puntos == 13

    def test_puntos_resto_cartas(self):
        for palo in range(3):
            for valor in range(2, 15):
                if palo == 2 and valor == 12:
                    continue
                c = Carta(palo=palo, valor=valor)
                assert c.puntos == 0

    def test_es_dos_de_treboles(self):
        assert Carta(palo=0, valor=2).es_dos_de_treboles is True
        assert Carta(palo=0, valor=3).es_dos_de_treboles is False
        assert Carta(palo=1, valor=2).es_dos_de_treboles is False

    def test_repr_carta(self):
        c = Carta(palo=0, valor=14)
        r = repr(c)
        # El As se representa como "A", no como "14"
        assert "A" in r and "♣" in r


# ============================================================
# Pruebas de la clase Jugador
# ============================================================

class TestJugador:
    def test_creacion_jugador(self):
        j = Jugador("Norte")
        assert j.nombre == "Norte"
        assert len(j.mano) == 0
        assert len(j.bazas_ganadas) == 0
        assert j.puntuacion_historica == 0

    def test_recibir_cartas(self):
        j = Jugador("Sur")
        cartas = [Carta(0, 2), Carta(0, 3), Carta(1, 5)]
        j.recibir_mano(cartas)
        assert len(j.mano) == 3
        assert j.mano[0] == Carta(0, 2)

    def test_jugar_carta_la_remueve_de_mano(self):
        j = Jugador("Este")
        cartas = [Carta(0, 2), Carta(0, 3), Carta(1, 5)]
        j.recibir_mano(cartas)
        jugada = j.jugar_carta(Carta(0, 3))
        assert jugada == Carta(0, 3)
        assert len(j.mano) == 2
        assert Carta(0, 3) not in j.mano

    def test_jugar_carta_no_en_mano_lanza_error(self):
        j = Jugador("Oeste")
        j.recibir_mano([Carta(0, 2)])
        with pytest.raises(ValueError):
            j.jugar_carta(Carta(3, 14))

    def test_sumar_puntos(self):
        j = Jugador("Norte")
        j.sumar_puntos(5)
        assert j.puntuacion_historica == 5
        j.sumar_puntos(13)
        assert j.puntuacion_historica == 18

    def test_contar_puntos_bazas(self):
        j = Jugador("Sur")
        bazas = [Carta(3, 2), Carta(3, 5), Carta(2, 12), Carta(0, 10)]
        j.bazas_ganadas = bazas
        puntos = j.contar_puntos_bazas()
        assert puntos == 1 + 1 + 13 + 0  # 15


# ============================================================
# Pruebas de la clase Baraja
# ============================================================

class TestBaraja:
    def test_baraja_tiene_52_cartas(self):
        b = Baraja()
        assert len(b.cartas) == 52

    def test_todas_las_cartas_son_unicas(self):
        b = Baraja()
        ids = [(c.palo, c.valor) for c in b.cartas]
        assert len(ids) == len(set(ids))

    def test_barajar_no_pierde_cartas(self):
        b = Baraja()
        b.barajar()
        assert len(b.cartas) == 52

    def test_repartir_da_13_a_cada_jugador(self):
        b = Baraja()
        jugadores = [Jugador(n) for n in ["N", "S", "E", "O"]]
        b.repartir(jugadores)
        for j in jugadores:
            assert len(j.mano) == 13

    def test_repartir_no_duplica_cartas(self):
        b = Baraja()
        jugadores = [Jugador(n) for n in ["N", "S", "E", "O"]]
        b.repartir(jugadores)
        todas = []
        for j in jugadores:
            todas.extend(j.mano)
        ids = [(c.palo, c.valor) for c in todas]
        assert len(ids) == len(set(ids)) == 52


# ============================================================
# Pruebas de MotorCorazones
# ============================================================

class TestMotorCorazones:
    def test_inicializacion(self):
        m = MotorCorazones()
        assert len(m.jugadores) == 4
        assert m.corazones_rotos is False
        assert m.numero_baza == 0
        assert len(m.mesa) == 0

    def test_repartir_inicializa_mano(self):
        m = MotorCorazones()
        m.repartir()
        for j in m.jugadores:
            assert len(j.mano) == 13
        assert m.numero_baza == 1
        assert m.corazones_rotos is False

    # --- Filtro 1: Salida Inicial (2 de Tréboles) ---

    def test_filtro1_primera_baza_mesa_vacia_solo_dos_de_treboles(self):
        m = MotorCorazones()
        m.repartir()
        # Encontrar el jugador que tiene 2♣
        for idx, jug in enumerate(m.jugadores):
            if Carta(0, 2) in jug.mano:
                legales = m.obtener_jugadas_legales(idx)
                assert legales == [Carta(0, 2)]
                return
        pytest.fail("Ningún jugador tiene el 2 de Tréboles")

    def test_filtro1_sin_dos_de_treboles_no_aplica(self):
        """Si un jugador no tiene 2♣ y la mesa está vacía en baza 1,
        no debería ser su turno porque el que tiene 2♣ debe abrir."""
        m = MotorCorazones()
        m.repartir()
        # El motor maneja esto por orden de turno; validamos que
        # obtener_jugadas_legales no crashee para ningún jugador
        for idx in range(4):
            legales = m.obtener_jugadas_legales(idx)
            assert len(legales) > 0

    # --- Filtro 2: Primera Baza Segura (sin puntos en baza 1) ---

    def test_filtro2_primera_baza_no_jugar_puntos_si_hay_alternativa(self):
        m = MotorCorazones()
        m.repartir()
        # Simular: forzar que un jugador tenga corazones y otras cartas en baza 1
        m.mesa = [(0, Carta(0, 3))]  # alguien ya jugó 3♣
        m.palo_de_salida = 0
        # Dar a jugador 1 una mano con tréboles (incluyendo alguno que no sea puntos)
        # y también corazones
        j = m.jugadores[1]
        j.mano = [
            Carta(0, 4), Carta(0, 5),  # tréboles sin puntos
            Carta(3, 2), Carta(3, 3),  # corazones (puntos)
        ]
        legales = m.obtener_jugadas_legales(1)
        # Debe poder jugar tréboles (seguir palo) y en baza 1 no corazones
        for c in legales:
            assert c.puntos == 0 or c.palo == 0

    def test_filtro2_primera_baza_si_solo_tiene_puntos_puede_jugarlos(self):
        m = MotorCorazones()
        m.repartir()
        m.mesa = [(0, Carta(1, 3))]  # palo de salida: diamantes
        m.palo_de_salida = 1
        j = m.jugadores[1]
        # Solo tiene corazones (void en diamantes)
        j.mano = [Carta(3, 2), Carta(3, 3), Carta(3, 4)]
        legales = m.obtener_jugadas_legales(1)
        assert len(legales) == 3
        # Todas son corazones, pero es legal porque no tiene alternativa
        for c in legales:
            assert c.palo == 3

    # --- Filtro 3: Asistir al Palo / Voids ---

    def test_filtro3_debe_asistir_al_palo(self):
        m = MotorCorazones()
        m.repartir()
        m.corazones_rotos = True
        m.numero_baza = 5
        m.mesa = [(0, Carta(1, 5))]
        m.palo_de_salida = 1  # diamantes
        j = m.jugadores[1]
        j.mano = [
            Carta(1, 7), Carta(1, 10),  # diamantes
            Carta(0, 2), Carta(3, 14),  # trébol y corazón
        ]
        legales = m.obtener_jugadas_legales(1)
        assert len(legales) == 2
        for c in legales:
            assert c.palo == 1

    def test_filtro3_void_puede_jugar_otro_palo(self):
        m = MotorCorazones()
        m.repartir()
        m.corazones_rotos = True
        m.numero_baza = 5
        m.mesa = [(0, Carta(1, 5))]
        m.palo_de_salida = 1  # diamantes
        j = m.jugadores[1]
        # Sin diamantes (void)
        j.mano = [Carta(0, 2), Carta(3, 14)]
        legales = m.obtener_jugadas_legales(1)
        assert len(legales) == 2  # puede jugar ambas

    # --- Filtro 4: Liderar con Corazones ---

    def test_filtro4_corazones_no_rotos_no_puede_abrir_con_corazones(self):
        m = MotorCorazones()
        m.repartir()
        m.corazones_rotos = False
        m.numero_baza = 3
        m.mesa = []  # mesa vacía
        m.palo_de_salida = None
        j = m.jugadores[0]
        j.mano = [
            Carta(0, 2), Carta(0, 3),
            Carta(3, 5), Carta(3, 6),
        ]
        legales = m.obtener_jugadas_legales(0)
        for c in legales:
            assert c.palo != 3, f"No debe poder abrir con {c}"

    def test_filtro4_solo_corazones_en_mano_puede_abrir(self):
        m = MotorCorazones()
        m.repartir()
        m.corazones_rotos = False
        m.numero_baza = 3
        m.mesa = []
        m.palo_de_salida = None
        j = m.jugadores[0]
        j.mano = [Carta(3, 2), Carta(3, 3), Carta(3, 4)]
        legales = m.obtener_jugadas_legales(0)
        assert len(legales) == 3
        for c in legales:
            assert c.palo == 3

    def test_filtro4_corazones_rotos_si_puede_abrir_con_corazones(self):
        m = MotorCorazones()
        m.repartir()
        m.corazones_rotos = True
        m.numero_baza = 3
        m.mesa = []
        m.palo_de_salida = None
        j = m.jugadores[0]
        j.mano = [
            Carta(0, 2), Carta(3, 5),
        ]
        legales = m.obtener_jugadas_legales(0)
        assert len(legales) == 2  # ambas son legales

    # --- Gatillo: Romper Corazones ---

    def test_gatillo_corazones_rotos_al_jugar_corazon_en_void(self):
        m = MotorCorazones()
        m.repartir()
        m.numero_baza = 2
        m.mesa = [(0, Carta(0, 3))]  # tréboles
        m.palo_de_salida = 0
        j = m.jugadores[1]
        j.mano = [Carta(3, 5)]  # void en tréboles, solo corazón
        assert m.corazones_rotos is False
        m.jugar_carta(1, Carta(3, 5))
        assert m.corazones_rotos is True

    def test_gatillo_corazones_no_se_rompen_si_abre_con_corazones(self):
        """Abrir con corazones cuando ya están rotos no es un gatillo nuevo."""
        m = MotorCorazones()
        m.repartir()
        m.corazones_rotos = True
        m.numero_baza = 4
        m.mesa = []
        m.palo_de_salida = None
        j = m.jugadores[0]
        j.mano = [Carta(3, 5), Carta(0, 2)]
        m.jugar_carta(0, Carta(3, 5))
        assert m.corazones_rotos is True  # ya estaba roto

    # --- Puntuación ---

    def test_puntuacion_simple(self):
        m = MotorCorazones()
        m.repartir()
        # Asignar bazas directamente para probar puntuación
        m.jugadores[0].bazas_ganadas = [Carta(3, 2), Carta(3, 3)]  # 2 pts
        m.jugadores[1].bazas_ganadas = [Carta(2, 12)]              # 13 pts
        m.jugadores[2].bazas_ganadas = [Carta(3, 4)]               # 1 pt
        m.jugadores[3].bazas_ganadas = [Carta(3, 5), Carta(3, 6), Carta(3, 7), Carta(3, 8),
                                        Carta(3, 9), Carta(3, 10), Carta(
                                            3, 11), Carta(3, 12),
                                        Carta(3, 13), Carta(3, 14)]  # 10 pts
        puntuaciones = m.calcular_puntuacion_mano()
        assert puntuaciones[0] == 2
        assert puntuaciones[1] == 13
        assert puntuaciones[2] == 1
        assert puntuaciones[3] == 10

    def test_deteccion_pleno(self):
        m = MotorCorazones()
        m.repartir()
        # Jugador 0 se lleva TODOS los 26 puntos
        todos_los_puntos = []
        for v in range(2, 15):
            todos_los_puntos.append(Carta(3, v))  # 13 corazones
        todos_los_puntos.append(Carta(2, 12))      # dama de picas
        # 14 cartas con puntos, pero solo 13 bazas.
        # En realidad hay 13 corazones + 1 dama = 14 cartas de puntos
        # Pero un jugador gana 13 bazas. Si gana todas, se lleva los 26 pts.
        # Simulamos 13 bazas: 12 corazones + dama de picas = 25 pts? No,
        # son 13 corazones + dama = 14 cartas de puntos en 52 cartas.
        # En 13 bazas de 4 cartas cada una = 52 cartas jugadas.
        # Un jugador que gana todas las bazas se lleva TODAS las cartas.
        m.jugadores[0].bazas_ganadas = [
            Carta(3, v) for v in range(2, 15)] + [Carta(2, 12)]
        # Asignamos cartas sin puntos a otros
        m.jugadores[1].bazas_ganadas = []
        m.jugadores[2].bazas_ganadas = []
        m.jugadores[3].bazas_ganadas = []
        puntuaciones = m.calcular_puntuacion_mano()
        # Pleno: el que hizo pleno suma 0, los demás +26
        assert puntuaciones[0] == 0
        assert puntuaciones[1] == 26
        assert puntuaciones[2] == 26
        assert puntuaciones[3] == 26

    def test_suma_cero_en_mano_normal(self):
        m = MotorCorazones()
        m.repartir()
        # Simular bazas que sumen 26 puntos en total
        m.jugadores[0].bazas_ganadas = [Carta(3, 2), Carta(3, 3)]
        m.jugadores[1].bazas_ganadas = [Carta(2, 12)]
        m.jugadores[2].bazas_ganadas = [Carta(3, 4)]
        m.jugadores[3].bazas_ganadas = [
            Carta(3, v) for v in range(5, 15)
        ]
        puntuaciones = m.calcular_puntuacion_mano()
        # Total debe ser 26 (suma de todos los puntos en el mazo)
        assert sum(puntuaciones) == 26

    # --- Jugar carta ilegal debe crashear ---

    def test_jugar_carta_ilegal_lanza_exception(self):
        m = MotorCorazones()
        m.repartir()
        # En baza 1, forzar mesa vacía e intentar jugar algo que no sea 2♣
        idx_dos_trebol = None
        for idx, j in enumerate(m.jugadores):
            if Carta(0, 2) in j.mano:
                idx_dos_trebol = idx
                break
        # Intentar que ese jugador juegue otra carta
        with pytest.raises(ValueError):
            m.jugar_carta(idx_dos_trebol, Carta(3, 14))

    # --- Resolución de baza ---

    def test_resolver_baza_gana_carta_mas_alta_del_palo_de_salida(self):
        m = MotorCorazones()
        m.repartir()
        m.numero_baza = 2
        m.corazones_rotos = True
        m.palo_de_salida = 0  # tréboles
        m.mesa = [
            (0, Carta(0, 5)),
            (1, Carta(0, 10)),
            (2, Carta(0, 3)),
            (3, Carta(1, 14)),  # diamante no compite
        ]
        ganador = m.resolver_baza()
        assert ganador == 1  # jugador 1 tiene 10 de tréboles (la más alta)

    def test_resolver_baza_solo_compiten_palo_de_salida(self):
        m = MotorCorazones()
        m.repartir()
        m.numero_baza = 2
        m.corazones_rotos = True
        m.palo_de_salida = 2  # picas
        m.mesa = [
            (0, Carta(2, 3)),
            (1, Carta(2, 14)),  # as de picas: gana
            (2, Carta(3, 14)),  # as de corazones: no compite
            (3, Carta(0, 14)),  # as de tréboles: no compite
        ]
        ganador = m.resolver_baza()
        assert ganador == 1

    # --- Test de integración: mano completa ---

    def test_jugar_mano_completa_sin_excepciones(self):
        m = MotorCorazones()
        m.repartir()
        resultados = m.jugar_mano()
        assert len(resultados) == 4
        assert sum(resultados) == 26 or sum(resultados) == 26 * 4
        # Después de 13 bazas, nadie debe tener cartas
        for j in m.jugadores:
            assert len(j.mano) == 0

    def test_numero_baza_avanza_correctamente(self):
        m = MotorCorazones()
        m.repartir()
        assert m.numero_baza == 1
        # Jugar las 13 bazas
        for _ in range(13):
            # Cada baza: 4 jugadores
            for _ in range(4):
                idx = m.obtener_jugador_actual()
                legales = m.obtener_jugadas_legales(idx)
                m.jugar_carta(idx, legales[0])
            m.resolver_baza()
        # Después de resolver 13 bazas, numero_baza llega a 14 (fin de mano)
        assert m.numero_baza == 14

    def test_puntuacion_acumulada_historica(self):
        m = MotorCorazones()
        m.repartir()
        m.jugar_mano()
        # Verificar que las puntuaciones históricas se actualizaron
        total_historico = sum(j.puntuacion_historica for j in m.jugadores)
        assert total_historico == 26 or total_historico == 26 * 4
