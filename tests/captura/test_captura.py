"""Tests del núcleo de captura: modelos, escritor, recolector."""
from __future__ import annotations

from typing import Iterator

import pytest

from src.captura.escritor import EscritorJsonl, cargar_partidas
from src.captura.modelos import (
    Jugada, RegistroMano, RegistroPartida, carta_a_str, str_a_carta_id,
)
from src.captura.puerto import (
    AdaptadorJuego, Evento, FinMano, FinPartida, InicioMano, InicioPartida,
    JugadaObservada, PaseAgente,
)
from src.captura.recolector import RecolectorPartidas


class TestHelpersCarta:
    def test_roundtrip_todas_las_cartas(self):
        for cid in range(52):
            assert str_a_carta_id(carta_a_str(cid)) == cid

    @pytest.mark.parametrize("texto,esperado_str", [
        ("10C", "10C"), ("ap", "AP"), ("Q♠", "QP"), ("2t", "2T"),
    ])
    def test_formatos_variados(self, texto, esperado_str):
        assert carta_a_str(str_a_carta_id(texto)) == esperado_str

    def test_carta_invalida(self):
        with pytest.raises(ValueError):
            str_a_carta_id("ZZ")


class TestRegistroSerializacion:
    def test_roundtrip_dict(self):
        partida = RegistroPartida(
            partida_id="abc", timestamp="2026-01-01T00:00:00+00:00",
            asiento_agente=1, fuente="test",
            manos=[RegistroMano(
                numero_mano=1, direccion_pase="izquierda",
                mano_inicial_agente=list(range(13)),
                pase_dado=[0, 1, 2], pase_recibido=[13, 14, 15],
                jugadas=[Jugada(0, 5, 1), Jugada(1, 9, 1)],
                puntuacion_mano=[1, 2, 3, 20],
            )],
            marcador_final=[10, 20, 30, 40], ranking_final=[0, 1, 2, 3],
        )
        d = partida.to_dict()
        recuperada = RegistroPartida.from_dict(d)
        assert recuperada == partida


class _AdaptadorFake(AdaptadorJuego):
    def __init__(self, eventos):
        self._evs = list(eventos)
        self.cerrado = False

    def eventos(self) -> Iterator[Evento]:
        yield from self._evs

    def cerrar(self) -> None:
        self.cerrado = True


def _stream_minimo():
    return [
        InicioPartida(asiento_agente=2, fuente="test"),
        InicioMano(numero_mano=1, direccion_pase="izquierda",
                   mano_agente=list(range(13))),
        PaseAgente(dadas=[0, 1, 2], recibidas=[13, 14, 15]),
        JugadaObservada(asiento=2, carta_id=0, baza=1),
        JugadaObservada(asiento=3, carta_id=1, baza=1),
        FinMano(puntuacion=[0, 0, 13, 13]),
        FinPartida(marcador=[10, 20, 5, 30]),
    ]


class TestRecolector:
    def test_agrega_y_calcula_ranking(self):
        fake = _AdaptadorFake(_stream_minimo())
        rec = RecolectorPartidas(adaptador=fake)
        partidas = rec.ejecutar()

        assert len(partidas) == 1
        p = partidas[0]
        assert p.asiento_agente == 2
        assert p.fuente == "test"
        assert len(p.manos) == 1
        assert p.manos[0].pase_dado == [0, 1, 2]
        assert len(p.manos[0].jugadas) == 2
        assert p.marcador_final == [10, 20, 5, 30]
        # menos puntos = mejor: asiento 2 (5) primero, luego 0,1,3
        assert p.ranking_final == [2, 0, 1, 3]
        assert fake.cerrado is True

    def test_escribe_y_relee(self, tmp_path):
        ruta = tmp_path / "sub" / "partidas.jsonl"
        escritor = EscritorJsonl(ruta)
        fake = _AdaptadorFake(_stream_minimo())
        RecolectorPartidas(adaptador=fake, escritor=escritor).ejecutar()

        leidas = cargar_partidas(ruta)
        assert len(leidas) == 1
        assert leidas[0].marcador_final == [10, 20, 5, 30]
        assert leidas[0].manos[0].jugadas[0].carta_id == 0

    def test_stream_inconsistente_falla(self):
        fake = _AdaptadorFake([JugadaObservada(asiento=0, carta_id=0, baza=1)])
        with pytest.raises(ValueError):
            RecolectorPartidas(adaptador=fake).ejecutar()
