"""
Tests for Issue 2 · Fix del SOLO canónico.

El filtro original `consenso_individual > 0` en `identificar_solo`
descarta precisamente el caso estructuralmente más interesante: un
modelo con consenso_individual == 0 (ortogonal al resto del ensemble)
es el SOLO canónico en la narrativa del paper. Estos tests exigen
que se use `>= 0`, con la excepción del caso degenerado (todos a cero)
que debe devolver None sin lanzar excepción.
"""

import pytest

from analizador import identificar_solo


def _mk_resultados(nombres):
    """Stub de resultados del ensamblador con response por modelo."""
    return [{"model_name": n, "response": f"respuesta de {n}"} for n in nombres]


def _mk_consensos(pares):
    """
    pares: lista de tuplas (modelo, consenso_individual), en el orden
    que devolvería `calcular_consensos_individuales` (desc por consenso).
    """
    return [{"modelo": m, "consenso_individual": c} for m, c in pares]


# ── SOLO canónico: un modelo con consenso exactamente 0 debe exponerse ──────
def test_solo_canonico_con_consenso_cero():
    consensos = _mk_consensos([("A", 0.9), ("B", 0.9), ("C", 0.0)])
    resultados = _mk_resultados(["A", "B", "C"])

    solo = identificar_solo(consensos, resultados)

    assert solo is not None, "El SOLO canónico (consenso_individual == 0) no debe ser silenciado"
    assert solo["modelo"] == "C"
    assert solo["consenso_individual"] == 0.0
    assert solo["respuesta"] == "respuesta de C"


# ── SOLO ordinario: modelo con consenso bajo pero no cero ───────────────────
def test_solo_ordinario_consenso_bajo():
    consensos = _mk_consensos([("A", 0.5), ("B", 0.5), ("C", 0.1)])
    resultados = _mk_resultados(["A", "B", "C"])

    solo = identificar_solo(consensos, resultados)

    assert solo is not None
    assert solo["modelo"] == "C"
    assert solo["consenso_individual"] == 0.1


# ── Caso degenerado: todos los consensos a cero → None sin excepción ────────
def test_todos_consensos_cero_devuelve_none():
    consensos = _mk_consensos([("A", 0.0), ("B", 0.0), ("C", 0.0)])
    resultados = _mk_resultados(["A", "B", "C"])

    # No debe lanzar excepción
    solo = identificar_solo(consensos, resultados)

    assert solo is None, (
        "Cuando todos los modelos son mutuamente ortogonales no hay SOLO "
        "distinguible: la función debe devolver None."
    )


# ── Lista vacía → None (guarda la llamada desde pipelines vacíos) ───────────
def test_lista_vacia_devuelve_none():
    assert identificar_solo([], []) is None


# ── Payload incluye nota explicativa ────────────────────────────────────────
def test_solo_incluye_nota():
    consensos = _mk_consensos([("A", 0.8), ("B", 0.2)])
    resultados = _mk_resultados(["A", "B"])

    solo = identificar_solo(consensos, resultados)

    assert solo is not None
    assert "nota" in solo
    assert isinstance(solo["nota"], str) and len(solo["nota"]) > 0
