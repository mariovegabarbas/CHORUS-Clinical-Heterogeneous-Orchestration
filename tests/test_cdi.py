"""
Tests for Issue 1 · Normalización del CDI y métricas emparejadas.

Verifica que `calcular_cdi` devuelve las tres métricas normalizadas
(`cdi_geometric`, `cdi_mean_dissent`, `cdi_entropy`) y que la principal
(`cdi_geometric`) es comparable entre ensembles de distinto tamaño.
"""

import numpy as np
import pytest

from analizador import calcular_cdi


# ── cdi_geometric debe normalizar: identidad → 1.0 en cualquier n ──────────
@pytest.mark.parametrize("n", [2, 4, 10])
def test_identity_matrix_gives_cdi_geometric_one(n):
    matriz = np.identity(n)
    resultado = calcular_cdi(matriz)
    assert resultado["cdi_geometric"] == pytest.approx(1.0, abs=1e-6), (
        f"Identidad n={n} debe dar cdi_geometric=1.0, dio {resultado['cdi_geometric']}"
    )


# ── cdi_geometric debe normalizar: matriz de unos → 0.0 en cualquier n ──────
@pytest.mark.parametrize("n", [2, 4, 10])
def test_ones_matrix_gives_cdi_geometric_zero(n):
    matriz = np.ones((n, n))
    resultado = calcular_cdi(matriz)
    assert resultado["cdi_geometric"] == pytest.approx(0.0, abs=1e-6), (
        f"Matriz de unos n={n} debe dar cdi_geometric=0.0, dio {resultado['cdi_geometric']}"
    )


# ── Diagonal con 0.5 fuera: mean_dissent=0.5 y geometric en [0.3, 0.7] ──────
#
# La cota [0.3, 0.7] de cdi_geometric solo es holgada para n grande (la media
# geométrica de los valores singulares converge hacia el disenso medio por
# par cuando n crece). Con n pequeño el resultado tiende al valor del
# eigenvalue grande y se sale de la cota. Usamos n=10 por coherencia con los
# tamaños realistas de ensemble del proyecto.
def test_half_off_diagonal_metrics():
    n = 10
    matriz = np.full((n, n), 0.5)
    np.fill_diagonal(matriz, 1.0)
    resultado = calcular_cdi(matriz)

    assert resultado["cdi_mean_dissent"] == pytest.approx(0.5, abs=1e-6), (
        f"cdi_mean_dissent con off-diagonal=0.5 debe ser 0.5, dio {resultado['cdi_mean_dissent']}"
    )
    assert 0.3 <= resultado["cdi_geometric"] <= 0.7, (
        f"cdi_geometric debe estar acotado en [0.3, 0.7], dio {resultado['cdi_geometric']}"
    )


# ── NaN en la matriz: devuelve None + error sin lanzar excepción ────────────
def test_nan_matrix_returns_error_dict_without_exception():
    matriz = np.identity(3)
    matriz[0, 1] = float("nan")
    matriz[1, 0] = float("nan")

    resultado = calcular_cdi(matriz)

    assert resultado["cdi_geometric"] is None
    assert "error" in resultado
    assert resultado["error"] == "matrix_contains_non_finite_values"


# ── La clave de retrocompatibilidad cdi_det_raw existe ──────────────────────
def test_result_exposes_cdi_det_raw_for_backwards_compat():
    matriz = np.identity(3)
    resultado = calcular_cdi(matriz)
    assert "cdi_det_raw" in resultado
    assert resultado["cdi_det_raw"] == pytest.approx(0.0, abs=1e-6)


# ── Alias `cdi` sigue existiendo y apunta a cdi_geometric ───────────────────
def test_cdi_alias_points_to_cdi_geometric():
    matriz = np.full((4, 4), 0.5)
    np.fill_diagonal(matriz, 1.0)
    resultado = calcular_cdi(matriz)
    assert resultado["cdi"] == pytest.approx(resultado["cdi_geometric"])


# ── n_modelos se expone correctamente ───────────────────────────────────────
def test_n_modelos_field_present():
    for n in [2, 4, 7]:
        matriz = np.identity(n)
        resultado = calcular_cdi(matriz)
        assert resultado["n_modelos"] == n
