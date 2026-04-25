"""
Tests del umbral de selección para la fusión clínica.

Bug histórico (smoke test grande): con n_modelos=3,
`n_top = len(consensos_ind) * 2 // 3` daba 2, y la condición
`if len(mayores) >= 3` impedía invocar `generar_fusion`. Resultado:
ensembles de 3 modelos no producían síntesis y el meta.json salía con
`respuesta_fusionada_sha256: None`.

Contrato actual:
  - Ensembles de hasta 4 modelos: se usan todos para fusión, sin filtrar.
  - Ensembles mayores: filtrado a los 2/3 más consensuados (mínimo 3).
  - `generar_fusion` se invoca cuando hay al menos 2 mayores; con 1
    modelo válido el pipeline rechaza el análisis antes de llegar a la
    fusión, así que tampoco se invoca.
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def analizador_modulo(monkeypatch):
    """Importa analizador con API_KEY vacía: las llamadas reales a la
    API de embeddings se cortan en seco (caen a TF-IDF) y la fusión se
    mockea directamente sobre el símbolo del módulo en cada test."""
    for mod in list(sys.modules.keys()):
        if mod == "analizador":
            del sys.modules[mod]
    import analizador
    # load_dotenv() reinyecta OPENAI_API_KEY desde .env al importar; aquí
    # forzamos vacío para que `_obtener_embeddings` no haga HTTP real.
    monkeypatch.setattr(analizador, "API_KEY", "")
    return analizador


def _resp(name, texto):
    return {
        "model_name": name,
        "response": texto,
        "timestamp": "2026-04-25T00:00:00",
        "provider_version": None,
        "latency_ms": 100,
        "api_error": None,
    }


def test_ensemble_3_modelos_dispara_fusion(analizador_modulo):
    """Caso del bug: 3 modelos válidos deben pasar por generar_fusion."""
    a = analizador_modulo
    fake = AsyncMock(return_value=("síntesis simulada", 123))
    with patch.object(a, "generar_fusion", fake):
        rep = asyncio.run(a.dataAnalisis_interno([
            _resp("model/a", "Diagnóstico: depresión moderada con buen pronóstico tras inicio de ISRS."),
            _resp("model/b", "Cuadro depresivo de intensidad moderada, recomendable terapia cognitivo-conductual."),
            _resp("model/c", "Episodio depresivo con criterios DSM-5, considerar farmacoterapia y psicoterapia."),
        ]))

    assert "error" not in rep, f"pipeline devolvió error: {rep}"
    assert fake.await_count == 1, "generar_fusion debe invocarse para n=3"

    (top_arg,), _ = fake.await_args
    assert len(top_arg) == 3
    assert {d["model_name"] for d in top_arg} == {"model/a", "model/b", "model/c"}

    assert rep["respuesta_fusionada"] == "síntesis simulada"
    assert rep["fusion_latency_ms"] == 123
    assert set(rep["modelos_base"]) == {"model/a", "model/b", "model/c"}


def test_ensemble_1_modelo_no_dispara_fusion(analizador_modulo):
    """1 modelo válido: el pipeline aborta antes de la fusión y
    `generar_fusion` no se invoca. Diferencia clave con el bug: aquí
    la ausencia de síntesis es correcta y esperada."""
    a = analizador_modulo
    fake = AsyncMock(return_value=("no debería llamarse", 0))
    with patch.object(a, "generar_fusion", fake):
        rep = asyncio.run(a.dataAnalisis_interno([
            _resp("model/solo", "Una sola respuesta sin con quién compararse."),
        ]))

    assert "error" in rep, "ensembles <2 deben devolver error"
    assert fake.await_count == 0, "generar_fusion no debe invocarse con 1 modelo"
    assert rep.get("respuesta_fusionada") is None
