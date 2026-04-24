"""
Test de integración para Issue 4.

Verifica que, cuando `calcular_cdi` devuelve el dict de error
(`matrix_contains_non_finite_values`), el pipeline completo:

  - no crashea (response 200),
  - propaga el estado de error al meta.json,
  - produce un meta.json v1.0 válido,
  - no emite NaN/Infinity crudos en el JSON de respuesta.

Monkey-patcheamos `calcular_cdi` para inyectar el estado de error de
forma determinista. Esto testea la propagación del error, que es lo
que la Issue 4 debe garantizar — la detección del non-finite está
cubierta por tests unitarios directos en `test_cdi.py`.
"""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _FakeEnsambladorNormal:
    """Ensamblador mock con tres modelos que devuelven textos razonables.
    El NaN se inyecta después, en `calcular_cdi`."""
    def __init__(self, modelos=None, timeout_seg=60):
        self.modelos = modelos or []
        self.modelos_filtrados = []
        self.resultados_crudos = []

    async def run(self, prompt):
        self.resultados_crudos = [
            {"model_name": "openai/gpt-4o",
             "response": "Diagnóstico A: episodio depresivo leve, seguimiento ambulatorio.",
             "timestamp": "2026-04-24T12:00:00", "provider_version": "v1",
             "latency_ms": 1000, "api_error": None},
            {"model_name": "openai/gpt-4o-mini",
             "response": "Diagnóstico B: síntomas compatibles con trastorno adaptativo, evaluación adicional.",
             "timestamp": "2026-04-24T12:00:01", "provider_version": "v1",
             "latency_ms": 1200, "api_error": None},
            {"model_name": "anthropic/claude-3.5-sonnet",
             "response": "Diagnóstico C: cuadro mixto ansioso-depresivo, psicoterapia recomendada.",
             "timestamp": "2026-04-24T12:00:02", "provider_version": "v1",
             "latency_ms": 1400, "api_error": None},
        ]
        return list(self.resultados_crudos)

    def guardar_resultados(self, respuestas, output_dir=None):
        return None


_CDI_ERROR_PAYLOAD = {
    "cdi": None,
    "cdi_geometric": None,
    "cdi_mean_dissent": None,
    "cdi_entropy": None,
    "cdi_det_raw": None,
    "n_modelos": 3,
    "nivel": "indeterminado",
    "etiqueta": "Matriz de similitud inestable. Alguna respuesta del ensemble no es procesable.",
    "color": "#888888",
    "determinante": None,
    "entropia": None,
    "error": "matrix_contains_non_finite_values",
}


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    monkeypatch.setenv("CHORUS_OUTPUT_PATH", str(tmp_path))
    for mod in list(sys.modules.keys()):
        if mod in ("app", "Ensambladores.ensamblador_LLM",
                   "analizador", "schemas", "schemas.meta_v1"):
            del sys.modules[mod]
    import Ensambladores.ensamblador_LLM as ens_mod
    monkeypatch.setattr(ens_mod, "Ensamblador", _FakeEnsambladorNormal)
    import analizador as ana_mod
    # Forzar el dict de error en cada llamada — simula matriz no finita.
    monkeypatch.setattr(ana_mod, "calcular_cdi", lambda _m: dict(_CDI_ERROR_PAYLOAD))
    import app as app_module
    monkeypatch.setattr(app_module, "Ensamblador", _FakeEnsambladorNormal)
    app_module.app.testing = True
    return app_module, app_module.app.test_client(), tmp_path


def test_pipeline_propaga_cdi_error_al_meta(app_client):
    app_module, client, out_dir = app_client

    resp = client.post(
        "/api/run-ensemble",
        json={
            "prompt": "Caso clínico forzando degradación del CDI vía matriz no finita.",
            "models": ["openai/gpt-4o", "openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet"],
            "modelType": "pay",
        },
    )
    assert resp.status_code == 200, resp.data
    body = resp.get_json()
    assert body.get("success") is True, body

    # El JSON de respuesta no debe contener NaN ni Infinity crudos
    # (no son JSON estándar; romperían parsers estrictos del frontend).
    raw = resp.get_data(as_text=True)
    assert "NaN" not in raw
    assert "Infinity" not in raw

    # El endpoint propaga el estado de error en consenso_data.
    cdi_ui = body["consenso_data"]["cdi"]
    assert cdi_ui["cdi"] is None
    assert cdi_ui["error"] == "matrix_contains_non_finite_values"

    # Meta.json producido y valido contra schema v1.0.
    metas = list(out_dir.glob("ensamble_*.meta.json"))
    assert len(metas) == 1
    with open(metas[0], "r", encoding="utf-8") as f:
        meta = json.load(f)
    app_module.validar_meta(meta)

    cdi = meta["cdi"]
    assert cdi is not None
    assert cdi["error"] == "matrix_contains_non_finite_values"
    assert cdi["cdi"] is None
    assert cdi["cdi_geometric"] is None
    assert cdi["nivel"] == "indeterminado"
