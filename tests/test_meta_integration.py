"""
Test de integración para Issue 3.

Ejecuta /api/run-ensemble con un Ensamblador mockeado (2 modelos, uno
válido y uno con api_error) y verifica que el meta.json generado:
  - pasa `validar_meta` contra el schema v1.0,
  - contiene los dos modelos en ensemble.modelos[],
  - persiste el sha256 del prompt (no el prompt en claro),
  - emite cookie chorus_browser_token y la refleja en el meta.
"""

import asyncio
import json
import os
import sys
import tempfile
import types
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _FakeEnsamblador:
    """Sustituye a Ensambladores.ensamblador_LLM.Ensamblador durante los tests.
    Simula dos modelos: uno responde bien, otro devuelve un error.
    """
    def __init__(self, modelos=None, timeout_seg=60):
        self.modelos = modelos or []
        self.modelos_filtrados = []
        self.resultados_crudos = []

    async def run(self, prompt):
        self.resultados_crudos = [
            {
                "model_name": "openai/gpt-4o",
                "response": (
                    "Análisis clínico simulado. El paciente presenta un cuadro "
                    "depresivo moderado sin ideación activa. Se recomienda "
                    "seguimiento ambulatorio con evaluación psiquiátrica."
                ),
                "timestamp": "2026-04-24T12:00:00",
                "provider_version": "gpt-4o-2024-05-13",
                "latency_ms": 1234,
                "api_error": None,
            },
            {
                "model_name": "anthropic/claude-3.5-sonnet",
                "response": (
                    "Otra perspectiva clínica simulada con léxico algo distinto "
                    "pero convergente: episodio depresivo mayor, leve-moderado, "
                    "sin urgencia, seguimiento indicado."
                ),
                "timestamp": "2026-04-24T12:00:01",
                "provider_version": "claude-3-5-sonnet-20241022",
                "latency_ms": 1600,
                "api_error": None,
            },
            {
                "model_name": "modelo/fallido",
                "response": "timeout consultando modelo/fallido",
                "timestamp": "2026-04-24T12:00:02",
                "provider_version": None,
                "latency_ms": 60000,
                "api_error": "timeout",
            },
        ]
        validos = [r for r in self.resultados_crudos if not r.get("api_error")]
        self.modelos_filtrados = [
            {
                "model_name": r["model_name"],
                "reason": r["response"][:120],
                "provider_version": r.get("provider_version"),
                "latency_ms": r.get("latency_ms"),
                "api_error": r.get("api_error"),
            }
            for r in self.resultados_crudos if r.get("api_error")
        ]
        return validos

    def guardar_resultados(self, respuestas, output_dir=None):
        return None


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    """Importa app.py con OUTPUT_PATH apuntando a tmp_path y Ensamblador mockeado."""
    monkeypatch.setenv("CHORUS_OUTPUT_PATH", str(tmp_path))
    # Recargar app.py para que tome el env var del tmp_path.
    for mod in list(sys.modules.keys()):
        if mod in ("app", "Ensambladores.ensamblador_LLM", "schemas", "schemas.meta_v1", "analizador"):
            del sys.modules[mod]
    import Ensambladores.ensamblador_LLM as ens_mod
    monkeypatch.setattr(ens_mod, "Ensamblador", _FakeEnsamblador)
    import app as app_module
    # También en el namespace de app (import ya ejecutado dentro del módulo).
    monkeypatch.setattr(app_module, "Ensamblador", _FakeEnsamblador)
    # Anular llamadas reales a OpenAI (embeddings + fusión) para que los
    # asserts sobre fallback_aplicado y la sección fusion sean
    # deterministas: load_dotenv reinyecta OPENAI_API_KEY desde .env al
    # importar analizador, así que limpiamos la global del módulo.
    import analizador
    monkeypatch.setattr(analizador, "API_KEY", "")
    app_module.app.testing = True
    return app_module, app_module.app.test_client(), tmp_path


def test_endpoint_produce_meta_v1_valido(app_client):
    app_module, client, out_dir = app_client
    prompt = (
        "Paciente adulto con síntomas compatibles con depresión: bajo estado de ánimo, "
        "anhedonia, alteración del sueño. Sin ideación suicida activa."
    )
    resp = client.post(
        "/api/run-ensemble",
        json={
            "prompt": prompt,
            "models": ["openai/gpt-4o", "anthropic/claude-3.5-sonnet", "modelo/fallido"],
            "modelType": "pay",
        },
    )
    assert resp.status_code == 200, resp.data
    body = resp.get_json()
    assert body["success"] is True
    assert "case_uuid" in body, body

    # Cookie emitida.
    set_cookie = resp.headers.get("Set-Cookie", "")
    assert "chorus_browser_token=" in set_cookie

    # Fichero meta.json creado.
    metas = list(out_dir.glob("ensamble_*.meta.json"))
    assert len(metas) == 1, f"se esperaba 1 meta.json, hay {len(metas)}"

    with open(metas[0], "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Validación contra el schema v1.0.
    app_module.validar_meta(meta)

    # Invariantes clave.
    import hashlib
    assert meta["schema_version"] == "1.0"
    assert meta["prompt_sha256"] == hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    assert meta["prompt_length_chars"] == len(prompt)
    assert meta["prompt_preview"] == prompt[:120]
    # El texto completo del prompt no debe aparecer en ningún punto del meta.
    assert prompt not in json.dumps(meta, ensure_ascii=False)

    # Ensemble refleja los tres modelos solicitados (válidos + filtrado).
    assert meta["ensemble"]["n_modelos"] == 3
    names = [m["name"] for m in meta["ensemble"]["modelos"]]
    assert set(names) == {"openai/gpt-4o", "anthropic/claude-3.5-sonnet", "modelo/fallido"}
    fallido = next(m for m in meta["ensemble"]["modelos"] if m["name"] == "modelo/fallido")
    assert fallido["api_error"] == "timeout"
    assert fallido["latency_ms"] == 60000

    # Browser token presente (el cliente no tenía cookie, el backend la emitió).
    assert meta["browser_token"] is not None
    assert len(meta["browser_token"]) >= 16  # uuid4

    # Sección embeddings presente y bien tipada. Sin OPENAI_API_KEY real
    # en el entorno de test la API no responde, por lo que el pipeline
    # cae a TF-IDF y `fallback_aplicado` debe ser True.
    emb = meta["embeddings"]
    assert isinstance(emb["modelo"], str) and emb["modelo"]
    assert isinstance(emb["dimensiones"], int)
    assert isinstance(emb["fallback_aplicado"], bool)
    assert emb["fallback_aplicado"] is True
