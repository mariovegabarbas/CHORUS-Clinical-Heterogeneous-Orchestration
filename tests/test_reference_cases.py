"""
Tests for Issue 5 · Casos de referencia y session_code opcional.

Contrato:
  - GET /api/reference_cases lista los casos SIN `texto_completo`.
  - POST /api/run-ensemble acepta `case_reference_id`: el backend carga
    el texto desde el catálogo y el visitante no necesita haberlo visto.
  - `prompt_sha256` es el sha del `texto_completo`, no del id (permite
    detectar reejecuciones del mismo caso de forma consistente).
  - `session_code` se persiste literal en el meta.json.
  - `case_reference_id` se persiste en el meta.json.
  - Un `case_reference_id` inexistente devuelve error controlado.
"""

import hashlib
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_CASO_TEXTO = (
    "Caso clínico de prueba, anonimizado, con suficiente longitud como para "
    "pasar el umbral de 10 caracteres del prompt. Paciente ficticio con sintomatología "
    "neutra descrita de forma extensa y reproducible entre ejecuciones."
)

_CASOS_FIXTURE = {
    "schema_version": "1.0",
    "casos": [
        {
            "id": "TEST-001",
            "titulo": "Caso de test",
            "descripcion_corta": "Descripción corta del caso de test.",
            "texto_completo": _CASO_TEXTO,
            "ensemble_recomendado": ["openai/gpt-4o", "anthropic/claude-3.5-sonnet"],
            "notas_investigador": "Fixture para pytest.",
        },
    ],
}


class _FakeEnsamblador:
    def __init__(self, modelos=None, timeout_seg=60):
        self.modelos = modelos or []
        self.modelos_filtrados = []
        self.resultados_crudos = []

    async def run(self, prompt):
        self.resultados_crudos = [
            {"model_name": m["name"],
             "response": f"respuesta simulada para {m['name']}: diagnóstico tentativo con matices clínicos razonables.",
             "timestamp": "2026-04-24T12:00:00",
             "provider_version": "v1",
             "latency_ms": 1000,
             "api_error": None}
            for m in self.modelos
        ]
        return list(self.resultados_crudos)

    def guardar_resultados(self, respuestas, output_dir=None):
        return None


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    cases_path = tmp_path / "casos_test.json"
    cases_path.write_text(json.dumps(_CASOS_FIXTURE), encoding="utf-8")
    monkeypatch.setenv("CHORUS_OUTPUT_PATH", str(tmp_path))
    monkeypatch.setenv("CHORUS_REFERENCE_CASES", str(cases_path))

    for mod in list(sys.modules.keys()):
        if mod in ("app", "Ensambladores.ensamblador_LLM",
                   "analizador", "schemas", "schemas.meta_v1"):
            del sys.modules[mod]
    import Ensambladores.ensamblador_LLM as ens_mod
    monkeypatch.setattr(ens_mod, "Ensamblador", _FakeEnsamblador)
    import app as app_module
    monkeypatch.setattr(app_module, "Ensamblador", _FakeEnsamblador)
    app_module.app.testing = True
    return app_module, app_module.app.test_client(), tmp_path


def test_reference_cases_endpoint_no_expone_texto_completo(app_client):
    _, client, _ = app_client
    resp = client.get("/api/reference_cases")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert len(body["casos"]) == 1
    caso = body["casos"][0]
    assert caso["id"] == "TEST-001"
    assert caso["titulo"] == "Caso de test"
    assert caso["descripcion_corta"] == "Descripción corta del caso de test."
    assert caso["ensemble_recomendado"] == ["openai/gpt-4o", "anthropic/claude-3.5-sonnet"]
    assert "texto_completo" not in caso, (
        "texto_completo no debe exponerse al navegador (decisión Issue 5)"
    )
    assert "notas_investigador" not in caso


def test_ejecutar_con_case_reference_id_persiste_id_y_sha(app_client):
    app_module, client, out_dir = app_client
    resp = client.post(
        "/api/run-ensemble",
        json={
            "prompt": "",  # se ignora cuando viene case_reference_id
            "case_reference_id": "TEST-001",
            "models": ["openai/gpt-4o", "anthropic/claude-3.5-sonnet"],
            "modelType": "pay",
            "session_code": "KARO-T0-P002",
        },
    )
    assert resp.status_code == 200, resp.data
    body = resp.get_json()
    assert body["success"] is True

    metas = list(out_dir.glob("ensamble_*.meta.json"))
    assert len(metas) == 1
    with open(metas[0], "r", encoding="utf-8") as f:
        meta = json.load(f)
    app_module.validar_meta(meta)

    assert meta["case_reference_id"] == "TEST-001"
    assert meta["session_code"] == "KARO-T0-P002"
    # sha256 = sha del texto_completo del catálogo, NO del id.
    expected_sha = hashlib.sha256(_CASO_TEXTO.encode("utf-8")).hexdigest()
    assert meta["prompt_sha256"] == expected_sha
    assert meta["prompt_length_chars"] == len(_CASO_TEXTO)
    assert meta["prompt_preview"] == _CASO_TEXTO[:120]


def test_reejecucion_mismo_caso_produce_mismo_sha(app_client):
    """La detección de reejecuciones del mismo caso se basa en
    `prompt_sha256`, que debe ser estable entre llamadas distintas
    sobre el mismo `case_reference_id`."""
    import time
    _, client, out_dir = app_client
    for _ in range(2):
        r = client.post(
            "/api/run-ensemble",
            json={"case_reference_id": "TEST-001",
                  "models": ["openai/gpt-4o", "anthropic/claude-3.5-sonnet"],
                  "modelType": "pay"},
        )
        assert r.status_code == 200
        # El filename_base usa timestamp con resolución de segundo:
        # dos llamadas en el mismo segundo sobrescriben el mismo meta.
        # Esperar 1.05s para garantizar dos ficheros distintos.
        time.sleep(1.05)

    metas = sorted(out_dir.glob("ensamble_*.meta.json"))
    assert len(metas) >= 2
    shas = []
    for p in metas:
        with open(p, encoding="utf-8") as f:
            shas.append(json.load(f)["prompt_sha256"])
    assert len(set(shas)) == 1, (
        f"prompt_sha256 debe ser estable entre reejecuciones del mismo caso, "
        f"obtuve {len(set(shas))} shas distintos."
    )


def test_case_reference_id_inexistente_devuelve_error(app_client):
    _, client, _ = app_client
    resp = client.post(
        "/api/run-ensemble",
        json={"case_reference_id": "NO-EXISTE",
              "models": ["openai/gpt-4o"],
              "modelType": "pay"},
    )
    assert resp.status_code == 200  # no 500: error controlado
    body = resp.get_json()
    assert body["success"] is False
    assert "NO-EXISTE" in body["error"]


def test_session_code_sin_case_reference_persiste(app_client):
    """session_code también funciona en análisis ad hoc (sin caso
    de referencia): el investigador puede marcar con session_code
    análisis hechos sobre texto propio."""
    app_module, client, out_dir = app_client
    resp = client.post(
        "/api/run-ensemble",
        json={
            "prompt": "Caso clínico ad hoc con suficiente longitud para pasar validación básica.",
            "models": ["openai/gpt-4o", "anthropic/claude-3.5-sonnet"],
            "modelType": "pay",
            "session_code": "KARO-T1-P002",
        },
    )
    assert resp.status_code == 200
    metas = list(out_dir.glob("ensamble_*.meta.json"))
    assert len(metas) == 1
    with open(metas[0], encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["session_code"] == "KARO-T1-P002"
    assert meta["case_reference_id"] is None


def test_casos_referencia_json_del_repo_carga_sin_errores():
    """El fichero real `casos_referencia.json` del repo debe parsear
    limpiamente y respetar la estructura esperada. Esto evita que un
    typo o un caso mal formado rompa el endpoint en producción."""
    import json as _json
    from pathlib import Path as _Path
    p = _Path(__file__).resolve().parent.parent / "casos_referencia.json"
    assert p.exists(), "Falta casos_referencia.json en la raíz del repo"
    data = _json.loads(p.read_text(encoding="utf-8"))
    assert data.get("schema_version") == "1.0"
    casos = data.get("casos")
    assert isinstance(casos, list) and len(casos) >= 1
    for c in casos:
        for campo in ("id", "titulo", "descripcion_corta", "texto_completo"):
            assert c.get(campo), f"Caso {c.get('id')} sin {campo}"
