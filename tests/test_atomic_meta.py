"""
Tests for Issue 6 · Guardado atómico de meta.json.

Contrato:
  - `_guardar_meta` escribe primero en `<base>.meta.json.tmp`, hace
    fsync y usa `os.replace` para producir el fichero final.
  - Un fallo durante la escritura (simulado) NO deja `<base>.meta.json`
    a medio escribir: o bien no existe, o bien está completo.
  - Un fallo durante la escritura NO deja `<base>.meta.json.tmp`
    huérfano: se limpia en el except.
  - Reemplazar un meta.json existente no cruza un estado intermedio
    en el que el fichero final esté parcialmente escrito (os.replace
    es atómico y nunca borra el destino antes de tener el origen).
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def app_module(tmp_path, monkeypatch):
    monkeypatch.setenv("CHORUS_OUTPUT_PATH", str(tmp_path))
    for mod in list(sys.modules.keys()):
        if mod in ("app", "schemas", "schemas.meta_v1"):
            del sys.modules[mod]
    import app as app_module
    return app_module, tmp_path


def _meta_payload():
    return {
        "schema_version": "1.0",
        "case_uuid": "test-uuid",
        "timestamp_utc": "2026-04-24T12:00:00Z",
        "anything": "test",
    }


def test_guardar_meta_produce_fichero_final(app_module):
    m, out = app_module
    m._guardar_meta("ensamble_AAA", _meta_payload())
    final = out / "ensamble_AAA.meta.json"
    assert final.exists(), "El meta final debe existir"
    data = json.loads(final.read_text(encoding="utf-8"))
    assert data["case_uuid"] == "test-uuid"


def test_guardar_meta_no_deja_tmp_en_ruta_feliz(app_module):
    m, out = app_module
    m._guardar_meta("ensamble_BBB", _meta_payload())
    tmps = list(out.glob("*.meta.json.tmp"))
    assert tmps == [], f"No debe quedar fichero .tmp tras éxito, encontrado: {tmps}"


def test_guardar_meta_fallo_no_deja_meta_final_parcial(app_module, monkeypatch):
    """Si la serialización JSON falla a mitad de la escritura, el
    `<base>.meta.json` final nunca debe existir en estado parcial:
    el patrón tmp + os.replace lo garantiza porque el destino solo
    se toca cuando el tmp ya está completo y fsync'd."""
    m, out = app_module

    class _Boom(Exception):
        pass

    def _raise(*a, **kw):
        raise _Boom("simulated serialization failure")

    monkeypatch.setattr(m.json, "dump", _raise)

    with pytest.raises(_Boom):
        m._guardar_meta("ensamble_CCC", _meta_payload())

    final = out / "ensamble_CCC.meta.json"
    assert not final.exists(), (
        "Un fallo de escritura NO debe producir un meta.json final corrupto. "
        "La atomicidad de os.replace lo garantiza."
    )


def test_guardar_meta_fallo_limpia_tmp(app_module, monkeypatch):
    """Si la escritura falla después de abrir el .tmp, el .tmp se
    limpia para no dejar basura en el directorio de resultados."""
    m, out = app_module

    def _dump_then_fail(obj, fp, *a, **kw):
        # Escribimos parcialmente antes de fallar, para simular un .tmp
        # con contenido basura en el volumen.
        fp.write("{partial")
        raise RuntimeError("simulated mid-write failure")

    monkeypatch.setattr(m.json, "dump", _dump_then_fail)

    with pytest.raises(RuntimeError):
        m._guardar_meta("ensamble_DDD", _meta_payload())

    tmps = list(out.glob("*.meta.json.tmp"))
    assert tmps == [], (
        f"El .tmp huérfano debe limpiarse tras un fallo de escritura, "
        f"encontrado: {tmps}"
    )
    # Y por supuesto no debe existir el meta final.
    assert not (out / "ensamble_DDD.meta.json").exists()


def test_guardar_meta_sobrescribe_atomicamente_existente(app_module):
    """Si ya existe un meta.json del mismo `filename_base` (p.ej.
    tras un reintento manual), la sobrescritura debe dejar el fichero
    final completo, no un estado intermedio."""
    m, out = app_module
    final = out / "ensamble_EEE.meta.json"

    m._guardar_meta("ensamble_EEE", _meta_payload())
    assert final.exists()
    first_len = final.stat().st_size

    bigger = _meta_payload()
    bigger["extra"] = "x" * 5000
    m._guardar_meta("ensamble_EEE", bigger)

    assert final.exists()
    data = json.loads(final.read_text(encoding="utf-8"))
    assert data["extra"] == "x" * 5000
    assert final.stat().st_size > first_len
    # No deben quedar ficheros .tmp tras la sobrescritura.
    assert list(out.glob("*.meta.json.tmp")) == []
