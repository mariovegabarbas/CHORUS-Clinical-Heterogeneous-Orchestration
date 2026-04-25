"""
Tests for Issue 3 · Schema v1.0 del meta.json.

Verifica que `validar_meta` acepta un payload completo y rechaza
payloads con campos obligatorios ausentes o tipos incorrectos.
"""

import copy
import pytest

from schemas.meta_v1 import (
    SCHEMA_VERSION,
    MetaValidationError,
    construir_meta_base,
    validar_meta,
)


def _meta_valido_minimo():
    """Devuelve un payload v1.0 mínimo y bien tipado para testing."""
    meta = construir_meta_base()
    meta["case_uuid"] = "abc-123"
    meta["timestamp_utc"] = "2026-04-24T10:00:00Z"
    meta["timestamp_local"] = "2026-04-24T12:00:00+02:00"
    meta["prompt_sha256"] = "0" * 64
    meta["prompt_length_chars"] = 42
    meta["prompt_preview"] = "ejemplo de preview"
    meta["ensemble"] = {
        "n_modelos": 2,
        "model_type": "free",
        "modelos": [
            {
                "name": "openai/gpt-4o",
                "provider_version": "gpt-4o-2024-05-13",
                "latency_ms": 1200,
                "response_length_chars": 800,
                "embedding_truncated": False,
                "api_error": None,
            },
            {
                "name": "anthropic/claude-3.5-sonnet",
                "provider_version": None,
                "latency_ms": 1500,
                "response_length_chars": 0,
                "embedding_truncated": False,
                "api_error": "timeout",
            },
        ],
    }
    meta["matrices"] = {
        "tfidf": [[1.0, 0.5], [0.5, 1.0]],
        "embed": None,
        "principal": "tfidf",
    }
    meta["consenso_global"] = 0.5
    meta["chorus_version"] = "deadbeef"
    return meta


def test_validar_acepta_meta_minimo_valido():
    meta = _meta_valido_minimo()
    # No debe lanzar.
    validar_meta(meta)


def test_validar_rechaza_schema_version_incorrecta():
    meta = _meta_valido_minimo()
    meta["schema_version"] = "0.9"
    with pytest.raises(MetaValidationError, match="schema_version"):
        validar_meta(meta)


@pytest.mark.parametrize(
    "missing_field",
    [
        "schema_version", "case_uuid", "timestamp_utc", "timestamp_local",
        "prompt_sha256", "prompt_length_chars", "prompt_preview",
        "ensemble", "determinismo", "fusion", "matrices", "embeddings",
        "consensos_individuales", "chorus_version",
    ],
)
def test_validar_rechaza_campos_top_ausentes(missing_field):
    meta = _meta_valido_minimo()
    del meta[missing_field]
    with pytest.raises(MetaValidationError, match=missing_field):
        validar_meta(meta)


@pytest.mark.parametrize(
    "missing_subfield",
    ["modelo", "dimensiones", "fallback_aplicado"],
)
def test_validar_rechaza_embeddings_subcampos_ausentes(missing_subfield):
    meta = _meta_valido_minimo()
    del meta["embeddings"][missing_subfield]
    with pytest.raises(MetaValidationError, match=missing_subfield):
        validar_meta(meta)


def test_validar_rechaza_embeddings_tipo_incorrecto():
    meta = _meta_valido_minimo()
    meta["embeddings"]["dimensiones"] = "no soy int"
    with pytest.raises(MetaValidationError, match="dimensiones"):
        validar_meta(meta)


def test_validar_rechaza_modelo_sin_campos():
    meta = _meta_valido_minimo()
    meta["ensemble"]["modelos"].append({"name": "solo_name"})
    with pytest.raises(MetaValidationError, match=r"modelos\[2\]"):
        validar_meta(meta)


def test_validar_rechaza_principal_invalido():
    meta = _meta_valido_minimo()
    meta["matrices"]["principal"] = "otra_cosa"
    with pytest.raises(MetaValidationError, match="principal"):
        validar_meta(meta)


def test_validar_rechaza_tipo_incorrecto():
    meta = _meta_valido_minimo()
    meta["prompt_length_chars"] = "no soy int"
    with pytest.raises(MetaValidationError, match="prompt_length_chars"):
        validar_meta(meta)


def test_construir_meta_base_es_valido_tras_rellenar_minimos():
    """El helper produce una base que, rellenada con los mínimos
    obligatorios, pasa la validación sin excepción."""
    meta = construir_meta_base()
    assert meta["schema_version"] == SCHEMA_VERSION
    assert meta["determinismo"]["temperature"] == 0.2
    assert meta["determinismo"]["seed"] is None
    assert isinstance(meta["ensemble"]["modelos"], list)
    # Un payload recién construido SIN campos rellenados debe rechazarse
    # por case_uuid vacío? No — case_uuid es str, "" cumple el tipo.
    # Pero timestamp_utc="" también cumple str. El schema valida tipo,
    # no contenido. Esto es consistente con el diseño: la semántica
    # (p.ej. que el UUID sea real) es responsabilidad del llamante.
    validar_meta(meta)


def test_validar_rechaza_payload_no_dict():
    with pytest.raises(MetaValidationError, match="dict"):
        validar_meta("esto no es un dict")
    with pytest.raises(MetaValidationError, match="dict"):
        validar_meta(None)
