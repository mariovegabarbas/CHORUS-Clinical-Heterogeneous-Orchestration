"""
Schema v1.0 del meta.json de CHORUS.

Decisión clave (RGPD):
    El meta.json **NO** contiene el texto completo del prompt del
    visitante. Solo `prompt_sha256` (detectar reejecuciones del mismo
    caso) y `prompt_preview` (primeros 120 caracteres, orientación
    humana). Esto saca a la demo pública del perímetro RGPD de forma
    limpia.

El schema es estricto: `validar_meta` lanza `MetaValidationError`
si falta cualquier campo obligatorio o si un campo tiene tipo
incorrecto. Los campos cuyo valor es opcional (`case_reference_id`,
`session_code`) deben estar presentes en el payload aunque sea con
valor `None`.
"""

from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "1.0"


class MetaValidationError(ValueError):
    """Payload meta.json no cumple el schema v1.0."""


# ── Campos requeridos a nivel top ─────────────────────────────────────────────
_REQUIRED_TOP = {
    "schema_version": str,
    "case_uuid": str,
    "case_reference_id": (str, type(None)),
    "session_code": (str, type(None)),
    "browser_token": (str, type(None)),
    "timestamp_utc": str,
    "timestamp_local": str,
    "prompt_sha256": str,
    "prompt_length_chars": int,
    "prompt_preview": str,
    "ensemble": dict,
    "determinismo": dict,
    "fusion": dict,
    "matrices": dict,
    "cdi": (dict, type(None)),
    "solo": (dict, type(None)),
    "divergencia_capas": (dict, type(None)),
    "consenso_global": (int, float, type(None)),
    "consensos_individuales": list,
    "respuesta_fusionada_sha256": (str, type(None)),
    "chorus_version": str,
}

_REQUIRED_ENSEMBLE = {
    "n_modelos": int,
    "model_type": str,
    "modelos": list,
}

_REQUIRED_MODEL = {
    "name": str,
    "provider_version": (str, type(None)),
    "latency_ms": (int, float, type(None)),
    "response_length_chars": int,
    "embedding_truncated": bool,
    "api_error": (str, type(None)),
}

_REQUIRED_DETERMINISMO = {
    "temperature": (int, float),
    "seed": (int, type(None)),
    "nota": str,
}

_REQUIRED_FUSION = {
    "modelo": (str, type(None)),
    "latency_ms": (int, float, type(None)),
    "max_tokens": (int, type(None)),
    "temperature": (int, float, type(None)),
}

_REQUIRED_MATRICES = {
    "tfidf": (list, type(None)),
    "embed": (list, type(None)),
    "principal": str,
}


def _check(payload: dict, specs: dict, path: str) -> None:
    for key, expected in specs.items():
        if key not in payload:
            raise MetaValidationError(f"falta campo obligatorio: {path}.{key}")
        if not isinstance(payload[key], expected):
            raise MetaValidationError(
                f"tipo incorrecto en {path}.{key}: "
                f"esperado {expected}, recibido {type(payload[key]).__name__}"
            )


def validar_meta(payload: Any) -> None:
    """Valida un payload meta.json contra el schema v1.0.

    Lanza `MetaValidationError` si:
      - el payload no es un dict,
      - falta algún campo obligatorio,
      - algún campo tiene tipo incorrecto,
      - `schema_version` no es exactamente "1.0".
    """
    if not isinstance(payload, dict):
        raise MetaValidationError("payload debe ser un dict")

    _check(payload, _REQUIRED_TOP, "meta")

    if payload["schema_version"] != SCHEMA_VERSION:
        raise MetaValidationError(
            f"schema_version debe ser {SCHEMA_VERSION!r}, "
            f"recibido {payload['schema_version']!r}"
        )

    _check(payload["ensemble"], _REQUIRED_ENSEMBLE, "meta.ensemble")
    for i, m in enumerate(payload["ensemble"]["modelos"]):
        if not isinstance(m, dict):
            raise MetaValidationError(
                f"meta.ensemble.modelos[{i}] debe ser dict"
            )
        _check(m, _REQUIRED_MODEL, f"meta.ensemble.modelos[{i}]")

    _check(payload["determinismo"], _REQUIRED_DETERMINISMO, "meta.determinismo")
    _check(payload["fusion"], _REQUIRED_FUSION, "meta.fusion")
    _check(payload["matrices"], _REQUIRED_MATRICES, "meta.matrices")

    if payload["matrices"]["principal"] not in ("tfidf", "embed"):
        raise MetaValidationError(
            f"meta.matrices.principal debe ser 'tfidf' o 'embed', "
            f"recibido {payload['matrices']['principal']!r}"
        )


def construir_meta_base() -> dict:
    """Devuelve un payload meta v1.0 con estructura completa y valores por
    defecto. Los campos que dependen de la ejecución se rellenan después.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "case_uuid": "",
        "case_reference_id": None,
        "session_code": None,
        "browser_token": None,
        "timestamp_utc": "",
        "timestamp_local": "",
        "prompt_sha256": "",
        "prompt_length_chars": 0,
        "prompt_preview": "",
        "ensemble": {
            "n_modelos": 0,
            "model_type": "free",
            "modelos": [],
        },
        "determinismo": {
            "temperature": 0.2,
            "seed": None,
            "nota": (
                "Output no determinista. Repeticiones del mismo prompt "
                "producen outputs distintos."
            ),
        },
        "fusion": {
            "modelo": None,
            "latency_ms": None,
            "max_tokens": None,
            "temperature": None,
        },
        "matrices": {
            "tfidf": None,
            "embed": None,
            "principal": "tfidf",
        },
        "cdi": None,
        "solo": None,
        "divergencia_capas": None,
        "consenso_global": None,
        "consensos_individuales": [],
        "respuesta_fusionada_sha256": None,
        "chorus_version": "unknown",
    }
