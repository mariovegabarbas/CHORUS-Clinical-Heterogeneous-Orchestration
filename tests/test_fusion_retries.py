"""
Tests for Issue 6 · Fusión no bloqueante y reintentos con backoff.

Contrato:
  - `_llamar_chatgpt` es async (no bloquea el event loop).
  - Un 200 en el primer intento devuelve el texto sin esperas.
  - Un 429 seguido de 200 termina en éxito tras una espera (asyncio.sleep
    patcheado para no dormir realmente en los tests).
  - Tres 429 consecutivos agotan reintentos y devuelven (None, latency).
  - Un 5xx seguido de 200 termina en éxito.
  - Un 400 no se reintenta (devuelve None inmediatamente).
  - Sin API_KEY, no se llama a la API (devuelve None, 0ms).
"""

import asyncio
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _FakeResp:
    def __init__(self, status, json_data=None):
        self.status = status
        self._json = json_data or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def json(self):
        return self._json


class _FakeSession:
    """Sustituye `aiohttp.ClientSession` en `analizador`. Devuelve
    respuestas de la cola `responses` en orden."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def post(self, url, json=None, headers=None):
        self.calls += 1
        if not self.responses:
            return _FakeResp(500)
        return self.responses.pop(0)


def _patch_module(analizador, responses):
    """Sustituye aiohttp.ClientSession y asyncio.sleep en el módulo
    analizador. Devuelve la instancia de sesión para poder inspeccionarla."""
    session = _FakeSession(responses)

    def _client_session_factory(*a, **kw):
        return session

    # Evita esperas reales en los tests.
    async def _no_sleep(_):
        return None

    return session, _client_session_factory, _no_sleep


@pytest.fixture
def analizador_con_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    for mod in list(sys.modules.keys()):
        if mod == "analizador":
            del sys.modules[mod]
    import analizador
    return analizador


def _run(coro):
    return asyncio.run(coro)


def test_primer_200_termina_sin_esperar(analizador_con_api_key):
    a = analizador_con_api_key
    session, factory, no_sleep = _patch_module(a, [
        _FakeResp(200, {"choices": [{"message": {"content": "síntesis ok"}}]}),
    ])
    with patch.object(a.aiohttp, "ClientSession", factory), \
         patch.object(a.asyncio, "sleep", no_sleep):
        texto, ms = _run(a._llamar_chatgpt([{"role": "user", "content": "x"}]))
    assert texto == "síntesis ok"
    assert isinstance(ms, int) and ms >= 0
    assert session.calls == 1


def test_429_luego_200_termina_en_exito(analizador_con_api_key):
    a = analizador_con_api_key
    session, factory, no_sleep = _patch_module(a, [
        _FakeResp(429),
        _FakeResp(200, {"choices": [{"message": {"content": "retry ok"}}]}),
    ])
    with patch.object(a.aiohttp, "ClientSession", factory), \
         patch.object(a.asyncio, "sleep", no_sleep):
        texto, _ = _run(a._llamar_chatgpt([{"role": "user", "content": "x"}]))
    assert texto == "retry ok"
    assert session.calls == 2


def test_tres_429_agota_reintentos(analizador_con_api_key):
    a = analizador_con_api_key
    session, factory, no_sleep = _patch_module(a, [
        _FakeResp(429), _FakeResp(429), _FakeResp(429),
    ])
    with patch.object(a.aiohttp, "ClientSession", factory), \
         patch.object(a.asyncio, "sleep", no_sleep):
        texto, ms = _run(a._llamar_chatgpt([{"role": "user", "content": "x"}]))
    assert texto is None
    assert session.calls == a.RETRY_MAX_ATTEMPTS == 3
    assert isinstance(ms, int)


def test_503_luego_200_termina_en_exito(analizador_con_api_key):
    a = analizador_con_api_key
    session, factory, no_sleep = _patch_module(a, [
        _FakeResp(503),
        _FakeResp(200, {"choices": [{"message": {"content": "ok tras 5xx"}}]}),
    ])
    with patch.object(a.aiohttp, "ClientSession", factory), \
         patch.object(a.asyncio, "sleep", no_sleep):
        texto, _ = _run(a._llamar_chatgpt([{"role": "user", "content": "x"}]))
    assert texto == "ok tras 5xx"
    assert session.calls == 2


def test_400_no_se_reintenta(analizador_con_api_key):
    """Un 4xx no-429 no es transitorio: no tiene sentido esperar."""
    a = analizador_con_api_key
    session, factory, no_sleep = _patch_module(a, [
        _FakeResp(400),
        _FakeResp(200, {"choices": [{"message": {"content": "no se llega"}}]}),
    ])
    with patch.object(a.aiohttp, "ClientSession", factory), \
         patch.object(a.asyncio, "sleep", no_sleep):
        texto, _ = _run(a._llamar_chatgpt([{"role": "user", "content": "x"}]))
    assert texto is None
    assert session.calls == 1, "4xx no-429 no debe reintentar"


def test_sin_api_key_no_llama(analizador_con_api_key, monkeypatch):
    """Sin API_KEY, ni se abre la sesión: devuelve (None, latencia).

    Se usa `monkeypatch.setattr` sobre la variable de módulo porque
    `.env` puede reinyectar la clave si se elimina solo del entorno.
    """
    a = analizador_con_api_key
    monkeypatch.setattr(a, "API_KEY", "")

    session_was_opened = {"flag": False}

    def factory(*args, **kwargs):
        session_was_opened["flag"] = True
        return _FakeSession([])

    with patch.object(a.aiohttp, "ClientSession", factory):
        texto, ms = _run(a._llamar_chatgpt([{"role": "user", "content": "x"}]))
    assert texto is None
    assert isinstance(ms, int)
    assert session_was_opened["flag"] is False, (
        "Sin API_KEY no debe abrirse una sesión aiohttp."
    )


def test_generar_fusion_menor_que_dos_respuestas_no_llama_api(analizador_con_api_key):
    """Con 0 o 1 respuesta no hay nada que sintetizar: devuelve el
    texto único (o "") y latencia None, sin tocar la API."""
    a = analizador_con_api_key
    texto_vacio, ms_vacio = _run(a.generar_fusion([]))
    assert texto_vacio == ""
    assert ms_vacio is None

    texto_uno, ms_uno = _run(a.generar_fusion([
        {"model_name": "m1", "response": "única respuesta", "consenso_individual": 1.0},
    ]))
    assert texto_uno == "única respuesta"
    assert ms_uno is None
