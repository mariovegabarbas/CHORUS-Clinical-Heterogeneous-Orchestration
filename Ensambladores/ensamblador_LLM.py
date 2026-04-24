import aiohttp
import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

API_URL    = "https://openrouter.ai/api/v1/chat/completions"
API_KEY    = os.environ.get("OPENROUTER_API_KEY", "")
OUTPUT_PATH = os.environ.get("CHORUS_OUTPUT_PATH", "resultados")

# Reintentos con backoff exponencial para 429 y 5xx. Tres intentos con
# espera 1s/2s/4s absorben rate limiting ocasional de OpenRouter o fallos
# transitorios del proveedor upstream. 4xx no-429 no se reintentan.
RETRY_MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = (1.0, 2.0, 4.0)

def _es_error(texto: str) -> bool:
    t = texto.strip()
    if len(t) < 20:
        return True
    # Errores de API: empiezan con "Error {" / "Error [" (dict/lista JSON) o prefijos técnicos exactos
    tl = t.lower()
    if tl.startswith("request failed:"):
        return True
    if tl.startswith("rate limit"):
        return True
    if tl.startswith("timeout "):
        return True
    if tl.startswith("server error "):
        return True
    # "Error {'error': ...}" o "Error [..." — error de API en formato dict/lista
    if (t.startswith("Error ") or t.startswith("error ")) and len(t) > 6 and t[6] in "{[":
        return True
    return False


class Ensamblador:
    def __init__(self, modelos=None, timeout_seg=60):
        self.modelos = modelos or []
        self.timeout = aiohttp.ClientTimeout(total=timeout_seg)
        self.modelos_filtrados = []

    async def query_modelo(self, sesion: aiohttp.ClientSession, modelo: dict, prompt: str) -> dict:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://chorus.innotep.upm",
            "X-Title": "CHORUS Clinical Reasoning"
        }
        payload = {
            "model": modelo["name"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4,
            "max_tokens": 2048
        }
        texto_respuesta = ""
        provider_version = None
        api_error = None
        t0 = time.perf_counter()
        # Reintentos con backoff para 429/5xx y errores transitorios de red.
        # La latencia acumulada se mide desde el primer intento; el
        # `api_error` refleja el resultado del último intento.
        for attempt in range(RETRY_MAX_ATTEMPTS):
            try:
                async with sesion.post(API_URL, headers=headers, json=payload) as rep:
                    if rep.status == 429 or rep.status >= 500:
                        if rep.status == 429:
                            texto_respuesta = f"rate limit exceeded for {modelo['name']}"
                            api_error = "http_429"
                        else:
                            texto_respuesta = f"server error {rep.status} for {modelo['name']}"
                            api_error = f"http_{rep.status}"
                        if attempt < RETRY_MAX_ATTEMPTS - 1:
                            wait = RETRY_BACKOFF_SECONDS[attempt]
                            print(f"[ensamblador] {modelo['name']} {rep.status}, "
                                  f"reintento en {wait}s ({attempt+1}/{RETRY_MAX_ATTEMPTS})")
                            await asyncio.sleep(wait)
                            continue
                        break
                    data = await rep.json()
                    if isinstance(data, dict) and "choices" in data and data["choices"]:
                        texto_respuesta = data["choices"][0]["message"]["content"]
                        provider_version = data.get("model")
                        api_error = None
                    elif isinstance(data, dict) and "error" in data:
                        msg = data["error"].get("message", str(data["error"]))
                        texto_respuesta = f"error {msg}"
                        api_error = str(data["error"].get("code", msg))[:120]
                    else:
                        texto_respuesta = f"error respuesta inesperada: {str(data)[:100]}"
                        api_error = "unexpected_response_shape"
                    break
            except asyncio.TimeoutError:
                texto_respuesta = f"timeout consultando {modelo['name']}"
                api_error = "timeout"
                if attempt < RETRY_MAX_ATTEMPTS - 1:
                    wait = RETRY_BACKOFF_SECONDS[attempt]
                    print(f"[ensamblador] {modelo['name']} timeout, "
                          f"reintento en {wait}s ({attempt+1}/{RETRY_MAX_ATTEMPTS})")
                    await asyncio.sleep(wait)
                    continue
                break
            except aiohttp.ClientError as e:
                texto_respuesta = f"request failed: {e}"
                api_error = f"exception: {type(e).__name__}"
                if attempt < RETRY_MAX_ATTEMPTS - 1:
                    wait = RETRY_BACKOFF_SECONDS[attempt]
                    print(f"[ensamblador] {modelo['name']} {api_error}, "
                          f"reintento en {wait}s ({attempt+1}/{RETRY_MAX_ATTEMPTS})")
                    await asyncio.sleep(wait)
                    continue
                break
            except Exception as e:
                texto_respuesta = f"request failed: {e}"
                api_error = f"exception: {type(e).__name__}"
                break

        latency_ms = int((time.perf_counter() - t0) * 1000)

        return {
            "model_name": modelo["name"],
            "response": texto_respuesta,
            "timestamp": datetime.now().isoformat(),
            "provider_version": provider_version,
            "latency_ms": latency_ms,
            "api_error": api_error,
        }

    async def run(self, prompt: str) -> list:
        async with aiohttp.ClientSession(timeout=self.timeout) as sesion:
            tasks = [self.query_modelo(sesion, m, prompt) for m in self.modelos]
            resultados = await asyncio.gather(*tasks, return_exceptions=False)

        validos = []
        self.modelos_filtrados = []
        self.resultados_crudos = resultados
        for r in resultados:
            if _es_error(r["response"]):
                motivo = r["response"][:120].strip()
                print(f"[ensamblador] Filtrado: {r['model_name']} — {motivo}")
                self.modelos_filtrados.append({
                    "model_name": r["model_name"],
                    "reason": motivo,
                    "provider_version": r.get("provider_version"),
                    "latency_ms": r.get("latency_ms"),
                    "api_error": r.get("api_error"),
                })
            else:
                validos.append(r)

        print(f"[ensamblador] Respuestas válidas: {len(validos)}/{len(self.modelos)}")
        return validos

    def guardar_resultados(self, respuestas: list, output_dir: str = OUTPUT_PATH):
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        filename = f"ensamble_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = Path(output_dir) / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(respuestas, f, indent=2, ensure_ascii=False)
        return str(filepath)
