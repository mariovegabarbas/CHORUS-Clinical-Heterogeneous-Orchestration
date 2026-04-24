import aiohttp
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

API_URL    = "https://openrouter.ai/api/v1/chat/completions"
API_KEY    = os.environ.get("OPENROUTER_API_KEY", "")
OUTPUT_PATH = os.environ.get("CRML_OUTPUT_PATH", "resultados")

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

    async def query_modelo(self, sesion: aiohttp.ClientSession, modelo: dict, prompt: str) -> dict:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://crml.local",
            "X-Title": "CRML Clinical Reasoning"
        }
        payload = {
            "model": modelo["name"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4,
            "max_tokens": 2048
        }
        texto_respuesta = ""
        try:
            async with sesion.post(API_URL, headers=headers, json=payload) as rep:
                if rep.status == 429:
                    texto_respuesta = f"rate limit exceeded for {modelo['name']}"
                elif rep.status >= 500:
                    texto_respuesta = f"server error {rep.status} for {modelo['name']}"
                else:
                    data = await rep.json()
                    if "choices" in data and data["choices"]:
                        texto_respuesta = data["choices"][0]["message"]["content"]
                    elif "error" in data:
                        msg = data["error"].get("message", str(data["error"]))
                        texto_respuesta = f"error {msg}"
                    else:
                        texto_respuesta = f"error respuesta inesperada: {str(data)[:100]}"
        except asyncio.TimeoutError:
            texto_respuesta = f"timeout consultando {modelo['name']}"
        except Exception as e:
            texto_respuesta = f"request failed: {e}"

        return {
            "model_name": modelo["name"],
            "response": texto_respuesta,
            "timestamp": datetime.now().isoformat()
        }

    async def run(self, prompt: str) -> list:
        async with aiohttp.ClientSession(timeout=self.timeout) as sesion:
            tasks = [self.query_modelo(sesion, m, prompt) for m in self.modelos]
            resultados = await asyncio.gather(*tasks, return_exceptions=False)

        validos = []
        for r in resultados:
            if _es_error(r["response"]):
                print(f"[ensamblador] Filtrado: {r['model_name']} — {r['response'][:80]}")
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
