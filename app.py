from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
import asyncio
import hashlib
import json
import subprocess
import sys
import os
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__, static_folder="static")
CORS(app, supports_credentials=True)

OUTPUT_PATH = os.environ.get("CHORUS_OUTPUT_PATH", "resultados")
BROWSER_COOKIE = "chorus_browser_token"
REFERENCE_CASES_PATH = os.environ.get(
    "CHORUS_REFERENCE_CASES", "casos_referencia.json"
)

with open("modelos.json", "r", encoding="UTF-8") as f:
    MODELS_DATA = json.load(f)


def _load_reference_cases():
    """Carga los casos de referencia desde disco. Devuelve dict vacío
    si el fichero no existe o está mal formado (la web sigue funcionando
    sin ellos; el endpoint devolverá una lista vacía)."""
    try:
        with open(REFERENCE_CASES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        casos = data.get("casos") or []
        return {c["id"]: c for c in casos if isinstance(c, dict) and c.get("id")}
    except FileNotFoundError:
        print(f"[app] No se encontró {REFERENCE_CASES_PATH}: casos de referencia deshabilitados")
        return {}
    except Exception as e:
        print(f"[app] Error leyendo casos de referencia: {e}")
        return {}


REFERENCE_CASES = _load_reference_cases()

try:
    from analizador import dataAnalisis, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS
    from Ensambladores.ensamblador_LLM import Ensamblador
    from schemas.meta_v1 import (
        SCHEMA_VERSION,
        construir_meta_base,
        validar_meta,
        MetaValidationError,
    )
    MODULES_OK = True
    print("Modulos importados correctamente")
except ImportError as e:
    print(f"Error importando modulos: {e}")
    traceback.print_exc()
    MODULES_OK = False


def _chorus_version():
    """Hash del commit actual; 'unknown' si no hay git o falla la llamada."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except Exception:
        pass
    return "unknown"


CHORUS_VERSION = _chorus_version()


@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "modules": MODULES_OK,
                    "schema_version": SCHEMA_VERSION if MODULES_OK else None,
                    "chorus_version": CHORUS_VERSION,
                    "time": datetime.now().isoformat()})

@app.route("/api/models", methods=["GET"])
def get_models():
    return jsonify({"success": True,
                    "free_models": MODELS_DATA["LLM"]["FREE_MODELS"],
                    "pay_models":  MODELS_DATA["LLM"]["PAY_MODELS"]})


@app.route("/api/reference_cases", methods=["GET"])
def get_reference_cases():
    """Devuelve la lista pública de casos de referencia.

    El texto completo (`texto_completo`) NO se expone al navegador:
    cuando el visitante selecciona un caso, envía solo el `id` en el
    POST de /api/run-ensemble y el backend carga el texto desde disco.
    Así los casos quedan auditables y el frontend no los expone en claro.
    """
    casos = [
        {
            "id": c.get("id"),
            "titulo": c.get("titulo"),
            "descripcion_corta": c.get("descripcion_corta"),
            "ensemble_recomendado": c.get("ensemble_recomendado", []),
        }
        for c in REFERENCE_CASES.values()
    ]
    return jsonify({"success": True, "casos": casos})


def _browser_token():
    """Devuelve el token de la cookie (si existe) o None."""
    return request.cookies.get(BROWSER_COOKIE)


@app.route("/api/history", methods=["GET"])
def get_history():
    try:
        token = _browser_token()
        path = Path(OUTPUT_PATH)
        if not path.exists():
            return jsonify({"success": True, "analyses": []})
        files = sorted(path.glob("ensamble_*.meta.json"), reverse=True)[:50]
        analyses = []
        for f in files:
            try:
                with open(f, "r", encoding="utf-8") as mf:
                    meta = json.load(mf)
                # Filtrar por browser_token cuando existe en ambos lados.
                # Si el visitante no tiene cookie, no se le muestra nada
                # para no mezclar análisis de otros navegadores.
                if token is None:
                    continue
                if meta.get("browser_token") != token:
                    continue
                # Resumen ligero para la lista.
                analyses.append({
                    "filename": f.name.replace(".meta.json", ".json"),
                    "case_uuid": meta.get("case_uuid"),
                    "timestamp": meta.get("timestamp_utc") or meta.get("timestamp_local"),
                    "prompt_preview": meta.get("prompt_preview"),
                    "models_count": (meta.get("ensemble") or {}).get("n_modelos"),
                    "cdi": meta.get("cdi"),
                    "consenso_global": meta.get("consenso_global"),
                })
                if len(analyses) >= 20:
                    break
            except Exception:
                continue
        return jsonify({"success": True, "analyses": analyses})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/history/<filename>", methods=["GET"])
def get_history_item(filename):
    try:
        safe = Path(filename).name
        if not safe.startswith("ensamble_") or not safe.endswith(".json"):
            return jsonify({"success": False, "error": "Nombre de archivo inválido"})
        meta_path = Path(OUTPUT_PATH) / safe.replace(".json", ".meta.json")
        if not meta_path.exists():
            return jsonify({"success": False, "error": "Análisis no encontrado"})
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        token = _browser_token()
        if data.get("browser_token") and data["browser_token"] != token:
            return jsonify({"success": False, "error": "No autorizado"}), 403
        return jsonify({"success": True, "data": data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


def _serializar(obj):
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    raise TypeError(f"No serializable: {type(obj)}")


def format_consenso_data(reporte, resultados_raw):
    if not reporte or "error" in reporte:
        return None

    consensos_ind = reporte.get("consensos_individuales", [])
    nombre_mas_consensuado = None
    if consensos_ind:
        validos = [c for c in consensos_ind if c["consenso_individual"] > 0]
        if validos:
            nombre_mas_consensuado = validos[0]["modelo"]

    top3 = [c["modelo"] for c in consensos_ind[:3]] if consensos_ind else []

    formateado = {
        "consenso_global": reporte.get("consenso_global", 0.0),
        "consensos_individuales": consensos_ind,
        "nombres_filtrados": reporte.get("nombres_filtrados", []),
        "top3_modelos": top3,
        "modelo_mas_consensuado": nombre_mas_consensuado,
        "cdi": reporte.get("cdi"),
        "solo": reporte.get("solo"),
        "divergencia_capas": reporte.get("divergencia_capas"),
        "embedding_disponible": reporte.get("embedding_disponible", False),
    }

    if reporte.get("respuesta_mas_consensuada"):
        formateado["respuesta_mas_consensuada"] = reporte["respuesta_mas_consensuada"]
    if reporte.get("respuesta_fusionada"):
        formateado["respuesta_fusionada"] = reporte["respuesta_fusionada"]
        formateado["modelos_base"] = reporte.get("modelos_base", [])

    return formateado


def _matriz_a_lista(m):
    if m is None:
        return None
    try:
        return m.tolist()
    except AttributeError:
        return list(m)


def _construir_meta(*, filename_base, prompt, model_type, modelos_solicitados,
                   resultados_crudos, reporte, browser_token,
                   case_reference_id=None, session_code=None):
    """Construye el payload meta.json v1.0 completo a partir de:
      - el prompt original,
      - resultados crudos del ensamblador (todos, válidos y filtrados),
      - el reporte de dataAnalisis.
    El prompt completo NO se persiste: solo sha256, preview y longitud.
    """
    meta = construir_meta_base()

    ts_utc = datetime.now(timezone.utc)
    ts_local = datetime.now().astimezone()

    meta["case_uuid"] = str(uuid.uuid4())
    meta["case_reference_id"] = case_reference_id
    meta["session_code"] = session_code
    meta["browser_token"] = browser_token
    meta["timestamp_utc"] = ts_utc.isoformat().replace("+00:00", "Z")
    meta["timestamp_local"] = ts_local.isoformat()
    meta["prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    meta["prompt_length_chars"] = len(prompt)
    meta["prompt_preview"] = prompt[:120]

    # Mapa model_name → flag embedding_truncated (válidos solamente).
    truncated_map = {}
    flags = reporte.get("embedding_truncated_flags") or []
    nombres_validos = reporte.get("nombres_filtrados") or []
    for name, flag in zip(nombres_validos, flags):
        truncated_map[name] = bool(flag)

    modelos_lista = []
    for r in resultados_crudos:
        name = r.get("model_name", "")
        modelos_lista.append({
            "name": name,
            "provider_version": r.get("provider_version"),
            "latency_ms": r.get("latency_ms"),
            "response_length_chars": len(r.get("response") or ""),
            "embedding_truncated": truncated_map.get(name, False),
            "api_error": r.get("api_error"),
        })

    meta["ensemble"] = {
        "n_modelos": len(modelos_lista),
        "model_type": model_type,
        "modelos": modelos_lista,
    }

    meta["fusion"] = {
        "modelo": reporte.get("fusion_modelo"),
        "latency_ms": reporte.get("fusion_latency_ms"),
        "max_tokens": reporte.get("fusion_max_tokens"),
        "temperature": reporte.get("fusion_temperature"),
    }

    m_embed = reporte.get("_matriz_embed")
    m_tfidf = reporte.get("_matriz_tfidf")
    meta["matrices"] = {
        "tfidf": _matriz_a_lista(m_tfidf),
        "embed": _matriz_a_lista(m_embed),
        "principal": "embed" if m_embed is not None else "tfidf",
    }

    meta["embeddings"] = {
        "modelo": EMBEDDING_MODEL,
        "dimensiones": EMBEDDING_DIMENSIONS.get(EMBEDDING_MODEL, 0),
        "fallback_aplicado": not reporte.get("embedding_disponible", False),
    }

    meta["cdi"] = reporte.get("cdi")
    meta["solo"] = reporte.get("solo")
    meta["divergencia_capas"] = reporte.get("divergencia_capas")
    meta["consenso_global"] = reporte.get("consenso_global")
    meta["consensos_individuales"] = reporte.get("consensos_individuales") or []

    fusion_text = reporte.get("respuesta_fusionada")
    meta["respuesta_fusionada_sha256"] = (
        hashlib.sha256(fusion_text.encode("utf-8")).hexdigest()
        if fusion_text else None
    )

    meta["chorus_version"] = CHORUS_VERSION

    # Normalizar tipos numpy antes de validar/guardar.
    meta = json.loads(json.dumps(meta, default=_serializar))
    validar_meta(meta)
    return meta


def _guardar_meta(filename_base, meta_payload):
    """Escritura atómica de `<base>.meta.json`.

    Se escribe primero en `<base>.meta.json.tmp`, se hace `fsync` para
    forzar el volcado a disco y se usa `os.replace` para mover el
    fichero a su ruta final. `os.replace` es atómico en POSIX y en
    Windows (cuando origen y destino están en el mismo volumen), así
    que el meta.json final nunca existe en estado parcial: o no existe,
    o está completo y validado.

    Si algo falla durante la escritura se elimina el `.tmp` para no
    dejar basura y se re-lanza la excepción (el caller decide qué hacer).
    """
    out_dir = Path(OUTPUT_PATH)
    out_dir.mkdir(exist_ok=True, parents=True)
    final_path = out_dir / f"{filename_base}.meta.json"
    tmp_path = out_dir / f"{filename_base}.meta.json.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, ensure_ascii=False, indent=2, default=_serializar)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, final_path)
    except Exception as e:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        print(f"[app] Error guardando meta: {e}")
        raise


@app.route("/api/run-ensemble", methods=["POST"])
def run_ensamble():
    if not MODULES_OK:
        return jsonify({"success": False, "error": "Modulos no disponibles"})
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "Body JSON requerido"})

        prompt = data.get("prompt", "").strip()
        model_names = data.get("models", [])
        model_type  = data.get("modelType", "free")
        session_code = data.get("session_code") or None
        case_reference_id = data.get("case_reference_id") or None

        # Si llega un case_reference_id, el texto se carga desde el
        # catálogo del backend. El frontend no necesita haber recibido
        # `texto_completo`, lo que permite que /api/reference_cases
        # nunca lo exponga al navegador.
        if case_reference_id:
            caso = REFERENCE_CASES.get(case_reference_id)
            if not caso:
                return jsonify({"success": False,
                                "error": f"Caso de referencia no encontrado: {case_reference_id}"})
            prompt = (caso.get("texto_completo") or "").strip()
            # Fallback implícito: si el request no trajo modelos pero el
            # caso tiene ensemble_recomendado, lo usamos tal cual.
            if not model_names and caso.get("ensemble_recomendado"):
                model_names = list(caso["ensemble_recomendado"])

        if not prompt:
            return jsonify({"success": False, "error": "El prompt no puede estar vacío"})
        if len(prompt) > 8000:
            return jsonify({"success": False, "error": "El caso clínico no puede superar 8000 caracteres"})
        if not model_names:
            return jsonify({"success": False, "error": "Selecciona al menos un modelo"})
        if len(model_names) > 20:
            return jsonify({"success": False, "error": "Máximo 20 modelos por consulta"})

        source = MODELS_DATA["LLM"]["FREE_MODELS" if model_type == "free" else "PAY_MODELS"]
        modelos = [m for m in source if m["name"] in model_names]
        if not modelos:
            return jsonify({"success": False, "error": "Modelos no encontrados en el catálogo"})

        # Cookie anónima del navegador. Se emite si no existe y se vuelve a
        # emitir (set-cookie) siempre, para refrescar maxAge.
        token = _browser_token() or str(uuid.uuid4())

        ts_now = datetime.now()
        filename_base = f"ensamble_{ts_now.strftime('%Y%m%d_%H%M%S')}"

        ensamble = Ensamblador(modelos=modelos)

        async def ejecutar():
            resultados = await ensamble.run(prompt)
            modelos_filtrados = ensamble.modelos_filtrados
            if hasattr(ensamble, "guardar_resultados"):
                ensamble.guardar_resultados(resultados)

            resultados_fmt = [
                {"model_name": r.get("model_name", f"Modelo_{i}"),
                 "response":   r.get("response", ""),
                 "timestamp":  r.get("timestamp", ts_now.isoformat()),
                 "index": i}
                for i, r in enumerate(resultados)
            ]

            reporte = await dataAnalisis(resultados)
            reporte_limpio = {
                k: v for k, v in reporte.items()
                if not k.startswith("_") and k != "embedding_truncated_flags"
            }
            consenso_data = format_consenso_data(reporte, resultados)

            return {
                "success": True,
                "results": resultados_fmt,
                "report": reporte_limpio,
                "consenso_data": consenso_data,
                "prompt_preview": prompt[:120] + ("…" if len(prompt) > 120 else ""),
                "models_count": len(resultados),
                "models_requested": len(modelos),
                "models_used": [r.get("model_name") for r in resultados],
                "models_failed": modelos_filtrados,
                "filename": filename_base + ".json",
                "timestamp": ts_now.isoformat()
            }, reporte

        resultado_ui, reporte_completo = asyncio.run(ejecutar())
        serializado = json.loads(json.dumps(resultado_ui, default=_serializar))

        # Construir y guardar meta v1.0.
        try:
            meta = _construir_meta(
                filename_base=filename_base,
                prompt=prompt,
                model_type=model_type,
                modelos_solicitados=modelos,
                resultados_crudos=getattr(ensamble, "resultados_crudos", []),
                reporte=reporte_completo,
                browser_token=token,
                case_reference_id=case_reference_id,
                session_code=session_code,
            )
            serializado["case_uuid"] = meta["case_uuid"]
            try:
                _guardar_meta(filename_base, meta)
            except OSError as ose:
                # El análisis ya está entregado al visitante; la escritura
                # atómica falló por I/O (permisos, disco lleno, etc.). No
                # degrada la respuesta, solo se pierde la traza persistente.
                print(f"[app] No se pudo persistir meta.json: {ose}")
        except MetaValidationError as mve:
            print(f"[app] Meta schema v1.0 inválido: {mve}")
            traceback.print_exc()

        resp = make_response(jsonify(serializado))
        resp.set_cookie(
            BROWSER_COOKIE, token,
            max_age=60 * 60 * 24 * 365,   # 1 año
            httponly=True, samesite="Lax", path="/",
        )
        return resp

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    print("\n\t\t CHORUS — Clinical Heterogeneous Orchestration for Reasoning Under Supervision")
    print(f"\t\t URL: http://localhost:8282\n")
    app.run(host="0.0.0.0", port=8282, debug=False, threaded=True, use_reloader=False)
