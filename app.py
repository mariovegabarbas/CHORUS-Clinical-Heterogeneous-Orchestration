from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import asyncio
import json
import sys
import os
import traceback
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__, static_folder="static")
CORS(app)

OUTPUT_PATH = os.environ.get("CHORUS_OUTPUT_PATH", "resultados")

with open("modelos.json", "r", encoding="UTF-8") as f:
    MODELS_DATA = json.load(f)

try:
    from analizador import dataAnalisis
    from Ensambladores.ensamblador_LLM import Ensamblador
    MODULES_OK = True
    print("Modulos importados correctamente")
except ImportError as e:
    print(f"Error importando modulos: {e}")
    traceback.print_exc()
    MODULES_OK = False


@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "modules": MODULES_OK,
                    "time": datetime.now().isoformat()})

@app.route("/api/models", methods=["GET"])
def get_models():
    return jsonify({"success": True,
                    "free_models": MODELS_DATA["LLM"]["FREE_MODELS"],
                    "pay_models":  MODELS_DATA["LLM"]["PAY_MODELS"]})


@app.route("/api/history", methods=["GET"])
def get_history():
    try:
        path = Path(OUTPUT_PATH)
        if not path.exists():
            return jsonify({"success": True, "analyses": []})
        files = sorted(path.glob("ensamble_*.json"), reverse=True)[:20]
        analyses = []
        for f in files:
            try:
                meta_path = f.with_suffix(".meta.json")
                if meta_path.exists():
                    with open(meta_path, "r", encoding="utf-8") as mf:
                        meta = json.load(mf)
                    analyses.append(meta)
                else:
                    ts_str = f.stem.replace("ensamble_", "")
                    analyses.append({
                        "filename": f.name,
                        "timestamp": ts_str,
                        "models_count": None,
                        "cdi": None,
                        "prompt_preview": None
                    })
            except Exception:
                pass
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


def _guardar_meta(filename_base, payload):
    try:
        Path(OUTPUT_PATH).mkdir(exist_ok=True, parents=True)
        meta_path = Path(OUTPUT_PATH) / f"{filename_base}.meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=_serializar)
    except Exception as e:
        print(f"[app] Error guardando meta: {e}")


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

        ts_now = datetime.now()
        filename_base = f"ensamble_{ts_now.strftime('%Y%m%d_%H%M%S')}"

        async def ejecutar():
            ensamble = Ensamblador(modelos=modelos)
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

            reporte = dataAnalisis(resultados)
            reporte_limpio = {k: v for k, v in reporte.items() if not k.startswith("_")}
            consenso_data = format_consenso_data(reporte, resultados)

            return {
                "success": True,
                "results": resultados_fmt,
                "report": reporte_limpio,
                "consenso_data": consenso_data,
                "prompt": prompt,
                "prompt_preview": prompt[:120] + ("…" if len(prompt) > 120 else ""),
                "models_count": len(resultados),
                "models_requested": len(modelos),
                "models_used": [r.get("model_name") for r in resultados],
                "models_failed": modelos_filtrados,
                "filename": filename_base + ".json",
                "timestamp": ts_now.isoformat()
            }

        resultado = asyncio.run(ejecutar())
        serializado = json.loads(json.dumps(resultado, default=_serializar))

        # Guardar metadatos para historial
        meta = {
            "filename": filename_base + ".json",
            "timestamp": ts_now.isoformat(),
            "prompt_preview": resultado["prompt_preview"],
            "models_count": resultado["models_count"],
            "models_used": resultado["models_used"],
            "cdi": resultado.get("consenso_data", {}).get("cdi"),
            "consenso_global": resultado.get("consenso_data", {}).get("consenso_global"),
            "full": serializado
        }
        _guardar_meta(filename_base, meta)

        return jsonify(serializado)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    print("\n\t\t CHORUS — Clinical Heterogeneous Orchestration for Reasoning Under Supervision")
    print(f"\t\t URL: http://localhost:8282\n")
    app.run(host="0.0.0.0", port=8282, debug=False, threaded=True, use_reloader=False)
