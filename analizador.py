"""
analizador.py — CHORUS
======================
  1. Clinical Dissent Index (CDI) formalizado con umbrales clínicos
  2. Voz Solista (SOLO): el modelo más disidente expuesto explícitamente
  3. Doble capa de similitud: TF-IDF (léxica) + Embeddings OpenAI (semántica)
  4. Taxonomía de complejidad diagnóstica basada en CDI
  5. Reporte enriquecido para el frontend
"""

import asyncio
import os
import time
import numpy as np
import json
import aiohttp
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

# ── Configuración ────────────────────────────────────────────────────────────
API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODELO_FUSION = os.environ.get("CHORUS_FUSION_MODEL", "gpt-4o-mini")
FUSION_MAX_TOKENS = 1500
FUSION_TEMPERATURE = 0.2

# Reintentos con backoff exponencial para 429/5xx y errores transitorios de red.
# Tres intentos con espera 1s, 2s, 4s absorben rate limiting ocasional sin
# degradar perceptiblemente la experiencia del visitante.
RETRY_MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = (1.0, 2.0, 4.0)

CDI_UMBRALES = {
    "baja":    (0.0,  0.25),
    "moderada":(0.25, 0.60),
    "alta":    (0.60, 0.85),
    "maxima":  (0.85, 1.01),
}
CDI_ETIQUETAS = {
    "baja":    "Convergencia diagnostica — alta certeza del ensemble",
    "moderada":"Ambiguedad moderada — se recomienda deliberacion clinica",
    "alta":    "Alta diversidad diagnostica — caso complejo",
    "maxima":  "Divergencia critica — incertidumbre maxima del ensemble",
}
CDI_COLORES = {
    "baja":    "#28a745",
    "moderada":"#ffc107",
    "alta":    "#fd7e14",
    "maxima":  "#dc3545",
}


def _nivel_cdi(cdi_val):
    for nivel, (lo, hi) in CDI_UMBRALES.items():
        if lo <= cdi_val < hi:
            return nivel
    return "maxima"


# ── Capa 1: TF-IDF ───────────────────────────────────────────────────────────

def _matriz_tfidf(textos):
    vectorizer = TfidfVectorizer(min_df=1, max_features=1000, strip_accents="unicode")
    try:
        mat = vectorizer.fit_transform(textos)
        if mat.shape[1] == 0:
            return np.identity(len(textos))
        return cosine_similarity(mat)
    except Exception as e:
        print(f"[analizador] Error TF-IDF: {e}")
        return np.identity(len(textos))


# ── Capa 2: Embeddings OpenAI ────────────────────────────────────────────────

# Configurable por env var. Default 16000 (subido desde 8000 tras el smoke
# test grande del 25 abril 2026: gemini-2.5-flash truncaba sistemáticamente).
# text-embedding-3-large tolera hasta ~32000 chars sin problema.
EMBEDDING_MAX_CHARS = int(os.environ.get("CHORUS_EMBEDDING_MAX_CHARS", "16000"))

# Modelo de embeddings configurable. Default `text-embedding-3-large`:
# mejora discriminación semántica en textos clínicos largos respecto a
# `text-embedding-3-small` a coste marginal.
EMBEDDING_MODEL = os.environ.get("CHORUS_EMBEDDING_MODEL", "text-embedding-3-large")

# Dimensiones nativas por modelo. Se persiste en el meta.json para
# trazabilidad del análisis. Modelos no listados → 0 (desconocido).
EMBEDDING_DIMENSIONS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}


def _obtener_embeddings(textos):
    """Devuelve (vectores_or_None, truncated_flags).

    truncated_flags es una lista de bools paralela a `textos` que indica
    si el texto excedió EMBEDDING_MAX_CHARS y tuvo que ser truncado
    antes de enviarlo a la API de embeddings.
    """
    truncated_flags = [len(t) > EMBEDDING_MAX_CHARS for t in textos]
    if any(truncated_flags):
        n_trunc = sum(truncated_flags)
        print(f"[analizador] WARNING: {n_trunc}/{len(textos)} textos truncados "
              f"a {EMBEDDING_MAX_CHARS} caracteres para embeddings.")
    if not API_KEY or API_KEY.startswith("//"):
        return None, truncated_flags
    try:
        url = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        payload = {"model": EMBEDDING_MODEL,
                   "input": [t[:EMBEDDING_MAX_CHARS] for t in textos]}
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            return np.array([d["embedding"] for d in data["data"]]), truncated_flags
        print(f"[analizador] Embeddings API {resp.status_code}")
        return None, truncated_flags
    except Exception as e:
        print(f"[analizador] Error embeddings: {e}")
        return None, truncated_flags


def _matriz_embeddings(textos):
    """Devuelve (matriz_similitud_or_None, truncated_flags)."""
    vecs, flags = _obtener_embeddings(textos)
    if vecs is None:
        return None, flags
    return cosine_similarity(vecs), flags


# ── CDI ───────────────────────────────────────────────────────────────────────

def calcular_cdi(matriz_sim):
    """
    Devuelve tres métricas emparejadas de disenso del ensemble, todas
    acotadas en [0, 1] y comparables entre ensembles de distinto tamaño.

    - cdi_geometric : 1 - |det(M)|**(1/n). Métrica principal. La media
      geométrica del determinante normaliza el colapso exponencial del
      det(M) con n y conserva una interpretación clara (producto de
      valores singulares elevado a 1/n).
    - cdi_mean_dissent : 1 - mean(off_diagonal(M)). Disenso par-a-par.
    - cdi_entropy : entropía de Shannon normalizada sobre las
      similitudes off-diagonal (disperso ↔ concentrado).
    - cdi_det_raw : 1 - |det(M)|, solo para retrocompatibilidad con
      meta.json generados antes del schema v1.0.

    El alias `cdi` apunta a `cdi_geometric` para no romper consumidores
    existentes (frontend, reportes previos).
    """
    matriz_sim = np.asarray(matriz_sim, dtype=float)
    n = len(matriz_sim)

    if n < 2:
        return {
            "cdi": 0.0,
            "cdi_geometric": 0.0,
            "cdi_mean_dissent": 0.0,
            "cdi_entropy": 0.0,
            "cdi_det_raw": 0.0,
            "n_modelos": n,
            "nivel": "baja",
            "etiqueta": CDI_ETIQUETAS["baja"],
            "color": CDI_COLORES["baja"],
            "determinante": 1.0,
            "entropia": 0.0,
        }

    if not np.all(np.isfinite(matriz_sim)):
        return {
            "cdi": None,
            "cdi_geometric": None,
            "cdi_mean_dissent": None,
            "cdi_entropy": None,
            "cdi_det_raw": None,
            "n_modelos": n,
            "nivel": "indeterminado",
            "etiqueta": "Matriz de similitud inestable. Alguna respuesta del ensemble no es procesable.",
            "color": "#888888",
            "determinante": None,
            "entropia": None,
            "error": "matrix_contains_non_finite_values",
        }

    try:
        det = float(np.linalg.det(matriz_sim))
        det_abs = abs(det)
        cdi_det_raw = float(np.clip(1.0 - det_abs, 0.0, 1.0))
        cdi_geometric = float(np.clip(det_abs ** (1.0 / n), 0.0, 1.0))
    except Exception:
        det = 0.5
        cdi_det_raw = 0.5
        cdi_geometric = 0.5

    off = [float(matriz_sim[i, j]) for i in range(n) for j in range(n) if i != j]
    if off:
        off_arr = np.array(off)
        mean_off = float(np.mean(off_arr))
        cdi_mean_dissent = float(np.clip(1.0 - mean_off, 0.0, 1.0))

        v = np.clip(off_arr, 1e-10, 1.0)
        v = v / v.sum()
        cdi_entropy = float(-np.sum(v * np.log(v + 1e-10)))
    else:
        cdi_mean_dissent = 0.0
        cdi_entropy = 0.0

    nivel = _nivel_cdi(cdi_geometric)
    return {
        "cdi": round(cdi_geometric, 4),
        "cdi_geometric": round(cdi_geometric, 4),
        "cdi_mean_dissent": round(cdi_mean_dissent, 4),
        "cdi_entropy": round(cdi_entropy, 4),
        "cdi_det_raw": round(cdi_det_raw, 4),
        "n_modelos": n,
        "nivel": nivel,
        "etiqueta": CDI_ETIQUETAS[nivel],
        "color": CDI_COLORES[nivel],
        "determinante": round(det, 6),
        "entropia": round(cdi_entropy, 4),
    }


# ── Divergencia entre capas ───────────────────────────────────────────────────

def calcular_divergencia_capas(m_tfidf, m_embed):
    if m_embed is None:
        return None
    n = len(m_tfidf)
    diff = m_tfidf - m_embed
    criticos = []
    for i in range(n):
        for j in range(i + 1, n):
            delta = float(diff[i, j])
            if delta > 0.15:
                criticos.append({"i": i, "j": j,
                                  "sim_tfidf": round(float(m_tfidf[i,j]), 3),
                                  "sim_embed": round(float(m_embed[i,j]), 3),
                                  "delta": round(delta, 3)})
    criticos.sort(key=lambda x: x["delta"], reverse=True)
    frob = float(np.linalg.norm(diff, "fro"))
    return {
        "distancia_frobenius": round(frob, 4),
        "pares_criticos": criticos[:5],
        "n_pares_criticos": len(criticos),
        "descripcion": (
            "Alta divergencia entre similitud lexica y semantica — "
            "algunos modelos usan vocabulario similar para hipotesis distintas."
            if criticos else
            "Coherencia entre capas lexica y semantica."
        )
    }


# ── Consensos individuales ────────────────────────────────────────────────────

def calcular_consensos_individuales(matriz_sim, nombres):
    n = len(matriz_sim)
    resultado = []
    for i in range(n):
        otros = [float(matriz_sim[i, j]) for j in range(n) if j != i]
        consenso = round(float(np.mean(otros)), 4) if otros else 0.0
        resultado.append({"indice": i, "modelo": nombres[i],
                           "consenso_individual": consenso,
                           "respuesta_idx": i})
    return sorted(resultado, key=lambda x: x["consenso_individual"], reverse=True)


# ── Voz Solista (SOLO) ────────────────────────────────────────────────────────

def identificar_solo(consensos_ind, resultados_raw):
    """
    Expone el modelo con menor consenso individual como Voz Solista (SOLO).

    El filtro acepta `consenso_individual >= 0` (no `> 0`): un modelo con
    consenso exactamente cero es ortogonal al resto del ensemble y
    constituye el SOLO canónico — estructuralmente es el caso más
    interesante y justamente el que el sistema está diseñado para no
    silenciar. Solo se devuelve None en el caso degenerado en el que
    TODOS los modelos son mutuamente ortogonales (consenso == 0 en todos):
    ahí no hay SOLO distinguible porque cada uno es tan disidente como los
    demás.
    """
    validos = [c for c in consensos_ind if c["consenso_individual"] >= 0]
    if not validos:
        return None
    if all(c["consenso_individual"] == 0 for c in validos):
        return None
    minoritario = validos[-1]
    texto = next((r.get("response","") for r in resultados_raw
                  if r.get("model_name") == minoritario["modelo"]), "")
    return {
        "modelo": minoritario["modelo"],
        "consenso_individual": minoritario["consenso_individual"],
        "respuesta": texto,
        "nota": (
            "Este modelo discrepa significativamente del ensemble. "
            "Su perspectiva puede representar una hipotesis diagnostica alternativa "
            "que merece evaluacion clinica independiente."
        )
    }


def imprimir_matriz_consenso(matriz, nombres_filtrados):
    if not nombres_filtrados:
        return
    cortos = [n.split("/")[-1][:15] if "/" in n else n[:15] for n in nombres_filtrados]
    print("\n\t\tMATRIZ DE SIMILITUD COSENO")
    header = " " * 17
    for c in cortos:
        header += f"{c:>17}"
    print(header)
    for i, fila in enumerate(matriz):
        row = f"{cortos[i]:<17}"
        for j, v in enumerate(fila):
            row += f"{'1.000' if i==j else f'{v:.3f}':>17}"
        print(row)


# ── Fusión ────────────────────────────────────────────────────────────────────

async def _llamar_chatgpt(mensajes):
    """Llama al modelo de fusión con aiohttp y backoff exponencial.

    Async nativo: ya no bloquea el event loop del pipeline. Devuelve
    (texto, latency_ms). `latency_ms` se mide siempre, incluso en caso
    de error. `texto` es None si la llamada falló tras agotar reintentos.

    Reintentos: 3 intentos con espera 1s/2s/4s para 429, 5xx y errores
    transitorios de red. Los 4xx no-429 no se reintentan (malformación
    del request, auth, etc.). Sin API_KEY no se llama a la API.
    """
    t0 = time.perf_counter()
    if not API_KEY or API_KEY.startswith("//"):
        return None, int((time.perf_counter() - t0) * 1000)

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODELO_FUSION, "messages": mensajes,
               "max_tokens": FUSION_MAX_TOKENS, "temperature": FUSION_TEMPERATURE}
    timeout = aiohttp.ClientTimeout(total=60)

    print("[analizador] Generando fusion...")
    last_error = None
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for attempt in range(RETRY_MAX_ATTEMPTS):
            try:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        latency_ms = int((time.perf_counter() - t0) * 1000)
                        return data["choices"][0]["message"]["content"], latency_ms
                    # 429 / 5xx son transitorios: reintentar.
                    if resp.status == 429 or resp.status >= 500:
                        last_error = f"http_{resp.status}"
                        if attempt < RETRY_MAX_ATTEMPTS - 1:
                            wait = RETRY_BACKOFF_SECONDS[attempt]
                            print(f"[analizador] Fusion {resp.status}, reintento en {wait}s "
                                  f"({attempt+1}/{RETRY_MAX_ATTEMPTS})")
                            await asyncio.sleep(wait)
                            continue
                    # 4xx no-429: no reintentar.
                    print(f"[analizador] Error ChatGPT: {resp.status}")
                    return None, int((time.perf_counter() - t0) * 1000)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = f"{type(e).__name__}: {e}"
                if attempt < RETRY_MAX_ATTEMPTS - 1:
                    wait = RETRY_BACKOFF_SECONDS[attempt]
                    print(f"[analizador] Fusion network error ({last_error}), "
                          f"reintento en {wait}s ({attempt+1}/{RETRY_MAX_ATTEMPTS})")
                    await asyncio.sleep(wait)
                    continue
                print(f"[analizador] Error fusion tras {RETRY_MAX_ATTEMPTS} intentos: {e}")
                return None, int((time.perf_counter() - t0) * 1000)

    print(f"[analizador] Error fusion tras {RETRY_MAX_ATTEMPTS} intentos: {last_error}")
    return None, int((time.perf_counter() - t0) * 1000)


async def generar_fusion(top_respuestas):
    """Devuelve (texto_fusionado, latency_ms_o_None).

    Si no hay al menos 2 respuestas, no se llama a la API: se devuelve
    el texto de la única respuesta (o "") con latencia None.
    """
    if len(top_respuestas) < 2:
        return (top_respuestas[0]["response"] if top_respuestas else ""), None
    partes = "\n\n".join(
        f"PERSPECTIVA {i+1} ({r['model_name']}, consenso={r['consenso_individual']:.3f}):\n{r['response']}"
        for i, r in enumerate(top_respuestas)
    )
    contexto = f"""Sintetiza estas {len(top_respuestas)} perspectivas clinicas del ensemble:

{partes}

Instrucciones:
1. Integra los puntos de acuerdo en una recomendacion coherente.
2. NO elimines divergencias importantes — si las perspectivas difieren en hipotesis diagnosticas, incluyelo explicitamente en una seccion "Puntos de deliberacion clinica".
3. Mantén tono clinico profesional.
4. Responde UNICAMENTE con la sintesis clinica estructurada."""
    mensajes = [
        {"role": "system", "content": "Eres un experto en sintesis clinica multi-perspectiva."},
        {"role": "user",   "content": contexto}
    ]
    resultado, latency_ms = await _llamar_chatgpt(mensajes)
    texto = resultado.strip() if resultado else top_respuestas[0]["response"]
    return texto, latency_ms


# ── Análisis principal ────────────────────────────────────────────────────────

async def dataAnalisis_interno(resultados_ensamblador):
    if not resultados_ensamblador or len(resultados_ensamblador) < 2:
        return {"error": "Se necesitan al menos 2 respuestas"}
    try:
        validos = [r for r in resultados_ensamblador
                   if r.get("response") and len(r["response"].strip()) > 10]
        if len(validos) < 2:
            return {"error": "Respuestas validas insuficientes"}

        textos  = [r["response"] for r in validos]
        nombres = [r["model_name"] for r in validos]

        m_tfidf = _matriz_tfidf(textos)
        m_embed, embedding_truncated_flags = _matriz_embeddings(textos)
        m_principal = m_embed if m_embed is not None else m_tfidf

        cdi_info = calcular_cdi(m_principal)
        consensos_ind = calcular_consensos_individuales(m_principal, nombres)

        vals_off = [float(m_principal[i,j]) for i in range(len(m_principal))
                    for j in range(len(m_principal)) if i != j]
        cg = round(float(np.mean(vals_off)), 4) if vals_off else 0.5

        solo_voz = identificar_solo(consensos_ind, validos)
        divergencia = calcular_divergencia_capas(m_tfidf, m_embed)

        # Para ensembles pequeños (<=4 modelos) usamos todos para fusión.
        # Para ensembles mayores filtramos outliers quedándonos con los 2/3 más consensuados.
        if len(consensos_ind) <= 4:
            mayores = list(consensos_ind)
        else:
            n_top = max(3, len(consensos_ind) * 2 // 3)
            mayores = consensos_ind[:n_top]

        respuesta_mc = consensos_ind[0] if consensos_ind else None
        fusionada, modelos_base, fusion_latency_ms = None, [], None

        # La fusión necesita al menos 2 respuestas válidas (consistente con el resto del módulo).
        if len(mayores) >= 2:
            top3 = mayores[:3]  # como mucho 3 entran a fusion
            top3_datos = []
            for c in top3:
                for r in validos:
                    if r["model_name"] == c["modelo"]:
                        top3_datos.append({"model_name": r["model_name"],
                                           "response": r["response"],
                                           "consenso_individual": c["consenso_individual"]})
                        break
            fusionada, fusion_latency_ms = await generar_fusion(top3_datos)
            modelos_base = [d["model_name"] for d in top3_datos]

        return {
            "consenso_global": cg,
            "consensos_individuales": consensos_ind,
            "respuesta_mas_consensuada": respuesta_mc,
            "mayores_consensos": mayores,
            "indices_filtrados": list(range(len(validos))),
            "nombres_filtrados": nombres,
            "cdi": cdi_info,
            "solo": solo_voz,
            "divergencia_capas": divergencia,
            "embedding_disponible": m_embed is not None,
            "respuesta_fusionada": fusionada,
            "modelos_base": modelos_base,
            "fusion_modelo": MODELO_FUSION if fusionada else None,
            "fusion_latency_ms": fusion_latency_ms,
            "fusion_max_tokens": FUSION_MAX_TOKENS if fusionada else None,
            "fusion_temperature": FUSION_TEMPERATURE if fusionada else None,
            "embedding_truncated_flags": embedding_truncated_flags,
            "_matriz_tfidf": m_tfidf,
            "_matriz_embed": m_embed,
            "_matriz_principal": m_principal,
        }
    except Exception as e:
        print(f"[analizador] Error: {e}")
        import traceback; traceback.print_exc()
        return {"error": str(e)}


async def dataAnalisis(resultados_ensamblador):
    print("\n\t\tANALISIS CHORUS")
    rep = await dataAnalisis_interno(resultados_ensamblador)
    if "error" in rep:
        print(f"[analizador] {rep['error']}")
        return rep
    cdi = rep.get("cdi") or {}
    print(f"Consenso global : {rep['consenso_global']:.3f}")
    cdi_val = cdi.get("cdi")
    cdi_str = f"{cdi_val:.4f}" if isinstance(cdi_val, (int, float)) else "indeterminado"
    print(f"CDI             : {cdi_str} — {cdi.get('etiqueta','')}")
    if rep.get("embedding_disponible"):
        print("Capa semantica  : embeddings OpenAI activos")
    else:
        print("Capa semantica  : solo TF-IDF")
    rm = rep.get("respuesta_mas_consensuada")
    if rm:
        print(f"Mas consensuado : {rm['modelo']} ({rm['consenso_individual']:.3f})")
    pm = rep.get("solo")
    if pm:
        print(f"Mas disidente   : {pm['modelo']} ({pm['consenso_individual']:.3f})")
    div = rep.get("divergencia_capas")
    if div and div.get("n_pares_criticos", 0) > 0:
        print(f"Pares lex!=sem  : {div['n_pares_criticos']} detectados")
    if "_matriz_tfidf" in rep:
        imprimir_matriz_consenso(rep["_matriz_tfidf"], rep["nombres_filtrados"])
    if rep.get("respuesta_fusionada"):
        print("\n\t\tRESPUESTA FUSIONADA:")
        print(rep["respuesta_fusionada"])
    return rep
