"""
analizador.py — CHORUS
======================
  1. Clinical Dissent Index (CDI) formalizado con umbrales clínicos
  2. Voz Solista (SOLO): el modelo más disidente expuesto explícitamente
  3. Doble capa de similitud: TF-IDF (léxica) + Embeddings OpenAI (semántica)
  4. Taxonomía de complejidad diagnóstica basada en CDI
  5. Reporte enriquecido para el frontend
"""

import os
import time
import numpy as np
import json
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

EMBEDDING_MAX_CHARS = 8000


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
        payload = {"model": "text-embedding-3-small",
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


# ── Compat con app.py ─────────────────────────────────────────────────────────

def obtener_matriz_consenso_completa(respuestas):
    if len(respuestas) < 2:
        return np.array([[1.0]]), 1.0, [0]
    try:
        limpias = [(i, r.strip()) for i, r in enumerate(respuestas)
                   if r and len(r.strip()) > 10]
        if len(limpias) < 2:
            n = len(respuestas)
            return np.identity(n), 0.5, list(range(n))
        indices = [i for i, _ in limpias]
        textos  = [t for _, t in limpias]
        sim = _matriz_tfidf(textos)
        n_total = len(respuestas)
        full = np.zeros((n_total, n_total))
        for ii, idx_i in enumerate(indices):
            for jj, idx_j in enumerate(indices):
                full[idx_i, idx_j] = sim[ii, jj]
        for i in range(n_total):
            if i not in indices:
                full[i, i] = 1.0
        vals = [full[i, j] for i in indices for j in indices if i != j]
        cg = round(float(np.mean(vals)), 4) if vals else 0.5
        return full, cg, indices
    except Exception as e:
        print(f"[analizador] Error obtener_matriz: {e}")
        n = len(respuestas)
        return np.identity(n), 0.5, list(range(n))


def calcular_consenso_semantico(respuestas, nombres):
    raws = [{"model_name": n, "response": r, "timestamp": ""}
            for n, r in zip(nombres, respuestas)]
    rep = dataAnalisis_interno(raws)
    return {
        "matriz_consenso": rep.get("_matriz_tfidf", np.identity(len(respuestas))),
        "consenso_global": rep.get("consenso_global", 0.5),
        "consensos_individuales": rep.get("consensos_individuales", []),
        "mayores_consensos": rep.get("mayores_consensos", []),
        "respuesta_mas_consensuada": rep.get("respuesta_mas_consensuada"),
        "indices_filtrados": rep.get("indices_filtrados", list(range(len(respuestas)))),
        "nombres_filtrados": nombres,
        "cdi": rep.get("cdi"),
        "solo": rep.get("solo"),
        "divergencia_capas": rep.get("divergencia_capas"),
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

def _llamar_chatgpt(mensajes):
    """Llama al modelo de fusión. Devuelve (texto, latency_ms).

    latency_ms se mide siempre, incluso en caso de error. El texto
    es None si la llamada falló.
    """
    t0 = time.perf_counter()
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        payload = {"model": MODELO_FUSION, "messages": mensajes,
                   "max_tokens": FUSION_MAX_TOKENS, "temperature": FUSION_TEMPERATURE}
        print("[analizador] Generando fusion...")
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"], latency_ms
        print(f"[analizador] Error ChatGPT: {resp.status_code}")
        return None, latency_ms
    except Exception as e:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        print(f"[analizador] Error fusion: {e}")
        return None, latency_ms


def generar_fusion(top_respuestas):
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
    resultado, latency_ms = _llamar_chatgpt(mensajes)
    texto = resultado.strip() if resultado else top_respuestas[0]["response"]
    return texto, latency_ms


# ── Análisis principal ────────────────────────────────────────────────────────

def dataAnalisis_interno(resultados_ensamblador):
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

        n_top = max(1, len(consensos_ind) * 2 // 3)
        mayores = consensos_ind[:n_top]
        respuesta_mc = consensos_ind[0] if consensos_ind else None

        fusionada, modelos_base, fusion_latency_ms = None, [], None
        if len(mayores) >= 3:
            top3 = mayores[:3]
            top3_datos = []
            for c in top3:
                for r in validos:
                    if r["model_name"] == c["modelo"]:
                        top3_datos.append({"model_name": r["model_name"],
                                           "response": r["response"],
                                           "consenso_individual": c["consenso_individual"]})
                        break
            fusionada, fusion_latency_ms = generar_fusion(top3_datos)
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


def dataAnalisis(resultados_ensamblador):
    print("\n\t\tANALISIS CHORUS")
    rep = dataAnalisis_interno(resultados_ensamblador)
    if "error" in rep:
        print(f"[analizador] {rep['error']}")
        return rep
    cdi = rep.get("cdi", {})
    print(f"Consenso global : {rep['consenso_global']:.3f}")
    print(f"CDI             : {cdi.get('cdi','N/A')} — {cdi.get('etiqueta','')}")
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
