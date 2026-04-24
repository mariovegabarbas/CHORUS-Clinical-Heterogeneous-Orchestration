"""
analizador.py — CRML v2
=======================
Mejoras sobre v1:
  1. Clinical Dissent Index (CDI) formalizado con umbrales clínicos
  2. Perspectiva Minoritaria Destacada: el modelo más disidente expuesto explícitamente
  3. Doble capa de similitud: TF-IDF (léxica) + Embeddings OpenAI (semántica)
  4. Taxonomía de complejidad diagnóstica basada en CDI
  5. Reporte enriquecido para el frontend
"""

import os
import numpy as np
import json
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

# ── Configuración ────────────────────────────────────────────────────────────
API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODELO_FUSION = "gpt-3.5-turbo"

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

def _obtener_embeddings(textos):
    if not API_KEY or API_KEY.startswith("//"):
        return None
    try:
        url = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        payload = {"model": "text-embedding-3-small", "input": [t[:8000] for t in textos]}
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            return np.array([d["embedding"] for d in data["data"]])
        print(f"[analizador] Embeddings API {resp.status_code}")
        return None
    except Exception as e:
        print(f"[analizador] Error embeddings: {e}")
        return None


def _matriz_embeddings(textos):
    vecs = _obtener_embeddings(textos)
    if vecs is None:
        return None
    return cosine_similarity(vecs)


# ── CDI ───────────────────────────────────────────────────────────────────────

def calcular_cdi(matriz_sim):
    n = len(matriz_sim)
    if n < 2:
        return {"cdi": 0.0, "nivel": "baja", "etiqueta": CDI_ETIQUETAS["baja"],
                "color": CDI_COLORES["baja"], "determinante": 1.0, "entropia": 0.0}
    try:
        det = float(np.linalg.det(matriz_sim))
        cdi = float(np.clip(1.0 - det, 0.0, 1.0))
    except Exception:
        cdi, det = 0.5, 0.5

    off = [float(matriz_sim[i, j]) for i in range(n) for j in range(n) if i != j]
    if off:
        v = np.clip(np.array(off), 1e-10, 1.0)
        v = v / v.sum()
        entropia = float(-np.sum(v * np.log(v + 1e-10)))
    else:
        entropia = 0.0

    nivel = _nivel_cdi(cdi)
    return {
        "cdi": round(cdi, 4),
        "nivel": nivel,
        "etiqueta": CDI_ETIQUETAS[nivel],
        "color": CDI_COLORES[nivel],
        "determinante": round(det, 6),
        "entropia": round(entropia, 4)
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


# ── Perspectiva Minoritaria ───────────────────────────────────────────────────

def identificar_perspectiva_minoritaria(consensos_ind, resultados_raw):
    validos = [c for c in consensos_ind if c["consenso_individual"] > 0]
    if not validos:
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
        "perspectiva_minoritaria": rep.get("perspectiva_minoritaria"),
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
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        payload = {"model": MODELO_FUSION, "messages": mensajes,
                   "max_tokens": 1500, "temperature": 0.2}
        print("[analizador] Generando fusion...")
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        print(f"[analizador] Error ChatGPT: {resp.status_code}")
        return None
    except Exception as e:
        print(f"[analizador] Error fusion: {e}")
        return None


def generar_fusion(top_respuestas):
    if len(top_respuestas) < 2:
        return top_respuestas[0]["response"] if top_respuestas else ""
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
    resultado = _llamar_chatgpt(mensajes)
    return resultado.strip() if resultado else top_respuestas[0]["response"]


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
        m_embed = _matriz_embeddings(textos)
        m_principal = m_embed if m_embed is not None else m_tfidf

        cdi_info = calcular_cdi(m_principal)
        consensos_ind = calcular_consensos_individuales(m_principal, nombres)

        vals_off = [float(m_principal[i,j]) for i in range(len(m_principal))
                    for j in range(len(m_principal)) if i != j]
        cg = round(float(np.mean(vals_off)), 4) if vals_off else 0.5

        perspectiva_min = identificar_perspectiva_minoritaria(consensos_ind, validos)
        divergencia = calcular_divergencia_capas(m_tfidf, m_embed)

        n_top = max(1, len(consensos_ind) * 2 // 3)
        mayores = consensos_ind[:n_top]
        respuesta_mc = consensos_ind[0] if consensos_ind else None

        fusionada, modelos_base = None, []
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
            fusionada = generar_fusion(top3_datos)
            modelos_base = [d["model_name"] for d in top3_datos]

        return {
            "consenso_global": cg,
            "consensos_individuales": consensos_ind,
            "respuesta_mas_consensuada": respuesta_mc,
            "mayores_consensos": mayores,
            "indices_filtrados": list(range(len(validos))),
            "nombres_filtrados": nombres,
            "cdi": cdi_info,
            "perspectiva_minoritaria": perspectiva_min,
            "divergencia_capas": divergencia,
            "embedding_disponible": m_embed is not None,
            "respuesta_fusionada": fusionada,
            "modelos_base": modelos_base,
            "_matriz_tfidf": m_tfidf,
            "_matriz_embed": m_embed,
            "_matriz_principal": m_principal,
        }
    except Exception as e:
        print(f"[analizador] Error: {e}")
        import traceback; traceback.print_exc()
        return {"error": str(e)}


def dataAnalisis(resultados_ensamblador):
    print("\n\t\tANALISIS CRML v2")
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
    pm = rep.get("perspectiva_minoritaria")
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
