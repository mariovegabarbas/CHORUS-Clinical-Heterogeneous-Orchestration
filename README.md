# CRML — Clinical Reasoning Multi-LLM

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=flat-square&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/Licencia-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Estado-En%20investigación-orange?style=flat-square)

> **Sistema de razonamiento clínico multi-modelo que cuantifica y expone el disenso diagnóstico como señal clínica.**  
> Grupo de investigación **InnoTep** · Universidad Politécnica de Madrid · En preparación para publicación en *JMIR Mental Health* / *IEEE JBHI*

---

## Índice

1. [Motivación](#motivación)
2. [Arquitectura del sistema](#arquitectura-del-sistema)
3. [Contribuciones originales](#contribuciones-originales)
4. [Clinical Dissent Index (CDI)](#clinical-dissent-index-cdi)
5. [Instalación](#instalación)
6. [Uso](#uso)
7. [Estructura del proyecto](#estructura-del-proyecto)
8. [Stack tecnológico](#stack-tecnológico)
9. [Para investigadores](#para-investigadores)
10. [Aviso clínico](#aviso-clínico)
11. [Licencia](#licencia)

---

## Motivación

Cuando varios LLMs analizan un mismo caso clínico, sus respuestas rara vez coinciden plenamente. El enfoque habitual consiste en promediar o ignorar esas discrepancias. **CRML invierte esa lógica**: el desacuerdo entre modelos no es un artefacto que suprimir, sino la señal más informativa que el sistema puede producir.

Esta filosofía se fundamenta en el trabajo **MEDLEY** (Abtahi, Astaraki & Seoane, 2025), que demuestra cómo el sesgo y la imperfección de los modelos individuales pueden aprovecharse constructivamente en aplicaciones médicas. CRML extiende ese marco al dominio de la psicoterapia, donde la incertidumbre diagnóstica no indica fallo del sistema sino la naturaleza inherentemente compleja de la condición humana.

---

## Arquitectura del sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                        ENTRADA CLÍNICA                          │
│              (descripción del caso por el psicoterapeuta)       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ENSAMBLADOR MULTI-LLM                        │
│              ensamblador_LLM.py  ·  asyncio / aiohttp           │
│                                                                 │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│   │  LLM-1   │  │  LLM-2   │  │  LLM-3   │  │  LLM-N   │      │
│   │(OpenRtr) │  │(OpenRtr) │  │(OpenRtr) │  │(OpenRtr) │      │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘      │
└────────┼─────────────┼─────────────┼──────────────┼────────────┘
         │             │             │              │
         └─────────────┴──────┬──────┴──────────────┘
                              │  respuestas individuales
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ANALIZADOR (CDI)                           │
│                      analizador.py                              │
│                                                                 │
│  ┌─────────────────────────┐   ┌─────────────────────────────┐  │
│  │  Similitud TF-IDF        │   │  Similitud semántica        │  │
│  │  (scikit-learn)          │   │  (text-embedding-3-small)   │  │
│  │  Capa léxica             │   │  Capa semántica             │  │
│  └───────────┬─────────────┘   └─────────────┬───────────────┘  │
│              └──────────────┬─────────────────┘                 │
│                             │                                   │
│                    CDI = 1 − det(M_sim)                         │
│             Perspectiva Minoritaria Destacada (PMD)             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FUSIÓN PRESERVADORA DEL DISENSO                │
│               GPT-3.5-turbo  ·  OpenAI API                      │
│         (síntesis con sección "Puntos de deliberación")         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      INTERFAZ DE SALIDA                         │
│                                                                 │
│   Web (Flask + HTML/JS)          CLI (main.py)                  │
│   · Banner CDI semafórico        · Salida estructurada          │
│   · Mapa consenso-disenso        · JSON exportable              │
│   · Tarjeta PMD destacada                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Contribuciones originales

CRML no es una reimplementación de MEDLEY, sino una extensión con contribuciones metodológicas propias orientadas al entorno clínico:

| Componente | MEDLEY (base) | CRML (extensión) |
|---|---|---|
| **Estrategia de ensemble** | Múltiples modelos médicos | Múltiples LLMs heterogéneos (OpenRouter) |
| **Tratamiento del sesgo** | Aprovechar el sesgo individual | Cuantificar y exponer el disenso como señal |
| **Métrica de diversidad** | No definida formalmente | **CDI = 1 − det(M_sim)** (contribución propia) |
| **Capas de similitud** | — | TF-IDF (léxica) + embeddings (semántica) simultáneas |
| **Perspectiva minoritaria** | — | **PMD**: extracción y exposición explícita del outlier diagnóstico |
| **Fusión** | Votación / promedio | Síntesis instructada para **preservar** las divergencias clínicas |
| **Dominio** | Diagnóstico médico general | Psicoterapia y razonamiento clínico en salud mental |

### Clinical Dissent Index (CDI)

El CDI es la métrica central del sistema. Se define como:

```
CDI = 1 − |det(M_sim)|
```

donde `M_sim` es la matriz de similitud entre todas las respuestas del ensemble. Un determinante cercano a 1 indica respuestas casi idénticas (bajo disenso); al alejarse de 1, el determinante cae y el CDI sube, reflejando mayor diversidad diagnóstica.

| Nivel CDI | Rango | Color | Significado clínico |
|---|---|---|---|
| **Bajo** | 0.00 – 0.25 | 🟢 Verde | Consenso amplio. Los modelos convergen en una hipótesis diagnóstica similar. La síntesis es fiable. |
| **Moderado** | 0.26 – 0.50 | 🟡 Amarillo | Divergencia notable. Existen matices de interpretación. Se recomienda revisar las perspectivas individuales. |
| **Alto** | 0.51 – 0.75 | 🟠 Naranja | Desacuerdo sustancial. El caso presenta complejidad diagnóstica real. La PMD merece evaluación independiente. |
| **Máximo** | 0.76 – 1.00 | 🔴 Rojo | Disenso extremo. El ensemble no converge. Indicador de alta ambigüedad clínica o presentación atípica. |

### Perspectiva Minoritaria Destacada (PMD)

El modelo con menor índice de consenso individual (suma de sus similitudes con el resto del ensemble) se expone de forma explícita, con su respuesta completa inalterada. La PMD no es un error: es una hipótesis diagnóstica alternativa que merece evaluación clínica independiente, especialmente en casos de CDI alto o máximo.

### Doble capa de similitud

La divergencia entre la matriz léxica (TF-IDF) y la semántica (embeddings) actúa como detector de un fenómeno de especial relevancia clínica: modelos que emplean el mismo vocabulario técnico para describir fenómenos distintos, o modelos que difieren en terminología pero coinciden en la estructura conceptual del diagnóstico.

---

## Instalación

**Requisitos previos:** Python 3.11+, claves de API de OpenAI y OpenRouter.

```bash
# 1. Clonar el repositorio
git clone https://github.com/mariovegabarbas/CRML-Clinical-Reasoning-Multi-LLM-framework
cd CRML-Clinical-Reasoning-Multi-LLM-framework

# 2. Instalar dependencias
pip install flask flask-cors scikit-learn numpy requests aiohttp openai python-dotenv

# 3. Configurar claves de API
cp .env.example .env
# Edita .env y rellena tus claves:
#   OPENAI_API_KEY=sk-...
#   OPENROUTER_API_KEY=sk-or-...

# 4. Lanzar el servidor
python app.py
```

Abre [http://localhost:8282](http://localhost:8282) en tu navegador.

---

## Uso

### Interfaz web

1. Introduce la descripción del caso clínico en el área de texto.
2. Selecciona los modelos del ensemble desde el catálogo (`modelos.json`).
3. Ejecuta el análisis. El sistema mostrará:
   - **Banner CDI** con código de color y nivel de disenso.
   - **Mapa de consenso-disenso** entre modelos.
   - **Respuestas individuales** de cada LLM.
   - **Tarjeta PMD** con la perspectiva minoritaria destacada.
   - **Síntesis fusionada** con los puntos de deliberación clínica.

### Interfaz CLI

```bash
python main.py
```

La salida estructurada incluye el CDI, las respuestas individuales, la PMD y la síntesis, en formato texto con opción de exportación JSON.

---

## Estructura del proyecto

```
CRML-v2/
├── app.py                   # Backend Flask — API REST (puerto 8282)
├── main.py                  # Interfaz de línea de comandos
├── analizador.py            # Cálculo del CDI, consenso, similitud y fusión
├── cargador_modelos.py      # Carga dinámica de modelos desde modelos.json
├── modelos.json             # Catálogo de modelos disponibles (OpenRouter)
├── Ensambladores/
│   └── ensamblador_LLM.py   # Consultas asíncronas al ensemble de LLMs
└── static/
    └── index.html           # Frontend web (HTML/CSS/JS vanilla)
```

---

## Stack tecnológico

| Capa | Tecnología |
|---|---|
| **Orquestación** | Python 3.11, asyncio, aiohttp |
| **Backend web** | Flask, Flask-CORS |
| **Análisis léxico** | scikit-learn (TF-IDF + cosine similarity) |
| **Análisis semántico** | OpenAI `text-embedding-3-small` |
| **Fusión** | OpenAI `gpt-3.5-turbo` |
| **Acceso a modelos** | OpenRouter API (400+ modelos) |
| **Frontend** | HTML5 / CSS3 / JavaScript vanilla |

**Variables de entorno requeridas:**

```
OPENAI_API_KEY        — Fusión (GPT-3.5-turbo) y embeddings semánticos
OPENROUTER_API_KEY    — Ensemble de modelos heterogéneos
```

---

## Para investigadores

### Cómo citar este trabajo

Si utilizas CRML en tu investigación, por favor cita el trabajo asociado (próxima publicación) y la referencia metodológica principal:

```bibtex
@misc{crml2025,
  author    = {Vega-Barbas, Mario and Grimaldos, Javier},
  title     = {{CRML}: Clinical Reasoning Multi-{LLM} — Quantifying Diagnostic Dissent as Clinical Signal},
  year      = {2025},
  note      = {InnoTep Research Group, Universidad Politécnica de Madrid. En preparación para publicación.},
  url       = {https://github.com/mariovegabarbas/CRML-Clinical-Reasoning-Multi-LLM-framework}
}

@misc{abtahi2025medley,
  author        = {Abtahi, Faryar and Astaraki, Mehdi and Seoane, Fernando},
  title         = {Leveraging Imperfection with {MEDLEY}: A Multi-Model Approach Harnessing Bias in Medical {AI}},
  year          = {2025},
  eprint        = {2508.21648},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}
```

### Cómo extender el sistema

**Añadir nuevos modelos al ensemble**

Edita `modelos.json` siguiendo la estructura existente. CRML soporta cualquier modelo disponible en OpenRouter sin cambios en el código:

```json
{ "name": "proveedor/nombre-modelo", "origin": "País", "size": "7B" }
```

**Implementar una métrica de disenso alternativa**

El módulo `analizador.py` expone las matrices de similitud léxica y semántica. Cualquier función `f(M_sim) → [0,1]` puede sustituir al CDI actual para experimentación comparativa.

**Modificar el prompt de fusión**

El prompt de síntesis en `analizador.py` puede adaptarse a distintos marcos clínicos (DSM-5, CIE-11, formulación psicodinámica, etc.) o a otros dominios de razonamiento clínico.

**Extender la PMD**

Actualmente se expone un único outlier. Es posible generalizar a un ranking completo de modelos por índice de consenso individual, o aplicar clustering sobre la matriz de similitud para identificar subgrupos diagnósticos cohesivos.

---

## Aviso clínico

> ⚠️ **CRML es una herramienta de apoyo al razonamiento clínico, no un sistema de diagnóstico.**

Las salidas del sistema —incluyendo la síntesis fusionada, el CDI y la PMD— son perspectivas generadas por modelos de lenguaje con fines de exploración y deliberación clínica. **No sustituyen, en ningún caso, el criterio profesional del psicoterapeuta o del equipo clínico responsable.**

El sistema no tiene acceso a la historia clínica del paciente, no puede realizar una evaluación directa, y sus respuestas están condicionadas por los sesgos inherentes a los modelos de lenguaje utilizados. Cualquier decisión diagnóstica o terapéutica debe recaer exclusivamente en el profesional de la salud mental.

---

## Licencia

Distribuido bajo licencia [MIT](LICENSE). Consulta el fichero `LICENSE` para más detalles.

---

<p align="center">
  <em>CRML · <a href="https://github.com/mariovegabarbas/CRML-Clinical-Reasoning-Multi-LLM-framework">InnoTep Research Group</a> · Universidad Politécnica de Madrid · 2025</em>
</p>
