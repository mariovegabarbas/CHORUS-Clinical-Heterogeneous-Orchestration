# CHORUS — Clinical Heterogeneous Orchestration

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

Cuando varios LLMs analizan un mismo caso clínico, sus respuestas rara vez coinciden plenamente. El enfoque habitual consiste en promediar o ignorar esas discrepancias. **CHORUS invierte esa lógica**: el desacuerdo entre modelos no es un artefacto que suprimir, sino la señal más informativa que el sistema puede producir.

Esta filosofía se fundamenta en el trabajo **MEDLEY** (Abtahi, Astaraki & Seoane, 2025), que demuestra cómo el sesgo y la imperfección de los modelos individuales pueden aprovecharse constructivamente en aplicaciones médicas. CHORUS extiende ese marco al dominio de la psicoterapia, donde la incertidumbre diagnóstica no indica fallo del sistema sino la naturaleza inherentemente compleja de la condición humana.

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
│  │  (scikit-learn)          │   │  (text-embedding-3-large)   │  │
│  │  Capa léxica             │   │  Capa semántica             │  │
│  └───────────┬─────────────┘   └─────────────┬───────────────┘  │
│              └──────────────┬─────────────────┘                 │
│                             │                                   │
│                    CDI = 1 − det(M_sim)                         │
│                      Voz Solista (SOLO)                         │
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
│   · Tarjeta SOLO destacada                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Contribuciones originales

CHORUS no es una reimplementación de MEDLEY, sino una extensión con contribuciones metodológicas propias orientadas al entorno clínico:

| Componente | MEDLEY (base) | CHORUS (extensión) |
|---|---|---|
| **Estrategia de ensemble** | Múltiples modelos médicos | Múltiples LLMs heterogéneos (OpenRouter) |
| **Tratamiento del sesgo** | Aprovechar el sesgo individual | Cuantificar y exponer el disenso como señal |
| **Métrica de diversidad** | No definida formalmente | **CDI = 1 − det(M_sim)** (contribución propia) |
| **Capas de similitud** | — | TF-IDF (léxica) + embeddings (semántica) simultáneas |
| **Voz Solista** | — | **SOLO**: extracción y exposición explícita de la Voz Solista |
| **Fusión** | Votación / promedio | Síntesis instructada para **preservar** las divergencias clínicas |
| **Dominio** | Diagnóstico médico general | Psicoterapia y razonamiento clínico en salud mental |

### Clinical Dissent Index (CDI)

El CDI es la métrica central del sistema. A partir del schema v1.0, el CDI se reporta como un conjunto de **tres métricas emparejadas** que capturan fenómenos complementarios y son comparables entre ensembles de distinto tamaño.

#### Motivación de la normalización

La definición original `CDI = 1 − |det(M_sim)|` no es comparable entre ensembles de tamaño distinto: el determinante del Gramiano de n vectores semánticamente relacionados tiende a cero a velocidad geométrica en n, de modo que un CDI de 0.8 con diez modelos no representa el mismo grado de disenso clínico que un CDI de 0.8 con cuatro. Por ese motivo, a partir de la versión del schema v1.0, la métrica principal es la **media geométrica de los valores singulares** de la matriz de similitud.

#### Las tres métricas

| Métrica | Fórmula | Interpretación |
|---|---|---|
| **`cdi_geometric`** (principal) | `|det(M)|^(1/n)` | Media geométrica de los valores singulares. Acotada en [0, 1]. Es la métrica que gobierna el banner semafórico y la que debe citarse por defecto en análisis y publicaciones. Normaliza correctamente por el tamaño del ensemble. |
| **`cdi_mean_dissent`** | `1 − mean(off-diagonal(M))` | Disenso medio par-a-par. Interpretación directa e intuitiva: 0 = todas las respuestas semánticamente idénticas, 1 = todas ortogonales. Útil como referencia rápida. |
| **`cdi_entropy`** | Entropía de Shannon sobre las similitudes off-diagonal normalizadas | Capta si el disenso está repartido uniformemente entre los pares o concentrado en pocos de ellos. Complementa a las otras dos cuando hace falta distinguir "disenso global" vs "disenso localizado". |

Además, el payload incluye `cdi_det_raw = 1 − |det(M)|` únicamente como **alias de retrocompatibilidad** para poder releer meta.json generados antes del schema v1.0. No debe usarse en análisis nuevos.

La clave `cdi` del payload del endpoint sigue existiendo y es un alias directo de `cdi_geometric`, lo que evita romper a los consumidores antiguos del JSON.

#### Por qué `cdi_geometric` como métrica principal

Dos razones:

1. **Comparabilidad entre tamaños de ensemble.** La media geométrica de los valores singulares está acotada en [0, 1] independientemente de n, mientras que `|det(M)|` colapsa exponencialmente con n.
2. **Interpretación limpia.** Un `cdi_geometric` alto significa que los vectores de respuesta son aproximadamente ortogonales entre sí (matriz de similitud cercana a la identidad), es decir, el ensemble no converge. Un `cdi_geometric` bajo significa que todas las respuestas están concentradas en un mismo "cono" semántico.

#### Umbrales clínicos

Los umbrales y colores se aplican sobre `cdi_geometric`:

| Nivel | Rango | Color | Significado clínico |
|---|---|---|---|
| **Baja** | 0.00 – 0.25 | 🟢 Verde | Convergencia diagnóstica. Los modelos confluyen en una hipótesis similar. La síntesis es fiable. |
| **Moderada** | 0.25 – 0.60 | 🟡 Amarillo | Ambigüedad moderada. Existen matices. Se recomienda revisar las perspectivas individuales. |
| **Alta** | 0.60 – 0.85 | 🟠 Naranja | Alta diversidad diagnóstica. Caso complejo. La Voz Solista (SOLO) merece evaluación independiente. |
| **Máxima** | 0.85 – 1.00 | 🔴 Rojo | Divergencia crítica. Incertidumbre máxima del ensemble. Presentación atípica o caso límite. |

### Voz Solista (SOLO)

El modelo con menor índice de consenso individual (media de sus similitudes con el resto del ensemble) se expone de forma explícita, con su respuesta completa inalterada. La Voz Solista (SOLO) no es un error: es una hipótesis diagnóstica alternativa que merece evaluación clínica independiente, especialmente en casos de CDI alto o máximo.

#### SOLO canónico (consenso individual = 0)

El caso estructuralmente más interesante es el del modelo con consenso individual **exactamente cero**: semánticamente ortogonal al resto del ensemble. Ese modelo es el *SOLO canónico* en la narrativa del paper y es, justamente, el fenómeno que el sistema está diseñado para no silenciar. Por eso el filtro interno acepta `consenso_individual >= 0` (no `> 0`).

La única excepción es el caso degenerado en el que **todos** los modelos son mutuamente ortogonales (todos los consensos son cero): ahí no existe un SOLO distinguible — cada modelo es tan disidente como los demás — y la función devuelve `None`.

### Doble capa de similitud

La divergencia entre la matriz léxica (TF-IDF) y la semántica (embeddings) actúa como detector de un fenómeno de especial relevancia clínica: modelos que emplean el mismo vocabulario técnico para describir fenómenos distintos, o modelos que difieren en terminología pero coinciden en la estructura conceptual del diagnóstico.

**Modelo de embeddings configurable.** El modelo de embeddings que alimenta la capa semántica se configura mediante la variable de entorno `CHORUS_EMBEDDING_MODEL`. El **default es `text-embedding-3-large`** (3072 dimensiones), elegido por su mayor discriminación semántica en textos clínicos largos respecto a `text-embedding-3-small` a coste marginal. El modelo concreto que se usó en cada ejecución, su dimensionalidad y si hubo fallback (TF-IDF puro por fallo de la API) quedan registrados en el campo `embeddings` del `meta.json` v1.0.

### Trazabilidad del análisis (schema meta.json v1.0)

Cada ejecución produce dos ficheros en `resultados/`:

- `ensamble_YYYYMMDD_HHMMSS.json` — respuestas crudas del ensamble (uso interno del ensamblador).
- `ensamble_YYYYMMDD_HHMMSS.meta.json` — metadatos estructurados del análisis (schema v1.0), que es el que alimenta la trazabilidad científica y el historial en el frontend.

**Decisión RGPD.** El `meta.json` **no contiene el texto completo del prompt** introducido por el visitante. Solo se conservan:

- `prompt_sha256` — hash SHA-256 del prompt (permite detectar reejecuciones del mismo caso sin almacenar su contenido).
- `prompt_preview` — primeros 120 caracteres, orientación humana para el historial.
- `prompt_length_chars` — longitud del prompt en caracteres.

Esto saca a la demo pública del perímetro RGPD de forma limpia. Bajo el campo de entrada del caso en la web se muestra un aviso explícito de este comportamiento.

**Campos del schema v1.0** (extracto): `schema_version`, `case_uuid`, `case_reference_id`, `session_code`, `browser_token`, `timestamp_utc`/`timestamp_local`, `ensemble.{n_modelos, model_type, modelos[]}` con latencia y `api_error` por modelo, `determinismo.{temperature, seed, nota}`, `fusion.{modelo, latency_ms, max_tokens, temperature}`, `matrices.{tfidf, embed, principal}`, `embeddings.{modelo, dimensiones, fallback_aplicado}`, `cdi`, `solo`, `divergencia_capas`, `consenso_global`, `consensos_individuales`, `respuesta_fusionada_sha256`, `chorus_version` (hash del commit en ejecución). La definición y validador están en `schemas/meta_v1.py`; cualquier payload sin todos los campos obligatorios o con tipos incorrectos se rechaza con `MetaValidationError`.

**Versionado.** Los `meta.json` producidos antes del schema v1.0 no se migran automáticamente; se consideran legacy. Futuras versiones del schema subirán la `schema_version` y deberán migrarse de forma explícita.

**Historial por navegador.** La web emite una cookie anónima `chorus_browser_token` (uuid4, `httpOnly`, sin datos personales) la primera vez que un visitante ejecuta un análisis. El endpoint `/api/history` filtra los meta.json por ese token, de modo que cada navegador ve solo sus propios análisis sin necesidad de login ni registro. Si la cookie se borra, se pierde ese historial — consistente con el diseño "sin registro".

### Casos de referencia y `session_code`

**Casos de referencia.** El fichero `casos_referencia.json` en la raíz del repo cataloga casos clínicos anonimizados que el ensemble reejecuta periódicamente como medida estable. Sirven para dos cosas:

1. **Demo sin fricción.** Un visitante sin caso propio puede lanzar CHORUS con un click sobre cualquier caso de ejemplo. Es la mejor forma de mostrar la herramienta al estilo de medley.smile.ki.se.
2. **Medida longitudinal de Jensen-Shannon.** Reejecutar los mismos casos en distintos momentos permite cuantificar la deriva de las distribuciones de outputs del ensemble en el tiempo (la componente dinámica del constructo, diferenciadora respecto a otros sistemas de IA clínica).

El endpoint `GET /api/reference_cases` devuelve la lista con `id`, `titulo`, `descripcion_corta` y `ensemble_recomendado`. **No** expone `texto_completo`: cuando el visitante selecciona un caso, el frontend envía solo el `id` en el POST de `/api/run-ensemble` y el backend carga el texto desde el catálogo. Así los casos quedan auditables en el servidor y no se exponen en claro al navegador.

El `prompt_sha256` del meta.json se calcula sobre el `texto_completo` del caso (no sobre el id), lo que hace que dos reejecuciones del mismo caso produzcan el mismo sha y sean detectables como tales.

**`session_code` (opcional).** `POST /api/run-ensemble` acepta un campo libre `session_code` que se persiste literal en el `meta.json`. Es la única conexión operativa de CHORUS con los estudios de validación gestionados externamente (Google Forms, Qualtrics, ConfAI). La mecánica — que es **protocolo de investigación y no código**:

1. El investigador enrola a un terapeuta y le asigna un identificador de estudio (p. ej. `KARO-P002`).
2. En cada sesión de recogida (T0, T1, T2), le pide que ejecute N casos de referencia con el `session_code` correspondiente (p. ej. `KARO-T0-P002`).
3. Al final, el terapeuta rellena el cuestionario ConfAI en la plataforma externa. El primer campo del cuestionario es el mismo `session_code`.
4. En el análisis, el investigador cruza los `meta.json` filtrados por `session_code` con las respuestas del cuestionario del mismo `session_code`.

En la web pública, el campo `session_code` vive en "Opciones avanzadas (investigadores)", oculto del visitante casual. El catálogo de casos de referencia se puede apuntar a un fichero alternativo mediante la variable de entorno `CHORUS_REFERENCE_CASES`.

### Robustez del pipeline

**Escritura atómica del `meta.json`.** `_guardar_meta` escribe primero en `<base>.meta.json.tmp`, fuerza el volcado a disco con `fsync` y usa `os.replace` para mover el fichero a su ruta final. `os.replace` es atómico en POSIX y en Windows (cuando origen y destino están en el mismo volumen): el `meta.json` definitivo nunca existe en estado parcial — o no existe, o está completo y validado contra el schema v1.0. Un SIGKILL a mitad de escritura, un fallo de serialización o un `fsync` fallido dejan el directorio de salida limpio (el `.tmp` huérfano se borra en el `except`).

**Fusión no bloqueante.** `analizador._llamar_chatgpt` y `analizador.generar_fusion` son `async` y usan `aiohttp` en lugar de `requests` síncrono. Antes, la llamada de fusión se ejecutaba sincrónicamente dentro de `asyncio.run(ejecutar())` y bloqueaba el event loop mientras esperaba la respuesta de OpenAI; ahora encaja naturalmente con el resto del pipeline asíncrono del ensamblador.

**Reintentos con backoff exponencial.** Tanto la fusión como cada llamada individual del ensamblador a OpenRouter reintentan hasta 3 veces con espera 1s / 2s / 4s para `429` y `5xx` (y errores transitorios de red: `ClientError`, `TimeoutError`). Los `4xx` no-429 no se reintentan: no son transitorios. La configuración está en `RETRY_MAX_ATTEMPTS` y `RETRY_BACKOFF_SECONDS` dentro de `analizador.py` y `Ensambladores/ensamblador_LLM.py`.

---

## Instalación

**Requisitos previos:** Python 3.11+, claves de API de OpenAI y OpenRouter.

```bash
# 1. Clonar el repositorio
git clone https://github.com/mariovegabarbas/CHORUS-Clinical-Heterogeneous-Orchestration
cd CHORUS-Clinical-Heterogeneous-Orchestration

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
   - **Tarjeta SOLO** con la Voz Solista.
   - **Síntesis fusionada** con los puntos de deliberación clínica.

### Interfaz CLI

```bash
python main.py
```

La salida estructurada incluye el CDI, las respuestas individuales, la Voz Solista (SOLO) y la síntesis, en formato texto con opción de exportación JSON.

---

## Estructura del proyecto

```
CHORUS/
├── app.py                   # Backend Flask — API REST (puerto 8282)
├── main.py                  # Interfaz de línea de comandos
├── analizador.py            # Cálculo del CDI, consenso, similitud y fusión
├── cargador_modelos.py      # Carga dinámica de modelos desde modelos.json
├── modelos.json             # Catálogo de modelos disponibles (OpenRouter)
├── Ensambladores/
│   ├── __init__.py
│   └── ensamblador_LLM.py   # Consultas asíncronas al ensemble de LLMs
├── static/
│   └── index.html           # Frontend web (HTML/CSS/JS vanilla)
└── resultados/              # Salida de análisis (generado en ejecución)
    ├── ensamble_YYYYMMDD_HHMMSS.json        # Respuestas + CDI + SOLO + síntesis
    └── ensamble_YYYYMMDD_HHMMSS.meta.json   # Metadatos del análisis
```

---

## Stack tecnológico

| Capa | Tecnología |
|---|---|
| **Orquestación** | Python 3.11, asyncio, aiohttp |
| **Backend web** | Flask, Flask-CORS |
| **Análisis léxico** | scikit-learn (TF-IDF + cosine similarity) |
| **Análisis semántico** | OpenAI `text-embedding-3-large` (configurable vía `CHORUS_EMBEDDING_MODEL`) |
| **Fusión** | OpenAI `gpt-3.5-turbo` |
| **Acceso a modelos** | OpenRouter API (400+ modelos) |
| **Frontend** | HTML5 / CSS3 / JavaScript vanilla |

**Variables de entorno requeridas:**

```
OPENAI_API_KEY            — Fusión (GPT-3.5-turbo) y embeddings semánticos
OPENROUTER_API_KEY        — Ensemble de modelos heterogéneos
CHORUS_OUTPUT_PATH        — (opcional) Ruta de salida de resultados. Por defecto: resultados/
CHORUS_EMBEDDING_MODEL    — (opcional) Modelo de embeddings. Por defecto: text-embedding-3-large
```

---

## Para investigadores

### Cómo citar este trabajo

Si utilizas CHORUS en tu investigación, por favor cita el trabajo asociado (próxima publicación) y la referencia metodológica principal:

```bibtex
@misc{chorus2025,
  author    = {Vega-Barbas, Mario and Grimaldos, Javier},
  title     = {{CHORUS}: Clinical Heterogeneous Orchestration — Quantifying Diagnostic Dissent as Clinical Signal},
  year      = {2025},
  note      = {InnoTep Research Group, Universidad Politécnica de Madrid. En preparación para publicación.},
  url       = {https://github.com/mariovegabarbas/CHORUS-Clinical-Heterogeneous-Orchestration}
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

Edita `modelos.json` siguiendo la estructura existente. CHORUS soporta cualquier modelo disponible en OpenRouter sin cambios en el código:

```json
{ "name": "proveedor/nombre-modelo", "origin": "País", "size": "7B" }
```

**Implementar una métrica de disenso alternativa**

El módulo `analizador.py` expone las matrices de similitud léxica y semántica. Cualquier función `f(M_sim) → [0,1]` puede sustituir al CDI actual para experimentación comparativa.

**Modificar el prompt de fusión**

El prompt de síntesis en `analizador.py` puede adaptarse a distintos marcos clínicos (DSM-5, CIE-11, formulación psicodinámica, etc.) o a otros dominios de razonamiento clínico.

**Extender la Voz Solista (SOLO)**

Actualmente se expone un único outlier. Es posible generalizar a un ranking completo de modelos por índice de consenso individual, o aplicar clustering sobre la matriz de similitud para identificar subgrupos diagnósticos cohesivos.

---

## Aviso clínico

> ⚠️ **CHORUS es una herramienta de apoyo al razonamiento clínico, no un sistema de diagnóstico.**

Las salidas del sistema —incluyendo la síntesis fusionada, el CDI y la Voz Solista (SOLO)— son perspectivas generadas por modelos de lenguaje con fines de exploración y deliberación clínica. **No sustituyen, en ningún caso, el criterio profesional del psicoterapeuta o del equipo clínico responsable.**

El sistema no tiene acceso a la historia clínica del paciente, no puede realizar una evaluación directa, y sus respuestas están condicionadas por los sesgos inherentes a los modelos de lenguaje utilizados. Cualquier decisión diagnóstica o terapéutica debe recaer exclusivamente en el profesional de la salud mental.

---

## Licencia

Distribuido bajo licencia [MIT](LICENSE). Consulta el fichero `LICENSE` para más detalles.

---

<p align="center">
  <em>CHORUS · <a href="https://github.com/mariovegabarbas/CHORUS-Clinical-Heterogeneous-Orchestration">InnoTep Research Group</a> · Universidad Politécnica de Madrid · 2025</em>
</p>
