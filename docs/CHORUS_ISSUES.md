# CHORUS — Documento de trabajo

**Seis issues priorizadas para Claude Code**
Web pública como demo de bajísima fricción · Estudios gestionados externamente · Abril 2026

---

## Encuadre del proyecto y propósito de este documento

CHORUS va a vivir en una página web pública al estilo de medley.smile.ki.se. Esa página cumple tres funciones simultáneamente: escaparate del programa de investigación (explica la idea, enlaza al paper, muestra casos de ejemplo), demo funcional del sistema (cualquier visitante puede probar CHORUS contra un caso propio o de ejemplo) y evidencia de que el sistema produce señal útil (a través de los papers derivados del estudio de validación).

Las tres funciones tienen audiencias distintas: investigadores curiosos, potenciales colaboradores, revisores, evaluadores de financiación, clínicos que descubren el sistema. Ninguna de esas audiencias debe encontrar fricción alguna al usar la demo. El sistema tiene que ser satisfactorio en los primeros 60 segundos, funcionar sin pedir registro, sin cuestionarios, sin condiciones.

La recogida de datos para el estudio de validación (Karolinska y los que vengan después) se gestiona por fuera, manualmente. El investigador enrola a los terapeutas, firma consentimientos informados, les guía a través de sesiones de uso con casos de referencia concretos, y administra los cuestionarios en los tres momentos del diseño psicométrico (T0, T1, T2) usando una plataforma externa (Google Forms, Qualtrics o equivalente). La escala ConfAI vive fuera de CHORUS. Esa decisión es deliberada y tiene ventajas conceptuales importantes: ConfAI queda desacoplado del sistema concreto al que se aplica, lo que refuerza su carácter transversal.

Este documento contiene las seis tareas que debe ejecutar Claude Code para que CHORUS pase de su estado actual al estado objetivo: demo pública robusta, con trazabilidad suficiente para auditar y para medir la divergencia de Jensen Shannon sobre casos de referencia a lo largo del tiempo, y con una única pieza de conexión ligera hacia los estudios gestionados manualmente.

### Nomenclatura a usar a partir de ahora en el código

- Sistema: **CHORUS** (Clinical Heterogeneous Orchestration for Reasoning Under Supervision).
- Voz minoritaria: **SOLO** (Solo Voice). En el código: `identificar_solo`, `extraer_solo`, etc. "Perspectiva minoritaria" se sustituye en todas sus formas.
- Métrica central: **CDI** (Clinical Dissent Index). Sin cambios en el acrónimo, sí en la implementación.
- Variables de entorno: todo `CRML_*` pasa a `CHORUS_*`.

---

## Resumen de las seis issues

Las cuatro primeras son críticas para la integridad científica de cualquier dato que se genere. La quinta introduce la única conexión operativa necesaria entre la demo pública y los estudios externos. La sexta resuelve la robustez del pipeline. Un paquete de fixes menores cierra el documento.

| # | Título | Bloquea | Ficheros |
|---|--------|---------|----------|
| 1 | Normalización del CDI y métricas emparejadas | Validez de constructo (paper) | `analizador.py`, `app.py`, frontend |
| 2 | Fix del SOLO canónico (consenso cero) | Validez del mecanismo antisupresión | `analizador.py`, `tests/` |
| 3 | Trazabilidad del meta.json (schema v1.0) | Reproducibilidad y auditoría RGPD | `app.py`, `schemas/` |
| 4 | Saneamiento del CDI contra NaN | Fiabilidad del demo público | `analizador.py`, frontend |
| 5 | Casos de referencia y `session_code` opcional | Medida longitudinal de D_JS | `app.py`, `casos_referencia.json`, frontend |
| 6 | Fusión no bloqueante y guardado atómico | Integridad del dataset | `analizador.py`, `app.py` |

---

## Issue 1 · Normalización del CDI y métricas emparejadas

### Prioridad
Crítica. Es la issue número uno del documento. Ninguna otra tiene más peso para la validez del paper.

### Problema
La definición actual `CDI = 1 - |det(M)|` no es comparable entre ensembles de distinto tamaño. El determinante del Gramiano de n vectores casi unitarios en un espacio de alta dimensión tiende a cero a velocidad geométrica en n. Un CDI de 0.8 con diez modelos no refleja el mismo grado de disenso clínico que un CDI de 0.8 con cuatro. Un revisor metodológico de JMIR Mental Health lo detectará en el abstract.

### Objetivo
Reemplazar la métrica única por un conjunto de tres métricas emparejadas que captan fenómenos distintos y son comparables entre ensembles de tamaño variable. Conservar el cálculo antiguo bajo otro nombre únicamente para retrocompatibilidad con los meta.json ya guardados.

### Ficheros afectados
- `analizador.py`: refactor de `calcular_cdi`.
- `app.py`: el endpoint devuelve las tres métricas en `consenso_data.cdi`, manteniendo la clave `cdi` como alias de `cdi_geometric` para no romper el frontend.
- `static/index.html`: el banner semafórico se gobierna por `cdi_geometric`. Añadir un selector discreto para alternar la vista entre las tres métricas.
- `tests/test_cdi.py` (fichero nuevo).

### Cambios en analizador.py
La función `calcular_cdi` se reescribe para devolver:

```python
{
  "cdi_geometric": float,         # 1 - |det(M)|^(1/n), principal
  "cdi_mean_dissent": float,      # 1 - mean(off_diagonal(M))
  "cdi_entropy": float,           # la entrópica ya calculada
  "cdi_det_raw": float,           # 1 - |det(M)|, solo retrocompatibilidad
  "n_modelos": int,
  "nivel": str,                   # sobre cdi_geometric
  "etiqueta": str,
  "color": str,
  "determinante": float
}
```

La media geométrica del determinante tiene una interpretación limpia: es la desviación promedio respecto a la unidad en el producto de los valores singulares de la matriz, elevada a uno partido por n. Está acotada en [0, 1] independientemente de n y converge al desacuerdo medio por par cuando la matriz es casi diagonal.

### Tests mínimos (tests/test_cdi.py)
- Matriz identidad n = 2, 4, 10: `cdi_geometric` debe ser 1.0 en los tres casos. Esto demuestra que la métrica normaliza correctamente.
- Matriz de unos n = 2, 4, 10: `cdi_geometric` debe ser 0.0 en los tres casos.
- Matriz diagonal con 0.5 fuera de la diagonal: `cdi_mean_dissent` debe ser 0.5, y `cdi_geometric` debe estar entre 0.3 y 0.7 (acotado y monótono en el disenso).
- Matriz con un valor NaN: la función devuelve `cdi_geometric = None` y un campo `error` explícito, sin lanzar excepción (conecta con la Issue 4).

### Criterios de aceptación
- Los tests pasan.
- El endpoint `/api/run-ensemble` devuelve las tres métricas en la clave `cdi` del `consenso_data`.
- El banner semafórico del frontend sigue funcionando para el visitante casual sin tocar nada.
- Un apartado en el README documenta las tres métricas y explica cuál se usa como principal y por qué.

---

## Issue 2 · Fix del SOLO canónico

### Prioridad
Crítica. Silencia justamente el fenómeno que el sistema está diseñado para detectar.

### Problema
En `analizador.py`, función `identificar_perspectiva_minoritaria`, línea 173:

```python
validos = [c for c in consensos_ind if c["consenso_individual"] > 0]
```

El consenso individual de un modelo es la media de sus cosenos con los demás. Si esa media es exactamente cero, el modelo es ortogonal al resto del ensemble, que es el caso estructuralmente más interesante: el SOLO canónico en la narrativa del paper. El filtro `> 0` lo descarta.

### Ficheros afectados
- `analizador.py`: función `identificar_perspectiva_minoritaria` (renombrar a `identificar_solo` por coherencia post-migración).
- `tests/test_solo.py` (fichero nuevo).

### Cambios
- Cambiar `> 0` por `>= 0`.
- Renombrar la función a `identificar_solo` y ajustar el único punto donde se llama (`dataAnalisis_interno`).
- Renombrar `perspectiva_minoritaria` a `solo` en todo el payload de salida.
- Actualizar el docstring para explicar por qué `consenso_individual == 0` sí se considera.

### Test mínimo (tests/test_solo.py)
- Construir un `consensos_ind` sintético con tres modelos: dos con consenso 0.9 y uno con consenso 0.0. Verificar que el tercero se expone como SOLO.
- Construir otro con tres modelos, dos con consenso 0.5 y uno con consenso 0.1. Verificar que el tercero se expone como SOLO.
- Caso degenerado: todos los consensos a cero. La función devuelve `None` sin lanzar excepción.

### Criterios de aceptación
- Los tests pasan.
- El payload del endpoint devuelve la clave `solo` en lugar de `perspectiva_minoritaria`.
- El frontend renderiza correctamente la tarjeta SOLO en los tres escenarios del test.

---

## Issue 3 · Trazabilidad del meta.json (schema v1.0)

### Prioridad
Crítica. Dos razones simultáneas: cualquier dato que vaya a un paper necesita reproducibilidad, y la demo pública entra en el perímetro RGPD si guarda texto clínico. El schema resuelve ambas cosas no guardando el prompt completo, solo un hash y un preview.

### Problema
El meta.json actual contiene timestamp sin zona horaria, nombres de modelo pero no versiones, no guarda las matrices de similitud (lo que impide recalcular el CDI con métricas alternativas sin volver a llamar a los LLMs), no guarda latencias, no hay hash del prompt, y no hay versión de schema. Además, el sistema no es determinista (temperature 0.2, sin seed) pero esa falta de determinismo no está documentada explícitamente en el propio registro.

### Decisión sobre el texto del prompt
**El meta.json NO contiene el texto completo del prompt del visitante.** Solo guarda `prompt_sha256` (para detectar que un mismo caso se ha ejecutado más de una vez) y `prompt_preview` (primeros 120 caracteres, para orientación humana). Esta decisión saca a la demo pública del perímetro RGPD de forma limpia. La web debe incluir un aviso explícito de este comportamiento en el formulario de entrada del caso.

### Ficheros afectados
- `schemas/meta_v1.py` (nuevo): define la estructura y una función `validar_meta(payload)` que lanza `ValueError` si falta algún campo obligatorio.
- `app.py`: `_guardar_meta` usa el schema y rellena todos los campos.
- `Ensambladores/ensamblador_LLM.py`: captura latencia por modelo y el string de versión que devuelve la API para cada respuesta.
- `static/index.html`: aviso claro bajo el campo del caso, del estilo *"El texto del caso no se guarda en el servidor. Solo se conservan metadatos del análisis."*

### Estructura del meta.json v1.0

```json
{
  "schema_version": "1.0",
  "case_uuid": "uuid4-generado-al-recibir-el-prompt",
  "case_reference_id": "null en casos ad hoc · rellenado si es caso de referencia",
  "session_code": "null salvo cuando un investigador lo introduzca (ver Issue 5)",
  "browser_token": "uuid4 de cookie anónima — permite filtrar historial por navegador",
  "timestamp_utc": "2026-05-18T10:32:11.234Z",
  "timestamp_local": "2026-05-18T12:32:11.234+02:00",
  "prompt_sha256": "hex64",
  "prompt_length_chars": 1432,
  "prompt_preview": "primeros 120 caracteres",
  "ensemble": {
    "n_modelos": 5,
    "model_type": "free|pay",
    "modelos": [
      {
        "name": "openai/gpt-4o",
        "provider_version": "string tal como lo devuelve OpenRouter",
        "latency_ms": 2340,
        "response_length_chars": 1820,
        "embedding_truncated": false,
        "api_error": null
      }
    ]
  },
  "determinismo": {
    "temperature": 0.2,
    "seed": null,
    "nota": "Output no determinista. Repeticiones del mismo prompt producen outputs distintos."
  },
  "fusion": {
    "modelo": "gpt-4o-mini",
    "latency_ms": 1820,
    "max_tokens": 1500,
    "temperature": 0.2
  },
  "matrices": {
    "tfidf": [[]],
    "embed": [[]],
    "principal": "embed"
  },
  "cdi": "objeto del Issue 1",
  "solo": "objeto del Issue 2",
  "divergencia_capas": {},
  "consenso_global": 0.74,
  "consensos_individuales": [],
  "respuesta_fusionada_sha256": "hex64",
  "chorus_version": "git commit hash del código en el momento de la ejecución"
}
```

### Notas de implementación
- `case_uuid` se genera con `uuid.uuid4()` en el endpoint `/api/run-ensemble` antes de lanzar `asyncio.run`.
- `prompt_sha256` se calcula con `hashlib.sha256(prompt.encode("utf-8")).hexdigest()`. Permite decir "este caso es el mismo que X" sin guardar el texto.
- `chorus_version` se obtiene con `subprocess.run(["git", "rev-parse", "HEAD"])`. Fallback a `"unknown"`.
- Las matrices se guardan como listas de listas. El `_serializar` existente ya convierte `np.ndarray` correctamente.
- Latencia por modelo: el ensamblador la mide con `time.perf_counter()` alrededor de cada llamada HTTP.

### Historial en el frontend (cookie anónima)
La ruta `/api/history` y el panel de historial existen actualmente y muestran análisis pasados. **Decisión**: el historial se vincula a una cookie anónima del navegador (`chorus_browser_token`, uuid4, httpOnly, sin dato personal asociado). Cada navegador ve solo sus propios análisis. El servidor guarda el token en el meta.json como `browser_token`. Esto evita que un visitante vea casos introducidos por otros, sin requerir login ni registro.

La implementación es trivial: al primer POST a `/api/run-ensemble` sin cookie, el servidor emite una. En los siguientes requests, la cookie viaja automáticamente. El endpoint `/api/history` filtra por `browser_token`. Si la cookie se borra, se pierde el historial de ese navegador, lo cual es consistente con el diseño sin registro.

### Criterios de aceptación
- `validar_meta(payload)` acepta un meta.json completo y rechaza uno con campos obligatorios ausentes.
- Un test de integración: ejecución con dos modelos mock, lectura del meta.json, todos los campos del schema presentes y tipados correctamente.
- El README documenta el schema, advierte que entre versiones los meta.json se migran explícitamente, y documenta el aviso RGPD que aparece en la web.
- El historial por cookie funciona: un segundo navegador en la misma máquina no ve los análisis del primero.

---

## Issue 4 · Saneamiento del CDI contra NaN

### Prioridad
Crítica para la demo pública. Una métrica que peta el frontend en condiciones límite no se puede presentar en abierto, donde cualquiera puede pegar cualquier cosa en el campo del caso.

### Problema
En `analizador.py`, `calcular_cdi`, la línea `np.linalg.det(M)` devuelve NaN si la matriz contiene NaN (por ejemplo, cuando una respuesta del ensemble es vacía o no procesable). `np.clip(nan, 0, 1)` también devuelve NaN, y NaN no es JSON estándar: el frontend falla al parsear.

### Objetivo
Detectar el caso de matriz inestable antes del cálculo y degradar con elegancia a un estado explícito que el frontend sabe manejar.

### Ficheros afectados
- `analizador.py`: `calcular_cdi`.
- `static/index.html`: el banner CDI maneja el estado de error sin romperse.
- `tests/test_cdi.py`: añadir casos de error.

### Cambios
Al inicio de `calcular_cdi`, después de validar n > 2:

```python
if not np.all(np.isfinite(matriz_sim)):
    return {
        "cdi_geometric": None,
        "cdi_mean_dissent": None,
        "cdi_entropy": None,
        "cdi_det_raw": None,
        "n_modelos": len(matriz_sim),
        "nivel": "indeterminado",
        "etiqueta": "Matriz de similitud inestable. Alguna respuesta del ensemble no es procesable.",
        "color": "#888888",
        "error": "matrix_contains_non_finite_values"
    }
```

En el frontend, el banner del CDI, cuando detecta `cdi_geometric == None`, muestra un estado neutro (gris, icono de advertencia, texto "CDI no computable en este caso") en lugar de intentar renderizar valores numéricos.

### Tests
- `calcular_cdi` con matriz que contiene NaN: devuelve el diccionario de error, no lanza excepción.
- `calcular_cdi` con matriz que contiene Inf: mismo comportamiento.
- Pipeline completo con una respuesta vacía en el ensemble: produce un meta.json válido con `cdi.error` rellenado.

---

## Issue 5 · Casos de referencia y session_code opcional

### Prioridad
Crítica para la componente longitudinal del programa. Es la única conexión operativa que CHORUS necesita con los estudios que se gestionan por fuera. Es una issue deliberadamente ligera: todo lo relacionado con cuestionarios, enrolamiento de terapeutas y consentimientos queda fuera del sistema.

### Contexto
El investigador ejecuta los estudios de validación por fuera, usando Google Forms o Qualtrics para administrar ConfAI a terapeutas enrolados. Durante las sesiones de recogida, el investigador le pide al terapeuta que analice ciertos casos con CHORUS. La única pieza que CHORUS necesita para que esos datos sean cruzables con los cuestionarios externos es un identificador compartido: el `session_code`.

Por otra parte, el programa necesita casos de referencia reutilizables. Un caso de referencia es un caso clínico concreto, consensuado, que se reejecuta periódicamente contra el ensemble. Sirve para medir la divergencia de Jensen Shannon entre distribuciones de outputs a lo largo del tiempo. Esa medida es la componente dinámica del constructo y es una de las propiedades que hacen a CHORUS único respecto a otros sistemas de IA clínica.

### Objetivo de la issue
Dos piezas muy pequeñas:
- Un campo `session_code` opcional en `/api/run-ensemble`, persistido en el meta.json.
- Un catálogo de casos de referencia (`casos_referencia.json`) con endpoint de listado y carga.

### Ficheros afectados
- `casos_referencia.json` (nuevo, raíz del repo): lista de casos de referencia.
- `app.py`: nuevo endpoint `GET /api/reference_cases`, extensión de `/api/run-ensemble` para aceptar `case_reference_id` y `session_code`.
- `static/index.html`: selector de caso de referencia (opcional) y campo `session_code` opcional en la interfaz avanzada.

### Estructura de casos_referencia.json

```json
{
  "schema_version": "1.0",
  "casos": [
    {
      "id": "REF-001",
      "titulo": "Depresión con ideación pasiva en paciente joven",
      "descripcion_corta": "Varón 28 años, episodio depresivo mayor, sin plan activo.",
      "texto_completo": "... texto anonimizado del caso ...",
      "ensemble_recomendado": ["openai/gpt-4o", "anthropic/claude-3.5-sonnet"],
      "notas_investigador": "caso semilla para medir D_JS semana a semana"
    }
  ]
}
```

### Endpoints
- `GET /api/reference_cases`: devuelve SOLO `id`, `titulo`, `descripcion_corta` y `ensemble_recomendado` de cada caso. **NO** devuelve `texto_completo` (el texto no se expone al navegador).
- `POST /api/run-ensemble`: acepta dos campos nuevos opcionales:
  - `case_reference_id` (si se pasa, el prompt se carga desde el JSON en el backend; el frontend no necesita ver el texto).
  - `session_code` (cadena libre que el investigador introduce; se persiste tal cual en el meta.json).
- Cuando se pasa `case_reference_id`, el meta.json rellena el campo correspondiente y el campo `prompt_sha256` sigue siendo del texto completo del caso (no del id), lo que permite detectar reejecuciones del mismo caso de forma consistente.

### Interfaz
En la web pública, los casos de referencia aparecen en un bloque visible del tipo **"Casos de ejemplo"** bajo el campo principal. El visitante puede pulsar en cualquiera para cargarlo y ejecutarlo sin tener que escribir un caso propio. Esto sirve para dos cosas simultáneamente: es la mejor forma de mostrar la herramienta a un visitante sin caso propio (como en medley.smile.ki.se), y es también el mecanismo por el que un terapeuta en sesión de recogida ejecuta los casos previstos por el investigador.

El campo `session_code` vive en una sección "Opciones avanzadas" o similar, no en la interfaz principal. Un visitante casual no lo ve. Un investigador que está recogiendo datos lo introduce manualmente (por ejemplo: `KARO-T0-P002` para el terapeuta 002 en el momento T0 del estudio de Karolinska).

### Cómo se cruzan los datos con los cuestionarios externos
Esta parte no es código, es protocolo de investigación. Se documenta aquí por coherencia pero no requiere implementación en CHORUS:
- El investigador enrola un terapeuta y le asigna un identificador de estudio (p. ej. `KARO-P002`).
- En cada sesión de recogida (T0, T1, T2), el investigador le pide al terapeuta que ejecute N casos de referencia introduciendo el `session_code` correspondiente (p. ej. `KARO-T0-P002`).
- Al final de la sesión, el terapeuta rellena el cuestionario ConfAI en Google Forms o equivalente. El primer campo del cuestionario es el mismo `session_code`.
- En el análisis, el investigador cruza los meta.json filtrados por `session_code` con las respuestas del cuestionario del mismo `session_code`. Sin base de datos común, sin sincronización automática, sin complejidad.

### Criterios de aceptación
- El endpoint `/api/reference_cases` devuelve la lista sin `texto_completo`.
- Ejecutar un caso con `case_reference_id` produce un meta.json con ese id rellenado y el `prompt_sha256` igual al sha del `texto_completo`.
- Ejecutar cualquier caso con `session_code` lo persiste en el meta.json.
- Los casos de referencia se pueden cargar y ejecutar desde la interfaz con un solo click.
- El README documenta qué es un caso de referencia y cómo usar `session_code` para estudios externos.

---

## Issue 6 · Fusión no bloqueante y guardado atómico

### Prioridad
Importante. En ensembles grandes el pipeline actual tiene riesgo de dejar meta.json corruptos o incompletos, lo que introduce ruido silencioso en el dataset.

### Problema
La fusión llamada desde `dataAnalisis_interno` usa `requests.post` síncrono, pero la cadena completa se llama desde `asyncio.run` en `app.py`. La llamada síncrona bloquea el event loop, y si hay timeouts o fallos durante la fusión, el proceso puede dejar un meta.json a medio escribir porque la escritura no es atómica.

### Ficheros afectados
- `analizador.py`: reescribir `_llamar_chatgpt` y `generar_fusion` con `aiohttp` (preferido) o envolver la llamada actual en `loop.run_in_executor`.
- `app.py`: `_guardar_meta` se reescribe con patrón de escritura atómica.

### Escritura atómica del meta.json

```python
def _guardar_meta(filename_base, payload):
    out_dir = Path(OUTPUT_PATH)
    out_dir.mkdir(exist_ok=True, parents=True)
    final_path = out_dir / f"{filename_base}.meta.json"
    tmp_path = out_dir / f"{filename_base}.meta.json.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=_serializar)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, final_path)
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
```

`os.replace` es atómico en POSIX y en Windows (cuando ambos paths están en el mismo volumen). Garantiza que el meta.json final nunca existe en estado parcial: o no existe, o está completo.

### Fusión no bloqueante
Dos opciones. La preferida es reescribir `_llamar_chatgpt` con `aiohttp` para que encaje con el resto del pipeline asíncrono. La alternativa, si se quiere tocar menos código, es envolver la llamada síncrona existente en `loop.run_in_executor(None, _llamar_chatgpt_sync, mensajes)` y hacer `generar_fusion` una función async. En ambos casos, la fusión pasa a no bloquear el event loop.

### Reintentos con backoff
Añadir reintentos exponenciales en la fusión y en las llamadas al ensamblador para errores 429 y 5xx. Tres intentos con espera de 1, 2 y 4 segundos es suficiente para absorber limitaciones de tasa sin degradar perceptiblemente la experiencia del visitante.

### Criterios de aceptación
- El pipeline completo pasa el test de integración incluso con un modelo configurado para devolver timeout intermitente.
- Un SIGKILL al proceso durante la fusión no deja meta.json corruptos (se verifica con un test que aborta el proceso a medio camino y comprueba el directorio de salida).

---

## Paquete de fixes menores

Después de cerrar las seis issues anteriores, aplicar estos fixes como un único pull request de limpieza.

- **Variables de entorno**: renombrar toda ocurrencia de `CRML_OUTPUT_PATH` y similares a `CHORUS_*`. Documentar en README.
- **analizador.py `MODELO_FUSION`**: mover a variable de entorno `CHORUS_FUSION_MODEL`, por defecto `gpt-4o-mini` (no `gpt-3.5-turbo`). El modelo que sintetiza no puede ser peor que los que sintetiza.
- **ensamblador_LLM.py `_es_error`**: sustituir la verificación frágil del prefijo por comprobación explícita del tipo de objeto devuelto por la API (si la respuesta decodificada es dict con clave `"error"`, entonces es error).
- **main.py**: aprovechar el reporte devuelto por `dataAnalisis` (actualmente se calcula y se descarta).
- **analizador.py embeddings truncados a 8000 caracteres**: añadir warning al log. El campo `embedding_truncated` por modelo ya está previsto en el schema del Issue 3.
- **ensamblador_LLM.py referer**: cambiar `HTTP-Referer` de `crml.local` a `chorus.innotep.upm`.
- **cargador_modelos.py**: eliminar print de debug dentro de la rama `"1"`.
- **app.py banner de arranque**: cambiar `"CRML v2 — Clinical Reasoning Multi-LLM"` por `"CHORUS — Clinical Heterogeneous Orchestration for Reasoning Under Supervision"`.
- **README.md**: actualizar completamente. Sigue diciendo CRML en el título, en la cabecera y en la mitad del texto. El repo está migrado, el README no.
- **Crear requirements.txt** con dependencias fijadas: `flask`, `flask-cors`, `scikit-learn`, `numpy`, `requests`, `aiohttp`, `openai`, `python-dotenv`.

---

## Nota de cierre

Lo que este documento deja fuera: CORS abierto, ausencia de rate limiting, falta de autenticación, purga automática del directorio `resultados`, separación de CSS y JS del HTML, migración de `print` a `logging`, accesibilidad de los botones de navegación, escape completo de HTML en `escHtml`. Son cuestiones legítimas de producción, no de demo científica. Cuando la web pública reciba tráfico relevante o cuando se plantee ofrecer CHORUS como herramienta institucional, tocará abordarlas. Ahora no.

**Orden propuesto para Claude Code**: issues 1 a 4 primero (integridad científica de la medida), después 5 (conexión con estudios externos), después 6 (robustez del pipeline), al final el paquete de fixes menores. Si Claude Code trabaja sobre ramas separadas por issue y abre pull requests pequeñas, la revisión será más fácil y los riesgos de regresión más bajos.

---

*CHORUS · InnoTep Research Group · UPM · Abril 2026*
