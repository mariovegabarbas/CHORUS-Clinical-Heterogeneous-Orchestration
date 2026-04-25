# Hallazgos del smoke test grande de CHORUS — Abril 2026

Documento vivo. Va capturando aprendizajes, sorpresas y decisiones que han ido saliendo durante el smoke test grande tras cerrar las seis issues iniciales y el cambio de embeddings. Sirve como fuente de verdad sobre el comportamiento real del sistema y como insumo para futuras issues.

Última actualización: 25 de abril de 2026.

## Hallazgos validados

### El catálogo de modelos en OpenRouter es volátil

Modelos que aparecían en el catálogo original de CHORUS (`anthropic/claude-3.5-sonnet`, `google/gemini-pro-1.5`) estaban retirados a fecha del smoke. OpenRouter los lista en `/v1/models` pero las llamadas devuelven `No endpoints found`. Implicaciones:

- **El listado de `/v1/models` no garantiza operatividad**. Hay que verificar con una llamada real, no solo con el listado.
- **Sustituidos por la familia "mediana" actual de cada laboratorio**: gpt-4o, claude-sonnet-4.5, gemini-2.5-pro (luego sustituido por gemini-2.5-flash, ver más abajo).
- **Pendiente de programa**: snapshot del catálogo al inicio de cada ronda del piloto, citable en el paper.

### El sistema trataba mensajes de error como respuestas válidas

Cuando un modelo retirado devolvía `error No endpoints found for ...`, CHORUS lo embebía, lo metía en la matriz de similitud y calculaba CDI sobre datos basura. La función `_es_error` no detectaba estos casos. Reportaba "Respuestas válidas: 3/3" siendo en realidad "1/3 + 2 errores".

Pendiente de issue: detección robusta de respuestas-error y exclusión del ensemble (heredado del informe inicial de Claude Code, ahora elevado a prioritario).

### Modelos de razonamiento extendido (thinking) rompen el CDI

`gemini-2.5-pro` resultó ser un modelo thinking. Con `max_tokens: 2048` consumía 1965 tokens en razonamiento interno y devolvía respuestas truncadas de 79 tokens (≈400 chars). Esto producía dos problemas:

- **Respuestas sistemáticamente más cortas que los demás modelos**, lo que sesgaba la matriz de similitud.
- **Aparición artificial como SOLO** por longitud, no por contenido.

Verificación de la familia entera: de los 8 modelos de pago canónicos, solo `gemini-2.5-pro` es thinking. Los siete restantes (gpt-4o-mini, gpt-4o, gpt-4.1, claude-3.5-haiku, claude-sonnet-4.5, claude-opus-4.5, gemini-2.5-flash) son no-thinking.

**Decisión adoptada**: catálogo de pago homogéneamente no-thinking. Sustitución de `gemini-2.5-pro` por `gemini-2.5-flash` en `modelos.json`, `ensemble_recomendado` y `ensemble_extendido` de los tres casos de referencia.

Pendiente para futuro paper: comparativa "CHORUS sobre modelos no-thinking vs CHORUS sobre modelos thinking" como estudio aparte. Incluiría una métrica de similitud sobre el reasoning además del content.

### Las longitudes de respuesta son heterogéneas entre laboratorios incluso a igualdad de prompt

Sobre el mismo prompt corto, las longitudes visibles varían así (caracteres):

- openai/gpt-4o-mini: 1735
- openai/gpt-4o: 1549
- openai/gpt-4.1: 844
- anthropic/claude-3.5-haiku: 790
- anthropic/claude-sonnet-4.5: 996
- anthropic/claude-opus-4.5: 713
- google/gemini-2.5-flash: 1719
- openai/gpt-oss-120b:free: 5855

Esto indica que **parte del CDI medido actualmente es disenso de longitud, no de contenido**. Un system prompt común que estructure la salida (Issue 7 hipotética) reduciría esta variabilidad.

### Embeddings: el fallback a TF-IDF es silencioso

Durante el smoke test inicial, los embeddings de OpenAI fallaron de forma transitoria y CHORUS cayó a TF-IDF léxico sin avisar al usuario. El meta.json sí lo registra (`embeddings.fallback_aplicado: true`, `matrices.principal: tfidf`), pero el frontend no lo señala visualmente.

Pendiente de mejora: indicador visual claro en la web cuando se aplica fallback (sema en gris al lado del banner

## Actualización 25 abril 2026 — Auditoría de catálogo y resolución de modelos thinking

### Auditoría del catálogo gratis: volatilidad real

En el primer barrido del catálogo gratis (smoke test), de los 5 modelos originales 3 dieron error en una sola sesión. Se hizo una auditoría más profunda con 12 candidatos adicionales: solo 4 respondieron limpiamente, y de esos 4 dos eran thinking. La conclusión:

- **El listado `/v1/models` de OpenRouter no garantiza operatividad de los modelos gratis**. Los providers gratis tienen instancias volátiles, rate limits agresivos y caídas frecuentes.
- **Solo 2 modelos gratis no-thinking respondieron en el momento de la auditoría**: `openai/gpt-oss-120b:free` y `google/gemma-4-26b-a4b-it:free`.
- El catálogo final (4 modelos) incluye también `meta-llama/llama-3.3-70b-instruct:free` y `qwen/qwen3-next-80b-a3b-instruct:free`. Aunque estaban caídos en el momento de la auditoría, históricamente son las opciones más estables a medio plazo. Se asume que la disponibilidad fluctuará.

Implicación operativa: la web pública debe asumir que en cualquier momento dado, una porción de los 4 modelos gratis puede estar caída. La gestión robusta vendrá con Issue 7 (detección de errores). Mientras tanto, un visitante que use modo gratis puede ver que solo 2 de 4 modelos responden — eso es esperable, no un fallo del sistema.

### Solución a la simetría del catálogo PAY: gemini-2.5-flash-lite

Eliminar `gemini-2.5-pro` por ser thinking dejaba el catálogo PAY asimétrico (3 OpenAI + 3 Anthropic + 1 Google). Se descubrió que `google/gemini-2.5-flash-lite` está vivo en OpenRouter y es no-thinking. Se incorporó como talla Small de Google, dejando `gemini-2.5-flash` como Mediano. El catálogo PAY queda así con 8 modelos en simetría perfecta:

| Talla | OpenAI | Anthropic | Google |
|---|---|---|---|
| Small | gpt-4o-mini | claude-3.5-haiku | gemini-2.5-flash-lite |
| Medium | gpt-4o | claude-sonnet-4.5 | gemini-2.5-flash |
| Premium | gpt-4.1 | claude-opus-4.5 | (sin equivalente estable no-thinking) |

Google no tiene actualmente un modelo Premium no-thinking estable equivalente a gpt-4.1 o claude-opus-4.5. Se acepta la asimetría y se documenta.

### Heterogeneidad de longitud entre modelos no-thinking

Aun entre modelos no-thinking, las longitudes de respuesta para el mismo prompt son notablemente distintas. Datos del primer smoke test sobre REF-001:

- openai/gpt-4o: 2256 chars
- anthropic/claude-sonnet-4.5: 3890 chars
- google/gemini-2.5-pro (thinking, truncado): 375 chars

En el test exploratorio sobre prompt corto:

- openai/gpt-4o-mini: 1735 chars
- openai/gpt-4o: 1549 chars
- openai/gpt-4.1: 844 chars
- anthropic/claude-3.5-haiku: 790 chars
- anthropic/claude-sonnet-4.5: 996 chars
- anthropic/claude-opus-4.5: 713 chars
- google/gemini-2.5-flash: 1719 chars
- openai/gpt-oss-120b:free: 5855 chars

Patrón observado: **OpenAI tiende a respuestas más largas, Anthropic más concisas, Google variable, modelos OSS pueden ser muy verbosos**. Esto se traduce en que **una parte significativa del CDI medido actualmente es disenso de longitud y formato, no de contenido clínico**. La Issue 8 (system prompts del ensemble) es el principal mecanismo previsto para reducir este sesgo.

### Decisión metodológica para el paper

El paper de CHORUS debe declarar explícitamente:

1. La fecha exacta del piloto y los IDs de modelos usados, con sus versiones según OpenRouter.
2. La política "ensemble homogéneamente no-thinking" como decisión de diseño.
3. La existencia del system prompt común (cuando esté implementado) y su versión.
4. La aceptación de heterogeneidad residual de longitud entre laboratorios como limitación.

Pendiente en el backlog: Issue 10 (snapshot de catálogo) cubre los puntos 1 y 4. Issue 8 (system prompts versionados) cubre el punto 3.

## Bug detectado el 25 abril 2026 — Fusión clínica no se ejecutaba en ensembles de 3 modelos

### Síntoma

Durante el smoke test grande tras la sustitución de gemini-2.5-pro por gemini-2.5-flash, las ejecuciones de REF-001 producían un meta.json con la sección `fusion` enteramente en None:

```json
"fusion": {"modelo": null, "latency_ms": null, "max_tokens": null, "temperature": null}
```

Y el campo `respuesta_fusionada_sha256` también `None`. La web mostraba el mensaje "Se necesitan al menos 3 respuestas válidas para generar la síntesis clínica" pese a que el ensemble había producido 3 respuestas válidas.

### Causa raíz

En `analizador.py`, el cálculo del subconjunto de modelos que entra a fusión filtraba con la heurística "los 2/3 mejores":

```python
n_top = max(1, len(consensos_ind) * 2 // 3)
mayores = consensos_ind[:n_top]
```

Con `n_modelos = 3`, esto da `n_top = 2` y `mayores` con 2 elementos. La condición posterior `if len(mayores) >= 3` impedía la llamada a `generar_fusion`. Resultado: para el caso por defecto del programa (ensemble_recomendado de 3 modelos), la fusión no se ejecutaba **nunca**.

El bug era invisible en pruebas porque:

- Los smoke tests previos también daban `fusion: None` y se interpretó como secundario.
- Los tests unitarios cubrían `_guardar_meta` y `_llamar_chatgpt` pero no la **disparación** de la fusión a partir de un ensemble real.
- El frontend tenía además un mensaje de error con umbral 3 (`"Se necesitan al menos 3 respuestas válidas"`) que reforzaba la idea de "es esperable", aunque el backend en realidad usaba un umbral de 2 en otras partes (`return {"error": "Se necesitan al menos 2 respuestas"}` en línea 416).

### Resolución

- **`analizador.py`**: lógica de selección rediseñada. Para ensembles pequeños (≤ 4 modelos) se usan todos sin filtro. Para ensembles mayores se mantiene el filtro 2/3 con un mínimo de 3. La fusión se dispara con ≥ 2 respuestas, consistente con el resto del módulo.
- **`static/index.html`**: mensaje del frontend reformulado a "No se ha generado síntesis clínica. Esto puede ocurrir si la fusión con el modelo síntesis ha fallado." Ya no menciona un umbral incorrecto.
- **Tests**: añadidos casos para ensemble de 3 modelos (debe fusionar) y de 1 modelo (no debe fusionar).

### Lección

Dos lecciones acumuladas:

Una. **Los criterios de "respuestas válidas" estaban duplicados en backend y frontend con valores distintos** (frontend exigía 3, backend toleraba 2). Cuando un valor está duplicado en dos sitios, la divergencia silenciosa es solo cuestión de tiempo. Anotar como ítem para una limpieza futura: que la condición de fusión la decida solo el backend y que el frontend simplemente reporte qué encontró en el JSON.

Dos. **El smoke test técnico (Issue 6) verificó que el meta.json se guardaba bien, pero no verificó que el contenido del meta.json fuera completo**. La fusión vacía pasó el filtro porque "técnicamente el JSON está bien formado". Para futuros smoke tests, conviene tener un script que valide no solo el schema sino la **completitud semántica** del payload: si un ensemble de 3 modelos válidos no produce fusión, eso es un fallo aunque el meta sea sintácticamente correcto. Posible Issue 13 hipotética: validador de completitud semántica del meta.json.

## Cierre del smoke test grande — 25 abril 2026

El smoke test grande se completó tras dos iteraciones (la primera contaminada por modelos retirados, gemini-2.5-pro thinking, y bug de fusión). El sistema, en su estado final, responde correctamente a los ocho criterios de validación previstos. A continuación se resumen las observaciones que el smoke ha dejado sobre el comportamiento real del sistema, que conviene tener presentes para Issues 7-12 y para el piloto.

### Lo que el smoke confirma

- **Pipeline técnico estable**: tres casos de referencia ejecutados sin errores, tres modelos respondiendo, embeddings activos, fusión generada, meta.json completo y atómico.
- **Aislamiento de sesiones**: la cookie `chorus_browser_token` aísla historiales por navegador. Sin fugas entre sesiones.
- **Trazabilidad longitudinal**: el `prompt_sha256` permanece estable entre reejecuciones del mismo caso de referencia, independientemente del catálogo de modelos, del estado de embeddings, o de cambios estructurales del sistema. Verificado con seis ejecuciones de REF-001 a lo largo de varias horas y dos catálogos distintos. Todas dieron el mismo hash. Es la propiedad básica que necesita la medida de D_JS longitudinal.

### Patrón estilístico inter-laboratorio recurrente

Las matrices de similitud de los tres casos muestran un patrón que no varía con la complejidad clínica:

```
Caso       gpt-4o↔claude   gpt-4o↔gemini   claude↔gemini
REF-001    0.451           0.780           0.480
REF-002    0.534           0.796           0.623
REF-003    0.570           0.802           0.598
```

En los tres casos, gpt-4o y gemini-flash están sistemáticamente más cerca entre sí que cualquiera de ellos con claude-sonnet-4.5. La variación caso a caso es pequeña (≤0.15 en cualquier celda) comparada con la consistencia del patrón. Esto sugiere que el CDI medido actualmente captura disenso estilístico/formal entre laboratorios además del disenso clínico específico al caso.

**Implicación para el programa**: Issue 8 (system prompts del ensemble) deja de ser opcional. Sin un system prompt común que homogeneice formato y estructura, los datos del piloto mezclan dos fuentes de variación que el paper no podrá separar.

### Métricas del CDI: posibles redundancias

Los tres casos del smoke producen los siguientes valores:

```
Caso       cdi_geometric   cdi_mean_dissent   cdi_entropy
REF-001    0.405           0.158              1.792
REF-002    0.361           0.136              1.791
REF-003    0.385           0.145              1.792
```

Dos observaciones:

- `cdi_geometric` y `cdi_mean_dissent` están perfectamente correlacionados en estos casos (mismo orden, escalado lineal). Una de las dos puede ser redundante.
- `cdi_entropy` sale idéntica a tres decimales en los tres casos. La métrica posiblemente está saturada por la normalización por suma. Hay que revisarla.

**Implicación**: tras Issue 8, conviene reevaluar el set de métricas. Es posible que con un system prompt común el rango de variación sea suficiente para que las métricas se diferencien, o puede ser señal de que la familia entera necesita repensarse. La decisión depende de los datos post-Issue 8.

### El sistema no discriminó complejidad clínica como predijimos

El programa diseñó los tres casos con expectativas de CDI ascendente: REF-001 (cuadro clásico) < REF-002 (ambigüedad de manual) < REF-003 (caso traumático complejo). El resultado real es:

```
REF-002: 0.361 (bajo)
REF-003: 0.385 (intermedio)
REF-001: 0.405 (alto)
```

El caso prototípico produce el CDI más alto. El rango total es estrecho (0.04). Combinado con el patrón estilístico recurrente, refuerza la hipótesis de que el CDI actual mide más estilo que contenido.

No es un fallo del sistema. Es información sobre dónde estamos antes de Issue 8.

### Truncamiento de embeddings: gemini-flash es verboso

Gemini-flash produce respuestas notablemente más largas que los otros dos modelos del ensemble (entre 6800 y 9000 chars en los tres casos). En REF-002 superó el límite actual de 8000 chars del embedding, y la respuesta se truncó. El campo `embedding_truncated: true` en el meta.json registra la incidencia.

Mejora prevista: subir `EMBEDDING_MAX_CHARS` a 16000 (o configurable por env var) y mostrar el flag de truncamiento en la UI cuando ocurra. Va al backlog.

### Hallazgo trivial pero útil

python-dotenv no encuentra el `.env` automáticamente cuando se ejecuta dentro de un heredoc bash. Workaround: pasar `dotenv_path=".env"` explícito.

### Estado del sistema al cierre del smoke

- 6 issues iniciales mergeadas.
- PR de embeddings configurables mergeado.
- PR de catálogo limpio (catalog-fix-thinking-and-free) mergeado.
- PR de fusión en ensembles pequeños (fusion-fix-small-ensembles) mergeado.
- 65 tests verdes.

El sistema está estable y listo para abordar las Issues 7 (detección de errores) y 8 (system prompts versionados). Ambas se identifican durante el smoke como necesarias antes del piloto.
