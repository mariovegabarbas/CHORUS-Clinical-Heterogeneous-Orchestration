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
