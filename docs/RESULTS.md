# Resultados científicos y trazabilidad

La configuración final se identifica por GF SHA-256 `74e3c2064d61ee4762c2d1cac8853017124a3c47df3e351ae7a3dd066c765326` y TCMR SHA-256 `8a30ff3e331fe40a121a0b6e3f6c13182fd3216caee5664fcf124336ad569eb9`. Las rutas no sustituyen esa identidad de contenido.

## O1: comparación pareada

`results/gf_vs_tcmr_bs1/gf_vs_tcmr_per_question.csv` conserva 7098 IDs, etiquetas, categorías y predicciones GF/TCMR. Su resumen contiene 3878 y 3963 aciertos respectivamente, 264 correcciones y 179 pérdidas; McNemar exacto global p≈0,0000627256. La reproducción completa obtuvo los mismos CSV byte a byte. Las nuevas probabilidades y logits de ambos modelos están en `results/reproducibility/o1_probabilities.jsonl`.

## O2: intervenciones sobre el mismo TCMR

| Condición | Aciertos | Accuracy | Flip rate | Media abs(Δp) |
|---|---:|---:|---:|---:|
| Original | 3963 | 55,832629 % | — | — |
| TOP | 3974 | 55,987602 % | 2,409129 % | 0,019911406 |
| RANDOM | 3990 | 56,213018 % | 1,944210 % | 0,015193809 |
| BOTTOM | 3977 | 56,029868 % | 1,535644 % | 0,010968645 |

Fuentes: `results/causal_importance_bs1/` y `results/final_statistics_bs1/`. Se reprodujeron las 7098 preguntas y 70 980 intervenciones; los CSV principales y los dos CSV estadísticos coincidieron byte a byte con la serie histórica. `results/reproducibility/o2_all_interventions.jsonl` conserva las diez distribuciones y efectos por pregunta. Sus 7098 distribuciones basales nuevas coinciden exactamente con O1-TCMR (28 392 probabilidades).

Los intervalos bootstrap usan 5000 remuestreos y semilla 42. Original frente a TOP, BOTTOM y RANDOM tiene p≈0,367297; 0,145635 y **0,0100633**, respectivamente. RANDOM muestra una diferencia nominal de accuracy; esta conclusión no debe reemplazarse por la ausencia de significancia de la serie posterior O2-R.

La serie utilizada en el capítulo 5 del PDF existente tenía baseline 54,64 %, con otro estado TCMR (`fc7aa1889f0014029039ed1f9e153131028dc8b397b8cf3f9498c3262213a9f4`). Sus cuatro CSV se conservan en `results/reference_intervention_posterior/` para contrastar las tablas del documento, **no como resultado final de esta configuración**. El caso del Anexo G se conserva en `results/reference_annex_g/all_memory_interventions_bs1.csv`; pertenece a esa serie posterior. La Tabla I3 del PDF reutilizó valores p de flip para accuracy: los CSV estadísticos de referencia permiten recalcular los p correctos (0,701811; 0,165441; 0,362032 para TOP, RANDOM y BOTTOM). No combinar esta serie con O2-H ni sus conclusiones.

## O3: asociación visual final

`results/memory_visual_sensitivity_final/` utiliza los pesos finales O1/O2, glancing y TCMR después de cada perturbación, ventana de tres tokens (recortada en extremos) y alineación por identidad de query del baseline. Sensibilidad: `1 − cosine_similarity`; selectividad: Z entre las diez memorias por token, con desviación poblacional y mínimo numérico 1e-12. Las copias por alternativa se promedian después de TCMR.

Se validaron diez casos y 100 memorias, matrices finitas, vectores basales/perturbados, tiempos aproximados y frames locales. Los cálculos S/L2/Z se contrastaron con los vectores guardados. La alineación se comprobó con el criterio original `rtol=1e-5`, `atol=1e-5`, sin ampliar tolerancias.

| Métrica | Serie anterior de Tabla 17/Anexo K | Serie final, mismos pesos O1/O2 |
|---|---:|---:|
| Correlación media | 0,977858 | 0,956143 |
| Rango de correlaciones intracasos | 0,872350–0,998568 | 0,228046–0,998532 |
| Concentración top-5 media | 14,193423 % | 14,230743 % |
| Máximo Z | 2,898092 | 2,901326 |
| Memorias con máximo Z positivo | 82/100 | 85/100 |

El máximo final corresponde a DQEC3/E08, token 25, ≈9,351475 s. Cambió el token de máxima selectividad de 14/100 memorias respecto de la serie TCMR anterior. Son cifras nuevas, **no las cifras originales de la tesis**. `comparison_with_previous_series.{csv,json}` registra las diferencias. Las 100 filas históricas publicadas como referencia están en `results/reference_visual_tcmr/`; no alimentan el prototipo.

La media global de correlación es la media no ponderada de las medias por caso; su rango recoge extremos intracasos. La concentración se calcula por memoria. No confundir correlación de perfiles, sensibilidad y selectividad relativa.

El cálculo visual promedia las copias de alternativas después de TCMR; el análisis intervencional selecciona la primera copia y la expande. Las copias fueron idénticas en los diez casos finales. La frase heredada `Mean after TCMR, as in the interventional analysis` de los metadatos describe imprecisamente esa segunda operación; se conserva como registro, con esta aclaración. La alineación mantiene la tolerancia original `atol=1e-5, rtol=1e-5`; el error absoluto máximo, cercano a `1,08e-5`, pasa la comparación combinada y no debe confundirse con un criterio exclusivamente absoluto.

## Prototipo y validación

El backend verifica los hashes de pesos frente a `configs/final_experiment.json`; el adaptador visual exige la etapa `tcmr_refined` y los mismos hashes. No se combinan VIS-PRE y la nueva serie. La inferencia y las diez intervenciones son en vivo; el análisis visual es precomputado con esa configuración.

La validación nueva del backend recorrió diez casos, 100 intervenciones y 300 JPEG comparados byte a byte con archivos locales. Playwright comprobó la selección de las 100 memorias, frames decodificados, navegación temporal y limpieza al cambiar de caso. Evidencia: `results/prototype_final_validation/`. Las pruebas no usan mocks de inferencia. El cumplimiento funcional no equivale a un estimador de rendimiento sobre datos no seleccionados.

## Procedencia y límites

`results/reproducibility/manifest.json` vincula los resultados seleccionados con sus hashes, configuración y salidas de reproducción, sin incluir registros internos. `configs/final_experiment.json` define la configuración activa. Los CSV/JSON numéricos se conservan; al preparar la distribución solo se normalizan rutas de metadatos, con hashes de origen/destino y verificación de igualdad numérica.

No se recuperaron los argumentos y pasos totales del entrenamiento original ni todos los intentos de selección. La evaluación es STAR validation, con checkpoint seleccionado mediante validación; no constituye una prueba independiente de generalización. Los p por categoría son nominales, el bootstrap remuestrea preguntas y TOP/BOTTOM se eligen por el efecto observado. No se demuestra causalidad física, significado semántico exclusivo de memorias ni exactitud temporal de los spans.
