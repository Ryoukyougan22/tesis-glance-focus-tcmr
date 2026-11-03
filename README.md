# Inferencia temporal y análisis causal intervencional en modelos multimodales de Video Question Answering

Código y evidencia experimental de una tesis de Ingeniería Informática de la Pontificia Universidad Católica del Perú. Se estudia cómo refinar las memorias internas de un modelo VideoQA y medir su influencia sobre la respuesta, conservando una configuración identificable entre evaluación predictiva, intervenciones y asociación visual.

## Modelo y objetivos

Glance-Focus genera diez memorias mediante **glancing visual**. TCMR construye relaciones SELF, BEFORE, AFTER y OVERLAP a partir de los spans predichos y refina las memorias mediante atención relacional y una conexión residual controlada. **Focusing** combina video, memorias y pregunta/alternativas para responder. Las features S3D están precomputadas; el prototipo no extrae features del MP4.

| Objetivo | Procedimiento | Evidencia final |
|---|---|---|
| O1 | Evaluación pareada GF frente a GF+TCMR, STAR validation, 7098 preguntas y batch_size=1; McNemar exacto | `results/gf_vs_tcmr_bs1/` |
| O2 | Diez intervenciones independientes sobre memorias refinadas; TOP/BOTTOM por abs(Δp), RANDOM con semilla 42; bootstrap y McNemar | `results/causal_importance_bs1/`, `results/final_statistics_bs1/` |
| O3 | Ventanas de tres tokens visuales anuladas; sensibilidad coseno y selectividad Z por token; diez casos, 100 memorias | `results/memory_visual_sensitivity_final/` |

En O1 se obtuvieron **3878/7098 aciertos (54,64 %)** para GF y **3963/7098 (55,83 %)** para GF+TCMR: 264 correcciones, 179 pérdidas y McNemar p≈6,27×10⁻⁵. En O2 el baseline es el mismo GF+TCMR; sus 7098 distribuciones basales coincidieron exactamente entre las ejecuciones de O1 y O2. Las predicciones, efectos seleccionados y estadísticas conservadas se reprodujeron con los hashes indicados abajo.

La serie visual final usa estos mismos pesos. Es **una ejecución nueva**, diferente de la serie que originó la Tabla 17 y el Anexo K de la tesis: correlación media 0,956143 y concentración top-5 media 14,2307 %, con 85 máximos Z positivos. [Resultados y alcance de cada serie](docs/RESULTS.md).

## Configuración reproducible

| Recurso externo | SHA-256 requerido |
|---|---|
| GF `checkpoints/star/model_3000.tar` | `74e3c2064d61ee4762c2d1cac8853017124a3c47df3e351ae7a3dd066c765326` |
| TCMR `checkpoints/tcmr/temporal_relation_encoder_historical.pt` | `8a30ff3e331fe40a121a0b6e3f6c13182fd3216caee5664fcf124336ad569eb9` |
| STAR `txt_db/val.jsonl` | `99dd65080a3d50eda8c875e7a10bdd5567b6654839074d7d189f64c3de5a3528` |
| STAR `vis_db/s3d.pth` | `7a4e6a6a4a12caed1a91ff2130e857a9d28fa396361f6dce1857834875846ecb` |

La ubicación local del modelo se configura en [configs/final_experiment.json](configs/final_experiment.json) o mediante `VIDEOQA_GF_CHECKPOINT` y `VIDEOQA_TCMR_CHECKPOINT`. El backend comprueba los hashes para impedir mezclar pesos y asociaciones visuales.

**Los pesos no están incluidos.** El checkpoint TCMR exacto debe solicitarse al responsable de la investigación: no se dispone de una URL pública verificada para ese artefacto. Entrenarlo de nuevo no garantiza obtener los mismos bytes o métricas. Los argumentos y pasos completos de su entrenamiento original no se recuperaron; sí se verificó la reproducción de la inferencia y las intervenciones con el artefacto identificado.

## Instalación y recursos

Entorno de referencia: Python 3.7.16, PyTorch 1.11.0+cu113, NumPy 1.21.6, transformers 4.30.2, CUDA 11.3 y RTX3060. [Versiones y recursos](docs/ENVIRONMENT.md).

```text
python -m pip install -r requirements-reproduction.txt
python -m pip install -r webapp/requirements.txt
cd webapp/frontend
npm ci
```

Instalar PyTorch/CUDA de forma compatible con el equipo antes de ejecutar el modelo. Los manifiestos reproducen las dependencias; no se afirma haber verificado una reinstalación completa de ese entorno antiguo. Las comprobaciones sin pesos son independientes de CUDA.

Obtener STAR y features desde sus fuentes oficiales, conservar los términos aplicables y disponer de RoBERTa/tokenizador en caché. Para el prototipo, preparar los diez MP4 en `../demo_videos/` y sus features oficiales en `../demo_features/`, conforme a `results/demo_library/demo_cases_10_prepared.json`. Las rutas son relativas a la raíz. [STAR](https://github.com/csbobby/STAR_Benchmark) · [Glance-Focus y recursos originales](https://github.com/ByZ0e/Glance-Focus).

Los JPEG no se distribuyen. Tras disponer de los MP4 autorizados:

```text
python scripts/prepare_demo_frames.py
```

Este comando reconstruye solo imágenes faltantes desde los tiempos/selectividad registrados; no recalcula métricas ni sobrescribe frames existentes.

## Prototipo web

Desde la raíz, con el entorno del modelo activo:

```text
python -m uvicorn webapp.backend.main:app --host 127.0.0.1 --port 8000 --workers 1
```

En otra terminal:

```text
cd webapp/frontend
npm run dev
```

Abrir `http://127.0.0.1:5173`. Seleccionar un caso y pulsar una vez **Ejecutar Glance-Focus + TCMR** calcula la condición original y las diez intervenciones. Elegir E01–E10 muestra sus efectos, predicciones, sensibilidad/selectividad, tiempos y frames de la **misma configuración final**. Cambiar de caso limpia el estado. Las asociaciones son precomputadas; las predicciones e intervenciones son inferencias reales.

## Reproducir las evaluaciones

Ejecutar desde la raíz, sustituir `data/STAR` por el dataset local y usar siempre directorios de salida nuevos:

```text
python compare_gf_tcmr_bs1.py --base_data_dir data/STAR --reload_model_path checkpoints/star/model_3000.tar --temporal_checkpoint checkpoints/tcmr/temporal_relation_encoder_historical.pt --batch_size 1 --seed 42 --output_dir results/new_o1

python evaluate_causal_importance.py --base_data_dir data/STAR --reload_model_path checkpoints/star/model_3000.tar --temporal_checkpoint checkpoints/tcmr/temporal_relation_encoder_historical.pt --batch_size 1 --seed 42 --max_batches -1 --output_dir results/new_o2

python scripts/analysis/analyze_final_statistics.py --input_csv results/new_o2/causal_importance_per_question.csv --output_dir results/new_statistics --bootstrap 5000 --seed 42

python run_visual_analysis_final_10.py --output-root results/new_visual --temporal-checkpoint checkpoints/tcmr/temporal_relation_encoder_historical.pt
```

El análisis visual conserva la identidad de queries del baseline, recalcula spans perturbados y valida matrices desde vectores refinados guardados. El entrenamiento TCMR está en `train_temporal_refinement.py`; para una investigación nueva usar una carpeta de pesos nueva y registrar todos los argumentos. Sus defaults incluyen un límite de pasos para pruebas: `--max_steps -1` elimina ese límite. [Guía de scripts](docs/SCRIPTS.md).

## Pruebas

Sin videos, features ni checkpoints:

```text
python -m unittest discover -s tests -v
```

Con recursos y backend disponibles, definir primero `VIDEOQA_VALIDATION_OUTPUT` como una ruta de archivo nueva (por ejemplo, `../../results/new_browser_validation.json`, relativa a `webapp/frontend`). La evidencia publicada ya existe y no se sobrescribe:

```text
python webapp/tests/validate_final_configuration.py --base-url http://127.0.0.1:8000 --output results/new_web_validation
cd webapp/frontend
npm run typecheck
npm run build
npx playwright test tests/final-configuration.spec.ts tests/localization.spec.ts tests/real-model.spec.js
```

Playwright usa Microsoft Edge. Elegir un archivo nuevo con `VIDEOQA_VALIDATION_OUTPUT` para conservar su evidencia de navegador. El frontend admite `VIDEOQA_BACKEND` y las pruebas `VIDEOQA_FRONTEND`; las variables no se cargan automáticamente desde `.env`.

## Estructura

```text
model/, dataset/, utils/       Glance-Focus, TCMR y carga de datos
train_*.py, compare_*.py       Entrenamiento y evaluación
evaluate_causal_*.py           Intervenciones
inspect_memory_*.py            Sensibilidad de memorias
scripts/analysis/              Estadística y consolidación
configs/                       Identidad de la configuración final
results/                       Resultados, matrices y manifiestos científicos
webapp/                        FastAPI, React/Vite y validación funcional
tests/                         Comprobaciones sin recursos externos
docs/                          Entorno, resultados y procedimientos
```

## Interpretación y redistribución

`Δp = p(y_original) − p(y_original | do(Ei=0))`. Las intervenciones describen influencia interna del modelo, no causalidad física. Sensibilidad y selectividad son relativas: no asignan un significado semántico exclusivo a una memoria y los tiempos no son anotaciones temporales de verdad terreno.

La evaluación usa validación STAR; no equivale a una prueba independiente de generalización. El bootstrap remuestrea preguntas y no modela dependencia entre preguntas del mismo video. TOP/BOTTOM se seleccionan usando el efecto observado; las significancias por categoría son nominales. No significancia no demuestra equivalencia.

Se preserva [LICENSE](LICENSE) del código Glance-Focus y [NOTICE](NOTICE) con las atribuciones y modificaciones de tesis. Los extractos STAR mantienen su [licencia Apache 2.0](licenses/STAR-APACHE-2.0.txt). Esto no acredita permisos de redistribución de videos Charades/Action Genome, frames, features o pesos: esos recursos permanecen externos. Sin ellos, una clonación permite examinar resultados y ejecutar las pruebas estáticas, pero no ejecutar inmediatamente la demostración completa.
