# Procedimientos principales

Ejecutar desde la raíz con salidas nuevas. Los resultados de referencia no se sobrescriben.

| Procedimiento | Fuente | Salida / función |
|---|---|---|
| Entrenar GF supervisado/no supervisado | `train_glance_focus_sup.py`, `train_glance_focus_uns.py` | Código original; no repetido como parte del cierre |
| Entrenar TCMR con GF congelado | `train_temporal_refinement.py` | Nuevo checkpoint y validación. Registrar argumentos; no reconstruye automáticamente el entrenamiento original |
| O1 pareado batch1 | `compare_gf_tcmr_bs1.py` | Clases por pregunta, accuracy y McNemar |
| O2 completo | `evaluate_causal_importance.py --max_batches -1` | Diez intervenciones por pregunta y exportación TOP/BOTTOM/RANDOM |
| Estadística O2 | `scripts/analysis/analyze_final_statistics.py` | Bootstrap, IC, diferencias pareadas y McNemar; usar 5000 remuestreos y seed42 |
| Serie visual final | `run_visual_analysis_final_10.py` | Diez casos/100 memorias, validación de matrices y consolidación; rechaza salida existente |
| Sensibilidad post-TCMR | `inspect_memory_visual_sensitivity_tcmr.py` | Perturbaciones, vectores y matrices S/L2; helpers en `complete_visual_analysis_10.py` |
| Selectividad y frames | `analyze_memory_visual_selectivity.py`, `export_selective_memory_frames.py` | Z por token e imágenes locales |
| Consolidar visual | `scripts/analysis/consolidate_visual_analysis.py` | Resúmenes por caso/memoria; rechaza sobrescritura |
| Reconstruir imágenes faltantes | `scripts/prepare_demo_frames.py` | Usa métricas existentes y MP4 locales; sin recalcular efectos |
| Validar backend final | `webapp/tests/validate_final_configuration.py` | Diez casos,100 memorias/intervenciones,300 JPEG; salida nueva |

Los evaluadores originales exportan clases/efectos seleccionados. Las capturas completas de probabilidades de las ejecuciones verificadas se suministran en `results/reproducibility/`. Los helpers matemáticos no se sustituyeron para producir esas capturas. Los nombres de las carpetas BS1 se conservan como identificadores históricos; los hashes del manifiesto establecen su configuración final.

Los comandos completos están en el README y cada CLI dispone de `--help`. Ejecutar un `--help` no equivale a repetir la evaluación o el entrenamiento. Las pruebas de navegador final usan expectativas de la serie seleccionada; no usan cifras VIS-PRE como oráculo de la configuración nueva.
