# Entorno y recursos

Referencia comprobada: Python 3.7.16; torch 1.11.0+cu113; NumPy 1.21.6; pandas 1.3.5; scipy 1.7.3; transformers 4.30.2; OpenCV 4.8.1.78; h5py 3.8.0; PyYAML 6.0.1; tensorboard 2.11.2; tqdm 4.68.2. GPU utilizada: RTX3060, CUDA11.3, cuDNN8200.

`requirements.txt` mantiene las dependencias del código original. `requirements-reproduction.txt` fija las versiones observadas del entorno científico. La compatibilidad de PyTorch/CUDA debe resolverse para el equipo de destino; los manifiestos no garantizan una reinstalación limpia en otro sistema ni la reproducción con las versiones más recientes.

Web: `webapp/requirements.txt` fija FastAPI/uvicorn/httpx/pydantic. Frontend: `webapp/frontend/package-lock.json`, React19.1.1, Vite6.4.3, TypeScript5.9.2 y Playwright1.55.1. Las pruebas usan Microsoft Edge. No se distribuyen dependencias instaladas ni entornos.

## Recursos externos obligatorios

- Pesos GF/TCMR exactos, con los hashes del README y `configs/final_experiment.json`. No están incluidos; el TCMR debe obtenerse del responsable de la investigación. No hay URL pública verificada de ese artefacto.
- STAR: `txt_db/{train,val}.jsonl`, `events.json`, `action_mapping.txt`, `vis_db/{s3d.pth,strID2numID.json}`; pasar la ruta mediante `--base_data_dir`.
- `roberta-base` y tokenizador en caché de transformers. Usar offline únicamente después de disponer de esos recursos.
- MP4 de los diez videos en `../demo_videos/` y features oficiales NPY en `../demo_features/`, según `results/demo_library/demo_cases_10_prepared.json`. El script `prepare_demo_library_assets.py --help` documenta la preparación; obtener los recursos bajo sus condiciones aplicables.
- Los frames se reconstruyen con `python scripts/prepare_demo_frames.py` después de obtener los MP4. No se recalculan métricas ni se sobrescriben imágenes.

Para exportar las features de los diez casos cuando ya se dispone de los MP4 autorizados, ejecutar desde la raíz (salida nueva):

```text
python prepare_demo_library_assets.py --manifest results/demo_library/demo_cases_10_prepared.json --s3d_pth data/STAR/vis_db/s3d.pth --output_manifest results/new_assets_manifest.json --skip_download
```

Se indican los argumentos explícitamente para no depender del nombre del catálogo anterior usado como default del preparador. El JSON publicado mantiene las rutas relativas de los recursos externos; no reemplazarlo por una salida con rutas personales.

No se redistribuyen pesos, features originales, videos, frames o capturas audiovisuales. Verificar permisos en las fuentes; la licencia del código no basta para esos recursos.

## Variables de configuración

`VIDEOQA_GF_CHECKPOINT`, `VIDEOQA_TCMR_CHECKPOINT`, `VIDEOQA_DEVICE`, `VIDEOQA_BACKEND`, `VIDEOQA_FRONTEND`, `VIDEOQA_VALIDATION_OUTPUT`, `HF_HUB_OFFLINE` y `TRANSFORMERS_OFFLINE`. Las rutas de pesos pueden ser externas; los hashes deben coincidir. `.env.example` es una guía para exportar variables, no una carga automática de dotenv.

El backend necesita pesos y medios al arrancar. Las pruebas en `tests/` solo leen los resultados públicos y no requieren esos recursos. Compilar el frontend requiere Node/npm y las dependencias del lockfile; no requiere inferir con el modelo.
