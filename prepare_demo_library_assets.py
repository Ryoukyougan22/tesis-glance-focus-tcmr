import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from remotezip import RemoteZip
except ImportError:
    RemoteZip = None


CHARADES_ZIP_URL = (
    "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/"
    "charades/Charades_v1_480.zip"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepara los assets de la librería de 10 casos STAR: "
            "exporta features S3D oficiales y descarga los MP4 de Charades."
        )
    )

    parser.add_argument(
        "--manifest",
        default=r"results\demo_library\demo_cases_10.json",
        help="JSON generado por build_demo_library_10.py",
    )

    parser.add_argument(
        "--s3d_pth",
        default=r"C:\tesis_data\STAR\vis_db\s3d.pth",
        help="Archivo oficial s3d.pth de STAR.",
    )

    parser.add_argument(
        "--video_dir",
        default=r"..\demo_videos",
        help="Carpeta donde guardar los MP4.",
    )

    parser.add_argument(
        "--feature_dir",
        default=r"..\demo_features",
        help="Carpeta donde guardar los .npy oficiales.",
    )

    parser.add_argument(
        "--output_manifest",
        default=r"results\demo_library\demo_cases_10_prepared.json",
        help="Manifest final con rutas locales verificadas.",
    )

    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="No descargar MP4; solo exportar features y validar lo existente.",
    )

    parser.add_argument(
        "--skip_features",
        action="store_true",
        help="No exportar features; solo descargar/validar videos.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescribir assets locales existentes.",
    )

    return parser.parse_args()


def normalize_feature_tensor(value):
    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim != 2:
        raise ValueError(
            "Feature con shape inesperado: {}. Se esperaba [T, D].".format(
                arr.shape
            )
        )

    if arr.shape[1] != 1024:
        raise ValueError(
            "Dimensión S3D inesperada: {}. Se esperaba 1024.".format(
                arr.shape[1]
            )
        )

    return arr.astype(np.float32, copy=False)


def validate_video(path):
    path = Path(path)

    if not path.is_file() or path.stat().st_size == 0:
        return False, None

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        return False, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    duration = None

    if fps and fps > 0 and frames and frames > 0:
        duration = float(frames / fps)

    return True, duration


def validate_feature(path):
    path = Path(path)

    if not path.is_file() or path.stat().st_size == 0:
        return False, None

    try:
        arr = np.load(str(path), mmap_mode="r")
    except Exception:
        return False, None

    if arr.ndim != 2 or arr.shape[1] != 1024:
        return False, tuple(arr.shape)

    return True, tuple(arr.shape)


def download_charades_video(video_id, output_path, remote_zip):
    filename = video_id + ".mp4"

    names = remote_zip.namelist()

    matches = [
        name
        for name in names
        if name.replace("\\", "/").endswith("/" + filename)
        or name == filename
    ]

    if not matches:
        raise FileNotFoundError(
            "No se encontró {} dentro del ZIP oficial de Charades.".format(
                filename
            )
        )

    remote_name = matches[0]
    info = remote_zip.getinfo(remote_name)

    print(
        "    Descargando {} ({:.2f} MB)...".format(
            filename,
            info.file_size / (1024.0 * 1024.0),
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with remote_zip.open(remote_name) as src, open(output_path, "wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)

            if not chunk:
                break

            dst.write(chunk)


def main():
    args = parse_args()

    manifest_path = Path(args.manifest)
    s3d_path = Path(args.s3d_pth)
    video_dir = Path(args.video_dir)
    feature_dir = Path(args.feature_dir)
    output_manifest_path = Path(args.output_manifest)

    if not manifest_path.is_file():
        raise FileNotFoundError(
            "No existe el manifest: {}".format(manifest_path)
        )

    with manifest_path.open("r", encoding="utf-8") as f:
        cases = json.load(f)

    if not isinstance(cases, list) or len(cases) != 10:
        raise ValueError(
            "Se esperaban exactamente 10 casos en el manifest; se obtuvieron {}.".format(
                len(cases) if isinstance(cases, list) else "?"
            )
        )

    video_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 100)
    print("PREPARACIÓN DE ASSETS - LIBRERÍA STAR DE 10 VIDEOS")
    print("=" * 100)
    print("Manifest :", manifest_path)
    print("Videos   :", video_dir.resolve())
    print("Features :", feature_dir.resolve())
    print()

    # =========================================================
    # 1. CARGAR S3D UNA SOLA VEZ
    # =========================================================
    s3d_db = None

    if not args.skip_features:
        if not s3d_path.is_file():
            raise FileNotFoundError(
                "No existe s3d.pth: {}".format(s3d_path)
            )

        print("Cargando s3d.pth una sola vez...")
        print("Esto puede tardar porque el archivo es grande.")

        s3d_db = torch.load(
            str(s3d_path),
            map_location="cpu",
        )

        print(
            "s3d.pth cargado. Entradas:",
            len(s3d_db),
        )
        print()

    # =========================================================
    # 2. EXPORTAR FEATURES OFICIALES
    # =========================================================
    if not args.skip_features:
        print("-" * 100)
        print("EXPORTANDO FEATURES S3D OFICIALES")
        print("-" * 100)

        for index, case in enumerate(cases, start=1):
            qid = case["question_id"]
            video_id = case["video_id"]

            output_feature = feature_dir / (
                video_id + "_s3d_official.npy"
            )

            print(
                "[{}/10] {} | {}".format(
                    index,
                    qid,
                    video_id,
                )
            )

            if output_feature.exists() and not args.overwrite:
                ok, shape = validate_feature(output_feature)

                if ok:
                    print(
                        "    Ya existe y es válida:",
                        output_feature,
                        shape,
                    )
                    continue

                print(
                    "    Existe pero no es válida; se regenerará."
                )

            if qid not in s3d_db:
                raise KeyError(
                    "No se encontró question_id {} en s3d.pth.".format(
                        qid
                    )
                )

            arr = normalize_feature_tensor(
                s3d_db[qid]
            )

            np.save(
                str(output_feature),
                arr,
            )

            print(
                "    Guardado:",
                output_feature,
                arr.shape,
            )

        print()

        # Liberar memoria RAM antes de descargar videos.
        del s3d_db

    # =========================================================
    # 3. DESCARGAR LOS MP4
    # =========================================================
    if not args.skip_download:
        if RemoteZip is None:
            raise ImportError(
                "Falta remotezip. Instálalo con: pip install remotezip"
            )

        print("-" * 100)
        print("DESCARGANDO VIDEOS CHARADES QUE FALTEN")
        print("-" * 100)
        print("Fuente:", CHARADES_ZIP_URL)
        print()

        with RemoteZip(CHARADES_ZIP_URL) as remote_zip:
            for index, case in enumerate(cases, start=1):
                video_id = case["video_id"]
                output_video = video_dir / (
                    video_id + ".mp4"
                )

                print(
                    "[{}/10] {}".format(
                        index,
                        video_id,
                    )
                )

                if output_video.exists() and not args.overwrite:
                    ok, duration = validate_video(
                        output_video
                    )

                    if ok:
                        print(
                            "    Ya existe y es válido: {} ({:.2f}s)".format(
                                output_video,
                                duration if duration is not None else 0.0,
                            )
                        )
                        continue

                    print(
                        "    Existe pero no es válido; se descargará de nuevo."
                    )

                download_charades_video(
                    video_id,
                    output_video,
                    remote_zip,
                )

                ok, duration = validate_video(
                    output_video
                )

                if not ok:
                    raise RuntimeError(
                        "El video descargado no pasó la validación: {}".format(
                            output_video
                        )
                    )

                print(
                    "    OK: {:.2f}s".format(
                        duration if duration is not None else 0.0
                    )
                )

        print()

    # =========================================================
    # 4. VALIDACIÓN FINAL 10/10
    # =========================================================
    print("-" * 100)
    print("VALIDACIÓN FINAL")
    print("-" * 100)

    prepared_cases = []
    all_ok = True

    for index, case in enumerate(cases, start=1):
        qid = case["question_id"]
        video_id = case["video_id"]

        video_path = (
            video_dir / (video_id + ".mp4")
        )
        feature_path = (
            feature_dir / (video_id + "_s3d_official.npy")
        )

        video_ok, duration = validate_video(
            video_path
        )
        feature_ok, feature_shape = validate_feature(
            feature_path
        )

        status = (
            "OK"
            if video_ok and feature_ok
            else "FALTA"
        )

        if status != "OK":
            all_ok = False

        print(
            "[{:02d}] {:5s} | {:12s} | video={} | features={} {}".format(
                index,
                status,
                video_id,
                "OK" if video_ok else "NO",
                "OK" if feature_ok else "NO",
                feature_shape if feature_shape is not None else "",
            )
        )

        item = dict(case)
        item["video_path"] = str(
            video_path.resolve()
        )
        item["feature_path"] = str(
            feature_path.resolve()
        )
        item["video_valid"] = bool(video_ok)
        item["feature_valid"] = bool(feature_ok)
        item["video_duration_seconds"] = duration
        item["feature_shape"] = (
            list(feature_shape)
            if feature_shape is not None
            else None
        )

        prepared_cases.append(item)

    with output_manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            prepared_cases,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 100)

    if all_ok:
        print("RESULTADO: 10/10 CASOS CON VIDEO + FEATURES OFICIALES LISTOS.")
    else:
        print(
            "RESULTADO: hay assets pendientes. Revisa las filas marcadas como FALTA."
        )

    print(
        "Manifest preparado:",
        output_manifest_path,
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
