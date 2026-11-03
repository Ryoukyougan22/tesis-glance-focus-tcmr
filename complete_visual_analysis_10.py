"""Run the original visual pipeline for nine demos; preserve RKGG5 byte for byte.

Run with the project's glance-focus Conda interpreter. All artifacts of this
orchestration use new filenames; existing case folders are never overwritten.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import cv2
import numpy as np
import torch


PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT / "results/memory_visual_sensitivity"
MANIFEST = PROJECT / "results/demo_library/demo_cases_10_prepared.json"
CHECKPOINTS = [PROJECT / "checkpoints/star/model_3000.tar",
               PROJECT / "checkpoints/tcmr/temporal_relation_encoder_seed42_best.pt"]
EXPECTED = ["E{:02d}".format(i) for i in range(1, 11)]
SCRIPTS = ["inspect_memory_visual_sensitivity.py", "analyze_memory_visual_selectivity.py",
           "export_selective_memory_frames.py"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(folder):
    return {str(p.relative_to(folder)): sha256(p) for p in folder.rglob("*") if p.is_file()}


def write_json_new(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def validate(case):
    folder = ROOT / case["video_id"]
    sens = json.loads((folder / "memory_visual_sensitivity.json").read_text(encoding="utf-8"))
    sel = json.loads((folder / "memory_visual_selectivity.json").read_text(encoding="utf-8"))
    assert [m["memory"] for m in sens["memories"]] == EXPECTED
    assert [m["memory"] for m in sel["memories"]] == EXPECTED
    assert sens["perturbation"]["window_size"] == 3
    assert Path(sens["video"]).resolve() == Path(case["video_path"]).resolve()
    assert Path(sens["features"]).resolve() == Path(case["feature_path"]).resolve()
    matrices = [np.load(str(folder / ("memory_token_" + name + ".npy")))
                for name in ("cosine_distance", "relative_l2", "specificity_z")]
    for matrix in matrices:
        assert matrix.shape == (10, sens["valid_visual_tokens"])
        assert np.isfinite(matrix).all()
    cosine, l2, z = matrices
    values = cosine.astype(np.float64)
    std = values.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1e-12
    np.testing.assert_allclose(z, (values - values.mean(axis=0, keepdims=True)) / std,
                               rtol=1e-7, atol=1e-10)
    for index, memory in enumerate(sens["memories"]):
        assert len(memory["top_regions"]) == 5
        assert max(r["cosine_distance"] for r in memory["top_regions"]) == float(cosine[index].max())
        for region in memory["top_regions"]:
            frame = Path(region["frame_path"])
            assert frame.is_file() and cv2.imread(str(frame)) is not None, str(frame)
        selective = sel["memories"][index]["top_distinctive_regions"]
        assert len(selective) == 5
        assert np.isclose(max(r["specificity_z"] for r in selective), z[index].max())
        if case["video_id"] != "RKGG5":
            frames = list((folder / "selective_frames" / memory["memory"]).glob("*.jpg"))
            assert len(frames) == 3
            assert all(cv2.imread(str(p)) is not None for p in frames)
    for filename in ("top_visual_regions.csv", "memory_visual_selectivity.csv"):
        with (folder / filename).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 50 and sorted({r["memory"] for r in rows}) == EXPECTED
    return {"case_id": case["video_id"], "category": case["category"],
            "question": case["question"], "num_memories": 10,
            "sensitivity_ok": True, "selectivity_ok": True, "output_folder": str(folder)}


def run(command, logfile):
    print("Ejecutando: " + subprocess.list2cmdline(command), flush=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with logfile.open("x", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=str(PROJECT), env=env, stdout=stream,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        print(logfile.read_text(encoding="utf-8"), flush=True)
        raise RuntimeError("Pipeline fallido; revisar " + str(logfile))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(str(PROJECT))
    cases = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert len(cases) == 10 and len({c["video_id"] for c in cases}) == 10
    assert "RKGG5" in {c["video_id"] for c in cases}
    for path in CHECKPOINTS + [PROJECT / s for s in SCRIPTS]:
        assert path.is_file(), str(path)
    for case in cases:
        for key in ("video_path", "feature_path"):
            assert Path(case[key]).is_file(), case[key]
        features = np.load(case["feature_path"])
        assert features.ndim == 2 and features.shape[1] == 1024 and np.isfinite(features).all()
        video = cv2.VideoCapture(case["video_path"])
        readable, frame = video.read()
        video.release()
        assert readable and frame is not None, case["video_path"]
    protected = ROOT / "RKGG5"
    before = snapshot(protected)
    reference = next(c for c in cases if c["video_id"] == "RKGG5")
    validate(reference)
    # Only an empty directory or logs from a failed initialization can be retried.
    for case in cases:
        if case["video_id"] != "RKGG5" and (ROOT / case["video_id"]).exists():
            if any(p.is_dir() or p.suffix != ".log" for p in (ROOT / case["video_id"]).iterdir()):
                raise FileExistsError("No se sobrescribe " + str(ROOT / case["video_id"]))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    provenance = {
        "run_id": run_id, "python": sys.executable, "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "offline_model_cache": True,
        "checkpoints_sha256": {str(p): sha256(p) for p in CHECKPOINTS},
        "scripts_sha256": {s: sha256(PROJECT / s) for s in SCRIPTS},
        "rkgg5_original_sha256": before,
        "parameters": {"num_queries": 10, "window": 3, "top_tokens": 5,
                       "max_feats": 80, "feature_dim": 1024, "selective_frames_per_memory": 3},
        "cases": cases,
        "note": "Original pipeline measures pre-TCMR Glancing memory_prompt. TCMR is loaded but "
                "not applied to these sensitivity vectors. RKGG5 has no historical checkpoint hash. "
                "QA correctness comes from the existing demo validation, not this visual analysis.",
    }
    provenance["inputs_sha256"] = {case["video_id"]: {
        key: sha256(Path(case[key])) for key in ("video_path", "feature_path")}
        for case in cases}
    write_json_new(ROOT / ("execution_10_" + run_id + ".json"), provenance)
    rows = []
    try:
        for case in cases:
            case_id = case["video_id"]
            folder = ROOT / case_id
            if case_id != "RKGG5":
                folder.mkdir(exist_ok=True)
                common = [sys.executable, "-u"]
                run(common + [SCRIPTS[0], "--video", case["video_path"],
                    "--features", case["feature_path"], "--reload_model_path", str(CHECKPOINTS[0]),
                    "--temporal_checkpoint", str(CHECKPOINTS[1]), "--num_queries", "10",
                    "--window", "3", "--top_tokens", "5", "--max_feats", "80",
                    "--feature_dim", "1024", "--output_dir", str(folder)], folder / ("sensitivity_" + run_id + ".log"))
                run(common + [SCRIPTS[1], "--input_dir", str(folder), "--top_k", "5"],
                    folder / "selectivity_execution.log")
                run(common + [SCRIPTS[2], "--video", case["video_path"], "--selectivity_csv",
                    str(folder / "memory_visual_selectivity.csv"), "--top_k", "3", "--output_dir",
                    str(folder / "selective_frames")], folder / "frames_execution.log")
            rows.append(validate(case))
            print("VALIDADO: {} | 10 memorias | sensibilidad OK | selectividad OK".format(case_id), flush=True)
    finally:
        assert snapshot(protected) == before, "Los archivos de RKGG5 cambiaron"
        print("RKGG5: integridad SHA-256 verificada; sin cambios.", flush=True)
    assert len(rows) == 10 and sum(r["num_memories"] for r in rows) == 100
    assert len([p for p in ROOT.iterdir() if p.is_dir()]) == 10
    report = {"cases_processed": 10, "memories_per_case": 10, "total_memories": 100,
              "new_cases_executed": 9, "rkgg5_unchanged": True, "rows": rows,
              "run_id": run_id}
    write_json_new(ROOT / ("validation_100_memories_" + run_id + ".json"), report)
    fields = ["case_id", "category", "question", "num_memories", "sensitivity_ok", "selectivity_ok", "output_folder"]
    with (ROOT / ("validation_100_memories_" + run_id + ".csv")).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("\n" + " | ".join(fields))
    for row in rows:
        print(" | ".join(str(row[k]) for k in fields))
    print("\nCasos procesados = 10\nMemorias por caso = 10\nTotal de memorias = 100")


if __name__ == "__main__":
    main()
