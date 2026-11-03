"""Execute, audit and consolidate the ten existing cases after TCMR refinement."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

import complete_visual_analysis_10 as shared


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(str(shared.PROJECT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--temporal-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    shared.CHECKPOINTS[1] = args.temporal_checkpoint.resolve()
    original_root = shared.ROOT
    original_hashes = shared.snapshot(original_root)
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError("No se sobrescriben resultados: " + str(root))
    cases = json.loads(shared.MANIFEST.read_text(encoding="utf-8"))
    expected = {"RKGG5": "Sequence", "DQEC3": "Sequence", "UEP20": "Sequence",
                "6CAZU": "Interaction", "8SDK5": "Interaction", "QPX3S": "Interaction",
                "IO712": "Prediction", "I1OLQ": "Prediction",
                "ZNQVC": "Feasibility", "KFI7N": "Feasibility"}
    assert len(cases) == 10
    assert {c["video_id"]: c["category"] for c in cases} == expected
    for case in cases:
        for key in ("video_path", "feature_path"):
            assert Path(case[key]).is_file(), case[key]
    for checkpoint in shared.CHECKPOINTS:
        assert checkpoint.is_file(), str(checkpoint)
    expected_hashes = ["74e3c2064d61ee4762c2d1cac8853017124a3c47df3e351ae7a3dd066c765326", "8a30ff3e331fe40a121a0b6e3f6c13182fd3216caee5664fcf124336ad569eb9"]
    assert [shared.sha256(p) for p in shared.CHECKPOINTS] == expected_hashes
    root.mkdir()
    shared.ROOT = root
    shared.write_json_new(root / "execution_manifest.json", {
        "representation_stage": "tcmr_refined", "cases": cases,
        "python": sys.executable, "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "checkpoints_sha256": {str(p): shared.sha256(p) for p in shared.CHECKPOINTS},
        "scripts_sha256": {p: shared.sha256(shared.PROJECT / p) for p in (
            "inspect_memory_visual_sensitivity_tcmr.py", "analyze_memory_visual_selectivity.py",
            "export_selective_memory_frames.py", "infer_causal_video_final.py",
            "model/temporal_relation_encoder.py", "train_temporal_refinement.py",
            "scripts/analysis/consolidate_visual_analysis.py")},
        "previous_results_sha256": original_hashes,
        "methodology": "Baseline follows final pipeline sorting and TCMR. Perturbations use "
            "baseline query order and newly predicted spans. Each refined perturbation is checked "
            "against the final pipeline after query alignment. Options are averaged after TCMR."
    })
    rows = []
    try:
        for case in cases:
            folder = root / case["video_id"]
            folder.mkdir()
            common = [sys.executable, "-u"]
            shared.run(common + ["inspect_memory_visual_sensitivity_tcmr.py", "--video", case["video_path"],
                "--features", case["feature_path"], "--reload_model_path", str(shared.CHECKPOINTS[0]),
                "--temporal_checkpoint", str(shared.CHECKPOINTS[1]), "--num_queries", "10",
                "--window", "3", "--top_tokens", "5", "--max_feats", "80",
                "--feature_dim", "1024", "--output_dir", str(folder)], folder / "sensitivity.log")
            shared.run(common + ["analyze_memory_visual_selectivity.py", "--input_dir", str(folder),
                                  "--top_k", "5"], folder / "selectivity.log")
            shared.run(common + ["export_selective_memory_frames.py", "--video", case["video_path"],
                "--selectivity_csv", str(folder / "memory_visual_selectivity.csv"),
                "--top_k", "3", "--output_dir", str(folder / "selective_frames")], folder / "frames.log")
            row = shared.validate(case)
            metadata = json.loads((folder / "memory_visual_sensitivity.json").read_text(encoding="utf-8"))
            assert metadata["representation_stage"] == "tcmr_refined"
            assert metadata["refinement"]["final_pipeline_alignment_checked_tokens"] == metadata["valid_visual_tokens"]
            baseline = torch.from_numpy(np.load(str(folder / "baseline_tcmr_memories.npy")))
            raw = torch.from_numpy(np.load(str(folder / "baseline_glancing_memories.npy")))
            perturbed = torch.from_numpy(np.load(str(folder / "perturbed_tcmr_memories.npy")))
            assert baseline.shape == (10, 512) and not torch.equal(raw, baseline)
            assert perturbed.shape == (metadata["valid_visual_tokens"], 10, 512)
            cosine = (1 - F.cosine_similarity(baseline.unsqueeze(0), perturbed, dim=-1, eps=1e-8)).T.numpy()
            l2 = (torch.linalg.vector_norm(perturbed - baseline, dim=-1) /
                  torch.linalg.vector_norm(baseline, dim=-1).clamp_min(1e-8)).T.numpy()
            np.testing.assert_allclose(cosine, np.load(str(folder / "memory_token_cosine_distance.npy")), atol=2e-7, rtol=1e-5)
            np.testing.assert_allclose(l2, np.load(str(folder / "memory_token_relative_l2.npy")), atol=1e-7, rtol=1e-5)
            assert len(list((folder / "selective_frames").rglob("*.jpg"))) == 30
            row["representation_stage"] = "tcmr_refined"
            row["metrics_recomputed_from_saved_refined_vectors_ok"] = True
            row["final_pipeline_alignment_max_abs_error"] = metadata["refinement"]["final_pipeline_alignment_max_abs_error"]
            rows.append(row)
            print("VALIDADO TCMR: {} | 10 memorias | error alineación {:.3g}".format(
                case["video_id"], row["final_pipeline_alignment_max_abs_error"]), flush=True)
    finally:
        assert shared.snapshot(original_root) == original_hashes, "Los resultados previos cambiaron"
        print("Resultados anteriores: integridad SHA-256 verificada, sin cambios.", flush=True)
    assert len(rows) == 10 and sum(r["num_memories"] for r in rows) == 100
    shared.write_json_new(root / "validation_100_memories.json", {
        "cases_processed": 10, "memories_per_case": 10, "total_memories": 100,
        "representation_stage": "tcmr_refined", "previous_results_unchanged": True, "rows": rows})
    with (root / "validation_100_memories.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # The standard Python interpreter is not needed: the Conda environment runs the same consolidator.
    subprocess.run([sys.executable, "scripts/analysis/consolidate_visual_analysis.py", "--root", str(root)], check=True)
    summary = json.loads((root / "visual_analysis_summary.json").read_text(encoding="utf-8"))
    assert summary["total_cases"] == 10 and summary["total_memories"] == 100
    assert summary["representation_stage"] == "tcmr_refined" and not summary["warnings"]
    print("\ncase_id | category | num_memories | sensitivity_ok | selectivity_ok | output_folder")
    for row in rows:
        print(" | ".join(str(row[k]) for k in (
            "case_id", "category", "num_memories", "sensitivity_ok", "selectivity_ok", "output_folder")))
    print("\nCONFIRMADO: 10 casos, 100 memorias refinadas mediante TCMR; no memorias crudas de Glancing.")


if __name__ == "__main__":
    main()
