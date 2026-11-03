"""TCMR-refined variant; original pipeline and pre-TCMR results are preserved."""
import argparse
import csv
import json
import os
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from infer_causal_video_final import (
    load_and_prepare_features,
    load_models,
)
from train_temporal_refinement import run_glancing, sort_memories_and_spans


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Atribución visual por perturbación: apaga tokens visuales y mide "
            "cuánto cambia cada memoria interna de Glance-Focus."
        )
    )

    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--features", type=str, required=True)

    parser.add_argument(
        "--reload_model_path",
        type=str,
        default="checkpoints/star/model_3000.tar",
    )
    parser.add_argument(
        "--temporal_checkpoint",
        type=str,
        default="checkpoints/tcmr/temporal_relation_encoder_seed42_best.pt",
    )

    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_queries", type=int, default=10)
    parser.add_argument("--event_pred_dim", type=int, default=111)
    parser.add_argument("--max_feats", type=int, default=80)
    parser.add_argument("--feature_dim", type=int, default=1024)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--num_options", type=int, default=4)
    parser.add_argument("--qa_dataset", type=str, default="star")
    parser.add_argument("--task_type", type=str, default="star")

    parser.add_argument(
        "--window",
        type=int,
        default=3,
        help=(
            "Número de tokens visuales consecutivos a poner en cero por perturbación. "
            "Debe ser impar. Recomendado: 3."
        ),
    )
    parser.add_argument(
        "--top_tokens",
        type=int,
        default=5,
        help="Número de regiones más influyentes a reportar por memoria.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/memory_visual_sensitivity_tcmr",
    )

    return parser.parse_args()


def get_video_duration_seconds(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps is None or fps <= 0 or frame_count is None or frame_count <= 0:
        return None

    return float(frame_count / fps)


def extract_frame_at_time(video_path, time_seconds, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    cap.set(cv2.CAP_PROP_POS_MSEC, float(time_seconds) * 1000.0)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return bool(cv2.imwrite(output_path, frame))


def prepared_index_to_approx_time(
    prepared_index,
    raw_feature_length,
    max_feats,
    duration_seconds,
):
    if duration_seconds is None or raw_feature_length <= 0:
        return None

    if raw_feature_length <= max_feats:
        raw_idx = prepared_index
    else:
        raw_idx = (prepared_index * raw_feature_length) // max_feats

    normalized_center = (float(raw_idx) + 0.5) / float(raw_feature_length)
    normalized_center = min(max(normalized_center, 0.0), 1.0)

    return normalized_center * duration_seconds


def make_model_input(single_features, num_options, device):
    return (
        single_features
        .unsqueeze(0)
        .repeat(num_options, 1, 1)
        .to(device)
    )


def collapse_option_rows(x, num_options):
    """
    Durante Glancing se repite el mismo video para las cuatro opciones.
    Promediamos esas copias para obtener una representación única del caso.
    """
    if x.shape[0] != num_options:
        # Para mantener el script explícito: esta demo usa una pregunta STAR
        # con cuatro alternativas.
        raise ValueError(
            "Se esperaban {} filas en Glancing, pero llegaron {}.".format(
                num_options,
                x.shape[0],
            )
        )
    return x.mean(dim=0)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gather_queries(values, order):
    return values.gather(1, order.unsqueeze(-1).expand(-1, -1, values.size(-1)))


def main():
    args = parse_args()

    if args.window < 1 or args.window % 2 == 0:
        raise ValueError("--window debe ser un entero impar >= 1.")

    if not os.path.isfile(args.video):
        raise FileNotFoundError("No existe el video: {}".format(args.video))

    if not os.path.isfile(args.features):
        raise FileNotFoundError("No existe el archivo de features: {}".format(args.features))

    if os.path.isdir(args.output_dir) and any(
        p.suffix != ".log" for p in Path(args.output_dir).iterdir()
    ):
        raise FileExistsError("No se sobrescriben resultados: " + args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    args.device = torch.device(
        "cuda:{}".format(args.device_id)
        if torch.cuda.is_available()
        else "cpu"
    )

    raw_features = np.load(args.features)
    raw_features = np.asarray(raw_features)

    if raw_features.ndim == 3 and raw_features.shape[0] == 1:
        raw_features = raw_features[0]

    if raw_features.ndim != 2:
        raise ValueError(
            "Se esperaban features [T,D], se obtuvo {}".format(raw_features.shape)
        )

    raw_feature_length = int(raw_features.shape[0])

    visual_single = load_and_prepare_features(
        args.features,
        args.max_feats,
        args.feature_dim,
    )

    prepared_length = int(visual_single.shape[0])
    valid_tokens = min(raw_feature_length, args.max_feats)

    print("Dispositivo:", args.device)
    print("Features originales:", tuple(raw_features.shape))
    print("Tokens preparados:", prepared_length)
    print("Tokens visuales reales:", valid_tokens)
    print("Ventana de perturbación:", args.window)

    model, temporal_encoder = load_models(args)
    assert not model.training and not temporal_encoder.training

    # ==========================================================
    # BASELINE
    # ==========================================================
    baseline_input = make_model_input(
        visual_single,
        args.num_options,
        args.device,
    )

    with torch.no_grad():
        baseline_outputs = run_glancing(
            model,
            baseline_input,
            args,
        )

    # Exactly the baseline path used in infer_causal_video_final.py.
    baseline_order = baseline_outputs["pred_spans"][..., 0].sort(dim=1)[1]
    assert torch.equal(baseline_order, baseline_order[:1].expand_as(baseline_order))
    sort_idx = baseline_order[0].cpu()
    with torch.no_grad():
        sorted_memory, sorted_spans = sort_memories_and_spans(
            baseline_outputs["memory_prompt"], baseline_outputs["pred_spans"]
        )
        refined_baseline = temporal_encoder(sorted_memory, sorted_spans)
    baseline_memory = collapse_option_rows(refined_baseline, args.num_options).cpu()
    baseline_raw = collapse_option_rows(sorted_memory, args.num_options).cpu()
    np.save(os.path.join(args.output_dir, "baseline_tcmr_memories.npy"), baseline_memory.numpy())
    np.save(os.path.join(args.output_dir, "baseline_glancing_memories.npy"), baseline_raw.numpy())
    refined_perturbations = []
    max_alignment_error = 0.0

    # ==========================================================
    # PERTURBACIONES DE TOKENS VISUALES
    # ==========================================================
    num_memories = args.num_queries
    sensitivity_cos = torch.zeros(
        num_memories,
        valid_tokens,
        dtype=torch.float32,
    )
    sensitivity_rel_l2 = torch.zeros_like(
        sensitivity_cos
    )

    half_window = args.window // 2

    print()
    print("Ejecutando perturbaciones visuales...")

    for token_idx in range(valid_tokens):
        perturbed_single = visual_single.clone()

        start = max(0, token_idx - half_window)
        end = min(valid_tokens, token_idx + half_window + 1)

        perturbed_single[start:end] = 0.0

        perturbed_input = make_model_input(
            perturbed_single,
            args.num_options,
            args.device,
        )

        with torch.no_grad():
            perturbed_outputs = run_glancing(
                model,
                perturbed_input,
                args,
            )

        # Preserve baseline query identity, gathering BOTH memories and fresh spans.
        with torch.no_grad():
            ordered_memory = gather_queries(perturbed_outputs["memory_prompt"], baseline_order)
            ordered_spans = gather_queries(perturbed_outputs["pred_spans"], baseline_order)
            refined = temporal_encoder(ordered_memory, ordered_spans)

            # Verify equivalence to the unchanged final pipeline on every perturbation.
            final_memory, final_spans = sort_memories_and_spans(
                perturbed_outputs["memory_prompt"], perturbed_outputs["pred_spans"]
            )
            final_refined = temporal_encoder(final_memory, final_spans)
            current_order = perturbed_outputs["pred_spans"][..., 0].sort(dim=1)[1]
            restore_baseline = current_order.argsort(dim=1).gather(1, baseline_order)
            aligned_final = gather_queries(final_refined, restore_baseline)
            error = float((refined - aligned_final).abs().max().item())
            max_alignment_error = max(max_alignment_error, error)
            if not torch.allclose(refined, aligned_final, rtol=1e-5, atol=1e-5):
                raise RuntimeError("Incompatibilidad con el pipeline final TCMR: error=" + str(error))
        perturbed_memory = collapse_option_rows(refined, args.num_options).cpu()
        refined_perturbations.append(perturbed_memory.numpy())

        cosine = F.cosine_similarity(
            baseline_memory,
            perturbed_memory,
            dim=-1,
            eps=1e-8,
        )

        cosine_distance = 1.0 - cosine

        l2 = torch.linalg.vector_norm(
            perturbed_memory - baseline_memory,
            dim=-1,
        )

        baseline_norm = torch.linalg.vector_norm(
            baseline_memory,
            dim=-1,
        ).clamp_min(1e-8)

        rel_l2 = l2 / baseline_norm

        sensitivity_cos[:, token_idx] = cosine_distance
        sensitivity_rel_l2[:, token_idx] = rel_l2

        if (
            token_idx == 0
            or (token_idx + 1) % 10 == 0
            or token_idx == valid_tokens - 1
        ):
            print(
                "  {}/{} perturbaciones".format(
                    token_idx + 1,
                    valid_tokens,
                )
            )

    duration_seconds = get_video_duration_seconds(
        args.video
    )

    # ==========================================================
    # REPORTAR RESULTADOS
    # ==========================================================
    print()
    print("=" * 82)
    print("SENSIBILIDAD VISUAL DE LAS MEMORIAS REFINADAS TCMR")
    print("=" * 82)
    print(
        "Métrica principal: 1 - cosine_similarity(memoria_baseline, memoria_perturbada)"
    )
    print(
        "Interpretación: cuanto mayor el valor, más cambia esa memoria al apagar "
        "esa región visual."
    )
    print(
        "IMPORTANTE: esto identifica contribución visual al estado interno; "
        "NO asigna una etiqueta semántica humana."
    )
    print("=" * 82)

    frames_root = os.path.join(
        args.output_dir,
        "top_frames",
    )

    result_memories = []
    csv_rows = []

    for memory_idx in range(num_memories):
        values = sensitivity_cos[memory_idx]

        total = float(values.sum().item())
        k = min(args.top_tokens, valid_tokens)

        top_values, top_indices = torch.topk(
            values,
            k=k,
        )

        top_share = (
            float(top_values.sum().item()) / total
            if total > 0
            else 0.0
        )

        memory_name = "E{:02d}".format(
            memory_idx + 1
        )

        print()
        print(memory_name)
        print(
            "  query original: Q{:02d}".format(
                int(sort_idx[memory_idx].item()) + 1
            )
        )
        print(
            "  concentración top-{}: {:.2%}".format(
                k,
                top_share,
            )
        )

        top_regions = []

        for rank in range(k):
            token_idx = int(
                top_indices[rank].item()
            )
            effect_cos = float(
                top_values[rank].item()
            )
            effect_rel_l2 = float(
                sensitivity_rel_l2[
                    memory_idx,
                    token_idx,
                ].item()
            )

            approx_time = prepared_index_to_approx_time(
                token_idx,
                raw_feature_length,
                args.max_feats,
                duration_seconds,
            )

            start = max(0, token_idx - half_window)
            end = min(valid_tokens, token_idx + half_window + 1)

            frame_path = None

            if approx_time is not None:
                frame_path = os.path.join(
                    frames_root,
                    memory_name,
                    "rank_{:02d}_token_{:03d}_{:.2f}s.jpg".format(
                        rank + 1,
                        token_idx,
                        approx_time,
                    ),
                )
                extract_frame_at_time(
                    args.video,
                    approx_time,
                    frame_path,
                )

            item = {
                "rank": rank + 1,
                "center_visual_token": token_idx,
                "perturbed_token_range": [start, end - 1],
                "cosine_distance": effect_cos,
                "relative_l2": effect_rel_l2,
                "approx_video_time_seconds": approx_time,
                "frame_path": frame_path,
            }
            top_regions.append(item)

            print(
                "  #{} token {:02d} ({}-{}) | cos-dist {:.6f} | rel-L2 {:.6f}{}".format(
                    rank + 1,
                    token_idx,
                    start,
                    end - 1,
                    effect_cos,
                    effect_rel_l2,
                    (
                        " | ~{:.2f}s".format(approx_time)
                        if approx_time is not None
                        else ""
                    ),
                )
            )

            csv_rows.append(
                {
                    "memory": memory_name,
                    "rank": rank + 1,
                    "center_visual_token": token_idx,
                    "window_start": start,
                    "window_end": end - 1,
                    "cosine_distance": effect_cos,
                    "relative_l2": effect_rel_l2,
                    "approx_video_time_seconds": approx_time,
                    "frame_path": frame_path,
                }
            )

        result_memories.append(
            {
                "memory": memory_name,
                "original_query_index_before_temporal_sort": int(
                    sort_idx[memory_idx].item()
                ),
                "top_k_concentration": top_share,
                "top_regions": top_regions,
            }
        )

    np.save(os.path.join(args.output_dir, "perturbed_tcmr_memories.npy"),
            np.stack(refined_perturbations, axis=0))  # [token, E01..E10, hidden]
    # Matrices completas
    np.save(
        os.path.join(
            args.output_dir,
            "memory_token_cosine_distance.npy",
        ),
        sensitivity_cos.numpy(),
    )
    np.save(
        os.path.join(
            args.output_dir,
            "memory_token_relative_l2.npy",
        ),
        sensitivity_rel_l2.numpy(),
    )

    csv_path = os.path.join(
        args.output_dir,
        "top_visual_regions.csv",
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "memory",
                "rank",
                "center_visual_token",
                "window_start",
                "window_end",
                "cosine_distance",
                "relative_l2",
                "approx_video_time_seconds",
                "frame_path",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    result = {
        "representation_stage": "tcmr_refined",
        "refinement": {
            "mechanism": "train_temporal_refinement.sort_memories_and_spans + loaded TemporalRelationEncoder",
            "glance_focus_checkpoint": os.path.abspath(args.reload_model_path),
            "temporal_checkpoint": os.path.abspath(args.temporal_checkpoint),
            "checkpoint_sha256": file_sha256(args.temporal_checkpoint),
            "alpha": float(temporal_encoder.alpha.item()),
            "baseline_query_order": sort_idx.tolist(),
            "perturbation_spans": "Recomputed by Glancing; gathered in baseline query order, not frozen.",
            "option_collapse": "Mean after TCMR, as in the interventional analysis.",
            "final_pipeline_alignment_max_abs_error": max_alignment_error,
            "final_pipeline_alignment_checked_tokens": valid_tokens,
            "alignment_tolerance": {"rtol": 1e-5, "atol": 1e-5},
            "baseline_refinement_l2": float(torch.linalg.vector_norm(baseline_memory - baseline_raw).item()),
        },
        "video": os.path.abspath(args.video),
        "features": os.path.abspath(args.features),
        "raw_feature_length": raw_feature_length,
        "prepared_feature_length": prepared_length,
        "valid_visual_tokens": valid_tokens,
        "video_duration_seconds": duration_seconds,
        "perturbation": {
            "operation": (
                "Zero a local window of prepared visual S3D feature tokens, "
                "then rerun Glancing and TCMR with recomputed spans."
            ),
            "window_size": args.window,
            "primary_metric": (
                "1 - cosine_similarity between baseline and perturbed memory vector"
            ),
            "secondary_metric": (
                "L2(memory_perturbed - memory_baseline) / ||memory_baseline||"
            ),
        },
        "memory_identity_note": (
            "Perturbed memories are aligned using the baseline query order after "
            "sorting by baseline pred_spans center. This avoids redefining E01...E10 "
            "after each perturbation."
        ),
        "interpretation_warning": (
            "This is a visual sensitivity analysis of TCMR-refined internal memories, not raw Glancing memories. A high score means "
            "that removing a visual feature region changes a memory representation more. "
            "It does not prove that the memory has a single semantic meaning, and the "
            "approximate video times are visualization aids rather than ground-truth "
            "temporal annotations."
        ),
        "memories": result_memories,
    }

    json_path = os.path.join(
        args.output_dir,
        "memory_visual_sensitivity.json",
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 82)
    print("Resultados guardados:")
    print(json_path)
    print(csv_path)
    print(
        os.path.join(
            args.output_dir,
            "memory_token_cosine_distance.npy",
        )
    )
    print(
        os.path.join(
            args.output_dir,
            "memory_token_relative_l2.npy",
        )
    )
    print("Frames:")
    print(frames_root)
    print("=" * 82)


if __name__ == "__main__":
    main()
