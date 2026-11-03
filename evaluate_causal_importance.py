import os
import csv
import random
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.star import VideoQACollator
from model.transformer_gf import build_transformer
from model.glance_focus import GF
from model.temporal_relation_encoder import TemporalRelationEncoder

from train_temporal_refinement import (
    build_dataset,
    prepare_batch,
    sort_memories_and_spans,
    run_glancing,
    run_focusing,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--base_data_dir", type=str, required=True)
    parser.add_argument("--reload_model_path", type=str, required=True)
    parser.add_argument("--temporal_checkpoint", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--max_batches",
        type=int,
        default=100,
        help="Smoke test. Usa -1 para evaluar toda validation.",
    )

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
        "--output_dir",
        type=str,
        default="results/causal_importance",
    )

    return parser.parse_args()


def collapse_option_memories(memory_bank, args):
    """
    [B * num_options, T, D] -> [B, T, D]
    """
    total_rows, num_events, hidden_dim = memory_bank.shape

    assert total_rows % args.num_options == 0, (
        "El número de filas de memory_bank no es divisible por num_options."
    )

    num_questions = total_rows // args.num_options

    memory_bank = memory_bank.reshape(
        num_questions,
        args.num_options,
        num_events,
        hidden_dim,
    )

    return memory_bank[:, 0, :, :]


def expand_question_memories(memory_bank, args):
    """
    [B, T, D] -> [B * num_options, T, D]
    """
    return memory_bank.repeat_interleave(
        args.num_options,
        dim=0,
    )


def new_metric_bucket():
    return {
        "count": 0,
        "baseline_correct": 0,
        "top_correct": 0,
        "bottom_correct": 0,
        "random_correct": 0,
        "top_flips": 0,
        "bottom_flips": 0,
        "random_flips": 0,
        "top_abs_change": 0.0,
        "bottom_abs_change": 0.0,
        "random_abs_change": 0.0,
        "top_signed_change": 0.0,
        "bottom_signed_change": 0.0,
        "random_signed_change": 0.0,
    }


def update_bucket(
    bucket,
    baseline_correct,
    top_correct,
    bottom_correct,
    random_correct,
    top_flip,
    bottom_flip,
    random_flip,
    top_abs_change,
    bottom_abs_change,
    random_abs_change,
    top_signed_change,
    bottom_signed_change,
    random_signed_change,
):
    bucket["count"] += 1

    bucket["baseline_correct"] += int(baseline_correct)
    bucket["top_correct"] += int(top_correct)
    bucket["bottom_correct"] += int(bottom_correct)
    bucket["random_correct"] += int(random_correct)

    bucket["top_flips"] += int(top_flip)
    bucket["bottom_flips"] += int(bottom_flip)
    bucket["random_flips"] += int(random_flip)

    bucket["top_abs_change"] += float(top_abs_change)
    bucket["bottom_abs_change"] += float(bottom_abs_change)
    bucket["random_abs_change"] += float(random_abs_change)

    bucket["top_signed_change"] += float(top_signed_change)
    bucket["bottom_signed_change"] += float(bottom_signed_change)
    bucket["random_signed_change"] += float(random_signed_change)


def summarize_bucket(bucket):
    n = bucket["count"]

    if n == 0:
        return None

    return {
        "count": n,
        "baseline_acc": bucket["baseline_correct"] / n,
        "top_acc": bucket["top_correct"] / n,
        "bottom_acc": bucket["bottom_correct"] / n,
        "random_acc": bucket["random_correct"] / n,
        "top_flip_rate": bucket["top_flips"] / n,
        "bottom_flip_rate": bucket["bottom_flips"] / n,
        "random_flip_rate": bucket["random_flips"] / n,
        "top_abs_change": bucket["top_abs_change"] / n,
        "bottom_abs_change": bucket["bottom_abs_change"] / n,
        "random_abs_change": bucket["random_abs_change"] / n,
        "top_signed_change": bucket["top_signed_change"] / n,
        "bottom_signed_change": bucket["bottom_signed_change"] / n,
        "random_signed_change": bucket["random_signed_change"] / n,
    }


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.device = torch.device(
        "cuda:{}".format(args.device_id)
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Seed:", args.seed)
    print("Dispositivo:", args.device)

    print("Cargando VALIDATION...")

    val_dataset = build_dataset(
        args,
        "val",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=VideoQACollator(
            task_type=args.task_type
        ).collate_batch,
    )

    print("Construyendo Glance-Focus...")

    transformer = build_transformer(args)

    model = GF(
        transformer,
        num_queries=args.num_queries,
        feature_dim=args.feature_dim,
        output_dim=args.output_dim,
        event_pred_dim=args.event_pred_dim,
        qa_dataset=args.qa_dataset,
    ).to(args.device)

    print("Cargando baseline:", args.reload_model_path)

    checkpoint = torch.load(
        args.reload_model_path,
        map_location=args.device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()

    temporal_encoder = TemporalRelationEncoder(
        hidden_dim=args.hidden_dim,
        dropout=0.1,
    ).to(args.device)

    print("Cargando TCMR temporal:", args.temporal_checkpoint)

    temporal_checkpoint = torch.load(
        args.temporal_checkpoint,
        map_location=args.device,
    )

    temporal_encoder.load_state_dict(
        temporal_checkpoint["temporal_encoder_state_dict"]
    )

    for parameter in temporal_encoder.parameters():
        parameter.requires_grad = False

    temporal_encoder.eval()

    print(
        "Accuracy TCMR guardado:",
        temporal_checkpoint.get("accuracy", "N/A"),
    )
    print("Alpha temporal:", temporal_encoder.alpha.item())

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    global_bucket = new_metric_bucket()

    category_buckets = {
        "Interaction": new_metric_bucket(),
        "Sequence": new_metric_bucket(),
        "Prediction": new_metric_bucket(),
        "Feasibility": new_metric_bucket(),
    }

    per_question_rows = []
    rng = random.Random(args.seed)

    with torch.no_grad():

        pbar = tqdm(
            val_loader,
            desc="Evaluando importancia causal",
        )

        for batch_idx, batch in enumerate(pbar):

            if (
                args.max_batches > 0
                and batch_idx >= args.max_batches
            ):
                break

            batch = prepare_batch(
                batch,
                args,
            )

            labels = batch["labels"]
            frame_features = batch["visual_inputs"]

            # 1. Glancing
            outputs_event = run_glancing(
                model,
                frame_features,
                args,
            )

            memory_prompt = outputs_event["memory_prompt"]
            pred_spans = outputs_event["pred_spans"]

            # 2. Orden temporal
            memory_prompt, pred_spans = sort_memories_and_spans(
                memory_prompt,
                pred_spans,
            )

            # 3. TCMR temporal
            temporal_memory = temporal_encoder(
                memory_prompt,
                pred_spans,
            )

            question_memory = collapse_option_memories(
                temporal_memory,
                args,
            )

            B, T, _ = question_memory.shape

            # 4. Predicción original
            baseline_outputs = run_focusing(
                model,
                frame_features,
                temporal_memory,
                batch["text_str_list"],
                args,
            )

            baseline_logits = baseline_outputs["pred_answer"]

            baseline_probs = torch.softmax(
                baseline_logits,
                dim=1,
            )

            baseline_pred = torch.argmax(
                baseline_probs,
                dim=1,
            )

            row_idx = torch.arange(
                B,
                device=args.device,
            )

            baseline_pred_prob = baseline_probs[
                row_idx,
                baseline_pred,
            ]

            # 5. Intervenir cada memoria: do(m_i = 0)
            intervened_probs_list = []

            for event_idx in range(T):

                intervened_memory = question_memory.clone()
                intervened_memory[:, event_idx, :] = 0.0

                expanded_memory = expand_question_memories(
                    intervened_memory,
                    args,
                )

                intervened_outputs = run_focusing(
                    model,
                    frame_features,
                    expanded_memory,
                    batch["text_str_list"],
                    args,
                )

                intervened_logits = intervened_outputs["pred_answer"]

                intervened_probs = torch.softmax(
                    intervened_logits,
                    dim=1,
                )

                intervened_probs_list.append(
                    intervened_probs
                )

            # [B, T, C]
            intervened_probs_all = torch.stack(
                intervened_probs_list,
                dim=1,
            )

            # 6. Influencia sobre la respuesta originalmente predicha
            target_idx = baseline_pred.view(
                B,
                1,
                1,
            ).expand(
                -1,
                T,
                1,
            )

            intervened_pred_prob = torch.gather(
                intervened_probs_all,
                dim=2,
                index=target_idx,
            ).squeeze(2)

            signed_effects = (
                baseline_pred_prob.unsqueeze(1)
                - intervened_pred_prob
            )

            abs_effects = torch.abs(
                signed_effects
            )

            # 7. TOP / BOTTOM / RANDOM
            top_idx = torch.argmax(
                abs_effects,
                dim=1,
            )

            bottom_idx = torch.argmin(
                abs_effects,
                dim=1,
            )

            random_idx = torch.tensor(
                [
                    rng.randrange(T)
                    for _ in range(B)
                ],
                dtype=torch.long,
                device=args.device,
            )

            top_probs = intervened_probs_all[
                row_idx,
                top_idx,
                :
            ]

            bottom_probs = intervened_probs_all[
                row_idx,
                bottom_idx,
                :
            ]

            random_probs = intervened_probs_all[
                row_idx,
                random_idx,
                :
            ]

            top_pred = torch.argmax(
                top_probs,
                dim=1,
            )

            bottom_pred = torch.argmax(
                bottom_probs,
                dim=1,
            )

            random_pred = torch.argmax(
                random_probs,
                dim=1,
            )

            top_original_class_prob = top_probs[
                row_idx,
                baseline_pred,
            ]

            bottom_original_class_prob = bottom_probs[
                row_idx,
                baseline_pred,
            ]

            random_original_class_prob = random_probs[
                row_idx,
                baseline_pred,
            ]

            top_signed_change = (
                baseline_pred_prob
                - top_original_class_prob
            )

            bottom_signed_change = (
                baseline_pred_prob
                - bottom_original_class_prob
            )

            random_signed_change = (
                baseline_pred_prob
                - random_original_class_prob
            )

            top_abs_change = torch.abs(
                top_signed_change
            )

            bottom_abs_change = torch.abs(
                bottom_signed_change
            )

            random_abs_change = torch.abs(
                random_signed_change
            )

            # 8. Métricas por pregunta
            for i, qid in enumerate(
                batch["question_ids"]
            ):
                category = qid.split("_")[0]

                baseline_correct = (
                    baseline_pred[i].item()
                    == labels[i].item()
                )

                top_correct = (
                    top_pred[i].item()
                    == labels[i].item()
                )

                bottom_correct = (
                    bottom_pred[i].item()
                    == labels[i].item()
                )

                random_correct = (
                    random_pred[i].item()
                    == labels[i].item()
                )

                top_flip = (
                    top_pred[i].item()
                    != baseline_pred[i].item()
                )

                bottom_flip = (
                    bottom_pred[i].item()
                    != baseline_pred[i].item()
                )

                random_flip = (
                    random_pred[i].item()
                    != baseline_pred[i].item()
                )

                metric_values = {
                    "baseline_correct": baseline_correct,
                    "top_correct": top_correct,
                    "bottom_correct": bottom_correct,
                    "random_correct": random_correct,
                    "top_flip": top_flip,
                    "bottom_flip": bottom_flip,
                    "random_flip": random_flip,
                    "top_abs_change": top_abs_change[i].item(),
                    "bottom_abs_change": bottom_abs_change[i].item(),
                    "random_abs_change": random_abs_change[i].item(),
                    "top_signed_change": top_signed_change[i].item(),
                    "bottom_signed_change": bottom_signed_change[i].item(),
                    "random_signed_change": random_signed_change[i].item(),
                }

                update_bucket(
                    global_bucket,
                    **metric_values,
                )

                if category not in category_buckets:
                    category_buckets[
                        category
                    ] = new_metric_bucket()

                update_bucket(
                    category_buckets[category],
                    **metric_values,
                )

                per_question_rows.append(
                    {
                        "question_id": qid,
                        "category": category,
                        "label": labels[i].item(),
                        "baseline_prediction": baseline_pred[i].item(),
                        "baseline_correct": int(baseline_correct),
                        "top_event": top_idx[i].item(),
                        "bottom_event": bottom_idx[i].item(),
                        "random_event": random_idx[i].item(),
                        "top_abs_effect": top_abs_change[i].item(),
                        "bottom_abs_effect": bottom_abs_change[i].item(),
                        "random_abs_effect": random_abs_change[i].item(),
                        "top_signed_effect": top_signed_change[i].item(),
                        "bottom_signed_effect": bottom_signed_change[i].item(),
                        "random_signed_effect": random_signed_change[i].item(),
                        "top_prediction": top_pred[i].item(),
                        "bottom_prediction": bottom_pred[i].item(),
                        "random_prediction": random_pred[i].item(),
                        "top_correct": int(top_correct),
                        "bottom_correct": int(bottom_correct),
                        "random_correct": int(random_correct),
                        "top_flip": int(top_flip),
                        "bottom_flip": int(bottom_flip),
                        "random_flip": int(random_flip),
                    }
                )

    global_summary = summarize_bucket(
        global_bucket
    )

    print()
    print("=" * 78)
    print("VALIDACIÓN DE IMPORTANCIA CAUSAL")
    print("=" * 78)

    print(
        "Preguntas evaluadas:",
        global_summary["count"],
    )

    print()
    print("ACCURACY")
    print("-" * 78)

    print(
        "Original TCMR : {:6.2f}%".format(
            global_summary["baseline_acc"] * 100
        )
    )

    print(
        "Quitar TOP    : {:6.2f}%  ({:+.2f} pp)".format(
            global_summary["top_acc"] * 100,
            (
                global_summary["top_acc"]
                - global_summary["baseline_acc"]
            ) * 100,
        )
    )

    print(
        "Quitar BOTTOM : {:6.2f}%  ({:+.2f} pp)".format(
            global_summary["bottom_acc"] * 100,
            (
                global_summary["bottom_acc"]
                - global_summary["baseline_acc"]
            ) * 100,
        )
    )

    print(
        "Quitar RANDOM : {:6.2f}%  ({:+.2f} pp)".format(
            global_summary["random_acc"] * 100,
            (
                global_summary["random_acc"]
                - global_summary["baseline_acc"]
            ) * 100,
        )
    )

    print()
    print("CAMBIO ABSOLUTO EN CONFIANZA DE LA RESPUESTA ORIGINAL")
    print("-" * 78)

    print(
        "TOP           : {:.6f}".format(
            global_summary["top_abs_change"]
        )
    )
    print(
        "BOTTOM        : {:.6f}".format(
            global_summary["bottom_abs_change"]
        )
    )
    print(
        "RANDOM        : {:.6f}".format(
            global_summary["random_abs_change"]
        )
    )

    print()
    print("CAMBIO FIRMADO EN CONFIANZA")
    print("(positivo = al quitar memoria baja la confianza)")
    print("-" * 78)

    print(
        "TOP           : {:.6f}".format(
            global_summary["top_signed_change"]
        )
    )
    print(
        "BOTTOM        : {:.6f}".format(
            global_summary["bottom_signed_change"]
        )
    )
    print(
        "RANDOM        : {:.6f}".format(
            global_summary["random_signed_change"]
        )
    )

    print()
    print("FLIP RATE")
    print("-" * 78)

    print(
        "TOP           : {:6.2f}%".format(
            global_summary["top_flip_rate"] * 100
        )
    )
    print(
        "BOTTOM        : {:6.2f}%".format(
            global_summary["bottom_flip_rate"] * 100
        )
    )
    print(
        "RANDOM        : {:6.2f}%".format(
            global_summary["random_flip_rate"] * 100
        )
    )

    print()
    print("POR TIPO DE PREGUNTA")
    print("=" * 78)

    category_order = [
        "Interaction",
        "Sequence",
        "Prediction",
        "Feasibility",
    ]

    summary_rows = []

    for category in category_order:

        summary = summarize_bucket(
            category_buckets.get(
                category,
                new_metric_bucket(),
            )
        )

        if summary is None:
            continue

        print()
        print(category)

        print(
            "  Original     : {:6.2f}%".format(
                summary["baseline_acc"] * 100
            )
        )
        print(
            "  Quitar TOP   : {:6.2f}%".format(
                summary["top_acc"] * 100
            )
        )
        print(
            "  Quitar BOTTOM: {:6.2f}%".format(
                summary["bottom_acc"] * 100
            )
        )
        print(
            "  Quitar RANDOM: {:6.2f}%".format(
                summary["random_acc"] * 100
            )
        )
        print(
            "  Flip TOP     : {:6.2f}%".format(
                summary["top_flip_rate"] * 100
            )
        )
        print(
            "  Flip BOTTOM  : {:6.2f}%".format(
                summary["bottom_flip_rate"] * 100
            )
        )
        print(
            "  Flip RANDOM  : {:6.2f}%".format(
                summary["random_flip_rate"] * 100
            )
        )
        print(
            "  |Δp| TOP     : {:.6f}".format(
                summary["top_abs_change"]
            )
        )
        print(
            "  |Δp| BOTTOM  : {:.6f}".format(
                summary["bottom_abs_change"]
            )
        )
        print(
            "  |Δp| RANDOM  : {:.6f}".format(
                summary["random_abs_change"]
            )
        )

        summary_rows.append(
            {
                "category": category,
                **summary,
            }
        )

    summary_rows.append(
        {
            "category": "Overall",
            **global_summary,
        }
    )

    per_question_path = os.path.join(
        args.output_dir,
        "causal_importance_per_question.csv",
    )

    with open(
        per_question_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=per_question_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(
            per_question_rows
        )

    summary_path = os.path.join(
        args.output_dir,
        "causal_importance_summary.csv",
    )

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=summary_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    print()
    print("=" * 78)
    print("Archivos guardados:")
    print(per_question_path)
    print(summary_path)
    print("=" * 78)


if __name__ == "__main__":
    main()