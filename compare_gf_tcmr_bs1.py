import argparse
import csv
import math
import os
import random

import numpy as np
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


CATEGORIES = [
    "Interaction",
    "Sequence",
    "Prediction",
    "Feasibility",
]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Comparación pareada Glance-Focus vs TCMR "
            "sobre STAR validation con batch_size=1."
        )
    )

    p.add_argument(
        "--base_data_dir",
        default=r"C:\tesis_data\STAR",
    )
    p.add_argument(
        "--reload_model_path",
        default="checkpoints/star/model_3000.tar",
    )
    p.add_argument(
        "--temporal_checkpoint",
        default="checkpoints/tcmr/temporal_relation_encoder_seed42_best.pt",
    )
    p.add_argument(
        "--output_dir",
        default="results/gf_vs_tcmr_bs1",
    )
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    # Configuración usada en los experimentos.
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--num_queries", type=int, default=10)
    p.add_argument("--event_pred_dim", type=int, default=111)
    p.add_argument("--max_feats", type=int, default=80)
    p.add_argument("--feature_dim", type=int, default=1024)
    p.add_argument("--output_dim", type=int, default=1)
    p.add_argument("--num_options", type=int, default=4)
    p.add_argument("--qa_dataset", default="star")
    p.add_argument("--task_type", default="star")

    return p.parse_args()


def combination(n, k):
    """
    Compatible con Python 3.7.
    """
    if k < 0 or k > n:
        return 0

    k = min(k, n - k)

    result = 1

    for i in range(1, k + 1):
        result = (
            result
            * (n - k + i)
            // i
        )

    return result


def mcnemar_exact(gf_correct, tcmr_correct):
    """
    n01: GF falla y TCMR acierta.
    n10: GF acierta y TCMR falla.
    """
    a = np.asarray(gf_correct, dtype=np.int64)
    b = np.asarray(tcmr_correct, dtype=np.int64)

    n01 = int(
        np.sum(
            (a == 0)
            & (b == 1)
        )
    )

    n10 = int(
        np.sum(
            (a == 1)
            & (b == 0)
        )
    )

    n = n01 + n10

    if n == 0:
        return 1.0, n01, n10

    k = min(n01, n10)

    numerator = sum(
        combination(n, i)
        for i in range(k + 1)
    )

    p_value = min(
        1.0,
        2.0
        * (
            numerator
            / float(2 ** n)
        ),
    )

    return (
        float(p_value),
        n01,
        n10,
    )


def load_models(args):
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

    print(
        "Glance-Focus cargado:",
        args.reload_model_path,
    )

    temporal_encoder = TemporalRelationEncoder(
        hidden_dim=args.hidden_dim,
        dropout=0.1,
    ).to(args.device)

    temporal_checkpoint = torch.load(
        args.temporal_checkpoint,
        map_location=args.device,
    )

    temporal_encoder.load_state_dict(
        temporal_checkpoint[
            "temporal_encoder_state_dict"
        ]
    )

    for parameter in temporal_encoder.parameters():
        parameter.requires_grad = False

    temporal_encoder.eval()

    print(
        "TCMR cargado:",
        args.temporal_checkpoint,
    )

    print(
        "Alpha temporal:",
        temporal_encoder.alpha.item(),
    )

    return model, temporal_encoder


def new_bucket():
    return {
        "gf_correct": [],
        "tcmr_correct": [],
    }


def summarize_bucket(name, bucket):
    gf = np.asarray(
        bucket["gf_correct"],
        dtype=np.int64,
    )

    tcmr = np.asarray(
        bucket["tcmr_correct"],
        dtype=np.int64,
    )

    n = len(gf)

    if n == 0:
        return None

    gf_acc = float(gf.mean())
    tcmr_acc = float(tcmr.mean())

    delta = (
        tcmr_acc
        - gf_acc
    )

    p_value, n01, n10 = mcnemar_exact(
        gf,
        tcmr,
    )

    return {
        "category": name,
        "n": n,
        "gf_correct_count": int(gf.sum()),
        "tcmr_correct_count": int(tcmr.sum()),
        "gf_accuracy": gf_acc,
        "tcmr_accuracy": tcmr_acc,
        "delta_pp": delta * 100.0,
        "gf_wrong_tcmr_correct": n01,
        "gf_correct_tcmr_wrong": n10,
        "mcnemar_p_value": p_value,
    }


def main():
    args = parse_args()

    if args.batch_size != 1:
        raise ValueError(
            "Este diagnóstico debe ejecutarse con --batch_size 1."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
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
        batch_size=1,
        shuffle=False,
        collate_fn=VideoQACollator(
            task_type=args.task_type
        ).collate_batch,
    )

    model, temporal_encoder = load_models(args)

    buckets = {
        "Overall": new_bucket(),
    }

    for category in CATEGORIES:
        buckets[category] = new_bucket()

    per_question_rows = []

    with torch.no_grad():

        pbar = tqdm(
            val_loader,
            desc="GF vs TCMR batch=1",
        )

        for batch in pbar:

            batch = prepare_batch(
                batch,
                args,
            )

            labels = batch["labels"]
            frame_features = batch[
                "visual_inputs"
            ]

            qid = batch[
                "question_ids"
            ][0]

            category = qid.split("_")[0]

            # ----------------------------------------------------
            # 1. Glancing compartido
            # ----------------------------------------------------
            outputs_event = run_glancing(
                model,
                frame_features,
                args,
            )

            memory_prompt = outputs_event[
                "memory_prompt"
            ]

            pred_spans = outputs_event[
                "pred_spans"
            ]

            (
                memory_prompt,
                pred_spans,
            ) = sort_memories_and_spans(
                memory_prompt,
                pred_spans,
            )

            # ----------------------------------------------------
            # 2. Glance-Focus original
            # ----------------------------------------------------
            gf_outputs = run_focusing(
                model,
                frame_features,
                memory_prompt,
                batch["text_str_list"],
                args,
            )

            gf_logits = gf_outputs[
                "pred_answer"
            ]

            gf_prediction = torch.argmax(
                gf_logits,
                dim=1,
            )

            # ----------------------------------------------------
            # 3. TCMR
            # ----------------------------------------------------
            temporal_memory = temporal_encoder(
                memory_prompt,
                pred_spans,
            )

            tcmr_outputs = run_focusing(
                model,
                frame_features,
                temporal_memory,
                batch["text_str_list"],
                args,
            )

            tcmr_logits = tcmr_outputs[
                "pred_answer"
            ]

            tcmr_prediction = torch.argmax(
                tcmr_logits,
                dim=1,
            )

            label = int(
                labels[0].item()
            )

            gf_pred = int(
                gf_prediction[0].item()
            )

            tcmr_pred = int(
                tcmr_prediction[0].item()
            )

            gf_correct = int(
                gf_pred == label
            )

            tcmr_correct = int(
                tcmr_pred == label
            )

            buckets["Overall"][
                "gf_correct"
            ].append(
                gf_correct
            )

            buckets["Overall"][
                "tcmr_correct"
            ].append(
                tcmr_correct
            )

            if category in buckets:

                buckets[category][
                    "gf_correct"
                ].append(
                    gf_correct
                )

                buckets[category][
                    "tcmr_correct"
                ].append(
                    tcmr_correct
                )

            per_question_rows.append(
                {
                    "question_id": qid,
                    "category": category,
                    "label": label,
                    "gf_prediction": gf_pred,
                    "tcmr_prediction": tcmr_pred,
                    "gf_correct": gf_correct,
                    "tcmr_correct": tcmr_correct,
                }
            )

    summaries = []

    for name in (
        CATEGORIES
        + ["Overall"]
    ):
        result = summarize_bucket(
            name,
            buckets[name],
        )

        if result is not None:
            summaries.append(result)

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    per_question_path = os.path.join(
        args.output_dir,
        "gf_vs_tcmr_per_question.csv",
    )

    summary_path = os.path.join(
        args.output_dir,
        "gf_vs_tcmr_summary.csv",
    )

    with open(
        per_question_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                per_question_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            per_question_rows
        )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                summaries[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            summaries
        )

    print()
    print("=" * 86)
    print(
        "COMPARACIÓN PAREADA GLANCE-FOCUS vs TCMR - BATCH SIZE 1"
    )
    print("=" * 86)

    for row in summaries:

        print()
        print(
            "{} (n={})".format(
                row["category"],
                row["n"],
            )
        )

        print(
            "  Glance-Focus : {:6.2f}%  ({}/{})".format(
                row["gf_accuracy"] * 100.0,
                row["gf_correct_count"],
                row["n"],
            )
        )

        print(
            "  TCMR         : {:6.2f}%  ({}/{})".format(
                row["tcmr_accuracy"] * 100.0,
                row["tcmr_correct_count"],
                row["n"],
            )
        )

        print(
            "  Mejora       : {:+.2f} pp".format(
                row["delta_pp"],
            )
        )

        print(
            "  GF mal -> TCMR bien : {}".format(
                row[
                    "gf_wrong_tcmr_correct"
                ]
            )
        )

        print(
            "  GF bien -> TCMR mal : {}".format(
                row[
                    "gf_correct_tcmr_wrong"
                ]
            )
        )

        print(
            "  McNemar exacto p     : {:.8g}".format(
                row[
                    "mcnemar_p_value"
                ]
            )
        )

    print()
    print("=" * 86)
    print("Archivos guardados:")
    print(per_question_path)
    print(summary_path)
    print("=" * 86)


if __name__ == "__main__":
    main()
