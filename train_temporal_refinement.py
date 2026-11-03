import os
import argparse

import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.star import (
    VideoQADataset,
    VideoQACollator,
    repeat_tensor_rows
)

from model.transformer_gf import build_transformer
from model.glance_focus import GF
from model.temporal_relation_encoder import TemporalRelationEncoder


MULTI_CHOICE_QA = [
    "star",
    "action",
    "transition",
    "nextqa_mc"
]


# ARGUMENTOS

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device_id",
        type=int,
        default=0
    )

    parser.add_argument(
        "--base_data_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--reload_model_path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512
    )

    parser.add_argument(
        "--num_layers",
        type=int,
        default=2
    )

    parser.add_argument(
        "--num_queries",
        type=int,
        default=10
    )

    parser.add_argument(
        "--event_pred_dim",
        type=int,
        default=111
    )

    parser.add_argument(
        "--max_feats",
        type=int,
        default=80
    )

    parser.add_argument(
        "--feature_dim",
        type=int,
        default=1024
    )

    parser.add_argument(
        "--output_dim",
        type=int,
        default=1
    )

    parser.add_argument(
        "--num_options",
        type=int,
        default=4
    )

    parser.add_argument(
        "--qa_dataset",
        type=str,
        default="star"
    )

    parser.add_argument(
        "--task_type",
        type=str,
        default="star"
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Número máximo de batches para la primera prueba. -1 usa todo."
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints/tcmr"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    return parser.parse_args()


# PREPARAR BATCH


def prepare_batch(batch, args):

    repeat_counts = [
        e * args.num_options
        for e in batch["n_examples_list"]
    ]

    del batch["n_examples_list"]

    batch["visual_inputs"] = torch.stack(
        repeat_tensor_rows(
            batch["visual_inputs"],
            repeat_counts
        )
    ).to(args.device)

    batch["span_lst"] = repeat_tensor_rows(
        batch["span_lst"],
        repeat_counts
    )

    batch["hoi_lst"] = repeat_tensor_rows(
        batch["hoi_lst"],
        repeat_counts
    )

    batch["labels"] = batch["labels"].to(
        args.device
    )

    return batch


# ORDENAR MEMORIAS Y SPANS JUNTOS

def sort_memories_and_spans(
    memory_prompt,
    pred_spans
):
    """
    Ordena memory_prompt y pred_spans usando exactamente
    los mismos índices temporales.

    pred_spans:
        [B, Q, 2]

    memory_prompt:
        [B, Q, D]
    """

    pred_centers = pred_spans[:, :, 0]

    _, indices = pred_centers.sort(
        dim=1
    )

    # Ordenar memorias

    memory_indices = indices.unsqueeze(-1).expand(
        -1,
        -1,
        memory_prompt.size(-1)
    )

    sorted_memory = torch.gather(
        memory_prompt,
        1,
        memory_indices
    )

    # Ordenar spans con los mismos índices

    span_indices = indices.unsqueeze(-1).expand(
        -1,
        -1,
        pred_spans.size(-1)
    )

    sorted_spans = torch.gather(
        pred_spans,
        1,
        span_indices
    )

    return sorted_memory, sorted_spans


# CONSTRUIR DATASET

def build_dataset(
    args,
    split
):

    if split == "train":

        data_path = os.path.join(
            args.base_data_dir,
            "txt_db",
            "train.jsonl"
        )

        is_train = True

    else:

        data_path = os.path.join(
            args.base_data_dir,
            "txt_db",
            "val.jsonl"
        )

        is_train = False

    dataset = VideoQADataset(
        args.task_type,
        data_path,
        os.path.join(
            args.base_data_dir,
            "vis_db",
            "s3d.pth"
        ),
        os.path.join(
            args.base_data_dir,
            "vis_db",
            "strID2numID.json"
        ),
        os.path.join(
            args.base_data_dir,
            "txt_db",
            "events.json"
        ),
        os.path.join(
            args.base_data_dir,
            "txt_db",
            "action_mapping.txt"
        ),
        args.max_feats,
        num_queries=args.num_queries,
        is_train=is_train,
        return_label=True
    )

    return dataset


# GLANCING STAGE

def run_glancing(
    model,
    frame_features,
    args
):
    """
    Glance-Focus está congelado.

    El Glancing Stage solamente genera:
        - memory_prompt
        - pred_spans

    No necesitamos gradientes aquí.
    """

    B, num_frames, D = frame_features.shape

    visual_attention_mask = torch.ones(
        frame_features.shape[:-1],
        dtype=torch.float,
        device=args.device
    )

    with torch.no_grad():

        memory_cache = model(
            frame_features,
            visual_attention_mask,
            None,
            encode_and_save=True,
            glance=True
        )

        outputs_event = model(
            frame_features,
            visual_attention_mask,
            None,
            encode_and_save=False,
            glance=True,
            memory_cache=memory_cache,
            query_type="event"
        )

    return outputs_event


# FOCUSING STAGE

def run_focusing(
    model,
    frame_features,
    refined_memory,
    text_input,
    args
):
    """
    Los parámetros de Glance-Focus están congelados,
    pero NO usamos torch.no_grad aquí.

    Necesitamos que el gradiente pueda viajar desde
    la respuesta hacia TemporalRelationEncoder.
    """

    B, num_frames, D = frame_features.shape

    visual_attention_mask = torch.ones(
        (
            B,
            num_frames + args.num_queries
        ),
        dtype=torch.float,
        device=args.device
    )

    combined_features = (
        frame_features,
        refined_memory
    )

    memory_cache = model(
        combined_features,
        visual_attention_mask,
        text_input,
        encode_and_save=True,
        glance=False
    )

    outputs_qa = model(
        combined_features,
        visual_attention_mask,
        text_input,
        encode_and_save=False,
        glance=False,
        memory_cache=memory_cache,
        query_type="qa"
    )

    return outputs_qa


# VALIDACIÓN

def validate(
    model,
    temporal_encoder,
    val_loader,
    args
):

    model.eval()
    temporal_encoder.eval()

    total_correct = 0
    total_examples = 0

    category_stats = {}

    with torch.no_grad():

        for batch in tqdm(
            val_loader,
            desc="Validando"
        ):

            batch = prepare_batch(
                batch,
                args
            )

            labels = batch["labels"]

            frame_features = batch[
                "visual_inputs"
            ]

            outputs_event = run_glancing(
                model,
                frame_features,
                args
            )

            memory_prompt = outputs_event[
                "memory_prompt"
            ]

            pred_spans = outputs_event[
                "pred_spans"
            ]

            memory_prompt, pred_spans = (
                sort_memories_and_spans(
                    memory_prompt,
                    pred_spans
                )
            )

            refined_memory = temporal_encoder(
                memory_prompt,
                pred_spans
            )

            outputs_qa = run_focusing(
                model,
                frame_features,
                refined_memory,
                batch["text_str_list"],
                args
            )

            logits = outputs_qa[
                "pred_answer"
            ]

            predictions = torch.argmax(
                logits,
                dim=1
            )

            total_correct += (
                predictions ==
                labels
            ).sum().item()

            total_examples += labels.numel()

            # Accuracy por categoría


            question_ids = batch[
                "question_ids"
            ]

            for qid, pred, gt in zip(
                question_ids,
                predictions.tolist(),
                labels.tolist()
            ):

                category = qid.split("_")[0]

                if category not in category_stats:

                    category_stats[
                        category
                    ] = {
                        "correct": 0,
                        "total": 0
                    }

                category_stats[
                    category
                ]["total"] += 1

                if pred == gt:

                    category_stats[
                        category
                    ]["correct"] += 1

    overall_accuracy = (
        total_correct /
        total_examples
    )

    print()
    print("=" * 60)
    print("VALIDACIÓN TCMR")
    print("=" * 60)

    category_order = [
        "Interaction",
        "Sequence",
        "Prediction",
        "Feasibility"
    ]

    for category in category_order:

        stats = category_stats.get(
            category
        )

        if stats is None:
            continue

        accuracy = (
            stats["correct"] /
            stats["total"]
        )

        print(
            "{:<12}: {:6.2f}% ({}/{})".format(
                category,
                accuracy * 100,
                stats["correct"],
                stats["total"]
            )
        )

    print("-" * 60)

    print(
        "Overall     : {:6.2f}% ({}/{})".format(
            overall_accuracy * 100,
            total_correct,
            total_examples
        )
    )

    print(
        "Alpha       : {:.6f}".format(
            temporal_encoder.alpha.item()
        )
    )

    print("=" * 60)

    temporal_encoder.train()

    return overall_accuracy


# MAIN

def main():

    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Seed:", args.seed)

    args.device = torch.device(
        "cuda:{}".format(
            args.device_id
        )
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Dispositivo:",
        args.device
    )

    # DATASETS

    print(
        "Cargando TRAIN..."
    )

    train_dataset = build_dataset(
        args,
        "train"
    )

    print(
        "Cargando VALIDATION..."
    )

    val_dataset = build_dataset(
        args,
        "val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=VideoQACollator(
            task_type=args.task_type
        ).collate_batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=VideoQACollator(
            task_type=args.task_type
        ).collate_batch
    )


    # BASELINE GLANCE-FOCUS

    print(
        "Construyendo Glance-Focus..."
    )

    transformer = build_transformer(
        args
    )

    model = GF(
        transformer,
        num_queries=args.num_queries,
        feature_dim=args.feature_dim,
        output_dim=args.output_dim,
        event_pred_dim=args.event_pred_dim,
        qa_dataset=args.qa_dataset
    ).to(
        args.device
    )

    # CARGAR CHECKPOINT OFICIAL

    print(
        "Cargando checkpoint:",
        args.reload_model_path
    )

    checkpoint = torch.load(
        args.reload_model_path,
        map_location=args.device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print(
        "Checkpoint cargado."
    )

    # CONGELAR GLANCE-FOCUS

    for parameter in model.parameters():

        parameter.requires_grad = False

    model.eval()

    print(
        "Glance-Focus congelado."
    )

    # NUESTRO MÓDULO

    temporal_encoder = (
        TemporalRelationEncoder(
            hidden_dim=args.hidden_dim,
            dropout=0.1
        )
        .to(args.device)
    )

    trainable_params = sum(
        p.numel()
        for p in temporal_encoder.parameters()
        if p.requires_grad
    )

    print(
        "Parámetros entrenables TCMR:",
        trainable_params
    )

    optimizer = optim.AdamW(
        temporal_encoder.parameters(),
        lr=args.lr
    )

    os.makedirs(
        args.save_dir,
        exist_ok=True
    )

    best_accuracy = 0.0

    verification_printed = False

    # ENTRENAMIENTO

    for epoch in range(
        args.epochs
    ):

        temporal_encoder.train()

        running_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc="Epoch {}".format(
                epoch + 1
            )
        )

        for step, batch in enumerate(
            pbar
        ):

            # Primer smoke-test:
            # solo ejecutamos N batches.
            if (
                args.max_steps > 0
                and step >= args.max_steps
            ):

                break

            batch = prepare_batch(
                batch,
                args
            )

            labels = batch[
                "labels"
            ]

            frame_features = batch[
                "visual_inputs"
            ]

            # 1. Glancing congelado

            outputs_event = run_glancing(
                model,
                frame_features,
                args
            )

            memory_prompt = outputs_event[
                "memory_prompt"
            ]

            pred_spans = outputs_event[
                "pred_spans"
            ]

            if not verification_printed:
                print("\n" + "=" * 60)
                print("VERIFICACIÓN DE INTEGRACIÓN TCMR")
                print("=" * 60)

                print("Forma de memorias generadas por Glance-Focus:",
                    tuple(memory_prompt.shape))

                print("Número de memorias por instancia:",
                    memory_prompt.size(1))

                print("Forma de pred_spans:",
                    tuple(pred_spans.shape))

            # 2. Orden temporal consistente


            memory_prompt, pred_spans = (
                sort_memories_and_spans(
                    memory_prompt,
                    pred_spans
                )
            )

            # 3. NUESTRA PROPUESTA


            refined_memory = (
                temporal_encoder(
                    memory_prompt,
                    pred_spans
                )
            )

            if not verification_printed:
                print("Forma de memorias refinadas por TCMR:",
                    tuple(refined_memory.shape))

                print("Número de memorias refinadas:",
                    refined_memory.size(1))

                print("Se conserva la dimensionalidad:",
                    memory_prompt.shape == refined_memory.shape)


            # 4. Focusing congelado


            outputs_qa = run_focusing(
                model,
                frame_features,
                refined_memory,
                batch[
                    "text_str_list"
                ],
                args
            )

            logits = outputs_qa[
                "pred_answer"
            ]

            predictions = torch.argmax(
                logits,
                dim=1
            )

            if not verification_printed:
                print("Forma de salida del modelo:",
                    tuple(logits.shape))

                print("Predicción generada:",
                    predictions[0].item())

                print("=" * 60)

                verification_printed = True


            # 5. QA Loss


            loss = F.cross_entropy(
                logits,
                labels.long()
            )

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                temporal_encoder.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            running_loss += loss.item()

            predictions = torch.argmax(
                logits,
                dim=1
            )

            batch_acc = (
                predictions ==
                labels
            ).float().mean().item()

            pbar.set_postfix(
                loss="{:.4f}".format(
                    loss.item()
                ),
                acc="{:.3f}".format(
                    batch_acc
                ),
                alpha="{:.4f}".format(
                    temporal_encoder.alpha.item()
                )
            )


        # VALIDACIÓN

        val_accuracy = validate(
            model,
            temporal_encoder,
            val_loader,
            args
        )

        # GUARDAR MEJOR MODELO

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            save_path = os.path.join(
                args.save_dir,
                "temporal_relation_encoder_seed{}_best.pt".format(
                    args.seed
                )
            )

            torch.save(
                {
                    "temporal_encoder_state_dict":
                        temporal_encoder.state_dict(),

                    "accuracy":
                        val_accuracy,

                    "epoch":
                        epoch + 1
                },
                save_path
            )

            print(
                "Mejor modelo guardado en:",
                save_path
            )

    print()
    print(
        "Entrenamiento terminado."
    )

    print(
        "Mejor accuracy:",
        "{:.2f}%".format(
            best_accuracy * 100
        )
    )



if __name__ == "__main__":

    main()