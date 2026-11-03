import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from model.transformer_gf import build_transformer
from model.glance_focus import GF
from model.temporal_relation_encoder import TemporalRelationEncoder

from train_temporal_refinement import (
    sort_memories_and_spans,
    run_glancing,
    run_focusing,
)


# ============================================================
# ARGUMENTOS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Demo final de VideoQA con TCMR + análisis intervencional "
            "sobre memorias internas del modelo."
        )
    )

    # Caso de demo
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)

    parser.add_argument("--option_a", type=str, required=True)
    parser.add_argument("--option_b", type=str, required=True)
    parser.add_argument("--option_c", type=str, required=True)
    parser.add_argument("--option_d", type=str, required=True)

    parser.add_argument(
        "--question_id",
        type=str,
        default=None,
        help="ID opcional de la pregunta STAR.",
    )

    parser.add_argument(
        "--correct_option",
        type=str,
        choices=["A", "B", "C", "D"],
        default=None,
        help="Alternativa correcta opcional para mostrar en la demo.",
    )

    # Checkpoints
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

    # Configuración del modelo
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

    # Salida
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/demo_causal_video_final",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Número de memorias más influyentes a mostrar.",
    )

    return parser.parse_args()


# ============================================================
# FEATURES
# ============================================================

def load_and_prepare_features(
    feature_path,
    max_feats,
    feature_dim,
):
    """
    Replica el preprocesamiento de features visuales usado en STAR:
    - entrada [T, D]
    - padding si T < max_feats
    - muestreo uniforme si T >= max_feats
    """
    features = np.load(feature_path)
    features = np.asarray(features)

    if (
        features.ndim == 3
        and features.shape[0] == 1
    ):
        features = features[0]

    if features.ndim != 2:
        raise ValueError(
            "Se esperaban features [T, D], pero se obtuvo shape {}.".format(
                features.shape
            )
        )

    if features.shape[1] != feature_dim:
        raise ValueError(
            "La dimensión visual debe ser {}, pero se obtuvo {}.".format(
                feature_dim,
                features.shape[1],
            )
        )

    features = torch.from_numpy(
        features
    ).float()

    if len(features) < max_feats:

        padding = torch.zeros(
            max_feats - len(features),
            features.shape[1],
            dtype=features.dtype,
        )

        features = torch.cat(
            [
                features,
                padding,
            ],
            dim=0,
        )

    else:

        sampled = []

        for j in range(max_feats):
            sampled.append(
                features[
                    (j * len(features))
                    // max_feats
                ]
            )

        features = torch.stack(
            sampled,
            dim=0,
        )

    return features


# ============================================================
# MEMORIAS POR PREGUNTA
# ============================================================

def collapse_option_memories(
    memory_bank,
    num_options,
):
    """
    Glance-Focus representa cada alternativa como una fila.
    Para la intervención usamos una única copia de las memorias
    de la pregunta, ya que el glancing es visual.
    """
    total_rows, num_memories, hidden_dim = (
        memory_bank.shape
    )

    if total_rows % num_options != 0:
        raise ValueError(
            "El número de filas de memoria no es divisible "
            "entre el número de opciones."
        )

    num_questions = (
        total_rows
        // num_options
    )

    memory_bank = memory_bank.reshape(
        num_questions,
        num_options,
        num_memories,
        hidden_dim,
    )

    return memory_bank[
        :,
        0,
        :,
        :
    ]


def expand_question_memories(
    memory_bank,
    num_options,
):
    return memory_bank.repeat_interleave(
        num_options,
        dim=0,
    )


# ============================================================
# CARGA DE MODELOS
# ============================================================

def load_models(args):
    device = args.device

    print()
    print("Cargando Glance-Focus...")

    transformer = build_transformer(
        args
    )

    model = GF(
        transformer,
        num_queries=args.num_queries,
        feature_dim=args.feature_dim,
        output_dim=args.output_dim,
        event_pred_dim=args.event_pred_dim,
        qa_dataset=args.qa_dataset,
    ).to(device)

    checkpoint = torch.load(
        args.reload_model_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()

    print(
        "Glance-Focus:",
        args.reload_model_path,
    )

    temporal_encoder = TemporalRelationEncoder(
        hidden_dim=args.hidden_dim,
        dropout=0.1,
    ).to(device)

    temporal_checkpoint = torch.load(
        args.temporal_checkpoint,
        map_location=device,
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
        "TCMR:",
        args.temporal_checkpoint,
    )

    print(
        "Alpha temporal:",
        temporal_encoder.alpha.item(),
    )

    return (
        model,
        temporal_encoder,
    )


# ============================================================
# UTILIDADES
# ============================================================

def letter_to_index(letter):
    return {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
    }[letter]


def yes_no(value):
    return "SÍ" if value else "NO"


# ============================================================
# INFERENCIA
# ============================================================

def main():
    args = parse_args()

    if not os.path.isfile(args.video):
        raise FileNotFoundError(
            "No existe el video: {}".format(
                args.video
            )
        )

    if not os.path.isfile(args.features):
        raise FileNotFoundError(
            "No existe el archivo de features: {}".format(
                args.features
            )
        )

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    args.device = torch.device(
        "cuda:{}".format(args.device_id)
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Dispositivo:", args.device)
    print(
        "Video de referencia:",
        args.video,
    )
    print(
        "Features usadas por el modelo:",
        args.features,
    )

    # ========================================================
    # 1. FEATURES VISUALES
    # ========================================================

    visual_single = load_and_prepare_features(
        args.features,
        args.max_feats,
        args.feature_dim,
    )

    # Una pregunta STAR tiene cuatro alternativas.
    # Se replica la misma representación visual para cada opción.
    frame_features = (
        visual_single
        .unsqueeze(0)
        .repeat(
            args.num_options,
            1,
            1,
        )
        .to(args.device)
    )

    # ========================================================
    # 2. TEXTO
    # ========================================================

    options = [
        args.option_a,
        args.option_b,
        args.option_c,
        args.option_d,
    ]

    letters = [
        "A",
        "B",
        "C",
        "D",
    ]

    text_str_list = [
        args.question
        + " "
        + option
        for option in options
    ]

    # ========================================================
    # 3. MODELOS
    # ========================================================

    (
        model,
        temporal_encoder,
    ) = load_models(args)

    # ========================================================
    # 4. GLANCING + TCMR
    # ========================================================

    with torch.no_grad():

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

        temporal_memory = temporal_encoder(
            memory_prompt,
            pred_spans,
        )

        # ====================================================
        # 5. RESPUESTA TCMR ORIGINAL
        # ====================================================

        qa_outputs = run_focusing(
            model,
            frame_features,
            temporal_memory,
            text_str_list,
            args,
        )

        logits = qa_outputs[
            "pred_answer"
        ]

        probs = torch.softmax(
            logits,
            dim=1,
        )

        predicted_idx = int(
            torch.argmax(
                probs,
                dim=1,
            )[0].item()
        )

        predicted_prob = float(
            probs[
                0,
                predicted_idx
            ].item()
        )

        # ====================================================
        # 6. MEMORIA ÚNICA DE LA PREGUNTA
        # ====================================================

        question_memory = collapse_option_memories(
            temporal_memory,
            args.num_options,
        )

        num_memories = (
            question_memory.shape[1]
        )

        # ====================================================
        # 7. INTERVENCIONES do(m_i = 0)
        # ====================================================

        effects = []

        for memory_idx in range(
            num_memories
        ):

            intervened_memory = (
                question_memory.clone()
            )

            intervened_memory[
                :,
                memory_idx,
                :
            ] = 0.0

            expanded_memory = (
                expand_question_memories(
                    intervened_memory,
                    args.num_options,
                )
            )

            intervened_outputs = run_focusing(
                model,
                frame_features,
                expanded_memory,
                text_str_list,
                args,
            )

            intervened_logits = (
                intervened_outputs[
                    "pred_answer"
                ]
            )

            intervened_probs = torch.softmax(
                intervened_logits,
                dim=1,
            )

            original_answer_prob_after = float(
                intervened_probs[
                    0,
                    predicted_idx
                ].item()
            )

            signed_effect = (
                predicted_prob
                - original_answer_prob_after
            )

            intervened_pred_idx = int(
                torch.argmax(
                    intervened_probs,
                    dim=1,
                )[0].item()
            )

            intervened_pred_prob = float(
                intervened_probs[
                    0,
                    intervened_pred_idx
                ].item()
            )

            effects.append(
                {
                    "memory_index":
                        memory_idx + 1,
                    "original_answer_probability_after_intervention":
                        original_answer_prob_after,
                    "signed_effect":
                        signed_effect,
                    "absolute_effect":
                        abs(signed_effect),
                    "intervened_prediction_index":
                        intervened_pred_idx,
                    "intervened_prediction_letter":
                        letters[
                            intervened_pred_idx
                        ],
                    "intervened_prediction_answer":
                        options[
                            intervened_pred_idx
                        ],
                    "intervened_prediction_probability":
                        intervened_pred_prob,
                    "answer_flip":
                        bool(
                            intervened_pred_idx
                            != predicted_idx
                        ),
                }
            )

    # ========================================================
    # 8. RANKING DE INFLUENCIA
    # ========================================================

    ranked_effects = sorted(
        effects,
        key=lambda item: item[
            "absolute_effect"
        ],
        reverse=True,
    )

    top_k = min(
        args.top_k,
        len(ranked_effects),
    )

    displayed_effects = (
        ranked_effects[
            :top_k
        ]
    )

    top_effect = (
        ranked_effects[0]
    )

    # ========================================================
    # 9. SALIDA LIMPIA PARA SUSTENTACIÓN
    # ========================================================

    print()
    print("=" * 78)
    print("DEMO FINAL - VIDEOQA + TCMR + INTERVENCIÓN CAUSAL INTERNA")
    print("=" * 78)

    if args.question_id is not None:
        print(
            "Caso STAR:",
            args.question_id,
        )

    print(
        "Video:",
        Path(args.video).name,
    )

    print()
    print(
        "Pregunta:",
        args.question,
    )

    for i, option in enumerate(
        options
    ):
        suffix = ""

        if (
            args.correct_option is not None
            and letters[i]
            == args.correct_option
        ):
            suffix = "  [CORRECTA]"

        print(
            "{}. {}{}".format(
                letters[i],
                option,
                suffix,
            )
        )

    print()
    print("-" * 78)
    print("RESPUESTA DEL MODELO CON TCMR")
    print("-" * 78)

    print(
        "Predicción: {}. {}".format(
            letters[predicted_idx],
            options[predicted_idx],
        )
    )

    print(
        "Confianza: {:.2f}%".format(
            predicted_prob
            * 100.0
        )
    )

    if args.correct_option is not None:

        correct_idx = letter_to_index(
            args.correct_option
        )

        print(
            "¿Predicción correcta?: {}".format(
                yes_no(
                    predicted_idx
                    == correct_idx
                )
            )
        )

    print()
    print("-" * 78)
    print("ANÁLISIS INTERVENCIONAL SOBRE MEMORIAS")
    print("-" * 78)

    print(
        "Probabilidad original P({} | M) = {:.6f}".format(
            letters[predicted_idx],
            predicted_prob,
        )
    )

    print(
        "Ranking definido por |Δp| al aplicar do(Ei = 0)."
    )

    print()

    for rank, item in enumerate(
        displayed_effects,
        start=1,
    ):

        memory_name = "E{:02d}".format(
            item["memory_index"]
        )

        print(
            "#{}  Memoria {}".format(
                rank,
                memory_name,
            )
        )

        print(
            "    P({} | do({}=0)) = {:.6f}".format(
                letters[predicted_idx],
                memory_name,
                item[
                    "original_answer_probability_after_intervention"
                ],
            )
        )

        print(
            "    Δp = {:+.6f}   |Δp| = {:.6f}".format(
                item[
                    "signed_effect"
                ],
                item[
                    "absolute_effect"
                ],
            )
        )

        print(
            "    Nueva predicción: {}. {} ({:.2f}%)".format(
                item[
                    "intervened_prediction_letter"
                ],
                item[
                    "intervened_prediction_answer"
                ],
                item[
                    "intervened_prediction_probability"
                ]
                * 100.0,
            )
        )

        print(
            "    ¿Cambia la respuesta?: {}".format(
                yes_no(
                    item[
                        "answer_flip"
                    ]
                )
            )
        )

    top_name = "E{:02d}".format(
        top_effect[
            "memory_index"
        ]
    )

    print()
    print("-" * 78)
    print("RESULTADO PRINCIPAL DE LA INTERVENCIÓN")
    print("-" * 78)

    print(
        "Memoria de mayor influencia:",
        top_name,
    )

    print(
        "Predicción original: {}. {}".format(
            letters[predicted_idx],
            options[predicted_idx],
        )
    )

    print(
        "Tras do({}=0): {}. {}".format(
            top_name,
            top_effect[
                "intervened_prediction_letter"
            ],
            top_effect[
                "intervened_prediction_answer"
            ],
        )
    )

    if top_effect[
        "answer_flip"
    ]:

        print(
            "Cambio de respuesta: SÍ ({} -> {})".format(
                letters[predicted_idx],
                top_effect[
                    "intervened_prediction_letter"
                ],
            )
        )

    else:

        print(
            "Cambio de respuesta: NO"
        )

    print(
        "Efecto absoluto |Δp|: {:.6f}".format(
            top_effect[
                "absolute_effect"
            ]
        )
    )

    print()
    print("-" * 78)
    print("NOTA METODOLÓGICA")
    print("-" * 78)

    print(
        "E01...E10 son memorias internas aprendidas por el modelo."
    )

    print(
        "En esta demo NO se asignan directamente a intervalos físicos "
        "del video, porque la semántica temporal de los spans del código "
        "base no es suficientemente consistente para sostener esa "
        "interpretación."
    )

    print(
        "La operación do(Ei=0) mide influencia intervencional interna "
        "sobre la decisión del modelo; no demuestra causalidad física "
        "entre eventos del mundo real."
    )

    print(
        "El MP4 se usa como referencia visual y el modelo consume las "
        "features precomputadas proporcionadas mediante --features."
    )

    print("=" * 78)

    # ========================================================
    # 10. GUARDAR JSON
    # ========================================================

    result = {
        "question_id":
            args.question_id,
        "video_reference":
            os.path.abspath(
                args.video
            ),
        "feature_path":
            os.path.abspath(
                args.features
            ),
        "feature_input_note":
            (
                "The model consumes the precomputed feature file; "
                "the MP4 is the corresponding visual reference."
            ),
        "question":
            args.question,
        "options":
            options,
        "correct_option":
            args.correct_option,
        "predicted_index":
            predicted_idx,
        "predicted_letter":
            letters[
                predicted_idx
            ],
        "predicted_answer":
            options[
                predicted_idx
            ],
        "predicted_probability":
            predicted_prob,
        "prediction_correct":
            (
                None
                if args.correct_option is None
                else bool(
                    predicted_idx
                    == letter_to_index(
                        args.correct_option
                    )
                )
            ),
        "ranking_definition":
            "Descending absolute interventional effect |Δp| under do(Ei=0).",
        "memory_effects":
            ranked_effects,
        "top_memory":
            top_effect,
        "methodological_note":
            (
                "E01...E10 are learned internal memories. This final demo "
                "does not map them directly to physical video intervals. "
                "do(Ei=0) measures model-internal interventional influence "
                "and does not establish physical causality in the real world."
            ),
    }

    output_json = os.path.join(
        args.output_dir,
        "{}_causal_final.json".format(
            Path(
                args.video
            ).stem
        ),
    )

    with open(
        output_json,
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
    print(
        "Resultado guardado en:",
        output_json,
    )


if __name__ == "__main__":
    main()
