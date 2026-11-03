import argparse
import csv
import json
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analiza qué regiones visuales afectan de forma ESPECÍFICA a cada memoria, "
            "comparando su sensibilidad contra las demás memorias."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Carpeta generada por inspect_memory_visual_sensitivity.py",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Número de tokens distintivos a reportar por memoria.",
    )
    return parser.parse_args()


def safe_corrcoef(matrix):
    corr = np.corrcoef(matrix)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr


def main():
    args = parse_args()

    matrix_path = os.path.join(
        args.input_dir,
        "memory_token_cosine_distance.npy",
    )
    json_path = os.path.join(
        args.input_dir,
        "memory_visual_sensitivity.json",
    )

    if not os.path.isfile(matrix_path):
        raise FileNotFoundError(matrix_path)

    if not os.path.isfile(json_path):
        raise FileNotFoundError(json_path)

    sensitivity = np.load(matrix_path).astype(np.float64)

    with open(json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if sensitivity.ndim != 2:
        raise ValueError(
            "Se esperaba matriz [memorias, tokens], se obtuvo {}".format(
                sensitivity.shape
            )
        )

    num_memories, num_tokens = sensitivity.shape

    duration = metadata.get("video_duration_seconds")
    raw_feature_length = metadata.get("raw_feature_length")
    max_feats = metadata.get("prepared_feature_length", num_tokens)

    # ----------------------------------------------------------
    # 1. SIMILITUD ENTRE PERFILES DE MEMORIA
    # ----------------------------------------------------------
    corr = safe_corrcoef(sensitivity)

    upper = corr[np.triu_indices(num_memories, k=1)]

    mean_corr = float(np.mean(upper)) if upper.size else 0.0
    median_corr = float(np.median(upper)) if upper.size else 0.0
    min_corr = float(np.min(upper)) if upper.size else 0.0
    max_corr = float(np.max(upper)) if upper.size else 0.0

    # ----------------------------------------------------------
    # 2. SELECTIVIDAD RELATIVA POR TOKEN
    # ----------------------------------------------------------
    # Para cada token visual, comparamos cuánto cambia cada memoria respecto
    # a las otras 9. Esto NO da significado semántico; solo "especificidad".
    token_mean = sensitivity.mean(axis=0, keepdims=True)
    token_std = sensitivity.std(axis=0, keepdims=True)
    token_std[token_std < 1e-12] = 1e-12

    z_specificity = (sensitivity - token_mean) / token_std

    # También calculamos una diferencia interpretable:
    # sensibilidad de Ei - media de las otras memorias.
    sum_all = sensitivity.sum(axis=0, keepdims=True)
    mean_other = (sum_all - sensitivity) / max(num_memories - 1, 1)
    excess_over_others = sensitivity - mean_other

    def token_to_time(token_idx):
        if (
            duration is None
            or raw_feature_length is None
            or raw_feature_length <= 0
        ):
            return None

        if raw_feature_length <= max_feats:
            raw_idx = token_idx
        else:
            raw_idx = (token_idx * raw_feature_length) // max_feats

        normalized = (float(raw_idx) + 0.5) / float(raw_feature_length)
        normalized = min(max(normalized, 0.0), 1.0)
        return normalized * float(duration)

    print()
    print("=" * 86)
    print("SELECTIVIDAD VISUAL ENTRE MEMORIAS")
    print("=" * 86)
    print("Shape sensibilidad:", sensitivity.shape)
    print()
    print("Correlación entre perfiles de sensibilidad:")
    print("  media   : {:.4f}".format(mean_corr))
    print("  mediana : {:.4f}".format(median_corr))
    print("  mínima  : {:.4f}".format(min_corr))
    print("  máxima  : {:.4f}".format(max_corr))
    print()
    print(
        "Lectura rápida: correlaciones cercanas a 1 indican que distintas memorias "
        "reaccionan a regiones visuales muy parecidas; valores menores indican "
        "perfiles más diferenciados."
    )
    print("=" * 86)

    rows = []
    summary = []

    top_k = min(args.top_k, num_tokens)

    for memory_idx in range(num_memories):
        memory_name = "E{:02d}".format(memory_idx + 1)

        # Priorizamos regiones donde esta memoria es MÁS sensible que las demás.
        order = np.argsort(
            -z_specificity[memory_idx]
        )[:top_k]

        print()
        print(memory_name)

        memory_items = []

        for rank, token_idx in enumerate(order, start=1):
            time_sec = token_to_time(int(token_idx))

            item = {
                "memory": memory_name,
                "rank": rank,
                "token": int(token_idx),
                "specificity_z": float(
                    z_specificity[memory_idx, token_idx]
                ),
                "sensitivity": float(
                    sensitivity[memory_idx, token_idx]
                ),
                "excess_over_other_memories": float(
                    excess_over_others[memory_idx, token_idx]
                ),
                "approx_video_time_seconds": (
                    None if time_sec is None else float(time_sec)
                ),
            }

            memory_items.append(item)
            rows.append(item)

            time_text = (
                ""
                if time_sec is None
                else " | ~{:.2f}s".format(time_sec)
            )

            print(
                "  #{} token {:02d} | especificidad z={:+.3f} | "
                "sens={:.6f} | exceso={:+.6f}{}".format(
                    rank,
                    int(token_idx),
                    item["specificity_z"],
                    item["sensitivity"],
                    item["excess_over_other_memories"],
                    time_text,
                )
            )

        summary.append(
            {
                "memory": memory_name,
                "top_distinctive_regions": memory_items,
            }
        )

    out_csv = os.path.join(
        args.input_dir,
        "memory_visual_selectivity.csv",
    )
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "memory",
                "rank",
                "token",
                "specificity_z",
                "sensitivity",
                "excess_over_other_memories",
                "approx_video_time_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    out_json = os.path.join(
        args.input_dir,
        "memory_visual_selectivity.json",
    )

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "profile_correlation": {
                    "mean": mean_corr,
                    "median": median_corr,
                    "min": min_corr,
                    "max": max_corr,
                },
                "interpretation": (
                    "The selectivity score compares each memory's response to a visual "
                    "token against the responses of the other memories. It measures "
                    "relative specificity, not semantic meaning."
                ),
                "memories": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    np.save(
        os.path.join(
            args.input_dir,
            "memory_token_specificity_z.npy",
        ),
        z_specificity,
    )

    print()
    print("=" * 86)
    print("Guardado:")
    print(out_csv)
    print(out_json)
    print(
        os.path.join(
            args.input_dir,
            "memory_token_specificity_z.npy",
        )
    )
    print("=" * 86)


if __name__ == "__main__":
    main()
