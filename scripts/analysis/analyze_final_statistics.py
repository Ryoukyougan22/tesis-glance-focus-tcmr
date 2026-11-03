import argparse
import csv
import math
import os

import numpy as np


# ============================================================
# ARGUMENTOS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csv",
        type=str,
        default="results/causal_importance/causal_importance_per_question.csv",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/final_statistics",
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=5000,
        help="Número de remuestreos bootstrap.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


# ============================================================
# LECTURA
# ============================================================

INT_FIELDS = {
    "label",
    "baseline_prediction",
    "baseline_correct",
    "top_event",
    "bottom_event",
    "random_event",
    "top_prediction",
    "bottom_prediction",
    "random_prediction",
    "top_correct",
    "bottom_correct",
    "random_correct",
    "top_flip",
    "bottom_flip",
    "random_flip",
}

FLOAT_FIELDS = {
    "top_abs_effect",
    "bottom_abs_effect",
    "random_abs_effect",
    "top_signed_effect",
    "bottom_signed_effect",
    "random_signed_effect",
}


def load_rows(path):
    rows = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            parsed = dict(row)

            for field in INT_FIELDS:
                parsed[field] = int(parsed[field])

            for field in FLOAT_FIELDS:
                parsed[field] = float(parsed[field])

            rows.append(parsed)

    return rows


# ============================================================
# ESTADÍSTICA
# ============================================================

def mean_ci(values, n_bootstrap, rng):
    """
    Media + intervalo de confianza bootstrap percentil al 95%.
    """
    values = np.asarray(values, dtype=np.float64)

    mean_value = float(np.mean(values))
    n = len(values)

    boot_means = np.empty(
        n_bootstrap,
        dtype=np.float64,
    )

    for i in range(n_bootstrap):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        boot_means[i] = np.mean(
            values[idx]
        )

    lower, upper = np.percentile(
        boot_means,
        [2.5, 97.5],
    )

    return (
        mean_value,
        float(lower),
        float(upper),
    )


def paired_difference_ci(a, b, n_bootstrap, rng):
    """
    Diferencia pareada a - b.
    Devuelve:
        media,
        IC95% bootstrap,
        Cohen's dz.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    diff = a - b

    mean_diff = float(
        np.mean(diff)
    )

    n = len(diff)

    boot_means = np.empty(
        n_bootstrap,
        dtype=np.float64,
    )

    for i in range(n_bootstrap):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        boot_means[i] = np.mean(
            diff[idx]
        )

    lower, upper = np.percentile(
        boot_means,
        [2.5, 97.5],
    )

    std_diff = np.std(
        diff,
        ddof=1,
    )

    if std_diff > 0:
        cohen_dz = mean_diff / std_diff
    else:
        cohen_dz = 0.0

    return (
        mean_diff,
        float(lower),
        float(upper),
        float(cohen_dz),
    )


def exact_mcnemar_p(a, b):
    """
    McNemar exacto para dos variables binarias pareadas.

    Compatible con Python 3.7: no utiliza math.comb(),
    que fue incorporado recién en Python 3.8.
    """
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)

    b01 = int(
        np.sum(
            (a == 0)
            & (b == 1)
        )
    )

    b10 = int(
        np.sum(
            (a == 1)
            & (b == 0)
        )
    )

    n = b01 + b10

    if n == 0:
        return 1.0, b01, b10

    k = min(
        b01,
        b10,
    )

    # P(X <= k), con X ~ Binomial(n, 0.5).
    # Se calcula en log-espacio usando lgamma para mantener
    # compatibilidad con Python 3.7 y evitar enteros gigantes.
    log_probs = []

    for i in range(k + 1):
        log_p = (
            math.lgamma(n + 1)
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            - n * math.log(2.0)
        )

        log_probs.append(log_p)

    max_log = max(log_probs)

    scaled_sum = sum(
        math.exp(lp - max_log)
        for lp in log_probs
    )

    log_tail = (
        max_log
        + math.log(scaled_sum)
    )

    # Si la probabilidad es extremadamente pequeña,
    # exp(log_tail) puede underflowear a 0.0, lo cual
    # es aceptable para reportar un p-value prácticamente nulo.
    tail_probability = (
        math.exp(log_tail)
        if log_tail > -745
        else 0.0
    )

    p_value = min(
        1.0,
        2.0 * tail_probability,
    )

    return (
        float(p_value),
        b01,
        b10,
    )


# ============================================================
# HELPERS
# ============================================================

def values(rows, field):
    return np.asarray(
        [
            row[field]
            for row in rows
        ],
        dtype=np.float64,
    )


def format_ci(
    mean_value,
    lower,
    upper,
    digits=6,
):
    return (
        f"{mean_value:.{digits}f} "
        f"[{lower:.{digits}f}, {upper:.{digits}f}]"
    )


def subgroup(rows, category):
    if category == "Overall":
        return rows

    return [
        row
        for row in rows
        if row["category"] == category
    ]


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    if not os.path.exists(
        args.input_csv
    ):
        raise FileNotFoundError(
            "No se encontró el archivo: "
            + args.input_csv
        )

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    rows = load_rows(
        args.input_csv
    )

    if len(rows) == 0:
        raise ValueError(
            "El CSV no contiene observaciones."
        )

    print(
        "Preguntas cargadas:",
        len(rows),
    )

    categories = [
        "Interaction",
        "Sequence",
        "Prediction",
        "Feasibility",
        "Overall",
    ]

    summary_rows = []
    test_rows = []
    report_lines = []

    report_lines.append(
        "=" * 80
    )
    report_lines.append(
        "ANÁLISIS ESTADÍSTICO FINAL - TCMR"
    )
    report_lines.append(
        "=" * 80
    )
    report_lines.append(
        f"N total: {len(rows)}"
    )
    report_lines.append(
        f"Bootstrap: {args.bootstrap}"
    )
    report_lines.append(
        f"Seed: {args.seed}"
    )
    report_lines.append("")

    for category_index, category in enumerate(
        categories
    ):
        current = subgroup(
            rows,
            category,
        )

        if len(current) == 0:
            continue

        rng = np.random.default_rng(
            args.seed + category_index
        )

        baseline_correct = values(
            current,
            "baseline_correct",
        )

        top_correct = values(
            current,
            "top_correct",
        )

        bottom_correct = values(
            current,
            "bottom_correct",
        )

        random_correct = values(
            current,
            "random_correct",
        )

        top_flip = values(
            current,
            "top_flip",
        )

        bottom_flip = values(
            current,
            "bottom_flip",
        )

        random_flip = values(
            current,
            "random_flip",
        )

        top_abs = values(
            current,
            "top_abs_effect",
        )

        bottom_abs = values(
            current,
            "bottom_abs_effect",
        )

        random_abs = values(
            current,
            "random_abs_effect",
        )

        top_signed = values(
            current,
            "top_signed_effect",
        )

        bottom_signed = values(
            current,
            "bottom_signed_effect",
        )

        random_signed = values(
            current,
            "random_signed_effect",
        )

        metric_map = {
            "baseline_accuracy":
                baseline_correct,
            "top_accuracy":
                top_correct,
            "bottom_accuracy":
                bottom_correct,
            "random_accuracy":
                random_correct,
            "top_flip_rate":
                top_flip,
            "bottom_flip_rate":
                bottom_flip,
            "random_flip_rate":
                random_flip,
            "top_abs_effect":
                top_abs,
            "bottom_abs_effect":
                bottom_abs,
            "random_abs_effect":
                random_abs,
            "top_signed_effect":
                top_signed,
            "bottom_signed_effect":
                bottom_signed,
            "random_signed_effect":
                random_signed,
        }

        for metric_name, metric_values in metric_map.items():

            mean_value, lower, upper = mean_ci(
                metric_values,
                args.bootstrap,
                rng,
            )

            summary_rows.append(
                {
                    "category":
                        category,
                    "n":
                        len(current),
                    "metric":
                        metric_name,
                    "mean":
                        mean_value,
                    "ci95_lower":
                        lower,
                    "ci95_upper":
                        upper,
                }
            )

        continuous_comparisons = [
            (
                "abs_effect_top_minus_random",
                top_abs,
                random_abs,
            ),
            (
                "abs_effect_top_minus_bottom",
                top_abs,
                bottom_abs,
            ),
            (
                "abs_effect_random_minus_bottom",
                random_abs,
                bottom_abs,
            ),
            (
                "signed_effect_top_minus_random",
                top_signed,
                random_signed,
            ),
            (
                "signed_effect_top_minus_bottom",
                top_signed,
                bottom_signed,
            ),
        ]

        for name, a, b in continuous_comparisons:

            (
                mean_diff,
                lower,
                upper,
                cohen_dz,
            ) = paired_difference_ci(
                a,
                b,
                args.bootstrap,
                rng,
            )

            test_rows.append(
                {
                    "category":
                        category,
                    "comparison":
                        name,
                    "test":
                        "paired_bootstrap",
                    "estimate":
                        mean_diff,
                    "ci95_lower":
                        lower,
                    "ci95_upper":
                        upper,
                    "effect_size":
                        cohen_dz,
                    "p_value":
                        "",
                    "discordant_01":
                        "",
                    "discordant_10":
                        "",
                }
            )

        binary_comparisons = [
            (
                "flip_top_vs_random",
                top_flip,
                random_flip,
            ),
            (
                "flip_top_vs_bottom",
                top_flip,
                bottom_flip,
            ),
            (
                "flip_random_vs_bottom",
                random_flip,
                bottom_flip,
            ),
            (
                "accuracy_baseline_vs_top",
                baseline_correct,
                top_correct,
            ),
            (
                "accuracy_baseline_vs_bottom",
                baseline_correct,
                bottom_correct,
            ),
            (
                "accuracy_baseline_vs_random",
                baseline_correct,
                random_correct,
            ),
        ]

        for name, a, b in binary_comparisons:

            p_value, b01, b10 = exact_mcnemar_p(
                a,
                b,
            )

            estimate = float(
                np.mean(a)
                - np.mean(b)
            )

            test_rows.append(
                {
                    "category":
                        category,
                    "comparison":
                        name,
                    "test":
                        "mcnemar_exact",
                    "estimate":
                        estimate,
                    "ci95_lower":
                        "",
                    "ci95_upper":
                        "",
                    "effect_size":
                        "",
                    "p_value":
                        p_value,
                    "discordant_01":
                        b01,
                    "discordant_10":
                        b10,
                }
            )

        b_acc = mean_ci(
            baseline_correct,
            args.bootstrap,
            rng,
        )

        top_acc = mean_ci(
            top_correct,
            args.bootstrap,
            rng,
        )

        bottom_acc = mean_ci(
            bottom_correct,
            args.bootstrap,
            rng,
        )

        random_acc = mean_ci(
            random_correct,
            args.bootstrap,
            rng,
        )

        top_abs_ci = mean_ci(
            top_abs,
            args.bootstrap,
            rng,
        )

        bottom_abs_ci = mean_ci(
            bottom_abs,
            args.bootstrap,
            rng,
        )

        random_abs_ci = mean_ci(
            random_abs,
            args.bootstrap,
            rng,
        )

        top_flip_ci = mean_ci(
            top_flip,
            args.bootstrap,
            rng,
        )

        bottom_flip_ci = mean_ci(
            bottom_flip,
            args.bootstrap,
            rng,
        )

        random_flip_ci = mean_ci(
            random_flip,
            args.bootstrap,
            rng,
        )

        report_lines.append(
            "-" * 80
        )
        report_lines.append(
            f"{category} (n={len(current)})"
        )
        report_lines.append(
            "-" * 80
        )

        report_lines.append(
            "Accuracy original: "
            + format_ci(
                b_acc[0] * 100,
                b_acc[1] * 100,
                b_acc[2] * 100,
                digits=2,
            )
            + "%"
        )

        report_lines.append(
            "Accuracy quitar TOP: "
            + format_ci(
                top_acc[0] * 100,
                top_acc[1] * 100,
                top_acc[2] * 100,
                digits=2,
            )
            + "%"
        )

        report_lines.append(
            "Accuracy quitar BOTTOM: "
            + format_ci(
                bottom_acc[0] * 100,
                bottom_acc[1] * 100,
                bottom_acc[2] * 100,
                digits=2,
            )
            + "%"
        )

        report_lines.append(
            "Accuracy quitar RANDOM: "
            + format_ci(
                random_acc[0] * 100,
                random_acc[1] * 100,
                random_acc[2] * 100,
                digits=2,
            )
            + "%"
        )

        report_lines.append("")

        report_lines.append(
            "|Δp| TOP: "
            + format_ci(
                *top_abs_ci,
                digits=6,
            )
        )

        report_lines.append(
            "|Δp| RANDOM: "
            + format_ci(
                *random_abs_ci,
                digits=6,
            )
        )

        report_lines.append(
            "|Δp| BOTTOM: "
            + format_ci(
                *bottom_abs_ci,
                digits=6,
            )
        )

        report_lines.append("")

        report_lines.append(
            "Flip TOP: "
            + format_ci(
                top_flip_ci[0] * 100,
                top_flip_ci[1] * 100,
                top_flip_ci[2] * 100,
                digits=2,
            )
            + "%"
        )

        report_lines.append(
            "Flip RANDOM: "
            + format_ci(
                random_flip_ci[0] * 100,
                random_flip_ci[1] * 100,
                random_flip_ci[2] * 100,
                digits=2,
            )
            + "%"
        )

        report_lines.append(
            "Flip BOTTOM: "
            + format_ci(
                bottom_flip_ci[0] * 100,
                bottom_flip_ci[1] * 100,
                bottom_flip_ci[2] * 100,
                digits=2,
            )
            + "%"
        )

        report_lines.append("")

    summary_path = os.path.join(
        args.output_dir,
        "final_statistics_summary.csv",
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "n",
                "metric",
                "mean",
                "ci95_lower",
                "ci95_upper",
            ],
        )
        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    tests_path = os.path.join(
        args.output_dir,
        "final_statistics_tests.csv",
    )

    with open(
        tests_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "comparison",
                "test",
                "estimate",
                "ci95_lower",
                "ci95_upper",
                "effect_size",
                "p_value",
                "discordant_01",
                "discordant_10",
            ],
        )
        writer.writeheader()
        writer.writerows(
            test_rows
        )

    report_lines.append(
        "=" * 80
    )
    report_lines.append(
        "NOTA METODOLÓGICA"
    )
    report_lines.append(
        "=" * 80
    )
    report_lines.append(
        "TOP y BOTTOM se seleccionan usando la magnitud del efecto "
        "intervencional calculado sobre las mismas memorias. Por ello, "
        "las diferencias de |Δp| TOP > BOTTOM son en parte consecuencia "
        "de la definición del ranking y deben interpretarse como análisis "
        "descriptivo/validación interna, no como prueba independiente de "
        "causalidad del mundo real."
    )
    report_lines.append("")
    report_lines.append(
        "Las pruebas de flip rate y accuracy cuantifican consecuencias "
        "conductuales de las intervenciones sobre la decisión del modelo. "
        "Estas intervenciones permiten hablar de influencia causal dentro "
        "del mecanismo de decisión del modelo, bajo la operación "
        "do(m_i = 0), pero no demuestran causalidad física entre eventos "
        "del video."
    )

    report_path = os.path.join(
        args.output_dir,
        "final_statistics_report.txt",
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "\n".join(
                report_lines
            )
        )

    print()
    print("=" * 80)
    print("ANÁLISIS ESTADÍSTICO TERMINADO")
    print("=" * 80)
    print("Resumen:")
    print(summary_path)
    print()
    print("Pruebas:")
    print(tests_path)
    print()
    print("Reporte:")
    print(report_path)
    print("=" * 80)


if __name__ == "__main__":
    main()