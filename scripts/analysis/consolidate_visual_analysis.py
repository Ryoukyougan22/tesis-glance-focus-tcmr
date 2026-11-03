"""Consolida resultados existentes sin modificar ni sobrescribir archivos.

Uso: python scripts/analysis/consolidate_visual_analysis.py [--root DIRECTORIO]
Solo biblioteca estándar. Los NA se conservan como texto también en JSON.
Los máximos se buscan por valor, no por orden de las regiones exportadas.
"""

import argparse
import csv
import io
import json
import math
import sys
from pathlib import Path
from statistics import mean


NA = "NA"
INPUTS = (
    "memory_visual_sensitivity.json", "memory_visual_selectivity.json",
    "top_visual_regions.csv", "memory_visual_selectivity.csv",
)
CASE_FIELDS = """case_id video_duration_seconds valid_visual_tokens num_memories
profile_correlation_mean profile_correlation_median profile_correlation_min
profile_correlation_max top5_concentration_mean top5_concentration_min
top5_concentration_max memory_with_highest_specificity max_specificity_z
token_of_max_specificity approx_time_of_max_specificity
num_memories_positive_max_z num_memories_negative_max_z""".split()
MEMORY_FIELDS = """case_id memory top_k_concentration most_sensitive_token
max_cosine_distance sensitive_time_seconds most_selective_token max_specificity_z
selective_time_seconds excess_over_other_memories""".split()


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else NA
    except (TypeError, ValueError):
        return NA


def aggregate(values, operation):
    valid = [v for v in values if number(v) != NA]
    return operation(valid) if valid else NA


def fmt(value):
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def load(path, warnings):
    if not path.is_file():
        warnings.append(f"Archivo faltante: {path}")
        return [] if path.suffix == ".csv" else {}
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            data = list(csv.DictReader(stream)) if path.suffix == ".csv" else json.load(stream)
        if path.suffix == ".json" and not isinstance(data, dict):
            raise ValueError("Se esperaba un objeto JSON")
        return data
    except (OSError, ValueError, csv.Error) as error:
        warnings.append(f"No se pudo leer {path}: {error}")
        return [] if path.suffix == ".csv" else {}


def memories(data, label, warnings):
    result = {}
    entries = data.get("memories", [])
    if not isinstance(entries, list):
        warnings.append(f"{label}: campo memories inválido")
        return result
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("memory"):
            warnings.append(f"{label}: registro sin identificador de memoria")
            continue
        key = str(entry["memory"])
        if key in result:
            warnings.append(f"{label}: memoria duplicada {key}; se conserva la primera")
        else:
            result[key] = entry
    return result


def regions(entry, field, rows, memory, token_field):
    # JSON tiene prioridad; CSV completa campos o regiones que no estén en JSON.
    merged = {}
    items = entry.get(field, [])
    if not isinstance(items, list):
        items = []
    candidates = [r for r in rows if r.get("memory") == memory] + items
    for index, row in enumerate(candidates):
        if not isinstance(row, dict):
            continue
        token = number(row.get(token_field))
        key = token if token != NA else ("missing", index)
        target = merged.setdefault(key, {})
        for name, value in row.items():
            if value is not None and value != "" and value != NA:
                target[name] = value
    return list(merged.values())


def best_region(rows, metric):
    valid = [r for r in rows if number(r.get(metric)) != NA]
    return max(valid, key=lambda r: number(r[metric])) if valid else {}


def summarize_case(folder, root, warnings):
    case_id = folder.relative_to(root).as_posix()
    sens, sel, sens_csv, sel_csv = [load(folder / name, warnings) for name in INPUTS]
    sm = memories(sens, f"{case_id}/sensibilidad", warnings)
    zm = memories(sel, f"{case_id}/selectividad", warnings)
    ids = set(sm) | set(zm)
    for row in sens_csv + sel_csv:
        if row.get("memory"):
            ids.add(str(row["memory"]))
        else:
            warnings.append(f"{case_id}: fila CSV sin identificador de memoria")
    expected = {f"E{i:02d}" for i in range(1, 11)}
    if ids != expected:
        warnings.append(f"{case_id}: memorias faltantes {sorted(expected - ids)}; "
                        f"memorias inesperadas {sorted(ids - expected)}")
    rows = []
    for memory in sorted(ids):
        absolute = best_region(regions(sm.get(memory, {}), "top_regions", sens_csv,
                                       memory, "center_visual_token"), "cosine_distance")
        relative = best_region(regions(zm.get(memory, {}), "top_distinctive_regions",
                                       sel_csv, memory, "token"), "specificity_z")
        row = dict(zip(MEMORY_FIELDS, [
            case_id, memory, number(sm.get(memory, {}).get("top_k_concentration")),
            number(absolute.get("center_visual_token")), number(absolute.get("cosine_distance")),
            number(absolute.get("approx_video_time_seconds")), number(relative.get("token")),
            number(relative.get("specificity_z")), number(relative.get("approx_video_time_seconds")),
            number(relative.get("excess_over_other_memories")),
        ]))
        rows.append(row)
        missing = [k for k, v in row.items() if v == NA]
        if missing:
            warnings.append(f"{case_id}/{memory}: campos NA: {', '.join(missing)}")
    correlation = sel.get("profile_correlation", {})
    if not isinstance(correlation, dict):
        correlation = {}
    concentrations = [r["top_k_concentration"] for r in rows]
    best = best_region(rows, "max_specificity_z")
    maxima = [r["max_specificity_z"] for r in rows if r["max_specificity_z"] != NA]
    summary = dict(zip(CASE_FIELDS, [
        case_id, number(sens.get("video_duration_seconds")),
        number(sens.get("valid_visual_tokens")), len(rows),
        *[number(correlation.get(k)) for k in ("mean", "median", "min", "max")],
        *[aggregate(concentrations, op) for op in (mean, min, max)],
        best.get("memory", NA), best.get("max_specificity_z", NA),
        best.get("most_selective_token", NA), best.get("selective_time_seconds", NA),
        sum(z > 0 for z in maxima) if maxima else NA,
        sum(z <= 0 for z in maxima) if maxima else NA,
    ]))
    missing = [k for k, v in summary.items() if v == NA]
    if missing:
        warnings.append(f"{case_id}: campos de resumen NA: {', '.join(missing)}")
    return summary, rows


def csv_text(fields, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2] /
                        "results" / "memory_visual_sensitivity")
    parser.add_argument("--show-summary", action="store_true",
                        help="Muestra el resumen ya generado sin escribir archivos")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"No existe la carpeta: {root}")
    if args.show_summary:
        report = root / "visual_analysis_thesis_summary.txt"
        if not report.is_file():
            parser.error(f"No existe el resumen: {report}")
        print(report.read_text(encoding="utf-8"))
        return
    names = ["visual_analysis_summary.csv", "visual_memory_summary.csv",
             "visual_analysis_summary.json", "visual_analysis_thesis_summary.txt"]
    existing = [str(root / name) for name in names if (root / name).exists()]
    if existing:
        parser.error("No se sobrescriben archivos existentes: " + ", ".join(existing))
    # Las carpetas inmediatas también permiten advertir sobre casos sin archivos.
    folders = {p for p in root.iterdir() if p.is_dir()}
    for name in INPUTS:
        folders.update(p.parent for p in root.rglob(name) if p.parent != root)
    # Un contenedor con casos anidados no es un caso adicional.
    folders = {p for p in folders if any((p / n).is_file() for n in INPUTS)
               or not any(p in other.parents for other in folders)}
    warnings, cases, rows = [], [], []
    incomplete = []
    for folder in sorted(folders):
        before = len(warnings)
        case, memory_rows = summarize_case(folder, root, warnings)
        cases.append(case)
        rows.extend(memory_rows)
        if len(warnings) > before:
            incomplete.append(case["case_id"])
    stages = set()
    for folder in folders:
        source = folder / "memory_visual_sensitivity.json"
        if source.is_file():
            metadata = load(source, warnings)
            stages.add(metadata.get("representation_stage", "glancing_raw"))
    if len(stages) > 1:
        parser.error("No se consolidan representaciones de etapas distintas: " + str(sorted(stages)))
    stage = next(iter(stages), "NA")
    if len(cases) != 10:
        warnings.append(f"Se esperaban 10 casos; se encontraron {len(cases)} en {root}.")
    if len(rows) != 100:
        warnings.append(f"Se esperaban 100 memorias; se encontraron {len(rows)}.")
    maxima = [r["max_specificity_z"] for r in rows if r["max_specificity_z"] != NA]
    best = best_region(rows, "max_specificity_z")
    concentration = [r["top_k_concentration"] for r in rows]
    stats = {
        "profile_correlation_mean": aggregate([c["profile_correlation_mean"] for c in cases], mean),
        "profile_correlation_min": aggregate([c["profile_correlation_min"] for c in cases], min),
        "profile_correlation_max": aggregate([c["profile_correlation_max"] for c in cases], max),
        "top5_concentration_mean": aggregate(concentration, mean),
        "top5_concentration_min": aggregate(concentration, min),
        "top5_concentration_max": aggregate(concentration, max),
        "max_specificity_z": best.get("max_specificity_z", NA),
        "case_with_highest_specificity": best.get("case_id", NA),
        "memory_with_highest_specificity": best.get("memory", NA),
        "num_memories_positive_max_z": sum(z > 0 for z in maxima) if maxima else NA,
        "num_memories_negative_max_z": sum(z <= 0 for z in maxima) if maxima else NA,
        "num_memories_unknown_max_z": len(rows) - len(maxima),
        "num_memories_with_concentration": sum(v != NA for v in concentration),
        "num_cases_with_correlation_mean": sum(c["profile_correlation_mean"] != NA for c in cases),
    }
    notes = [
        ("Las métricas corresponden a memorias internas refinadas mediante TCMR, "
         "no a las memorias crudas de Glancing. Se recalculan memorias y spans tras cada "
         "perturbación, preservando la identidad de las queries del baseline; "
         "el promedio de copias por opción se realiza después del refinamiento."
         if stage == "tcmr_refined" else
         "Etapa de representación: " + stage + "."),
        "La correlación global es la media no ponderada de las medias por caso disponibles; "
        "su rango usa los mínimos y máximos intracasos, no correlaciones entre casos.",
        "La concentración se resume sobre top_k_concentration de las memorias disponibles; "
        "se usa como concentración top-5 conforme a los resultados exportados.",
        "Los máximos se obtienen de todas las regiones exportadas en JSON y CSV. "
        "Los exportadores ordenan por sensibilidad o Z descendente, por lo que incluyen el máximo; "
        "no se recalculan matrices ni se infieren tiempos ausentes.",
        "NA indica un dato no disponible; los promedios y rangos excluyen NA. "
        "Los conteos por signo solo incluyen máximos conocidos. No se crean memorias ausentes.",
        "Se describe sensibilidad absoluta y selectividad relativa de representaciones internas. "
        "No se establece correspondencia semántica exclusiva ni causalidad física. "
        "Los tiempos son aproximaciones de los archivos de origen.",
    ]
    lines = [
        "CONSOLIDACIÓN PARA LA SECCIÓN 6.2.1.2", f"Carpeta: {root}",
        f"Casos encontrados: {len(cases)}", f"Memorias evaluadas: {len(rows)}",
        f"¿Se encontraron 10 casos?: {'SÍ' if len(cases) == 10 else 'NO'}",
        f"¿Se evaluaron 100 memorias?: {'SÍ' if len(rows) == 100 else 'NO'}", "",
        "Correlación entre perfiles: promedio " + fmt(stats["profile_correlation_mean"]) +
        "; rango " + fmt(stats["profile_correlation_min"]) + " a " + fmt(stats["profile_correlation_max"]),
        "Concentración top-5: promedio " + fmt(stats["top5_concentration_mean"]) +
        "; rango " + fmt(stats["top5_concentration_min"]) + " a " + fmt(stats["top5_concentration_max"]),
        f"Mayor selectividad: caso {best.get('case_id', NA)}, memoria {best.get('memory', NA)}, "
        f"Z={fmt(best.get('max_specificity_z', NA))}, token={fmt(best.get('most_selective_token', NA))}, "
        f"tiempo≈{fmt(best.get('selective_time_seconds', NA))} s.",
        f"Memorias con máximo Z positivo (>0): {stats['num_memories_positive_max_z']}",
        f"Memorias con máximo Z no positivo (<=0, incluye cero): {stats['num_memories_negative_max_z']}",
        f"Memorias con máximo Z desconocido: {stats['num_memories_unknown_max_z']}", "",
        "TABLA POR CASO (solo casos encontrados)",
        "Caso | Memorias | Corr. media | Corr. mín. | Corr. máx. | Top5 media | Top5 mín. | Top5 máx. | Memoria máx. Z | Máx. Z | Z>0 | Z<=0",
    ]
    table_fields = ["case_id", "num_memories", "profile_correlation_mean", "profile_correlation_min",
                    "profile_correlation_max", "top5_concentration_mean", "top5_concentration_min",
                    "top5_concentration_max", "memory_with_highest_specificity", "max_specificity_z",
                    "num_memories_positive_max_z", "num_memories_negative_max_z"]
    lines.extend(" | ".join(fmt(c[k]) for k in table_fields) for c in cases)
    comparable = [r for r in rows if r["most_sensitive_token"] != NA and r["most_selective_token"] != NA]
    different = sum(r["most_sensitive_token"] != r["most_selective_token"] for r in comparable)
    lines.extend(["", "OBSERVACIONES PARA LA REDACCIÓN",
                  f"En {different} de {len(comparable)} memorias comparables, el token más sensible "
                  "difiere del más selectivo: ambas medidas describen propiedades distintas."])
    for c in cases:
        if c["profile_correlation_min"] != NA and c["profile_correlation_min"] >= 0.9:
            lines.append(f"{c['case_id']}: todas las correlaciones reportadas son >=0.9; "
                         "los perfiles de sensibilidad son similares entre memorias.")
    lines.extend(notes)
    lines.extend(["", "VALIDACIONES Y ADVERTENCIAS",
                  "Casos incompletos o con incidencias: " + (", ".join(incomplete) or "ninguno")])
    lines.extend(warnings or ["Sin advertencias."])
    thesis = "\n".join(lines) + "\n"
    document = {"total_cases": len(cases), "total_memories": len(rows),
                "representation_stage": stage,
                "case_ids": [c["case_id"] for c in cases], "case_summary": cases,
                "memory_summary": rows, "global_statistics": stats,
                "incomplete_cases": incomplete, "warnings": warnings, "methodology": notes}
    contents = [csv_text(CASE_FIELDS, cases), csv_text(MEMORY_FIELDS, rows),
                json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n", thesis]
    for name, content in zip(names, contents):
        with (root / name).open("x", encoding="utf-8", newline="") as stream:
            stream.write(content)
    print(thesis)
    print("Archivos generados:")
    for name in names:
        print(root / name)


if __name__ == "__main__":
    main()
