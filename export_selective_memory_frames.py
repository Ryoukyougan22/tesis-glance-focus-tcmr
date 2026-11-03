import argparse
import csv
import os

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extrae frames de las regiones visualmente distintivas por memoria."
    )
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--selectivity_csv", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/memory_visual_sensitivity/RKGG5/selective_frames",
    )
    return parser.parse_args()


def extract_frame(video_path, time_seconds, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el video: {}".format(video_path))

    cap.set(cv2.CAP_PROP_POS_MSEC, float(time_seconds) * 1000.0)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return bool(cv2.imwrite(output_path, frame))


def main():
    args = parse_args()

    if not os.path.isfile(args.video):
        raise FileNotFoundError(args.video)

    if not os.path.isfile(args.selectivity_csv):
        raise FileNotFoundError(args.selectivity_csv)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.selectivity_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    selected = {}

    for row in rows:
        memory = row["memory"]
        rank = int(row["rank"])

        if rank > args.top_k:
            continue

        time_text = row.get("approx_video_time_seconds", "")
        if not time_text:
            continue

        time_seconds = float(time_text)
        token = int(row["token"])
        z = float(row["specificity_z"])

        memory_dir = os.path.join(args.output_dir, memory)
        filename = "rank_{:02d}_token_{:03d}_{:.2f}s_z{:+.2f}.jpg".format(
            rank,
            token,
            time_seconds,
            z,
        )
        output_path = os.path.join(memory_dir, filename)

        ok = extract_frame(
            args.video,
            time_seconds,
            output_path,
        )

        selected.setdefault(memory, []).append(
            {
                "rank": rank,
                "time_seconds": time_seconds,
                "specificity_z": z,
                "path": output_path,
                "saved": ok,
            }
        )

    print()
    print("=" * 78)
    print("FRAMES DISTINTIVOS POR MEMORIA")
    print("=" * 78)

    for memory in sorted(selected):
        print()
        print(memory)
        for item in sorted(selected[memory], key=lambda x: x["rank"]):
            print(
                "  #{} ~{:.2f}s | z={:+.2f} | {}".format(
                    item["rank"],
                    item["time_seconds"],
                    item["specificity_z"],
                    item["path"],
                )
            )

    print()
    print("=" * 78)
    print("Guardado en:")
    print(args.output_dir)
    print("=" * 78)


if __name__ == "__main__":
    main()
