"""Rebuild only missing demo JPEGs from existing MP4s and recorded selectivity."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from export_selective_memory_frames import extract_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    config = json.loads((ROOT / 'configs/final_experiment.json').read_text(encoding='utf-8'))
    parser.add_argument('--output-root', type=Path, default=ROOT / config['visual_root'])
    args = parser.parse_args()
    library = ROOT / 'results/demo_library'
    cases = json.loads((library / 'demo_cases_10_prepared.json').read_text(encoding='utf-8'))
    created = existing = 0
    for case in cases:
        folder = case['video_id']
        video = (ROOT / case['video_path']).resolve()
        if not video.is_file():
            raise FileNotFoundError('Required MP4: ' + case['video_path'])
        with (ROOT / config['visual_root'] / folder / 'memory_visual_selectivity.csv').open(encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            rank = int(row['rank'])
            if rank > 3:
                continue
            filename = 'rank_{:02d}_token_{:03d}_{:.2f}s_z{:+.2f}.jpg'.format(
                rank, int(row['token']), float(row['approx_video_time_seconds']), float(row['specificity_z']))
            target = args.output_root / folder / 'selective_frames' / row['memory'] / filename
            if target.is_file():
                existing += 1
                continue
            if not extract_frame(str(video), float(row['approx_video_time_seconds']), str(target)):
                raise RuntimeError('Frame extraction failed: ' + str(target))
            created += 1
    print('Demo frames: {} created, {} existing; no metrics recalculated.'.format(created, existing))


if __name__ == '__main__':
    main()
