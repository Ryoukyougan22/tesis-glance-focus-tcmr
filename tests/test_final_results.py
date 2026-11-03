"""Resource-free checks of published scientific artifacts and configuration."""
import csv
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


class FinalResultsTests(unittest.TestCase):
    def test_paired_predictions_and_interventional_baseline(self):
        def rows(name):
            with (ROOT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
        o1=rows('results/gf_vs_tcmr_bs1/gf_vs_tcmr_per_question.csv')
        o2=rows('results/causal_importance_bs1/causal_importance_per_question.csv')
        self.assertEqual(len(o1),7098);self.assertEqual(len(o2),7098)
        self.assertEqual(len({r['question_id'] for r in o1}),7098)
        self.assertEqual([r['question_id'] for r in o1],[r['question_id'] for r in o2])
        self.assertEqual(sum(int(r['gf_correct']) for r in o1),3878)
        self.assertEqual(sum(int(r['tcmr_correct']) for r in o1),3963)
        for a,b in zip(o1,o2):
            self.assertEqual(a['label'],b['label']);self.assertEqual(a['tcmr_prediction'],b['baseline_prediction'])

    def test_visual_configuration_and_memory_coverage(self):
        cfg=json.loads((ROOT/'configs/final_experiment.json').read_text(encoding='utf-8'))
        root=ROOT/cfg['visual_root']
        summary=json.loads((root/'visual_analysis_summary.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['total_cases'],10);self.assertEqual(summary['total_memories'],100)
        self.assertEqual(summary['representation_stage'],'tcmr_refined')
        manifest=json.loads((root/'execution_manifest.json').read_text(encoding='utf-8'))
        self.assertTrue({cfg['gf_sha256'],cfg['tcmr_sha256']}.issubset(set(manifest['checkpoints_sha256'].values())))
        for case in summary['case_ids']:
            data=json.loads((root/case/'memory_visual_sensitivity.json').read_text(encoding='utf-8'))
            self.assertEqual(data['refinement']['checkpoint_sha256'],cfg['tcmr_sha256'])
            self.assertEqual({m['memory'] for m in data['memories']},{'E%02d'%i for i in range(1,11)})


if __name__=='__main__':unittest.main()
