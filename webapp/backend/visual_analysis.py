"""Read-only adapter for the definitive library and offline visual experiments.

No model calls, visual perturbation or feature transformations belong here.
Image requests resolve only to an allowlist assembled from the experiment files.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / 'configs/final_experiment.json').read_text(encoding='utf-8'))
LIBRARY = ROOT / 'results/demo_library'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


class VisualAnalysisStore:
    def __init__(self, library=LIBRARY):
        self.library = library.resolve()
        self.visual_root = ROOT / CONFIG['visual_root']
        final_manifest = read_json(self.visual_root / 'execution_manifest.json')
        hashes = set(final_manifest['checkpoints_sha256'].values())
        if not {CONFIG['gf_sha256'], CONFIG['tcmr_sha256']}.issubset(hashes):
            raise ValueError('La serie visual no corresponde a los checkpoints finales.')
        prepared = read_json(self.library / 'demo_cases_10_prepared.json')
        validated = {r['case']['question_id']: r for r in read_json(self.library / 'validation_10/validation_10_detailed.json')}
        self.cases, self.groundings, self.frames, self.errors = {}, {}, {}, {}
        for record in sorted(prepared, key=lambda r: r['demo_order']):
            qid = record['question_id']
            if qid in self.cases:
                raise ValueError('Duplicate library question ID: ' + qid)
            validation = validated[qid]
            if not validation['validation']['case_validation_ok']:
                raise ValueError('Case has not passed offline validation: ' + qid)
            runtime = validation['runtime_result']
            effects = sorted(runtime['effects'], key=lambda e: e['absolute_effect'], reverse=True)
            top = effects[0]
            reference = dict(source='precomputed_validation', top_memory=top['memory_name'],
                             probabilities=runtime['probabilities'],
                             top_abs_effect=top['absolute_effect'], top_flip=bool(top['flip']),
                             prediction=runtime['prediction_letter'], after=top['prediction_after_letter'])
            self.cases[qid] = dict(id=qid, question_id=qid, video_id=record['video_id'],
                                   category=record['category'], question=record['question'],
                                   options=[record['option_' + letter] for letter in 'abcd'],
                                   correct_option=record['answer_letter'], video=record['video_path'],
                                   features=record['feature_path'], reference=reference,
                                   random_memory=record['random_event'] + 1)
            try:
                self._load_grounding(qid, runtime, effects)
            except (OSError, ValueError, KeyError) as error:
                # An unavailable visual experiment must not prevent live inference.
                self.errors[qid] = str(error)

    def _load_grounding(self, qid, runtime, effects):
        # Resolve the case folder under the known results tree, ignoring old absolute roots.
        folder = self.visual_root / self.cases[qid]['video_id']
        metadata = read_json(folder / 'memory_visual_sensitivity.json')
        if metadata.get('representation_stage') != CONFIG['representation_stage'] or metadata['refinement']['checkpoint_sha256'] != CONFIG['tcmr_sha256']:
            raise ValueError('Etapa representacional o pesos visuales incompatibles.')
        content = read_json(folder / 'memory_visual_selectivity.json')
        memories = []
        if {m['memory'] for m in content['memories']} != {'E%02d' % i for i in range(1, 11)}:
            raise ValueError('Expected grounding for E01 through E10')
        for memory in content['memories']:
            name = memory['memory']
            regions = []
            for region in sorted(memory['top_distinctive_regions'], key=lambda r: r['rank']):
                pattern = 'rank_{:02d}_token_{:03d}_*.jpg'.format(region['rank'], region['token'])
                matches = list((folder / 'selective_frames' / name).glob(pattern))
                if len(matches) > 1:
                    raise ValueError('Ambiguous frame: ' + name + '/' + pattern)
                frame_url = None
                if matches:
                    resolved = matches[0].resolve()
                    resolved.relative_to(folder.resolve())
                    key = (qid, name, region['rank'])
                    self.frames[key] = resolved
                    frame_url = '/api/cases/{}/grounding/{}/frames/{}'.format(*key)
                regions.append(dict(region, frameUrl=frame_url))
            if not regions:
                raise ValueError('Empty selectivity profile: ' + name)
            memories.append(dict(memory=name, max_specificity_z=max(r['specificity_z'] for r in regions),
                                 regions=regions))
        baseline_index = runtime['prediction']
        reference_effects = []
        for rank, effect in enumerate(effects, 1):
            index = effect['prediction_after']
            reference_effects.append(dict(index=effect['memory_index'] + 1, memory=effect['memory_name'], rank=rank,
                delta=effect['signed_effect'], absolute_delta=effect['absolute_effect'], flip=bool(effect['flip']),
                original_answer_probability_after=effect['probabilities_after'][baseline_index],
                after=dict(prediction=effect['prediction_after_letter'],
                           answer=self.cases[qid]['options'][index], probabilities=effect['probabilities_after'],
                           probability=effect['probabilities_after'][index])))
        self.groundings[qid] = dict(case_id=qid, source='precomputed_visual_analysis',
            profile_correlation=content['profile_correlation'], memories=memories,
            reference_analysis=dict(source='precomputed_validation', memories=reference_effects,
                baseline=dict(prediction=runtime['prediction_letter'], probability=runtime['prediction_probability']),
                top=reference_effects[0]['index'], bottom=reference_effects[-1]['index']),
            files=dict(selectivity=(folder / 'memory_visual_selectivity.json').relative_to(ROOT).as_posix(),
                       summary=(self.visual_root / 'visual_analysis_summary.json').relative_to(ROOT).as_posix()),
            representation_stage=CONFIG['representation_stage'], checkpoint_sha256=dict(gf=CONFIG['gf_sha256'],tcmr=CONFIG['tcmr_sha256']))

    def runtime_cases(self):
        return list(self.cases.values())

    def thumbnail(self, qid):
        # A real exported frame is sufficient as the case preview; no new imagery.
        top = self.cases[qid]['reference']['top_memory']
        return self.frames.get((qid, top, 1))
