"""Thin orchestration of the final experiment's public functions; no result fixtures."""
import json
import hashlib
import os
import random
from time import perf_counter
from pathlib import Path
from types import SimpleNamespace
from threading import RLock

import torch
import infer_causal_video_final as inference
from model.causal_memory_intervention import CausalMemoryIntervention

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / 'configs/final_experiment.json').read_text(encoding='utf-8'))
LETTERS = 'ABCD'
NOTE = ('Las intervenciones permiten analizar la influencia de las representaciones internas '
        'sobre la decisión del modelo. No representan necesariamente relaciones causales '
        'físicas entre eventos del video.')


class ModelService:
    def __init__(self, records=None):
        if records is None:
            if os.getenv('VIDEOQA_CASES'):
                records = json.loads(Path(os.environ['VIDEOQA_CASES']).read_text(encoding='utf-8'))
            else:
                from .visual_analysis import VisualAnalysisStore
                records = VisualAnalysisStore().runtime_cases()
        self.cases = {}
        for case in records:
            if case['id'] in self.cases or len(case['options']) != 4:
                raise ValueError('El catálogo debe tener IDs únicos y cuatro alternativas.')
            if case.get('correct_option') not in [None, *LETTERS]:
                raise ValueError('Ground truth inválido.')
            for field in ('video', 'features'):
                case[field] = str((ROOT / case[field]).resolve())
                if not Path(case[field]).is_file():
                    raise FileNotFoundError(case[field])
            self.cases[case['id']] = case
        self.args = SimpleNamespace(
            device=torch.device(os.getenv('VIDEOQA_DEVICE', 'cuda:0' if torch.cuda.is_available() else 'cpu')),
            hidden_dim=512, num_layers=2, num_queries=10, event_pred_dim=111,
            max_feats=80, feature_dim=1024, output_dim=1, num_options=4,
            qa_dataset='star', task_type='star',
            reload_model_path=str(ROOT / os.getenv('VIDEOQA_GF_CHECKPOINT', CONFIG['gf_checkpoint'])),
            temporal_checkpoint=str(ROOT / os.getenv('VIDEOQA_TCMR_CHECKPOINT', CONFIG['tcmr_checkpoint'])))
        self.checkpoint_hashes = {}
        for kind, path in [('gf', self.args.reload_model_path), ('tcmr', self.args.temporal_checkpoint)]:
            digest = hashlib.sha256()
            with Path(path).open('rb') as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    digest.update(block)
            self.checkpoint_hashes[kind] = digest.hexdigest()
            if digest.hexdigest() != CONFIG[kind + '_sha256']:
                raise ValueError('Checkpoint incompatible con la serie visual final: ' + kind)
        self.model, self.temporal = inference.load_models(self.args)
        self.intervention = CausalMemoryIntervention('zero')
        self.lock = RLock()

    def public_case(self, case):
        return {k: v for k, v in case.items() if k not in ('video', 'features', 'thumbnail')}

    def technical_info(self, case):
        return dict(device=str(self.args.device),
                    checkpoint_sha256=self.checkpoint_hashes,
                    gf_checkpoint=Path(self.args.reload_model_path).name,
                    tcmr_checkpoint=Path(self.args.temporal_checkpoint).name,
                    feature_filename=Path(case['features']).name,
                    tcmr_alpha=float(self.temporal.alpha.item()), random_seed=42,
                    question_batch_size=1)

    def distribution(self, probs, case):
        index = int(probs.argmax(dim=1)[0])
        return dict(probabilities=probs[0].tolist(), prediction=LETTERS[index],
                    answer=case['options'][index], probability=float(probs[0, index]),
                    correct=None if case.get('correct_option') is None else LETTERS[index] == case['correct_option'])

    @torch.no_grad()
    def evaluate(self, case_id, selected=None, ranking=False):
        # Every request owns its tensors. Interventions are individual, never cumulative.
        with self.lock:
            started = perf_counter()
            case = self.cases[case_id]
            a = self.args
            visual = inference.load_and_prepare_features(case['features'], a.max_feats, a.feature_dim)
            frames = visual.unsqueeze(0).repeat(a.num_options, 1, 1).to(a.device)
            text = [case['question'] + ' ' + option for option in case['options']]
            outputs = inference.run_glancing(self.model, frames, a)
            memory, spans = inference.sort_memories_and_spans(outputs['memory_prompt'], outputs['pred_spans'])
            refined = self.temporal(memory, spans)
            def focus(m):
                return torch.softmax(inference.run_focusing(self.model, frames, m, text, a)['pred_answer'], dim=1)
            original = focus(refined)
            baseline = self.distribution(original, case)
            reference = case.get('reference', {}).get('probabilities')
            reference_match = (None if reference is None else
                               max(abs(p - q) for p, q in zip(baseline['probabilities'], reference)) <= 1e-5)
            result = dict(case_id=case_id, baseline=baseline, current=baseline, selected=None,
                          reference_match=reference_match,
                          source='live_model', methodological_note=NOTE)
            question_memory = inference.collapse_option_memories(refined, a.num_options)
            effects = []
            indices = range(a.num_queries) if ranking else ([] if selected is None else [selected - 1])
            for index in indices:
                changed = self.intervention.intervene(question_memory, index)
                after = focus(inference.expand_question_memories(changed, a.num_options))
                delta = float(self.intervention.compute_effect(original, after)[0])
                effects.append(dict(memory=f'E{index + 1:02d}', index=index + 1,
                                    description='Representación latente interna',
                                    delta=delta, absolute_delta=abs(delta),
                                    original_answer_probability_after=float(after[0, original.argmax(dim=1)[0]]),
                                    after=self.distribution(after, case),
                                    flip=bool(after.argmax(dim=1)[0] != original.argmax(dim=1)[0])))
            if ranking:
                effects.sort(key=lambda item: item['absolute_delta'], reverse=True)
                for rank, effect in enumerate(effects, 1):
                    effect['rank'] = rank
                result.update(memories=effects, top=effects[0]['index'], bottom=effects[-1]['index'],
                              random=case.get('random_memory', random.Random(42).randint(1, a.num_queries)), random_seed=42)
            elif selected is not None:
                result.update(current=effects[0]['after'], selected=selected, effect=effects[0])
            result['technical'] = self.technical_info(case)
            result['technical']['execution_ms'] = round((perf_counter() - started) * 1000, 2)
            result['technical']['operation'] = 'analysis' if ranking else 'intervention' if selected else 'inference'
            return result
