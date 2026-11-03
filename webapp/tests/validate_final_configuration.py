"""Validate final live cases against the selected checkpoint and visual source."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',default='http://127.0.0.1:8000')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=(ROOT/a.output).resolve()
    if out.exists():raise FileExistsError('Use a new validation output directory.')
    out.mkdir(parents=True)
    cfg=json.loads((ROOT/'configs/final_experiment.json').read_text(encoding='utf-8'))
    expected={r['case']['question_id']:r for r in json.loads((ROOT/cfg['reference']).read_text(encoding='utf-8'))}
    def request(path,body=None):
        r=Request(a.base_url+path,data=None if body is None else json.dumps(body).encode(),headers={} if body is None else {'Content-Type':'application/json'})
        with urlopen(r,timeout=180) as response:return response.read()
    def api(path,body=None):return json.loads(request(path,body))
    cases=api('/api/cases');assert len(cases)==10
    evidence=dict(configuration=cfg,cases=[],memories=0,frames=0,interventions=0,errors=[])
    for case in cases:
        qid=case['id'];ref=expected[qid]['runtime_result'];prefix='/api/cases/'+qid
        data=api(prefix+'/analysis',{})
        assert data['technical']['checkpoint_sha256']==dict(gf=cfg['gf_sha256'],tcmr=cfg['tcmr_sha256'])
        assert data['reference_match'] and len(data['memories'])==10
        assert all(abs(x-y)<=1e-5 for x,y in zip(data['baseline']['probabilities'],ref['probabilities']))
        grounding=api(prefix+'/grounding')
        assert grounding['representation_stage']=='tcmr_refined' and grounding['checkpoint_sha256']==data['technical']['checkpoint_sha256']
        source=ROOT/grounding['files']['selectivity'];original=json.loads(source.read_text(encoding='utf-8'))
        row=dict(question_id=qid,video_id=case['video_id'],analysis=data,grounding_source=source.relative_to(ROOT).as_posix(),memories=[],frames=[])
        for memory in grounding['memories']:
            name=memory['memory'];stored=next(m for m in original['memories'] if m['memory']==name)
            assert memory['max_specificity_z']==max(r['specificity_z'] for r in stored['top_distinctive_regions'])
            index=int(name[1:]);effect=next(m for m in data['memories'] if m['index']==index)
            after=api(prefix+'/intervene',{'memory':index})
            assert after['current']['probabilities']==effect['after']['probabilities']
            old=next(m for m in ref['effects'] if m['memory_index']==index-1)
            assert abs(effect['delta']-old['signed_effect'])<=1e-5 and effect['flip']==bool(old['flip'])
            row['memories'].append(name);evidence['memories']+=1;evidence['interventions']+=1
            for region in memory['regions']:
                if region['frameUrl']:
                    pattern='rank_{:02d}_token_{:03d}_*.jpg'.format(region['rank'],region['token'])
                    files=list((source.parent/'selective_frames'/name).glob(pattern));assert len(files)==1
                    image=request(region['frameUrl']);assert image==files[0].read_bytes() and image[:2]==b'\xff\xd8'
                    row['frames'].append(dict(memory=name,rank=region['rank'],sha256=hashlib.sha256(image).hexdigest()));evidence['frames']+=1
        assert api(prefix+'/restore',{})['current']['probabilities']==data['baseline']['probabilities']
        assert request(prefix+'/video')[:4] and request(prefix+'/thumbnail')[:2]==b'\xff\xd8'
        evidence['cases'].append(row)
        print('Validated '+qid,flush=True)
    assert evidence['memories']==100 and evidence['frames']==300 and evidence['interventions']==100
    with (out/'backend.json').open('x',encoding='utf-8') as f:json.dump(evidence,f,indent=2,ensure_ascii=False)
    with (out/'functional_cases.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['question_id','video_id','inference','memories','interventions','visual_stage','frames']);w.writeheader()
        for row in evidence['cases']:w.writerow(dict(question_id=row['question_id'],video_id=row['video_id'],inference='PASS',memories=10,interventions=10,visual_stage='tcmr_refined',frames=len(row['frames'])))


if __name__=='__main__':main()
