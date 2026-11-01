"""Run the full frozen FaceGrabber study, preserving every round and old ORL."""
import json
import shutil
import time
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from itml_repro.baseline_compare import run
from itml_repro.complete_pipeline import finish_only
from itml_repro.run import save_json,sha

ROOT=Path(__file__).resolve().parent/'face_input_v2'

def main():
    while not (ROOT/'baseline_config.json').exists():
        print('Waiting for complete face feature preparation',flush=True);time.sleep(10)
    snapshot=ROOT/'source_snapshot';snapshot.mkdir(exist_ok=False)
    for path in (ROOT.parents[2]/'itml_repro').glob('*.py'):shutil.copy2(path,snapshot/path.name)
    save_json(snapshot/'checksums.json',{p.name:sha(p) for p in snapshot.glob('*.py')})
    def status(phase):save_json(ROOT/'study_state.json',dict(phase=phase,updated_unix=time.time(),original_paper_reproduction=False))
    status('seven_method_full_grid_validation_training')
    names=[f'round_{i:02d}' for i in range(5)];shards=ROOT/'baseline_shards';shards.mkdir(exist_ok=False)
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(run,Namespace(input=ROOT,output=shards/name,config=ROOT/'baseline_config.json',rounds=[name])) for name in names]
        for future in futures:future.result()
    dest=ROOT/'baselines_final';dest.mkdir(exist_ok=False);records=[];protocol=None;sources=[]
    for name in names:
        src=shards/name;p=json.loads((src/'protocol.json').read_text())
        if protocol is None:protocol=p
        elif any(p[k]!=protocol[k] for k in ('config','code_sha256','methods')):raise ValueError('Shard settings mismatch')
        if any(sha(ROOT/n)!=h for n,h in p['input_sha256'].items()):raise ValueError('Shard input mismatch')
        protocol['input_sha256'].update(p['input_sha256'])
        records+=json.loads((src/'results.json').read_text());shutil.copytree(src/name,dest/name)
        sources.append(dict(path=str(src),protocol_sha256=sha(src/'protocol.json'),results_sha256=sha(src/'results.json')))
    if len(records)!=35 or len({(r['round'],r['method']) for r in records})!=35:raise ValueError('Incomplete baseline set')
    save_json(dest/'protocol.json',protocol);save_json(dest/'results.json',records)
    save_json(dest/'shards.json',sources);save_json(dest/'integrity.json',dict(source_unchanged=True,test_evaluated=False))
    status('fixed_parameter_paper_analyses')
    finish_only(ROOT)
    status('complete_face_alternative_study')
    print('FULL FACE STUDY COMPLETED',flush=True)

if __name__=='__main__':main()
