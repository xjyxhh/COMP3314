"""Run the queued fifth round concurrently, then merge frozen shard selections.

Only scheduling changes. The original controller and any redundant fifth-round
work are stopped only after the first four complete shards and this fifth shard
have passed integrity checks. No test is used until all 35 selections exist.
"""
import json
import os
import shutil
import signal
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from itml_repro.reid import fit_reid,evaluate_reid
from itml_repro.run import save_json,sha

HERE=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())

def main():
    inputs=HERE/'weight_input';extra=HERE/'weight_round04_parallel'
    with ProcessPoolExecutor(max_workers=1) as pool:
        pool.submit(fit_reid,inputs,extra,inputs/'baseline_config.json',['round_04']).result()
    first=[HERE/'weight_search_shards'/f'round_{i:02d}' for i in range(4)]
    while not all((p/'integrity.json').exists() for p in first):time.sleep(10)
    sources=first+[extra];records=[];protocol=read(extra/'protocol.json')
    for src in sources:
        assert read(src/'protocol.json')==protocol
        assert read(src/'integrity.json')['source_unchanged']
        rows=read(src/'results.json');assert len(rows)==7 and all(r['status']=='completed' for r in rows)
        records+=rows
    assert len({(r['round'],r['method']) for r in records})==35
    assert all(sha(inputs/n)==h for n,h in protocol['input_sha256'].items())
    # If the original controller already merged, keep its outcome and this extra
    # validation run as redundant; never compete with an active merge/test pass.
    out=HERE/'weight_search'
    if out.exists():
        save_json(HERE/'weight_parallel_recovery.json',dict(action='original_merge_already_started_extra_retained',new_training_settings=False))
        return
    table=[]
    for line in subprocess.check_output(['ps','-axo','pid=,ppid=,command='],text=True).splitlines():
        fields=line.strip().split(None,2)
        if len(fields)==3:table.append((int(fields[0]),int(fields[1]),fields[2]))
    parents=[p for p,_,cmd in table if cmd.endswith('python runs/external_rgbd_20261031/run_weight_study.py')]
    assert len(parents)==1,parents
    parent=parents[0];os.kill(parent,signal.SIGSTOP)
    # A merge can start between the prior check and stopping the controller.
    if out.exists():
        os.kill(parent,signal.SIGCONT)
        save_json(HERE/'weight_parallel_recovery.json',dict(action='original_merge_race_extra_retained',new_training_settings=False))
        return
    children=[p for p,pp,cmd in table if pp==parent and 'from multiprocessing.spawn import spawn_main' in cmd]
    for p in children:
        try:os.kill(p,signal.SIGTERM);os.kill(p,signal.SIGCONT)
        except ProcessLookupError:pass
    os.kill(parent,signal.SIGTERM);os.kill(parent,signal.SIGCONT)
    deadline=time.monotonic()+30
    for p in children+[parent]:
        while time.monotonic()<deadline:
            try:os.kill(p,0)
            except ProcessLookupError:break
            time.sleep(.1)
        else:raise RuntimeError('Original redundant controller did not exit')
    old=HERE/'weight_search_shards/round_04'
    if old.exists():old.rename(HERE/'weight_round04_redundant_attempt')
    shutil.copytree(extra,old)
    save_json(HERE/'weight_parallel_recovery.json',dict(action='merged_four_original_shards_and_concurrent_fifth',
        stopped_own_controller=parent,stopped_own_workers=children,previous_four_shards_unchanged=True,
        original_redundant_fifth_preserved=(HERE/'weight_round04_redundant_attempt').exists(),
        parameters_or_formulas_changed=False,all_validation_selections_finished=True,test_evaluated=False,
        input_sha256=protocol['input_sha256']))
    out.mkdir()
    for i,src in enumerate(sources):shutil.copytree(src/f'round_{i:02d}',out/f'round_{i:02d}')
    save_json(out/'protocol.json',protocol);save_json(out/'results.json',records)
    save_json(out/'integrity.json',dict(source_unchanged=True,test_evaluated=False))
    save_json(HERE/'weight_state.json',dict(phase='evaluating_frozen_parallel_merged_models',updated_unix=time.time()))
    evaluate_reid(inputs,out,HERE/'weight_final')
    save_json(HERE/'weight_state.json',dict(phase='complete_main_alternative_retrieval_study',updated_unix=time.time(),original_paper_reproduction=False))
    print('FULL WEIGHT STUDY COMPLETED WITH CONCURRENT FIFTH ROUND',flush=True)

if __name__=='__main__':main()
