"""Join completed round shards, run V-D searches, then freeze/evaluate/report.

The optional PID is ONLY the exact serial baseline command for this input;
stop it after snapshotting its completed round_00, avoiding duplicate work.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from .run import save_json,sha
from .paper_analyses import train_analyses
from .finalize_suite import finalize


def complete(root,serial_pid=None):
    state=root/'pipeline_state.json'
    def status(phase):
        save_json(state,dict(phase=phase,updated_unix=time.time(),original_paper_reproduction=False))
        print(phase,flush=True)
    status('waiting_for_baseline_shards')
    snapshot=root/'round_00_snapshot.json'
    if not snapshot.exists():
        while True:
            try:
                rows=json.loads((root/'baselines/results.json').read_text())
                rows=[r for r in rows if r['round']=='round_00']
                if len(rows)==7:break
            except (FileNotFoundError,json.JSONDecodeError):pass
            if serial_pid is not None:
                os.kill(serial_pid,0)
            time.sleep(5)
        save_json(snapshot,rows)
    if serial_pid is not None:
        command=subprocess.check_output(['ps','-p',str(serial_pid),'-o','command='],text=True).strip()
        expected='--output '+str(root/'baselines')
        if ' -m itml_repro.baseline_compare ' not in command or not command.endswith(expected):
            raise ValueError('Refusing to signal a different process: '+command)
        os.kill(serial_pid,signal.SIGINT)
        status('serial_round_00_preserved_duplicate_work_stopped')
    shards=[root/'shard_01',root/'shard_02',root/'shard_03_04']
    for folder in shards:
        while not (folder/'integrity.json').exists():time.sleep(5)
    combined=root/'baselines_final'
    combined.mkdir(exist_ok=False)
    records=json.loads(snapshot.read_text())
    shutil.copytree(root/'baselines/round_00',combined/'round_00')
    protocol=json.loads((root/'baselines/protocol.json').read_text())
    provenance=[]
    for shard in shards:
        data=json.loads((shard/'results.json').read_text())
        sp=json.loads((shard/'protocol.json').read_text())
        if sp['config']!=protocol['config'] or any(sha(root/n)!=h for n,h in sp['input_sha256'].items()):
            raise ValueError('Shard protocol/input mismatch')
        for folder in sorted(shard.glob('round_*')):shutil.copytree(folder,combined/folder.name)
        records.extend(data)
        provenance.append(dict(path=str(shard),protocol_sha256=sha(shard/'protocol.json'),results_sha256=sha(shard/'results.json')))
    if len(records)!=35 or len({(r['round'],r['method']) for r in records})!=35:
        raise ValueError('Expected exactly five rounds of seven methods')
    if not all(r['status']=='completed' for r in records):
        raise ValueError('Some baseline methods have no completed candidate; inspect shards')
    if any(sha(root/n)!=h for n,h in protocol['input_sha256'].items()):
        raise ValueError('Original inputs changed')
    records.sort(key=lambda r:(r['round'],protocol['methods'].index(r['method'])))
    save_json(combined/'protocol.json',protocol)
    save_json(combined/'shards.json',dict(serial_snapshot_sha256=sha(snapshot),shards=provenance,
        note='Independent round scheduling; numerical fit settings identical; timing measured separately'))
    save_json(combined/'results.json',records)
    save_json(combined/'integrity.json',dict(source_unchanged=True,test_evaluated=False))
    status('training_partial_and_sample_count_analyses')
    shard_root=root/'analysis_shards';shard_root.mkdir()
    round_names=[f'round_{n:02d}' for n in range(5)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(train_analyses,root,combined,shard_root/name,root/'baseline_config.json',[name]) for name in round_names]
        for future in futures:future.result()
    analysis=root/'analyses';analysis.mkdir()
    analysis_records=[];hashes={};sources=[]
    for name in round_names:
        shard=shard_root/name
        ap=json.loads((shard/'protocol.json').read_text())
        hashes.update(ap['input_sha256'])
        analysis_records.extend(json.loads((shard/'results.json').read_text()))
        shutil.copytree(shard/name,analysis/name)
        sources.append(dict(path=str(shard),protocol_sha256=sha(shard/'protocol.json'),results_sha256=sha(shard/'results.json')))
    save_json(analysis/'protocol.json',dict(config=protocol['config'],input_sha256=hashes,shards=sources,test_evaluated=False))
    save_json(analysis/'results.json',analysis_records)
    save_json(analysis/'integrity.json',dict(source_unchanged=all(sha(root/n)==h for n,h in hashes.items()),test_evaluated=False))
    status('all_validation_searches_finished_evaluating_frozen_models')
    finalize(root,combined,root/'analyses',root/'final')
    status('completed_available_data_extension')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--serial-pid',type=int)
    a=p.parse_args();complete(a.root,a.serial_pid)
