"""Finish fixed-parameter analyses from frozen five-round baselines.

The former ORL shard/PID recovery controller is archived under
archive/cleanup_20261101/itml_repro/complete_pipeline.py.
"""
import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from .run import save_json,sha
from .paper_analyses import train_analyses
from .finalize_suite import finalize


def finish_only(root):
    """Resume from frozen baselines with an explicit single-factor policy."""
    combined=root/'baselines_final'
    if not (combined/'integrity.json').exists():raise ValueError('Frozen baseline results required')
    if (root/'final').exists() or (root/'analyses').exists():raise FileExistsError('Refusing to replace analysis/test results')
    config=json.loads((combined/'protocol.json').read_text())['config']
    save_json(root/'analysis_policy_revision.json',dict(recorded_unix=time.time(),policy='fixed_main',
        reason='Single-factor interpretation; paper V-D does not state per-fraction retuning',
        previous=('analysis_shards: retuning began, preserved incomplete; no test evaluation' if (root/'analysis_shards').exists()
                  else 'No previous analysis attempt; fixed_main policy recorded before analysis training'),
        alternative_available='--parameter-policy retune_each_fraction',test_evaluated=False))
    def status(phase):
        save_json(root/'pipeline_state.json',dict(phase=phase,updated_unix=time.time(),original_paper_reproduction=False))
        print(phase,flush=True)
    status('training_fixed_parameter_analyses')
    shards=root/'analysis_fixed_shards';shards.mkdir(exist_ok=False)
    names=[f'round_{n:02d}' for n in range(5)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(train_analyses,root,combined,shards/name,root/'baseline_config.json',[name],'fixed_main') for name in names]
        for future in futures:future.result()
    out=root/'analyses';out.mkdir();records=[];hashes={};sources=[]
    for name in names:
        shard=shards/name
        protocol=json.loads((shard/'protocol.json').read_text())
        if protocol['parameter_policy']!='fixed_main':raise ValueError('Mixed analysis policies')
        hashes.update(protocol['input_sha256']);records.extend(json.loads((shard/'results.json').read_text()))
        shutil.copytree(shard/name,out/name)
        sources.append(dict(path=str(shard),protocol_sha256=sha(shard/'protocol.json'),results_sha256=sha(shard/'results.json')))
    save_json(out/'protocol.json',dict(config=config,input_sha256=hashes,shards=sources,parameter_policy='fixed_main',test_evaluated=False))
    save_json(out/'results.json',records)
    save_json(out/'integrity.json',dict(source_unchanged=all(sha(root/n)==h for n,h in hashes.items()),test_evaluated=False))
    status('all_validation_searches_finished_evaluating_frozen_models')
    finalize(root,combined,out,root/'final')
    status('completed_available_data_extension')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--finish-only',action='store_true',required=True,
                   help='Run analyses and finalization from frozen baselines')
    a=p.parse_args()
    finish_only(a.root)
