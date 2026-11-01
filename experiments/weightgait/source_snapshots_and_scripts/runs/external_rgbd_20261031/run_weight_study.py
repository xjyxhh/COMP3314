"""Full five-round alternative retrieval study; released display-depth caveat."""
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from itml_repro.weightgait import extract
from itml_repro.kdes_native import extract_archive
from itml_repro.reid import prepare,fit_reid,evaluate_reid
from itml_repro.run import save_json,sha

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
SOURCE=PROJECT/'data/external_rgbd_20261031/weightgait'

def main():
    def status(phase):
        save_json(HERE/'weight_state.json',dict(phase=phase,updated_unix=time.time(),original_paper_reproduction=False))
        print(phase,flush=True)
    while not (SOURCE/'All Depth Videos.zip').exists():time.sleep(10)
    status('extracting_frozen_RGB_crops_and_training_display_depth')
    crops=HERE/'weight_crops';extract(SOURCE,HERE/'weight_selection',crops)
    status('training_only_KDES_codebook_and_image_features')
    features=HERE/'weight_kdes.npz'
    extract_archive(crops/'manifest.csv',crops/'training_images.csv',features,'native_im2double')
    metadata=json.loads(features.with_suffix('.json').read_text())
    metadata['input_protocol_sha256']=sha(crops/'protocol.json')
    metadata['depth_caveat']='Colorized compressed uint8 display converted to grayscale; NOT raw metric depth or original BIWI KDES input'
    save_json(features.with_suffix('.json'),metadata)
    config=json.loads((crops/'reid_config.json').read_text())
    inputs=HERE/'weight_input';prepare(crops/'manifest.csv',features,inputs,config)
    search_config=json.loads((PROJECT/'configs/baselines_paper.json').read_text())
    search_config.update(lmnn_backend='projected',lmnn_max_iterations=1000,lmnn_tolerance=1e-7)
    save_json(inputs/'baseline_config.json',search_config)
    source=inputs/'source_snapshot';source.mkdir()
    for path in (PROJECT/'itml_repro').glob('*.py'):shutil.copy2(path,source/path.name)
    save_json(source/'checksums.json',{p.name:sha(p) for p in source.glob('*.py')})
    status('seven_method_full_grid_validation_Rank1_training')
    shards=HERE/'weight_search_shards';shards.mkdir();names=[f'round_{n:02d}' for n in range(5)]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(fit_reid,inputs,shards/n,inputs/'baseline_config.json',[n]) for n in names]
        for future in futures:future.result()
    out=HERE/'weight_search';out.mkdir();records=[];protocol=None
    for n in names:
        src=shards/n;p=json.loads((src/'protocol.json').read_text())
        if protocol is None:protocol=p
        elif p!=protocol:raise ValueError('Re-ID shard protocol mismatch')
        if not json.loads((src/'integrity.json').read_text())['source_unchanged']:raise ValueError('Re-ID integrity failed')
        records+=json.loads((src/'results.json').read_text());shutil.copytree(src/n,out/n)
    if len(records)!=35 or len({(r['round'],r['method']) for r in records})!=35:raise ValueError('Incomplete retrieval set')
    save_json(out/'protocol.json',protocol);save_json(out/'results.json',records)
    save_json(out/'integrity.json',dict(source_unchanged=all(sha(inputs/n)==h for n,h in protocol['input_sha256'].items()),test_evaluated=False))
    status('evaluating_frozen_models_on_two_heldout_normal_clips')
    evaluate_reid(inputs,out,HERE/'weight_final')
    status('complete_main_alternative_retrieval_study')
    print('FULL WEIGHT MAIN STUDY COMPLETED',flush=True)

if __name__=='__main__':main()
