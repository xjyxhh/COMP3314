"""Run predeclared controls as soon as the two main validation searches freeze."""
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from itml_repro.supplemental import controls

HERE=Path(__file__).resolve().parent

def wait_and_run(kind):
    face=kind=='face'
    inputs=HERE/'face_input_v2' if face else HERE/'weight_input'
    baseline=inputs/'baselines_final' if face else HERE/'weight_search'
    while not (baseline/'integrity.json').exists():time.sleep(10)
    controls(inputs,baseline,HERE/(kind+'_controls'),retrieval=not face)
    print(kind,'CONTROLS COMPLETED',flush=True)

if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(wait_and_run,['face','weight']))
