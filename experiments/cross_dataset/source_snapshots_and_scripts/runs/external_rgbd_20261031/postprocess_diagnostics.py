"""Run descriptive diagnostics after final audit and quiet timing finish."""
from pathlib import Path
import runpy
import time

if __name__=='__main__':
    here=Path(__file__).resolve().parent
    while not (here/'final_report/completion.json').exists():time.sleep(10)
    runpy.run_path(str(here/'final_diagnostics.py'),run_name='__main__')
