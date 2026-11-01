"""Bound active OWN training workers; preserve solver states and all settings.

SIGSTOP/SIGCONT scheduling changes elapsed search times only. Main fits have
iteration/norm stopping, no wall-time budget. Final timing is separately refit.
Always resume every worker we suspended on normal shutdown or interruption.
"""
import json
import os
import signal
import shutil
import subprocess
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
COMMANDS={'face':'python runs/external_rgbd_20261031/run_face_study.py',
          'body':'python runs/external_rgbd_20261031/run_weight_study.py',
          'robustness':'python runs/external_rgbd_20261031/run_robustness.py',
          'body_extra':'python runs/external_rgbd_20261031/complete_weight_parallel.py'}
LIMITS={'face':4,'body':2,'robustness':2,'body_extra':1}
STOP=False

def main():
    suspended=set();events=[];start=time.monotonic()
    def end(*_):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,end);signal.signal(signal.SIGINT,end)
    try:
        while not STOP:
            table=[]
            for line in subprocess.check_output(['ps','-axo','pid=,ppid=,command='],text=True).splitlines():
                parts=line.strip().split(None,2)
                if len(parts)==3:table.append((int(parts[0]),int(parts[1]),parts[2]))
            roots={name:pid for name,suffix in COMMANDS.items() for pid,_,command in table if command.endswith(suffix)}
            living={p for p,_,_ in table};suspended&=living
            if not roots:break
            limits=LIMITS if shutil.disk_usage(HERE).free>2*1024**3 else {'face':2,'body':1,'robustness':1,'body_extra':0}
            slot=int((time.monotonic()-start)//120)
            desired=set();tracked=set()
            for name,parent in roots.items():
                children=sorted(pid for pid,ppid,cmd in table if ppid==parent and 'from multiprocessing.spawn import spawn_main' in cmd)
                tracked.update(children)
                if children:
                    offset=(slot*limits[name])%len(children)
                    active=[children[(offset+i)%len(children)] for i in range(min(limits[name],len(children)))]
                    desired.update(active)
            for pid in tracked:
                try:
                    if pid not in desired and pid not in suspended:
                        os.kill(pid,signal.SIGSTOP);suspended.add(pid);events.append(dict(unix=time.time(),pid=pid,action='suspend_own_worker'))
                    elif pid in desired and pid in suspended:
                        os.kill(pid,signal.SIGCONT);suspended.remove(pid);events.append(dict(unix=time.time(),pid=pid,action='resume_own_worker'))
                except ProcessLookupError:pass
            (HERE/'memory_scheduling.json').write_text(json.dumps(dict(active_limits=limits,suspended=sorted(suspended),events=events,
                note='Search elapsed times include explicit scheduling pauses. Model settings unchanged. Final training times measured after all work.'),indent=2)+'\n')
            time.sleep(10)
    finally:
        for pid in suspended:
            try:os.kill(pid,signal.SIGCONT)
            except ProcessLookupError:pass
        print('All scheduled workers resumed',flush=True)

if __name__=='__main__':main()
