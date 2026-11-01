"""Synthetic plumbing check ONLY; does not represent a paper experiment."""
import argparse
import csv
from pathlib import Path
import numpy as np


def generate(output):
    path=Path(output); path.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(31)
    ids=np.repeat(np.arange(8),5)
    latent=rng.normal(size=(8,6))[ids]
    X=latent + .8*rng.normal(size=(40,6))
    Z=latent + .3*rng.normal(size=(40,6))
    sample_ids=np.array([f's{i:03d}' for i in range(40)])
    with (path/'manifest.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['sample_id','identity','rgb_path','depth_path'])
        writer.writerows((s,str(y),'','') for s,y in zip(sample_ids,ids))
    np.savez(path/'features.npz',sample_ids=sample_ids,X=X,Z=Z)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    generate(p.parse_args().output)
