"""Five-round ORL G-LBP extension; explicit non-depth/non-landmark deviations."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .olivetti import load_images
from .glbp import glbp
from .data import split_identities,make_pairs
from .pca import fit_pca,transform
from .run import save_json,sha


def prepare(source,out):
    out.mkdir(parents=True,exist_ok=False)
    rows,ordinary,_,hashes=load_images(source)
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/baselines_paper.json').read_text())
    config.update(lmnn_backend='projected',lmnn_max_iterations=1000,lmnn_tolerance=1e-7)
    save_json(out/'baseline_config.json',config)
    save_json(out/'protocol.json',dict(scope='ORL G-LBP clean-image privilege EXTENSION, not RGB-D numerical reproduction',
        source=str(source.resolve()),seed=2026,identity_split=[20,10,10],rounds=5,pca_dimensions=150,
        pair_counts=dict(train=[900,2904],val=[450,1452],test=[450,1452]),
        pairs='all positive pairs fixed, new negative samples each round, no replacement',
        ordinary='8x8 grayscale BOX + fixed noise 0.05, round to uint8 and resize bilinear to 120x105',
        privileged='original grayscale resized bilinear to 120x105, not physical depth',
        glbp='123 bins/block, 56 blocks, grid neighbors, valid block centers, histogram_l1',
        alignment='resize only: eye landmarks unavailable; extra deviation from paper',
        bounds='training ordinary L2 q03 for u, q97 for l; ambiguity disclosed',
        primary_metric='mean AP and AUC across five rounds; population std',
        prior_test_exposure='same 40 people appeared in earlier experiments; exploratory evaluation',
        feature_code_sha256=sha(Path(__file__).parent/'glbp.py'),test_evaluated=False))
    save_json(out/'source_checksums.json',hashes)
    X,Z=[],[]
    for i,row in enumerate(rows):
        with Image.open(row['rgb_path']) as im:
            clear=np.asarray(im.resize((105,120),Image.Resampling.BILINEAR))
        low=np.rint(ordinary[i].reshape(8,8)*255).astype('uint8')
        degraded=np.asarray(Image.fromarray(low).resize((105,120),Image.Resampling.BILINEAR))
        X.append(glbp(degraded,'histogram_l1'));Z.append(glbp(clear,'histogram_l1'))
    X,Z=np.asarray(X),np.asarray(Z)
    np.savez(out/'raw_glbp.npz',X=X,Z=Z,sample_ids=np.array([r['sample_id'] for r in rows]))
    split=split_identities(rows,[20,10,10],2026)
    np.savez(out/'split.npz',**split)
    save_json(out/'split.json',{k:[rows[i]['sample_id'] for i in idx] for k,idx in split.items()})
    for name,idx in split.items():
        with (out/f'{name}_manifest.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows[i] for i in idx)
    px,pz=fit_pca(X[split['train']],150),fit_pca(Z[split['train']],150)
    xs={k:transform(X[idx],px) for k,idx in split.items()}
    z=transform(Z[split['train']],pz)
    np.savez(out/'pca.npz',rgb_mean=px[0],rgb_components=px[1],depth_mean=pz[0],depth_components=pz[1])
    np.savez(out/'features_pca.npz',**{'X_'+k:v for k,v in xs.items()},Z_train=z)
    i,j=np.triu_indices(len(xs['train']),1)
    u,l=np.quantile(np.linalg.norm(xs['train'][i]-xs['train'][j],axis=1),[.03,.97])
    save_json(out/'bounds.json',dict(u=float(u),l=float(l)))
    for round_id in range(5):
        folder=out/f'round_{round_id:02d}';folder.mkdir()
        pairs={}
        for offset,(name,idx) in enumerate(split.items()):
            counts=(900,2904) if name=='train' else (450,1452)
            pairs[name]=make_pairs([rows[i]['identity'] for i in idx],*counts,2026+round_id*1000+offset,
                                  positive_seed=9000+offset)
        np.savez(folder/'pairs.npz',**pairs)
        np.save(folder/'xi0.npy',np.where(pairs['train'][:,2]==1,u,l))
    assert all(sha(source/name)==h for name,h in hashes.items())
    print('Prepared 400 images, 6888->150 dims, five fixed rounds',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.source,a.output)
