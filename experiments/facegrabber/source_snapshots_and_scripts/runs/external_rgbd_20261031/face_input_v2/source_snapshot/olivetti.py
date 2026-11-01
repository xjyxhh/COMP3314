"""ORL extension: clean training images as privileged information, degraded-only inference."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
from .data import split_identities, make_pairs
from .pca import fit_pca, transform
from .metrics import fit, distances
from .corrected import fit_corrected
from .evaluation import verification
from .run import save_json, sha
from .audit_washington import feasibility


def image_features(im, seed):
    z = np.asarray(im.resize((32, 32), Image.Resampling.BOX), dtype=float)/255
    x = np.asarray(im.resize((8, 8), Image.Resampling.BOX), dtype=float)/255
    rng = np.random.default_rng(seed)
    return np.clip(x+rng.normal(0, .05, x.shape), 0, 1).ravel(), z.ravel()


def load_images(source):
    expected = {source/f's{s}'/f'{i}.pgm' for s in range(1, 41) for i in range(1, 11)}
    if set(source.glob('s*/*.pgm')) != expected:
        raise ValueError('Expected exactly s1..s40, each with 1.pgm..10.pgm')
    rows, clean, degraded, hashes = [], [], [], {}
    for subject in range(1, 41):
        for image in range(1, 11):
            path = source/f's{subject}'/f'{image}.pgm'
            with Image.open(path) as im:
                im.load()
                if im.mode != 'L' or im.size != (92, 112):
                    raise ValueError('Unexpected grayscale image format: '+str(path))
                x, z = image_features(im, 91000+subject*10+image)
            # Stable per-image noise independent of identity split, labels and results.
            rows.append(dict(sample_id=f's{subject}_{image}', identity=f's{subject}',
                             rgb_path=str(path.resolve()), depth_path='', privileged_path=str(path.resolve())))
            clean.append(z); degraded.append(x)
            hashes[str(path.relative_to(source))] = sha(path)
    return rows, np.asarray(degraded), np.asarray(clean), hashes


def prepare(source, out):
    rows, X, Z, hashes = load_images(source)
    save_json(out/'source_checksums.json', hashes)
    np.savez(out/'image_features.npz', sample_ids=np.array([r['sample_id'] for r in rows]), X=X, Z=Z)
    for index, seed in enumerate((3314, 3315, 3316)):
        folder = out/f'split_{index:02d}'/'input'
        folder.mkdir(parents=True)
        splits = split_identities(rows, [20, 10, 10], seed)
        with (folder/'manifest.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        np.savez(folder/'split.npz', **splits)
        save_json(folder/'split.json', {s: sorted({rows[i]['identity'] for i in idx}) for s,idx in splits.items()})
        px, pz = fit_pca(X[splits['train']], 16), fit_pca(Z[splits['train']], 16)
        xx, zz = transform(X, px), transform(Z, pz)
        factor = float(np.sqrt(np.sum(xx[splits['train']]**2)/np.sum(zz[splits['train']]**2)))
        zz *= factor
        np.savez(folder/'pca.npz', X_mean=px[0], X_components=px[1], Z_mean=pz[0], Z_components=pz[1]*factor)
        # Deliberately omit clean validation/test features from training artifacts.
        np.savez(folder/'features_pca.npz', **{'X_'+s:xx[idx] for s,idx in splits.items()}, Z_train=zz[splits['train']])
        save_json(folder/'feature_definition.json', dict(dataset='ORL/Olivetti, original PGM', seed=seed,
            ordinary='8x8 BOX downsample + fixed Gaussian noise std .05, clipped [0,1]',
            privileged='32x32 BOX downsample of same original image, no added noise; not physical depth',
            pca='16 dimensions per modality, training only, centered, no whitening',
            training_rms_privileged_scale=factor, split_counts=[20,10,10]))
        pairs = {s:make_pairs([rows[i]['identity'] for i in idx], 90, 180, seed*10+k)
                 for k,(s,idx) in enumerate(splits.items())}
        rd = folder/'round_00'; rd.mkdir()
        np.savez(rd/'pairs.npz', **pairs)
        train = xx[splits['train']]
        i,j = np.triu_indices(len(train),1)
        bounds = np.quantile(np.linalg.norm(train[i]-train[j],axis=1), [.03,.97])
        np.save(rd/'xi0.npy', np.where(pairs['train'][:,2]==1,*bounds))
        save_json(folder/'bounds.json', dict(low=float(bounds[0]), high=float(bounds[1]), distance='euclidean'))
        audit = feasibility(train, zz[splits['train']], pairs['train'], rd)
        save_json(folder/'feasibility.json', audit)
        print('prepared', index, audit['classification'], flush=True)
    if any(sha(source/name) != digest for name,digest in hashes.items()):
        raise RuntimeError('Original images changed')


def evaluate(out):
    # All validation searches must finish before any test scores are computed.
    for index in range(3):
        search = out/f'split_{index:02d}'/'search'
        if not (search/'integrity.json').is_file():
            raise ValueError('Incomplete validation search: '+str(search))
    records = []
    for index in range(3):
        folder = out/f'split_{index:02d}'
        inputs, search = folder/'input', folder/'search'
        if not (search/'integrity.json').is_file():
            raise ValueError('Incomplete validation search')
        protocol = json.loads((search/'protocol.json').read_text())
        if any(sha(inputs/name) != digest for name,digest in protocol['sha256'].items()):
            raise ValueError('Input changed after validation selection')
        with np.load(inputs/'features_pca.npz', allow_pickle=False) as data:
            X, Z, test = data['X_train'], data['Z_train'], data['X_test']
        with np.load(inputs/'round_00/pairs.npz', allow_pickle=False) as data:
            pairs, test_pairs = data['train'], data['test']
        xi0 = np.load(inputs/'round_00/xi0.npy')
        for record in json.loads((search/'results.json').read_text()):
            record = dict(record, split=index)
            method = record['method']
            if record['status']=='completed':
                with np.load(search/'round_00'/f'{method}_model.npz', allow_pickle=False) as model:
                    record['test'] = verification(test, test_pairs, model['M'])
                    if 'plus' in method:
                        r, s = distances(X,pairs,model['M']), distances(Z,pairs,model['P'])
                        record['train_diagnostics'] = dict(
                            relative_violation=float(np.max(np.maximum(pairs[:,2]*(r-s),0)/np.maximum(r,s))),
                            relative_equality=float(np.max(np.abs(model['xi']-s)/np.maximum(model['xi'],s))),
                            min_eigenvalue=float(min(np.linalg.eigvalsh(model['M']).min(),np.linalg.eigvalsh(model['P']).min())))
                if 'plus' in method:
                    permutation = np.random.default_rng(5530+index).permutation(len(Z))
                    np.save(folder/f'{method}_shuffle.npy', permutation)
                    config = record['selected']
                    solver = fit_corrected if method.endswith('corrected') else fit
                    try:
                        model = solver(X,pairs,xi0,config['gamma'],Z=Z[permutation],lam=config['lam'],max_sweeps=100,tol=0.)
                        record['shuffle'] = dict(status='completed', test=verification(test,test_pairs,model['M']),final=model['history'][-1])
                        np.savez(folder/f'{method}_shuffled.npz', **{k:v for k,v in model.items() if isinstance(v,np.ndarray)})
                    except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
                        record['shuffle'] = dict(status='failed',error=str(error))
            records.append(record)
            save_json(out/'results.json',records)
            print('test', index, method, record.get('test'), flush=True)
    summary = {}
    for method in ('euclidean','itml','itml_plus','itml_plus_corrected'):
        rows = [r for r in records if r['method']==method and 'test' in r]
        summary[method] = dict(completed_splits=len(rows), **{metric:dict(
            mean=float(np.mean([r['test'][metric] for r in rows])),
            std=float(np.std([r['test'][metric] for r in rows]))) for metric in ('AP','AUC')}) if rows else dict(completed_splits=0)
        controls = [r['shuffle']['test'] for r in rows if r.get('shuffle',{}).get('status')=='completed']
        if controls:
            summary[method]['shuffle'] = dict(completed_splits=len(controls), **{
                metric:float(np.mean([r[metric] for r in controls])) for metric in ('AP','AUC')})
    save_json(out/'summary.json',summary)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage', choices=('prepare','train','evaluate'),required=True)
    args=p.parse_args()
    if args.stage=='prepare':
        args.output.mkdir(parents=True,exist_ok=False)
        save_json(args.output/'protocol.json',dict(dataset='ORL/Olivetti',source=str(args.source.resolve()),
            scope='clean-to-degraded privileged-information extension; not RGB-D or original paper numerical reproduction',
            identities=40,images=400,seeds=[3314,3315,3316],split_counts=[20,10,10],
            pairs_per_split=dict(positive=90,negative=180),pca_dimensions=[16,16],
            degradation=dict(resolution=[8,8],noise_std=.05),privileged_resolution=[32,32],
            gamma_grid=[.01,.1,1.],lambda_grid=[.1,1.,10.],max_sweeps=100,
            selection='validation AP only, first candidate on ties; no test until all searches complete',
            primary_metric='mean test AP across all 3 predefined identity splits',
            controls='Euclidean, ordinary-only ITML, literal ITML+, corrected ITML+; shuffled clean training rows at selected hyperparameters',
            limitations='small dataset; splits overlap across repeats; no guarantee of improvement; finite training budget'))
        prepare(args.source,args.output)
    elif args.stage=='train':
        for index in range(3):
            folder=args.output/f'split_{index:02d}'
            subprocess.run([sys.executable,'-m','itml_repro.viewpoint','--input',str(folder/'input'),
                            '--output',str(folder/'search'),'--defer-test'],check=True)
    else:
        if (args.output/'results.json').exists():
            raise FileExistsError('Refusing to overwrite test results')
        evaluate(args.output)


if __name__=='__main__':
    main()
