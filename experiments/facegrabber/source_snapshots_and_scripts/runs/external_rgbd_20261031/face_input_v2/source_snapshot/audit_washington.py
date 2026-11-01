"""Fixed Washington input audit. Basic spatial features, NOT paper GLBP."""
import argparse,csv,hashlib,json,os
from collections import defaultdict
from pathlib import Path
import cvxpy as cp
import numpy as np
from PIL import Image
from .data import read_manifest
from .pca import fit_pca,transform


def feasibility(X,Z,pairs,folder):
    v,w=X[pairs[:,0]]-X[pairs[:,1]],Z[pairs[:,0]]-Z[pairs[:,1]]
    C=np.einsum('ni,nj->nij',v,v)*pairs[:,2,None,None]
    D=-np.einsum('ni,nj->nij',w,w)*pairs[:,2,None,None]
    scale=np.sqrt((C*C).sum((1,2))+(D*D).sum((1,2)))
    assert np.all(scale>0)
    C,D=C/scale[:,None,None],D/scale[:,None,None]
    d,e=X.shape[1],Z.shape[1]
    M=cp.Variable((d,d),symmetric=True);P=cp.Variable((e,e),symmetric=True);t=cp.Variable()
    constraints=[M-t*np.eye(d)>>0,P-t*np.eye(e)>>0,cp.trace(M)+cp.trace(P)==1]
    constraints += [cp.sum(cp.multiply(c,M))+cp.sum(cp.multiply(q,P))<=0 for c,q in zip(C,D)]
    problem=cp.Problem(cp.Maximize(t),constraints)
    problem.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9,max_iter=500)
    if problem.status not in ('optimal','optimal_inaccurate'):
        return dict(status=problem.status,classification=('solver_reported_infeasible' if problem.status=='infeasible' else 'undetermined'))
    mu=np.array([c.dual_value for c in constraints[3:]]);eta=float(constraints[2].dual_value)
    # Enforce exact nonnegative multipliers before checking a conservative dual bound.
    mu=np.maximum(mu,0)
    A=eta*np.eye(d)+np.einsum('n,nij->ij',mu,C)
    B=eta*np.eye(e)+np.einsum('n,nij->ij',mu,D)
    delta=max(0.,-np.linalg.eigvalsh(A).min(),-np.linalg.eigvalsh(B).min())+1e-10
    normalizer=np.trace(A)+np.trace(B)+(d+e)*delta
    assert normalizer>0
    bound=(eta+delta)/normalizer
    dual_min=min(np.linalg.eigvalsh((A+delta*np.eye(d))/normalizer).min(),np.linalg.eigvalsh((B+delta*np.eye(e))/normalizer).min())
    # Seek an interior witness independently of the margin solver's status.
    witness_slack = None
    if float(t.value) > 1e-6:
        floor = float(t.value)/2
        N=cp.Variable((d,d),symmetric=True);Q=cp.Variable((e,e),symmetric=True);h=cp.Variable()
        interior=cp.Problem(cp.Maximize(h),[N-floor*np.eye(d)>>0,Q-floor*np.eye(e)>>0,
            cp.trace(N)+cp.trace(Q)==1]+[cp.sum(cp.multiply(c,N))+cp.sum(cp.multiply(q,Q))<=-h for c,q in zip(C,D)])
        interior.solve(solver='CLARABEL',tol_gap_abs=1e-8,tol_gap_rel=1e-8,tol_feas=1e-9,max_iter=500)
        if N.value is not None and Q.value is not None:
            raw=np.einsum('nij,ij->n',C,N.value)+np.einsum('nij,ij->n',D,Q.value)
            if raw.max() < -1e-9 and min(np.linalg.eigvalsh(N.value).min(),np.linalg.eigvalsh(Q.value).min())>1e-6:
                M.value=N.value;P.value=Q.value;witness_slack=float(-raw.max())
    violation=float(max(0.,np.max(np.einsum('nij,ij->n',C,M.value)+np.einsum('nij,ij->n',D,P.value))))
    eig=float(min(np.linalg.eigvalsh(M.value).min(),np.linalg.eigvalsh(P.value).min()))
    trace_error=float(abs(np.trace(M.value)+np.trace(P.value)-1))
    status='undetermined'
    if dual_min>=0 and bound < -1e-7: status='numerically_certified_spd_infeasible'
    elif witness_slack is not None and eig>1e-6 and violation==0 and trace_error<1e-8: status='numerically_spd_feasible'
    np.savez(folder/'feasibility_certificate.npz',M=M.value,P=P.value,mu=mu/normalizer,eta=bound,C=C,D=D)
    return dict(status=problem.status,classification=status,normalized_spd_margin=float(t.value),
                conservative_dual_upper_bound=float(bound),dual_psd_min=float(dual_min),
                primal_min_eigenvalue=eig,normalized_pair_violation=violation,trace_error=trace_error,strict_witness_pair_slack=witness_slack)


def extract(root,row):
    with Image.open(root/row['rgb_path']) as im: rgb=np.asarray(im.convert('RGB'),dtype=float)/255
    with Image.open(root/row['depth_path']) as im: depth=np.asarray(im)
    with Image.open(root/row['mask_path']) as im: mask=np.asarray(im,dtype=bool)
    if rgb.shape[:2]!=depth.shape or depth.shape!=mask.shape or depth.dtype!=np.uint16: raise ValueError(row['sample_id'])
    valid=mask & (depth>0)
    if not mask.any() or not valid.any(): raise ValueError('No object depth')
    relative=np.zeros(depth.shape);relative[valid]=depth[valid]/np.median(depth[valid])-1
    x=[];z=[]
    for ys in np.array_split(np.arange(mask.shape[0]),4):
        for xs in np.array_split(np.arange(mask.shape[1]),4):
            ix=np.ix_(ys,xs);m=mask[ix];d=valid[ix]
            x.extend(rgb[ix][m].mean(0) if m.any() else [0.,0.,0.]);x.append(float(m.mean()))
            z.extend([float(relative[ix][d].mean()) if d.any() else 0.,float(d.mean())])
    return x,z,float(valid.sum()/mask.sum())


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest',type=Path,default=Path('data/washington/manifest.csv'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    root=args.manifest.resolve().parent;rows=read_manifest(args.manifest)
    groups=defaultdict(lambda:defaultdict(list));issues=[];missing=0
    for r in rows:
        prefix=f"{r['identity']}_{r['sequence']}_{r['frame']}"
        if r['sample_id']!=prefix: issues.append(r['sample_id'])
        for key,suffix in [('rgb_path','_crop.png'),('depth_path','_depthcrop.png'),('mask_path','_maskcrop.png'),('loc_path','_loc.txt')]:
            p=root/r[key]
            if p.name!=prefix+suffix or p.parent.name!=r['identity'] or p.parent.parent.name!=r['category']: issues.append(r['sample_id']+':'+key)
            missing+=not p.is_file()
        groups[r['category']][r['identity']].append(r)
    if issues or missing: raise ValueError(f'Bad manifest: {issues[:10]}, missing {missing}')
    rng=np.random.default_rng(2026);selected=[];split_ids={s:[] for s in ('train','val','test')};eligible=[]
    for cat,identities in sorted(groups.items()):
        ids=[identity for identity,rr in sorted(identities.items()) if all(sum(r['sequence']==s for r in rr)>=4 for s in ('1','2','4'))]
        if len(ids)>=3: eligible.append((cat,ids))
    for cat,ids in eligible[:8]:
        for split,identity in zip(split_ids,rng.choice(ids,3,replace=False)):
            split_ids[split].append(str(identity))
            for seq in ('1','2','4'):
                rr=sorted((r for r in groups[cat][identity] if r['sequence']==seq),key=lambda r:int(r['frame']))
                selected.extend(dict(rr[k],split=split) for k in np.linspace(0,len(rr)-1,4,dtype=int))
    assert len(split_ids['train'])==8
    assert all(not(set(split_ids[a]) & set(split_ids[b])) for a,b in [('train','val'),('train','test'),('val','test')])
    extracted=[extract(root,r) for r in selected]
    X,Z=np.array([a[0] for a in extracted]),np.array([a[1] for a in extracted])
    assert np.isfinite(X).all() and np.isfinite(Z).all()
    outrows=[dict(r,**{k:os.path.relpath(root/r[k],args.output.resolve()) for k in ('rgb_path','depth_path','mask_path','loc_path')}) for r in selected]
    with (args.output/'manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(outrows[0]));writer.writeheader();writer.writerows(outrows)
    np.savez(args.output/'features.npz',sample_ids=np.array([r['sample_id'] for r in selected]),X=X,Z=Z)
    splits={s:np.array([i for i,r in enumerate(selected) if r['split']==s]) for s in split_ids}
    px,pz=fit_pca(X[splits['train']],8),fit_pca(Z[splits['train']],8)
    x,z=transform(X,px),transform(Z,pz)
    np.savez(args.output/'pca.npz',X_mean=px[0],X_components=px[1],Z_mean=pz[0],Z_components=pz[1])
    np.savez(args.output/'split.npz',**splits)
    np.savez(args.output/'features_pca.npz',**{f'{axis}_{s}':array[idx] for s,idx in splits.items() for axis,array in [('X',x),('Z',z)]})
    result=dict(seed=2026,scope='Washington diagnostic subset; basic spatial features, NOT GLBP',
                full_manifest=dict(samples=len(rows),categories=len(groups),instances=sum(map(len,groups.values())),missing_files=int(missing),path_label_issues=issues),
                selected=dict(samples=len(selected),split_identities=split_ids,features_rgb=X.shape[1],features_depth=Z.shape[1],pca_dimension=8,min_valid_depth_fraction=min(a[2] for a in extracted)),rounds=[])
    for rd in range(2):
        folder=args.output/f'round_{rd:02d}';folder.mkdir();pairs_by_split={};checks={}
        for s,idx in splits.items():
            rr=[selected[i] for i in idx];pos=[];neg=[]
            for i in range(len(rr)):
                for j in range(i+1,len(rr)):
                    if rr[i]['identity']==rr[j]['identity']:
                        if rr[i]['sequence']!=rr[j]['sequence']: pos.append((i,j,1))
                    else: neg.append((i,j,-1))
            sampler=np.random.default_rng(2026+rd*10+list(splits).index(s))
            positive_sampler=np.random.default_rng(2026+list(splits).index(s))
            pairs=np.array([pos[k] for k in positive_sampler.choice(len(pos),48,replace=False)]+[neg[k] for k in sampler.choice(len(neg),96,replace=False)],dtype=int)
            pairs=pairs[sampler.permutation(len(pairs))]
            assert len(set(map(tuple,pairs[:,:2])))==len(pairs)
            assert all((rr[i]['identity']==rr[j]['identity'])==(y==1) for i,j,y in pairs)
            r=((x[idx][pairs[:,0]]-x[idx][pairs[:,1]])**2).sum(1);q=((z[idx][pairs[:,0]]-z[idx][pairs[:,1]])**2).sum(1)
            assert min(r.min(),q.min())>0
            pairs_by_split[s]=pairs
            checks[s]=dict(positive=48,negative=96,cross_sequence_positives=48,same_category_negatives=sum(rr[i]['category']==rr[j]['category'] for i,j,y in pairs if y==-1),min_rgb_distance=float(r.min()),min_depth_distance=float(q.min()))
        np.savez(folder/'pairs.npz',**pairs_by_split)
        train=splits['train'];ii,jj=np.triu_indices(len(train),1)
        bounds=np.quantile(np.linalg.norm(x[train][ii]-x[train][jj],axis=1),[.03,.97])
        np.save(folder/'xi0.npy',np.where(pairs_by_split['train'][:,2]==1,*bounds))
        result['rounds'].append(dict(round=rd,pair_checks=checks,feasibility=feasibility(x[train],z[train],pairs_by_split['train'],folder)))
        (args.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result['rounds'][-1]),flush=True)
    coverage=np.array([a[2] for a in extracted])
    result['depth_quality']=dict(samples_below_10_percent=int((coverage<.1).sum()),
        samples_below_25_percent=int((coverage<.25).sum()),samples_below_50_percent=int((coverage<.5).sum()),
        median_valid_fraction=float(np.median(coverage)))
    result['feature_definition']=dict(rgb='4x4 cells: masked RGB means in [0,1] and mask occupancy; 64 dimensions',
        depth='4x4 cells: mean of depth/within-image valid median - 1, plus valid depth occupancy; 32 dimensions',
        missing='depth zero excluded; empty cell mean zero with explicit occupancy; no sample removed',
        normalization='per-image relative depth, no physical unit assumption; training-only centered unwhitened PCA')
    result['manifest_sha256']=hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    result['versions']=dict(numpy=np.__version__,cvxpy=cp.__version__)
    (args.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__': main()
