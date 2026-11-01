"""Evaluate frozen face-verification models after ALL validation searches finish."""
import argparse
import json
import platform
from pathlib import Path
from time import perf_counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .baselines import decision_scores
from .baseline_compare import METHODS,train,training_labels
from .evaluation import verification_scores
from .metrics import distances
from .paper_analyses import variants
from .run import save_json,sha


def finalize(inputs,baseline,analyses,out):
    rounds=sorted(inputs.glob('round_*/pairs.npz'))
    base=json.loads((baseline/'results.json').read_text())
    extra=json.loads((analyses/'results.json').read_text())
    if len(base)!=len(rounds)*7 or len(extra)!=len(rounds)*len(variants()):
        raise ValueError('All baseline and analysis searches must finish before test evaluation')
    for root,records in ((baseline,base),(analyses,extra)):
        if any(r['status']!='completed' for r in records):
            raise ValueError('Incomplete selected model set')
        integrity=json.loads((root/'integrity.json').read_text())
        if not integrity['source_unchanged']:
            raise ValueError('Training input integrity failed')
        protocol=json.loads((root/'protocol.json').read_text())
        if any(sha(inputs/n)!=h for n,h in protocol['input_sha256'].items()):
            raise ValueError('Input changed after model selection')
    # Test arrays are accessed only after the completion/hash barrier above.
    with np.load(inputs/'features_pca.npz',allow_pickle=False) as data:
        X,Z,test=data['X_train'],data['Z_train'],data['X_test']
    out.mkdir(parents=True,exist_ok=False)
    config=json.loads((baseline/'protocol.json').read_text())['config']
    labels=training_labels(inputs,len(X))
    evaluated=[]
    for r in base:
        with np.load(inputs/r['round']/'pairs.npz') as p:test_pairs=p['test']
        path=baseline/r['round']/f"{r['method']}_model.npz"
        with np.load(path) as model:score=decision_scores(test,test_pairs,model)
        evaluated.append(dict(round=r['round'],method=r['method'],kind='main',
            test=verification_scores(score,test_pairs[:,2]),model_sha256=sha(path)))
    for r in extra:
        with np.load(inputs/r['round']/'pairs.npz') as p:test_pairs=p['test']
        path=analyses/r['folder']/'model.npz'
        with np.load(path) as model:score=decision_scores(test,test_pairs,model)
        evaluated.append(dict(round=r['round'],method=r['method'],kind=r['kind'],fraction=r['fraction'],
            test=verification_scores(score,test_pairs[:,2]),model_sha256=sha(path)))
    save_json(out/'test_results.json',evaluated)
    summary=[]
    for kind in ('main','partial','sample_count'):
        for method in METHODS:
            selected=[r for r in evaluated if r['kind']==kind and r['method']==method]
            for fraction in sorted({r.get('fraction',1.) for r in selected}):
                rows=[r for r in selected if r.get('fraction',1.)==fraction]
                summary.append(dict(kind=kind,method=method,fraction=fraction,rounds=len(rows),
                    **{m:dict(mean=float(np.mean([r['test'][m] for r in rows])),
                              std=float(np.std([r['test'][m] for r in rows],ddof=0))) for m in ('AP','AUC')}))
    save_json(out/'summary.json',summary)
    # V-D3: fixed first 200 positive/negative training pairs; no cherry picking.
    with np.load(rounds[0]) as p:training_pairs=p['train']
    chosen=np.r_[np.flatnonzero(training_pairs[:,2]==1)[:200],np.flatnonzero(training_pairs[:,2]==-1)[:200]]
    if len(chosen)!=400:
        raise ValueError('Distance analysis needs at least 200 training pairs per label')
    figure,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    arrays=dict(pair_indices=chosen,pairs=training_pairs[chosen])
    for ax,method in zip(axes,('itml','itml_plus')):
        with np.load(baseline/rounds[0].parent.name/f'{method}_model.npz') as model:
            d=distances(X,training_pairs[chosen],model['M'])
        arrays[method]=d
        ax.scatter(np.arange(200),d[:200],s=9,c='#b24536',label='same identity')
        ax.scatter(np.arange(200,400),d[200:],s=9,c='#28546b',label='different identity')
        ax.axhline(d[:200].max(),ls='--',color='#b24536',lw=.8)
        ax.axhline(d[200:].min(),ls='--',color='#28546b',lw=.8)
        ax.set(title=method,xlabel='Fixed training pair index',ylabel='Squared learned distance')
        ax.legend(fontsize=8)
    figure.tight_layout();figure.savefig(out/'distance_analysis.png',dpi=160);plt.close(figure)
    np.savez(out/'distance_analysis.npz',**arrays)
    # V-D4: fixed selected parameters, fit-only wall time, same process/hardware.
    times=[]
    for record in base:
        if record['method'] not in ('lmnn','itml','itml_s','itml_plus'):continue
        with np.load(inputs/record['round']/'pairs.npz') as f:pairs=f['train']
        xi0=np.load(inputs/record['round']/'xi0.npy')
        start=perf_counter()
        model=train(record['method'],X,Z,labels,pairs,xi0,record['selected']['parameters'],config)
        elapsed=perf_counter()-start
        with np.load(baseline/record['round']/f"{record['method']}_model.npz") as old:
            same=bool(np.allclose(model['M'],old['M'],rtol=1e-10,atol=1e-10))
        times.append(dict(round=record['round'],method=record['method'],fit_seconds=elapsed,model_matches=same))
        print('timing',record['round'],record['method'],round(elapsed,3),flush=True)
    save_json(out/'training_times.json',dict(platform=platform.platform(),python=platform.python_version(),
        scope='local backend timing, cannot reproduce author hardware/software seconds',records=times))
    plot_summary(summary,times,out)
    source=json.loads((inputs/'protocol.json').read_text()) if (inputs/'protocol.json').exists() else {}
    lines=['# 固定协议扩展实验结果','',
        '**范围：'+source.get('scope','未声明数据源；仅作流程诊断')+'。不是原论文数值复现。**',
        f'固定输入，{len(rounds)}轮重抽负对，特征及参数网格见输入protocol与搜索protocol。',
        '本地ORL人物已在既往实验出现；以下只作探索性结果。标准差是负对重抽的描述性波动，不是独立人物泛化的置信区间。','',
        '|方法|AP均值±标准差 (%)|AUC均值±标准差 (%)|','|---|---:|---:|']
    for row in summary:
        if row['kind']=='main':
            lines.append(f"|{row['method']}|{100*row['AP']['mean']:.2f} ± {100*row['AP']['std']:.2f}|{100*row['AUC']['mean']:.2f} ± {100*row['AUC']['std']:.2f}|")
    main={r['method']:r for r in summary if r['kind']=='main'}
    effect=100*(main['itml_plus']['AP']['mean']-main['itml']['AP']['mean'])
    lines+=['',f'原版ITML+相对ITML的平均AP差为 {effect:+.3f} 个百分点。结果按实际差值报告，不要求出现正收益。',
        '各候选和选定模型的停止原因/残差见训练目录；有限预算及LMNN数值停止不等于全局最优。',
        '未根据这些测试成绩再次选参或更改特征。不进行将相关pairs当独立样本的显著性检验。',
        '四项分析：partial_curve.png、sample_count_curve.png、distance_analysis.png、training_time.png。',
        '距离图使用训练对，不能单独证明减少泛化过拟合；本机时间不能验证作者时间比例。',
        '行人重识别没有对应真实RGB-D数据，不能把人脸/物体结果称为BIWI验证。']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    save_json(out/'completion.json',dict(test_models=len(evaluated),all_timing_models_match=all(r['model_matches'] for r in times),
        source_sha256=sha(Path(__file__)),status='completed_extension',original_paper_reproduction=False))


def plot_summary(summary,times,out):
    for kind,name in [('partial','partial_curve.png'),('sample_count','sample_count_curve.png')]:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for method in ('itml','itml_plus'):
            rows=sorted((r for r in summary if r['kind']==kind and r['method']==method),key=lambda r:r['fraction'])
            if not rows:continue
            for ax,metric in zip(axes,('AP','AUC')):
                ax.errorbar([100*r['fraction'] for r in rows],[100*r[metric]['mean'] for r in rows],
                            yerr=[100*r[metric]['std'] for r in rows],marker='o',capsize=3,label=method)
                ax.set(xlabel='Privileged pairs (%)' if kind=='partial' else 'Training pairs (%)',ylabel=metric+' (%)')
                ax.legend();ax.grid(alpha=.15)
        fig.tight_layout();fig.savefig(out/name,dpi=160);plt.close(fig)
    names=['lmnn','itml','itml_s','itml_plus']
    values=[[r['fit_seconds'] for r in times if r['method']==name] for name in names]
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(names,[np.mean(v) for v in values],yerr=[np.std(v) for v in values],capsize=4,color='#28546b')
    ax.set(ylabel='Selected-model fit time (seconds)',title=f"Local numerical backends; {len({r['round'] for r in times})} rounds")
    fig.tight_layout();fig.savefig(out/'training_time.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('input','baseline','analyses','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();finalize(a.input,a.baseline,a.analyses,a.output)
