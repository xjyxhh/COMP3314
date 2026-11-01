"""Final independent audit, quiet serial timing, cross-clip retrieval and report."""
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itml_repro.baseline_compare import train,training_labels,METHODS
from itml_repro.baselines import decision_scores
from itml_repro.run import save_json,sha

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
FACE=HERE/'face_input_v2'

def read(path):return json.loads(path.read_text())

def independent_scores(X,pairs,model):
    a,b=X[pairs[:,0]],X[pairs[:,1]]
    if 'M' in model:
        M=model['M'];assert np.linalg.eigvalsh(M).min()>-1e-7
        delta=a-b;return -np.sum((delta@M)*delta,axis=1)
    g=np.exp(-.5*((a+b)/2)**2)/np.sqrt(2*np.pi)
    return np.column_stack([np.abs(a-b)*g,a*b*g])@model['w']+model['b']

def independent_metrics(score,labels):
    positive=labels==1;pos,neg=score[positive],score[~positive]
    auc=float(np.mean((pos[:,None]>neg).astype(float)+.5*(pos[:,None]==neg)))
    _,inverse=np.unique(score,return_inverse=True)
    tp=np.bincount(inverse,weights=positive)[::-1];all_count=np.bincount(inverse)[::-1]
    ap=float(np.sum(tp*np.cumsum(tp)/np.cumsum(all_count))/positive.sum())
    return dict(AP=ap,AUC=auc)

def independent_retrieval(X,labels,model,shots):
    queries=np.array([np.flatnonzero(labels==n)[0] for n in dict.fromkeys(labels.tolist())])
    gallery=np.setdiff1d(np.arange(len(labels)),queries)
    pairs=np.array([(i,j,1 if labels[i]==labels[j] else -1) for i in queries for j in gallery])
    score=independent_scores(X,pairs,model).reshape(len(queries),len(gallery))
    ranks=[]
    for scores,label in zip(score,labels[queries]):
        order=np.argsort(-scores,kind='stable');ranks.append(np.flatnonzero(labels[gallery][order]==label)[0]+1)
    return np.mean(np.array(ranks)[:,None]<=np.arange(1,len(gallery)+1)[None,:],axis=0)

def audit_face():
    expected=read(FACE/'final/test_results.json');analyses=read(FACE/'analyses/results.json');max_error=0.
    for directory in ('baselines_final','analyses'):
        protocol=read(FACE/directory/'protocol.json')
        assert all(sha(FACE/n)==h for n,h in protocol['input_sha256'].items())
    with np.load(FACE/'features_pca.npz') as f:X=f['X_test']
    for row in expected:
        if row['kind']=='main':path=FACE/'baselines_final'/row['round']/f"{row['method']}_model.npz"
        else:
            selected=next(a for a in analyses if all(a[k]==row[k] for k in ('round','kind','method','fraction')))
            path=FACE/'analyses'/selected['folder']/'model.npz'
        assert sha(path)==row['model_sha256']
        with np.load(FACE/row['round']/'pairs.npz') as p:pairs=p['test']
        with np.load(path) as model:metrics=independent_metrics(independent_scores(X,pairs,model),pairs[:,2])
        error=max(abs(metrics[m]-row['test'][m]) for m in metrics);assert error<1e-10,(row,error)
        max_error=max(max_error,error)
    return dict(models=len(expected),max_metric_error=max_error)

def audit_body_and_cross(out):
    inputs=HERE/'weight_input';base=HERE/'weight_search';expected=read(HERE/'weight_final/results.json')
    protocol=read(base/'protocol.json');assert all(sha(inputs/n)==h for n,h in protocol['input_sha256'].items())
    with np.load(inputs/'features_pca.npz') as f:
        tests={s:(f['X_'+s],f['labels_'+s]) for s in ('normal_A','normal_B')}
    max_error=0.;cross=[]
    for row in read(base/'results.json'):
        path=base/row['round']/f"{row['method']}_model.npz"
        with np.load(path) as model:
            for setting,(X,labels) in tests.items():
                cmc=independent_retrieval(X,labels,model,20)
                actual=next(a for a in expected if all(a[k]==row[k] for k in ('round','method')) and a['setting']==setting)
                error=float(np.max(np.abs(cmc-actual['CMC'])));assert error<1e-12
                max_error=max(error,max_error)
            for a,b in [('normal_A','normal_B'),('normal_B','normal_A')]:
                X,labels=tests[a];G,gl=tests[b];q=np.array([np.flatnonzero(labels==n)[0] for n in dict.fromkeys(labels.tolist())])
                combined=np.vstack([X,G]);pairs=np.array([(i,len(X)+j,1 if labels[i]==gl[j] else -1) for i in q for j in range(len(G))])
                scores=independent_scores(combined,pairs,model).reshape(len(q),len(G))
                order=np.argsort(-scores,axis=1,kind='stable');match=gl[order]==labels[q,None]
                cmc=np.maximum.accumulate(match,axis=1).mean(0)
                cross.append(dict(round=row['round'],method=row['method'],direction=a+'_to_'+b,rank1=float(cmc[0]),CMC=cmc.tolist(),
                    model_sha256=sha(path),scope='different heldout clips, same normal condition; no shared sample/clip on query/gallery sides'))
    save_json(out/'cross_clip_results.json',cross)
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,setting in zip(axes,tests):
        for method in METHODS:
            values=np.array([r['CMC'] for r in expected if r['method']==method and r['setting']==setting])
            ax.plot(np.arange(1,31),100*values.mean(0)[:30],label=method)
        ax.set(title=setting,xlabel='Rank (first 30 shown)',ylim=(0,102));ax.grid(alpha=.2)
    axes[0].set_ylabel('Recognition (%)');axes[1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'retrieval_cmc.png',dpi=170);plt.close(fig)
    return dict(evaluations=len(expected),max_CMC_error=max_error,cross_clip_evaluations=len(cross))

def quiet_timing(out):
    times=[]
    for kind,inputs,base in [('face',FACE,FACE/'baselines_final'),('body',HERE/'weight_input',HERE/'weight_search')]:
        records=read(base/'results.json');config=read(base/'protocol.json')['config']
        with np.load(inputs/'features_pca.npz') as f:
            X,Z=f['X_train'],f['Z_train'];labels=f['labels_train'] if kind=='body' else training_labels(inputs,len(X))
        for r in records:
            if r['method'] not in ('itml','lmnn','itml_s','itml_plus'):continue
            with np.load(inputs/r['round']/'pairs.npz') as f:pairs=f['train']
            xi0=np.load(inputs/r['round']/'xi0.npy');start=time.perf_counter()
            model=train(r['method'],X,Z,labels,pairs,xi0,r['selected']['parameters'],config)
            seconds=time.perf_counter()-start
            with np.load(base/r['round']/f"{r['method']}_model.npz") as old:
                match=bool(np.allclose(model['M'],old['M'],rtol=1e-10,atol=1e-10))
            assert match
            times.append(dict(dataset=kind,round=r['round'],method=r['method'],fit_seconds=seconds,model_matches=True))
            print('quiet timing',kind,r['round'],r['method'],round(seconds,2),flush=True)
    save_json(out/'quiet_training_times.json',dict(records=times,platform=platform.platform(),
        scope='Sequential refits after all our main/robustness/control training completed; other user/system load not controlled; original author hardware/backend not reproduced'))
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    methods=['lmnn','itml','itml_s','itml_plus']
    for ax,kind in zip(axes,('face','body')):
        values=[[r['fit_seconds'] for r in times if r['dataset']==kind and r['method']==m] for m in methods]
        ax.bar(methods,[np.mean(v) for v in values],yerr=[np.std(v) for v in values],capsize=3,color='#28546b')
        ax.set(title=kind+' selected-model refit',ylabel='Seconds')
    fig.tight_layout();fig.savefig(out/'quiet_training_time.png',dpi=170);plt.close(fig)


def audit_actions():
    expected=read(HERE/'action_final/results.json');max_error=0.
    with np.load(HERE/'action_features.npz') as f:
        X,labels,subsets=f['X'],f['labels'],f['subsets']
    for row in expected:
        keep=subsets==row['setting']
        with np.load(HERE/'weight_search'/row['round']/f"{row['method']}_model.npz") as model:
            cmc=independent_retrieval(X[keep],labels[keep],model,20)
        error=float(np.max(np.abs(cmc-row['CMC'])));assert error<1e-12
        max_error=max(error,max_error)
    return dict(frozen_condition_evaluations=len(expected),max_CMC_error=max_error,new_training=False)

def main():
    targets=[FACE/'final/completion.json',HERE/'weight_final/summary.json',HERE/'robustness/completion.json',
             HERE/'face_controls/completion.json',HERE/'weight_controls/completion.json',HERE/'action_final/completion.json']
    while not all(p.exists() for p in targets):time.sleep(10)
    out=HERE/'final_report';out.mkdir(exist_ok=False)
    previous=read(HERE/'previous_results_sha256.json')
    assert all(sha(PROJECT/n)==h for n,h in previous.items())
    audit=dict(previous_result_files_unchanged=len(previous),face=audit_face(),body=audit_body_and_cross(out),extra_actions=audit_actions())
    save_json(out/'independent_audit.json',audit)
    quiet_timing(out)
    face={r['method']:r for r in read(FACE/'final/summary.json') if r['kind']=='main'}
    body=read(HERE/'weight_final/summary.json');cross=read(out/'cross_clip_results.json');robust=read(HERE/'robustness/completion.json')
    lines=['# 替代数据完整实验结果','',
        '原有ORL最终11个结果文件哈希均未改变。新增真实FaceGrabber人脸研究及WeightGait展示深度人物检索扩展；不称原论文数值复现。','',
        '|方法|FaceGrabber AP (%)|FaceGrabber AUC (%)|Weight normal_A Rank-1 (%)|Weight normal_B Rank-1 (%)|','|---|---:|---:|---:|---:|']
    for m in METHODS:
        a=next(r for r in body if r['method']==m and r['setting']=='normal_A');b=next(r for r in body if r['method']==m and r['setting']=='normal_B')
        lines.append(f"|{m}|{100*face[m]['AP']['mean']:.2f} ± {100*face[m]['AP']['std']:.2f}|{100*face[m]['AUC']['mean']:.2f} ± {100*face[m]['AUC']['std']:.2f}|{100*a['mean_rank1']:.2f} ± {100*a['std_rank1']:.2f}|{100*b['mean_rank1']:.2f} ± {100*b['std_rank1']:.2f}|")
    effect=100*(face['itml_plus']['AP']['mean']-face['itml']['AP']['mean'])
    lines+=['',f'FaceGrabber ITML+相对ITML平均AP差：{effect:+.3f}个百分点。五轮为固定人物划分的负对重抽；表中标准差仅作描述，不能当作独立人物置信区间。','',
        f"额外10个人物划分，ITML+优于ITML的划分有{robust['positive_partitions']}/10；AP差均值{robust['AP_difference_mean']:+.3f}、中位数{robust['AP_difference_median']:+.3f}个百分点，范围[{robust['AP_difference_min']:+.3f},{robust['AP_difference_max']:+.3f}]。各划分重叠，不做独立样本t检验。",'',
        '|方法|normal_A首帧 → normal_B全部帧 Rank-1 (%)|反方向 Rank-1 (%)|','|---|---:|---:|']
    for m in METHODS:
        values=[[r['rank1'] for r in cross if r['method']==m and r['direction']==a+'_to_'+b] for a,b in [('normal_A','normal_B'),('normal_B','normal_A')]]
        lines.append(f'|{m}|{100*np.mean(values[0]):.2f}|{100*np.mean(values[1]):.2f}|')
    actions=read(HERE/'action_final/summary.json');action_cross=read(HERE/'action_final/cross_action_results.json')
    lines+=['','## 冻结模型的跨动作检验','',
        '模型和参数仍只由正常行走训练/验证确定。额外读取单腿负重、双腿负重的RGB，沿用既有训练码本与PCA，不使用这些条件的深度，不重新训练或选参。',
        '|方法|单腿负重本条件Rank-1 (%)|双腿负重本条件Rank-1 (%)|正常→单腿负重 (%)|正常→双腿负重 (%)|','|---|---:|---:|---:|---:|']
    for m in METHODS:
        values=[next(r['mean_rank1'] for r in actions if r['method']==m and r['setting']==s) for s in ('single_leg_load','both_leg_load')]
        cross_values=[np.mean([r['rank1'] for r in action_cross if r['method']==m and r['direction']=='normal_A_to_'+s]) for s in ('single_leg_load','both_leg_load')]
        lines.append(f'|{m}|{100*values[0]:.2f}|{100*values[1]:.2f}|{100*cross_values[0]:.2f}|{100*cross_values[1]:.2f}|')
    lines.append('反方向结果及完整CMC保存在../action_final/cross_action_results.json。负重动作变化仍不能替代原BIWI的Still或跨日换装。')
    for kind in ('face','weight'):
        results=read(HERE/(kind+'_controls')/'test_results.json');training=read(HERE/(kind+'_controls')/'training_results.json')
        lines+=['',f'## {kind} 同参数机制对照','', '|对照|'+('AP (%)' if kind=='face' else 'normal_A / normal_B Rank-1 (%)')+'|','|---|---:|']
        names=['no_PI_matched_gamma','shuffle_PI','random_PI','duplicate_RGB_PI','extended_100_sweeps']
        for name in names:
            chosen=[r for r in results if r['variant']==name or r['variant'].startswith(name+'_')]
            if kind=='face':value=f"{100*np.mean([r['AP'] for r in chosen]):.2f}" if chosen else '全部拟合失败'
            else:value=' / '.join(f"{100*np.mean([r['rank1'] for r in chosen if r['setting']==s]):.2f}" for s in ('normal_A','normal_B')) if chosen else '全部拟合失败'
            lines.append('|'+name+'|'+value+'|')
        completed=sum(r['status']=='completed' for r in training)
        lines.append(f'预定45次拟合完成{completed}/45。打乱/随机对照各使用3个固定种子，表格汇总全部种子与轮次，无按测试成绩挑选。100遍为额外诊断，未替换主比较。逐轮停止原因和残差保存在controls目录。')
    lines+=['','## 图表与证据','',
        '- 人脸比例、距离图：../face_input_v2/final/；人物CMC：retrieval_cmc.png。',
        '- 等待本次所有重训练完成后，额外串行复算40次训练时间；所有度量模型均匹配。见quiet_training_times.json及quiet_training_time.png。Face初次流水线中的时间受其他实验负载影响，采用此后测量为主要本机时间结果。',
        '- 110个人脸模型、70个正常人物条件评估及70个负重动作条件评估均独立复算；输入和模型哈希检查通过，见independent_audit.json。',
        '- 输入检查、来源、固定方案、失败预处理尝试与源码快照保留在上级目录；原有结果不变。','',
        '## 结论边界','',
        'FaceGrabber使用真实传感器深度，但采用新的RGB眼点检测和点云投影/插值预处理，人数/图片及样本对数量不同于原论文。WeightGait的特权输入是有损彩色深度展示，不是原始米制深度；两项测试均是正常行走，不包含原BIWI的Still或跨日换装条件。原作者未公开设置和KDES历史版本仍无法确认。',
        '因此这些结果用于检验替代条件下的方法收益、机制和稳定性。出现提升、无提升或负收益均按实际报告，不把距离图、单次正收益或未收敛的模型当作完整泛化证明。']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    save_json(out/'completion.json',dict(status='completed_available_alternative_studies',original_paper_numerical_reproduction=False,
        face_main_candidates=700,body_main_candidates=700,robustness_candidates=900,face_frozen_models=110,body_main_models=35,
        cross_clip_evaluations=70,control_fits=90,quiet_timing_fits=40,previous_results_unchanged=True,
        extra_action_within_condition_evaluations=70,extra_cross_action_evaluations=140,
        finalizer_sha256=sha(Path(__file__))))
    print('ALL EXTERNAL STUDIES AND FINAL AUDIT COMPLETED',flush=True)

if __name__=='__main__':main()
