"""Descriptive generalization gaps and classifier numerical score sensitivity.

No model/parameter selection. Main scores remain unchanged and are retained.
"""
import json
import runpy
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itml_repro.evaluation import verification
from itml_repro.run import save_json

HERE=Path(__file__).resolve().parent

def main():
    out=HERE/'final_report'
    if not (out/'completion.json').exists():raise ValueError('Full study must complete first')
    reference=runpy.run_path(str(HERE/'finish_external.py'))
    independent=reference['independent_scores'];errors=[];extra_checked=0
    for kind,inputs in [('face',HERE/'face_input_v2'),('weight',HERE/'weight_input')]:
        rows=json.loads((HERE/(kind+'_controls')/'test_results.json').read_text())
        with np.load(inputs/'features_pca.npz') as f:
            arrays={n:f[n] for n in f.files if n.startswith(('X_','labels_'))}
        for row in rows:
            path=HERE/(kind+'_controls')/row['round']/(row['variant']+'_model.npz')
            with np.load(path) as model:
                if kind=='face':
                    with np.load(inputs/row['round']/'pairs.npz') as p:pairs=p['test']
                    got=reference['independent_metrics'](independent(arrays['X_test'],pairs,model),pairs[:,2])
                    errors.append(max(abs(got[m]-row[m]) for m in got))
                else:
                    s=row['setting'];got=reference['independent_retrieval'](arrays['X_'+s],arrays['labels_'+s],model,20)
                    errors.append(float(np.max(np.abs(got-row['CMC']))))
            extra_checked+=1
    for row in json.loads((HERE/'robustness/results.json').read_text()):
        inputs=HERE/'robustness'/f"seed_{row['seed']}"
        with np.load(inputs/'features_pca.npz') as f:X=f['X_test']
        with np.load(inputs/'round_00/pairs.npz') as f:pairs=f['test']
        for method,scores in row['scores'].items():
            with np.load(inputs/'search/round_00'/f'{method}_model.npz') as model:
                got=reference['independent_metrics'](independent(X,pairs,model),pairs[:,2])
            errors.append(max(abs(got[m]-scores[m]) for m in got));extra_checked+=1
    assert max(errors)<1e-10
    save_json(out/'supplemental_independent_audit.json',dict(evaluations=extra_checked,max_metric_error=max(errors),all_match=True))
    # Independently check both cross-action directions, retaining the frozen models.
    with np.load(HERE/'action_features.npz') as f:
        action_data={s:(f['X'][f['subsets']==s],f['labels'][f['subsets']==s]) for s in ('single_leg_load','both_leg_load')}
    with np.load(HERE/'weight_input/features_pca.npz') as f:
        action_data['normal_A']=(f['X_normal_A'],f['labels_normal_A'])
    cross_errors=[]
    for row in json.loads((HERE/'action_final/cross_action_results.json').read_text()):
        a,b=row['direction'].split('_to_');X,labels=action_data[a];G,gl=action_data[b]
        q=np.array([np.flatnonzero(labels==n)[0] for n in dict.fromkeys(labels.tolist())])
        pairs=np.array([(i,len(X)+j,1 if labels[i]==gl[j] else -1) for i in q for j in range(len(G))])
        with np.load(HERE/'weight_search'/row['round']/f"{row['method']}_model.npz") as model:
            scores=independent(np.vstack([X,G]),pairs,model).reshape(len(q),len(G))
        order=np.argsort(-scores,axis=1,kind='stable')
        cmc=np.maximum.accumulate(gl[order]==labels[q,None],axis=1).mean(0)
        cross_errors.append(float(np.max(np.abs(cmc-row['CMC']))))
    assert len(cross_errors)==140 and max(cross_errors)<1e-12
    save_json(out/'cross_action_independent_audit.json',dict(evaluations=len(cross_errors),max_CMC_error=max(cross_errors),all_match=True))
    solver_records=[]
    for dataset,base in [('face',HERE/'face_input_v2/baselines_final'),('weight',HERE/'weight_search')]:
        for row in json.loads((base/'results.json').read_text()):
            selected=row['selected']
            solver_records.append(dict(dataset=dataset,round=row['round'],method=row['method'],
                **{k:selected[k] for k in ('parameters','stop','final','diagnostics') if k in selected}))
    save_json(out/'main_solver_diagnostics.json',dict(records=solver_records,
        scope='Original stopping convention and finite iteration budget; stopping on norm change is not a constraint-feasibility or global-optimum certificate.'))
    gaps=[]
    folders=[('primary',HERE/'face_input_v2',HERE/'face_input_v2/baselines_final')]
    folders += [(p.name,p,p/'search') for p in sorted((HERE/'robustness').glob('seed_*'))]
    for kind,inputs,base in folders:
        for r in json.loads((base/'results.json').read_text()):
            if r['method'] not in ('itml','itml_plus'):continue
            with np.load(base/r['round']/f"{r['method']}_model.npz") as model,np.load(inputs/'features_pca.npz') as f,np.load(inputs/r['round']/'pairs.npz') as p:
                scores={n:verification(f['X_'+n],p[n],model['M']) for n in ('train','val','test')}
            gaps.append(dict(partition=kind,round=r['round'],method=r['method'],scores=scores,
                train_minus_test_AP_percentage_points=100*(scores['train']['AP']-scores['test']['AP'])))
    save_json(out/'generalization_gaps.json',dict(records=gaps,
        scope='Descriptive, matched positive prevalence; smaller gap alone can mean underfitting. Shared people/partitions are dependent. No causal overfitting proof.'))
    sensitivity=[];identity_groups=[]
    with np.load(HERE/'weight_input/features_pca.npz') as f:
        data={s:(f['X_'+s],f['labels_'+s]) for s in ('normal_A','normal_B')}
        train_ids=set(f['labels_train'].tolist())
    for r in json.loads((HERE/'weight_search/results.json').read_text()):
        with np.load(HERE/'weight_search'/r['round']/f"{r['method']}_model.npz") as model:
            for setting,(X,labels) in data.items():
                q=np.array([np.flatnonzero(labels==n)[0] for n in dict.fromkeys(labels.tolist())]);gallery=np.setdiff1d(np.arange(len(labels)),q)
                pairs=np.array([(i,j,1 if labels[i]==labels[j] else -1) for i in q for j in gallery])
                score=independent(X,pairs,model).reshape(len(q),len(gallery));values={}
                for name,scores in [('raw',score),('rounded6',np.round(score,6)),('rounded9',np.round(score,9))]:
                    order=np.argsort(-scores,axis=1,kind='stable')
                    values[name]=float(np.mean(labels[gallery][order[:,0]]==labels[q]))
                scale=np.ptp(score,axis=1,keepdims=True)
                normalized=(score-score.mean(axis=1,keepdims=True))/np.where(scale>0,scale,1.)
                order=np.argsort(-np.round(normalized,9),axis=1,kind='stable')
                values['relative_rounded9']=float(np.mean(labels[gallery][order[:,0]]==labels[q]))
                if r['method'] in ('svm','svm_plus'):
                    sensitivity.append(dict(round=r['round'],method=r['method'],setting=setting,rank1=values,
                        w_norm=float(np.linalg.norm(model['w'])),score_abs_max=float(np.max(np.abs(score))),score_range=float(np.ptp(score))))
                top=np.argsort(-score,axis=1,kind='stable')[:,0];hits=labels[gallery][top]==labels[q]
                seen=np.array([p in train_ids for p in labels[q]])
                identity_groups.append(dict(round=r['round'],method=r['method'],setting=setting,
                    trained_identity_queries=int(seen.sum()),untrained_identity_queries=int((~seen).sum()),
                    trained_identity_rank1=float(hits[seen].mean()),untrained_identity_rank1=float(hits[~seen].mean())))
    save_json(out/'classifier_score_sensitivity.json',dict(records=sensitivity,
        scope='Additional numerical diagnostic, not a replacement scoring rule; six/nine decimal rounding never selects parameters'))
    save_json(out/'retrieval_identity_groups.json',dict(records=identity_groups,
        scope='Same full gallery for both query groups. Untrained means absent from model training, but these identities appeared in validation; not never-seen confirmation.'))
    # Original V-D3 task type: fixed 200+200 body training pairs, not test pairs.
    with np.load(HERE/'weight_input/features_pca.npz') as f:X=f['X_train']
    with np.load(HERE/'weight_input/round_00/pairs.npz') as f:pairs=f['train']
    chosen=np.r_[np.flatnonzero(pairs[:,2]==1)[:200],np.flatnonzero(pairs[:,2]==-1)[:200]]
    arrays=dict(pair_indices=chosen,pairs=pairs[chosen]);fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,method in zip(axes,('itml','itml_plus')):
        with np.load(HERE/'weight_search/round_00'/f'{method}_model.npz') as model:d=-independent(X,pairs[chosen],model)
        arrays[method]=d
        ax.scatter(np.arange(200),d[:200],s=8,label='same identity',color='#b24536')
        ax.scatter(np.arange(200,400),d[200:],s=8,label='different identity',color='#28546b')
        ax.axhline(d[:200].max(),ls='--',color='#b24536',lw=.8)
        ax.axhline(d[200:].min(),ls='--',color='#28546b',lw=.8)
        ax.set(title=method,xlabel='Fixed body training pair index');ax.legend(fontsize=8)
    axes[0].set_ylabel('Squared learned distance');fig.tight_layout();fig.savefig(out/'body_distance_analysis.png',dpi=170);plt.close(fig)
    np.savez(out/'body_distance_analysis.npz',**arrays)
    # Same selected gamma at 0% removes the independently tuned endpoint change.
    summary=json.loads((HERE/'face_input_v2/final/summary.json').read_text())
    partial=sorted([r for r in summary if r['kind']=='partial'],key=lambda r:r['fraction'])
    controls=json.loads((HERE/'face_controls/test_results.json').read_text())
    matched=[r for r in controls if r['variant']=='no_PI_matched_gamma']
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,metric in zip(axes,('AP','AUC')):
        values=[r[metric]['mean'] for r in partial]
        values[0]=float(np.mean([r[metric] for r in matched]))
        ax.plot([100*r['fraction'] for r in partial],100*np.array(values),marker='o')
        ax.set(xlabel='Privileged pairs (%)',ylabel=metric+' (%)',title='Same main ITML+ gamma at all fractions');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'matched_parameter_partial.png',dpi=170);plt.close(fig)
    primary=[r for r in gaps if r['partition']=='primary']
    lines=['','## 补充：泛化差距与数值稳定性','',
        '|方法|训练AP (%)|验证AP (%)|测试AP (%)|训练−测试AP差 (百分点)|','|---|---:|---:|---:|---:|']
    for method in ('itml','itml_plus'):
        group=[r for r in primary if r['method']==method]
        values=[100*np.mean([r['scores'][n]['AP'] for r in group]) for n in ('train','val','test')]
        lines.append(f'|{method}|{values[0]:.2f}|{values[1]:.2f}|{values[2]:.2f}|{values[0]-values[2]:.2f}|')
    affected=[r for r in sensitivity if abs(r['rank1']['raw']-r['rank1']['rounded6'])>1e-12]
    relative_affected=[r for r in sensitivity if abs(r['rank1']['raw']-r['rank1']['relative_rounded9'])>1e-12]
    lines += ['',f'人物分类器共有{len(affected)}/{len(sensitivity)}个条件评估的Rank-1对绝对分数保留6位小数敏感；按每次查询的分数跨度归一化后保留9位小数，敏感条件为{len(relative_affected)}/{len(sensitivity)}。全部原始分数结果保留，细节见classifier_score_sensitivity.json。',
        '绝对小数位舍入依赖分数量纲，不能单独判定数值错误或模型无效；相对跨度检查也只作诊断，需结合优化残差和重复实验解释，未据此筛除或重选任何模型。',
        '机制对照固定主比较参数，用于单因素诊断；没有为每种对照重新调参，不能当作各对照最优性能的比较。',
        '泛化差距只作描述，较小差距也可能来自欠拟合；不能仅靠训练距离图或差距就证明减少过拟合。额外十划分的完整训练/验证/测试指标见generalization_gaps.json。',
        '另存人物训练对距离图body_distance_analysis.png及原始数组；同gamma的特权比例曲线见matched_parameter_partial.png，以补充独立选参0%端点的比较。',
        '延长迭代对照将上限提高至100遍，仍按原范数停止准则允许提前停止，不宣称所有模型取得全局最优。']
    lines.append(f'另独立核验{extra_checked}个对照/稳定性条件结果，最大指标差{max(errors):.3g}，见supplemental_independent_audit.json。')
    lines.append(f'另独立核验全部{len(cross_errors)}个双方向跨动作结果，最大CMC差{max(cross_errors):.3g}，见cross_action_independent_audit.json。')
    lines+=['','## 主比较的优化停止状态','',
        '|数据|方法|达到遍数/迭代上限的轮次|最终相对等式残差中位数|','|---|---|---:|---:|']
    for dataset in ('face','weight'):
        for method in ('itml','itml_plus','lmnn'):
            group=[r for r in solver_records if r['dataset']==dataset and r['method']==method]
            limited=sum(r.get('stop') in ('max_sweeps','max_iterations') for r in group)
            residual=[r['final']['relative_equality_residual'] for r in group if 'relative_equality_residual' in r.get('final',{})]
            value=f'{np.median(residual):.3g}' if residual else '—'
            lines.append(f'|{dataset}|{method}|{limited}/{len(group)}|{value}|')
    lines.append('完整绝对/相对约束残差、最小特征值、分类器最优性诊断及停止原因见main_solver_diagnostics.json。达到范数变化阈值不代表约束全部满足，投影LMNN也未取得全局最优证书；本次比较是声明预算和停止规则下的模型表现，不是所有方法全局最优性能的比较。')
    lines+=['','## 人物查询是否参与过模型训练','',
        '两组使用完全相同的完整图库，只区分查询人物是否参与过模型训练。未参与训练的这批人物曾用于验证选参，因此不能称为从未接触过的独立确认样本。',
        '|方法|normal_A训练人物 / 未训练人物 Rank-1 (%)|normal_B训练人物 / 未训练人物 Rank-1 (%)|','|---|---:|---:|']
    for method in dict.fromkeys(r['method'] for r in identity_groups):
        cells=[]
        for setting in ('normal_A','normal_B'):
            group=[r for r in identity_groups if r['method']==method and r['setting']==setting]
            cells.append(f"{100*np.mean([r['trained_identity_rank1'] for r in group]):.2f} / {100*np.mean([r['untrained_identity_rank1'] for r in group]):.2f}")
        lines.append('|'+method+'|'+'|'.join(cells)+'|')
    report=out/'RESULTS.md';report.write_text(report.read_text()+'\n'.join(lines)+'\n')
    save_json(out/'diagnostics_completion.json',dict(generalization_records=len(gaps),classifier_conditions=len(sensitivity),rounding6_sensitive_conditions=len(affected),relative_rounding9_sensitive_conditions=len(relative_affected)))
    print('Final descriptive diagnostics complete',flush=True)

if __name__=='__main__':main()
