"""Export local research artifacts with source/metric folders and SHA256 provenance."""
import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

PLOTS = {'comparison.png','diagnostics.png','training_time.png','sample_count_curve.png',
         'distance_analysis.png','partial_curve.png','retrieval_cmc.png',
         'matched_parameter_partial.png','quiet_training_time.png','body_distance_analysis.png'}
CHUNK = 40 * 1024 * 1024

def source(path):
    s = path.as_posix().lower()
    if 'external_rgbd_' in s:
        if any(x in s for x in ('weight','body','action')): return 'weightgait'
        if any(x in s for x in ('face_input','face_controls','face_secondary','face_study','face_prepare','facegrabber','robustness')): return 'facegrabber'
        return 'cross_dataset'
    if 'orl_glbp' in s or 'olivetti' in s: return 'orl_olivetti'
    if any(x in s for x in ('washington','depth_augmented','depth_improvement','depth_selected')): return 'washington'
    if 'kdes_source' in s or 'glbp_source' in s: return 'feature_validation'
    return 'synthetic_and_formula_validation'

def metric(path):
    s = path.as_posix().lower(); n = path.name.lower()
    if 'source_snapshot' in s or path.suffix == '.py': return 'source_snapshots_and_scripts'
    if any(x in s for x in ('attempt_v1','failed_size','redundant_attempt')): return 'failed_attempts'
    if any(x in s for x in ('robustness','secondary_depth')): return 'robustness'
    if 'controls' in s or 'supplemental' in s: return 'mechanism_controls'
    if 'partial_' in s or 'matched_parameter_partial' in s: return 'privileged_ratio'
    if 'sample_count' in s: return 'training_pair_count'
    if 'distance_analysis' in s: return 'distance_distribution'
    if any(x in n for x in ('training_time','timing','quiet_training','times')): return 'training_time'
    if any(x in s for x in ('history','convergence','constraints','corrected_acceptance','formula_reaudit','review_20261031_corrected','diagnostics','acceptance')): return 'convergence_kkt'
    if any(x in n for x in ('check','audit','integrity','sha256','provenance','revision','qc','receipt')): return 'integrity_and_provenance'
    if any(x in n for x in ('features','pca','pairs','xi0','bounds','split','manifest','selection','basis','codebook','raw_glbp','kdes.npz')): return 'inputs_features_and_protocol'
    if n.endswith('_model.npz') or n == 'model.npz' or 'model' in n: return 'models'
    if 'cmc' in s or 'rank' in s or 'action_final' in s or 'weight_final' in s: return 'rank1_cmc'
    if 'final_report' in s: return 'joint_metrics_and_report'
    if any(x in n for x in ('results','summary','comparison','search','itml','euclidean','svm','lmnn')):
        return 'rank1_cmc' if source(path) == 'weightgait' else 'ap_auc' if source(path) in ('facegrabber','orl_olivetti','washington') else 'joint_metrics_and_report'
    return 'protocol_and_logs'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path)
    a=p.parse_args();root=a.root.resolve();out=a.output or root/'experiments';out.mkdir(exist_ok=False)
    runs=root/'itml_reproduction/runs'
    candidates=list(runs.rglob('*'))
    # Acquisition receipts describe sources without copying downloaded image/video archives.
    candidates+=list((root/'itml_reproduction/data').rglob('*.receipt.json'))
    synthetic=root/'itml_reproduction/data/synthetic'
    if synthetic.exists():candidates+=list(synthetic.iterdir())
    records=[];excluded=[];seen={};totals=Counter()
    for f in sorted(candidates):
        if not f.is_file():continue
        rel=f.relative_to(root);s=rel.as_posix();size=f.stat().st_size
        if '__pycache__' in f.parts or f.name=='.DS_Store' or f.suffix=='.pyc':reason='regenerable cache'
        elif f.suffix.lower() in {'.png','.jpg','.jpeg','.pgm','.pdf'} and f.name not in PLOTS:reason='raw/sample/QA imagery or paper page; retained locally'
        elif f.suffix=='.npz' and ('attempt_v1' in s or 'redundant_attempt' in s):reason='superseded failed-attempt binary cache; metadata and failure records retained'
        elif f.suffix.lower() not in {'.json','.csv','.npz','.npy','.png','.md','.txt','.py','.log'}:reason='non-result auxiliary file'
        else:reason=None
        if reason:
            excluded.append(dict(original_path=s,bytes=size,reason=reason));continue
        digest=hashlib.sha256(f.read_bytes()).hexdigest();src=source(rel);kind=metric(rel)
        record=dict(original_path=s,data_source=src,metric_group=kind,bytes=size,sha256=digest)
        if digest in seen:
            record['duplicate_of']=seen[digest]
        else:
            dest=out/src/kind/rel.relative_to('itml_reproduction');dest.parent.mkdir(parents=True,exist_ok=True)
            if size>CHUNK:
                parts=[]
                with f.open('rb') as stream:
                    i=0
                    while block:=stream.read(CHUNK):
                        target=dest.with_name(dest.name+f'.part{i:03d}');target.write_bytes(block)
                        parts.append(dict(path=target.relative_to(out).as_posix(),bytes=len(block),sha256=hashlib.sha256(block).hexdigest()));i+=1
                record['parts']=parts
            else:
                shutil.copy2(f,dest);record['stored_path']=dest.relative_to(out).as_posix()
            seen[digest]=s;totals[src]+=size
        records.append(record)
    manifest=dict(format_version=1,created_date='2026-11-01',repository='https://github.com/xjyxhh/COMP3314',
        policy='All result metadata and useful final binary artifacts; no person images/videos/download archives. Exact-byte deduplication; failed-attempt binary caches excluded with reasons.',
        records=records,excluded=excluded,unique_bytes_by_source=dict(totals))
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (out/'README.md').write_text('''# 实验数据与检验指标

按数据来源与指标归档；正收益、负收益和失败候选的记录均保留。manifest.json逐文件记录原始路径、SHA256、存放位置、重复关系或排除原因。混合指标的报告放在cross_dataset或joint_metrics_and_report中，不将不同任务AP合并。模型、共享输入及源码快照各有独立目录。

|数据来源|内容|
|---|---|
|facegrabber|真实传感器深度人脸验证、机制对照、10个重叠人物划分|
|weightgait|展示深度人物检索、跨片段/负重动作；不是原BIWI Walking/Still|
|orl_olivetti|清晰图像特权信息，包括早期负收益及后续G-LBP扩展|
|washington|RGB-D物体实例验证及深度改进，非人物任务|
|synthetic_and_formula_validation|合成流程、Partial、凸解及公式核验|
|feature_validation|G-LBP/KDES实现与标量对照|
|cross_dataset|跨数据集报告、计时、独立核验及来源记录|

各数据来源下的ap_auc、rank1_cmc、privileged_ratio、training_pair_count、distance_distribution、training_time、convergence_kkt等目录对应检验主题；inputs_features_and_protocol、models、integrity_and_provenance保存复算依据。一个文件同时包含多个指标时原样保留一次，不拆改原实验文件。

同内容文件只保存一份。超过40 MiB文件按字节分块。执行`python tools/restore_experiments.py --output /path/to/new_directory`，可校验并恢复被纳入清单的文件到原始相对目录；目标必须不存在。未纳入的人物图像、下载归档、失败尝试的大型二进制缓存仍在原工作区，排除原因见manifest。图表保留，人物示例/质量检查图不上传。

恢复后原实验报告的相对链接和复算脚本可使用原目录结构；部分记录仍含原机器绝对路径，跨机器运行需指定新路径。原始数据及外部依赖须按项目说明另行取得，不能把“可还原实验产物”当作完整原论文数值复现。
''')
    print(json.dumps(dict(included=len(records),excluded=len(excluded),unique_files=len(seen),unique_MiB=round(sum(totals.values())/1024**2,2),sources=dict(totals)),indent=2))
if __name__=='__main__':main()
