"""Write the readable paper-order report only after all audits complete."""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]

def read(p):return json.loads(p.read_text())
def mean(rows,key):return float(np.mean([r[key] for r in rows]))

def main():
    final=HERE/'final_report'
    assert (final/'diagnostics_completion.json').exists()
    face=read(HERE/'face_input_v2/final/summary.json')
    main_face={r['method']:r for r in face if r['kind']=='main'}
    body=read(HERE/'weight_final/summary.json')
    robust=read(HERE/'robustness/completion.json')
    controls={k:read(HERE/(k+'_controls')/'test_results.json') for k in ('face','weight')}
    checks=read(HERE/'final_checks.json')
    fg=100*(main_face['itml_plus']['AP']['mean']-main_face['itml']['AP']['mean'])
    def rank(method,setting):return next(r['mean_rank1'] for r in body if r['method']==method and r['setting']==setting)
    bg=[100*(rank('itml_plus',s)-rank('itml',s)) for s in ('normal_A','normal_B')]
    lines=['# 替代数据实验报告：按原论文顺序','',
        '本次计划的两个替代数据集实验已完成。旧ORL的11个最终结果文件及110个冻结模型记录保持不变；此前其他实验也保留。原论文的三个数据库仍未取得，因此本报告不属于原论文原数据数值复现。','',
        f'主要观察：人脸ITML+相对ITML的AP差为{fg:+.3f}个百分点；人物normal_A/normal_B的Rank-1差分别为{bg[0]:+.3f}/{bg[1]:+.3f}个百分点。十个重叠人物划分的AP差均值为{robust["AP_difference_mean"]:+.3f}个百分点，正收益{robust["positive_partitions"]}/10。','',
        '## 研究方向与公式','',
        '仍研究“训练阶段用额外深度信息辅助学习，测试只用RGB”的人脸验证与人物再识别。ITML+采用公开Algorithm 1的更新，Partial采用有/无特权训练对混合。没有用后续修正版替代主比较。公式及印刷歧义见[公式说明](FORMULAS.md)；参数和停止规则均有记录。','',
        '## V-B 人脸验证','',
        'FaceGrabber固定40人、560张RGB；20/10/10人物互斥划分。训练特权信息来自真实传感器点云z值，使用RGB眼点对齐和训练深度投影/插值；这些预处理与原文存在差异。主比较五轮固定人物、重抽负对，各方法完整验证网格选参。',
        '|方法|AP (%)|AUC (%)|','|---|---:|---:|']
    for m,r in main_face.items():lines.append(f"|{m}|{100*r['AP']['mean']:.2f} ± {100*r['AP']['std']:.2f}|{100*r['AUC']['mean']:.2f} ± {100*r['AUC']['std']:.2f}|")
    best=max(main_face,key=lambda m:main_face[m]['AP']['mean'])
    lines += ['',f'本次人脸平均AP最高的方法是{best}。五轮标准差描述负对重抽的变化，不是独立人物置信区间。',
        f'额外10个人物划分分别重新训练PCA、完整验证选参，所有20个模型选定后才评分。ITML+的AP差范围为[{robust["AP_difference_min"]:+.3f}, {robust["AP_difference_max"]:+.3f}]个百分点；这些划分共享人物，不做独立样本t检验。','',
        '## V-C 人物再识别','',
        'WeightGait主实验32人、1920张RGB，16人训练、16人验证；训练仅使用320帧展示深度。该数据公开的是有损彩色深度展示视频，转灰度不能恢复原始深度。主实验两个条件都是正常行走的不同片段组，并非原BIWI的Walking/Still。',
        '|方法|normal_A Rank-1 (%)|normal_B Rank-1 (%)|','|---|---:|---:|']
    for m in main_face:
        rows=[next(r for r in body if r['method']==m and r['setting']==s) for s in ('normal_A','normal_B')]
        lines.append('|'+m+'|'+'|'.join(f"{100*r['mean_rank1']:.2f} ± {100*r['std_rank1']:.2f}" for r in rows)+'|')
    best_body=[max((r for r in body if r['setting']==s),key=lambda r:r['mean_rank1']) for s in ('normal_A','normal_B')]
    lines+=['',f"normal_A最高为{best_body[0]['method']}（{100*best_body[0]['mean_rank1']:.2f}%）；normal_B最高为{best_body[1]['method']}（{100*best_body[1]['mean_rank1']:.2f}%）。因此未复现ITML+在所有数据条件下都超过全部基线的结论。"]
    lines += ['','每个人物组20帧，首帧查询，其余19帧图库。两组测试包含全部32人，其中16人参与训练、16人参与验证选参；不能称为全新人物确认实验。每轮32个固定查询，五轮只改变训练负对，差异均为描述性结果。','',
        '另取单腿/双腿负重共1280张RGB，沿用冻结的正常行走码本、PCA和模型，不读取额外条件深度。与主实验没有共享片段或重复图片。以下列出所有跨片段/跨动作方向；七方法完整结果见[完整数值报告](runs/external_rgbd_20261031/final_report/RESULTS.md)。',
        '|跨条件方向|ITML Rank-1 (%)|ITML+ Rank-1 (%)|差 (百分点)|','|---|---:|---:|---:|']
    cross=read(final/'cross_clip_results.json')+read(HERE/'action_final/cross_action_results.json')
    for direction in dict.fromkeys(r['direction'] for r in cross):
        values=[100*mean([r for r in cross if r['direction']==direction and r['method']==m],'rank1') for m in ('itml','itml_plus')]
        lines.append(f'|{direction}|{values[0]:.2f}|{values[1]:.2f}|{values[1]-values[0]:+.2f}|')
    lines += ['','这些仍是同一批人物、相近衣着和同一相机场景，不能替代Still或跨日换装。','',
        '## V-D1 特权信息比例','',
        '按0%、25%、50%、75%、100%逐步增加有特权信息的正/负训练对。中间点固定主比较ITML+参数；原曲线0%端点为独立验证选参的ITML，另补同gamma的0%端点以隔离参数变化。',
        '|比例|原端点规则 AP (%)|AUC (%)|同gamma AP (%)|同gamma AUC (%)|','|---|---:|---:|---:|---:|']
    partial=sorted([r for r in face if r['kind']=='partial'],key=lambda r:r['fraction'])
    matched=[r for r in controls['face'] if r['variant']=='no_PI_matched_gamma']
    matched_ap=[]
    for r in partial:
        ap,auc=r['AP']['mean'],r['AUC']['mean']
        ma,mu=(mean(matched,'AP'),mean(matched,'AUC')) if r['fraction']==0 else (ap,auc)
        matched_ap.append(ma)
        lines.append(f"|{100*r['fraction']:.0f}%|{100*ap:.2f}|{100*auc:.2f}|{100*ma:.2f}|{100*mu:.2f}|")
    lines += ['','图：[原端点比例曲线](runs/external_rgbd_20261031/face_input_v2/final/partial_curve.png)、[同gamma补充曲线](runs/external_rgbd_20261031/final_report/matched_parameter_partial.png)。','',
        '## V-D2 训练对数量','',
        '按20%、40%、60%、80%、100%使用训练对，两种方法各固定自己在完整训练集上验证选定的参数；测试集合不变。',
        '|训练对比例|ITML AP / AUC (%)|ITML+ AP / AUC (%)|AP差 (百分点)|','|---|---:|---:|---:|']
    gains=[]
    for fraction in (.2,.4,.6,.8,1.):
        rows=[next(r for r in face if r['kind']=='sample_count' and r['method']==m and abs(r['fraction']-fraction)<1e-10) for m in ('itml','itml_plus')]
        values=[100*r['AP']['mean'] for r in rows];gains.append(values[1]-values[0])
        cells=[f"{100*r['AP']['mean']:.2f} / {100*r['AUC']['mean']:.2f}" for r in rows]
        lines.append(f'|{100*fraction:.0f}%|{cells[0]}|{cells[1]}|{gains[-1]:+.3f}|')
    lines += ['','图：[训练对数量曲线](runs/external_rgbd_20261031/face_input_v2/final/sample_count_curve.png)。逐点变化按实列出，不将测试波动作为选参依据。','',
        '## V-D3 学得距离与机制','',
        '分别保存固定首200正对、200负对的[人脸距离图](runs/external_rgbd_20261031/face_input_v2/final/distance_analysis.png)及[人物距离图](runs/external_rgbd_20261031/final_report/body_distance_analysis.png)，并保留原始数组。它们使用训练对，只能描述训练距离，不能单独证明测试泛化或减少过拟合。','',
        '每个数据集另有45次固定主参数对照：同gamma普通ITML、3种子打乱、3种子随机、复制RGB特征、延长到100遍；90次均按预定方案保存，无测试成绩择优。各控制组未分别重调参数，因此不能将其看作各控制组最优性能。',
        '|数据/指标|真实特权信息|打乱信息均值|随机信息均值|','|---|---:|---:|---:|']
    for kind,setting,truth in [('face',None,100*main_face['itml_plus']['AP']['mean'])]+[('weight',s,100*rank('itml_plus',s)) for s in ('normal_A','normal_B')]:
        values=[]
        for prefix in ('shuffle_PI','random_PI'):
            chosen=[r for r in controls[kind] if r['variant'].startswith(prefix) and (setting is None or r['setting']==setting)]
            values.append(100*mean(chosen,'AP' if kind=='face' else 'rank1'))
        lines.append(f'|{kind+" "+(setting or "AP")}|{truth:.2f}|{values[0]:.2f}|{values[1]:.2f}|')
    lines+=['','打乱/随机信息也取得明显收益，因此不能把相对ITML的全部提升归因于真实深度语义。真实特权信息的均值更高，但控制组没有各自重调参数，机制证据仍有限。']
    lines += ['','训练/验证/测试差距、全部控制组、停止原因和约束残差见完整数值报告。相邻更新范数停止不代表全部约束已满足；主比较与100遍诊断均不宣称取得全局最优。','',
        '## V-D4 训练时间','',
        '全部主实验、稳定性和对照训练完成后，另做40次串行训练计时，所得度量矩阵与原冻结模型一致。受本机硬件、实现后端、参数及停止状态影响，不能用来复现作者工作站的秒数或固定“两倍”比例。',
        '|数据|ITML平均秒数|ITML+平均秒数|比值|','|---|---:|---:|---:|']
    timing=read(final/'quiet_training_times.json')['records']
    for kind in ('face','body'):
        values=[mean([r for r in timing if r['dataset']==kind and r['method']==m],'fit_seconds') for m in ('itml','itml_plus')]
        lines.append(f'|{kind}|{values[0]:.3f}|{values[1]:.3f}|{values[1]/values[0]:.2f}×|')
    lines += ['','图：[串行训练时间](runs/external_rgbd_20261031/final_report/quiet_training_time.png)，另含LMNN和ITML-S。','',
        '## 能支持到什么程度','',
        '|原论文相关论点|本次证据与边界|','|---|---|',
        f'|特权信息可以带来收益|人脸AP差{fg:+.3f}个百分点；人物两条件差{bg[0]:+.3f}/{bg[1]:+.3f}个百分点。适用范围限于本次数据和协议；是否超过全部其他方法应看完整七方法表。|',
        f'|更多特权训练对通常更好|同gamma曲线AP总体提高，但是否逐点不下降：{"是" if np.all(np.diff(matched_ap)>=-1e-12) else "否"}；75%到100%变化{100*(matched_ap[-1]-matched_ap[-2]):+.3f}个百分点。不能把总体收益写成所有比例都严格单调提升。|',
        f'|小样本时优势更大|本次未复现：20%训练对AP差{gains[0]:+.3f}，100%差{gains[-1]:+.3f}个百分点，优势反而随这条固定参数曲线增加。|',
        '|减少过拟合的机制|有打乱/随机信息、训练测试差距和延长迭代诊断；训练距离更分离或差距更小本身不能证明因果机制。|',
        '|原数据上的精度、Still/跨日泛化和作者时间比例|未复现：仍缺EURECOM、CurtinFaces、BIWI及原作者完整设置和历史特征包。替代数据上的正收益不能填补这些条件。|','',
        '## 核验与保留','',
        '正式网格共尝试2300组参数，2299组成功；人物round_03的ITML-S在gamma=0.0001时过滤后缺少一种标签，按预定规则拒绝，不作回退填充。该失败完整保留，其余有效候选用于验证选参；两数据集共70个主比较选定模型全部可用。',
        f"46项测试通过。{checks['independently_recomputed_evaluations']}个条件评估已用独立评分实现核验；90次机制对照完成，40次串行训练复算模型一致，旧结果和旧模型哈希未变。具体计数和误差见[最终核验](runs/external_rgbd_20261031/final_checks.json)、[独立核验记录](runs/external_rgbd_20261031/final_report/independent_audit.json)及同目录补充audit文件。",
        '以前低维ORL及Washington扩展中的负收益结果仍保留，未用这次数据覆盖，也未合并不同任务的AP来制造整体收益。数据来源、预处理修订和许可记录见[获取记录](EXTERNAL_RGBD_PROGRESS.md)。']
    formatted=[]
    for line in lines:
        if formatted and line and formatted[-1] and line.startswith('|')!=formatted[-1].startswith('|'):formatted.append('')
        formatted.append(line)
    (PROJECT/'EXTERNAL_RGBD_RESULTS.md').write_text('\n'.join(formatted)+'\n')
    print('PAPER-ORDER REPORT WRITTEN',flush=True)

if __name__=='__main__':main()
