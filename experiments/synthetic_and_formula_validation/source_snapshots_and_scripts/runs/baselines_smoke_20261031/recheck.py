"""Recheck saved final baseline models; run as python -m runs.baselines_smoke_20261031.recheck."""
import json
from pathlib import Path
import numpy as np
from itml_repro.baselines import decision_scores
from itml_repro.evaluation import verification_scores
from itml_repro.run import save_json,sha

root=Path(__file__).resolve().parent
out=root/'comparison_final'
rows=json.loads((out/'results.json').read_text())
checks=[]
with np.load(root/'input/features_pca.npz') as data:
    val=data['X_val']
    assert 'X_test' not in data.files
for r in rows:
    with np.load(root/'input'/r['round']/'pairs.npz') as data:
        pairs=data['val']
        assert 'test' not in data.files
    with np.load(out/r['round']/(r['method']+'_model.npz')) as model:
        result=verification_scores(decision_scores(val,pairs,model),pairs[:,2])
    assert all(abs(result[m]-r['selected']['validation'][m])<1e-12 for m in result)
    checks.append(dict(round=r['round'],method=r['method'],saved_model_scores_match=True))
residuals={}
searches=list(out.glob('round_*/*_search.json'))
for path in searches:
    for candidate in json.loads(path.read_text()):
        for name,value in candidate.get('diagnostics',{}).items():
            if name.startswith('stationarity') or name in ('primal_violation','complementarity','relative_objective_recheck'):
                residuals[name]=max(residuals.get(name,0.),value)
save_json(root/'acceptance.json',dict(comparison_directory=str(out),saved_models_checked=len(checks),checks=checks,
    candidate_count=sum(len(json.loads(p.read_text())) for p in searches),
    failed_candidates=sum(r['failed_candidates'] for r in rows),test_inputs_absent=True,
    maximum_convex_diagnostics=residuals,
    source_sha256={name:sha(root.parents[1]/'itml_repro'/name) for name in ('baselines.py','baseline_compare.py','evaluation.py')}))
print('Saved model recheck passed:',len(checks),'models')
