"""Training-only empirical power-law vs exponential comparison on supplied Excel values.
Run from the repository root. Does not modify the workbook or train neural networks.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, importlib.metadata, json, platform, shutil
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy.optimize import least_squares, minimize_scalar
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut

BLOCKS=[(5.5,'Test',3),(5.5,'Control',6),(6.5,'Test',9),(6.5,'Control',12),(7.4,'Test',15),(7.4,'Control',18)]
N_LO,N_HI=1e-6,10.0  # numerical search bounds, not mechanism-specific exponent limits

def read_excel(path):
    f=load_workbook(path,data_only=False);v=load_workbook(path,data_only=True)
    if 'Releases' not in f:raise ValueError('Updated workbook with Releases sheet required')
    records=[]
    for ph,form,c in BLOCKS:
        for row in range(9,15):
            t=float(v['Releases'].cell(row,2).value)
            mass=float(v['Releases'].cell(row,c+1).value)
            formula=f['Releases'].cell(row,c+2).value
            if not isinstance(formula,str) or not formula.startswith('='):raise ValueError('Missing percentage formula')
            denominator=float(formula.rsplit('/',1)[1].strip())
            y=float(v['Releases'].cell(row,c+2).value)
            if denominator<=0 or not np.isclose(y,mass*100/denominator,rtol=1e-10,atol=1e-12):raise ValueError('Formula/cache mismatch')
            records.append(dict(row_id=len(records),Time_h=t,pH=ph,Formulation=form,value=y,denominator=denominator,source_cell=v['Releases'].cell(row,c+2).coordinate,source_formula=formula))
    frame=pd.DataFrame(records)
    assert len(frame)==36 and not frame.duplicated(['Time_h','pH','Formulation']).any()
    assert np.isfinite(frame[['Time_h','value']]).all().all() and (frame.value>=0).all()
    assert set(frame.Time_h)=={0,6,12,24,48,96}
    frame['profile_group']=frame.groupby(['pH','Formulation']).ngroup()
    return frame

def fit_equation(frame,kind):
    mean=frame.groupby('Time_h',as_index=False).value.mean()
    t=mean.Time_h.to_numpy()/96.;y=mean.value.to_numpy();scale=max(float(np.max(abs(y))),1e-8)
    if np.unique(t[t>0]).size<2:
        coefficient=max(float(t@y/max(t@t,1e-12)),0.)
        return (lambda q:coefficient*np.asarray(q)/96.),dict(route='linear_fallback_one_positive_time',coefficient=coefficient,parameter_1=np.nan,parameter_2=np.nan,boundary=False,Jacobian_condition=np.nan)
    if kind=='saturating_exponential':
        def residual(theta):return (theta[0]*(-np.expm1(-theta[1]*t))-y)/scale
        fits=[least_squares(residual,[scale,k],bounds=([0,1e-6],[100,1000]),max_nfev=2000) for k in [1.,10.,100.]]
        good=[q for q in fits if q.success and np.isfinite(q.cost)]
        if not good:raise RuntimeError('Exponential fit did not converge')
        best=min(good,key=lambda q:q.cost);a,k=best.x
        condition=float(np.linalg.cond(best.jac))
        boundary=bool(a<1e-7 or a>100-1e-5 or k<1.01e-6 or k>999.99)
        return (lambda q:a*(-np.expm1(-k*np.asarray(q)/96.))),dict(route='two_parameter_fit',parameter_1=float(a),parameter_2=float(k),boundary=boundary,Jacobian_condition=condition)
    if kind!='power_law':raise ValueError(kind)
    # For each n, the least-squares coefficient has an analytic training-only solution.
    def solve(n):
        u=np.power(t,n);a=max(float(u@y/max(u@u,1e-300)),0.)
        return float(np.sum(((a*u-y)/scale)**2)),a
    grid=np.geomspace(N_LO,N_HI,81);cost=np.array([solve(n)[0] for n in grid]);candidates=[(float(cost[0]),float(grid[0])),(float(cost[-1]),float(grid[-1]))]
    for i in range(1,len(grid)-1):
        if cost[i]<=cost[i-1] and cost[i]<=cost[i+1]:
            fit=minimize_scalar(lambda n:solve(n)[0],bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':1e-12})
            if fit.success:candidates.append((float(fit.fun),float(fit.x)))
    _,n=min(candidates);_,a=solve(n)
    u=t**n;second=np.zeros_like(t);mask=t>0;second[mask]=a*u[mask]*np.log(t[mask])
    jac=np.column_stack([u,second])/scale
    return (lambda q:a*(np.asarray(q)/96.)**n),dict(route='two_parameter_fit',parameter_1=a,parameter_2=n,boundary=bool(n<=N_LO*1.01 or n>=N_HI*.999),Jacobian_condition=float(np.linalg.cond(jac)))

def metrics(g):
    return dict(n=len(g),R2=float(r2_score(g.observed,g.predicted)) if len(g)>1 and np.var(g.observed)>1e-24 else np.nan,RMSE=float(np.sqrt(mean_squared_error(g.observed,g.predicted))),MAE=float(mean_absolute_error(g.observed,g.predicted)))

def run(path,out):
    frame=read_excel(path);out.mkdir(parents=True,exist_ok=False);frame.to_csv(out/'source_release_values.csv',index=False)
    splits=[('held_out_profiles',i,tr,te) for i,(tr,te) in enumerate(LeaveOneGroupOut().split(frame,groups=frame.profile_group))]
    splits += [('forward_time',i,np.flatnonzero(frame.Time_h<t),np.flatnonzero(frame.Time_h==t)) for i,t in enumerate([12.,24.,48.,96.])]
    predictions=[];parameters=[];checks=[];assignments=[]
    for protocol,fold,tr,te in splits:
        train=frame.iloc[tr];test=frame.iloc[te]
        assert set(train.row_id).isdisjoint(test.row_id)
        if protocol=='held_out_profiles':assert set(train.profile_group).isdisjoint(test.profile_group)
        else:assert train.Time_h.max()<test.Time_h.min()
        checks.append(dict(protocol=protocol,fold=fold,training_rows=len(train),test_rows=len(test),row_overlap=0,whole_profile_separation=protocol=='held_out_profiles',strictly_earlier_training=protocol=='forward_time'))
        for role,part in [('train',train),('test',test)]:
            for idx in part.row_id:assignments.append(dict(protocol=protocol,fold=fold,role=role,row_id=int(idx)))
        # Query covariates exclude targets. Outcomes are joined only after prediction.
        query=test[['row_id','Time_h','pH','Formulation']]
        for kind in ['power_law','saturating_exponential']:
            cache={}
            for record in query.to_dict('records'):
                key=(record['pH'],record['Formulation'])
                if key not in cache:
                    subset=train[(train.pH==key[0])&(train.Formulation==key[1])];source='same_profile'
                    if subset.empty:subset=train[train.pH==key[0]];source='same_pH_other_formulation'
                    if subset.empty:raise ValueError('Unexpected absent training pH')
                    predict,details=fit_equation(subset,kind);cache[key]=predict
                    parameters.append(dict(protocol=protocol,fold=fold,model=kind,pH=key[0],Formulation=key[1],training_source=source,training_row_ids=json.dumps(subset.row_id.tolist()),**details))
                prediction=float(cache[key]([record['Time_h']])[0]);assert np.isfinite(prediction)
                observed=float(test.loc[test.row_id==record['row_id'],'value'].item())
                predictions.append(dict(protocol=protocol,fold=fold,model=kind,**record,observed=observed,predicted=prediction))
        print(protocol,'fold',fold+1,'completed',flush=True)
    p=pd.DataFrame(predictions);assert len(p)==120 and not p.duplicated(['protocol','fold','model','row_id']).any()
    p.to_csv(out/'held_out_predictions.csv',index=False);pd.DataFrame(parameters).to_csv(out/'training_fit_parameters.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'split_checks.csv',index=False);pd.DataFrame(assignments).to_csv(out/'split_assignments.csv',index=False)
    summaries=[]
    for fields,name in [(['protocol','model'],'comparison.csv'),(['protocol','model','pH','Formulation'],'metrics_by_profile.csv'),(['protocol','model','fold'],'metrics_by_fold.csv')]:
        rows=[dict(zip(fields,key))|metrics(g) for key,g in p.groupby(fields)];pd.DataFrame(rows).to_csv(out/name,index=False)
    stability=[]
    for (ph,form),group in frame.groupby(['pH','Formulation']):
        for kind in ['power_law','saturating_exponential']:
            for excluded in [None]+sorted(group.loc[group.Time_h>0,'Time_h'].unique().tolist()):
                subset=group if excluded is None else group[group.Time_h!=excluded]
                _,details=fit_equation(subset,kind)
                stability.append(dict(pH=ph,Formulation=form,model=kind,excluded_time_h=excluded,scope='descriptive_parameter_stability_not_validation',**details))
    pd.DataFrame(stability).to_csv(out/'descriptive_parameter_stability.csv',index=False)
    cfg=dict(workbook_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),script_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),target='Supplied cumulative-release percentage unchanged',power_law='y=a*(t/96h)^n; a>=0; n numerical search bounds [1e-6,10]; a is value at 96h, not an asymptote',exponential='y=A*(1-exp(-k*t/96h)); A in [0,100]; k in [1e-6,1000]; same bounds/multistart procedure as prior comparison',fallback='Nonnegative line through origin for both models if fewer than two positive training times',fit='Least squares on original response scale using training-only numerical residual scaling; zero-time values retained; no M_infinity inferred; no 60%-fraction cutoff imposed because fractional normalization is unverified',parameter_names={'power_law':['a_at_96h','n'],'saturating_exponential':['A_asymptote','k_normalized_time']},independent_experimental_validation=False,neural_models_trained=False,platform=platform.platform(),versions={x:importlib.metadata.version(x) for x in ['numpy','scipy','pandas','scikit-learn','openpyxl']})
    (out/'run_config.json').write_text(json.dumps(cfg,indent=2));(out/'completion.json').write_text(json.dumps(dict(completed=True,predictions=len(p),splits=len(splits)),indent=2))
    print(pd.read_csv(out/'comparison.csv').to_string(index=False));return out

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--excel',type=Path,default=Path('data/SMART NANO DATA OLA 2.xlsx'));parser.add_argument('--output',type=Path)
    args=parser.parse_args();output=args.output or Path('exports/powerlaw_benchmark')/datetime.now(timezone.utc).strftime('run_%Y%m%d_%H%M%S_%f')
    result=run(args.excel,output);print('COMPLETED:',result);print('ZIP:',shutil.make_archive(str(result),'zip',root_dir=result))
