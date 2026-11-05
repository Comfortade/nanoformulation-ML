"""Matched exploratory tests of fixed extract representations; CPU only.
This is not the manuscript's original Korsmeyer–Peppas PIGNN.
"""
import argparse, hashlib, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import revision_core as core

ATOM_CIP=['R','S','unspecified_or_not_applicable']
HYBRID=['SP','SP2','SP3','UNSPECIFIED']
BOND=['SINGLE','DOUBLE','TRIPLE','AROMATIC']
STEREO=['STEREONONE','STEREOANY','STEREOZ','STEREOE','STEREOCIS','STEREOTRANS']
def onehot(value,choices):
    return [float(value==v) for v in choices]+[float(value not in choices)]

def tensor_graphs(variant):
    atoms=[];edges=[];attributes=[];membership=[];weights=[];offset=0
    for g,e in enumerate(variant['entries']):
        for a in e['graph']['atoms']:
            atoms.append([a['atomic_number']/20,a['formal_charge'],a['hydrogens']/4,float(a['aromatic'])]+onehot(a['hybridization'],HYBRID)+onehot(a['CIP'],ATOM_CIP))
            membership.append(g)
        for b in e['graph']['directed_bonds']:
            edges.append([offset+b['source'],offset+b['target']])
            attributes.append(onehot(b['bond_type'],BOND)+onehot(b['stereo'],STEREO)+[float(b['conjugated']),float(b['in_ring'])])
        offset+=len(e['graph']['atoms']);weights.append(e['pooling_weight'])
    return [torch.tensor(atoms,dtype=torch.float32),torch.tensor(edges,dtype=torch.long).T,
            torch.tensor(attributes,dtype=torch.float32),torch.tensor(membership,dtype=torch.long),
            torch.tensor(weights,dtype=torch.float32)]

class ExtractModel(nn.Module):
    def __init__(self,variant,kinetic,scale):
        super().__init__();self.kinetic=kinetic;self.has_graph=variant is not None
        self.register_buffer('capacity',torch.tensor(100/scale))
        if self.has_graph:
            tensors=tensor_graphs(variant)
            for name,t in zip(['atoms','edges','bond_features','membership','weights'],tensors):self.register_buffer(name,t)
            self.embed=nn.Linear(tensors[0].shape[1],8)
            self.messages=nn.ModuleList([nn.Linear(8+tensors[2].shape[1],8) for _ in range(2)])
            self.updates=nn.ModuleList([nn.Linear(16,8) for _ in range(2)])
        # Identical prediction head and initialization across variants and zero-context baseline.
        self.head=nn.Sequential(nn.Linear((2 if kinetic else 3)+8,16),nn.Tanh(),nn.Linear(16,8),nn.Tanh(),nn.Linear(8,2 if kinetic else 1))
        if kinetic:
            with torch.no_grad():
                f=np.clip(.5*scale/100,1e-8,1-1e-8)
                self.head[-1].bias[0]=float(np.log(f/(1-f)));self.head[-1].bias[1]=20
    def context(self):
        if not self.has_graph:return torch.zeros(8)
        h=torch.tanh(self.embed(self.atoms));src,dst=self.edges
        counts=torch.bincount(dst,minlength=len(h)).clamp(min=1).float().unsqueeze(1)
        for message,update in zip(self.messages,self.updates):
            m=torch.tanh(message(torch.cat([h[src],self.bond_features],dim=1)))
            agg=torch.zeros_like(h).index_add(0,dst,m)/counts
            h=torch.tanh(update(torch.cat([h,agg],dim=1)))
        pooled=torch.zeros((len(self.weights),8),dtype=h.dtype).index_add(0,self.membership,h)
        n=torch.bincount(self.membership,minlength=len(self.weights)).float().unsqueeze(1)
        return ((pooled/n)*self.weights[:,None]).sum(0)
    def forward(self,x,t):
        if self.kinetic:x=x[:,1:]
        z=self.context().unsqueeze(0).expand(len(x),-1)
        out=self.head(torch.cat([x,z],dim=1))
        if not self.kinetic:return out.flatten()
        a=self.capacity*torch.sigmoid(out[:,0]);k=torch.nn.functional.softplus(out[:,1])+1e-6
        return a*(-torch.expm1(-k*t))

def fit(train,query,variant,kinetic,seed,epochs):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    scaler=StandardScaler().fit(train[core.FEATURES])
    scale=max(float(train.value.abs().max()),1e-8)
    model=ExtractModel(variant,kinetic,scale)
    # Reset head identically irrespective of graph parameter initialization order.
    torch.manual_seed(seed+10000)
    for layer in model.head:
        if isinstance(layer,nn.Linear):layer.reset_parameters()
    if kinetic:
        with torch.no_grad():
            f=np.clip(.5*scale/100,1e-8,1-1e-8)
            model.head[-1].bias[0]=float(np.log(f/(1-f)));model.head[-1].bias[1]=20
    x=torch.tensor(scaler.transform(train[core.FEATURES]),dtype=torch.float32)
    y=torch.tensor(train.value.to_numpy()/scale,dtype=torch.float32)
    t=torch.tensor(train.Time_h.to_numpy()/96,dtype=torch.float32)
    prior=None
    if kinetic:
        prior=torch.tensor(core.curve_predictions(train,train)[0]/scale,dtype=torch.float32)
    optimizer=torch.optim.Adam(model.parameters(),lr=.003,weight_decay=1e-5)
    for _ in range(epochs):
        optimizer.zero_grad();pred=model(x,t);loss=((pred-y)**2).mean()
        if kinetic:loss=loss+.1*((pred-prior)**2).mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
    model.eval()
    with torch.no_grad():
        pred=model(torch.tensor(scaler.transform(query[core.FEATURES]),dtype=torch.float32),torch.tensor(query.Time_h.to_numpy()/96,dtype=torch.float32)).numpy()*scale
    assert np.isfinite(pred).all()
    return pred

def run(args):
    root=Path(args.root).resolve();book=root/'data'/args.excel
    variants_path=root/'data/chemical_audit/graph_input_variants.json'
    payload=json.loads(variants_path.read_text());variants=payload['variants']
    mapping=root/'data/chemical_audit/reviewed_structure_mapping.csv'
    assert hashlib.sha256(mapping.read_bytes()).hexdigest()==payload['source_SHA256'],'Mapping changed: rerun verification/export.'
    conc,summary,release,loading,meancheck,releasecheck=core.read_workbook(book)
    datasets={'concentration_as_supplied':conc,'cumulative_percent_as_supplied':release}
    if args.target!='both':datasets={args.target:datasets[args.target]}
    out=root/'exports/graph_sensitivity'/core.datetime.now(core.timezone.utc).strftime('run_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    config=dict(epochs=args.epochs,seeds=args.seeds,learning_rate=.003,weight_decay=1e-5,kinetic_weight=.1,graph_layers=2,graph_width=8,targets=list(datasets),variant_names=list(variants),status='exploratory representation sensitivity; not independent experimental validation',chemistry_assignment='Same fixed extract context across all observations; not an assertion that free-citral control contains extract.',kinetic_model='Saturating exponential plus training-only empirical prior; not Korsmeyer-Peppas.',workbook_SHA256=hashlib.sha256(book.read_bytes()).hexdigest(),graph_inputs_SHA256=hashlib.sha256(variants_path.read_bytes()).hexdigest(),script_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    core.write_json(out/'run_config.json',config)
    core.audit_splits(datasets,out)
    core.write_json(out/'environment.json',{p:core.importlib.metadata.version(p) for p in ['torch','pandas','scikit-learn','numpy','rdkit']})
    assignments=[];predictions=[]
    for target,frame in datasets.items():
        for protocol,fold,tr,te in core.splits_for(frame,target):
            train=frame.iloc[tr].copy();test=frame.iloc[te].copy()
            query=test[core.FEATURES+['Formulation']].copy()
            for role,ids in [('train',tr),('test',te)]:
                for idx in ids:assignments.append(dict(target=target,protocol=protocol,fold=fold,role=role,row_id=int(frame.iloc[idx].row_id)))
            settings=[False,True] if target.startswith('cumulative') else [False]
            for name in ['linear','random_forest']:
                for seed in (args.seeds if name=='random_forest' else [args.seeds[0]]):
                    if name=='linear':
                        estimator=core.make_pipeline(StandardScaler(),core.LinearRegression())
                    else:
                        estimator=core.RandomForestRegressor(n_estimators=200,min_samples_leaf=2,random_state=seed,n_jobs=1)
                    estimator.fit(train[core.FEATURES],train.value)
                    values=estimator.predict(query[core.FEATURES])
                    for r,prediction in zip(test.to_dict('records'),values):predictions.append(dict(target=target,protocol=protocol,fold=fold,model=name,variant='not_applicable',seed=seed,row_id=r['row_id'],pH=r['pH'],Formulation=r['Formulation'],Time_h=r['Time_h'],observed=r['value'],predicted=float(prediction)))
            # Classical comparison reuses the same exact held-out observations.
            if target.startswith('cumulative'):
                for name,kind in [('saturating_exponential','saturating'),('higuchi','higuchi')]:
                    pred=core.curve_predictions(train,query,kind)[0]
                    for r,p in zip(test.to_dict('records'),pred):predictions.append(dict(target=target,protocol=protocol,fold=fold,model=name,variant='not_applicable',seed=-1,row_id=r['row_id'],pH=r['pH'],Formulation=r['Formulation'],Time_h=r['Time_h'],observed=r['value'],predicted=float(p)))
            for kinetic in settings:
                for label,variant in [('zero_context_baseline',None)]+list(variants.items()):
                    for seed in args.seeds:
                        pred=fit(train,query,variant,kinetic,seed,args.epochs)
                        for r,p in zip(test.to_dict('records'),pred):predictions.append(dict(target=target,protocol=protocol,fold=fold,model='kinetic' if kinetic else 'unconstrained',variant=label,seed=seed,row_id=r['row_id'],pH=r['pH'],Formulation=r['Formulation'],Time_h=r['Time_h'],observed=r['value'],predicted=float(p)))
            pd.DataFrame(predictions).to_csv(out/'held_out_predictions.csv',index=False)
            print(target,protocol,'fold',fold+1,'completed',flush=True)
    pred=pd.DataFrame(predictions)
    assert not pred.duplicated(['target','protocol','fold','model','variant','seed','row_id']).any()
    pd.DataFrame(assignments).to_csv(out/'split_assignments.csv',index=False)
    by=['target','protocol','model','variant','seed']
    metrics=[]
    for key,part in pred.groupby(by):
        metrics.append(dict(zip(by,key))|core.metrics_for(part))
    scores=pd.DataFrame(metrics);scores.to_csv(out/'metrics_by_seed.csv',index=False)
    scores.groupby(by[:-1],as_index=False).agg(R2_mean=('R2','mean'),R2_seed_SD=('R2','std'),RMSE_mean=('RMSE','mean'),MAE_mean=('MAE','mean')).to_csv(out/'comparison.csv',index=False)
    subgroup=[]
    for key,part in pred.groupby(by+['pH','Formulation']):subgroup.append(dict(zip(by+['pH','Formulation'],key))|core.metrics_for(part))
    pd.DataFrame(subgroup).to_csv(out/'metrics_by_profile.csv',index=False)
    # Paired changes relative to the all-entry reviewed mapping, within identical folds/seeds.
    reference=pred[pred.variant=='reported_mapping'].drop(columns='variant').rename(columns={'predicted':'reference_prediction'})
    join=['target','protocol','fold','model','seed','row_id']
    paired=pred[pred.variant!='not_applicable'].merge(reference[join+['reference_prediction']],on=join,validate='many_to_one')
    paired['prediction_change']=paired.predicted-paired.reference_prediction
    paired.to_csv(out/'paired_prediction_changes.csv',index=False)
    core.write_json(out/'completion.json',dict(completed=True,prediction_count=len(pred),new_laboratory_experiments=False,interpretation='Do not select chemical identities by prediction score. Seed SD is not a confidence interval. Fixed chemistry cannot establish chemical generalization.'))
    import shutil
    archive=shutil.make_archive(str(out),'zip',root_dir=out)
    print('COMPLETED:',out,'\nResults archive:',archive,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--excel',default='SMART NANO DATA OLA 2.xlsx');p.add_argument('--epochs',type=int,default=500);p.add_argument('--seeds',type=int,nargs='+',default=[11,42,97]);p.add_argument('--target',choices=['both','concentration_as_supplied','cumulative_percent_as_supplied'],default='both')
    args=p.parse_args();assert args.epochs>0;run(args)
