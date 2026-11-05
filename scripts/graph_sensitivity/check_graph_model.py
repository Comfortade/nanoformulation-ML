"""Focused implementation checks; these do not establish scientific validity."""
import copy,json
from pathlib import Path
import numpy as np
import torch
import run_graph_sensitivity as s

def check(path):
    variants=json.loads(Path(path).read_text())['variants'];v=variants['reported_mapping']
    torch.manual_seed(11);m=s.ExtractModel(v,False,1)
    before=m.context().detach().clone()
    # A relabeling of atoms must not change the pooled molecular representation.
    perm=torch.randperm(len(m.atoms));inverse=torch.argsort(perm)
    with torch.no_grad():
        m.atoms=m.atoms[perm];m.membership=m.membership[perm];m.edges=inverse[m.edges]
    assert torch.allclose(before,m.context(),atol=1e-6), 'Atom permutation changed representation'
    # Defined stereo is actually consumed, not just stored in the export.
    a=copy.deepcopy(v);b=copy.deepcopy(v)
    for entry in b['entries']:
        for bond in entry['graph']['directed_bonds']:
            if bond['stereo']=='STEREOZ':bond['stereo']='STEREOE'
    torch.manual_seed(11);model_a=s.ExtractModel(a,False,1)
    torch.manual_seed(11);model_b=s.ExtractModel(b,False,1)
    assert not torch.allclose(model_a.context(),model_b.context(),atol=1e-8,rtol=1e-7),'Stereo unused'
    # All variant representations can execute forward/backward passes.
    for variant in [None]+list(variants.values()):
        for kinetic in [False,True]:
            model=s.ExtractModel(variant,kinetic,1)
            x=torch.zeros((4,3));t=torch.tensor([0.,.25,1.,1.25])
            y=model(x,t);assert torch.isfinite(y).all();y.sum().backward()
            if variant is not None:assert model.messages[0].weight.grad is not None
            if kinetic:assert abs(y[0].item())<1e-8 and bool((y[1:]>=y[:-1]).all()) and bool((y<=100).all())
    print('PASS: atom permutation invariance, bond-stereo sensitivity, all variant gradients, constrained shape.')

if __name__=='__main__':
    import sys
    check(sys.argv[1] if len(sys.argv)>1 else 'data/chemical_audit/graph_input_variants.json')
