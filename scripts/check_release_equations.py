"""Focused fitting checks. Run from the project root."""
import numpy as np
import pandas as pd
from compare_release_equations import fit_equation

t=np.array([0.,6.,12.,24.,48.,96.])
for kind,expected in [('power_law',2*(t/96)**.7),('saturating_exponential',2*(-np.expm1(-3*t/96)))]:
 frame=pd.DataFrame({'Time_h':t,'value':expected})
 predict,meta=fit_equation(frame,kind)
 assert np.allclose(predict(t),expected,rtol=1e-5,atol=1e-8),(kind,meta)
 assert np.allclose([meta['parameter_1'],meta['parameter_2']],[2,.7] if kind=='power_law' else [2,3],rtol=1e-4)
for kind in ['power_law','saturating_exponential']:
 f=pd.DataFrame({'Time_h':[0.,6.],'value':[0.,.5]})
 predict,meta=fit_equation(f,kind)
 assert meta['route']=='linear_fallback_one_positive_time' and np.isclose(predict([12.])[0],1.)
print('PASS: known-parameter recovery for both equations and shared one-positive-time fallback.')
