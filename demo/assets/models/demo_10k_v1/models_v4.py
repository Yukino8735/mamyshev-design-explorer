"""Prediction models: use only four inputs and observed training labels.
No access to generator coefficients, intermediate physics fields or lookup table.
"""
from __future__ import annotations
import numpy as np
from sklearn.pipeline import make_pipeline,make_union
from sklearn.preprocessing import PolynomialFeatures,StandardScaler,SplineTransformer
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import HistGradientBoostingRegressor
FEATURES=('pump_power_W','filter_offset_nm','filter2_bw_nm','gain_length_m')
TARGETS=('compressed_energy_nJ','compressed_duration_fs','compressed_peak_power_MW')
class LogSurrogate:
    def __init__(self,family,domain,parameters,random_state=11):
        self.family=family;self.domain=domain;self.parameters=parameters;self.random_state=random_state;self.models=[]
    def _x(self,x):
        x=np.asarray(x,float)
        if x.ndim!=2 or x.shape[1]!=4 or not np.isfinite(x).all():raise ValueError('Expected finite n-by-4 features')
        low=np.array([self.domain[k][0] for k in FEATURES]);high=np.array([self.domain[k][1] for k in FEATURES])
        if (x<low-1e-10).any() or (x>high+1e-10).any():raise ValueError('Prediction outside released domain; do not clip silently.')
        if np.any(np.abs(x[:,3]*10-np.round(x[:,3]*10))>1e-8):raise ValueError('Unsupported gain length: not on 0.1m grid')
        return (x-low)/(high-low)
    def fit(self,x,y):
        x=self._x(x);y=np.asarray(y,float)
        if y.shape!=(len(x),3) or not np.isfinite(y).all() or (y<=0).any():raise ValueError('Expected positive finite n-by-3 training labels')
        self.models=[];p=self.parameters
        for j in range(3):
            if self.family=='ridge2':m=make_pipeline(PolynomialFeatures(p['degree'],include_bias=False),StandardScaler(),Ridge(alpha=p['alpha']))
            elif self.family=='spline_interactions':m=make_pipeline(make_union(SplineTransformer(n_knots=p['n_knots'],degree=p['degree'],knots=p['knots'],include_bias=False),PolynomialFeatures(p['interaction_polynomial_degree'],include_bias=False)),StandardScaler(),Ridge(alpha=p['alpha']))
            elif self.family=='rbf_svr':m=SVR(C=p['C'],epsilon=p['epsilon'],gamma=p['gamma'],tol=p['tol'],cache_size=p['cache_size_MB'])
            elif self.family=='hist_gb':m=HistGradientBoostingRegressor(max_iter=p['max_iter'],max_leaf_nodes=p['max_leaf_nodes'],learning_rate=p['learning_rate'],min_samples_leaf=p['min_samples_leaf'],l2_regularization=p['l2_regularization'],early_stopping=p['early_stopping'],random_state=self.random_state)
            else:raise ValueError('Unknown model family')
            m.fit(x,np.log(y[:,j]));self.models.append(m)
        return self
    def predict(self,x):
        if len(self.models)!=3:raise RuntimeError('Model not fitted')
        x=self._x(x);p=np.exp(np.column_stack([m.predict(x) for m in self.models]))
        if not np.isfinite(p).all():raise ArithmeticError('Prediction overflow')
        return p
