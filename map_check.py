import numpy as np,time
from probe import *
from scipy.optimize import least_squares
def mapfit(tag,te,mb,mp,ne=60):
    pred=[];conv=[];t=time.time();nf=te['t0'].shape[1]
    sd=np.r_[5,5,4]/12**.5
    for e in range(ne):
        def f(v):
            h,b,bl,pa=v[:3],v[3:7],v[7:7+nf],v[7+nf:]
            y,_=solve((te['t0'][e],te['p0'][e],te['n'][e],bl[:,None]*.01,pa[:,None]*1e4),h/100)
            r=((te['y'][:,e]/y-1)/SIG[:,0]-b[:,None,None]).ravel()
            return np.r_[r,(h+[2.5,2.5,2])/sd,b,(bl-mb)/(6/12**.5),(pa-mp)/((8 if mp==4 else 6)/12**.5)]
        s=least_squares(f,np.r_[-2.5,-2.5,-2,np.zeros(4),np.full(nf,mb),np.full(nf,mp)],max_nfev=200)
        pred.append(s.x[:3]/100);conv.append(s.status>0)
    sc,a=skill(np.array(pred),dict(h=te['h'][:ne])); print(f"{tag:45s} {sc:.2f} {a.round(2)} converged {np.mean(conv):.2f} {(time.time()-t)/ne:.1f}s/eng",flush=True)
te=fleet(200,20,2);sh=fleet(200,20,3,shift=True)
tr=fleet(1500,20,1)
for d in (te,): ladder("same",tr,d)
s60={k:(v[:,:60] if k=='y' else v[:60]) if isinstance(v,np.ndarray) and v.ndim else v for k,v in te.items()}
ladder("same, first 60 only",tr,s60)
mapfit("exact model, correct priors, same",te,3.,4.)
mapfit("exact model, correct priors, hot/high",sh,7.,9.)
mapfit("exact model, priors of 'same' on hot/high",sh,3.,4.)
