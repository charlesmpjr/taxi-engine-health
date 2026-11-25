"""Probe, not a task: single-spool turbojet, ground static, speed-governed. Health known by construction."""
import numpy as np, time, warnings; warnings.filterwarnings("ignore")
CP,CPG,G,GG,LHV=1005.,1148.,1.4,1.333,43e6
NDES=1.0; PRD=10.; WD=20.; ETC=.84; ETT=.88; TT4=1300.
def phi(x,g):                       # nozzle flow fn, x=p0/pt, choked limited
    xc=(2/(g+1))**(g/(g-1)); x=np.clip(x,xc,.999999)
    return np.sqrt(np.maximum(2*g/(g-1)/287.*(x**(2/g)-x**((g+1)/g)),1e-14))
def comp(nc,pr,h):
    prw=1+(PRD-1)*nc**2.6                      # working-line reference pressure ratio
    z=(pr-1)/(prw-1)
    wc=WD*nc**1.5*(1+h[1])*(1+.35*(1-z**2)*.6)
    eta=(ETC-.30*(1-nc)**2-.25*(z-1)**2)*(1+h[0])
    return wc,np.maximum(eta,.3)
def res(u,c,h,K):
    pr,t4,p5r=np.exp(u)             # p5r=p5/p0
    t0,p0,n,bl,pacc=c; th=t0/288.15; nc=n/np.sqrt(th)
    wc,eta=comp(nc,pr,h); w=wc*(p0/101325)/np.sqrt(th)
    p3=pr*p0; t3=t0*(1+(pr**((G-1)/G)-1)/eta)
    w4=w*(1-bl); p4=.95*p3; p5=p5r*p0; pt=np.clip(p5/p4,1e-3,.99999)
    t5=t4*(1-ETT*(1+h[2])*(1-pt**((GG-1)/GG)))
    r1=np.log(w4*np.sqrt(t4)/p4)-np.log(K[0]*np.sqrt(1-pt**2))
    r2=np.log(w4*np.sqrt(np.maximum(t5,50))/p5)-np.log(K[1]*phi(1/np.maximum(p5r,1.0000001),GG))
    r3=(w4*CPG*(t4-t5)*.99-w*CP*(t3-t0)-pacc)/(w*CP*t0)
    y=np.array([p3,t3,t5,w4*CPG*(t4-t3)/(LHV*.99)])
    return np.array([r1,r2,r3]),y
# design-point constants
def design():
    w=WD;p3=PRD*101325;p4=.95*p3;t3=288.15*(1+(PRD**.2857-1)/ETC)
    dt=w*CP*(t3-288.15)/(w*CPG*.99);t5=TT4-dt;pt=(1-dt/TT4/ETT)**(GG/(GG-1))
    p5=p4*pt;return (w*np.sqrt(TT4)/p4/np.sqrt(1-pt**2), w*np.sqrt(t5)/p5/phi(101325/p5,GG)),p5
K,p5d=design()
def solve(c,h):
    c=[np.asarray(x,float) for x in np.broadcast_arrays(*c)]; h=[np.broadcast_to(np.asarray(x,float),c[0].shape) for x in h]
    nc=c[2]/np.sqrt(c[0]/288.15); s=1+(PRD-1)*nc**2.6
    u=np.log(np.array([s,700+(TT4-700)*nc**2,1+(p5d/101325-1)*nc**2.6]))
    for _ in range(60):
        r,y=res(u,c,h,K); J=np.empty((3,3)+c[0].shape)
        for k in range(3):
            v=u.copy();v[k]+=1e-6;J[:,k]=(res(v,c,h,K)[0]-r)/1e-6
        d=np.linalg.solve(np.moveaxis(J,(0,1),(-2,-1))+1e-12*np.eye(3),np.moveaxis(r,0,-1)[...,None])[...,0]
        d=np.moveaxis(np.clip(d,-.15,.15),-1,0); u=u-d; u[2]=np.maximum(u[2],1e-5)
        if np.abs(d).max()<1e-10:break
    r,y=res(u,c,h,K); return y,np.abs(r).max(0)
NSET=np.array([.55,.62,.70]); NT=.97
SIG=np.array([.003,.003,.004,.008])[:,None,None,None]
def fleet(ne,nf,seed,shift=False,confound=True,takeoff=False):
    r=np.random.default_rng(seed); ns=np.append(NSET,NT) if takeoff else NSET
    h=np.array([-r.uniform(0,.05,ne),-r.uniform(0,.05,ne),-r.uniform(0,.04,ne)])
    t0=(r.uniform(303,316,(ne,nf,1)) if shift else r.uniform(275,300,(ne,nf,1)))
    p0=(r.uniform(80e3,90e3,(ne,nf,1)) if shift else r.uniform(96e3,103e3,(ne,nf,1)))
    bl=(r.uniform(.04,.10,(ne,nf,1)) if shift else r.uniform(0,.06,(ne,nf,1)))*confound
    pa=(r.uniform(60e3,120e3,(ne,nf,1)) if shift else r.uniform(0,80e3,(ne,nf,1)))*confound
    n=ns[None,None]*r.uniform(.98,1.02,(ne,nf,len(ns)))
    H=[x[:,None,None] for x in h]
    y,rr=solve((t0,p0,n,bl,pa),H)
    bias=r.normal(0,1,(4,ne,1,1))*SIG*confound     # fixed per-engine sensor bias
    ym=y*(1+bias+r.normal(0,1,y.shape)*SIG)
    return dict(h=h.T,t0=t0,p0=p0,n=n,y=ym,res=rr.max(),bad=(rr>1e-6).mean(),bl=bl,pa=pa)
def feats(d):
    t0,p0,n=d['t0'],d['p0'],d['n']; th=t0/288.15; de=p0/101325
    f=np.concatenate([d['y'][0]/de,d['y'][1]/th,d['y'][2]/th,d['y'][3]/de/np.sqrt(th),n/np.sqrt(th)],-1)
    return f
def skill(p,d): # 1 - RMSE/RMSE(const mean of prior), per parameter, clipped, averaged (PROVISIONAL probe metric)
    pr=np.array([-.025,-.025,-.02]); e=np.sqrt(((p-d['h'])**2).mean(0)); b=np.sqrt(((pr-d['h'])**2).mean(0))
    s=np.clip(1-e/b,0,1); return s.mean(),s
def agg(F): return np.concatenate([F.mean(1),F.std(1)],1)
def ladder(tag,tr,te):
    from sklearn.linear_model import RidgeCV
    from sklearn.ensemble import HistGradientBoostingRegressor as H
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline as P; from sklearn.preprocessing import StandardScaler as Sc
    from sklearn.multioutput import MultiOutputRegressor as M
    X,Y=agg(feats(tr)),agg(feats(te)); T=tr['h']*100
    out={'ridge':P(Sc(),RidgeCV(np.logspace(-4,3,15))),'HGB':M(H(max_iter=200)),'MLP':P(Sc(),MLPRegressor((64,64),max_iter=1500,early_stopping=True,random_state=0))}
    for k,m in out.items():
        m.fit(X,T); s,a=skill(m.predict(Y)/100,te); print(f"  {tag:26s} {k:6s} {s:.2f} {a.round(2)}")
def physics(tag,te,nuis,known=False):
    from scipy.optimize import least_squares
    pred=[]; ne=min(40,len(te['h'])); t=time.time()
    nf,nn=te['t0'].shape[1],te['n'].shape[2]
    for e in range(ne):
        def f(v):
            h=v[:3]; b=v[3:7] if nuis else np.zeros(4); q=v[7:] if nuis else None
            bl=q[:nf,None]*.01 if nuis else (te['bl'][e] if known else 0*te['t0'][e]); pa=q[nf:,None]*1e4 if nuis else (te['pa'][e] if known else 0*te['t0'][e])
            y,_=solve((te['t0'][e],te['p0'][e],te['n'][e],bl,pa),h/100)
            r=((te['y'][:,e]/y-1)/SIG[:,0]-b[:,None,None]).ravel()
            pri=[(h+[2.5,2.5,2])/1.4]
            if nuis: pri+=[b,(q[:nf]-5)/3,(q[nf:]-6)/4]
            return np.concatenate([r]+pri)
        x0=np.concatenate([[-2.5,-2.5,-2]]+([np.zeros(4),np.full(nf,5.),np.full(nf,6.)] if nuis else []))
        pred.append(least_squares(f,x0,max_nfev=25).x[:3]/100)
    s,a=skill(np.array(pred),dict(h=te['h'][:ne])); print(f"  {tag:40s} {s:.2f} {a.round(2)}  ({(time.time()-t)/ne:.1f}s/engine)",flush=True)
if __name__=="__main__":
    y,r=solve((288.15,101325.,1.0,0.,0.),(0,0,0)); print("design check",y,r)
    for n in (.55,1.0):
        y0,_=solve((288.15,101325.,n,0.,0.),(0,0,0)); y1,_=solve((288.15,101325.,n,0.,0.),(-.02,0,0)); y2,_=solve((288.15,101325.,n,.04,40e3),(0,0,0))
        print(f"N={n}: %change P3,T3,EGT,Wf  wear(-2% eta_c)",((y1/y0-1)*100).round(2)," healthy+bleed4%+40kW",((y2/y0-1)*100).round(2))
    tr=fleet(1500,20,1); te=fleet(200,20,2); sh=fleet(200,20,3,shift=True)
    print("solver res",tr['res'],tr['bad'],te['bad'],sh['bad'])
    print("LIBRARY, 1500 labelled engines"); ladder("same conditions",tr,te); ladder("hot/high airports",tr,sh)
    mix=fleet(1500,20,4,shift=True); both={k:(np.concatenate([tr[k],mix[k]],1 if k=='y' else 0) if isinstance(tr[k],np.ndarray) and tr[k].ndim>0 else 0) for k in tr}
    ladder("hot/high, trained on both",both,sh)
    print("PHYSICS least squares, exact model known")
    physics("ignores bleed/accessory, same",te,False); physics("ignores bleed/accessory, hot/high",sh,False)
    physics("MAP nuisance+bias, same",te,True); physics("MAP nuisance+bias, hot/high",sh,True)
    physics("oracle bleed/power, same",te,False,True)
