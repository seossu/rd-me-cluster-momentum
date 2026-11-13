exec(open("loader2.py").read())
import networkx as nx, statsmodels.api as sm
from scipy.sparse.csgraph import minimum_spanning_tree
from networkx.algorithms.community import louvain_communities
T1=pd.Timestamp("2026-08-31"); RDW=RD*(me*1e6)
pxd=pn.sort_index(); DR=np.log(pxd.where(pxd>0)).diff(); WR=DR.resample("W-FRI").sum(min_count=3)
def resid(R):
    R=R.fillna(0.0); mkt=R.mean(axis=1); mc=mkt-mkt.mean(); Rc=R-R.mean()
    beta=(Rc.mul(mc,axis=0)).sum()/(mc**2).sum(); return Rc-np.outer(mc,beta)
H={}; h=None
for t in [x for x in me.index if T0<=x<=T1]:
    if h is None or t.month in (3,6,9,12): h=MErank.loc[t]>=0.70
    H[t]=h
def build(win,seeds=(0,)):
    out={}
    for t in [d for d in me.index if T0<=d<=T1]:
        if t not in mom.index: continue
        tv=TVv.loc[t].dropna(); okl=tv[tv>tv.quantile(.10)].index
        u=[c for c in CM if H[t].get(c,False) and c in okl and c in DR.columns and pd.notna(mom.at[t,c])]
        ww=WR.loc[:t,u].tail(win); ok=[c for c in u if ww[c].notna().sum()>=int(win*0.77)]
        if len(ok)<60: continue
        C=np.nan_to_num(np.corrcoef(resid(ww[ok]).values.T))
        Dm=np.sqrt(np.clip(2*(1-C),1e-9,None)); np.fill_diagonal(Dm,0)
        Tm=minimum_spanning_tree(Dm).toarray(); G=nx.Graph()
        for a,b in np.argwhere(Tm>0): G.add_edge(ok[a],ok[b],weight=float(1-Dm[a,b]**2/2)+1.0)
        out[t]={s:pd.Series({n:i for i,c in enumerate(louvain_communities(G,weight="weight",seed=s)) for n in c}).reindex(ok) for s in seeds}
    return out
MOM112=(ma.shift(2)/ma.shift(13)-1)[CM]
def bt(LB,seed=0,n=30,tc=50.,borrow=300.):
    pw=None;rec=[];Dt=sorted(LB)
    for t,nxt in zip(Dt[:-1],Dt[1:]):
        lab=LB[t][seed].dropna(); m=MOM112.loc[t,lab.index].astype(float)
        sig=(2*m.rank(pct=True)+m.groupby(lab).transform("mean").rank(pct=True))/3
        tv=TVv.loc[t].dropna(); okl=tv[tv>tv.quantile(.10)].index
        s0=RD.loc[t]; s=s0[H[t].fillna(False)&s0.index.isin(okl)&s0.notna()&np.isfinite(s0)&(s0>0)]
        if len(s)<60: continue
        mv=sig.reindex(s.index); pas=(mv<=mv.quantile(.5)).fillna(False)
        lb=lab.reindex(s.index); sg=s[lb.notna()]; lg=lb.dropna()
        a=RDW.loc[t].reindex(sg.index); am=a.groupby(lg).transform("mean")
        Lc=sg[(a>am).fillna(False)]; Sc=sg[pas.reindex(sg.index).fillna(False)&(a<am).fillna(False)]
        if len(Lc)<n or len(Sc)<n: continue
        L=Lc.nlargest(n).index; S=Sc.drop(index=L,errors="ignore").nsmallest(n).index
        if len(S)<n: continue
        w=pd.concat([pd.Series(1/n,index=L),pd.Series(-1/n,index=S)])
        idx=w.index.union(pw.index) if pw is not None else w.index
        turn=(w.reindex(idx,fill_value=0)-(pw.reindex(idx,fill_value=0) if pw is not None else 0)).abs().sum()/2
        r=RW.loc[nxt].reindex(w.index); g=(w*r).sum(skipna=True); pw=(w*(1+r.fillna(0)))/(1+g)
        rec.append(dict(date=nxt,형성월=t,turn=turn,gross=g,롱수익=r[L].mean(),숏수익=r[S].mean(),
                        유니버스=len(s),군집수=lg.nunique(),롱=", ".join(L),숏=", ".join(S)))
    b=pd.DataFrame(rec).set_index("date"); b["net"]=b.gross-b.turn*tc/1e4-borrow/1e4/12; return b
print("트리 생성 104주 (seed 0~2)…"); LB104=build(104,(0,1,2))
B=bt(LB104); print(f"메인 백테스트 완료 {len(B)}개월 SR {S_(B.net):.2f}")
Bs={s:bt(LB104,seed=s) for s in (0,1,2)}
# rd_me 5분위 (롱온리 EW)
qnav={}
rows=[]
for t,nxt in zip(sorted(LB104)[:-1],sorted(LB104)[1:]):
    tv=TVv.loc[t].dropna(); okl=tv[tv>tv.quantile(.10)].index
    s0=RD.loc[t]; s=s0[H[t].fillna(False)&s0.index.isin(okl)&s0.notna()&np.isfinite(s0)&(s0>0)]
    if len(s)<60: continue
    q=pd.qcut(s.rank(method="first"),5,labels=[1,2,3,4,5])
    r=RW.loc[nxt].reindex(s.index)
    rows.append(dict(date=nxt,**{f"Q{i}": r[q==i].mean() for i in range(1,6)}))
Q=pd.DataFrame(rows).set_index("date")
# 파라미터 그리드
grid={}
for win in (52,78,104,156):
    LBw=LB104 if win==104 else build(win,(0,))
    for n_ in (20,25,30,40):
        grid[(win,n_)]=S_(bt(LBw,n=n_).net)
    print(f"  격자 {win}주 완료")
G=pd.DataFrame([[grid[(w,n_)] for n_ in (20,25,30,40)] for w in (52,78,104,156)],
               index=[f"{w}주" for w in (52,78,104,156)],columns=[f"{n_}종목" for n_ in (20,25,30,40)])
pd.to_pickle({"B":B,"Bs":Bs,"Q":Q,"G":G},"deliver_out.pkl")
print("\n파라미터 격자 (Sharpe)\n"+G.round(2).to_string())
