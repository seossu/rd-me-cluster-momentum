exec(open("loader2.py").read())
import networkx as nx
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
LB={}
for t in [d for d in me.index if T0<=d<=T1]:
    if t not in mom.index: continue
    tv=TVv.loc[t].dropna(); okl=tv[tv>tv.quantile(.10)].index
    u=[c for c in CM if H[t].get(c,False) and c in okl and c in DR.columns and pd.notna(mom.at[t,c])]
    ww=WR.loc[:t,u].tail(104); ok=[c for c in u if ww[c].notna().sum()>=80]
    if len(ok)<60: continue
    C=np.nan_to_num(np.corrcoef(resid(ww[ok]).values.T))
    Dm=np.sqrt(np.clip(2*(1-C),1e-9,None)); np.fill_diagonal(Dm,0)
    Tm=minimum_spanning_tree(Dm).toarray(); G=nx.Graph()
    for a,b in np.argwhere(Tm>0): G.add_edge(ok[a],ok[b],weight=float(1-Dm[a,b]**2/2)+1.0)
    LB[t]={s:pd.Series({n:i for i,c in enumerate(louvain_communities(G,weight="weight",seed=s)) for n in c}).reindex(ok) for s in (0,1,2)}
def MOM(J,sk): return (ma.shift(sk)/ma.shift(sk+J)-1)[CM]
def run(J=12,sk=1,seed=0,n=30,D=True,fix=True,tc=50.,borrow=300.):
    MO_=MOM(J,sk) if (J,sk)!=(12,1) else mom[CM]
    pw=None;rec=[];Dt=sorted(LB)
    for t,nxt in zip(Dt[:-1],Dt[1:]):
        lab=LB[t][seed].dropna(); m=MO_.loc[t,lab.index].astype(float)
        sig=(2*m.rank(pct=True)+m.groupby(lab).transform("mean").rank(pct=True))/3
        tv=TVv.loc[t].dropna(); okl=tv[tv>tv.quantile(.10)].index
        s0=RD.loc[t]; s=s0[H[t].fillna(False)&s0.index.isin(okl)&s0.notna()&np.isfinite(s0)&(s0>0)]
        if len(s)<60: continue
        mv=sig.reindex(s.index); pas=(mv<=mv.quantile(.5)).fillna(False)
        lb=lab.reindex(s.index); sg=s[lb.notna()]; lg=lb.dropna()
        a=RDW.loc[t].reindex(sg.index); am=a.groupby(lg).transform("mean")
        ps=pas.reindex(sg.index).fillna(False)
        if D:
            Lc=sg[(a>am).fillna(False)]; Sc=sg[ps&(a<am).fillna(False)]
        else:
            Lc=sg; Sc=sg[ps]
        if len(Lc)<n or len(Sc)<n: continue
        L=Lc.nlargest(n).index
        S=(Sc.drop(index=L,errors="ignore").nsmallest(n).index if fix
           else Sc.nsmallest(2*n).index.difference(L)[:n])
        if len(S)<n: continue
        w=pd.concat([pd.Series(1/n,index=L),pd.Series(-1/n,index=S)])
        idx=w.index.union(pw.index) if pw is not None else w.index
        turn=(w.reindex(idx,fill_value=0)-(pw.reindex(idx,fill_value=0) if pw is not None else 0)).abs().sum()/2
        r=RW.loc[nxt].reindex(w.index); g=(w*r).sum(skipna=True); pw=(w*(1+r.fillna(0)))/(1+g)
        rec.append(dict(date=nxt,turn=turn,gross=g,lr=r[L].mean(),sr=r[S].mean(),srd=sg.rank(pct=True).reindex(S).mean()))
    b=pd.DataFrame(rec).set_index("date"); b["net"]=b.gross-b.turn*tc/1e4-borrow/1e4/12; return b
def LN(tag,b):
    x=b.net.dropna()
    return f"{tag:30s} CAGR {CAGR_(x):6.2f}% VOL {x.std()*np.sqrt(12)*100:5.2f}% SR {S_(x):5.2f} MDD {MDD_(x):6.1f}% 회전 {b.turn.mean()*100:3.0f}% IS {S_(x.loc[IS]):5.2f} OOS {S_(x.loc[OOS]):5.2f} ~25 {S_(x.loc[:'2025-12']):5.2f} 숏rd_me백분위 {b.srd.mean()*100:4.1f}%"
print("\n=== 버그 vs 수정 (D, 12-1, seed0) ===")
print(LN("버그(가나다순 30개)",run(fix=False)))
print(LN("수정(진짜 하위 30개)",run(fix=True)))
print("\n=== 수정 엔진에서 모멘텀 창 ===")
for J in (11,12):
    for sk in (1,2,3):
        print(LN(f"J{J} skip{sk}",run(J=J,sk=sk)))
print("\n=== 수정 엔진, seed 교란 ===")
for s in (0,1,2):
    print(LN(f"seed{s} 12-1",run(seed=s)) )
    print(LN(f"seed{s} 12-2",run(J=12,sk=2,seed=s)))
print("\n=== 수정 엔진, 게이트 없는 베이스라인 ===")
print(LN("베이스라인 12-1",run(D=False)))
print(LN("베이스라인 12-2",run(D=False,J=12,sk=2)))
b1=run(); b2=run(J=12,sk=2)
print("\n연도별 %"); print(pd.DataFrame({"D 12-1(수정)":b1.net.groupby(b1.net.index.year).apply(lambda x:(1+x).prod()-1)*100,
  "D 12-2(수정)":b2.net.groupby(b2.net.index.year).apply(lambda x:(1+x).prod()-1)*100,
  "D 12-1(버그)":run(fix=False).net.groupby(lambda d:d.year).apply(lambda x:(1+x).prod()-1)*100}).round(1).to_string())
pd.to_pickle({"fix12_1":b1,"fix12_2":b2,"bug":run(fix=False)},"fixsel_out.pkl")
