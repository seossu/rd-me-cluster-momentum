exec(open("fixsel.py").read().split('print("\\n=== 버그 vs 수정')[0])
def run2(J=11,sk=2,seed=0,n=30,t0=None,tc=50.,borrow=300.):
    MO_=MOM(J,sk) if (J,sk)!=(12,1) else mom[CM]
    pw=None;rec=[];Dt=[t for t in sorted(LB) if (t0 is None or t>=pd.Timestamp(t0))]
    for t,nxt in zip(Dt[:-1],Dt[1:]):
        lab=LB[t][seed].dropna(); m=MO_.loc[t,lab.index].astype(float)
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
        rec.append(dict(date=nxt,turn=turn,gross=g))
    b=pd.DataFrame(rec).set_index("date"); b["net"]=b.gross-b.turn*tc/1e4-borrow/1e4/12; return b
starts=["2015-01-31","2015-02-28","2015-03-31","2015-04-30","2015-05-31","2015-06-30","2015-09-30","2015-12-31","2016-06-30"]
for tag,J,sk in [("채택안 11개월/스킵2",11,2),("현행 12개월/스킵1",12,1)]:
    print(f"\n=== {tag} (수정 엔진, seed0) ===")
    print(f"{'시작':10s} {'첫 수익월':9s} {'개월':>4s} {'CAGR':>7s} {'변동성':>7s} {'SR':>6s} {'MDD':>7s} {'IS':>5s} {'OOS':>6s} {'~2025':>6s}")
    for st in starts:
        b=run2(J=J,sk=sk,t0=st); x=b.net.dropna()
        isr=S_(x.loc[IS]) if len(x.loc[IS])>12 else float('nan')
        print(f"{st[:7]:10s} {x.index[0]:%Y-%m}   {len(x):4d} {CAGR_(x):6.2f}% {x.std()*np.sqrt(12)*100:6.2f}% {S_(x):6.2f} {MDD_(x):6.1f}% {isr:5.2f} {S_(x.loc[OOS]):6.2f} {S_(x.loc[:'2025-12']):6.2f}")
print("\n=== seed 0~2 전체 분포 (채택안) ===")
for st in ["2015-01-31","2015-03-31","2015-06-30"]:
    v=[S_(run2(t0=st,seed=s).net) for s in (0,1,2)]
    c=[CAGR_(run2(t0=st,seed=s).net) for s in (0,1,2)]
    print(f"  {st[:7]}: SR {np.mean(v):.2f} (seed별 {' '.join(f'{x:.2f}' for x in v)}) | CAGR {np.mean(c):.2f}%")
