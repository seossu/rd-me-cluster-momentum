exec(open("startdt.py").read().split("starts=[")[0])
import statsmodels.api as sm
def fin(seed=0,n=30,tc=50.,borrow=300.):
    pw=None;rec=[];Dt=sorted(LB); MO_=MOM(11,2)
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
        rec.append(dict(date=nxt,turn=turn,gross=g,lr=r[L].mean(),sr=r[S].mean(),nu=len(s),ncl=lg.nunique(),L=list(L),S=list(S)))
    b=pd.DataFrame(rec).set_index("date"); b["net"]=b.gross-b.turn*tc/1e4-borrow/1e4/12; return b
B=fin(); x=B.net
print(f"기간 {x.index[0]:%Y-%m} ~ {x.index[-1]:%Y-%m} ({len(x)}개월) | 월평균 유니버스 {B.nu.mean():.0f}종목 / 군집 {B.ncl.mean():.1f}개")
print(f"CAGR {CAGR_(x):.2f}% | 변동성 {x.std()*np.sqrt(12)*100:.2f}% | Sharpe {S_(x):.2f} | MDD {MDD_(x):.1f}% | 회전율 {B.turn.mean()*100:.1f}% | 승률 {(x>0).mean()*100:.1f}% | 최악월 {x.min()*100:.2f}%")
print(f"롱 연율 {((1+B.lr).prod()**(12/len(B))-1)*100:.2f}% | 숏 연율 {((1+B.sr).prod()**(12/len(B))-1)*100:.2f}%")
print(f"IS(~2020-06) SR {S_(x.loc[IS]):.2f} CAGR {CAGR_(x.loc[IS]):.2f}% | OOS(2020-07~) SR {S_(x.loc[OOS]):.2f} CAGR {CAGR_(x.loc[OOS]):.2f}%")
t10=x.loc["2016-09":]; print(f"최근10년(2016-09~2026-08) CAGR {CAGR_(t10):.2f}% SR {S_(t10):.2f} MDD {MDD_(t10):.1f}%")
print(f"~2025-12 기준 CAGR {CAGR_(x.loc[:'2025-12']):.2f}% SR {S_(x.loc[:'2025-12']):.2f} MDD {MDD_(x.loc[:'2025-12']):.1f}%")
print("\n연도별 %:"); print((x.groupby(x.index.year).apply(lambda v:(1+v).prod()-1)*100).round(1).to_string())
print("\n2026 월별 %:", " ".join(f"{d:%m}:{v*100:+.1f}" for d,v in x.loc["2026"].items()))
sd=[S_(fin(seed=s).net) for s in (0,1,2)]; print(f"\nseed 0/1/2 SR {sd[0]:.2f}/{sd[1]:.2f}/{sd[2]:.2f} (평균 {np.mean(sd):.2f})")
X=sm.add_constant(F.reindex(x.index)[["MKT","SMB","HML","MOM"]]).dropna()
r4=sm.OLS(x.reindex(X.index),X,missing="drop").fit(cov_type="HAC",cov_kwds={"maxlags":12})
print(f"Carhart(~{X.index.max():%Y-%m}, {len(X)}개월) 알파 {r4.params['const']*1200:+.2f}% t={r4.tvalues['const']:.2f} | MKT {r4.params['MKT']:+.3f} SMB {r4.params['SMB']:+.3f}(p{r4.pvalues['SMB']:.2f}) HML {r4.params['HML']:+.3f} MOM {r4.params['MOM']:+.3f}")
last=B.iloc[-1]; print(f"\n최근 리밸런싱({B.index[-1]-pd.offsets.MonthEnd(1):%Y-%m}) 롱: {', '.join(last.L[:10])} ...")
print(f"                              숏: {', '.join(last.S[:10])} ...")
pd.to_pickle(B,"FINAL_out.pkl")
