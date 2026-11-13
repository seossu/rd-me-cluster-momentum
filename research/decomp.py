exec(open("startdt.py").read().split("starts=[")[0])
MEfz=me.copy()
for c in me.columns: pass
ME_dec=me.reindex(me.index).copy()
anchor=me[me.index.month==12].reindex(me.index).ffill()      # 직전 12월 시총으로 고정
def runc(mode="all",J=11,sk=2,seed=0,n=30,tc=50.,borrow=300.):
    MO_=MOM(J,sk); pw=None;rec=[];Dt=sorted(LB); lastsig=None; lastlab=None
    for t,nxt in zip(Dt[:-1],Dt[1:]):
        lab=LB[t][seed].dropna()
        if mode=="freeze_mom":                     # 모멘텀·군집을 연 1회(12월)만 갱신
            if lastlab is None or t.month==12: lastlab=lab; lastsig=None
            lab=lastlab.copy()
        m=MO_.loc[t if mode!="freeze_mom" else (t if t.month==12 or lastsig is None else lastsig)].reindex(lab.index).astype(float)
        if mode=="freeze_mom" and (lastsig is None or t.month==12): lastsig=t
        sig=(2*m.rank(pct=True)+m.groupby(lab).transform("mean").rank(pct=True))/3
        tv=TVv.loc[t].dropna(); okl=tv[tv>tv.quantile(.10)].index
        denom=anchor.loc[t] if mode=="freeze_me" else me.loc[t]
        rdme=(RDW.loc[t].reindex(CM)/(denom.reindex(CM)*1e6)).replace([np.inf,-np.inf],np.nan)
        s=rdme[H[t].fillna(False)&rdme.index.isin(okl)&rdme.notna()&(rdme>0)]
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
        rec.append(dict(date=nxt,turn=turn,gross=g,lme=me.loc[t,L].mean()/1e6,sme=me.loc[t,S].mean()/1e6,
                        lmed=me.loc[t,L].median()/1e6,smed=me.loc[t,S].median()/1e6))
    b=pd.DataFrame(rec).set_index("date"); b["net"]=b.gross-b.turn*tc/1e4-borrow/1e4/12; return b
print("=== 구성요소별 월간 갱신의 기여 (수정 엔진, 채택안, seed0) ===")
for tag,md in [("전부 월간 (채택안)","all"),("시총 고정(직전 12월) → rd_me는 공시 때만 변함","freeze_me"),("모멘텀·군집을 연 1회만 갱신","freeze_mom")]:
    b=runc(mode=md); x=b.net
    print(f"  {tag:42s} CAGR {CAGR_(x):6.2f}% SR {S_(x):5.2f} MDD {MDD_(x):6.1f}% 회전 {b.turn.mean()*100:3.0f}%")
b=runc("all")
print(f"\n=== 롱·숏 레그 시총 (채택안) ===")
print(f"  롱: 평균 {b.lme.mean():5.1f}조 / 중앙 {b.lmed.mean():4.1f}조 | 숏: 평균 {b.sme.mean():5.1f}조 / 중앙 {b.smed.mean():4.1f}조")
import statsmodels.api as sm
X=sm.add_constant(F.reindex(b.net.index)[["MKT","SMB","HML","MOM"]]).dropna()
r4=sm.OLS(b.net.reindex(X.index),X,missing="drop").fit(cov_type="HAC",cov_kwds={"maxlags":12})
print(f"\n=== Carhart 4팩터 (팩터데이터 한계로 {X.index.max():%Y-%m}까지 {len(X)}개월) ===")
for k in ["const","MKT","SMB","HML","MOM"]:
    v=r4.params[k]*(1200 if k=="const" else 1)
    print(f"  {k:6s} {v:+7.3f}{'%' if k=='const' else ' '}  t={r4.tvalues[k]:+5.2f}  p={r4.pvalues[k]:.3f}")
