exec(open("startdt.py").read().split("starts=[")[0])
import statsmodels.api as sm
def runv(var="rdme",J=11,sk=2,seed=0,n=30,rebal=1,tc=50.,borrow=300.):
    MO_=MOM(J,sk); pw=None;rec=[];Dt=sorted(LB); k=0; keep=None
    for t,nxt in zip(Dt[:-1],Dt[1:]):
        lab=LB[t][seed].dropna(); m=MO_.loc[t,lab.index].astype(float)
        sig=(2*m.rank(pct=True)+m.groupby(lab).transform("mean").rank(pct=True))/3
        tv=TVv.loc[t].dropna(); okl=tv[tv>tv.quantile(.10)].index
        s0=RD.loc[t]; s=s0[H[t].fillna(False)&s0.index.isin(okl)&s0.notna()&np.isfinite(s0)&(s0>0)]
        if len(s)<60: continue
        mv=sig.reindex(s.index); pas=(mv<=mv.quantile(.5)).fillna(False)
        lb=lab.reindex(s.index); sg=s[lb.notna()]; lg=lb.dropna()
        a=RDW.loc[t].reindex(sg.index); am=a.groupby(lg).transform("mean")
        mm=me.loc[t,sg.index].astype(float)
        if var=="rdme": key=sg
        elif var=="invme": key=1/mm                              # 순수 소형주 전략
        elif var=="me": key=mm                                   # 순수 대형주 전략
        elif var=="rnd": key=a                                   # R&D 금액 (스케일 미조정)
        elif var=="resid":                                       # 시총 중립화한 rd_me
            X=sm.add_constant(np.log(mm)); yv=np.log(sg)
            ok=np.isfinite(yv)&np.isfinite(X).all(axis=1)
            key=pd.Series(np.nan,index=sg.index); key[ok]=sm.OLS(yv[ok],X[ok]).fit().resid
        Lc=key[(a>am).fillna(False)].dropna(); Sc=key[pas.reindex(sg.index).fillna(False)&(a<am).fillna(False)].dropna()
        if len(Lc)<n or len(Sc)<n: continue
        if keep is None or k%rebal==0:
            L=Lc.nlargest(n).index; S=Sc.drop(index=L,errors="ignore").nsmallest(n).index
            if len(S)<n: continue
            keep=(L,S)
        else: L,S=keep
        k+=1
        w=pd.concat([pd.Series(1/n,index=L),pd.Series(-1/n,index=S)])
        idx=w.index.union(pw.index) if pw is not None else w.index
        turn=(w.reindex(idx,fill_value=0)-(pw.reindex(idx,fill_value=0) if pw is not None else 0)).abs().sum()/2
        r=RW.loc[nxt].reindex(w.index); g=(w*r).sum(skipna=True); pw=(w*(1+r.fillna(0)))/(1+g)
        rec.append(dict(date=nxt,turn=turn,gross=g,lme=me.loc[t,L].median()/1e6,sme=me.loc[t,S].median()/1e6,
                        lmr=MErank.loc[t,L].mean(),smr=MErank.loc[t,S].mean()))
    b=pd.DataFrame(rec).set_index("date"); b["net"]=b.gross-b.turn*tc/1e4-borrow/1e4/12; return b
print("=== 1) 순위 변수를 바꿔보면 (모든 나머지 규칙 동일, 수정 엔진, seed0) ===")
for tag,v in [("rd_me (채택안)","rdme"),("1/시총 = 순수 소형주","invme"),("시총 = 순수 대형주","me"),
              ("R&D 금액 (시총 미조정)","rnd"),("시총 중립화 rd_me","resid")]:
    b=runv(var=v); x=b.net
    print(f"  {tag:22s} CAGR {CAGR_(x):6.2f}% SR {S_(x):5.2f} MDD {MDD_(x):6.1f}% IS {S_(x.loc[IS]):5.2f} OOS {S_(x.loc[OOS]):5.2f} | 롱 중앙시총 {b.lme.mean():5.1f}조 숏 {b.sme.mean():4.1f}조")
print("\n=== 2) 리밸런싱 주기 (채택안) ===")
for rb,nm in [(1,"월간"),(3,"분기"),(6,"반기"),(12,"연간")]:
    b=runv(rebal=rb); x=b.net
    print(f"  {nm:5s} CAGR {CAGR_(x):6.2f}% SR {S_(x):5.2f} MDD {MDD_(x):6.1f}% 회전 {b.turn.mean()*100:3.0f}%")
print("\n=== 3) rd_me 순위가 시총 순위와 얼마나 같은가 ===")
cs=[]; rc=[]
for t in sorted(LB):
    tv=TVv.loc[t].dropna(); okl=tv[tv>tv.quantile(.10)].index
    s0=RD.loc[t]; s=s0[H[t].fillna(False)&s0.index.isin(okl)&s0.notna()&np.isfinite(s0)&(s0>0)]
    if len(s)<60: continue
    cs.append(s.rank(pct=True).corr(me.loc[t,s.index].rank(pct=True),method="spearman"))
    rc.append(s.rank(pct=True).corr(RDW.loc[t,s.index].rank(pct=True),method="spearman"))
print(f"  rd_me 순위 vs 시총 순위 상관 {np.mean(cs):+.3f} | rd_me 순위 vs R&D금액 순위 상관 {np.mean(rc):+.3f}")
print("\n=== 4) 월별 rd_me 변화의 원천 ===")
ch=[]; 
for t0_,t1_ in zip(sorted(LB)[:-1],sorted(LB)[1:]):
    rn0,rn1=RDW.loc[t0_],RDW.loc[t1_]
    both=rn0.notna()&rn1.notna(); chg=((rn1-rn0).abs()>1)&both
    ch.append(chg.sum()/both.sum())
print(f"  매월 R&D 값이 갱신되는 종목 비중 평균 {np.mean(ch)*100:.1f}% (연 1회 공시이므로 대부분 3~4월에 집중)")
mo=pd.Series(ch,index=sorted(LB)[1:]).groupby(lambda d:d.month).mean()*100
print("  월별 갱신 비중 %:", " ".join(f"{m}월 {v:.0f}" for m,v in mo.items()))
