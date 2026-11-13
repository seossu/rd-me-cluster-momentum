exec(open("startdt.py").read().split("starts=[")[0])
A=run2(J=11,sk=2); B=run2(J=12,sk=1)          # 전체구간 1회 실행 (롤링용)
def st(x): return CAGR_(x),x.std()*np.sqrt(12)*100,S_(x),MDD_(x)
print("=== 실제 재시작 기준 10년 구간 (seed0) ===")
print(f"{'구간':22s} {'개월':>4s} | {'채택안 CAGR':>10s} {'SR':>5s} {'MDD':>7s} | {'현행 CAGR':>9s} {'SR':>5s} {'MDD':>7s}")
wins=[("2016-09~2026-08","2016-08-31","2026-08-31"),("2015-02~2025-01","2015-01-31","2025-01-31"),
      ("2016-01~2025-12","2015-12-31","2025-12-31"),("2015-09~2025-08","2015-08-31","2025-08-31")]
for tag,t0,t1 in wins:
    a=run2(J=11,sk=2,t0=t0).net.loc[:t1]; b=run2(J=12,sk=1,t0=t0).net.loc[:t1]
    ca,va,sa,md_a=st(a); cb,vb,sb,md_b=st(b)
    print(f"{tag:22s} {len(a):4d} | {ca:9.2f}% {sa:5.2f} {md_a:6.1f}% | {cb:8.2f}% {sb:5.2f} {md_b:6.1f}%")
print("\n=== 가능한 모든 10년(120개월) 구간 분포 ===")
for nm,b in [("채택안 11개월/스킵2",A),("현행 12개월/스킵1",B)]:
    x=b.net.dropna(); rows=[]
    for i in range(len(x)-119):
        w=x.iloc[i:i+120]; c,v,s,m=st(w); rows.append((w.index[0],w.index[-1],c,s,m))
    R=pd.DataFrame(rows,columns=["시작","종료","CAGR","SR","MDD"])
    print(f"  {nm}: 구간 {len(R)}개 | SR 중앙 {R.SR.median():.2f} (최저 {R.SR.min():.2f} ~ 최고 {R.SR.max():.2f}) | CAGR 중앙 {R.CAGR.median():.2f}% (최저 {R.CAGR.min():.2f}%) | MDD 최악 {R.MDD.min():.1f}%")
    w=R.loc[R.SR.idxmin()]; print(f"     최악 10년: {w.시작:%Y-%m}~{w.종료:%Y-%m} SR {w.SR:.2f} CAGR {w.CAGR:.2f}% MDD {w.MDD:.1f}%")
    w=R.loc[R.SR.idxmax()]; print(f"     최고 10년: {w.시작:%Y-%m}~{w.종료:%Y-%m} SR {w.SR:.2f} CAGR {w.CAGR:.2f}% MDD {w.MDD:.1f}%")
a=run2(J=11,sk=2,t0="2016-08-31").net.loc[:"2026-08-31"]
print("\n=== 최근 10년(2016-09~2026-08) 채택안 연도별 % ===")
print((a.groupby(a.index.year).apply(lambda x:(1+x).prod()-1)*100).round(1).to_string())
print(f"\n승률 {(a>0).mean()*100:.1f}% | 최악월 {a.min()*100:.2f}% | 최고월 {a.max()*100:.2f}% | 연율변동성 {a.std()*np.sqrt(12)*100:.2f}%")
