exec(open("fixsel.py").read().split('print("\\n=== 버그 vs 수정')[0])
print("\n=== 수정 엔진: seed 0~2 ===")
for nm,J,sk in [("현행 (12개월, 1개월 스킵)",12,1),("후보 (11개월, 2개월 스킵)",11,2)]:
    xs=[run(J=J,sk=sk,seed=s) for s in (0,1,2)]
    n=[b.net for b in xs]
    print(f"  {nm:24s} SR {np.mean([S_(x) for x in n]):5.2f} (seed별 {' '.join(f'{S_(x):.2f}' for x in n)}) CAGR {np.mean([CAGR_(x) for x in n]):6.2f}% MDD {np.mean([MDD_(x) for x in n]):6.1f}% IS {np.mean([S_(x.loc[IS]) for x in n]):5.2f} OOS {np.mean([S_(x.loc[OOS]) for x in n]):5.2f} ~25 {np.mean([S_(x.loc[:'2025-12']) for x in n]):5.2f}")
print("\n=== 종목수 n 민감도 (seed0) ===")
for n_ in (20,25,30,40,50):
    a=run(n=n_).net; b=run(J=11,sk=2,n=n_).net
    print(f"  n={n_:2d}: 현행 SR {S_(a):5.2f} CAGR {CAGR_(a):6.2f}% | 11-2 SR {S_(b):5.2f} CAGR {CAGR_(b):6.2f}%")
print("\n=== 게이트 없는 베이스라인에서도 성립하나 ===")
for nm,J,sk in [("12-1",12,1),("11-2",11,2)]:
    x=run(D=False,J=J,sk=sk).net; print(f"  베이스라인 {nm}: SR {S_(x):5.2f} CAGR {CAGR_(x):6.2f}% MDD {MDD_(x):6.1f}% IS {S_(x.loc[IS]):5.2f} OOS {S_(x.loc[OOS]):5.2f}")
b=run(J=11,sk=2); a=run()
print("\n=== 연도별 % (수정 엔진, seed0) ===")
print(pd.DataFrame({"현행 12-1":a.net.groupby(a.net.index.year).apply(lambda x:(1+x).prod()-1)*100,
                    "후보 11-2":b.net.groupby(b.net.index.year).apply(lambda x:(1+x).prod()-1)*100}).round(1).T.to_string())
print("\n2026 월별 % 11-2:"," ".join(f"{d:%m}:{v*100:+5.1f}" for d,v in b.net.loc["2026"].items()))
print(f"롱 연율 {((1+b.lr).prod()**(12/len(b))-1)*100:.2f}% / 숏 연율 {((1+b.sr).prod()**(12/len(b))-1)*100:.2f}% | 승률 {(b.net>0).mean()*100:.1f}% | 최악월 {b.net.min()*100:.2f}%")
pd.to_pickle({"cur":a,"j11s2":b},"j11_out.pkl")
