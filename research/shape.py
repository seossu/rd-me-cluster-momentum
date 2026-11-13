exec(open("fixsel.py").read().split('print("\\n=== 버그 vs 수정')[0])
def S3(J,sk):
    xs=[run(J=J,sk=sk,seed=s).net for s in (0,1,2)]
    return np.mean([S_(x) for x in xs]),[round(S_(x),2) for x in xs],np.mean([CAGR_(x) for x in xs]),np.mean([MDD_(x) for x in xs]),np.mean([S_(x.loc[IS]) for x in xs]),np.mean([S_(x.loc[OOS]) for x in xs])
print("\n=== 12개월 창 유지, 스킵만 늘릴 때 (seed 0~2 평균) ===")
for sk in (1,2,3,4):
    m,ss,c,md,i_,o=S3(12,sk); print(f"  12개월 창, 스킵 {sk}개월 (t-{13+sk-1}→t-{sk}): SR {m:5.2f} {ss} CAGR {c:6.2f}% MDD {md:6.1f}% IS {i_:5.2f} OOS {o:5.2f}")
print("\n=== 창 길이를 고정 끝점(t-13)에서 줄일 때 ===")
for J,sk in [(12,1),(11,2),(10,3),(9,4)]:
    m,ss,c,md,i_,o=S3(J,sk); print(f"  {J}개월 창, 스킵 {sk} (t-13→t-{sk}): SR {m:5.2f} {ss} CAGR {c:6.2f}% MDD {md:6.1f}% IS {i_:5.2f} OOS {o:5.2f}")
print("\n=== Novy-Marx(2012) 에코 효과 검증: 최근 구간 vs 중간 구간 ===")
for tag,J,sk in [("최근 6개월 (t-7→t-1)",6,1),("중간 6개월 (t-13→t-7) = 에코",6,7),("최근 3개월 (t-4→t-1)",3,1),("중간 9개월 (t-13→t-4)",9,4)]:
    m,ss,c,md,i_,o=S3(J,sk); print(f"  {tag:28s} SR {m:5.2f} {ss} CAGR {c:6.2f}% MDD {md:6.1f}% IS {i_:5.2f} OOS {o:5.2f}")
