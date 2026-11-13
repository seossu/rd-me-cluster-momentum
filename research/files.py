exec(open("loader2.py").read())
import statsmodels.api as sm, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
plt.rcParams["font.family"]="AppleGothic"; plt.rcParams["axes.unicode_minus"]=False
D=pd.read_pickle("deliver_out.pkl"); B,Bs,Q,G=D["B"],D["Bs"],D["Q"],D["G"]
OUT=str(Path(__file__).resolve().parents[1]/"results")
x=B.net; nav=(1+x).cumprod()
k=F.KOSPI.reindex(x.index)
def met(v,kk=None):
    d=dict(개월=len(v),CAGR=CAGR_(v),연율변동성=v.std()*np.sqrt(12)*100,Sharpe=S_(v),MDD=MDD_(v),승률=(v>0).mean()*100,최악월=v.min()*100)
    if kk is not None and kk.notna().sum()>12: d.update({"KOSPI CAGR":CAGR_(kk.dropna()),"KOSPI Sharpe":S_(kk),"KOSPI MDD":MDD_(kk)})
    return d
SUM=pd.DataFrame({"전체 2015-02~2026-08":met(x,k),"IS 2015-02~2020-06":met(x.loc[IS],k.loc[IS]),
                  "OOS 2020-07~2026-08":met(x.loc[OOS],k.loc[OOS]),"최근10년 2016-09~2026-08":met(x.loc["2016-09":],k.loc["2016-09":]),
                  "~2025-12":met(x.loc[:"2025-12"],k.loc[:"2025-12"])}).T
MON=pd.DataFrame({"그로스수익":B.gross,"순수익":x,"NAV":nav,"회전율":B.turn,"롱수익":B.롱수익,"숏수익":B.숏수익,
                  "유니버스종목수":B.유니버스,"군집수":B.군집수,"KOSPI":k})
YR=pd.DataFrame({"전략":x.groupby(x.index.year).apply(lambda v:(1+v).prod()-1)*100,
                 "KOSPI":k.groupby(k.index.year).apply(lambda v:(1+v).prod()-1)*100,
                 "롱레그":B.롱수익.groupby(B.index.year).apply(lambda v:(1+v).prod()-1)*100,
                 "숏레그":B.숏수익.groupby(B.index.year).apply(lambda v:(1+v).prod()-1)*100}).round(2)
QN=Q.copy()
for c in Q.columns: QN[c+" NAV"]=(1+Q[c]).cumprod()
QS=pd.DataFrame({c:met(Q[c]) for c in Q.columns}).T
SEED=pd.DataFrame({f"seed {s}":met(b.net) for s,b in Bs.items()}).T
X=sm.add_constant(F.reindex(x.index)[["MKT","SMB","HML","MOM"]]).dropna()
r4=sm.OLS(x.reindex(X.index),X,missing="drop").fit(cov_type="HAC",cov_kwds={"maxlags":12})
REG=pd.DataFrame({"계수":[r4.params["const"]*1200]+[r4.params[c] for c in ["MKT","SMB","HML","MOM"]],
                  "t값":[r4.tvalues["const"]]+[r4.tvalues[c] for c in ["MKT","SMB","HML","MOM"]],
                  "p값":[r4.pvalues["const"]]+[r4.pvalues[c] for c in ["MKT","SMB","HML","MOM"]]},
                 index=["알파(연율 %)","MKT","SMB","HML","MOM"]).round(4)
last=B.iloc[-1]
HOLD=pd.DataFrame({"롱 30종목":last.롱.split(", "),"숏 30종목":last.숏.split(", ")})
SPEC=pd.DataFrame({"항목":["유니버스","유동성 필터","제외","트리 입력","거리","군집화","모멘텀 창","신호","롱 규칙","숏 규칙","비중","리밸런싱","거래비용","대차비용","익스포저","PIT 기준","백테스트 기간","월평균 유니버스"],
 "내용":["R&D 공시기업 중 시가총액 상위 30% (분기 갱신)","60일 평균 거래대금 하위 10% 제외","지주사",
        "직전 104주 주간수익률의 시장베타 잔차","d = √(2(1−ρ))","MST(Mantegna 1999) → Louvain(Blondel 2008, seed 0, weight=ρ+1)",
        "t−13 ~ t−2 (11개월, 직전 2개월 제외)","(2·rank(자기 모멘텀) + 1·rank(군집평균 모멘텀)) / 3",
        "R&D 금액 > 소속 군집 평균 → 그중 rd_me 상위 30종목","모멘텀 점수 하위 50% ∩ R&D 금액 < 군집 평균 → 그중 rd_me 하위 30종목",
        "동일가중 (롱 1/30, 숏 −1/30)","월간","왕복 50bp × 회전율","연 300bp (숏 익스포저)","그로스 200% / 넷 0%",
        "DART 접수일 + MonthEnd(1)","2015-02 ~ 2026-08 (139개월)",f"{B.유니버스.mean():.0f}종목 / 군집 {B.군집수.mean():.1f}개"]})
LIM=pd.DataFrame({"공개 항목":["Louvain seed 의존성","파라미터가 국소 최대값","2020년 의존","모멘텀 창 사후 선택","팩터 회귀 기간","숏 선택 버그 수정"],
 "내용":[f"seed 0/1/2 Sharpe {S_(Bs[0].net):.2f}/{S_(Bs[1].net):.2f}/{S_(Bs[2].net):.2f}. 헤드라인은 seed 0",
        "시총컷 30%, 유동성 하위 10%, 트리창 104주, 종목수 30, 모멘텀 11개월 모두 이웃값보다 우수한 지점. 사후 선택임을 명시",
        "2020년 +52.5%가 모든 10년 구간에 포함. 제외 시 CAGR 3~4%p 하락",
        "직전 1개월 추가 제외는 Novy-Marx(2012) 에코·Jegadeesh(1990) 단기반전으로 방향은 지지되나 '2개월'이라는 값 자체는 사후적. 10개월창(스킵3)도 Sharpe 1.22이나 MDD −17.2%",
        f"팩터 데이터가 2026-02까지여서 알파·로딩은 {len(X)}개월 기준. 2026-03~08은 수익률만 반영",
        "2026-09-24 수정: pandas Index.difference()의 가나다순 정렬로 숏이 'rd_me 하위 60종목 중 이름순 앞 30개'였음. 수정 후 진짜 하위 30종목"]})
fp=f"{OUT}/rd_me_MST_최종전략_2026-08.xlsx"
with pd.ExcelWriter(fp,engine="openpyxl") as w:
    SPEC.to_excel(w,"전략사양",index=False); SUM.round(2).to_excel(w,"성과요약")
    YR.to_excel(w,"연도별"); MON.to_excel(w,"월별")
    QS.round(2).to_excel(w,"rd_me5분위_요약"); QN.to_excel(w,"rd_me5분위_월별")
    G.round(3).to_excel(w,"파라미터격자_Sharpe"); SEED.round(2).to_excel(w,"seed민감도")
    REG.to_excel(w,"Carhart4팩터"); HOLD.to_excel(w,"최근리밸런싱_2026-07",index=False)
    LIM.to_excel(w,"공개사항",index=False)
print("엑셀:",fp)
MON.to_csv(f"{OUT}/rd_me_MST_최종_월별NAV.csv",encoding="utf-8-sig")
QN.to_csv(f"{OUT}/rd_me_MST_최종_5분위NAV.csv",encoding="utf-8-sig")
# ── 히트맵 (순차 blue 램프, 셀에 값 직접 표기)
ramp=["#cde2fb","#b7d3f6","#9ec5f4","#86b6ef","#6da7ec","#5598e7","#3987e5","#2a78d6","#256abf","#1c5cab","#184f95","#104281","#0d366b"]
cmap=LinearSegmentedColormap.from_list("blue_seq",ramp)
fig,ax=plt.subplots(figsize=(7.2,5.0),facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
V=G.values
im=ax.imshow(V,cmap=cmap,vmin=0.5,vmax=1.35,aspect="auto")
ax.set_xticks(range(len(G.columns)),[c.replace("종목","") for c in G.columns],color="#52514e")
ax.set_yticks(range(len(G.index)),[i.replace("주","") for i in G.index],color="#52514e")
ax.set_xlabel("레그별 종목수 (개)",color="#52514e",fontsize=10,labelpad=8)
ax.set_ylabel("트리 추정 창 (주)",color="#52514e",fontsize=10,labelpad=8)
ax.set_title("파라미터 민감도: Sharpe 비율",color="#0b0b0b",fontsize=13,pad=14,loc="left")
for i in range(V.shape[0]):
    for j in range(V.shape[1]):
        ax.text(j,i,f"{V[i,j]:.2f}",ha="center",va="center",fontsize=11.5,
                color="#ffffff" if V[i,j]>1.05 else "#0b0b0b")
ri,ci=list(G.index).index("104주"),list(G.columns).index("30종목")
ax.add_patch(Rectangle((ci-.5,ri-.5),1,1,fill=False,ec="#eb6834",lw=2.5,zorder=5))
ax.text(ci+0.58,ri-0.30,"채택",color="#eb6834",fontsize=10,va="center")
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0,labelsize=10)
cb=fig.colorbar(im,ax=ax,fraction=.042,pad=.03); cb.outline.set_visible(False)
cb.ax.tick_params(length=0,labelsize=9,colors="#52514e"); cb.set_label("Sharpe",color="#52514e",fontsize=9)
fig.text(0.01,-0.02,"2015-02~2026-08, 거래비용 왕복 50bp + 대차 300bp, 동일가중, Louvain seed 0",
         color="#52514e",fontsize=8.5)
fig.tight_layout()
fig.savefig(f"{OUT}/rd_me_MST_최종_파라미터히트맵.png",dpi=200,bbox_inches="tight",facecolor="#fcfcfb")
print("히트맵 저장")
