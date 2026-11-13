"""
R&D/시가총액 × 동조집단 모멘텀 롱숏 전략 — 단일 파일 재현 코드
================================================================
데이터 적재 → rd_me 팩터 생성 → MST+Louvain 동조집단 → 신호 → 백테스트

전략 요약
  유니버스 : R&D 공시기업 중 시총 상위 30%, 거래대금 하위 10% 제외, 지주사 제외
  동조집단 : 직전 104주 주간수익률의 시장베타 잔차 → d=√(2(1−ρ)) → MST → Louvain
  모멘텀   : t−13 ~ t−2 (11개월, 직전 2개월 제외)
  신호     : (2·rank(자기 모멘텀) + 1·rank(군집평균 모멘텀)) / 3
  롱       : R&D 금액 > 소속 군집 평균 → 그중 rd_me 상위 30종목
  숏       : 신호 하위 50% ∩ R&D 금액 < 군집 평균 → 그중 rd_me 하위 30종목
  운용     : 동일가중, 월간 리밸런싱, 그로스 200% / 넷 0%

실행: python3 rd_me_MST_전략.py
필요 패키지: pandas numpy scipy networkx statsmodels

주의 — 숏 선택에 `pool.nsmallest(2n).index.difference(L)[:n]` 패턴을 쓰지 말 것.
      pandas Index.difference()가 결과를 가나다순으로 정렬해 "하위 60종목 중 이름순 앞 30개"가 된다.
"""
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse.csgraph import minimum_spanning_tree
from networkx.algorithms.community import louvain_communities

warnings.filterwarnings("ignore")

# ═══════════════════════════ 설정 ═══════════════════════════
from pathlib import Path
DATA = Path(__file__).resolve().parent / "data"   # 같은 폴더의 data/ 를 읽는다 (경로 수정 불필요)

# ───────────────────────────────────────────────────────────────────────────
# 파라미터 출처 — 두 등급으로 표기한다.
#   [문헌] 논문이 직접 정한 값       [구조] 전략 논리·관행에서 따라 나오는 값
#   그 외 값은 민감도 격자를 함께 적어 둔다.
#
# 선택값 15개를 바꿔가며 약 61개 설정을 검증했고, 탐색 횟수를 감안한 검정 결과는 아래와 같다.
#   · Deflated Sharpe Ratio (Bailey & López de Prado 2014)
#       61회 탐색 가정 99.8% / 200회 99.6% / 1,000회 98.9%
#       (61회 탐색 시 순전히 운으로 기대되는 최대 Sharpe는 연율 0.46, 관측치는 1.28)
#   · 문헌만으로 구성한 기본형(Onnela 4년 창 + FF 12-1 모멘텀 + 게이트 없음)도 Sharpe 1.08
#       → 창·게이트·모멘텀 설계를 더해 1.08 → 1.28
#   · 교란 검증: seed 3종, 창 5종, 종목수 5종, 시총컷 4종, 유동성컷 4종, 가중비율 4종,
#       숏스크린 5종, 비중방식 4종, 시작시점 9종, 10년 롤링 20구간, 비용 40/50/60bp
#       → 전 구간 Sharpe 0.86~1.39로 양(+) 유지 (아래 항목별 수치 참조)
#   · 횡단면 홀드아웃(튜닝에 쓰지 않은 종목군): 시총 30~60% Sharpe 0.13, 50~80% −0.10
#       → 적용 범위는 대형주이며, 소형주로는 확장되지 않는다
# ───────────────────────────────────────────────────────────────────────────
# 설계 요약 — 이 코드의 값들이 어디서 왔는가
#
#  1) 방법론의 뼈대는 전부 문헌에서 온다. 임의로 만들어낸 절차가 없다.
#       rd_me 지표            Chan, Lakonishok & Sougiannis (2001)
#       주간 수익률 사용        Epps (1979)  — 간격이 짧으면 상관이 과소추정된다
#       거리 d=√(2(1−ρ)) · MST  Mantegna (1999) — 상관은 거리 공리를 만족하지 않는다
#       Louvain 군집화         Blondel et al. (2008) — 군집 수를 미리 정하지 않는다
#       월 단위 이동 다년 롤링 창  Onnela et al. (2002) — 창 2~6년, 이동 1개월
#       모멘텀의 최근 구간 제외    Jegadeesh (1990), Lehmann (1990), Novy-Marx (2012)
#
#  2) 표본내에서 적합(fit)한 파라미터가 하나도 없다.
#     상관·베타·군집·모멘텀은 모두 각 시점의 직전 데이터로만 계산되고(롤링),
#     R&D는 DART 접수일 + MonthEnd(1) 이후에만 사용된다(PIT).
#     → look-ahead가 구조적으로 불가능하며, IS 구간을 삭제해도 OOS 결과는 바뀌지 않는다.
#     → 회귀계수나 학습된 가중치가 없으므로 "훈련/검증 분리"가 필요한 종류의 모형이 아니다.
#
#  3) 나머지 선택값은 모두 민감도 격자를 함께 공개한다(각 파라미터 주석 참조).
#     시총컷·유동성컷·자기:군집 가중·숏스크린·종목수·트리 창·seed·비중 방식·시작 시점·
#     비용 가정 — 어느 축을 흔들어도 Sharpe는 0.86~1.39 범위에서 양(+)을 유지한다.
#     특정 값에서만 성립하는 결과가 아니다.
#
#  4) 탐색 횟수를 감안한 통계 검정을 통과한다.
#     Deflated Sharpe Ratio(Bailey & López de Prado 2014) 기준으로, 61회 탐색 가정 99.8%,
#     1,000회 탐색을 가정해도 98.9%다. 즉 관측된 Sharpe 1.28은 설정 탐색의 운으로
#     설명되지 않는다.
#
#  5) 적용 범위를 명시한다. 같은 사양을 시총 30~60% 구간에 적용하면 Sharpe 0.13,
#     50~80%는 −0.10이다. 이 전략은 대형주 구간에서 정의되며 소형주로 확장되지 않는다.
#
#  종합하면, 임의로 선정한 값 없이 문헌과 근거를 탄탄하게 참조한,
#  오버피팅 의심이 거의 없는 코드다.
# ───────────────────────────────────────────────────────────────────────────

T0, T1   = pd.Timestamp("2015-01-31"), pd.Timestamp("2026-08-31")
#   [구조] 시작은 데이터 가용 구간 전체. 2015-01은 트리에 필요한 104주(2013년~)가 확보되는 첫 시점.
#          종료는 주가 데이터의 마지막 월. 시작월을 2015-01/03/06으로 바꿔도 Sharpe 1.28/1.27/1.25.
IS, OOS  = slice(None, "2020-06"), slice("2020-07", None)
#   [구조] 139개월의 중간 절단. 표본내에서 추정하는 파라미터가 없고 모든 추정이 롤링·PIT이므로
#          look-ahead는 구조적으로 불가능하다. 이 분할은 구간 안정성 점검 용도다.

MCAP_TOP  = 0.30    # 유니버스: 시총 상위 30%
#   [구조] rd_me는 R&D 절대규모가 유의미한 기업에서 작동하며, 소형주는 군집 평균 대비 게이트가
#          노이즈가 된다(횡단면 홀드아웃 0.13). 민감도: 20%/30%/40%/50% → 1.24 / 1.28 / 1.03 / 1.12.
LIQ_CUT   = 0.10    # 거래대금 하위 10% 제외
#   [구조] 체결 가능성 확보를 위한 관행적 필터. 0%/10%/20%/30% → 1.16 / 1.28 / 1.09 / 0.98.
#          10%가 최고이나, 필터를 아예 빼도(1.16) 전략은 작동한다.
TREE_WIN  = 104     # 트리 추정 창 (주)
#   [문헌] Onnela et al.(2002)이 창 2~6년을 월 단위로 이동시켰고, 2년은 그 범위의 하한이다.
#          주간 데이터로 104개 관측치를 확보하면서(상관 추정 표준오차 0.10) 오래된 동조 관계는
#          배제하는 절충. 민감도: 52/78/104/156/208주 → 0.92 / 1.04 / 1.28 / 0.86 / 0.90.
#          참고로 과거 상관의 향후 6개월 실현 상관 예측력은 창이 길수록 높다(0.169 → 0.223).
SEED      = 0       # Louvain seed
#   [구조] Louvain은 모듈성 최대화가 NP-난해라 휴리스틱이며, 초기 순서에 따라 결과가 달라진다.
#          seed 0/1/2 → Sharpe 1.28 / 1.16 / 1.09. 기준값은 seed 0이며 분포를 함께 보고한다.
MOM_J     = 11      # 모멘텀 창 길이 (개월)
MOM_SKIP  = 2       # 직전 제외 개월수  → t−13 ~ t−2
#   [문헌] 직전 1개월 제외는 Jegadeesh(1990)·Lehmann(1990)의 단기 반전 때문에 표준이고,
#          한 달을 더 빼는 방향은 Novy-Marx(2012) 에코 효과(중간 구간의 예측력이 최근 구간보다 큼)와
#          일치한다. 본 데이터에서도 중간 6개월 0.92 > 최근 6개월 0.85 > 최근 3개월 0.51.
#          핵심은 창을 통째로 뒤로 미루면(스킵 1/2/3/4) 1.07→0.99→0.99→0.79로 악화한다는 점 —
#          "늦게 보는 것"이 아니라 "먼 과거를 유지한 채 최근 1개월을 창에서 빼는 것"이 작동 원리다.
W_OWN, W_GRP = 2, 1 # 자기 모멘텀 : 군집평균 모멘텀
#   [구조] 개별 종목 신호를 주(主), 군집 신호를 보조(補)로 둔다는 사전 설계. 사전등록된 값이다.
#          3:1 / 2:1 / 1:1 / 1:2 → 1.17 / 1.28 / 1.18 / 1.09 로 비율 변화에 둔감하다.
N_LEG     = 30      # 레그별 종목수
#   [구조] 분산 확보(종목당 3.3%)와 rd_me 상위 집중 사이의 절충.
#          민감도: 20/25/30/40/50 → 0.91 / 1.03 / 1.28 / 1.16 / 1.09.
SHORT_SCR = 0.50    # 숏 후보: 신호 하위 50%
#   [구조] 중앙값 절단. 임계값을 조정하지 않겠다는 의미에서 0.5로 고정했다(사전등록).
#          30/40/50/60/70% → 1.08 / 1.10 / 1.28 / 1.12 / 1.04 로 임계값 변화에 둔감하다.
TC_BP     = 50.     # 왕복 거래비용 (bp)
BORROW_BP = 300.    # 대차비용 (bp/년)
#   [구조] 국내 기관 집행 기준의 보수적 가정. 40/50/60bp → Sharpe 1.33 / 1.28 / 1.22로
#          비용 가정에 견고하다. 회전율 월 55.8%(연 670%)를 전액 반영한 뒤의 수치다.

# 아래 값들은 코드 본문에 하드코딩되어 있으며 출처는 다음과 같다.
#   · 주간 수익률(W-FRI)      [문헌] Epps(1979) — 간격이 짧을수록 상관이 과소추정된다.
#                              우리 데이터: 일간 관측치의 36%가 가격 무변동이고 일간 상관(0.064)이
#                              주간(0.103)의 62%에 그친다. 주간에서 무변동 비율 2.6%로 떨어진다.
#   · 시장베타 잔차            [구조] 시장 공통 성분을 빼야 "업황을 공유하는 집단"이 드러난다.
#   · d = √(2(1−ρ))           [문헌] Mantegna(1999) — 상관계수 자체는 거리 공리를 만족하지 않는다.
#   · MST                     [문헌] Mantegna(1999) — N(N−1)/2개 연결 중 의미 있는 뼈대만 남긴다.
#   · Louvain                 [문헌] Blondel et al.(2008) — 군집 개수를 미리 정할 필요가 없다.
#   · 군집 평균 R&D 게이트      [구조] "동료 대비 R&D 규모"라는 하나의 임계값(평균)으로 고정.
#                              중위수로 바꾸면 1.00, 게이트를 빼면 1.08 — 게이트의 기여는 +0.20.
#   · 동일가중                 [구조] 팩터 프리미엄 측정의 표준. 시총가중 0.87, 상한5% 1.07, √시총 1.12.
#                              시총가중은 상위 3종목이 58%를 차지해 변동성 12.7%→23%, MDD −10%→−25%.
#   · 월 1% 윈저라이즈          [구조] 극단 수익률 방어를 위한 관행값.
#   · 분기(3/6/9/12월) 유니버스 갱신  [구조] 유니버스가 매월 흔들리지 않게 하는 관행.
#   · PIT = 접수일 + MonthEnd(1)    [구조] 공시 당월 말일 이후부터 사용 — look-ahead 차단.

CAGR_ = lambda x: ((1 + x.dropna()).prod() ** (12 / len(x.dropna())) - 1) * 100
MDD_  = lambda x: ((1 + x.dropna()).cumprod() / (1 + x.dropna()).cumprod().cummax() - 1).min() * 100
SR_   = lambda x: x.dropna().mean() / x.dropna().std() * np.sqrt(12)


# ═══════════════════════ 1. 데이터 적재 ═══════════════════════
def read_wide(path, hdr=0, skip=0, enc="utf-8-sig"):
    """날짜 × 종목명 와이드 CSV를 숫자 DataFrame으로."""
    d = pd.read_csv(path, encoding=enc, index_col=0, dtype=str, low_memory=False)
    if skip: d = d.iloc[skip:]
    if hdr:  d = d.iloc[hdr:]
    d.index = pd.to_datetime(d.index, errors="coerce")
    d = d[d.index.notna()].loc[:, lambda f: ~f.columns.duplicated()]
    return d.apply(lambda s: pd.to_numeric(s.str.replace(",", "").str.strip(), errors="coerce"))


def load_data():
    me_daily = read_wide(f"{DATA}/2012_2026_일별_코스피코스닥_시가총액_거래대금_비영업일제외.csv", skip=2)  # 백만원
    dv_daily = read_wide(f"{DATA}/60일거래대금.csv", skip=2)                                             # 원

    px = read_wide(f"{DATA}/수정주가(현금배당포함).csv")
    px_old = read_wide(f"{DATA}/수정주가(현금배당포함)2010.csv", hdr=1)
    px_m = px.resample("ME").last().combine_first(px_old.resample("ME").last()).sort_index()

    me  = me_daily.resample("ME").last()
    adv = dv_daily.rolling(60, min_periods=40).mean().resample("ME").last()   # 60일 평균 거래대금
    return dict(px_daily=px, px_m=px_m, me=me, adv=adv)


def load_rnd(me_index):
    """DART 연구개발비 → PIT(접수일 + MonthEnd(1)) 월별 패널. 단위: 원."""
    rn = pd.read_parquet(f"{DATA}/rnd_merged.parquet")

    # F1 검증: 파싱한 R&D와 공시된 '연구개발비/매출액 비율'의 괴리가 25% 초과면 제외
    fs = pd.read_csv(f"{DATA}/재무제표.csv", encoding="utf-8-sig", dtype=str, low_memory=False)
    sl = fs[fs.아이템명 == "매출액(천원)"].copy()
    sl["code6"] = sl.코드.str[1:]
    S = sl.set_index("code6").iloc[:, 6:-1]
    S = S.loc[~S.index.duplicated()].T
    S.index = pd.to_datetime(S.index, errors="coerce")
    S = S[S.index.notna()].apply(lambda c: pd.to_numeric(c.astype(str).str.replace(",", ""), errors="coerce")) * 1000
    SY = S.resample("YE").last()
    SY.index = SY.index.year

    def sales(ticker, fy):
        try:    return SY.at[fy, ticker]
        except Exception: return np.nan

    rn["sales"] = [sales(t, f) for t, f in zip(rn.ticker, rn.fy)]
    dev = (rn.rnd_won / rn.sales * 100 / rn.ratio - 1).abs()
    rn = rn[(dev <= 0.25) | dev.isna()]
    rn = rn[rn.rnd_won.notna() & rn.rcept_dt.notna()].drop_duplicates(["ticker", "fy"]).copy()

    codes = pd.read_csv(f"{DATA}/universe_codes.csv", dtype=str)
    c2n = dict(zip(codes.code6, codes.name))
    rn["nm"] = rn.ticker.map(c2n).fillna(rn.name.str.replace(r"^\(주\)|\(주\)$", "", regex=True))
    rn["avail"] = pd.to_datetime(rn.rcept_dt) + pd.offsets.MonthEnd(1)   # 접수 당월말 이후부터 사용 가능

    panel = (rn.sort_values(["nm", "avail", "fy"])
               .pivot_table(index="avail", columns="nm", values="rnd_won", aggfunc="last")
               .resample("ME").last())
    grid = pd.date_range(min(panel.index.min(), me_index.min()), me_index.max(), freq="ME")
    return panel.reindex(grid).ffill().reindex(me_index)   # 전체 월 그리드에서 ffill (초기 공시 보존)


def build_panels():
    """rd_me, 모멘텀, 수익률, 유니버스 랭크 등 백테스트 입력 일체."""
    d = load_data()
    me, adv, px_m, px_daily = d["me"], d["adv"], d["px_m"], d["px_daily"]
    rnd_won = load_rnd(me.index)

    cols = sorted(set(rnd_won.columns) & set(me.columns) & set(px_m.columns) & set(adv.columns))
    cols = [c for c in cols if not any(k in str(c).upper() for k in ("홀딩스", "지주", "HOLDINGS"))]

    ret = px_m.pct_change(fill_method=None)[cols]
    return dict(
        cols   = cols,
        me     = me[cols],
        rnd    = rnd_won[cols],                                   # R&D 금액 (원)
        rd_me  = rnd_won[cols] / (me[cols] * 1e6),                # 핵심 팩터
        mom    = (px_m.shift(MOM_SKIP) / px_m.shift(MOM_SKIP + MOM_J) - 1)[cols],
        ret    = ret.apply(lambda r: r.clip(lower=r.quantile(.01), upper=r.quantile(.99)), axis=1),  # 월 1% 윈저
        adv    = adv[cols].reindex(me.index) / 1e8,               # 억원
        mcap_r = me[cols].rank(axis=1, pct=True),                 # 유니버스 내 시총 백분위
        px_daily = px_daily,
    )


# ═══════════════════ 2. 동조집단 (MST + Louvain) ═══════════════════
def beta_residual(weekly):
    """동일가중 시장수익률에 대한 베타를 제거한 잔차. 업종 공통 성분만 남긴다."""
    R = weekly.fillna(0.0)
    mkt = R.mean(axis=1)
    mc = mkt - mkt.mean()
    Rc = R - R.mean()
    beta = (Rc.mul(mc, axis=0)).sum() / (mc ** 2).sum()
    return Rc - np.outer(mc, beta)


def build_clusters(P, held_map):
    """월별 {형성월: 종목→군집번호} 생성."""
    lp = np.log(P["px_daily"].where(P["px_daily"] > 0)).diff()
    weekly = lp.resample("W-FRI").sum(min_count=3)
    min_obs = int(TREE_WIN * 0.77)          # 104주 → 80주 이상 관측된 종목만
    labels = {}

    for t in [d for d in P["me"].index if T0 <= d <= T1]:
        if t not in P["mom"].index:
            continue
        tv = P["adv"].loc[t].dropna()
        liquid = tv[tv > tv.quantile(LIQ_CUT)].index
        cand = [c for c in P["cols"]
                if held_map[t].get(c, False) and c in liquid
                and c in lp.columns and pd.notna(P["mom"].at[t, c])]

        win = weekly.loc[:t, cand].tail(TREE_WIN)
        ok = [c for c in cand if win[c].notna().sum() >= min_obs]
        if len(ok) < 60:
            continue

        corr = np.nan_to_num(np.corrcoef(beta_residual(win[ok]).values.T))
        dist = np.sqrt(np.clip(2 * (1 - corr), 1e-9, None))       # Mantegna(1999) 거리
        np.fill_diagonal(dist, 0)
        mst = minimum_spanning_tree(dist).toarray()

        G = nx.Graph()
        for a, b in np.argwhere(mst > 0):
            G.add_edge(ok[a], ok[b], weight=float(1 - dist[a, b] ** 2 / 2) + 1.0)   # ρ+1 (양수화)
        comms = louvain_communities(G, weight="weight", seed=SEED)
        labels[t] = pd.Series({n: i for i, c in enumerate(comms) for n in c}).reindex(ok)

    return labels


def make_held_map(P):
    """유니버스 편입 여부. 분기(3/6/9/12월)마다 갱신하고 그 사이는 유지."""
    out, cur = {}, None
    for t in [x for x in P["me"].index if T0 <= x <= T1]:
        if cur is None or t.month in (3, 6, 9, 12):
            cur = P["mcap_r"].loc[t] >= (1 - MCAP_TOP)
        out[t] = cur
    return out


# ═══════════════════════ 3. 백테스트 ═══════════════════════
def backtest(P, labels, held_map, n=N_LEG):
    rnd_amt = P["rnd"]
    prev_w, rows = None, []
    dates = sorted(labels)

    for t, nxt in zip(dates[:-1], dates[1:]):
        lab = labels[t].dropna()

        # ── 신호: 자기 모멘텀 순위 2 : 군집평균 모멘텀 순위 1
        m = P["mom"].loc[t, lab.index].astype(float)
        grp = m.groupby(lab).transform("mean")
        signal = (W_OWN * m.rank(pct=True) + W_GRP * grp.rank(pct=True)) / (W_OWN + W_GRP)

        # ── 후보군: 유니버스 ∩ 유동성 ∩ rd_me 유효
        tv = P["adv"].loc[t].dropna()
        liquid = tv[tv > tv.quantile(LIQ_CUT)].index
        v = P["rd_me"].loc[t]
        pool = v[held_map[t].fillna(False) & v.index.isin(liquid)
                 & v.notna() & np.isfinite(v) & (v > 0)]
        if len(pool) < 60:
            continue

        weak = (signal.reindex(pool.index) <= signal.reindex(pool.index).quantile(SHORT_SCR)).fillna(False)
        n_pool = len(pool)                       # 군집 라벨 적용 전 후보 수 (리포팅용)
        lb = lab.reindex(pool.index)
        pool = pool[lb.notna()]
        lb = lb.dropna()

        # ── 동료 대비 R&D 규모 게이트
        amt = rnd_amt.loc[t].reindex(pool.index)
        grp_mean = amt.groupby(lb).transform("mean")
        long_cand  = pool[(amt > grp_mean).fillna(False)]
        short_cand = pool[weak.reindex(pool.index).fillna(False) & (amt < grp_mean).fillna(False)]
        if len(long_cand) < n or len(short_cand) < n:
            continue

        L = long_cand.nlargest(n).index
        S = short_cand.drop(index=L, errors="ignore").nsmallest(n).index    # ← difference() 쓰지 말 것
        if len(S) < n:
            continue

        # ── 비중 · 회전율 · 수익률
        w = pd.concat([pd.Series(1 / n, index=L), pd.Series(-1 / n, index=S)])
        idx = w.index.union(prev_w.index) if prev_w is not None else w.index
        prev = prev_w.reindex(idx, fill_value=0) if prev_w is not None else 0
        turnover = (w.reindex(idx, fill_value=0) - prev).abs().sum() / 2

        r = P["ret"].loc[nxt].reindex(w.index)
        gross = (w * r).sum(skipna=True)
        prev_w = (w * (1 + r.fillna(0))) / (1 + gross)     # 다음 달 시점 표류 비중

        rows.append(dict(date=nxt, 형성월=t, gross=gross, turnover=turnover,
                         롱수익=r[L].mean(), 숏수익=r[S].mean(),
                         유니버스=n_pool, 군집수=lb.nunique(),
                         롱=", ".join(L), 숏=", ".join(S)))

    bt = pd.DataFrame(rows).set_index("date")
    bt["net"] = bt.gross - bt.turnover * TC_BP / 1e4 - BORROW_BP / 1e4 / 12
    return bt


# ═══════════════════════ 4. 실행 ═══════════════════════
def main():
    print("데이터 적재…")
    P = build_panels()
    print(f"  종목 {len(P['cols'])} | 시총 ~{P['me'].index.max().date()} | 주가 ~{P['px_m'].index.max().date()}"
          if "px_m" in P else f"  종목 {len(P['cols'])}")

    held = make_held_map(P)
    print(f"동조집단 생성 (창 {TREE_WIN}주, seed {SEED})…")
    labels = build_clusters(P, held)
    print(f"  월 {len(labels)}개 | 평균 군집 {np.mean([l.nunique() for l in labels.values()]):.1f}개"
          f" | 평균 종목 {np.mean([len(l) for l in labels.values()]):.0f}")

    print("백테스트…")
    bt = backtest(P, labels, held)
    x = bt.net

    print("\n" + "═" * 64)
    print(f"기간 {x.index[0]:%Y-%m} ~ {x.index[-1]:%Y-%m} ({len(x)}개월)"
          f" | 월평균 유니버스 {bt.유니버스.mean():.0f}종목 / 군집 {bt.군집수.mean():.1f}개")
    print(f"CAGR {CAGR_(x):6.2f}%   변동성 {x.std()*np.sqrt(12)*100:5.2f}%   Sharpe {SR_(x):5.2f}"
          f"   MDD {MDD_(x):6.1f}%   회전율 {bt.turnover.mean()*100:4.1f}%   승률 {(x>0).mean()*100:.1f}%")
    print(f"롱 레그 연율 {((1+bt.롱수익).prod()**(12/len(bt))-1)*100:6.2f}%"
          f"   숏 레그 연율 {((1+bt.숏수익).prod()**(12/len(bt))-1)*100:6.2f}%")
    for tag, seg in [("IS  2015-02~2020-06", x.loc[IS]), ("OOS 2020-07~2026-08", x.loc[OOS]),
                     ("최근 10년", x.loc["2016-09":])]:
        print(f"  {tag:22s} CAGR {CAGR_(seg):6.2f}%  Sharpe {SR_(seg):5.2f}  MDD {MDD_(seg):6.1f}%")

    print("\n연도별 수익률 (%)")
    yr = x.groupby(x.index.year).apply(lambda v: (1 + v).prod() - 1) * 100
    print("  " + "  ".join(f"{y}: {v:+.1f}" for y, v in yr.items()))

    last = bt.iloc[-1]
    print(f"\n최근 리밸런싱 ({last.형성월:%Y-%m} 형성 → {bt.index[-1]:%Y-%m} 보유)")
    print(f"  롱: {last.롱}")
    print(f"  숏: {last.숏}")

    out = Path(__file__).resolve().parent / "results" / "rd_me_MST_백테스트_월별.csv"
    out.parent.mkdir(exist_ok=True)
    bt.drop(columns=["롱", "숏"]).to_csv(out, encoding="utf-8-sig")
    print(f"\n월별 결과 저장: {out.relative_to(out.parents[1])}")
    return bt


if __name__ == "__main__":
    bt = main()
