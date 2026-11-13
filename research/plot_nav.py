"""README용 그래프: 전략 누적 NAV vs KOSPI (results/rd_me_MST_nav_vs_kospi.png)
실행: research/ 폴더에서  python3 plot_nav.py
"""
from pathlib import Path
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "AppleGothic"; matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
bt = pd.read_csv(ROOT / "results/rd_me_MST_백테스트_월별.csv", index_col=0, parse_dates=True)
k = pd.read_csv(ROOT / "data/extra/코스피최근.csv", index_col=0, skiprows=[1], thousands=",")
k.index = pd.to_datetime(k.index); k = k.iloc[:, 0].astype(float).resample("ME").last()

nav = (1 + bt.net).cumprod()
start = bt.index[0] - pd.offsets.MonthEnd(1)
nav = pd.concat([pd.Series({start: 1.0}), nav])
kos = (k / k.loc[start]).reindex(nav.index)
dd = nav / nav.cummax() - 1; kdd = kos / kos.cummax() - 1

fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
a1.plot(nav.index, nav, color="#1f5fa8", lw=2, label="전략 (롱숏, 비용 차감 후)")
a1.plot(kos.index, kos, color="#999", lw=1.5, label="KOSPI")
a1.set_ylabel("누적 NAV (2015-01 = 1)"); a1.grid(alpha=.3); a1.legend(loc="upper left")
a1.set_title("R&D/ME × 동조집단 모멘텀 롱숏 — 누적 성과 (2015-02 ~ 2026-08)", fontweight="bold")
a2.fill_between(dd.index, dd * 100, 0, color="#1f5fa8", alpha=.5, label="전략")
a2.plot(kdd.index, kdd * 100, color="#999", lw=1, label="KOSPI")
a2.set_ylabel("낙폭 (%)"); a2.grid(alpha=.3); a2.legend(loc="lower left")
plt.tight_layout()
fig.savefig(ROOT / "results/rd_me_MST_nav_vs_kospi.png", dpi=150, bbox_inches="tight")
print("저장 완료", nav.iloc[-1], kos.iloc[-1], dd.min(), kdd.min())
