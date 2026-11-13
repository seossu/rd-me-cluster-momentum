import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
import statsmodels.api as sm
from pathlib import Path
DATA=Path(globals().get("__file__", Path.cwd()/"_")).resolve().parents[1]/"data"   # 저장소의 data/ (research/ 폴더에서 실행)
def wide(p,hdr=0,enc="utf-8-sig",skip=0):
    d=pd.read_csv(p,encoding=enc,index_col=0,dtype=str,low_memory=False)
    if skip: d=d.iloc[skip:]
    if hdr: d=d.iloc[hdr:]
    d.index=pd.to_datetime(d.index,errors="coerce"); d=d[d.index.notna()]
    d=d.loc[:,~d.columns.duplicated()]
    return d.apply(lambda s: pd.to_numeric(s.str.replace(",","").str.strip(),errors="coerce"))
MEd=wide(f"{DATA}/2012_2026_일별_코스피코스닥_시가총액_거래대금_비영업일제외.csv",skip=2)   # 백만원
DVd=wide(f"{DATA}/60일거래대금.csv",skip=2)                                             # 원 (일별)
me=MEd.resample("ME").last()
ADV=DVd.rolling(60,min_periods=40).mean().resample("ME").last()      # 60일 평균 거래대금(원)
pn=wide(f"{DATA}/수정주가(현금배당포함).csv"); po=wide(f"{DATA}/수정주가(현금배당포함)2010.csv",1)
ma=pn.resample("ME").last().combine_first(po.resample("ME").last()).sort_index()
RETr=ma.pct_change(fill_method=None); mom=ma.pct_change(12,fill_method=None).shift(1)
rn=pd.read_parquet(f"{DATA}/rnd_merged.parquet")
_fs=pd.read_csv(f"{DATA}/재무제표.csv",encoding="utf-8-sig",dtype=str,low_memory=False)
_s=_fs[_fs.아이템명=="매출액(천원)"].copy(); _s["code6"]=_s.코드.str[1:]
_S=_s.set_index("code6").iloc[:,6:-1]; _S=_S.loc[~_S.index.duplicated()].T
_S.index=pd.to_datetime(_S.index,errors="coerce"); _S=_S[_S.index.notna()]
_S=_S.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(",",""),errors="coerce"))*1000
_SY=_S.resample("YE").last(); _SY.index=_SY.index.year
def _sale(tk,fy):
    try: return _SY.at[fy,tk]
    except Exception: return np.nan
rn["sales"]=[_sale(t,f) for t,f in zip(rn.ticker,rn.fy)]
_dev=(rn.rnd_won/rn.sales*100/rn.ratio-1).abs()
rn=rn[(_dev<=0.25)|_dev.isna()].copy()
rn=rn[rn.rnd_won.notna()&rn.rcept_dt.notna()].drop_duplicates(["ticker","fy"])
uc=pd.read_csv(f"{DATA}/universe_codes.csv",dtype=str); C2N=dict(zip(uc.code6,uc.name))
rn["nm"]=rn.ticker.map(C2N).fillna(rn.name.str.replace(r"^\(주\)|\(주\)$","",regex=True))
rn["avail"]=pd.to_datetime(rn.rcept_dt)+pd.offsets.MonthEnd(1)
rw_=rn.sort_values(["nm","avail","fy"]).pivot_table(index="avail",columns="nm",values="rnd_won",aggfunc="last")
rw_=rw_.resample("ME").last()
_grid=pd.date_range(min(rw_.index.min(),me.index.min()),me.index.max(),freq="ME")
rw_=rw_.reindex(_grid).ffill().reindex(me.index)      # 전체 월 그리드에서 ffill 후 정렬
ALL=sorted(set(rw_.columns)&set(me.columns)&set(RETr.columns)&set(ADV.columns))
CM=[c for c in ALL if not any(k in str(c).upper() for k in ("홀딩스","지주","HOLDINGS"))]
RD=rw_[CM]/(me[CM]*1e6); TVv=(ADV[CM].reindex(me.index)/1e8)          # 억원
RW=RETr[CM].apply(lambda r: r.clip(lower=r.quantile(.01),upper=r.quantile(.99)),axis=1)
MErank=me[CM].rank(axis=1,pct=True)          # R&D 공시 기업 내 시총 백분위
MO=mom[CM]
T0,T1=pd.Timestamp("2015-01-31"),pd.Timestamp("2026-06-30")
IS,OOS=slice(None,"2020-06"),slice("2020-07",None)
f1=pd.read_csv(f"{DATA}/extra/factors.csv",encoding="utf-8-sig",index_col=0)
f1.index=pd.to_datetime(f1.index,errors="coerce"); f1=f1[f1.index.notna()]
f1=f1.apply(lambda s: pd.to_numeric(s.astype(str).str.replace(",","").str.strip(),errors="coerce"))
f2=pd.read_csv(f"{DATA}/extra/factor_model_index_0826.csv",encoding="utf-8-sig",index_col=0)
f2.index=pd.to_datetime(f2.index,errors="coerce"); f2=f2.resample("ME").last()
g1=f1.resample("ME").last(); F=pd.DataFrame(index=g1.index)
F["KOSPI"]=g1.KOSPI.pct_change(); F["HML"]=g1.HML.pct_change(); F["SMB"]=g1.SMB.pct_change()
F["MOM"]=f2.Mom.reindex(g1.index).pct_change(); F["RF"]=(1+g1.RF/100)**(1/12)-1
F["MKT"]=F.KOSPI-F.RF; F=F.dropna(subset=["KOSPI","RF"])
CAGR_=lambda x:((1+x.dropna()).prod()**(12/len(x.dropna()))-1)*100
MDD_=lambda x:((1+x.dropna()).cumprod()/(1+x.dropna()).cumprod().cummax()-1).min()*100
S_=lambda x:x.dropna().mean()/x.dropna().std()*np.sqrt(12)
print(f"로더2 OK | 종목 {len(CM)} | 시총 ~{me.index.max().date()} | 거래대금 ~{ADV.index.max().date()} | 주가 ~{ma.index.max().date()} | 팩터 ~{F.index.max().date()}")
