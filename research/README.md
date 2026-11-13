# research

전략을 확정하기까지의 분석·검증 스크립트입니다. **이 폴더 안에서** 실행합니다. 스크립트가 `exec(open("loader2.py").read())`처럼 서로를 이어서 불러오기 때문입니다.

| 파일 | 내용 |
|---|---|
| `loader2.py` | 공통 데이터 적재 (모든 스크립트의 출발점) |
| `fixsel.py` | 숏 선택 버그(`Index.difference()` 가나다순 정렬) 수정 전후 비교, 수정 엔진 정의 |
| `j11chk.py`, `shape.py` | 모멘텀 창(12-1 vs 11-2) 비교, seed 0~2 |
| `startdt.py`, `ten.py` | 시작 시점 민감도, 10년 롤링 구간 |
| `sizecrit.py`, `decomp.py` | "시가총액 전략 아닌가" 반박 검증, 월간 갱신 효과 분해 |
| `FINAL.py` | 최종 사양 성과, 팩터 회귀 |
| `deliver.py` → `files.py`, `md.py` | 결과 엑셀·CSV·히트맵(`results/`)과 요약 문서(`docs/`) 생성 |
| `plot_nav.py` | README용 누적 NAV vs KOSPI 그래프 |
| `사전등록_*.txt` | 결과를 보기 **전에** 작성한 검증 계획 (과적합 방지) |
| `기록_모멘텀창_변경.txt` | 모멘텀 창 변경이 사후 선택이었음을 밝힌 의사결정 기록 |
