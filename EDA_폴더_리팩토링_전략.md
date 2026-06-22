# EDA 폴더 리팩토링 전략 (`1_eda` / `1_eda_` 통합 + 내부 재구조화)

> 작성 2026-06-22 · 스냅샷 커밋 `38f53a4` · **본 문서는 모든 의사결정 확정 반영본(v2)**

## 0. 목적

중복된 두 EDA 폴더(`1_eda`, `1_eda_`)를 **하나(`1_eda`)로 통합**하고, 내부를 전/후/웨이퍼맵
3관심사로 재구조화한다. 핵심: (1) 중복 제거(~149MB), (2) 폐기된 옛 전처리 사본 정리,
(3) 전/후 EDA 폴더 대칭화, (4) git 추적 범위 정상화.

---

## 1. 현황 진단 (검증 완료)

### 1-1. 두 폴더는 거의 완전 중복
`eda.ipynb`·`modules/*.py` 22개·`config`·**PNG 2,317장(체크섬 동일)**·report html 2종이 byte 동일.
실제 차이는 **6개 파일뿐이며 전부 `1_eda_`가 최신** → **`1_eda_`를 정본(base)으로 채택.**

### 1-2. 차이 6파일
| 파일 | `1_eda`(구) | `1_eda_`(신) |
|------|------|------|
| `aefer_pipeline.py` | 없음 | 신규(06-12) |
| `report/make_html_after.py`, `eda_clean_report.html` | 없음 | 신규(06-13) |
| `전처리 후 eda/eda_after.ipynb` | 04-15 구 | **06-13 최신**(2_preprocessing 참조) |
| `전처리 후 eda/scaling.py` | 없음 | 신규(06-12) |
| `전처리 후 eda/cleaning·preprocess·scale_v2.py` | 04-15 스테일 사본 | 삭제됨 |

### 1-3. 핵심 사실 (재구조화 근거)
- **깊이 독립화 완료**(다른 세션): 두 노트북 모두 cwd에서 상위로 `setup.py`+`utils/`를 자동탐색 → **어느 깊이에 둬도 동작**.
- **wafer 툴킷은 독립 도구**: `aefer_pipeline.py`(웨이퍼맵 시각화 클래스 `AeferPipeline`)와 `scaling.py`(그 전용 스케일링 헬퍼)는 **어느 노트북도 import 안 함**. `scaling.py`는 오직 `aefer_pipeline.py`만 사용. → 둘은 한 쌍, 별도 분리 대상.
- **git 추적**: 현재 `1_eda`는 `eda.ipynb`+`modules/*.py` 23개만 추적. `1_eda_`는 통째 미추적.
- **잠복 버그**: `.gitignore`의 `!1_eda/eda_style.mplstyle`가 실제 경로(`config/eda_style.mplstyle`)와 불일치 → 스타일 파일 미추적 상태.
- 외부에 `1_eda_` 하드코딩 경로 **없음** → 이동/스왑 안전.

---

## 2. 확정된 의사결정

| 항목 | 결정 |
|------|------|
| base 내용 | `1_eda_` (최신) |
| 최종 폴더명 | `1_eda` (git 경로 유지) |
| 전/후 네이밍 | `01_raw_eda` / `02_processed_eda` |
| 옛 사본 3개(cleaning/preprocess/scale_v2) | **삭제** |
| `aefer_pipeline.py` | **`wafer_pipeline.py`로 rename** |
| wafer 툴킷 위치 | **`wafer_map/`로 분리** (.py 2개 + 이미지) |
| 웨이퍼맵 이미지 추적 | **`median_auto_robust`(1087장)만 추적** |
| `★분류완료` | **git 미추적**(디스크엔 유지). EXCLUDE_COLS 근거는 CLAUDE.md + median_auto_robust로 충분 |
| 작업 방식 | `1_eda_`(미추적 샌드박스)에서 정리 후 `1_eda`로 스왑 |

---

## 3. 최종 폴더 구조

```
1_eda/                                   ← 최종(=정리된 1_eda_를 swap)
├── modules/                    [git: *.py]   공유 EDA 모듈 22개 + __init__
│   └── modules.zip             [ignore]      빌드 산출물
├── config/
│   └── eda_style.mplstyle      [git]         (whitelist 경로 버그 수정)
├── 01_raw_eda/
│   └── eda.ipynb               [git]         전처리 전 EDA
├── 02_processed_eda/
│   └── eda_after.ipynb         [ignore 5.6M] 전처리 후 EDA (스테일 3개 삭제됨)
├── wafer_map/                              ← 웨이퍼맵 시각화 툴킷 한 묶음
│   ├── wafer_pipeline.py       [git]         (← aefer_pipeline rename)
│   ├── scaling.py              [git]         (← 전처리 후 eda/에서 이동, import 1줄 수정)
│   └── images/
│       ├── median_auto_robust/ [git] 1087장  피처 웨이퍼맵 갤러리
│       └── ★분류완료/          [ignore 23M]  수동 분류(디스크 유지)
├── report/                                 ← 공유 (전/후 리포트 혼재)
│   ├── build_report.py         [git]
│   ├── make_html_after.py      [git]         (입력파일명 불일치 점검)
│   ├── *.html (3종)            [ignore]
│   └── images/ (72장)          [ignore]
└── viz_images/ (71장)          [ignore]      (루트 공유 자산 기본값)
```

---

## 4. 실행 단계

1. **`1_eda_` 샌드박스에서 재구조화** (git 무관, 자유)
   - `01_raw_eda/` 생성 → `eda.ipynb` 이동
   - `02_processed_eda/` 생성 → `eda_after.ipynb` 이동, 스테일 3개(cleaning/preprocess/scale_v2) 삭제
   - `wafer_map/` 생성 → `aefer_pipeline.py`→`wafer_pipeline.py` rename 후 이동, `전처리 후 eda/scaling.py` 이동(import 수정), `wafer_map_image/` → `wafer_map/images/` 이동
   - 빈 `전처리 후 eda/` 제거
2. **내부 정리**: `wafer_pipeline.py`의 `from scaling import`(같은 디렉토리로) 수정, `make_html_after.py` 입력파일명(`eda_clean.ipynb` vs 실제) 점검
3. **검증**: 노트북 setup 셀 깊이 자동탐색 동작 확인, `wafer_pipeline.py` import 동작 확인
4. **스왑**: 옛 `1_eda` 삭제 → `1_eda_` → `1_eda` rename (중복 149MB 회수)
5. **`.gitignore` 갱신**: 새 구조 whitelist (modules/*.py, config/eda_style.mplstyle, 01_raw_eda/eda.ipynb, wafer_map/*.py, wafer_map/images/median_auto_robust/**, report/*.py) + 나머지 ignore
6. **검증 후 리팩토링 완료 커밋**

---

## 5. 안전장치
- 스냅샷 커밋 `38f53a4` (작업 직전).
- 재구조화는 미추적 `1_eda_`에서 수행 → 옛 `1_eda` 무손상, 스왑 전까지 언제든 복구.
- `1_eda_`(=정본)는 스왑 시점까지 디스크 유지.
