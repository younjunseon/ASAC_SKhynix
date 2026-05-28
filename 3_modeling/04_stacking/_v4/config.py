"""SqueezeV4Config — 모든 파라미터를 한 dataclass에 집약.

노트북에서:
    from _v4.config import SqueezeV4Config
    cfg = SqueezeV4Config(random_trials=2000, top_refit=30,
                          shap_caches=["shap_cache/02_reg_single__lgbm__hp__002"], ...)
형태로 사용. v3의 모든 옵션 + v2의 SHAP 옵션을 die-level로 재구현 (shap_caches / shap_top_k /
shap_mode / shap_prefix_with_tag).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional


def _default_project_root() -> Path:
    """이 파일 위치(_v3/config.py)에서 프로젝트 루트를 자동 추론.

    경로: 3_modeling/04_stacking/_v3/config.py → parents[3] = 프로젝트 루트.
    노트북 cwd가 어디든 동일하게 동작.
    """
    return Path(__file__).resolve().parents[3]


# 1차 실험에서 "확실히 강한" base 모델 4개 — make_seed_subsets에서 강제 후보로 추가.
# 4개 모두 discover_models 결과에 존재할 때만 활용.
KNOWN_STRONG_SUBSET: tuple[str, ...] = (
    "03_two_stage__default__clf__lgbm__hp__002",
    "01_zit__zit_only__hp__002",
    "02_reg_single__et__hp__001",
    "02_reg_single__catboost__raw__001",
)


@dataclass
class SqueezeV4Config:
    # ─── 모델 풀 필터링 ─────────────────────────────────────────────
    oof_rmse_cutoff: float = 0.00555
    """후보 모델의 oof_rmse 상한. 이걸 넘는 모델은 base에서 제외."""

    no_clf: bool = False
    """True면 category='clf' 모델 제외 (분류 단독 출력)."""

    include_combined: bool = False
    """03_two_stage/default/combined/* 포함 여부. 기본 False (die-level CSV가 없음)."""

    include_ts_reg: bool = False
    """03_two_stage/default/reg/* 포함 여부. 기본 False (Stage 2만 단독 — 보통 미사용)."""

    # ─── 서브셋 크기 ────────────────────────────────────────────────
    min_subset_size: int = 2
    """메타에 들어가는 base 모델의 최소 개수."""

    max_subset_size: int = 18
    """메타에 들어가는 base 모델의 최대 개수."""

    # ─── 탐색 단계 1: random search ─────────────────────────────────
    random_trials: int = 6000
    """랜덤 서브셋 평가 개수. 0이면 random search 생략."""

    # ─── 탐색 단계 2: local improve (add/drop/swap) ────────────────
    local_seeds: int = 25
    """local_improve를 돌릴 seed subset의 개수 (상위 N개에서 출발)."""

    local_steps: int = 30
    """seed 1개당 add/drop/swap 시도 step 수."""

    local_candidate_limit: int = 35
    """add 후보로 둘 base 모델의 최대 개수 (정렬 상위 N개)."""

    # ─── 탐색 단계 3: refit (정밀 메타) ─────────────────────────────
    top_refit: int = 50
    """refit 단계에서 ENet + ENetPositive를 적용할 unique subset 개수."""

    combo_refit: int = 20
    """refit 단계에서 Combo (Bag+ENet+NNLS)도 적용할 상위 subset 개수."""

    # ─── 탐색 단계 4: Optuna (옵션) ─────────────────────────────────
    optuna_trials: int = 0
    """0이면 Optuna 생략. >0이면 method/alpha/iso_weight/zero_tau를 베이지안 탐색."""

    # ─── 선택 기준 ──────────────────────────────────────────────────
    select_by: Literal["oof", "val", "meta_cv_oof"] = "oof"
    """탐색 / 최종 선정 기준 metric. v2 기본값 'oof' 유지."""

    val_gap_penalty: float = 0.0
    """select_by='oof'일 때만 적용. score = oof + λ·max(0, val-oof). λ=0이면 순수 OOF.
    λ를 크게 하면 val-oof 갭이 큰 (과적합 의심) 후보를 페널티."""

    allow_val_fit: bool = False
    """True면 val label을 직접 fit에 쓰는 후처리도 시도 (강한 peek-bias 위험).
    v2와 마찬가지로 v3에서도 일단 미구현 — True 줘도 무시되고 경고만."""

    # ─── die→unit 집계 (postprocess.tune_and_apply 인자) ───────────
    agg_methods: tuple[str, ...] = (
        "mean", "median", "max", "min",
        "trimmed_mean", "weighted",
        "Q25", "Q75",
    )
    """집계 후보. find_best_aggregation이 train OOF로 1등을 고름."""

    baseline_agg: str = "mean"
    """val 개선이 없을 때 fallback할 baseline 집계 방식."""

    position_method: Literal["optuna", "slsqp"] = "optuna"
    """weighted 집계의 position 가중치 최적화 방법."""

    position_optuna_n_trials: int = 50
    """position_method='optuna'일 때 trial 수."""

    use_pi_threshold: bool = False
    """True면 die-level π를 unit 평균 후 threshold gating. ZITboost meta가 아니면 보통 False.
    base 모델의 die-level π 정보를 stacking에 활용하지 않으므로 v3 기본값 False."""

    use_iso: bool = True
    """True면 메타 raw 예측에 IsotonicRegression 후처리 (step function → 학습 분포에 맞춰 보정).
    **False면 raw 메타 출력을 그대로 사용**(non-negative clip만 적용) — 연속형 예측이 필요할 때.
    iso는 학습 데이터 분포의 step function이라 같은 값이 반복적으로 나올 수 있다."""

    zero_clip_range: tuple[float, float] = (0.001, 0.015)
    zero_clip_step: float = 0.001
    """zero_clip 탐색 그리드 (작은 양수 → 0)."""

    zero_clip_log_space: bool = False

    # ─── 메타 학습 하이퍼파라미터 ─────────────────────────────────
    ridge_alpha_grid: tuple[float, ...] = tuple(10 ** -e for e in range(2, 10))
    """ridge 메타의 alpha 그리드. 1e-2 ~ 1e-9 (8개) — np.logspace(-9,-2,8)과 동일."""

    enet_l1_ratio_grid: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
    enet_alpha_n: int = 30
    """ElasticNetCV의 alpha 개수 (np.logspace(-6, 0, n))."""

    enet_max_iter: int = 20000
    enet_cv_folds: int = 5

    combo_seeds: tuple[int, ...] = (42, 123, 456, 789, 2024)
    """Combo의 ENet bagging seeds."""

    # ─── CV (meta_cv_oof 계산) ──────────────────────────────────────
    cv_n_splits: int = 5
    """meta_cv_oof_rmse 계산 시 GroupKFold split 수."""

    cv_ridge_alpha: float = 1e-5
    """meta_cv_oof 내부에서 쓰는 ridge alpha (fold마다 fit)."""

    # ─── 실행 제어 ──────────────────────────────────────────────────
    seed: Optional[int] = None
    """RNG seed. None이면 매 실행마다 SeedSequence().entropy로 자동 생성."""

    deadline: Optional[str] = None
    """예: '2026-05-16 08:00'. 이 시각 이후에는 새 trial 중단 → save로 진행."""

    deadline_margin_minutes: float = 10.0
    """deadline에서 이만큼 여유를 두고 일찍 중단 (저장 시간 확보)."""

    # ─── SHAP X-stacking (v4 신규) ─────────────────────────────────
    shap_caches: tuple[str, ...] = ()
    """추가 die-level SHAP 캐시 폴더 리스트 (build_shap_features.py가 만든 폴더).
    예: ('shap_cache/02_reg_single__lgbm__hp__002', 'shap_cache/01_zit__zit_only__hp__002__mu').
    각 폴더는 die_shap.npz를 가지고 있어야 하며, **run_wf_xy 키를 포함**한 npz여야 한다
    (v4 시점부터 build_shap_features.py가 저장. 구버전 캐시는 재생성 필요)."""

    shap_top_k: int = 0
    """각 캐시 안에서 mean|SHAP| 상위 K개 feature만 사용. 0이면 전체.
    캐시별 meta.json의 importance_top50 순서를 따른다 (build_shap_features.py가 박제)."""

    shap_mode: Literal["always_include", "searchable"] = "always_include"
    """SHAP 컬럼 사용 방식.
    - 'always_include': subset search 후보에는 들어가지 않고, 메타 학습 시 **항상 입력**으로 포함 (v2 기본).
    - 'searchable':     base와 동등하게 subset search 후보 풀에 등록. ridge/L1이 자동 가중치 학습.
    """

    shap_prefix_with_tag: bool = True
    """True면 SHAP 컬럼 이름 앞에 캐시 태그 prefix (예: '02_reg_single__lgbm__hp__002::X146') —
    여러 캐시를 합칠 때 컬럼명 충돌 방지."""

    shap_cache_root: Optional[str] = None
    """shap_caches 항목이 상대경로일 때의 기준 폴더. None이면 04_stacking/ 폴더 기준."""

    # ─── 경로 ───────────────────────────────────────────────────────
    project_root: Path = field(default_factory=lambda: _default_project_root())
    """프로젝트 루트. 기본값: 이 파일(_v4/config.py) 위치에서 부모 디렉토리 3단계 위.
    (3_modeling/04_stacking/_v4/config.py → 3_modeling/04_stacking/_v4 → 3_modeling/04_stacking → 3_modeling → 프로젝트 루트)
    노트북에서 cwd가 다르더라도 자동으로 올바른 경로를 잡는다. 명시적으로 덮어쓰려면 인자로 전달."""

    output_subdir: str = "04_stacking/results_extreme_v4"
    """결과 저장 위치 (project_root / '4_output' / output_subdir / 'run_{ts}')."""

    # ─── 노트북 진단용 출력 ─────────────────────────────────────────
    verbose: bool = True
    """진행 로그 출력 여부."""

    log_every: int = 500
    """random search 중 log_every trial마다 best 진행상황 출력."""

    # ─── 강제 포함 subset ───────────────────────────────────────────
    known_strong_subset: tuple[str, ...] = KNOWN_STRONG_SUBSET

    # ─── 후처리 ─────────────────────────────────────────────────────
    def __post_init__(self):
        # 경로 정규화
        if not isinstance(self.project_root, Path):
            self.project_root = Path(self.project_root)
        # seed 자동 생성
        if self.seed is None:
            import numpy as np
            self.seed = int(np.random.SeedSequence().entropy & 0xFFFFFFFF)

    # ─── 편의 ───────────────────────────────────────────────────────
    @property
    def output_dir(self) -> Path:
        """4_output 루트."""
        return self.project_root / "4_output"

    @property
    def result_dir(self) -> Path:
        """run 들이 저장될 부모 디렉토리. mkdir은 io.save_outputs에서 처리."""
        return self.output_dir / self.output_subdir

    def to_dict(self) -> dict:
        d = asdict(self)
        d["project_root"] = str(self.project_root)
        d["output_dir"] = str(self.output_dir)
        d["result_dir"] = str(self.result_dir)
        return d


def parse_deadline(text: Optional[str]) -> Optional[datetime]:
    """SqueezeV4Config.deadline 문자열 → datetime. None이면 None."""
    if not text:
        return None
    text = text.strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%m-%d %H:%M"):
        try:
            dt = datetime.strptime(text, fmt)
            if fmt == "%m-%d %H:%M":
                dt = dt.replace(year=datetime.now().year)
            return dt
        except ValueError:
            continue
    raise ValueError(f"Unsupported deadline format: {text!r}")


def seconds_left(deadline: Optional[datetime]) -> float:
    if deadline is None:
        return float("inf")
    return (deadline - datetime.now()).total_seconds()


def should_stop(deadline: Optional[datetime], margin_minutes: float) -> bool:
    return seconds_left(deadline) <= margin_minutes * 60.0
