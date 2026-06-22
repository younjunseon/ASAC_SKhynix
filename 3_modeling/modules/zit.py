"""
ZITboost — Zero-Inflated Tweedie + LightGBM (EM 알고리즘) 회귀 모델.

우리 target(health)은 semicontinuous: 70.8%가 정확히 0이고, 나머지는 양의 연속값.
이를 단일 모델로 다루기 위해 다음 mixture를 가정한다:
  Y_i = 0                              확률 π(x_i)      ← structural zero (정상 제품)
  Y_i ~ Tweedie(μ_i, φ_i, ζ)          확률 1 - π(x_i)  ← Tweedie 상태 (0일 수도, 양수일 수도)

  최종 예측 E[Y] = (1 - π) × μ

세 개의 LightGBM을 EM으로 번갈아 학습한다:
  - lgb_pi  : P(structural zero)  — cross_entropy (soft label 0~1을 직접 학습)
  - lgb_mu  : Tweedie mean μ      — Tweedie deviance, weight = (1-Π)/φ
  - lgb_phi : dispersion φ        — gamma regression, weight = (1-Π)

Two-Stage(분류→회귀)와 달리 분류·회귀가 joint로 학습돼 분류가 약해도 μ가 보완한다.

이 모듈은 네 클래스를 제공한다 (φ 추정방식 × bag 제약의 2×2):
  - ZITboostRegressor      : 기본('우리버전'). φ를 Pearson 잔차(=(y-μ)²/μ^ζ)로 추정 (가볍고 빠른 근사).
  - ZITboostEQLRegressor   : '논문충실'. Gu 2024(arXiv:2405.14990) 방향.
                             φ를 extended quasi-likelihood/saddlepoint(=Tweedie unit deviance)로 추정하고
                             zero-truncated Tweedie로 초기화. ZITboostRegressor와 다른 점은 이 2곳뿐
                             (나머지 E/M-step·predict는 동일해서 상속).
  - BagZITboostRegressor   : ZITboost('우리버전' φ) + unit 제약(bag constraint) 변형 (논문 외).
                             fit(X, y, unit_id), die→unit 집계는 SUM(predict_unit).
  - BagZITEQLRegressor     : BagZIT('논문충실' φ) — bag 제약 + EQL φ M-step. _m_step만 ZITboostEQL과 동일.

ζ profile likelihood(Algorithm 2)용 score_loglik은 부모 ZITboostRegressor에 있어 네 클래스가 공유한다.

사용법:
    from modules.zit import ZITboostRegressor, ZITboostEQLRegressor

    model = ZITboostEQLRegressor(zeta=1.5, n_em_iters=10)   # 논문 충실 버전
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pi, mu, phi = model.predict_components(X_test)
"""

import numpy as np
import lightgbm as lgb
from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator, RegressorMixin

from utils.config import SEED


# --- Tweedie 분포 관련 수학 헬퍼 ---

def _tweedie_p0(mu, phi, zeta):
    """Tweedie 상태에서 Y가 정확히 0일 확률 P(Y=0 | Tweedie) = exp(-λ).

    Tweedie를 compound Poisson-Gamma로 보면 Poisson rate λ = μ^(2-ζ) / ((2-ζ)φ).
    "이벤트가 0번 발생" = Y=0 이므로 P(Y=0) = exp(-λ).
    ζ ∈ (1, 2) 에서만 유효 (ζ=1: Poisson, ζ=2: Gamma — 양 끝은 이 식이 안 됨).
    """
    mu = np.maximum(mu, 1e-10)     # log/거듭제곱에서 0·음수 방지
    phi = np.maximum(phi, 1e-10)
    lam = np.power(mu, 2 - zeta) / ((2 - zeta) * phi)
    return np.exp(-lam)


def _estimate_phi(y_pos, mu_pos, zeta):
    """Y>0 샘플들로부터 dispersion φ를 moment estimator로 한 번 추정 (EM 초기값용 스칼라).

    Tweedie의 분산 구조 Var(Y) ≈ φ·μ^ζ 를 뒤집어 φ ≈ Var(Y) / μ^ζ.
    """
    if len(y_pos) < 2:
        return 1.0   # 양수 샘플이 거의 없으면 그냥 1
    mu_mean = np.maximum(np.mean(mu_pos), 1e-10)
    var_y = np.var(y_pos, ddof=1)
    phi = var_y / np.maximum(np.power(mu_mean, zeta), 1e-10)
    return np.clip(phi, 1e-6, 1e6)   # 극단값 클립 (수치 안정성)


def _tweedie_unit_deviance(y, mu, zeta):
    """Tweedie unit deviance D_ζ(y, μ), 1 < ζ < 2 에서 유효.

    Gu 2024의 extended quasi-likelihood(saddlepoint 근사)에서 φ를 추정할 때 쓰는 통계량.
    표준형(Dunn & Smyth):
      D_ζ(y,μ) = 2[ y^(2-ζ)/((1-ζ)(2-ζ)) − y·μ^(1-ζ)/(1-ζ) + μ^(2-ζ)/(2-ζ) ]   (y>0)
               = 2·μ^(2-ζ)/(2-ζ)                                                (y=0; 앞 두 항 소거)
    saddlepoint 하에서 E[D_ζ] ≈ φ 이므로, D_ζ를 타깃으로 gamma 회귀를 돌리면 φ(x)를 추정한다.
    (프로젝트의 BagZITEQLRegressor와 동일한 수식·클립.)
    """
    y = np.asarray(y, dtype=np.float64)
    mu = np.maximum(np.asarray(mu, dtype=np.float64), 1e-10)
    y = np.maximum(y, 0.0)
    p = float(zeta)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        y_term = np.where(
            y > 0,
            np.power(y, 2.0 - p) / ((1.0 - p) * (2.0 - p)),
            0.0,
        )                                                   # y=0이면 0 (2-ζ>0이라 0^(2-ζ)=0)
        cross_term = -y * np.power(mu, 1.0 - p) / (1.0 - p)  # y=0이면 자동 0
        mu_term = np.power(mu, 2.0 - p) / (2.0 - p)
        dev = 2.0 * (y_term + cross_term + mu_term)

    dev = np.nan_to_num(dev, nan=1e6, posinf=1e6, neginf=1e-8)
    return np.clip(dev, 1e-8, 1e6)


def _zitweedie_loglik(y, pi, mu, phi, zeta, w=1.0):
    """ZI-Tweedie 로그우도 합 — Gu 2024가 채택한 EQL/saddlepoint **근사** 로그우도.

    주의: exact Tweedie 밀도가 아니라 saddlepoint 근사다(논문도 이 EQL을 씀 → 접근은 일치).
    아래 수식은 arXiv HTML에서 전사한 것으로, PDF 원문과 한 줄씩 대조한 'verbatim'은 아니다.

    y=0 :  log( π + (1-π)·exp(−w·μ^(2-ζ) / (φ(2-ζ))) )
    y>0 :  log(1-π) − ½·(w/φ)·D_ζ(y;μ) − ½·log(2π·φ·y^ζ / w)

    Algorithm 2(ζ를 profile likelihood로 선택)에서 후보 ζ들을 비교하는 목적함수.
    w(exposure)는 우리 데이터에 노출량 개념이 없어 1.0(스칼라) 기본 — 논문 일반식의 w_i=1 특수화.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    pi = np.clip(np.asarray(pi, dtype=np.float64), 1e-8, 1 - 1e-8)
    mu = np.maximum(np.asarray(mu, dtype=np.float64), 1e-10)
    phi = np.maximum(np.asarray(phi, dtype=np.float64), 1e-10)

    is_zero = (y == 0)
    is_pos = ~is_zero
    ll = np.zeros(len(y), dtype=np.float64)

    # y=0: structural zero 또는 Tweedie 상태에서 0이 나온 경우의 혼합 확률 (해당 행만 계산)
    if is_zero.any():
        piz, muz, phiz = pi[is_zero], mu[is_zero], phi[is_zero]
        lam = w * np.power(muz, 2 - zeta) / (phiz * (2 - zeta))   # Tweedie 상태에서 0일 rate
        ll[is_zero] = np.log(np.maximum(piz + (1 - piz) * np.exp(-lam), 1e-300))

    # y>0: Tweedie 상태(1-π) × saddlepoint Tweedie 밀도 (해당 행만 계산 → y=0의 log(0) 회피)
    if is_pos.any():
        yp, pip, mup, phip = y[is_pos], pi[is_pos], mu[is_pos], phi[is_pos]
        devp = _tweedie_unit_deviance(yp, mup, zeta)
        ll[is_pos] = (np.log(np.maximum(1 - pip, 1e-300))
                      - 0.5 * (w / phip) * devp
                      - 0.5 * np.log(2 * np.pi * phip * np.power(yp, zeta) / w))

    return float(ll.sum())


# --- ZITboostRegressor ---

class ZITboostRegressor(BaseEstimator, RegressorMixin):
    """Zero-Inflated Tweedie Boosting via EM + LightGBM.

    Parameters
    ----------
    zeta : float
        Tweedie power (1 < ζ < 2). 1.0→Poisson, 2.0→Gamma.
    n_em_iters : int
        EM 알고리즘 반복 횟수.

    mu_* : μ 모델 (핵심 회귀) HP — 9개
    pi_* : π 모델 (zero 확률 분류) HP — 5개
    phi_* : φ 모델 (분산) HP — 5개
    """

    def __init__(
        self,
        # ZIT 전용 파라미터
        zeta=1.5,
        n_em_iters=10,
        em_tol=1e-7,
        # μ 모델 (Tweedie mean — 가장 중요, HP 9개)
        mu_n_estimators=500,
        mu_learning_rate=0.05,
        mu_num_leaves=31,
        mu_max_depth=6,
        mu_min_child_samples=20,
        mu_subsample=0.8,
        mu_colsample_bytree=0.8,
        mu_reg_alpha=1e-3,
        mu_reg_lambda=1e-1,
        # π 모델 (zero 확률 — HP 5개)
        pi_n_estimators=200,
        pi_learning_rate=0.05,
        pi_num_leaves=31,
        pi_max_depth=6,
        pi_min_child_samples=20,
        # φ 모델 (dispersion — HP 5개)
        phi_n_estimators=200,
        phi_learning_rate=0.05,
        phi_num_leaves=31,
        phi_max_depth=6,
        phi_min_child_samples=20,
        # 공통/환경
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
        device="cpu",
    ):
        # sklearn 규약: __init__에선 인자를 그대로 attribute에 담기만 한다 (검증/연산 X)
        self.zeta = zeta
        self.n_em_iters = n_em_iters
        self.em_tol = em_tol
        # μ
        self.mu_n_estimators = mu_n_estimators
        self.mu_learning_rate = mu_learning_rate
        self.mu_num_leaves = mu_num_leaves
        self.mu_max_depth = mu_max_depth
        self.mu_min_child_samples = mu_min_child_samples
        self.mu_subsample = mu_subsample
        self.mu_colsample_bytree = mu_colsample_bytree
        self.mu_reg_alpha = mu_reg_alpha
        self.mu_reg_lambda = mu_reg_lambda
        # π
        self.pi_n_estimators = pi_n_estimators
        self.pi_learning_rate = pi_learning_rate
        self.pi_num_leaves = pi_num_leaves
        self.pi_max_depth = pi_max_depth
        self.pi_min_child_samples = pi_min_child_samples
        # φ
        self.phi_n_estimators = phi_n_estimators
        self.phi_learning_rate = phi_learning_rate
        self.phi_num_leaves = phi_num_leaves
        self.phi_max_depth = phi_max_depth
        self.phi_min_child_samples = phi_min_child_samples
        # 공통
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.device = device

    # --- 세 LightGBM의 파라미터 dict 구성 ---

    def _mu_params(self):
        """μ 모델용 LightGBM 파라미터 (objective=tweedie)."""
        return dict(
            objective="tweedie",
            tweedie_variance_power=self.zeta,   # Tweedie power ζ
            n_estimators=self.mu_n_estimators,
            learning_rate=self.mu_learning_rate,
            num_leaves=self.mu_num_leaves,
            max_depth=self.mu_max_depth,
            min_child_samples=self.mu_min_child_samples,
            subsample=self.mu_subsample,
            subsample_freq=1,  # subsample을 켜려면 freq>0이어야 함 (LGBM 기본 0이면 subsample 무시됨)
            colsample_bytree=self.mu_colsample_bytree,
            reg_alpha=self.mu_reg_alpha,
            reg_lambda=self.mu_reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            device=self.device,
        )

    def _pi_params(self):
        """π 모델용 LightGBM 파라미터 (objective=cross_entropy).

        soft label(0~1 연속값)을 직접 학습 → predict()가 곧 확률. (binary objective는
        soft label을 주면 상수만 출력하는 문제가 있어 cross_entropy를 씀.)
        또 LightGBM의 cross_entropy는 GPU 빌드에서 불안정하므로 device를 'cpu'로 강제한다.
        """
        return dict(
            objective="cross_entropy",
            n_estimators=self.pi_n_estimators,
            learning_rate=self.pi_learning_rate,
            num_leaves=self.pi_num_leaves,
            max_depth=self.pi_max_depth,
            min_child_samples=self.pi_min_child_samples,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            device="cpu",   # π 모델만 항상 CPU (cross_entropy의 GPU 지원 불안정)
        )

    def _phi_params(self):
        """φ 모델용 LightGBM 파라미터 (objective=gamma — dispersion은 양수 연속값)."""
        return dict(
            objective="gamma",
            n_estimators=self.phi_n_estimators,
            learning_rate=self.phi_learning_rate,
            num_leaves=self.phi_num_leaves,
            max_depth=self.phi_max_depth,
            min_child_samples=self.phi_min_child_samples,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            device=self.device,
        )

    def _fit_lgb_phi(self, X, phi_target, w_tw):
        """φ(gamma) 모델 적합 — 라이브러리 기본값이 퇴화 split을 만들면 보수적으로 1회 재적합.

        BagZIT는 unit health를 die에 ¼ 스케일로 분배한 타깃으로 φ를 적합하는데, 그 타깃이
        clip 경계([1e-8, 1e6])에 몰리고 가중치 w_tw(=1-Π)의 동적범위가 커서, LightGBM 기본값
        (min_child_samples 작음 + max_depth 무제한)이 한쪽 count=0 split을 골라
        'Check failed: best_split_info.left_count > 0'로 죽는 경우가 있다(특히 bag 변형).
        정상 적합되면 그대로 통과(=결과 불변), 그 CHECK로 죽을 때만 φ 트리를 보수적으로
        (잎 최소표본↑·깊이 cap) 재적합해 구조를 안정화한다. μ/π 및 그 외 HP는 건드리지 않음.
        base(Pearson)·EQL _m_step이 공유한다.
        """
        params = self._phi_params()
        lgb_phi = lgb.LGBMRegressor(**params)
        try:
            lgb_phi.fit(X, phi_target, sample_weight=w_tw)
        except lgb.basic.LightGBMError:
            safe = dict(params)
            safe["min_child_samples"] = max(int(params.get("min_child_samples", 20) or 20), 100)
            md = params.get("max_depth", -1)
            safe["max_depth"] = 8 if (md is None or md <= 0) else min(int(md), 8)
            lgb_phi = lgb.LGBMRegressor(**safe)
            lgb_phi.fit(X, phi_target, sample_weight=w_tw)
        return lgb_phi

    # --- EM 알고리즘 ---

    def _initialize(self, X, y):
        """EM 시작점: Y의 zero 비율로 π, Y>0 평균으로 μ, moment estimator로 φ를 초기화."""
        n = len(y)
        is_zero = (y == 0)
        is_pos = ~is_zero

        # π 초기값 = 전체 zero 비율 (단, 0/1로 붙지 않게 [0.01, 0.99]로 클립)
        pi_init = np.clip(is_zero.mean(), 0.01, 0.99)

        # μ 초기값 = Y>0의 평균을 모든 샘플에 똑같이 깔아 둠 (양수가 없으면 작은 값)
        mu_init_val = y[is_pos].mean() if is_pos.any() else 1e-4
        mu_arr = np.full(n, mu_init_val, dtype=np.float64)

        # φ 초기값 = Y>0로 moment estimate 한 스칼라를 전 샘플에 broadcast
        phi_scalar = _estimate_phi(y[is_pos], mu_arr[is_pos], self.zeta)
        phi_arr = np.full(n, phi_scalar, dtype=np.float64)

        pi_arr = np.full(n, pi_init, dtype=np.float64)

        return pi_arr, mu_arr, phi_arr

    def _e_step(self, y, pi_arr, mu_arr, phi_arr):
        """E-step: 각 샘플이 "structural zero에서 왔을 사후확률" Π_i 계산.

        y>0  → Π_i = 0  (양수는 반드시 Tweedie 상태에서 나옴)
        y=0  → Π_i = π / [π + (1-π)·P(Y=0|Tweedie)]   (Bayes 규칙)
        """
        n = len(y)
        posterior = np.zeros(n, dtype=np.float64)   # y>0 자리는 0으로 남음

        is_zero = (y == 0)
        if is_zero.any():
            pi_z = pi_arr[is_zero]
            p0_z = _tweedie_p0(mu_arr[is_zero], phi_arr[is_zero], self.zeta)   # Tweedie 상태에서 0일 확률

            numerator = pi_z
            denominator = pi_z + (1 - pi_z) * p0_z
            denominator = np.maximum(denominator, 1e-15)   # 0 나눗셈 방지

            posterior[is_zero] = numerator / denominator

        # 0/1에 정확히 붙으면 다음 M-step 가중치가 0이 되어 불안정 → 살짝 안쪽으로 클립
        posterior = np.clip(posterior, 1e-8, 1 - 1e-8)
        return posterior

    def _m_step(self, X, y, posterior):
        """M-step: posterior Π를 고정한 채 세 LightGBM을 순서대로 다시 학습.

        1) lgb_pi  : target = Π            (cross_entropy)
        2) lgb_mu  : target = y, weight = (1-Π)/φ   (tweedie)
        3) lgb_phi : target = Pearson 잔차², weight = (1-Π)  (gamma)
        """
        w_tw = 1 - posterior   # "Tweedie 상태일 확률" — μ/φ 학습 가중치의 베이스

        # 1) π 모델: structural zero 확률을 soft label Π로 회귀
        lgb_pi = lgb.LGBMRegressor(**self._pi_params())
        lgb_pi.fit(X, posterior)
        pi_pred = lgb_pi.predict(X)
        pi_pred = np.clip(pi_pred, 1e-8, 1 - 1e-8)

        # 2) μ 모델: target은 y 그대로 (y=0도 포함 — Tweedie는 0에도 확률질량이 있어 학습에 기여),
        #    가중치는 (1-Π)/φ — Tweedie 소속 확률이 높고 분산이 작은 샘플일수록 크게 반영
        phi_for_weight = np.maximum(self._phi_current, 1e-10)
        mu_weight = w_tw / phi_for_weight

        lgb_mu = lgb.LGBMRegressor(**self._mu_params())
        lgb_mu.fit(X, y, sample_weight=mu_weight)
        mu_pred = lgb_mu.predict(X)
        mu_pred = np.maximum(mu_pred, 1e-10)

        # 3) φ 모델: dispersion을 Pearson 잔차² ((y-μ)²/μ^ζ)로 근사한 값에 회귀, 가중치 (1-Π)
        residual_sq = np.square(y - mu_pred)
        mu_pow_zeta = np.power(mu_pred, self.zeta)
        phi_target = residual_sq / np.maximum(mu_pow_zeta, 1e-10)
        phi_target = np.clip(phi_target, 1e-8, 1e6)

        lgb_phi = self._fit_lgb_phi(X, phi_target, w_tw)   # 퇴화 split 시 보수적 재적합(성공 시 결과 불변)
        phi_pred = lgb_phi.predict(X)
        phi_pred = np.clip(phi_pred, 1e-8, 1e6)

        return lgb_pi, lgb_mu, lgb_phi, pi_pred, mu_pred, phi_pred

    def fit(self, X, y, sample_weight=None):
        """EM 알고리즘으로 ZI-Tweedie 학습.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        sample_weight : ignored (sklearn API 호환용, 향후 확장 가능)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        # 초기화 (모멘트 기반 추정값)
        pi_arr, mu_arr, phi_arr = self._initialize(X, y)
        self._phi_current = phi_arr   # _m_step의 μ 가중치 (1-Π)/φ 에서 직전 φ를 참조

        self.em_history_ = []   # iter별 RMSE/π평균 등 수렴 기록
        prev_rmse = np.inf

        for em_iter in range(self.n_em_iters):
            # E-step: 현재 π/μ/φ로 posterior Π 갱신
            posterior = self._e_step(y, pi_arr, mu_arr, phi_arr)

            # M-step: Π 고정하고 세 모델 재학습 → 새 π/μ/φ
            self._phi_current = phi_arr
            lgb_pi, lgb_mu, lgb_phi, pi_arr, mu_arr, phi_arr = \
                self._m_step(X, y, posterior)

            # 이 iter의 학습세트 RMSE (수렴 모니터링용)
            pred = (1 - pi_arr) * mu_arr
            pred = np.clip(pred, 0, None)
            rmse = np.sqrt(np.mean((y - pred) ** 2))

            self.em_history_.append({
                "iter": em_iter + 1,
                "rmse": rmse,
                "pi_mean": float(pi_arr.mean()),
                "pi_std": float(pi_arr.std()),
                "mu_mean": float(mu_arr.mean()),
                "phi_mean": float(phi_arr.mean()),
                "posterior_mean": float(posterior.mean()),
                "posterior_zero_pct": float((posterior > 0.5).mean()),
            })

            if self.verbose >= 0:
                print(f"  EM iter {em_iter+1}/{self.n_em_iters}: "
                      f"RMSE={rmse:.6f}, π_mean={pi_arr.mean():.4f}, "
                      f"μ_mean={mu_arr.mean():.6f}")

            # 조기 종료: 2 iter 이상 돌았고 RMSE 변화가 em_tol보다 작으면 수렴으로 보고 멈춤
            rmse_delta = prev_rmse - rmse
            if em_iter >= 2 and abs(rmse_delta) < self.em_tol:
                if self.verbose >= 0:
                    print(f"  EM early stop at iter {em_iter+1}: "
                          f"|ΔRMSE|={abs(rmse_delta):.2e} < tol={self.em_tol:.1e}")
                break
            prev_rmse = rmse

        self.n_em_iters_actual_ = em_iter + 1   # 실제로 돈 iter 수 (조기 종료 가능)

        # 마지막 iter의 세 모델을 최종 모델로 보관
        self.lgb_pi_ = lgb_pi
        self.lgb_mu_ = lgb_mu
        self.lgb_phi_ = lgb_phi
        self.fitted_ = True

        return self

    def predict(self, X):
        """E[Y] = (1 - π(x)) × μ(x), 음수는 0으로 clip.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        pred : ndarray of shape (n_samples,)
        """
        pi, mu, _ = self.predict_components(X)
        pred = (1 - pi) * mu
        return np.clip(pred, 0, None)

    def predict_components(self, X):
        """π, μ, φ 각각의 예측값 반환 (진단/후처리용 — 예: τ_π로 die를 0으로 누르기).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        pi : ndarray — P(structural zero)
        mu : ndarray — Tweedie mean
        phi : ndarray — dispersion
        """
        if not hasattr(self, "fitted_"):
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.asarray(X, dtype=np.float64)
        pi = self.lgb_pi_.predict(X)   # cross_entropy objective라 predict 결과가 곧 확률
        pi = np.clip(pi, 1e-8, 1 - 1e-8)
        mu = self.lgb_mu_.predict(X)
        mu = np.maximum(mu, 1e-10)
        phi = self.lgb_phi_.predict(X)
        phi = np.clip(phi, 1e-8, 1e6)
        return pi, mu, phi

    def score_loglik(self, X, y):
        """학습된 π/μ/φ로 (X, y)에서 ZI-Tweedie EQL 로그우도 합을 계산.

        Algorithm 2(ζ profile likelihood)에서 후보 ζ들을 비교할 때 쓴다.
        fit()은 self.zeta 한 값으로 EM(Algorithm 1)을 돌리므로, ζ 추정은 노트북에서
        여러 ζ로 fit→score_loglik 해 최대값을 고르는 식으로 수행한다 (논문 Algorithm 2).

        로그우도식 _zitweedie_loglik은 π/μ/φ 값만으로 계산되고 φ의 *학습 방식*(Pearson vs EQL)과
        무관하므로, 이 메서드는 부모에 두어 ZITboost/ZITboostEQL/BagZIT/BagZITEQL 4종이 공유한다.
        """
        pi, mu, phi = self.predict_components(X)
        return _zitweedie_loglik(np.asarray(y, dtype=np.float64).ravel(), pi, mu, phi, self.zeta)


# --- ZITboostEQLRegressor (Gu 2024 충실 버전) ---
#
# ZITboostRegressor 대비 논문(arXiv:2405.14990)과 다른 곳은 정확히 2곳뿐이라, 그 2곳만 오버라이드한다.
#   ① _initialize : φ₀ 를 zero-truncated Tweedie deviance moment 로 (기존: Var(y)/μ^ζ Pearson moment)
#   ② _m_step     : φ 타깃을 Tweedie unit deviance 로 (기존: (y-μ)²/μ^ζ Pearson) — extended quasi-likelihood
# 나머지(E-step posterior, π=cross_entropy soft-label, μ=tweedie weight (1-Π)/φ, predict=(1-π)μ,
#        세 LightGBM 파라미터 빌더 _pi/_mu/_phi_params)는 ZITboostRegressor가 이미 논문과 일치하므로 그대로 상속.

class ZITboostEQLRegressor(ZITboostRegressor):
    """Gu 2024(arXiv:2405.14990)에 100% 충실한 ZI-Tweedie + LightGBM EM.

    부모 ZITboostRegressor와 모델/EM 구조는 동일하고, 분산 φ의 추정 방식만 논문 방향으로 교체한다:
      - 초기화(스칼라부) : μ₀=Σ_{y>0}wy/Σ_{y>0}w (=mean(y>0), w=1),
                          φ₀=Σ_{y>0} w·D_ζ(y;μ₀)/ΣI(y>0) (deviance moment). 이 스칼라 식은 논문과 일치.
      - φ M-step : extended quasi-likelihood + saddlepoint → "weighted gamma regression"
                   (target = Tweedie unit deviance D_ζ(y;μ̂), sample_weight=(1-Π)).
    __init__/HP는 부모와 동일(추가 인자 없음) — 노트북에서 LightGBM 라이브러리 기본값을 넘기면 그대로 기본값으로 돈다.

    ⚠️ 논문과 다른/근사인 지점 (이 클래스는 'Gu 2024 구조를 따른 적응 구현'이지 100% 충실 구현이 아님):
      - 초기화(GBT부): 논문은 스칼라 μ₀/φ₀를 구한 뒤 *초기 GBT F̂μ^(0)/F̂φ^(0)/F̂π^(0)를 따로 적합*한다.
        여기서는 그 단계를 생략하고 스칼라 μ₀/φ₀/π₀(=zero율)를 첫 E-step에 그대로 써, 첫 M-step이 사실상 F^(0) 역할을 한다(근사).
      - 로그우도/φ: exact Tweedie가 아니라 EQL/saddlepoint 근사(논문도 EQL을 쓰지만, '근사'임을 명시).
      - ζ: fit()은 Algorithm 1(주어진 ζ EM). ζ는 Algorithm 2(profile likelihood)로 노트북에서 추정(score_loglik).
        단 ζ 그리드 간격은 논문 verbatim이 아니라 임의(0.1).
      - exposure w_i=1(노출량 없음), unit target→die broadcast→die 평균 집계: die/unit 구조에 맞춘 프로젝트 적응.
      - 위 수식들은 arXiv HTML 전사 기준이며 PDF 원문 한 줄 대조는 아직 아님.
    """

    def _initialize(self, X, y):
        """zero-truncated Tweedie 초기화 (논문 식).

        μ₀ = (Σ_{i:y_i>0} y_i) / n_pos                      (양수만의 평균; w_i=1)
        φ₀ = (Σ_{i:y_i>0} D_ζ(y_i; μ₀)) / n_pos             (deviance moment)
        π₀ = zero 비율 (0/1에 붙지 않게 [0.01,0.99] 클립)
        부모와 다른 곳은 φ₀ 한 줄뿐 (부모는 Var(y)/μ^ζ Pearson moment).
        """
        n = len(y)
        is_zero = (y == 0)
        is_pos = ~is_zero
        n_pos = int(is_pos.sum())

        pi_init = np.clip(is_zero.mean(), 0.01, 0.99)

        mu_init_val = y[is_pos].mean() if n_pos > 0 else 1e-4
        mu_arr = np.full(n, mu_init_val, dtype=np.float64)

        # φ₀: zero-truncated Tweedie deviance moment (saddlepoint 하 E[D_ζ]≈φ)
        if n_pos > 0:
            dev_pos = _tweedie_unit_deviance(y[is_pos], mu_arr[is_pos], self.zeta)
            phi_scalar = float(np.clip(dev_pos.sum() / n_pos, 1e-6, 1e6))
        else:
            phi_scalar = 1.0
        phi_arr = np.full(n, phi_scalar, dtype=np.float64)

        pi_arr = np.full(n, pi_init, dtype=np.float64)
        return pi_arr, mu_arr, phi_arr

    def _m_step(self, X, y, posterior):
        """M-step: 부모와 동일하되 φ 타깃만 Tweedie unit deviance(EQL/saddlepoint)로 교체.

        1) lgb_pi  : target = Π            (cross_entropy)                  — 부모와 동일
        2) lgb_mu  : target = y, weight=(1-Π)/φ̂   (tweedie)               — 부모와 동일
        3) lgb_phi : target = D_ζ(y;μ̂),  weight=(1-Π)  (gamma, EQL)        — 부모(Pearson)와 다름
        """
        w_tw = 1.0 - posterior   # "Tweedie 상태일 확률"

        # 1) π 모델 — structural zero 확률을 soft label Π로 회귀 (부모와 동일)
        lgb_pi = lgb.LGBMRegressor(**self._pi_params())
        lgb_pi.fit(X, posterior)
        pi_pred = np.clip(lgb_pi.predict(X), 1e-8, 1 - 1e-8)

        # 2) μ 모델 — weighted Tweedie deviance, weight=(1-Π)/φ̂^(k) (부모와 동일)
        phi_for_weight = np.maximum(self._phi_current, 1e-10)
        mu_weight = w_tw / phi_for_weight
        lgb_mu = lgb.LGBMRegressor(**self._mu_params())
        lgb_mu.fit(X, y, sample_weight=mu_weight)
        mu_pred = np.maximum(lgb_mu.predict(X), 1e-10)

        # 3) φ 모델 — EQL/saddlepoint: target = Tweedie unit deviance, weight=(1-Π) (논문 충실 지점)
        phi_target = _tweedie_unit_deviance(y, mu_pred, self.zeta)
        lgb_phi = self._fit_lgb_phi(X, phi_target, w_tw)   # 퇴화 split 시 보수적 재적합(성공 시 결과 불변)
        phi_pred = np.clip(lgb_phi.predict(X), 1e-8, 1e6)

        return lgb_pi, lgb_mu, lgb_phi, pi_pred, mu_pred, phi_pred


# --- BagZITboostRegressor ---
#
# ZITboost에 "unit 제약(bag constraint)"을 추가한 변형.
# 핵심: target이 unit 단위(unit 1개 = die 4개가 health 1개를 공유)인데, ZITboost는 die 단위로 학습한다.
#   그래서 매 EM iter마다 unit의 health를 그 unit의 4 die에 "현재 die 예측 기여도(=(1-π)μ)에 비례"하게 다시 나눠 준다 (B3 allocation).
# 시그니처가 다름: fit(X, y, unit_id)로 unit_id가 필수 → hpo.refit_best 같은 표준 경로와 호환 안 됨.
#   노트북에서 직접 fold 루프를 돌리고 model.predict_components(X)를 쓰는 식으로 사용한다.

class BagZITboostRegressor(ZITboostRegressor):
    """ZITboost + bag(unit) constraint — 매 EM iter마다 unit_y를 die에 기여도 비례 분배(B3)."""

    @staticmethod
    def _allocate_b3(unit_y_per_unit, contribution, inverse, n_units):
        """unit별 health를, 그 unit 안 die들의 contribution((1-π)μ) 비율대로 die-level로 쪼갬.

        contribution 합이 0에 가까운 unit은 비율을 못 정하니 die 수로 균등 분배.
        inverse: 각 die가 몇 번째 unit에 속하는지의 정수 인덱스 (np.unique(return_inverse) 결과).
        """
        contrib_sum_per_unit = np.zeros(n_units)
        np.add.at(contrib_sum_per_unit, inverse, contribution)   # unit별 contribution 합
        contrib_sum_die = contrib_sum_per_unit[inverse]          # 그 합을 die-level로 broadcast
        n_die_per_unit = np.bincount(inverse, minlength=n_units).astype(np.float64)
        n_die_die = n_die_per_unit[inverse]
        share = np.where(
            contrib_sum_die > 1e-12,
            contribution / np.maximum(contrib_sum_die, 1e-12),   # 정상: 기여도 비율
            1.0 / np.maximum(n_die_die, 1.0),                    # 기여도 0 unit: 균등 분배
        )
        return unit_y_per_unit[inverse] * share                  # die-level로 쪼갠 target

    def fit(self, X, y, unit_id):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        unit_id = np.asarray(unit_id)

        # unit 목록 + 각 die가 속한 unit 인덱스(inverse) + 각 unit의 대표 행(first_idx)
        unique_units, first_idx, inverse = np.unique(
            unit_id, return_index=True, return_inverse=True
        )
        n_units = len(unique_units)
        unit_y_per_unit = y[first_idx]   # unit별 health (같은 unit의 die는 y가 동일하므로 대표값 하나면 됨)

        # 첫 iter의 die-level target: 일단 unit health를 die 수로 균등 분배
        n_die_per_unit = np.bincount(inverse, minlength=n_units).astype(np.float64)
        y_die_alloc = unit_y_per_unit[inverse] / np.maximum(n_die_per_unit[inverse], 1.0)

        pi_arr, mu_arr, phi_arr = self._initialize(X, y_die_alloc)
        self._phi_current = phi_arr

        self.em_history_ = []
        prev_rmse = np.inf
        em_iter = 0

        for em_iter in range(self.n_em_iters):
            if em_iter > 0:
                # 직전 die 예측 기여도에 비례하게 unit_y를 다시 분배 (bag constraint의 핵심)
                contribution = np.clip((1 - pi_arr) * mu_arr, 0, None)
                y_die_alloc = self._allocate_b3(
                    unit_y_per_unit, contribution, inverse, n_units
                )

            posterior = self._e_step(y_die_alloc, pi_arr, mu_arr, phi_arr)

            self._phi_current = phi_arr
            lgb_pi, lgb_mu, lgb_phi, pi_arr, mu_arr, phi_arr = \
                self._m_step(X, y_die_alloc, posterior)

            # 수렴 모니터링은 die가 아니라 unit RMSE로 (die 합 = unit 예측)
            pred_die = np.clip((1 - pi_arr) * mu_arr, 0, None)
            pred_unit = np.zeros(n_units)
            np.add.at(pred_unit, inverse, pred_die)   # unit별 die 예측 합
            rmse_unit = float(np.sqrt(np.mean((unit_y_per_unit - pred_unit) ** 2)))

            self.em_history_.append({
                "iter":      em_iter + 1,
                "unit_rmse": rmse_unit,
                "pi_mean":   float(pi_arr.mean()),
                "mu_mean":   float(mu_arr.mean()),
            })

            rmse_delta = prev_rmse - rmse_unit
            if em_iter >= 2 and abs(rmse_delta) < self.em_tol:
                break
            prev_rmse = rmse_unit

        self.n_em_iters_actual_ = em_iter + 1
        self.lgb_pi_ = lgb_pi
        self.lgb_mu_ = lgb_mu
        self.lgb_phi_ = lgb_phi
        self.fitted_ = True
        return self

    def predict_unit(self, X, unit_id):
        """die-level 예측을 한 뒤 같은 unit끼리 합쳐 unit-level 예측으로 반환."""
        if not hasattr(self, "fitted_"):
            raise ValueError("Model not fitted")
        unit_id = np.asarray(unit_id)
        unique_units, inverse = np.unique(unit_id, return_inverse=True)
        n_units = len(unique_units)
        pred_die = self.predict(X)
        pred_unit = np.zeros(n_units)
        np.add.at(pred_unit, inverse, pred_die)   # unit별 die 예측 합 = unit 예측
        return pred_unit, unique_units


# --- BagZITEQLRegressor (BagZIT의 EQL/논문충실 φ 변종) ---
#
# BagZITboostRegressor(bag constraint) 에 ZITboostEQLRegressor 의 φ M-step(EQL/saddlepoint =
# Tweedie unit deviance 타깃) 만 얹은 변형. 즉 두 축의 조합이다:
#   ① bag(unit) constraint  : BagZITboostRegressor 로부터 상속 (fit(X,y,unit_id)·B3 allocation·predict_unit)
#   ② EQL φ M-step          : ZITboostEQLRegressor._m_step 재사용 (φ 타깃 = Tweedie unit deviance)
# _initialize 는 부모 BagZIT(→ZITboostRegressor)의 Pearson moment 초기화를 그대로 쓴다
#   (02_bag_zit_eql_parallel_hpo.py 워커의 BagZITEQLRegressor 와 동일: 워커도 _m_step 만 EQL로 교체했음).
# BagZIT 자체가 논문 외 변형이므로 '논문충실'은 φ M-step 한정 의미.

class BagZITEQLRegressor(BagZITboostRegressor):
    """BagZIT + EQL φ M-step (Tweedie unit deviance) — BagZIT의 논문충실(EQL) 변종.

    bag(unit) constraint·EM·예측·초기화는 BagZITboostRegressor 그대로 상속하고,
    φ의 M-step 타깃만 Pearson 잔차에서 Tweedie unit deviance(extended quasi-likelihood)로 교체한다.
    이 교체분이 ZITboostEQLRegressor._m_step 과 수식이 동일하므로 그대로 재사용한다.

    시그니처는 부모와 동일: fit(X, y, unit_id) — unit_id 필수, die→unit 집계는 SUM(predict_unit).
    score_loglik(부모 ZITboostRegressor 제공)도 상속하므로 ζ profile likelihood(Algorithm 2) 가능.
    """

    def _m_step(self, X, y, posterior):
        # ZITboostEQLRegressor 와 동일한 EQL(Tweedie unit deviance) φ M-step 을 재사용.
        # (self 는 BagZITEQL 인스턴스로 바인딩되어 _pi/_mu/_phi_params·_phi_current·zeta 는 BagZIT 것을 쓴다.)
        return ZITboostEQLRegressor._m_step(self, X, y, posterior)


# ---------------------------------------------------------------------------
# EMTboost — Zhou, Qian & Yang (2019, arXiv:1811.10192)
#   ZI-Tweedie의 또 다른 형식: π(=1-q)·φ가 **전역 스칼라**, μ만 LightGBM Tweedie f(X).
#   예측 E[Y]=q·μ. 위 ZITboost 4종(π/μ/φ 전부 f(X))과 달리 스칼라 q/φ를 EM으로 추정.
#   Tweedie 수학 헬퍼(_tweedie_p0 / _tweedie_unit_deviance / _zitweedie_loglik)는 위 정의를 공유.
#   (구 modules/zit_EMT.py를 zit.py로 통합 — 중복 헬퍼 제거. zit_EMT.py 삭제, import는 modules.zit 직접.)
# ---------------------------------------------------------------------------

def _estimate_phi_scalar(y, mu, delta1, zeta):
    """논문 eq.(23): δ₁ᵢ 가중 로그우도를 최대화하는 φ 스칼라를 golden section search로 추정.

    목적함수 (φ에 관한 부분만):
      Σ_i δ₁ᵢ · [ log a(yᵢ, φ, ρ) + (1/φ)(yᵢμᵢ^(1-ρ)/(1-ρ) - μᵢ^(2-ρ)/(2-ρ)) ]

    log a(y,φ,ρ) 의 exact 계산은 복잡하므로 saddlepoint 근사를 사용:
      ≈ -½ log(2πφ yᵢ^ρ)  (y>0인 항만; y=0은 φ 목적함수에 기여 없음)

    최종 목적함수 (음수 → 최소화):
      -Σ_{y>0} δ₁ᵢ · [ -½log(φ) - (1/2φ)·Dζ(yᵢ;μᵢ) ]
      = Σ_{y>0} δ₁ᵢ · [ ½log(φ) + Dζ/(2φ) ]
    """
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    delta1 = np.asarray(delta1, dtype=np.float64)

    is_pos = (y > 0)
    if not is_pos.any():
        return 1.0

    d1_pos = delta1[is_pos]
    dev_pos = _tweedie_unit_deviance(y[is_pos], mu[is_pos], zeta)

    def neg_loglik(log_phi):
        phi = np.exp(log_phi)
        # Σ δ₁ᵢ · [ -½log(φ) - Dζ/(2φ) ]  → 최대화 → 음수 취해 최소화
        val = float(np.sum(d1_pos * (-0.5 * log_phi - dev_pos / (2.0 * phi))))
        return -val

    result = minimize_scalar(neg_loglik, bounds=(-10, 10), method="bounded")
    return float(np.clip(np.exp(result.x), 1e-6, 1e6))


class EMTboost(BaseEstimator, RegressorMixin):
    """Zhou et al. (2019) EMTboost — π/φ 스칼라, μ=f(X) LightGBM.

    Parameters
    ----------
    zeta : float
        Tweedie power ζ ∈ (1, 2).
    n_em_iters : int
        EM 반복 횟수.
    em_tol : float
        조기 종료 RMSE 변화 임계값.
    mu_* : μ 모델 LightGBM 하이퍼파라미터.
    """

    def __init__(
        self,
        zeta=1.5,
        n_em_iters=10,
        em_tol=1e-7,
        mu_n_estimators=500,
        mu_learning_rate=0.05,
        mu_num_leaves=31,
        mu_max_depth=6,
        mu_min_child_samples=20,
        mu_subsample=0.8,
        mu_colsample_bytree=0.8,
        mu_reg_alpha=1e-3,
        mu_reg_lambda=1e-1,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
        device="cpu",
    ):
        self.zeta = zeta
        self.n_em_iters = n_em_iters
        self.em_tol = em_tol
        self.mu_n_estimators = mu_n_estimators
        self.mu_learning_rate = mu_learning_rate
        self.mu_num_leaves = mu_num_leaves
        self.mu_max_depth = mu_max_depth
        self.mu_min_child_samples = mu_min_child_samples
        self.mu_subsample = mu_subsample
        self.mu_colsample_bytree = mu_colsample_bytree
        self.mu_reg_alpha = mu_reg_alpha
        self.mu_reg_lambda = mu_reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.device = device

    def _mu_params(self):
        return dict(
            objective="tweedie",
            tweedie_variance_power=self.zeta,
            n_estimators=self.mu_n_estimators,
            learning_rate=self.mu_learning_rate,
            num_leaves=self.mu_num_leaves,
            max_depth=self.mu_max_depth,
            min_child_samples=self.mu_min_child_samples,
            subsample=self.mu_subsample,
            subsample_freq=1,
            colsample_bytree=self.mu_colsample_bytree,
            reg_alpha=self.mu_reg_alpha,
            reg_lambda=self.mu_reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            device=self.device,
        )

    # --- 초기화 (논문 eq.33/34/35) ---

    def _initialize(self, X, y):
        """논문 eq.(33)/(34)/(35): y>0 샘플로 F⁰, φ⁰, q⁰ 초기화.

        F⁰ = log( Σ_{y>0} y / n_{y>0} )   → μ⁰ = mean(y>0)
        φ⁰ = golden section search (y>0만, δ₁ᵢ=1 가정)
        q⁰ = n_{y>0} / n
        """
        n = len(y)
        is_pos = (y > 0)
        n_pos = int(is_pos.sum())

        # q⁰ = n_{y>0}/n (논문 eq.35)
        q0 = float(np.clip(n_pos / n, 0.01, 0.99))

        # μ⁰ = mean(y>0) → 전체 샘플에 broadcast (논문 eq.33)
        mu0_val = float(y[is_pos].mean()) if n_pos > 0 else 1e-4
        mu0_val = max(mu0_val, 1e-10)
        mu_arr = np.full(n, mu0_val, dtype=np.float64)

        # φ⁰: y>0 샘플, δ₁ᵢ=1 가정으로 golden section search (논문 eq.34)
        delta1_init = is_pos.astype(np.float64)
        phi0 = _estimate_phi_scalar(y, mu_arr, delta1_init, self.zeta)

        return q0, mu_arr, phi0

    # --- E-step (논문 eq.14/15) ---

    def _e_step(self, y, q, mu_arr, phi):
        """논문 eq.(14): δ₁ᵢ = P(Πᵢ=1|yᵢ) = Tweedie 상태일 사후확률.

        y > 0 → δ₁ᵢ = 1  (양수는 반드시 Tweedie에서 나옴)
        y = 0 → δ₁ᵢ = q·exp(−λ) / (q·exp(−λ) + (1−q))
                  λ = μ^(2-ζ) / ((2-ζ)·φ)
        """
        n = len(y)
        delta1 = np.ones(n, dtype=np.float64)

        is_zero = (y == 0)
        if is_zero.any():
            p0 = _tweedie_p0(mu_arr[is_zero], phi, self.zeta)  # exp(-λ)
            numer = q * p0                                       # q·exp(-λ)
            denom = np.maximum(numer + (1.0 - q), 1e-15)        # + (1-q)
            delta1[is_zero] = numer / denom

        return np.clip(delta1, 1e-8, 1 - 1e-8)

    # --- M-step (논문 eq.21/23/24) ---

    def _m_step(self, X, y, delta1, phi_current):
        """논문 eq.(21)/(23)/(24): δ₁ 고정 후 μ/φ/q 순서대로 갱신.

        eq.(21) μ : LightGBM Tweedie, weight=δ₁ᵢ
        eq.(23) φ : golden section search (스칼라)
        eq.(24) q : (1/n) Σ δ₁ᵢ (스칼라)
        """
        # eq.(21): μ 모델 — weight=δ₁ᵢ
        lgb_mu = lgb.LGBMRegressor(**self._mu_params())
        lgb_mu.fit(X, y, sample_weight=delta1)
        mu_pred = np.maximum(lgb_mu.predict(X), 1e-10)

        # eq.(23): φ 스칼라 — golden section search
        phi_new = _estimate_phi_scalar(y, mu_pred, delta1, self.zeta)

        # eq.(24): q 스칼라 — δ₁ᵢ 평균
        q_new = float(np.clip(delta1.mean(), 0.01, 0.99))

        return lgb_mu, mu_pred, phi_new, q_new

    # --- fit ---

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        # 초기화
        q, mu_arr, phi = self._initialize(X, y)

        self.em_history_ = []
        prev_rmse = np.inf

        for em_iter in range(self.n_em_iters):
            # E-step
            delta1 = self._e_step(y, q, mu_arr, phi)

            # M-step
            lgb_mu, mu_arr, phi, q = self._m_step(X, y, delta1, phi)

            # 수렴 모니터링 — 논문 eq.(41): E[Y] = q·μ
            pred = np.clip(q * mu_arr, 0, None)
            rmse = float(np.sqrt(np.mean((y - pred) ** 2)))

            self.em_history_.append({
                "iter":        em_iter + 1,
                "rmse":        rmse,
                "q_scalar":    q,
                "phi_scalar":  phi,
                "mu_mean":     float(mu_arr.mean()),
                "delta1_mean": float(delta1.mean()),
            })

            if self.verbose >= 0:
                print(f"  EM iter {em_iter+1}/{self.n_em_iters}: "
                      f"RMSE={rmse:.6f}, q={q:.4f}, φ={phi:.4f}")

            rmse_delta = prev_rmse - rmse
            if em_iter >= 2 and abs(rmse_delta) < self.em_tol:
                if self.verbose >= 0:
                    print(f"  EM early stop at iter {em_iter+1}: "
                          f"|ΔRMSE|={abs(rmse_delta):.2e} < tol={self.em_tol:.1e}")
                break
            prev_rmse = rmse

        self.n_em_iters_actual_ = em_iter + 1
        self.lgb_mu_ = lgb_mu
        self.q_ = q        # 스칼라
        self.phi_ = phi    # 스칼라
        self.fitted_ = True
        return self

    # --- 예측 ---

    def predict(self, X):
        """논문 eq.(41): E[Y|x] = q·μ(x) — q=P(Tweedie), 스칼라."""
        _, mu, _ = self.predict_components(X)
        return np.clip(self.q_ * mu, 0, None)

    def predict_components(self, X):
        """π(structural zero 확률 = 1-q, 스칼라 broadcast), μ(X함수), φ(스칼라 broadcast) 반환.

        ZITboostGu.predict_components()와 동일한 인터페이스 — 비교 실험 호환.
        π = 1-q (structural zero 확률) 로 변환해서 반환 → predict = (1-π)·μ = q·μ
        """
        if not hasattr(self, "fitted_"):
            raise ValueError("Model not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        mu = np.maximum(self.lgb_mu_.predict(X), 1e-10)
        pi = np.full(n, 1.0 - self.q_)   # structural zero 확률 = 1-q
        phi = np.full(n, self.phi_)
        return pi, mu, phi

    def score_loglik(self, X, y):
        """ZI-Tweedie EQL 로그우도 합 — ζ profile likelihood(Algorithm 2)용."""
        pi, mu, phi = self.predict_components(X)
        return _zitweedie_loglik(np.asarray(y, dtype=np.float64).ravel(), pi, mu, phi, self.zeta)
