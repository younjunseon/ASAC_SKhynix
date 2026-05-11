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

이 모듈은 ZITboostRegressor(기본)와 BagZITboostRegressor(unit 제약을 추가한 변형) 두 클래스를 제공한다.

사용법:
    from modules.zit import ZITboostRegressor

    model = ZITboostRegressor(zeta=1.5, n_em_iters=10)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pi, mu, phi = model.predict_components(X_test)
"""

import numpy as np
import lightgbm as lgb
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

        lgb_phi = lgb.LGBMRegressor(**self._phi_params())
        lgb_phi.fit(X, phi_target, sample_weight=w_tw)
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
