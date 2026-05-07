"""
ZITboost — Zero-Inflated Tweedie + LightGBM EM 모델

Gu 2024 (논문 2-16) 기반 구현.
semicontinuous 데이터(0에 확률질량 + 양수에 연속분포)를 단일 모델로 처리.

구조:
  Y_i = 0                              확률 π(x_i)
  Y_i ~ Tweedie(μ_i, φ_i, ζ)          확률 1-π(x_i)

  E[Y] = (1-π) × μ

3개 LightGBM 모델을 EM 알고리즘으로 동시 학습:
  - lgb_pi:  P(structural zero)   — cross-entropy (soft label)
  - lgb_mu:  Tweedie mean         — weighted Tweedie deviance
  - lgb_phi: dispersion           — weighted gamma regression

사용법:
    from modules.zi_tweedie import ZITboostRegressor

    model = ZITboostRegressor(zeta=1.5, n_em_iters=10)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pi, mu, phi = model.predict_components(X_test)
"""

import numpy as np
import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin

from utils.config import SEED


# ─── Tweedie 수학 유틸 ───────────────────────────────────────

def _tweedie_p0(mu, phi, zeta):
    """P(Y=0 | Tweedie) = exp(-λ), λ = μ^(2-ζ) / ((2-ζ)φ)

    compound Poisson-Gamma 표현에서 Poisson rate λ.
    ζ ∈ (1, 2) 범위에서만 유효 (ζ=1: Poisson, ζ=2: Gamma).
    """
    mu = np.maximum(mu, 1e-10)
    phi = np.maximum(phi, 1e-10)
    lam = np.power(mu, 2 - zeta) / ((2 - zeta) * phi)
    return np.exp(-lam)


def _estimate_phi(y_pos, mu_pos, zeta):
    """Y>0 샘플로부터 dispersion φ 추정 (moment estimator).

    Var(Y|Y>0) ≈ φ × μ^ζ  →  φ ≈ Var / μ^ζ
    """
    if len(y_pos) < 2:
        return 1.0
    mu_mean = np.maximum(np.mean(mu_pos), 1e-10)
    var_y = np.var(y_pos, ddof=1)
    phi = var_y / np.maximum(np.power(mu_mean, zeta), 1e-10)
    return np.clip(phi, 1e-6, 1e6)


# ─── ZITboostRegressor ──────────────────────────────────────

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
        # ZIT 전용
        zeta=1.5,
        n_em_iters=10,
        em_tol=1e-7,
        # μ 모델 (핵심, 9개 HP)
        mu_n_estimators=500,
        mu_learning_rate=0.05,
        mu_num_leaves=31,
        mu_max_depth=6,
        mu_min_child_samples=20,
        mu_subsample=0.8,
        mu_colsample_bytree=0.8,
        mu_reg_alpha=1e-3,
        mu_reg_lambda=1e-1,
        # π 모델 (분류, 5개 HP)
        pi_n_estimators=200,
        pi_learning_rate=0.05,
        pi_num_leaves=31,
        pi_max_depth=6,
        pi_min_child_samples=20,
        # φ 모델 (분산, 5개 HP)
        phi_n_estimators=200,
        phi_learning_rate=0.05,
        phi_num_leaves=31,
        phi_max_depth=6,
        phi_min_child_samples=20,
        # 공통
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
        device="cpu",
    ):
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

    # ─── 내부 LightGBM 파라미터 구성 ─────────────────────────

    def _mu_params(self):
        """μ 모델 (Tweedie regression) 파라미터."""
        return dict(
            objective="tweedie",
            tweedie_variance_power=self.zeta,
            n_estimators=self.mu_n_estimators,
            learning_rate=self.mu_learning_rate,
            num_leaves=self.mu_num_leaves,
            max_depth=self.mu_max_depth,
            min_child_samples=self.mu_min_child_samples,
            subsample=self.mu_subsample,
            subsample_freq=1,  # ← 없으면 subsample 무시됨 (LGBM 기본값 0)
            colsample_bytree=self.mu_colsample_bytree,
            reg_alpha=self.mu_reg_alpha,
            reg_lambda=self.mu_reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            device=self.device,
        )

    def _pi_params(self):
        """π 모델 (soft-label classification) 파라미터.

        cross_entropy objective: 연속 [0,1] soft label을 직접 학습.
        predict()가 확률을 직접 반환하므로 sigmoid 변환 불필요.
        (binary objective는 soft label 입력 시 상수 출력 버그 발생)

        GPU 가드: LightGBM의 cross_entropy objective는 GPU 빌드에서 지원이
        불안정하여(버전에 따라 실패) μ·φ 가 GPU여도 π 모델만 CPU로 강제.
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
            device="cpu",   # ← GPU 설정 무시, π 는 항상 CPU
        )

    def _phi_params(self):
        """φ 모델 (gamma regression for dispersion) 파라미터."""
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

    # ─── EM 알고리즘 ─────────────────────────────────────────

    def _initialize(self, X, y):
        """EM 초기화: Y>0으로부터 μ, φ, π 추정."""
        n = len(y)
        is_zero = (y == 0)
        is_pos = ~is_zero

        # π 초기값: 전체 zero 비율
        pi_init = np.clip(is_zero.mean(), 0.01, 0.99)

        # μ 초기값: Y>0 평균 (전체 샘플에 broadcast)
        mu_init_val = y[is_pos].mean() if is_pos.any() else 1e-4
        mu_arr = np.full(n, mu_init_val, dtype=np.float64)

        # φ 초기값: Y>0 moment estimator
        phi_scalar = _estimate_phi(y[is_pos], mu_arr[is_pos], self.zeta)
        phi_arr = np.full(n, phi_scalar, dtype=np.float64)

        # π array
        pi_arr = np.full(n, pi_init, dtype=np.float64)

        return pi_arr, mu_arr, phi_arr

    def _e_step(self, y, pi_arr, mu_arr, phi_arr):
        """E-step: posterior P(structural zero | y_i, x_i).

        y=0:  Π_i = π_i / [π_i + (1-π_i) × f_Tw(0; μ_i, φ_i, ζ)]
        y>0:  Π_i = 0  (반드시 Tweedie 상태에서 발생)
        """
        n = len(y)
        posterior = np.zeros(n, dtype=np.float64)

        is_zero = (y == 0)
        if is_zero.any():
            pi_z = pi_arr[is_zero]
            p0_z = _tweedie_p0(mu_arr[is_zero], phi_arr[is_zero], self.zeta)

            numerator = pi_z
            denominator = pi_z + (1 - pi_z) * p0_z
            denominator = np.maximum(denominator, 1e-15)

            posterior[is_zero] = numerator / denominator

        # 범위 제한 (수치 안정성)
        posterior = np.clip(posterior, 1e-8, 1 - 1e-8)
        return posterior

    def _m_step(self, X, y, posterior):
        """M-step: 3개 LightGBM 모델 순차 업데이트.

        1. lgb_pi:  target=Π_i, objective=cross_entropy (soft labels)
        2. lgb_mu:  target=y,   weight=(1-Π_i)/φ_i, objective=tweedie
        3. lgb_phi: target=deviance_residual², weight=(1-Π_i), objective=gamma
        """
        w_tw = 1 - posterior  # Tweedie 소속 확률

        # ── 1) π 모델: P(structural zero) ──
        # cross_entropy objective: 연속 [0,1] soft label 직접 학습.
        # predict()가 확률을 직접 반환하므로 sigmoid 변환 불필요.
        lgb_pi = lgb.LGBMRegressor(**self._pi_params())
        lgb_pi.fit(X, posterior)
        pi_pred = lgb_pi.predict(X)
        pi_pred = np.clip(pi_pred, 1e-8, 1 - 1e-8)

        # ── 2) μ 모델: Tweedie mean ──
        # weight = (1 - Π_i) / φ_i
        phi_for_weight = np.maximum(self._phi_current, 1e-10)
        mu_weight = w_tw / phi_for_weight

        # target은 y 자체. y=0인 샘플도 포함 (Tweedie P(0)>0이므로 학습에 기여)
        lgb_mu = lgb.LGBMRegressor(**self._mu_params())
        lgb_mu.fit(X, y, sample_weight=mu_weight)
        mu_pred = lgb_mu.predict(X)
        mu_pred = np.maximum(mu_pred, 1e-10)

        # ── 3) φ 모델: dispersion ──
        # target = Tweedie deviance residual² (unit deviance)
        # d_i = 2 × [y·(y^(1-ζ) - μ^(1-ζ))/(1-ζ) - (y^(2-ζ) - μ^(2-ζ))/(2-ζ)]
        # 간소화: φ ≈ (y - μ)² / μ^ζ  (Pearson residual² 기반, 보다 안정적)
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

        # 초기화
        pi_arr, mu_arr, phi_arr = self._initialize(X, y)
        self._phi_current = phi_arr

        self.em_history_ = []
        prev_rmse = np.inf

        for em_iter in range(self.n_em_iters):
            # E-step
            posterior = self._e_step(y, pi_arr, mu_arr, phi_arr)

            # M-step
            self._phi_current = phi_arr
            lgb_pi, lgb_mu, lgb_phi, pi_arr, mu_arr, phi_arr = \
                self._m_step(X, y, posterior)

            # E[Y] = (1 - π) × μ
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

            # Early stopping: RMSE 변화가 tol 미만이면 수렴으로 판단
            rmse_delta = prev_rmse - rmse
            if em_iter >= 2 and abs(rmse_delta) < self.em_tol:
                if self.verbose >= 0:
                    print(f"  EM early stop at iter {em_iter+1}: "
                          f"|ΔRMSE|={abs(rmse_delta):.2e} < tol={self.em_tol:.1e}")
                break
            prev_rmse = rmse

        self.n_em_iters_actual_ = em_iter + 1

        # 최종 모델 저장
        self.lgb_pi_ = lgb_pi
        self.lgb_mu_ = lgb_mu
        self.lgb_phi_ = lgb_phi
        self.fitted_ = True

        return self

    def predict(self, X):
        """E[Y] = (1 - π(x)) × μ(x), clipped to ≥ 0.

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
        """π, μ, φ 각각의 예측값 반환 (진단/후처리용).

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
        pi = self.lgb_pi_.predict(X)  # cross_entropy: 확률 직접 반환
        pi = np.clip(pi, 1e-8, 1 - 1e-8)
        mu = self.lgb_mu_.predict(X)
        mu = np.maximum(mu, 1e-10)
        phi = self.lgb_phi_.predict(X)
        phi = np.clip(phi, 1e-8, 1e6)
        return pi, mu, phi