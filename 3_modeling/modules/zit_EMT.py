"""
EMTboost — Zhou, Qian & Yang (2019, arXiv:1811.10192) 구조 충실 구현.

논문 제목: "Tweedie Gradient Boosting for Extremely Unbalanced Zero-inflated Data"

논문과의 대응:
  - μ  : f(X), LightGBM Tweedie  (논문 Algorithm 3 TDboost 대신 LightGBM 사용 — 구조 동일)
  - π  : 스칼라 q                 (논문 eq.24: q = (1/n) Σ δ₁ᵢ)
  - φ  : 스칼라                   (논문 eq.23: golden section search로 단일 값 추정)
  - E-step : 논문 eq.14/15 그대로
  - 초기화 : 논문 eq.33/34/35 그대로

논문과 다른 점 (의도적 근사):
  - μ 추정: 자체 TDboost → LightGBM Tweedie (구조는 동일, 라이브러리만 교체)
  - exposure w_i=1 고정 (논문 일반식의 w_i=1 특수화)

ZITboostGu(zit_Gu.py)와의 핵심 차이:
  - π: 스칼라 (전체 평균) vs f(X) LightGBM
  - φ: 스칼라 (golden section) vs f(X) LightGBM custom obj
  → 모든 샘플이 같은 zero 확률·분산을 공유 vs 샘플마다 다른 값

파라미터 방향 주의 (논문 vs ZITboostGu):
  논문:  q   = P(Tweedie 상태),  1-q = P(structural zero), 예측 = q·μ
  ZITGu: π   = P(structural zero), 1-π = P(Tweedie 상태), 예측 = (1-π)·μ
  → 이 파일은 논문 방향을 따름: q = Tweedie 확률, predict = q·μ

사용법:
    from modules.zit_EMT import EMTboost

    model = EMTboost(zeta=1.5, n_em_iters=10)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pi, mu, phi = model.predict_components(X_test)
    loglik = model.score_loglik(X_val, y_val)
"""

import numpy as np
import lightgbm as lgb
from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator, RegressorMixin

from utils.config import SEED


# ---------------------------------------------------------------------------
# Tweedie 분포 수학 헬퍼
# ---------------------------------------------------------------------------

def _tweedie_p0(mu, phi, zeta):
    """Tweedie 상태에서 Y=0일 확률 exp(-λ), λ = μ^(2-ζ) / ((2-ζ)·φ)."""
    mu = np.maximum(mu, 1e-10)
    phi = max(float(phi), 1e-10)
    lam = np.power(mu, 2.0 - zeta) / ((2.0 - zeta) * phi)
    return np.exp(-lam)


def _tweedie_unit_deviance(y, mu, zeta):
    """Tweedie unit deviance D_ζ(y, μ), 1 < ζ < 2."""
    y = np.asarray(y, dtype=np.float64)
    mu = np.maximum(np.asarray(mu, dtype=np.float64), 1e-10)
    y = np.maximum(y, 0.0)
    p = float(zeta)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        y_term = np.where(
            y > 0,
            y * (np.power(y, 1.0 - p) - np.power(mu, 1.0 - p)) / (1.0 - p),
            0.0,
        )
        mu_term = np.where(
            y > 0,
            -(np.power(y, 2.0 - p) - np.power(mu, 2.0 - p)) / (2.0 - p),
            np.power(mu, 2.0 - p) / (2.0 - p),
        )
        dev = 2.0 * (y_term + mu_term)

    return np.clip(np.nan_to_num(dev, nan=1e6, posinf=1e6, neginf=1e-8), 1e-8, 1e6)


def _zitweedie_loglik(y, pi, mu, phi, zeta, w=1.0):
    """ZI-Tweedie EQL/saddlepoint 근사 로그우도 합."""
    y = np.asarray(y, dtype=np.float64).ravel()
    pi = np.clip(np.asarray(pi, dtype=np.float64), 1e-8, 1 - 1e-8)
    mu = np.maximum(np.asarray(mu, dtype=np.float64), 1e-10)
    phi = np.maximum(np.asarray(phi, dtype=np.float64), 1e-10)

    is_zero = (y == 0)
    is_pos = ~is_zero
    ll = np.zeros(len(y), dtype=np.float64)

    if is_zero.any():
        piz, muz, phiz = pi[is_zero], mu[is_zero], phi[is_zero]
        lam = w * np.power(muz, 2.0 - zeta) / (phiz * (2.0 - zeta))
        ll[is_zero] = np.log(np.maximum(piz + (1 - piz) * np.exp(-lam), 1e-300))

    if is_pos.any():
        yp, pip, mup, phip = y[is_pos], pi[is_pos], mu[is_pos], phi[is_pos]
        devp = _tweedie_unit_deviance(yp, mup, zeta)
        ll[is_pos] = (
            np.log(np.maximum(1 - pip, 1e-300))
            - 0.5 * (w / phip) * devp
            - 0.5 * np.log(2.0 * np.pi * phip * np.power(yp, zeta) / w)
        )

    return float(ll.sum())


# ---------------------------------------------------------------------------
# φ 스칼라 추정 — 논문 eq.(23) golden section search
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


# ---------------------------------------------------------------------------
# EMTboost
# ---------------------------------------------------------------------------

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
