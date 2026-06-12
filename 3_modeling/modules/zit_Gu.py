"""
ZITboostGu — Gu (2024, arXiv:2405.14990) 논문 Algorithm 1 충실 구현.

논문 제목: "Dispersion Modeling in Zero-inflated Tweedie Models
           with Applications to Insurance Claim Data Analysis"
저자: Yuwen Gu

기존 ZITboostEQLRegressor(zit.py)와 달리, 논문과 다른 점을 정확히 두 가지 교정한다:
  ① 초기화: 스칼라 μ₀/φ₀/π₀를 구한 뒤, 각각에 대해 GBT F̂^(0)를 별도로 적합한 후 EM 시작
             (기존: 스칼라를 첫 E-step에 그대로 써 첫 M-step이 F^(0) 역할을 하는 근사)
  ② φ M-step: 논문 eq.(8) gamma regression의 response는 w_i·D_ζ(y_i;μ̂^(k+1)), weight=(1-Π_i)
               LightGBM gamma objective는 log-link log(φ)를 최적화 → 논문 log-link와 일치.
               (기존도 같은 구조이나, 초기화 GBT 생략이 첫 iter φ에 영향을 줬음)

나머지 (E-step posterior, π=cross-entropy soft-label, μ=weighted Tweedie deviance, predict=(1-π)μ,
score_loglik = EQL 로그우도, Tweedie unit deviance) 는 논문과 이미 일치하므로 그대로 유지.

논문과 완전히 동일한 지점 목록:
  - E-step: Π_i = π/(π+(1-π)·exp(-w·μ^(2-ζ)/(φ(2-ζ)))) · I(y_i=0)
  - M-step π: cross-entropy, target=Π_i, weight=1 (eq.6)
  - M-step μ: Tweedie deviance, weight=(1-Π_i)·w_i/φ̂ (eq.7)
  - M-step φ: gamma, response=w_i·D_ζ(y_i;μ̂), weight=(1-Π_i) (eq.8)
  - 초기화 스칼라: μ₀=Σ_{y>0}w_iy_i/Σ_{y>0}w_i, φ₀=Σ_{y>0}w_i·D_ζ(y_i;μ₀)/n_pos
  - 초기화 GBT: F̂^(0)_π/μ/φ 각각을 스칼라 초기값으로 적합 (Algorithm 1 step 1)
  - predict: E[Y|x] = (1-π(x))·μ(x)
  - ζ profile likelihood: score_loglik으로 외부에서 수행 (Algorithm 2)

논문과 여전히 다른/근사인 지점 (데이터 특성 또는 논문 미명시로 해결 불가):
  - w_i=1 고정: 노출량(exposure) 없는 데이터 특수화 (논문 일반식의 w_i=1 특수화)
  - ζ 그리드 간격: 논문 미명시, 노트북에서 임의 설정
  - 로그우도 c(·) 항: exact Tweedie 정규화 불가 → saddlepoint 근사 (논문도 EQL 근사 사용)
  - 수렴 조건·K: 논문 미명시 → 프로젝트 자체 설정

사용법:
    from modules.zit_Gu import ZITboostGu

    model = ZITboostGu(zeta=1.5, n_em_iters=10)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pi, mu, phi = model.predict_components(X_test)
    loglik = model.score_loglik(X_val, y_val)   # Algorithm 2용 ζ 비교
"""

import numpy as np
import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin

from utils.config import SEED


# ---------------------------------------------------------------------------
# Tweedie 분포 수학 헬퍼
# ---------------------------------------------------------------------------

def _tweedie_p0(mu, phi, zeta):
    """Tweedie 상태에서 Y=0일 확률 exp(-λ), λ = μ^(2-ζ) / ((2-ζ)·φ)."""
    mu = np.maximum(mu, 1e-10)
    phi = np.maximum(phi, 1e-10)
    lam = np.power(mu, 2.0 - zeta) / ((2.0 - zeta) * phi)
    return np.exp(-lam)


def _tweedie_unit_deviance(y, mu, zeta):
    """Tweedie unit deviance D_ζ(y, μ) — 논문 eq.(4), Dunn & Smyth 표준형.

    D_ζ(y,μ) = 2[y·(y^(1-ζ) - μ^(1-ζ))/(1-ζ)  −  (y^(2-ζ) - μ^(2-ζ))/(2-ζ)]   (y>0)
             = 2·μ^(2-ζ)/(2-ζ)                                                    (y=0)
    1 < ζ < 2 에서 유효.
    """
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
        mu_term = -(np.power(y, 2.0 - p) - np.power(mu, 2.0 - p)) / (2.0 - p)
        # y=0이면: y^(2-p)=0이므로 mu_term = μ^(2-p)/(2-p)
        mu_term = np.where(y > 0, mu_term, np.power(mu, 2.0 - p) / (2.0 - p))
        dev = 2.0 * (y_term + mu_term)

    dev = np.nan_to_num(dev, nan=1e6, posinf=1e6, neginf=1e-8)
    return np.clip(dev, 1e-8, 1e6)


def _zitweedie_loglik(y, pi, mu, phi, zeta, w=1.0):
    """ZI-Tweedie EQL/saddlepoint 근사 로그우도 합 — 논문 eq.(5).

    y=0: log(π + (1-π)·exp(−w·μ^(2-ζ)/(φ(2-ζ))))
    y>0: log(1-π) + (w/φ)·[y·μ^(1-ζ)/(1-ζ) − μ^(2-ζ)/(2-ζ)] − ½·log(2π·φ·y^ζ/w)
         (saddlepoint 정규화 항 포함)

    Algorithm 2(ζ profile likelihood)의 목적함수로 사용.
    """
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
# ZITboostGu — Algorithm 1 충실 구현
# ---------------------------------------------------------------------------

class ZITboostGu(BaseEstimator, RegressorMixin):
    """Gu 2024(arXiv:2405.14990) Algorithm 1 충실 구현.

    Parameters
    ----------
    zeta : float
        Tweedie power ζ ∈ (1, 2).
    n_em_iters : int
        EM 반복 횟수 K.
    em_tol : float
        조기 종료 RMSE 변화 임계값.

    mu_* / pi_* / phi_* : LightGBM 하이퍼파라미터 (세 모델 각각).
    """

    def __init__(
        self,
        zeta=1.5,
        n_em_iters=10,
        em_tol=1e-7,
        # μ 모델 (Tweedie deviance — eq.7)
        mu_n_estimators=500,
        mu_learning_rate=0.05,
        mu_num_leaves=31,
        mu_max_depth=6,
        mu_min_child_samples=20,
        mu_subsample=0.8,
        mu_colsample_bytree=0.8,
        mu_reg_alpha=1e-3,
        mu_reg_lambda=1e-1,
        # π 모델 (cross-entropy soft-label — eq.6)
        pi_n_estimators=200,
        pi_learning_rate=0.05,
        pi_num_leaves=31,
        pi_max_depth=6,
        pi_min_child_samples=20,
        # φ 모델 (gamma, log-link — eq.8)
        phi_n_estimators=200,
        phi_learning_rate=0.05,
        phi_num_leaves=31,
        phi_max_depth=6,
        phi_min_child_samples=20,
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
        self.pi_n_estimators = pi_n_estimators
        self.pi_learning_rate = pi_learning_rate
        self.pi_num_leaves = pi_num_leaves
        self.pi_max_depth = pi_max_depth
        self.pi_min_child_samples = pi_min_child_samples
        self.phi_n_estimators = phi_n_estimators
        self.phi_learning_rate = phi_learning_rate
        self.phi_num_leaves = phi_num_leaves
        self.phi_max_depth = phi_max_depth
        self.phi_min_child_samples = phi_min_child_samples
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.device = device

    # --- LightGBM 파라미터 빌더 ---

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

    def _pi_params(self):
        # cross_entropy: soft label(0~1) 직접 학습, GPU 불안정하므로 CPU 고정
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
            device="cpu",
        )

    def _phi_lgb_base_params(self):
        """φ 모델의 트리 구조 파라미터 (objective 제외)."""
        return dict(
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

    @staticmethod
    def _phi_custom_obj(d_w, w_arr):
        """논문 eq.(8) shape=½ gamma loss의 gradient/hessian 생성기를 반환.

        논문 eq.(8):
          LC+(F_φ) = Σ_i (1-Π_i) · [−w_i·D_ζ(y_i;μ̂) / (2·φ(x_i)) − ½·log(φ(x_i))]

        log-link: φ = exp(F_φ)  →  ∂φ/∂F = φ

        1차 미분 (gradient, LightGBM은 loss 최소화 → 부호 반전):
          g_i = −(1-Π_i) · [−w_i·D_i/(2φ_i) − ½]·φ_i / φ_i
              = (1-Π_i) · [½ − w_i·D_i/(2φ_i)]

          단, (1-Π_i)·w_i·D_i 는 d_w[i]로, (1-Π_i) 는 w_arr[i]로 전달.

        2차 미분 (hessian, 양수 보장):
          h_i = (1-Π_i) · w_i·D_i / (2φ_i)
              = d_w[i] / (2φ_i)

        Parameters
        ----------
        d_w : ndarray — (1-Π_i)·w_i·D_ζ(y_i;μ̂)   (고정값, closure로 캡처)
        w_arr : ndarray — (1-Π_i)                    (고정값, closure로 캡처)

        Returns
        -------
        callable : (preds, dataset) → (grad, hess) — LightGBM custom objective 형식
        """
        def _obj(preds, dataset):
            phi = np.exp(preds)                          # log-link 역변환
            phi = np.maximum(phi, 1e-10)
            grad = w_arr * 0.5 - d_w / (2.0 * phi)      # (1-Π)·½ − (1-Π)·w·D/(2φ)
            hess = np.maximum(d_w / (2.0 * phi), 1e-8)  # 양수 클램프 (GBDT 안정성)
            return grad, hess
        return _obj

    def _fit_lgb_phi(self, X, d_w, w_arr):
        """논문 eq.(8) shape=½ custom objective로 φ 모델 적합.

        LightGBM custom objective는 LGBMRegressor 대신 lgb.train()으로 직접 호출한다.
        초기 예측값(init_score)은 log(mean(d_w / w_arr))로 설정해 log-link 공간에서 시작.

        퇴화 split 발생 시 min_child_samples↑·max_depth cap으로 재적합 (수치 안정성).
        """
        base = self._phi_lgb_base_params()

        # log-link 초기값: φ₀ = mean(D_ζ) (상수 예측) → F_φ₀ = log(φ₀)
        d_mean = float(np.mean(d_w / np.maximum(w_arr, 1e-10)))
        init_score = np.full(len(X), np.log(max(d_mean, 1e-10)))

        params = dict(
            num_leaves=base["num_leaves"],
            max_depth=base["max_depth"] if base["max_depth"] and base["max_depth"] > 0 else -1,
            min_child_samples=base["min_child_samples"],
            learning_rate=base["learning_rate"],
            n_estimators=base["n_estimators"],
            random_state=base["random_state"],
            n_jobs=base["n_jobs"],
            verbose=base["verbose"],
            device=base["device"],
        )

        def _train(p):
            ds = lgb.Dataset(X, init_score=init_score, free_raw_data=False)
            # LightGBM 4.x: train()에서 fobj 인자가 제거됨 → custom objective는
            # params['objective']에 callable로 전달 (시그니처·반환값은 옛 fobj와 동일).
            train_params = {k: v for k, v in p.items()
                            if k not in ("n_estimators", "random_state", "n_jobs", "verbose")}
            train_params |= {"seed": p["random_state"], "num_threads": p["n_jobs"],
                             "verbosity": -1, "num_iterations": p["n_estimators"],
                             "objective": self._phi_custom_obj(d_w, w_arr)}
            booster = lgb.train(params=train_params, train_set=ds)
            return booster

        try:
            booster = _train(params)
        except lgb.basic.LightGBMError:
            safe = dict(params)
            safe["min_child_samples"] = max(int(params.get("min_child_samples", 20) or 20), 100)
            md = params.get("max_depth", -1)
            safe["max_depth"] = 8 if (md is None or md <= 0) else min(int(md), 8)
            booster = _train(safe)

        return booster

    # --- Algorithm 1 초기화 (논문 Step 1) ---

    def _initialize(self, X, y, w=1.0):
        """논문 Algorithm 1 Step 1: 스칼라 추정 → 초기 GBT F̂^(0) 적합.

        스칼라 추정 (논문 eq. 아래 설명):
          μ₀ = Σ_{y>0} w_i·y_i / Σ_{y>0} w_i        (zero-truncated weighted mean)
          φ₀ = Σ_{y>0} w_i·D_ζ(y_i;μ₀) / n_pos      (zero-truncated deviance moment)
          π₀ = n_zero / n                              (zero 비율, [0.01,0.99] 클립)

        초기 GBT (기존 ZITboostEQLRegressor에서 생략됐던 단계):
          F̂^(0)_μ : target=y, weight=(1-π₀)/φ₀ (스칼라)  — eq.(7)와 같은 구조
          F̂^(0)_φ : target=D_ζ(y;μ₀), weight=(1-π₀)      — eq.(8)와 같은 구조
          F̂^(0)_π : target=π₀ (상수), weight=1             — eq.(6)와 같은 구조
        """
        n = len(y)
        is_zero = (y == 0)
        is_pos = ~is_zero
        n_pos = int(is_pos.sum())

        # 스칼라 μ₀
        if n_pos > 0:
            mu0_scalar = float(np.sum(y[is_pos]) / n_pos)   # w_i=1이므로 단순 평균
        else:
            mu0_scalar = 1e-4
        mu0_scalar = max(mu0_scalar, 1e-10)

        # 스칼라 φ₀ — zero-truncated deviance moment
        if n_pos > 0:
            dev_pos = _tweedie_unit_deviance(y[is_pos], np.full(n_pos, mu0_scalar), self.zeta)
            phi0_scalar = float(np.clip(dev_pos.sum() / n_pos, 1e-6, 1e6))
        else:
            phi0_scalar = 1.0

        # 스칼라 π₀
        pi0_scalar = float(np.clip(is_zero.mean(), 0.01, 0.99))

        # 초기 GBT F̂^(0): 각 스칼라 초기값을 target으로 GBT 적합
        # F̂^(0)_π : constant target π₀
        init_pi_target = np.full(n, pi0_scalar)
        lgb_pi0 = lgb.LGBMRegressor(**self._pi_params())
        lgb_pi0.fit(X, init_pi_target)
        pi_arr = np.clip(lgb_pi0.predict(X), 1e-8, 1 - 1e-8)

        # F̂^(0)_μ : target=y, weight=(1-π₀)/φ₀ — eq.(7) 구조
        mu_w0 = np.full(n, (1.0 - pi0_scalar) / phi0_scalar)
        lgb_mu0 = lgb.LGBMRegressor(**self._mu_params())
        lgb_mu0.fit(X, y, sample_weight=mu_w0)
        mu_arr = np.maximum(lgb_mu0.predict(X), 1e-10)

        # F̂^(0)_φ : d_w=(1-π₀)·D_ζ(y;μ₀), w_arr=(1-π₀) — eq.(8) 구조
        dev0 = _tweedie_unit_deviance(y, np.full(n, mu0_scalar), self.zeta)
        phi_w0 = np.full(n, 1.0 - pi0_scalar)
        booster_phi0 = self._fit_lgb_phi(X, phi_w0 * dev0, phi_w0)
        phi_arr = np.clip(np.exp(booster_phi0.predict(X)), 1e-8, 1e6)

        return pi_arr, mu_arr, phi_arr

    # --- E-step (논문 eq. E-step) ---

    def _e_step(self, y, pi_arr, mu_arr, phi_arr, w=1.0):
        """E-step: structural zero 사후확률 Π_i 계산.

        Π_i = π_i / (π_i + (1-π_i)·exp(−w·μ_i^(2-ζ)/(φ_i·(2-ζ))))  · I(y_i=0)
        y>0이면 Π_i=0 (양수는 반드시 Tweedie 상태).
        """
        n = len(y)
        posterior = np.zeros(n, dtype=np.float64)
        is_zero = (y == 0)

        if is_zero.any():
            pi_z = pi_arr[is_zero]
            p0_z = _tweedie_p0(mu_arr[is_zero], phi_arr[is_zero], self.zeta)
            denom = np.maximum(pi_z + (1 - pi_z) * p0_z, 1e-15)
            posterior[is_zero] = pi_z / denom

        return np.clip(posterior, 1e-8, 1 - 1e-8)

    # --- M-step (논문 eq.6/7/8) ---

    def _m_step(self, X, y, posterior, phi_current, w=1.0):
        """M-step: Π 고정 후 세 GBT 순서대로 갱신.

        eq.(6) π : cross-entropy, target=Π, weight=1
        eq.(7) μ : Tweedie deviance, weight=(1-Π)·w/φ̂^(k)
        eq.(8) φ : shape=½ gamma custom obj(log-link), d_w=(1-Π)·w·D_ζ(y;μ̂^(k+1)), weight=(1-Π)
        """
        one_minus_Pi = 1.0 - posterior   # (1-Π_i)

        # eq.(6): π 모델
        lgb_pi = lgb.LGBMRegressor(**self._pi_params())
        lgb_pi.fit(X, posterior)
        pi_pred = np.clip(lgb_pi.predict(X), 1e-8, 1 - 1e-8)

        # eq.(7): μ 모델 — weight = (1-Π)·w/φ̂^(k)
        mu_weight = one_minus_Pi * w / np.maximum(phi_current, 1e-10)
        lgb_mu = lgb.LGBMRegressor(**self._mu_params())
        lgb_mu.fit(X, y, sample_weight=mu_weight)
        mu_pred = np.maximum(lgb_mu.predict(X), 1e-10)

        # eq.(8): φ 모델 — d_w=(1-Π)·w·D_ζ(y;μ̂^(k+1)), w_arr=(1-Π), shape=½ custom obj
        d_zeta = _tweedie_unit_deviance(y, mu_pred, self.zeta)
        d_w = one_minus_Pi * w * d_zeta
        booster_phi = self._fit_lgb_phi(X, d_w, one_minus_Pi)
        phi_pred = np.clip(np.exp(booster_phi.predict(X)), 1e-8, 1e6)

        return lgb_pi, lgb_mu, booster_phi, pi_pred, mu_pred, phi_pred

    # --- fit (Algorithm 1 전체) ---

    def fit(self, X, y, sample_weight=None):
        """Algorithm 1: 초기 GBT → EM 반복.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
        y : array-like (n_samples,)
        sample_weight : ignored (sklearn API 호환)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        # Step 1: 초기 GBT F̂^(0) 적합 (논문 Algorithm 1의 첫 단계)
        pi_arr, mu_arr, phi_arr = self._initialize(X, y)

        self.em_history_ = []
        prev_rmse = np.inf

        for em_iter in range(self.n_em_iters):
            # E-step
            posterior = self._e_step(y, pi_arr, mu_arr, phi_arr)

            # M-step
            lgb_pi, lgb_mu, lgb_phi, pi_arr, mu_arr, phi_arr = \
                self._m_step(X, y, posterior, phi_current=phi_arr)

            pred = np.clip((1 - pi_arr) * mu_arr, 0, None)
            rmse = float(np.sqrt(np.mean((y - pred) ** 2)))

            self.em_history_.append({
                "iter":           em_iter + 1,
                "rmse":           rmse,
                "pi_mean":        float(pi_arr.mean()),
                "pi_std":         float(pi_arr.std()),
                "mu_mean":        float(mu_arr.mean()),
                "phi_mean":       float(phi_arr.mean()),
                "posterior_mean": float(posterior.mean()),
            })

            if self.verbose >= 0:
                print(f"  EM iter {em_iter+1}/{self.n_em_iters}: "
                      f"RMSE={rmse:.6f}, π_mean={pi_arr.mean():.4f}, "
                      f"μ_mean={mu_arr.mean():.6f}")

            rmse_delta = prev_rmse - rmse
            if em_iter >= 2 and abs(rmse_delta) < self.em_tol:
                if self.verbose >= 0:
                    print(f"  EM early stop at iter {em_iter+1}: "
                          f"|ΔRMSE|={abs(rmse_delta):.2e} < tol={self.em_tol:.1e}")
                break
            prev_rmse = rmse

        self.n_em_iters_actual_ = em_iter + 1
        self.lgb_pi_ = lgb_pi
        self.lgb_mu_ = lgb_mu
        self.lgb_phi_ = lgb_phi
        self.fitted_ = True
        return self

    # --- 예측 ---

    def predict(self, X):
        """E[Y|x] = (1-π(x))·μ(x) — 논문 예측식."""
        pi, mu, _ = self.predict_components(X)
        return np.clip((1 - pi) * mu, 0, None)

    def predict_components(self, X):
        """π, μ, φ 각각 반환 (진단/후처리용).

        Returns
        -------
        pi  : ndarray — P(structural zero | x)
        mu  : ndarray — Tweedie mean
        phi : ndarray — dispersion
        """
        if not hasattr(self, "fitted_"):
            raise ValueError("Model not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        pi = np.clip(self.lgb_pi_.predict(X), 1e-8, 1 - 1e-8)
        mu = np.maximum(self.lgb_mu_.predict(X), 1e-10)
        # lgb_phi_는 lgb.Booster (custom obj) — F_φ를 예측하므로 exp로 역변환
        phi = np.clip(np.exp(self.lgb_phi_.predict(X)), 1e-8, 1e6)
        return pi, mu, phi

    def score_loglik(self, X, y):
        """ZI-Tweedie EQL 로그우도 합 — Algorithm 2(ζ profile likelihood)의 목적함수.

        여러 ζ 후보에 대해 fit() → score_loglik() 해서 최대값을 고르는 방식으로 ζ를 추정한다.
        """
        pi, mu, phi = self.predict_components(X)
        return _zitweedie_loglik(np.asarray(y, dtype=np.float64).ravel(), pi, mu, phi, self.zeta)
