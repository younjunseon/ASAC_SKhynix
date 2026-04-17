"""
공통 시각화 모듈 -- baseline / ensemble / reproduce 노트북에서 공유.

사용법:
    from modules.viz import plot_optuna_results, plot_shap_analysis, ...

shap / optuna 는 런타임 import (설치 안 된 환경에서도 모듈 로드 가능).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================================
# 1. Optuna 결과 시각화
# =====================================================================
def plot_optuna_results(study, title_prefix="", show_slice=True):
    """Main study: optimization history + param importance + slice plot.

    Parameters
    ----------
    study : optuna.Study
    title_prefix : str
    show_slice : bool
    """
    import optuna.visualization as ov

    fig_hist = ov.plot_optimization_history(study)
    fig_hist.update_layout(title=f"{title_prefix}Optimization History")
    fig_hist.show()

    try:
        fig_imp = ov.plot_param_importances(study)
        fig_imp.update_layout(title=f"{title_prefix}Param Importances")
        fig_imp.show()
    except Exception as e:
        print(f"[plot_optuna_results] param importance skipped: {e}")

    if show_slice:
        try:
            fig_s = ov.plot_slice(study)
            fig_s.update_layout(title=f"{title_prefix}Slice Plot")
            fig_s.show()
        except Exception as e:
            print(f"[plot_optuna_results] slice plot skipped: {e}")


def plot_optuna_substudy(substudies, model_names=None):
    """Sub-study optimization history (position weight 등).

    Parameters
    ----------
    substudies : dict {name: Study} or list[Study]
    model_names : list[str]  (substudies가 list일 때)
    """
    import optuna.visualization as ov

    if isinstance(substudies, dict):
        items = substudies.items()
    else:
        names = model_names or [f"model_{i}" for i in range(len(substudies))]
        items = zip(names, substudies)

    for name, sub in items:
        if sub is None:
            continue
        fig = ov.plot_optimization_history(sub)
        fig.update_layout(title=f"Sub-study -- {name}")
        fig.show()


# =====================================================================
# 2. SHAP 분석
# =====================================================================
def plot_shap_analysis(models, X, feature_names, model_names=None,
                       max_display=20, n_sample=1000, seed=42):
    """모델별 SHAP beeswarm + bar (mean|SHAP|).

    Parameters
    ----------
    models : list[model]
    X : np.ndarray  (n_samples, n_features)
    feature_names : list[str]
    model_names : list[str]
    max_display : int
    n_sample : int
    seed : int
    """
    import shap

    if model_names is None:
        model_names = [f"model_{i}" for i in range(len(models))]

    if len(X) > n_sample:
        idx = np.random.default_rng(seed).choice(len(X), n_sample, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    n_models = len(models)
    fig, axes = plt.subplots(n_models, 2, figsize=(20, 5 * n_models), squeeze=False)

    for row, (model, mname) in enumerate(zip(models, model_names)):
        if model is None:
            continue
        is_tree = any(kw in type(model).__name__ for kw in
                      ["LGBM", "XGB", "CatBoost", "Forest", "Tree"])
        explainer = shap.TreeExplainer(model) if is_tree else shap.Explainer(model, X_sample)
        X_df = pd.DataFrame(X_sample, columns=feature_names)
        shap_values = explainer(X_df)

        plt.sca(axes[row, 0])
        shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
        axes[row, 0].set_title(f"{mname} -- beeswarm (top {max_display})")

        plt.sca(axes[row, 1])
        shap.plots.bar(shap_values, max_display=max_display, show=False)
        axes[row, 1].set_title(f"{mname} -- mean |SHAP|")

    plt.tight_layout()
    plt.show()
    return fig


# =====================================================================
# 3. Feature Importance 비교
# =====================================================================
def plot_fi_comparison(df_imp, cols_labels, top_k=20):
    """다중 모델 FI Top-K bar + 겹침 분석 + 랭킹 산점도.

    Parameters
    ----------
    df_imp : pd.DataFrame  ('feature' + importance columns)
    cols_labels : list[tuple(col, label)]
        e.g. [('lgbm_gain', 'LGBM'), ('et_impurity', 'ET')]
    top_k : int
    """
    from itertools import combinations
    from scipy.stats import spearmanr

    n = len(cols_labels)

    fig, axes = plt.subplots(2, max(n, 2), figsize=(6 * max(n, 2), 10), squeeze=False)

    # Row 1: Top-K bar per model
    for ax, (col, label) in zip(axes[0], cols_labels):
        top = df_imp.nlargest(top_k, col)
        ax.barh(range(top_k), top[col].values, color="steelblue")
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(top["feature"].values, fontsize=8)
        ax.set_title(f"Top {top_k}: {label}")
        ax.invert_yaxis()
    for j in range(n, axes.shape[1]):
        axes[0, j].set_visible(False)

    # Row 2 left: overlap counts
    topk_sets = {}
    for col, label in cols_labels:
        topk_sets[label] = set(df_imp.nlargest(top_k, col)["feature"].tolist())

    pairs = list(combinations(topk_sets.keys(), 2))
    overlap_data = {f"{a} & {b}": len(topk_sets[a] & topk_sets[b]) for a, b in pairs}
    all_inter = set.intersection(*topk_sets.values()) if len(topk_sets) > 1 else set()
    overlap_data[f"ALL {len(topk_sets)}"] = len(all_inter)

    ax_ov = axes[1, 0]
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"][:len(overlap_data)]
    ax_ov.barh(list(overlap_data.keys()), list(overlap_data.values()), color=colors)
    for i, (k, v) in enumerate(overlap_data.items()):
        ax_ov.text(v + 0.3, i, str(v), va="center", fontsize=11)
    ax_ov.set_title(f"Top-{top_k} Feature overlap")
    ax_ov.set_xlabel("shared features")

    # Row 2 right: ranking scatter
    ax_sc = axes[1, 1]
    rank_cols = []
    for col, label in cols_labels:
        rcol = f"_rank_{label}"
        df_imp[rcol] = df_imp[col].rank(ascending=False)
        rank_cols.append((rcol, label))

    scatter_colors = ["steelblue", "coral", "forestgreen", "orchid"]
    ref_rcol, ref_label = rank_cols[0]
    for i, (rcol, label) in enumerate(rank_cols[1:]):
        ax_sc.scatter(df_imp[ref_rcol], df_imp[rcol], alpha=0.3, s=10,
                      color=scatter_colors[i % len(scatter_colors)],
                      label=f"{ref_label} vs {label}")
    ax_sc.plot([0, len(df_imp)], [0, len(df_imp)], "k--", alpha=0.3, lw=1)
    ax_sc.set_xlabel(f"{ref_label} rank")
    ax_sc.set_ylabel("other model rank")
    ax_sc.set_title("FI ranking scatter")
    ax_sc.legend()

    lines = ["Spearman:"]
    for (rcol_a, lab_a), (rcol_b, lab_b) in combinations(rank_cols, 2):
        r, _ = spearmanr(df_imp[rcol_a], df_imp[rcol_b])
        lines.append(f"  {lab_a}-{lab_b}: {r:.3f}")
    ax_sc.text(0.02, 0.98, "\n".join(lines), transform=ax_sc.transAxes,
               va="top", fontsize=9, family="monospace",
               bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    for j in range(2, axes.shape[1]):
        axes[1, j].set_visible(False)

    for rcol, _ in rank_cols:
        df_imp.drop(columns=rcol, inplace=True, errors="ignore")

    plt.tight_layout()
    plt.show()

    print(f"Top-{top_k} intersection ({len(topk_sets)} models): {len(all_inter)}")
    return fig


# =====================================================================
# 4. Stage 1 Clf 오차행렬 + PR 곡선
# =====================================================================
def plot_clf_confusion_pr(y_true_bin, proba, threshold=0.5):
    """Confusion matrix + Precision-Recall curve.

    Parameters
    ----------
    y_true_bin : array-like  (0/1)
    proba : array-like
    threshold : float
    """
    from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                                 precision_recall_curve, roc_auc_score,
                                 average_precision_score, f1_score)

    pred_bin = (np.asarray(proba) >= threshold).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm = confusion_matrix(y_true_bin, pred_bin)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Y=0", "Y>0"])
    disp.plot(ax=axes[0], cmap="Blues", values_format="d")
    axes[0].set_title(f"Confusion Matrix (thr={threshold})")

    prec, rec, thresholds = precision_recall_curve(y_true_bin, proba)
    axes[1].plot(rec, prec, color="darkorange", lw=2)
    axes[1].fill_between(rec, prec, alpha=0.15, color="darkorange")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])

    f1_arr = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    best_idx = np.argmax(f1_arr)
    axes[1].scatter(rec[best_idx], prec[best_idx], color="red", s=80, zorder=5,
                    label=f"Best F1={f1_arr[best_idx]:.3f} (thr={thresholds[best_idx]:.3f})")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    auc = roc_auc_score(y_true_bin, proba)
    ap = average_precision_score(y_true_bin, proba)
    f1 = f1_score(y_true_bin, pred_bin, zero_division=0)
    print(f"AUC={auc:.4f}  AP={ap:.4f}  F1={f1:.4f} (thr={threshold})")
    return fig


# =====================================================================
# 5. y_true vs y_pred 히스토그램
# =====================================================================
def plot_pred_histogram(splits, target_col="health", log_scale=True,
                        show_positive_only=True):
    """train/val/test: y_true vs y_pred overlay histogram.

    Parameters
    ----------
    splits : dict {split_name: (y_true_unit, y_pred_unit)}
    target_col : str
    log_scale : bool
    show_positive_only : bool  (Y>0 subset panel)
    """
    n = len(splits)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (split_name, (y_true, y_pred)) in zip(axes, splits.items()):
        vmax = max(y_true.max(), y_pred.max()) * 1.05
        bins = np.linspace(0, vmax if vmax > 0 else 1, 80)
        ax.hist(y_true, bins=bins, alpha=0.5, color="steelblue", label="y_true", density=True)
        ax.hist(y_pred, bins=bins, alpha=0.5, color="coral", label="y_pred", density=True)
        ax.set_title(f"{split_name} -- y_true vs y_pred")
        ax.set_xlabel(target_col)
        ax.set_ylabel("density")
        ax.legend()
        if log_scale:
            ax.set_yscale("log")

    plt.tight_layout()
    plt.show()

    if show_positive_only:
        fig2, axes2 = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes2 = [axes2]
        for ax, (split_name, (y_true, y_pred)) in zip(axes2, splits.items()):
            mask = y_true > 0
            if mask.sum() == 0:
                ax.set_title(f"{split_name} -- no Y>0")
                continue
            bins = np.linspace(0, y_true[mask].max() * 1.05, 60)
            ax.hist(y_true[mask], bins=bins, alpha=0.5, color="steelblue", label="y_true (Y>0)", density=True)
            ax.hist(y_pred[mask], bins=bins, alpha=0.5, color="coral", label="y_pred (Y>0)", density=True)
            ax.set_title(f"{split_name} -- Y>0 subset")
            ax.set_xlabel(target_col)
            ax.set_ylabel("density")
            ax.legend()
        plt.tight_layout()
        plt.show()

    return fig
