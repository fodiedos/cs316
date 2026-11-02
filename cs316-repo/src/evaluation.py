"""Evaluate the chosen model: Figures 5-8, per-class metrics and ablation."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             top_k_accuracy_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .config import CLIMATE, CV_FOLDS, FEATURES, FIG_DIR, NUTRIENTS, RANDOM_STATE


def plot_model_comparison(results):
    """Figure 5: CV and test accuracy per model."""
    r = results.sort_values("cv_accuracy_mean")
    y = np.arange(len(r))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(y - 0.2, r["cv_accuracy_mean"] * 100, height=0.4,
            xerr=r["cv_accuracy_std"] * 100, color="#3b6ea5", label="5-fold CV (train)")
    ax.barh(y + 0.2, r["test_accuracy"] * 100, height=0.4,
            color="#2a7f62", label="Held-out test")
    ax.set_yticks(y, r["model"])
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(75, 100.5)
    ax.axvline(80, color="grey", ls="--", lw=1)          # 80 % target
    ax.text(80.3, len(r) - 0.6, "80 % target", fontsize=8, color="grey")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8)
    ax.set_title("Model comparison")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_model_comparison.png", dpi=200)
    plt.close(fig)


def confusion_and_report(model, X_test, y_test):
    """Figure 6 and the per-class metrics table."""
    pred = model.predict(X_test)
    labels = np.unique(y_test)
    cm = confusion_matrix(y_test, pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, cmap="Greens", colorbar=False, xticks_rotation=75)
    ax.set_title("Confusion matrix - held-out test set (440 samples)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_confusion_matrix.png", dpi=200)
    plt.close(fig)

    report = pd.DataFrame(classification_report(y_test, pred, output_dict=True)).T

    errors = [(t, p) for t, p in zip(y_test, pred) if t != p]
    return report, errors


def feature_importance(model, X_test, y_test):
    """Figure 7: permutation importance (accuracy drop when a feature is shuffled)."""
    imp = permutation_importance(model, X_test, y_test, n_repeats=20,
                                 random_state=RANDOM_STATE, n_jobs=-1)
    s = pd.DataFrame({"feature": FEATURES,
                      "mean": imp.importances_mean,
                      "std": imp.importances_std}).sort_values("mean")
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.barh(s["feature"], s["mean"] * 100, xerr=s["std"] * 100, color="#b5651d")
    ax.set_xlabel("Accuracy drop when shuffled (percentage points)")
    ax.set_title("Permutation feature importance")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_feature_importance.png", dpi=200)
    plt.close(fig)
    return s.sort_values("mean", ascending=False)


def topk(model, X_test, y_test, k=3):
    """Share of test samples whose true crop is in the model's top k."""
    proba = model.predict_proba(X_test)
    return top_k_accuracy_score(y_test, proba, k=k, labels=model.classes_)


def noise_robustness(models, X_train, X_test, y_test, levels=(0, 5, 10, 20, 30)):
    """Figure 8: accuracy when Gaussian noise (level % of train std) is added."""
    rng = np.random.default_rng(RANDOM_STATE)
    std = X_train.std(axis=0)
    rows = []
    for level in levels:
        # same noise for every model so the comparison is fair
        noise = rng.normal(0, 1, X_test.shape).astype(np.float32) * std * level / 100
        Xn = X_test + noise
        for name, m in models.items():
            rows.append({"noise_pct": level, "model": name,
                         "accuracy": accuracy_score(y_test, m.predict(Xn))})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    sns.lineplot(data=df, x="noise_pct", y=df["accuracy"] * 100, hue="model",
                 marker="o", ax=ax)
    ax.set_xlabel("Added noise (% of feature std)")
    ax.set_ylabel("Test accuracy (%)")
    ax.axhline(80, color="grey", ls="--", lw=1)
    ax.set_title("Robustness to measurement noise")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig8_noise_robustness.png", dpi=200)
    plt.close(fig)
    return df


def ablation(model, X_train, y_train):
    """CV accuracy with all features, climate only, or soil N-P-K only."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    groups = {"All 7 features": FEATURES,
              "Climate only (T, RH, pH, rain)": CLIMATE,
              "Soil N-P-K only": NUTRIENTS}
    out = {}
    for label, cols in groups.items():
        idx = [FEATURES.index(c) for c in cols]
        scores = cross_val_score(clone(model), X_train[:, idx], y_train, cv=cv,
                                 scoring="accuracy", n_jobs=-1)
        out[label] = (scores.mean(), scores.std())
    return out
