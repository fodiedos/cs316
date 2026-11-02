"""Resource-aware re-ranking: prefer suitable crops that need less fertiliser and water.

Score = ALPHA * probability - (1 - ALPHA) * resource cost, over the model's
top-k crops. ALPHA = 1 gives the plain classifier.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ALPHA, FEATURES, FIG_DIR, MIN_PROB, NUTRIENTS, TOP_K

NPK_IDX = [FEATURES.index(n) for n in NUTRIENTS]
RAIN_IDX = FEATURES.index("rainfall")


class ResourceAwareRecommender:
    """Wraps a fitted classifier and re-ranks its suggestions by resource cost."""

    def __init__(self, model, profile, alpha=ALPHA, top_k=TOP_K, min_prob=MIN_PROB):
        self.model = model
        self.alpha = alpha
        self.top_k = top_k
        self.min_prob = min_prob
        # needs must follow the same class order as model.classes_
        prof = profile.loc[list(model.classes_)]
        self.need_npk = prof[NUTRIENTS].to_numpy(dtype=np.float32)   # (crops, 3)
        self.need_rain = prof["rainfall"].to_numpy(dtype=np.float32)  # (crops,)
        self.fert_scale = 1.0
        self.water_scale = 1.0

    def gaps(self, X):
        """Fertiliser and water shortfall for every (field, crop) pair."""
        # Only shortfalls count: surplus nutrients or rain cost nothing.
        soil = X[:, NPK_IDX][:, None, :]                       # (n, 1, 3)
        fert = np.clip(self.need_npk[None, :, :] - soil, 0, None).sum(axis=2)
        water = np.clip(self.need_rain[None, :] - X[:, [RAIN_IDX]], 0, None)
        return fert, water

    def fit_scales(self, X_train):
        """Set gap scales from training data (95th percentile, robust to outliers)."""
        fert, water = self.gaps(X_train)
        self.fert_scale = float(np.percentile(fert, 95)) or 1.0
        self.water_scale = float(np.percentile(water, 95)) or 1.0
        return self

    def cost(self, X):
        """Resource cost in 0-1, half fertiliser and half water."""
        fert, water = self.gaps(X)
        return 0.5 * np.clip(fert / self.fert_scale, 0, 1) + \
               0.5 * np.clip(water / self.water_scale, 0, 1)

    def recommend_index(self, X):
        """Index into model.classes_ of the chosen crop for each field."""
        proba = self.model.predict_proba(X)                     # (n, crops)
        cost = self.cost(X)
        order = np.argsort(-proba, axis=1)
        ranks = np.empty_like(order)
        rows = np.arange(len(X))[:, None]
        ranks[rows, order] = np.arange(proba.shape[1])[None, :]
        # candidates: top-k with enough probability; top-1 always allowed
        candidate = (ranks < self.top_k) & (proba >= self.min_prob)
        candidate[np.arange(len(X)), order[:, 0]] = True
        score = self.alpha * proba - (1 - self.alpha) * cost
        score[~candidate] = -np.inf
        return score.argmax(axis=1)

    def recommend(self, X):
        """Crop names chosen by the resource-aware rule."""
        return self.model.classes_[self.recommend_index(X)]

    def explain(self, x_row):
        """Table of probability, gaps, cost and score for one field's top crops."""
        X = np.asarray(x_row, dtype=np.float32).reshape(1, -1)
        proba = self.model.predict_proba(X)[0]
        fert, water = self.gaps(X)
        cost = self.cost(X)[0]
        table = pd.DataFrame({
            "crop": self.model.classes_,
            "probability": proba,
            "fertiliser_gap": fert[0],
            "water_gap_mm": water[0],
            "resource_cost": cost,
            "score": self.alpha * proba - (1 - self.alpha) * cost,
        }).sort_values("probability", ascending=False).head(self.top_k)
        return table.round(3).reset_index(drop=True)


def evaluate_reranking(rec, X_test, y_test):
    """Compare plain top-1 and resource-aware choices: gaps, cost and changes."""
    proba = rec.model.predict_proba(X_test)
    base_idx = proba.argmax(axis=1)
    new_idx = rec.recommend_index(X_test)
    fert, water = rec.gaps(X_test)
    cost = rec.cost(X_test)
    n = np.arange(len(X_test))
    classes = rec.model.classes_
    return {
        "base_cost": float(cost[n, base_idx].mean()),
        "new_cost": float(cost[n, new_idx].mean()),
        "changed_pct": 100 * float(np.mean(base_idx != new_idx)),
        "base_fert_gap": float(fert[n, base_idx].mean()),
        "new_fert_gap": float(fert[n, new_idx].mean()),
        "base_water_gap": float(water[n, base_idx].mean()),
        "new_water_gap": float(water[n, new_idx].mean()),
        "base_matches_label_pct": 100 * float(np.mean(classes[base_idx] == y_test)),
        "new_matches_label_pct": 100 * float(np.mean(classes[new_idx] == y_test)),
        "base_mean_probability": float(proba[n, base_idx].mean()),
        "new_mean_probability": float(proba[n, new_idx].mean()),
    }


def alpha_tradeoff(rec, X, alphas=np.linspace(0, 1, 11)):
    """Mean suitability and resource cost of the chosen crop for each alpha."""
    saved = rec.alpha
    proba = rec.model.predict_proba(X)
    cost = rec.cost(X)
    n = np.arange(len(X))
    rows = []
    for a in alphas:
        rec.alpha = float(a)
        idx = rec.recommend_index(X)
        rows.append({"alpha": round(float(a), 1),
                     "suitability": float(proba[n, idx].mean()),
                     "resource_cost": float(cost[n, idx].mean())})
    rec.alpha = saved
    return pd.DataFrame(rows)


def arid_scenario(rec, X_test, y_test, rain_factor=0.5):
    """Re-ranking results on the test fields with rainfall scaled by rain_factor."""
    Xd = X_test.copy()
    Xd[:, RAIN_IDX] *= rain_factor
    out = evaluate_reranking(rec, Xd, y_test)
    classes = rec.model.classes_
    out["base_top_crops"] = pd.Series(classes[rec.model.predict_proba(Xd).argmax(axis=1)]) \
        .value_counts().head(5).to_dict()
    out["new_top_crops"] = pd.Series(rec.recommend(Xd)).value_counts().head(5).to_dict()
    out["rain_factor"] = rain_factor
    return out


def noisy_fields(X_test, X_train, level=20, seed=0):
    """Test fields plus Gaussian noise (level % of each feature's train std)."""
    rng = np.random.default_rng(seed)
    std = X_train.std(axis=0)
    return X_test + rng.normal(0, 1, X_test.shape).astype(np.float32) * std * level / 100


def plot_tradeoff(curves):
    """Figure 9: suitability vs. resource cost, one curve per scenario."""
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for label, df in curves.items():
        ax.plot(df["resource_cost"], df["suitability"] * 100, marker="o", label=label)
        for a in (0.0, 0.5, 1.0):
            r = df[df["alpha"] == a].iloc[0]
            ax.annotate(f"α={a:g}", (r["resource_cost"], r["suitability"] * 100),
                        textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("Mean resource cost of recommended crop (0-1, lower is better)")
    ax.set_ylabel("Mean model probability (%)")
    ax.set_title("Suitability vs. resource cost trade-off")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig9_tradeoff.png", dpi=200)
    plt.close(fig)
