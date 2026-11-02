"""Run the full pipeline and write every figure and number used in the report.

Usage: python main.py
"""

import json
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import data_loader as dl
from src import eda, evaluation, models, sustainability
from src.config import FEATURES, FIG_DIR, RANDOM_STATE, RESULTS_DIR, TEST_SIZE


def section(title):
    """Print a section header in the console."""
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def main():
    FIG_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    metrics = {}

    section("1. Loading data with memory optimisation")
    mem = dl.memory_comparison()
    print(f"DataFrame memory : {mem['default_total_bytes']:,} B -> "
          f"{mem['optimised_total_bytes']:,} B  (-{mem['reduction_pct']} %)")
    print(f"tracemalloc peak : {mem['default_peak_bytes']:,} B -> "
          f"{mem['optimised_peak_bytes']:,} B  (-{mem['peak_reduction_pct']} %)")
    print(f"max value change from float32: {mem['max_abs_value_difference']:.2e}")
    # Table A1 in the report appendix
    mem["per_column"].to_csv(RESULTS_DIR / "memory_columns.csv")
    metrics["memory"] = {k: v for k, v in mem.items() if k != "per_column"}

    df = dl.load_optimised()
    # chunked means must equal in-memory means
    chunk_means = dl.chunked_crop_means(chunksize=500)
    full_means = df.groupby(df["label"].astype(str))[FEATURES].mean()
    same = np.allclose(chunk_means.sort_index().to_numpy(),
                       full_means.sort_index().to_numpy(), atol=1e-3)
    print(f"chunked (500 rows) per-crop means equal full-file means: {same}")
    metrics["memory"]["chunked_means_match"] = bool(same)

    section("2. Exploratory data analysis")
    info = eda.run_eda(df)
    print(info["summary"])
    print("strongest correlation:", info["strongest_correlation"])
    print("crops from lowest to highest input needs:", info["cheapest_to_most_expensive"])
    metrics["eda"] = {
        "rows": int(len(df)), "features": len(FEATURES),
        "classes": int(df["label"].nunique()),
        "missing_values": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "strongest_correlation": {"pair": list(info["strongest_correlation"]["pair"]),
                                  "r": info["strongest_correlation"]["r"]},
        "cheapest_to_most_expensive": info["cheapest_to_most_expensive"],
    }
    info["summary"].to_csv(RESULTS_DIR / "summary_statistics.csv")
    info["profile"].to_csv(RESULTS_DIR / "crop_resource_profile.csv")

    section("3. Train / test split")
    X, y = dl.to_model_arrays(df)
    X64 = df[FEATURES].to_numpy(dtype=np.float64)    # only to measure the saving
    metrics["memory"]["X_float64_bytes"] = int(X64.nbytes)
    metrics["memory"]["X_float32_bytes"] = int(X.nbytes)
    del X64
    # stratify keeps 20 samples of every crop in the test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    print(f"train {X_train.shape}, test {X_test.shape}")

    section("4. Model selection (grid search, 5-fold stratified CV)")
    results, fitted = models.tune_and_compare(X_train, y_train, X_test, y_test)
    print(results.drop(columns="best_params").round(4).to_string())
    results.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    # choose by CV accuracy to avoid selection bias
    best_name = models.pick_best(results)
    best = fitted[best_name]
    print("selected model:", best_name)
    metrics["models"] = results.assign(best_params=results["best_params"].astype(str)) \
                               .round(4).to_dict(orient="records")
    metrics["selected_model"] = best_name
    evaluation.plot_model_comparison(results)

    section("5. Detailed evaluation of the selected model")
    report, errors = evaluation.confusion_and_report(best, X_test, y_test)
    report.to_csv(RESULTS_DIR / "classification_report.csv")
    imp = evaluation.feature_importance(best, X_test, y_test)
    imp.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
    top3 = evaluation.topk(best, X_test, y_test, k=3)
    print(f"test accuracy {report.loc['accuracy'].iloc[0]:.4f}, top-3 accuracy {top3:.4f}")
    print("misclassified (true -> predicted):", errors)
    print(imp.round(4).to_string(index=False))

    top3_models = results["model"].head(3).tolist()
    noise = evaluation.noise_robustness({m: fitted[m] for m in top3_models},
                                        X_train, X_test, y_test)
    noise.to_csv(RESULTS_DIR / "noise_robustness.csv", index=False)
    print(noise.pivot(index="noise_pct", columns="model", values="accuracy").round(4))

    abl = evaluation.ablation(best, X_train, y_train)
    for k, (m, s) in abl.items():
        print(f"ablation - {k}: {m:.4f} ± {s:.4f}")

    metrics["evaluation"] = {
        "test_accuracy": float(report.loc["accuracy"].iloc[0]),
        "macro_precision": float(report.loc["macro avg", "precision"]),
        "macro_recall": float(report.loc["macro avg", "recall"]),
        "macro_f1": float(report.loc["macro avg", "f1-score"]),
        "top3_accuracy": float(top3),
        "misclassified": [list(e) for e in errors],
        "lowest_f1_classes": report.drop(["accuracy", "macro avg", "weighted avg"])
                                   ["f1-score"].sort_values().head(4).round(3).to_dict(),
        "feature_importance": imp.round(4).to_dict(orient="records"),
        "noise": noise.round(4).to_dict(orient="records"),
        "ablation": {k: [round(m, 4), round(s, 4)] for k, (m, s) in abl.items()},
    }

    section("6. Resource-aware re-ranking (sustainability layer)")
    # requirement profile from training rows only
    train_df = pd.DataFrame(X_train, columns=FEATURES).assign(label=y_train)
    profile = eda.crop_resource_profile(train_df)
    rec = sustainability.ResourceAwareRecommender(best, profile).fit_scales(X_train)
    rr = sustainability.evaluate_reranking(rec, X_test, y_test)
    X_noisy = sustainability.noisy_fields(X_test, X_train, level=20)
    rr_noisy = sustainability.evaluate_reranking(rec, X_noisy, y_test)
    arid = sustainability.arid_scenario(rec, X_test, y_test, rain_factor=0.5)
    for label, res in [("clean test", rr), ("noisy test (20 %)", rr_noisy),
                       ("arid (rain x0.5)", arid)]:
        print(label, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()})

    noisy_curve = sustainability.alpha_tradeoff(rec, X_noisy)
    Xd = X_test.copy()
    Xd[:, FEATURES.index("rainfall")] *= 0.5
    arid_curve = sustainability.alpha_tradeoff(rec, Xd)
    sustainability.plot_tradeoff({"Noisy fields (20 %)": noisy_curve,
                                  "Fields with 50 % less rain": arid_curve})
    pd.concat([noisy_curve.assign(scenario="noisy"),
               arid_curve.assign(scenario="arid")]) \
        .to_csv(RESULTS_DIR / "tradeoff.csv", index=False)

    # how often the re-ranker acts with each of the top-3 models
    activity = {}
    for m in top3_models:
        r_m = sustainability.ResourceAwareRecommender(fitted[m], profile).fit_scales(X_train)
        a_m = sustainability.arid_scenario(r_m, X_test, y_test, rain_factor=0.5)
        activity[m] = {"arid_changed_pct": round(a_m["changed_pct"], 2),
                       "arid_water_gap_change_pct": round(
                           100 * (a_m["new_water_gap"] / a_m["base_water_gap"] - 1), 1),
                       "test_mean_top1_prob": round(float(
                           fitted[m].predict_proba(X_test).max(axis=1).mean()), 3)}
    print("re-ranker activity by model:", activity)

    # worked example: a hot, dry, low-nutrient field
    example = [40, 35, 30, 30.0, 35.0, 7.6, 40.0]   # N, P, K, T, RH, pH, rain
    table = rec.explain(example)
    print("example field:", dict(zip(FEATURES, example)))
    print(table.to_string(index=False))
    print("plain model ->", best.predict(np.array([example], dtype=np.float32))[0],
          "| resource-aware ->", rec.recommend(np.array([example], dtype=np.float32))[0])

    # first arid test field where the decision changes
    base_idx = best.predict_proba(Xd).argmax(axis=1)
    new_idx = rec.recommend_index(Xd)
    i = int(np.flatnonzero(base_idx != new_idx)[0])
    changed_field = [round(float(v), 1) for v in Xd[i]]
    changed_table = rec.explain(Xd[i])
    print("arid field where the decision changes:", dict(zip(FEATURES, changed_field)))
    print(changed_table.to_string(index=False))

    metrics["sustainability"] = {
        "changed_example_field": dict(zip(FEATURES, changed_field)),
        "changed_example_table": changed_table.to_dict(orient="records"),
        "settings": {"alpha": rec.alpha, "top_k": rec.top_k, "min_prob": rec.min_prob,
                     "fert_scale": round(rec.fert_scale, 2),
                     "water_scale": round(rec.water_scale, 2)},
        "test": rr, "noisy": rr_noisy, "arid": arid,
        "activity_by_model": activity,
        "tradeoff_noisy": noisy_curve.round(4).to_dict(orient="records"),
        "tradeoff_arid": arid_curve.round(4).to_dict(orient="records"),
        "example_field": dict(zip(FEATURES, example)),
        "example_table": table.to_dict(orient="records"),
        "example_plain": str(best.predict(np.array([example], dtype=np.float32))[0]),
        "example_aware": str(rec.recommend(np.array([example], dtype=np.float32))[0]),
    }

    section("7. Saving the final model (compressed)")
    # measure the uncompressed size in a temporary folder
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "raw.joblib")
        joblib.dump(best, raw)
        raw_size = os.path.getsize(raw)
    final_path = RESULTS_DIR / "final_model.joblib"
    joblib.dump(best, final_path, compress=3)          # zlib level 3
    comp_size = os.path.getsize(final_path)
    print(f"model file: {raw_size:,} B -> {comp_size:,} B compressed")
    metrics["memory"]["model_raw_bytes"] = int(raw_size)
    metrics["memory"]["model_compressed_bytes"] = int(comp_size)

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print("\nAll results written to", RESULTS_DIR)


if __name__ == "__main__":
    main()
