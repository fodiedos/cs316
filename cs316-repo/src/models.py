"""Tune and compare eight classifiers with grid search and stratified CV."""

import time

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import CV_FOLDS, RANDOM_STATE


def candidate_models():
    """Return {name: (estimator, parameter grid)} for every model compared."""
    rs = RANDOM_STATE
    # Scale-sensitive models use a Pipeline so the scaler is fitted inside
    # each CV fold and never sees validation data.
    return {
        "Logistic Regression": (
            Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=3000))]),
            {"clf__C": [0.1, 1, 10, 100]},
        ),
        "Gaussian NB": (
            GaussianNB(),
            {"var_smoothing": [1e-9, 1e-8, 1e-7]},
        ),
        "k-NN": (
            Pipeline([("scale", StandardScaler()),
                      ("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": [3, 5, 7, 11], "clf__weights": ["uniform", "distance"]},
        ),
        # SVC has no native probabilities, so it is calibrated (Platt scaling)
        "SVM (RBF)": (
            Pipeline([("scale", StandardScaler()),
                      ("clf", CalibratedClassifierCV(SVC(kernel="rbf", random_state=rs),
                                                     ensemble=False))]),
            {"clf__estimator__C": [1, 10, 100], "clf__estimator__gamma": ["scale", 0.1]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=rs),
            {"max_depth": [8, 12, None], "min_samples_leaf": [1, 3]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=rs, n_jobs=-1),
            {"n_estimators": [100, 300], "max_depth": [None, 12],
             "max_features": ["sqrt", 0.5]},
        ),
        "Extra Trees": (
            ExtraTreesClassifier(random_state=rs, n_jobs=-1),
            {"n_estimators": [100, 300], "max_depth": [None, 12]},
        ),
        "Hist Gradient Boosting": (
            HistGradientBoostingClassifier(random_state=rs),
            {"learning_rate": [0.05, 0.1], "max_iter": [200], "max_leaf_nodes": [15, 31]},
        ),
    }


def tune_and_compare(X_train, y_train, X_test, y_test):
    """Grid-search every model on training data, then score it on the test set."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows, fitted = [], {}

    for name, (estimator, grid) in candidate_models().items():
        search = GridSearchCV(estimator, grid, cv=cv, scoring="accuracy", n_jobs=-1)
        start = time.perf_counter()
        search.fit(X_train, y_train)
        seconds = time.perf_counter() - start

        best = search.best_estimator_
        pred = best.predict(X_test)
        proba = best.predict_proba(X_test)
        i = search.best_index_
        rows.append({
            "model": name,
            "cv_accuracy_mean": search.cv_results_["mean_test_score"][i],
            "cv_accuracy_std": search.cv_results_["std_test_score"][i],
            "test_accuracy": accuracy_score(y_test, pred),
            "test_macro_f1": f1_score(y_test, pred, average="macro"),
            # lower log loss = better calibrated probabilities for the re-ranker
            "test_log_loss": log_loss(y_test, proba, labels=best.classes_),
            "search_seconds": seconds,
            "best_params": search.best_params_,
        })
        fitted[name] = best

    results = (pd.DataFrame(rows)
                 .sort_values("cv_accuracy_mean", ascending=False)
                 .reset_index(drop=True))
    return results, fitted


def pick_best(results):
    """Pick the model with the best CV accuracy; ties go to lower log loss."""
    # Selecting on test accuracy would leak test data into the choice.
    ranked = results.sort_values(["cv_accuracy_mean", "test_log_loss"],
                                 ascending=[False, True])
    return ranked.iloc[0]["model"]
