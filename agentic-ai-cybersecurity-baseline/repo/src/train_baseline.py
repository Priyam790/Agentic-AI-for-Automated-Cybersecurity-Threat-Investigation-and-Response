"""Train baseline classifiers (Logistic Regression, Random Forest) for
benign-vs-attack detection and report real metrics on the held-out test set."""
import argparse
import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)


def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_s = time.time() - t0
    t0 = time.time()
    pred = model.predict(X_te)
    pred_s = time.time() - t0
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    m = {
        "model": name,
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred, zero_division=0),
        "f1": f1_score(y_te, pred, zero_division=0),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) else 0.0,
        "train_seconds": round(train_s, 2),
        "predict_seconds_per_1k_flows": round(pred_s / len(X_te) * 1000, 4),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    print(f"{name}: acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
          f"rec={m['recall']:.4f} f1={m['f1']:.4f} "
          f"FPR={m['false_positive_rate']:.4f} FNR={m['false_negative_rate']:.4f}")
    return m


def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Attack"]); ax.set_yticklabels(["Benign", "Attack"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    mx = np.max(cm)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i][j]:,}", ha="center", va="center",
                    color="white" if cm[i][j] > mx / 2 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/data.npz")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X_tr, X_te, y_tr, y_te = d["X_train"], d["X_test"], d["y_train"], d["y_test"]
    print(f"train={X_tr.shape} test={X_te.shape}")

    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                                 random_state=args.seed)),
    ]
    os.makedirs(args.results_dir, exist_ok=True)
    results = []
    for name, model in models:
        m = evaluate(name, model, X_tr, y_tr, X_te, y_te)
        results.append(m)
        tag = name.lower().replace(" ", "_")
        plot_cm(m["confusion_matrix"], name, os.path.join(args.results_dir, f"cm_{tag}.png"))
        if name == "Random Forest":
            joblib.dump(model, os.path.join(args.results_dir, "rf_model.joblib"))

    with open(os.path.join(args.results_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.results_dir, "metrics.md"), "w") as f:
        f.write("| Model | Accuracy | Precision | Recall | F1 | FPR | FNR |\n|---|---|---|---|---|---|---|\n")
        for m in results:
            f.write(f"| {m['model']} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
                    f"{m['recall']:.4f} | {m['f1']:.4f} | {m['false_positive_rate']:.4f} | "
                    f"{m['false_negative_rate']:.4f} |\n")
    print(f"results saved in {args.results_dir}/")


if __name__ == "__main__":
    main()
