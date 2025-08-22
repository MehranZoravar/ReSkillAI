import argparse, joblib
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user_job", type=str, default="data/user_job.csv")
    ap.add_argument("--model_out", type=str, default="models/lr.pkl")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.user_job)
    skill_cols = df.columns[1:-1]
    X = df[skill_cols].values
    y = df["qualified"].values

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(Xtr, ytr)

    prob = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, prob)
    print(f"ROC AUC: {auc:.4f}")
    print(classification_report(yte, (prob >= 0.5).astype(int)))

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "skills": list(skill_cols)}, args.model_out)
    print(f"✅ Saved LR model to {args.model_out}")

if __name__ == "__main__":
    main()
