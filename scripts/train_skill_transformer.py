"""
Train a small Transformer to suggest missing skills for unqualified users.

Input:
  - data/user_job.csv  (columns: job_id, <one-hot skills...>, qualified)

Output:
  - models/skill_transformer.pt

"""
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
from src.transformer import SkillTransformer

def build_training_pairs(X: np.ndarray,
                         y: np.ndarray,
                         k: int = 5,
                         suggest_threshold: float = 0.2,
                         fallback_topk: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Create (inputs, targets) for training, per your prior logic."""
    qualified = X[y == 1]
    unqualified = X[y == 0]

    X_inputs, Y_targets = [], []

    if len(qualified) >= k:
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(qualified)

        for user_vec in unqualified:
            distances, indices = knn.kneighbors([user_vec])
            nearest = qualified[indices[0]]
            avg = nearest.mean(axis=0)

            suggest = ((avg > suggest_threshold) & (user_vec == 0)).astype(np.float32)
            if suggest.sum() == 0:
                continue
            X_inputs.append(user_vec.astype(np.float32))
            Y_targets.append(suggest)
    else:
        pass

    if len(X_inputs) == 0:
        # Fallback: most common skills among qualified users
        if len(qualified) == 0:
            raise ValueError("No qualified samples found in user_job.csv; cannot build fallback targets.")
        top_counts = qualified.sum(axis=0)
        sorted_idx = np.argsort(-top_counts)
        for user_vec in unqualified:
            missing_mask = (user_vec == 0)
            # pick top skills the user is missing
            fallback_indices = [i for i in sorted_idx if missing_mask[i]][:fallback_topk]
            if not fallback_indices:
                continue
            suggest = np.zeros_like(user_vec, dtype=np.float32)
            suggest[fallback_indices] = 1.0
            X_inputs.append(user_vec.astype(np.float32))
            Y_targets.append(suggest)

    return np.array(X_inputs, dtype=np.float32), np.array(Y_targets, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user_job", type=str, default="data/user_job.csv")
    ap.add_argument("--out", type=str, default="models/skill_transformer.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--knn_k", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--fallback_topk", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # Load user_job
    df = pd.read_csv(args.user_job)
    skill_cols = df.columns[1:-1]
    X = df[skill_cols].values.astype(np.float32)
    y = df["qualified"].values.astype(int)

    # Build training pairs
    X_inputs, Y_targets = build_training_pairs(
        X, y, k=args.knn_k, suggest_threshold=args.threshold, fallback_topk=args.fallback_topk
    )
    print(f"✅ Training samples prepared: {len(X_inputs)} (inputs: {X_inputs.shape}, targets: {Y_targets.shape})")

    if len(X_inputs) == 0:
        raise RuntimeError("No training pairs could be constructed. Check your user_job.csv balance.")

    # Model / device
    num_skills = X_inputs.shape[1]
    device = (
        torch.device("mps") if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    model = SkillTransformer(num_skills=num_skills).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    # Simple mini-batch training
    n = len(X_inputs)
    idx = np.arange(n)

    for epoch in range(1, args.epochs + 1):
        rng.shuffle(idx)
        epoch_loss = 0.0
        model.train()

        for start in range(0, n, args.batch_size):
            batch_idx = idx[start:start + args.batch_size]
            xb = torch.tensor(X_inputs[batch_idx], dtype=torch.float32, device=device)
            yb = torch.tensor(Y_targets[batch_idx], dtype=torch.float32, device=device)

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_idx)

        epoch_loss /= n
        print(f"Epoch {epoch:02d}/{args.epochs} - loss: {epoch_loss:.4f}")

    # Save weights
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"✅ Saved Transformer weights → {out_path}")

if __name__ == "__main__":
    main()
