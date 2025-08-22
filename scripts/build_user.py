"""
Build synthetic user-job training pairs (user_job.csv).

Input (expected in data/raw/):
  - postings.csv
  - job_skills.csv
  - skill_mapping.csv

Output:
  - data/user_job.csv
"""

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.utils import setup_path, load_cfg, load_raw_frames, ensure_dir
setup_path()

import argparse
import random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def build_user_job(df_postings, df_skills, df_map, seed=42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    df_sk_full = df_skills.merge(df_map, on="skill_abr", how="left")
    df_job_skills = df_sk_full.groupby("job_id")["skill_name"].apply(list).reset_index()
    df_full = df_postings.merge(df_job_skills, on="job_id", how="inner")
    df_full["skill_name"] = df_full["skill_name"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    mlb = MultiLabelBinarizer()
    mlb.fit(df_full["skill_name"])
    skills = list(mlb.classes_)

    samples = []
    labels = []

    for _, row in df_full.iterrows():
        job_id = row["job_id"]
        true_skills = set(row["skill_name"])
        if len(true_skills) < 2:
            continue

        # Qualified sample (remove ≤ 1 skill)
        remove_q = min(1, len(true_skills) - 1)
        partial_q = list(true_skills)
        removed_q = random.sample(partial_q, remove_q)
        remain_q = [s for s in partial_q if s not in removed_q]
        vec_q = []
        i = 0
        while i < len(skills):
            vec_q.append(1 if skills[i] in remain_q else 0)
            i += 1
        samples.append([job_id] + vec_q)
        labels.append(1)

        # Not-qualified sample (remove 2..50%)
        max_remove_nq = max(2, int(0.5 * len(true_skills)))
        if len(true_skills) > max_remove_nq:
            remove_nq = random.randint(2, max_remove_nq)
            partial_nq = list(true_skills)
            removed_nq = random.sample(partial_nq, remove_nq)
            remain_nq = [s for s in partial_nq if s not in removed_nq]
            vec_nq = []
            j = 0
            while j < len(skills):
                vec_nq.append(1 if skills[j] in remain_nq else 0)
                j += 1
            samples.append([job_id] + vec_nq)
            labels.append(0)

    columns = ["job_id"] + skills
    df_final = pd.DataFrame(samples, columns=columns)
    df_final["qualified"] = labels
    return df_final


def main():
    ap = argparse.ArgumentParser(description="Build synthetic user_job.csv")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--data_dir", type=str, default="data/raw")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", type=str, default="data/user_job.csv")
    args = ap.parse_args()

    cfg = load_cfg(args.config)  # currently unused, but kept for symmetry

    data_dir = Path(args.data_dir)
    df_postings, df_skills, df_map = load_raw_frames(
        data_dir / "postings.csv",
        data_dir / "job_skills.csv",
        data_dir / "skill_mapping.csv",
    )

    out_csv = Path(args.out_csv)
    ensure_dir(out_csv)

    final_df = build_user_job(df_postings, df_skills, df_map, seed=args.seed)
    final_df.to_csv(out_csv, index=False)

    positives = int(final_df["qualified"].sum())
    print(f"✅ Saved {out_csv} with {len(final_df)} samples and {positives} positives")


if __name__ == "__main__":
    main()
