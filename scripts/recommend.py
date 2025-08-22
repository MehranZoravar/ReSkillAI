import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.utils import (
    setup_path, load_cfg, load_lr_bundle, load_raw_frames, prepare_job_skill_space,
    user_vector_from_skills, filter_jobs_require_more, knn_retrieve_indices,
    score_job_row, load_transformer, transformer_suggestions
)

setup_path()

import argparse
from pathlib import Path
import numpy as np
from src.ocr import extract_skills_from_cv


def main():
    parser = argparse.ArgumentParser(description="Config-driven Job & Skill Recommender")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--skills", type=str, nargs="*")
    parser.add_argument("--cv", type=str)
    parser.add_argument("--top_jobs", type=int)
    parser.add_argument("--require_more", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    postings = Path(cfg["paths"]["postings"])
    job_skills = Path(cfg["paths"]["job_skills"])
    skill_map = Path(cfg["paths"]["skill_mapping"])
    lr_model_path = Path(cfg["paths"]["lr_model"])
    transformer_weights = Path(cfg["paths"]["skill_transformer"])

    top_k = args.top_jobs if args.top_jobs is not None else int(cfg["recommend"].get("top_jobs", 5))
    require_more = args.require_more or bool(cfg["recommend"].get("require_more_skills", False))

    # --- skills source + print first
    if args.skills:
        user_skills = [s for s in args.skills if s and str(s).strip()]
        src = "CLI --skills"
    elif args.cv:
        user_skills = extract_skills_from_cv(args.cv)
        src = f"CV ({args.cv})"
    elif cfg.get("recommend", {}).get("default_cv"):
        user_skills = extract_skills_from_cv(cfg["recommend"]["default_cv"])
        src = f"CV ({cfg['recommend']['default_cv']})"
    elif cfg.get("recommend", {}).get("default_skills"):
        user_skills = [s for s in cfg["recommend"]["default_skills"] if s and str(s).strip()]
        src = "config.default_skills"
    else:
        raise SystemExit("❌ Provide --skills or --cv, or set default_cv/default_skills in config.")

    print("\n=== User Skills (source: {}) ===".format(src))
    if user_skills:
        print(", ".join(sorted(set(user_skills))))
    else:
        print("⚠️ No skills extracted; aborting.")
        return

    # --- load models and data
    lr_model, lr_skill_cols = load_lr_bundle(lr_model_path)
    df_postings, df_skills, df_map = load_raw_frames(postings, job_skills, skill_map)
    df_full, mlb = prepare_job_skill_space(df_postings, df_skills, df_map)

    # --- branch require_more
    if require_more:
        df_used = filter_jobs_require_more(df_full, user_skills)
    else:
        df_used = df_full.copy()

    if df_used.empty:
        print("\n⚠️ No jobs matched the current setting (require_more_skills = {}).".format(require_more))
        return

    # --- retrieve and score
    user_vec = user_vector_from_skills(user_skills, mlb)
    idx = knn_retrieve_indices(df_used, mlb, user_vec, k=20)
    recs = df_used.iloc[idx]

    rows = []
    for _, row in recs.iterrows():
        comp = score_job_row(row, user_skills, lr_model, lr_skill_cols, mlb)
        out = {
            "Job Title": row.get("title", ""),
            "Missing": comp["Missing"],
            "Matched": comp["Matched"],
            "Match Ratio": comp["Match Ratio"],
            "LR": comp["LR"],
            "Score": comp["Score"],
        }
        rows.append(out)

    rows = sorted(rows, key=lambda r: r["Score"], reverse=True)[:top_k]

    print("\n=== Recommendations (require_more_skills = {}) ===".format(require_more))
    for r in rows:
        print(f"- {r['Job Title']}")
        print(f"  Missing: {r['Missing']}")
        print(f"  Matched: {r['Matched']}")
        print(f"  Score: {r['Score']:.3f} (Match {r['Match Ratio']:.2f} | LR {r['LR']:.2f})")

    # --- transformer (optional)
    model, device = load_transformer(transformer_weights, len(mlb.classes_))
    if model is not None:
        suggestions = transformer_suggestions(model, device, user_vec, mlb.classes_, top_k=10)
        if suggestions:
            # prioritize skills that actually help with top jobs
            miss_union = set()
            for r in rows:
                miss_union.update(r["Missing"])
            filtered = []
            for s, c in suggestions:
                if not miss_union or s in miss_union:
                    filtered.append((s, c))
            if not filtered:
                filtered = suggestions

            print("\n=== Suggested Skills (Transformer) ===")
            count = 0
            for s, c in filtered:
                print(f"- {s} (conf: {c:.2f})")
                count += 1
                if count >= 10:
                    break
    else:
        print("\n[INFO] No Transformer suggestions (weights missing or torch not installed).")


if __name__ == "__main__":
    main()
