from __future__ import annotations
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from pathlib import Path
import sys, os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
import torch
from src.transformer import SkillTransformer
from sklearn.preprocessing import MultiLabelBinarizer

# ----------------------------- Path / Config -----------------------------
def setup_path() -> None:
    """
    Ensure project root is on sys.path so 'src' can be imported from scripts/.
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.append(root)


def load_cfg(path: str = "configs/default.yaml") -> dict:
    """
    Load YAML config into a dict; {} if not found or PyYAML missing.
    """
    try:
        import yaml
    except ImportError:
        return {}

    p = Path(path)
    if not p.exists():
        return {}

    with open(p, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    return data


def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


# ----------------------------- Data Loading -----------------------------
def load_lr_bundle(path: Path):
    """
    Load joblib bundle with fields:
      - model: trained LogisticRegression
      - skills: list[str] skill columns used for LR
    """
    import joblib
    bundle = joblib.load(path)
    model = bundle["model"]
    skills = bundle["skills"]
    return model, skills


def load_raw_frames(postings_csv: Path, job_skills_csv: Path, skill_map_csv: Path):
    """
    Read the three raw csvs and return DataFrames.
    """
    if not postings_csv.exists():
        raise FileNotFoundError(f"Missing: {postings_csv}")
    if not job_skills_csv.exists():
        raise FileNotFoundError(f"Missing: {job_skills_csv}")
    if not skill_map_csv.exists():
        raise FileNotFoundError(f"Missing: {skill_map_csv}")

    df_postings = pd.read_csv(postings_csv)
    df_skills = pd.read_csv(job_skills_csv)
    df_map = pd.read_csv(skill_map_csv)
    return df_postings, df_skills, df_map


# ----------------------------- Job/Skill Space -----------------------------
def prepare_job_skill_space(
    df_postings: pd.DataFrame,
    df_skills: pd.DataFrame,
    df_map: pd.DataFrame
) -> Tuple[pd.DataFrame, MultiLabelBinarizer]:
    """
    Merge postings + job_skills + skill names; return (df_full, mlb).
    df_full has column 'skill_name' as list[str].
    """
    df_sk_full = df_skills.merge(df_map, on="skill_abr", how="left")
    df_job_skills = df_sk_full.groupby("job_id")["skill_name"].apply(list).reset_index()
    df_full = df_postings.merge(df_job_skills, on="job_id", how="left")
    df_full["skill_name"] = df_full["skill_name"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    mlb = MultiLabelBinarizer()
    _ = mlb.fit_transform(df_full["skill_name"])
    return df_full, mlb


def user_vector_from_skills(
    user_skills: List[str],
    mlb: MultiLabelBinarizer
) -> np.ndarray:
    """
    Build a binary vector in mlb space from a list of user skills.
    """
    user_set = set(s.strip() for s in user_skills if s and str(s).strip())
    vec = np.zeros(len(mlb.classes_), dtype=int)

    i = 0
    while i < len(mlb.classes_):
        skill = mlb.classes_[i]
        if skill in user_set:
            vec[i] = 1
        i += 1

    return vec


def filter_jobs_require_more(df_full: pd.DataFrame, user_skills: List[str]) -> pd.DataFrame:
    """
    Keep jobs that require at least one skill the user doesn't have (and >1 skills total).
    """
    user_set = set(s.strip() for s in user_skills if s and str(s).strip())

    def needs_more(sk_list: Any) -> bool:
        if not isinstance(sk_list, list):
            return False
        req = set(sk_list)
        missing = req - user_set
        if len(req) <= 1:
            return False
        if len(missing) <= 0:
            return False
        return True

    mask = df_full["skill_name"].apply(needs_more)
    return df_full[mask].copy()


# ----------------------------- Retrieval (KNN) -----------------------------
from sklearn.neighbors import NearestNeighbors

def knn_retrieve_indices(
    df: pd.DataFrame,
    mlb: MultiLabelBinarizer,
    user_vec: np.ndarray,
    k: int = 20
) -> np.ndarray:
    """
    Fit KNN on df's job skill matrix; return indices of top-k neighbors.
    """
    job_mat = mlb.transform(df["skill_name"])
    k_eff = min(k, len(df)) if len(df) > 0 else 0
    if k_eff <= 0:
        return np.array([], dtype=int)

    knn = NearestNeighbors(n_neighbors=k_eff)
    knn.fit(job_mat)
    _, indices = knn.kneighbors(user_vec.reshape(1, -1))
    return indices[0]


# ----------------------------- Scoring -----------------------------
def score_job_row(
    job_row: pd.Series,
    user_skills: List[str],
    lr_model,
    lr_skill_cols: List[str],
    mlb: MultiLabelBinarizer,
    alpha_match: float = 0.5,
    beta_lr: float = 0.3,
    gamma_div: float = 0.2
) -> Dict[str, Any]:
    """
    Compute scoring components and final score for a single job row.
    Returns dict with Missing, Matched, Match Ratio, LR, Score.
    """
    user_set = set(s.strip() for s in user_skills if s and str(s).strip())
    req = set(job_row["skill_name"]) if isinstance(job_row["skill_name"], list) else set()

    matched = sorted(req & user_set)
    missing = sorted(req - user_set)
    match_ratio = float(len(matched) / len(req)) if len(req) > 0 else 0.0

    # LR probability uses LR training skill order
    user_vec_lr = np.zeros(len(lr_skill_cols), dtype=int)
    i = 0
    while i < len(lr_skill_cols):
        if lr_skill_cols[i] in user_set:
            user_vec_lr[i] = 1
        i += 1
    lr_prob = float(lr_model.predict_proba(user_vec_lr.reshape(1, -1))[0, 1])

    # Diversity = how many new skills this job would add
    diversity = len(missing)
    denom = max(1, len(mlb.classes_))
    score = (alpha_match * match_ratio) + (beta_lr * lr_prob) + (gamma_div * diversity / denom)

    return {
        "Missing": missing,
        "Matched": matched,
        "Match Ratio": match_ratio,
        "LR": lr_prob,
        "Score": score
    }


# ----------------------------- Transformer Helpers -----------------------------

def detect_device():
    """
    Pick a sensible torch device if torch is available.
    """
    if torch is None:
        return None
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_transformer(weights: Path, num_skills: int):
    """
    Load a SkillTransformer with saved weights if available.
    Returns (model, device) or (None, None) if unavailable.
    """
    if torch is None or SkillTransformer is None or not weights.exists():
        return None, None

    device = detect_device()
    model = SkillTransformer(num_skills=num_skills)
    state = torch.load(str(weights), map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, device


def transformer_suggestions(
    model,
    device,
    user_vec: np.ndarray,
    mlb_classes: np.ndarray,
    top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    Suggest top-k missing skills based on transformer probabilities.
    Step-by-step style (no inline comprehensions in return).
    """
    if model is None or torch is None:
        return []

    with torch.no_grad():
        x = torch.tensor(user_vec, dtype=torch.float32, device=device).unsqueeze(0)
        probs = model(x).squeeze(0).cpu().numpy()

    existing = set(np.where(user_vec == 1)[0])
    order = np.argsort(probs)[::-1]

    suggestions: List[Tuple[str, float]] = []
    for idx in order:
        if idx not in existing:
            skill = str(mlb_classes[idx])
            confidence = float(probs[idx])
            suggestions.append((skill, confidence))
        if len(suggestions) >= top_k:
            break

    return suggestions
