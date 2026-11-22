"""
step_22_label_noise_simulation.py — attenuation of AUROC by label discordance (protocol analysis A5).

WHY THIS FILE EXISTS
--------------------
Receptor labels in this cohort may come from the primary tumor, a lymph node or a metastasis, and
receptor status can differ between primary tumor and brain metastasis. If a fraction epsilon of
labels does not describe the imaged lesion, an image signal with true AUROC A is observed with a
lower AUROC. This script quantifies that attenuation.

Model: binormal scores (negatives N(0,1), positives N(d,1), d = sqrt(2) * Phi^-1(A)), prevalence
0.5, each label flipped independently with probability epsilon regardless of class
(nondifferential). 200,000 cases per cell, seed 42. With prevalence 0.5 the expected observed
AUROC is exactly 0.5 + (A - 0.5)(1 - 2 epsilon) for any score distribution; that closed form is
written beside each simulated value as a check.

Protocol: docs/PROTOCOL_added_analyses.md, A5.

Usage:
    python src/analysis/step_22_label_noise_simulation.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = _ROOT / "bcbm_project"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "revision_R1"
META_DIR = PROJECT_ROOT / "metadata"

SEED = 42
N_CASES = 200_000
PREVALENCE = 0.5
TRUE_AUCS = [0.65, 0.75, 0.85]
EPSILONS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("step_22")


def main() -> int:
    log = setup_logger()
    rng = np.random.default_rng(SEED)
    rows = []
    for a in TRUE_AUCS:
        d = np.sqrt(2) * norm.ppf(a)
        y = (rng.random(N_CASES) < PREVALENCE).astype(int)
        s = rng.normal(0.0, 1.0, N_CASES) + d * y
        for eps in EPSILONS:
            flip = rng.random(N_CASES) < eps
            y_obs = np.where(flip, 1 - y, y)
            obs = float(roc_auc_score(y_obs, s))
            rows.append({"true_auroc": a, "epsilon": eps, "n_cases": N_CASES, "prevalence": PREVALENCE,
                         "observed_auroc": obs,
                         "closed_form_auroc": 0.5 + (a - 0.5) * (1 - 2 * eps),
                         "realized_flip_fraction": float(flip.mean())})
            log.info("true %.2f eps %.2f -> observed %.4f (closed form %.4f)", a, eps, obs,
                     rows[-1]["closed_form_auroc"])
    out = pd.DataFrame(rows)
    out["abs_diff_sim_vs_closed_form"] = (out["observed_auroc"] - out["closed_form_auroc"]).abs()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLES_DIR / "step22_label_noise_attenuation.csv", index=False, encoding="utf-8-sig")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=300)
    for a, m in zip(TRUE_AUCS, ["o", "s", "^"]):
        g = out[out["true_auroc"] == a]
        ax.plot(g["epsilon"], g["observed_auroc"], marker=m, color="black", lw=1,
                label=f"True AUROC {a:.2f}")
    ax.axhline(0.5, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("Fraction of labels not describing the imaged lesion (ε)")
    ax.set_ylabel("Observed AUROC")
    ax.set_ylim(0.45, 0.9)
    ax.legend(frameon=False, fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "tif"):
        fig.savefig(FIG_DIR / f"step22_label_noise_attenuation.{ext}", dpi=300)
    plt.close(fig)

    json.dump({"seed": SEED, "n_cases": N_CASES, "prevalence": PREVALENCE, "true_aucs": TRUE_AUCS,
               "epsilons": EPSILONS, "max_abs_diff_sim_vs_closed_form":
                   float(out["abs_diff_sim_vs_closed_form"].max())},
              open(META_DIR / "step22_label_noise_simulation.json", "w", encoding="utf-8"), indent=1)
    log.info("wrote step22_label_noise_attenuation.csv and figure; max |sim - closed form| = %.4f",
             out["abs_diff_sim_vs_closed_form"].max())
    return 0


if __name__ == "__main__":
    sys.exit(main())
