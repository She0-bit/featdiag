"""
MIDUS 2 Reproduction Example
=============================

Reproduces the three case studies from:

  Alotaibi SA. When SHAP, Incremental AUC, and Logistic Regression Diverge:
  A Simulation-Validated Diagnostic Framework for Feature Substitution in
  Clinical Machine Learning. npj Digital Medicine (submitted 2026).

Requirements
------------
  pip install featdiag
  pip install featdiag[lime]          # for run_lime=True

Data access
-----------
  MIDUS 2 (ICPSR 04652): https://www.icpsr.umich.edu/web/NACDA/studies/4652
  MIDUS 2 Biomarker (ICPSR 29282): https://www.icpsr.umich.edu/web/NACDA/studies/29282
  HRS (University of Michigan): https://hrs.isr.umich.edu/

Running
-------
  python examples/midus_example.py

The script will print a summary and save:
  midus_cvd_result.png
  hrs_result.png         (requires HRS data)
  biomarker_result.png   (requires Biomarker data)
"""

from __future__ import annotations

import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = pathlib.Path(__file__).parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)


# ── Utility: build cleaned feature matrix ─────────────────────────────────────

def _load_midus2_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract the 9-feature set from a cleaned MIDUS 2 main-file DataFrame.

    Expected columns (post-harmonisation):
        pil, age, bmi, smoke, phys_act, sleep, educ, income, female
        cvd_event   (binary outcome, cardiovascular event)
    """
    features = ["pil", "age", "bmi", "smoke", "phys_act", "sleep", "educ", "income", "female"]
    outcome = "cvd_event"
    keep = features + [outcome]
    sub = df[keep].dropna()
    return sub[features], sub[outcome].astype(int)


# ── Case Study 1: MIDUS 2 — PIL → CVD ─────────────────────────────────────────

def run_midus_cvd(df_midus2: pd.DataFrame):
    """SUBSTITUTION expected: OR significant, ΔAUC < 0.015."""
    from featdiag import DiagnosticFramework

    X, y = _load_midus2_features(df_midus2)
    print(f"\n{'='*60}")
    print(f"Case Study 1: MIDUS 2 — PIL → CVD")
    print(f"n={len(y)}, events={y.sum()} ({y.mean()*100:.1f}%)")
    print(f"{'='*60}")

    fw = DiagnosticFramework(X, y, feature="pil", outcome="CVD", threshold=0.015)
    result = fw.fit(run_lime=False, verbose=True)
    result.summary()
    result.save_plot(str(OUT / "midus_cvd_result.png"))
    print(f"  Figure saved → {OUT / 'midus_cvd_result.png'}")

    with open(OUT / "midus_cvd_result.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    assert result.classification == "SUBSTITUTION", (
        f"Expected SUBSTITUTION, got {result.classification}"
    )
    return result


# ── Case Study 2: HRS — PIL → incident CVD ────────────────────────────────────

def run_hrs(df_hrs: pd.DataFrame):
    """
    SUBSTITUTION expected prospectively.

    Expected HRS columns (post-harmonisation):
        pil, age, bmi, smoke, phys_act, sleep, educ, income, female
        incident_cvd  (binary outcome)
    """
    from featdiag import DiagnosticFramework

    features = ["pil", "age", "bmi", "smoke", "phys_act", "sleep", "educ", "income", "female"]
    sub = df_hrs[features + ["incident_cvd"]].dropna()
    X, y = sub[features], sub["incident_cvd"].astype(int)

    print(f"\n{'='*60}")
    print(f"Case Study 2: HRS — PIL → incident CVD")
    print(f"n={len(y)}, events={y.sum()} ({y.mean()*100:.1f}%)")
    print(f"{'='*60}")

    fw = DiagnosticFramework(X, y, feature="pil", outcome="incident_CVD", threshold=0.015)
    result = fw.fit(run_lime=False, verbose=True)
    result.summary()
    result.save_plot(str(OUT / "hrs_result.png"))
    print(f"  Figure saved → {OUT / 'hrs_result.png'}")

    with open(OUT / "hrs_result.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    return result


# ── Case Study 3: MIDUS 2 Biomarker — PIL → elevated CRP ──────────────────────

def run_biomarker(df_biomarker: pd.DataFrame):
    """
    REDUNDANT expected: OR not significant, validates discriminative validity.

    Expected biomarker columns (post-harmonisation, merged on M2ID):
        pil, age, bmi, smoke, phys_act, sleep, educ, income, female
        crp_high  (binary: CRP > 3 mg/L per ACC/AHA threshold)
    """
    from featdiag import DiagnosticFramework

    features = ["pil", "age", "bmi", "smoke", "phys_act", "sleep", "educ", "income", "female"]
    sub = df_biomarker[features + ["crp_high"]].dropna()
    X, y = sub[features], sub["crp_high"].astype(int)

    print(f"\n{'='*60}")
    print(f"Case Study 3: MIDUS 2 Biomarker — PIL → CRP > 3 mg/L")
    print(f"n={len(y)}, events={y.sum()} ({y.mean()*100:.1f}%)")
    print(f"{'='*60}")

    fw = DiagnosticFramework(X, y, feature="pil", outcome="CRP>3", threshold=0.015)
    result = fw.fit(run_lime=True, lime_seeds=10, lime_samples=300, verbose=True)
    result.summary()
    result.save_plot(str(OUT / "biomarker_result.png"))
    print(f"  Figure saved → {OUT / 'biomarker_result.png'}")

    with open(OUT / "biomarker_result.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    assert result.classification == "REDUNDANT", (
        f"Expected REDUNDANT, got {result.classification}"
    )
    return result


# ── Discriminative validity check ─────────────────────────────────────────────

def discriminative_validity_report(r_cvd, r_biomarker):
    print(f"\n{'='*60}")
    print("Discriminative Validity")
    print(f"{'='*60}")
    print(f"  PIL → CVD:        {r_cvd.classification}  "
          f"(OR={r_cvd.or_value:.3f}, p={r_cvd.p_value:.3f}, "
          f"ΔAUC={r_cvd.delta_auc_ablation:+.4f})")
    print(f"  PIL → CRP > 3:    {r_biomarker.classification}  "
          f"(OR={r_biomarker.or_value:.3f}, p={r_biomarker.p_value:.3f}, "
          f"ΔAUC={r_biomarker.delta_auc_ablation:+.4f})")
    print()
    print("  Same feature, same pipeline, same patients,")
    print("  different outcome → different verdict.")
    print("  This confirms the framework is not a default classifier.")
    print(f"{'='*60}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Load your harmonised data here ────────────────────────────────────────
    # Replace these with actual data loading from ICPSR downloads.
    # The script will run the MIDUS 2 CVD case study from a CSV if provided
    # as the first argument, otherwise it prints usage instructions.

    if len(sys.argv) >= 2:
        print(f"Loading MIDUS 2 data from: {sys.argv[1]}")
        df = pd.read_csv(sys.argv[1])
        r_cvd = run_midus_cvd(df)

        if len(sys.argv) >= 3:
            print(f"Loading Biomarker data from: {sys.argv[2]}")
            df_bio = pd.read_csv(sys.argv[2])
            r_bio = run_biomarker(df_bio)
            discriminative_validity_report(r_cvd, r_bio)

        if len(sys.argv) >= 4:
            print(f"Loading HRS data from: {sys.argv[3]}")
            df_hrs = pd.read_csv(sys.argv[3])
            run_hrs(df_hrs)

    else:
        print(__doc__)
        print("\nUsage:")
        print("  python examples/midus_example.py midus2.csv")
        print("  python examples/midus_example.py midus2.csv biomarker.csv")
        print("  python examples/midus_example.py midus2.csv biomarker.csv hrs.csv")
        print()
        print("Running synthetic demonstration instead...\n")

        # ── Synthetic demo (no real data needed) ──────────────────────────────
        from featdiag import DiagnosticFramework

        rng = np.random.RandomState(42)
        n = 800
        pil   = rng.randn(n)
        bmi   = 25 + 4 * rng.randn(n)
        age   = 55 + 10 * rng.randn(n)
        smoke = (rng.random(n) < 0.25).astype(float)
        phys  = rng.randn(n)
        sleep = 7 + rng.randn(n)
        educ  = rng.randint(1, 6, n).astype(float)
        income = rng.randn(n)
        female = (rng.random(n) < 0.52).astype(float)

        # CVD: PIL is a genuine predictor masked by BMI/age (substitution scenario)
        logit_cvd = -0.4 * pil + 0.5 * bmi / 4 + 0.3 * age / 10 + 0.4 * smoke
        y_cvd = (rng.random(n) < 1 / (1 + np.exp(-logit_cvd))).astype(int)

        # CRP: PIL has no real association (redundant scenario)
        logit_crp = 0.4 * bmi / 4 + 0.2 * smoke + 0.01 * pil
        y_crp = (rng.random(n) < 1 / (1 + np.exp(-logit_crp))).astype(int)

        X = pd.DataFrame({
            "pil": pil, "age": age, "bmi": bmi, "smoke": smoke,
            "phys_act": phys, "sleep": sleep, "educ": educ,
            "income": income, "female": female,
        })

        print("[Demo] Case Study 1 — PIL → CVD (expect SUBSTITUTION or INDEPENDENT)")
        fw1 = DiagnosticFramework(X, y_cvd, feature="pil", outcome="CVD_demo")
        r1 = fw1.fit(run_lime=False, verbose=True)
        r1.summary()
        r1.save_plot(str(OUT / "demo_cvd_result.png"))

        print("\n[Demo] Case Study 2 — PIL → CRP>3 (expect REDUNDANT)")
        fw2 = DiagnosticFramework(X, y_crp, feature="pil", outcome="CRP_demo")
        r2 = fw2.fit(run_lime=False, verbose=True)
        r2.summary()
        r2.save_plot(str(OUT / "demo_crp_result.png"))

        discriminative_validity_report(r1, r2)
        print(f"Figures saved to: {OUT}")
