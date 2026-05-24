"""Classification rules for the three-metric diagnostic framework."""

CLASSIFICATIONS = ("INDEPENDENT", "SUBSTITUTION", "REDUNDANT")


def classify(or_significant: bool, delta_auc_ablation: float, threshold: float = 0.015) -> str:
    """
    Apply the three-metric diagnostic rule.

    Parameters
    ----------
    or_significant : bool
        Whether the logistic regression OR is statistically significant (p < alpha).
    delta_auc_ablation : float
        Drop in 5-fold CV AUC when the feature is removed (full model AUC minus
        AUC of model trained without the feature). Positive = feature helps.
    threshold : float
        ΔAUC threshold separating INDEPENDENT from SUBSTITUTION. Default 0.015.
        Classifications are stable across 0.005–0.030 (see paper §3.8).

    Returns
    -------
    str
        One of 'INDEPENDENT', 'SUBSTITUTION', or 'REDUNDANT'.

    Decision table
    --------------
    OR significant  |  ΔAUC ≥ threshold  |  Classification
    ----------------|--------------------|----------------
    True            |  True              |  INDEPENDENT
    True            |  False             |  SUBSTITUTION
    False           |  either            |  REDUNDANT
    """
    if or_significant and delta_auc_ablation >= threshold:
        return "INDEPENDENT"
    elif or_significant:
        return "SUBSTITUTION"
    else:
        return "REDUNDANT"


def interpret(classification: str) -> str:
    """Return a plain-language interpretation of the classification."""
    interpretations = {
        "INDEPENDENT": (
            "The feature has a genuine independent association with the outcome "
            "that is both statistically significant (OR) and reflected in a "
            "meaningful AUC contribution (ΔAUC ≥ threshold). It is not merely "
            "substituting for correlated features."
        ),
        "SUBSTITUTION": (
            "The feature has a genuine independent association with the outcome "
            "(significant OR) but contributes negligible incremental AUC because "
            "correlated features absorb its predictive signal. Discarding it based "
            "on ΔAUC alone would be a false negative."
        ),
        "REDUNDANT": (
            "The feature has no detectable independent association with the outcome "
            "(non-significant OR) and contributes negligible incremental AUC. It adds "
            "no unique information beyond what the other features already provide."
        ),
    }
    return interpretations[classification]
