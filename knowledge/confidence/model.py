"""
Confidence estimation model.
"""

import numpy as np

from knowledge.confidence.features import ConfidenceFeatures


class ConfidenceModel:
    """
    Placeholder confidence model.

    This will later become
    Logistic Regression/XGBoost.
    """

    def predict(
        self,
        features: ConfidenceFeatures,
    ) -> float:

        # Temporary normalized confidence
        confidence = (
            0.40 * min(features.semantic_score, 1.0)
            + 0.25 * min(features.keyword_score / 10, 1.0)
            + 0.15 * (features.agreement / 5)
            + 0.10 * min(features.context_length / 1000, 1.0)
            + 0.10 * (1 / (features.retrieval_rank + 1))
        )

        return float(np.clip(confidence, 0, 1))