"""
Cross-encoder based reranker implementation.
"""

from typing import Any

from sentence_transformers import CrossEncoder

from .reranker import Reranker


class CrossEncoderReranker(Reranker):
    """
    Reranker that uses a cross-encoder model to score candidates
    based on their relevance to the query.
    """

    def __init__(self, config: dict[str, Any] = {}):
        """
        Initialize a CrossEncoderReranker
        with the provided configuration.

        Args:
            config (dict[str, Any], optional):
                Configuration dictionary containing:
                - model_name (str, optional):
                    Name of the pre-trained cross-encoder model to use
                    (default: "cross-encoder/ms-marco-MiniLM-L6-v2").
        """
        super().__init__()

        model_name = config.get(
            "model_name", "cross-encoder/ms-marco-MiniLM-L6-v2"
        )
        self._cross_encoder = CrossEncoder(model_name)

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        scores = [
            float(score)
            for score in self._cross_encoder.predict(
                [(query, candidate) for candidate in candidates]
            )
        ]
        return scores
