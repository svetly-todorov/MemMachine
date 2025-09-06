"""
Abstract base class for a reranker.

Defines the interface for scoring and reranking candidates
based on their relevance to a query.
"""

from abc import ABC, abstractmethod


class Reranker(ABC):
    """
    Abstract base class for a reranker.
    """

    async def rerank(
        self, query: str, candidates: list[str]
    ) -> list[tuple[float, str]]:
        """
        Rerank the candidates based on their relevance to the query.

        Args:
            query (str):
                The input query string.
            candidates (list[str]):
                A list of candidate strings to be reranked.

        Returns:
            list[tuple[float, str]]:
            A list of tuples where each tuple contains
            a score and the corresponding candidate string,
            sorted by score in descending order.
        """
        scores = await self.score(query, candidates)
        return sorted(
            zip(scores, candidates),
            key=lambda pair: pair[0],
            reverse=True,
        )

    @abstractmethod
    async def score(self, query: str, candidates: list[str]) -> list[float]:
        """
        Compute relevance scores for each candidate
        with respect to the query.

        Args:
            query (str):
                The input query string.
            candidates (list[str]):
                A list of candidate strings to be scored.

        Returns:
            list[float]:
                A list of scores corresponding to each candidate.
        """
        raise NotImplementedError
