"""
Abstract base class for a language model.
"""

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from memmachine.episodic_memory.data_types import MemoryContext


class LanguageModel(ABC):
    """
    Abstract base class for a language model.
    """

    DEFAULT_METRICS_LABELS: dict[str, str] = {
        "group_id": "",
        "agent_id": "",
        "user_id": "",
        "session_id": "",
    }

    def set_default_metrics_labels(
        self,
        user_id: str = "",
        agent_id: str = "",
        group_id: str = "",
        session_id: str = "",
    ) -> None:
        """
        Set the default metrics labels for the language model.
        
        Each language model call should be attributed to a specific user and agent
        for accurate token usage tracking.
        
        Args:
            user_id: The specific user ID for this LLM call (e.g., the message producer).
            agent_id: The specific agent ID for this LLM call (e.g., the agent being called).
            group_id: The group identifier.
            session_id: The session identifier.
        """
        self.DEFAULT_METRICS_LABELS["user_id"] = user_id
        self.DEFAULT_METRICS_LABELS["agent_id"] = agent_id
        self.DEFAULT_METRICS_LABELS["group_id"] = group_id
        self.DEFAULT_METRICS_LABELS["session_id"] = session_id

    @abstractmethod
    async def generate_response(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = None,
        max_attempts: int = 1,
    ) -> tuple[str, Any]:
        """
        Generate a response based on the provided prompts and tools.

        Args:
            system_prompt (str | None, optional):
                The system prompt to guide the model's behavior
                (default: None).
            user_prompt (str | None, optional):
                The user prompt containing the main input
                (default: None).
            tools (list[dict[str, Any]] | None, optional):
                A list of tools that the model can use in its response, if supported
                (default: None).
            tool_choice (str | dict[str, str] | None, optional):
                Strategy for selecting tools, if supported.
                Can be "auto" for automatic selection,
                "required" for using at least one tool,
                or a specific tool.
                If None, implementation-defined default is used
                (default: None).
            max_attempts (int, optional):
                The maximum number of attempts to make before giving up
                (default: 1).

        Returns:
            tuple[str, Any]:
                A tuple containing the generated response text
                and tool call outputs (if any).

        Raises:
            ExternalServiceAPIError:
                Errors from the underlying embedding API.
            ValueError:
                Invalid input or max_attempts.
        """
        raise NotImplementedError
