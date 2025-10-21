"""
Abstract base class for a language model.
"""

from abc import ABC, abstractmethod
from typing import Any

class LanguageModel(ABC):
    """
    Abstract base class for a language model.
    """

    def __init__(self):
        """Initialize the language model with an empty metrics labels dict."""
        # Instance variable for metrics labels, managed by base class
        self._user_metrics_labels: dict[str, str] = {}

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
        self._user_metrics_labels["user_id"] = user_id
        self._user_metrics_labels["agent_id"] = agent_id
        self._user_metrics_labels["group_id"] = group_id
        self._user_metrics_labels["session_id"] = session_id

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
