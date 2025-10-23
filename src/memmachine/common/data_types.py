"""
Common data types for MemMachine.
"""

from pydantic import BaseModel, Field, Self, model_validator

class ExternalServiceAPIError(Exception):
    """
    Raised when an API error occurs for an external service.
    """

    pass

class AppConst:
    """Constants for app and header key names."""

    DEFAULT_GROUP_ID = "default"
    """Default value for group id when not provided."""

    DEFAULT_SESSION_ID = "default"
    """Default value for session id when not provided."""

    DEFAULT_USER_ID = "default"
    """Default value for user id when not provided."""

    DEFAULT_PRODUCER_ID = "default"
    """Default value for producer id when not provided."""

    DEFAULT_EPISODE_TYPE = "message"
    """Default value for episode type when not provided."""

    GROUP_ID_KEY = "group-id"
    """Header key for group ID."""

    SESSION_ID_KEY = "session-id"
    """Header key for session ID."""

    AGENT_ID_KEY = "agent-id"
    """Header key for agent ID."""

    USER_ID_KEY = "user-id"
    """Header key for user ID."""

    GROUP_ID_DOC = (
        "Unique identifier for a group or shared context. "
        "Used as the main filtering property. "
        "For single-user use cases, this can be the same as `user_id`. "
        "Defaults to `default` if not provided and user ids is empty. "
        "Defaults to the first user id if user ids are provided."
    )

    AGENT_ID_DOC = (
        "List of agent identifiers associated with this session. "
        "Useful if multiple AI agents participate in the same context. "
        "Defaults to `[]` if not provided."
    )

    USER_ID_DOC = (
        "List of user identifiers participating in this session. "
        "Used to isolate memories and data per user. "
        "Defaults to `['default']` if not provided."
    )

    SESSION_ID_DOC = (
        "Unique identifier for a specific session or conversation. "
        "Can represent a chat thread, Slack channel, or conversation instance. "
        "Should be unique per conversation to avoid data overlap. "
        "Defaults 'default' if not provided and user ids is empty. "
        "Defaults to the first `user_id` if user ids are provided."
    )

    PRODUCER_DOC = (
        "Identifier of the entity producing the episode. "
        "Default to the first `user_id` in the session if not provided. "
        "Default to `default` if user_id is not available."
    )

    PRODUCER_FOR_DOC = "Identifier of the entity for whom the episode is produced."

    EPISODE_CONTENT_DOC = "Content of the memory episode."

    EPISODE_TYPE_DOC = "Type of the episode content (e.g., message)."

    EPISODE_META_DOC = "Additional metadata for the episode."

    GROUP_ID_EXAMPLES = ["group-1234", "project-alpha", "team-chat"]
    AGENT_ID_EXAMPLES = ["crm", "healthcare", "sales", "agent-007"]
    USER_ID_EXAMPLES = ["user-001", "alice@example.com"]
    SESSION_ID_EXAMPLES = ["session-5678", "chat-thread-42", "conversation-abc"]
    PRODUCER_EXAMPLES = ["chatbot", "user-1234", "agent-007"]
    PRODUCER_FOR_EXAMPLES = ["user-1234", "team-alpha", "project-xyz"]
    EPISODE_CONTENT_EXAMPLES = ["Met at the coffee shop to discuss project updates."]
    EPISODE_TYPE_EXAMPLES = ["message"]
    EPISODE_META_EXAMPLES = [{"mood": "happy", "location": "office"}]

# Request session data
class SessionData(BaseModel):
    """Metadata used to organize and filter memory or conversation context.

    Each ID serves a different level of data separation:
    - `group_id`: identifies a shared context (e.g., a group chat or project).
    - `user_id`: identifies individual participants within the group.
    - `agent_id`: identifies the AI agent(s) involved in the session.
    - `session_id`: identifies a specific conversation thread or session.
    """

    group_id: str = Field(
        default="",
        description=AppConst.GROUP_ID_DOC,
        examples=AppConst.GROUP_ID_EXAMPLES,
    )

    agent_id: list[str] = Field(
        default=[],
        description=AppConst.AGENT_ID_DOC,
        examples=AppConst.AGENT_ID_EXAMPLES,
    )

    user_id: list[str] = Field(
        default=[AppConst.DEFAULT_USER_ID],
        description=AppConst.USER_ID_DOC,
        examples=AppConst.USER_ID_EXAMPLES,
    )

    session_id: str = Field(
        default="",
        description=AppConst.SESSION_ID_DOC,
        examples=AppConst.SESSION_ID_EXAMPLES,
    )

    def merge(self, other: Self) -> None:
        """Merge another SessionData into this one in place.

        - Combine and deduplicate list fields.
        - Overwrite string fields if the new value is set.
        """

        def merge_lists(a: list[str], b: list[str]) -> list[str]:
            if a and b:
                ret = list(dict.fromkeys(a + b))  # preserve order & unique
            else:
                ret = a or b
            return sorted(ret)

        if other.group_id:
            self.group_id = other.group_id

        if other.session_id:
            self.session_id = other.session_id

        self.agent_id = merge_lists(self.agent_id, other.agent_id)
        self.user_id = merge_lists(self.user_id, other.user_id)

    def first_user_id(self) -> str:
        """Returns the first user ID if available, else default user id."""
        return self.user_id[0] if self.user_id else AppConst.DEFAULT_USER_ID

    def combined_user_ids(self) -> str:
        """format groups id to <size>#<user-id><size>#<user-id>..."""
        return "".join([f"{len(uid)}#{uid}" for uid in sorted(self.user_id)])

    def from_user_id_or(self, default_value: str) -> str:
        """returns the first user id or combined user ids as a default string."""
        size_user_id = len(self.user_id)
        if size_user_id == 0:
            return default_value
        elif size_user_id == 1:
            return self.first_user_id()
        else:
            return self.combined_user_ids()

    @model_validator(mode="after")
    def _set_default_group_id(self) -> Self:
        """Defaults group_id to default gr."""
        if not self.group_id:
            self.group_id = self.from_user_id_or(AppConst.DEFAULT_GROUP_ID)
        return self

    @model_validator(mode="after")
    def _set_default_session_id(self) -> Self:
        """Defaults session_id to 'default' if not set."""
        if not self.session_id:
            self.session_id = self.from_user_id_or(AppConst.DEFAULT_SESSION_ID)
        return self

    @model_validator(mode="after")
    def _set_default_user_id(self) -> Self:
        """Defaults user_id to ['default'] if not set."""
        if len(self.user_id) == 0:
            self.user_id = [AppConst.DEFAULT_USER_ID]
        else:
            self.user_id = sorted(self.user_id)
        return self

    def is_valid(self) -> bool:
        """Return False if the session data is invalid (both group_id and
        session_id are empty), True otherwise.
        """
        return (
            self.group_id != "" and self.session_id != "" and self.first_user_id() != ""
        )
