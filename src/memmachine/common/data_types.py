"""
Common data types for MemMachine.
"""

from typing import Protocol
from pydantic import BaseModel, Field, Self, model_validator

class ExternalServiceAPIError(Exception):
    """
    Raised when an API error occurs for an external service.
    """

    pass

class SessionDataProtocol(Protocol):
    """Protocol for any object that contains session data fields."""
    
    group_id: str
    user_id: list[str]
    session_id: str
    agent_id: list[str]
