from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

IsolationPropertyValue = bool | int | str
JSONValue = (
    None
    | bool
    | int
    | float
    | str
    | list["JSONValue"]
    | dict[str, "JSONValue"]
)


class ContentType(Enum):
    STRING = "string"


@dataclass(kw_only=True)
class Episode:
    uuid: UUID
    episode_type: str
    content_type: ContentType
    content: Any
    timestamp: datetime
    isolation_properties: dict[str, IsolationPropertyValue] = field(
        default_factory=dict
    )
    user_metadata: JSONValue = None


@dataclass(kw_only=True)
class EpisodeCluster:
    uuid: UUID
    episodes: list[Episode] = field(default_factory=list)
    timestamp: datetime | None = None
    isolation_properties: dict[str, IsolationPropertyValue] = field(
        default_factory=dict
    )
    user_metadata: JSONValue = None


@dataclass(kw_only=True)
class Derivative:
    uuid: UUID
    derivative_type: str
    content_type: ContentType
    content: Any
    timestamp: datetime | None = None
    isolation_properties: dict[str, IsolationPropertyValue] = field(
        default_factory=dict
    )
    user_metadata: JSONValue = None


def mangle_isolation_property_key(key: str) -> str:
    return f"isolation_{key}"


def demangle_isolation_property_key(mangled_key: str) -> str:
    return mangled_key.removeprefix("isolation_")


def is_mangled_isolation_property_key(candidate_key: str) -> bool:
    return candidate_key.startswith("isolation_")
