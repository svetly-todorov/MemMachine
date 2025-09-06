from abc import ABC, abstractmethod
from typing import Any

import numpy as np


## LEGACY CODE to be fixed.
class ProfileStorageBase(ABC):
    """
    The base class for profile storage
    """

    @abstractmethod
    async def get_profile(self, user_id: str) -> dict[str, Any]:
        """
        Get profile by id
        Return: A list of KV for eatch feature and value.
           The value is an array with: feature value, feature tag and deleted, update time, create time and delete time.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_profile(self, user_id: str):
        """
        Delete all the profile by id
        """
        raise NotImplementedError

    @abstractmethod
    async def add_profile_feature(
        self,
        user_id: str,
        feature: str,
        value: str,
        tag: str,
        embedding: np.ndarray,
    ):
        """
        Add a new feature to the profile.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_profile_feature(self, user_id: str, feature: str):
        """
        Delete a feature from the profile with the key from the given user
        """
        raise NotImplementedError

    @abstractmethod
    async def get_large_profile_sections(
        self, user_id: str, thresh: int
    ) -> list[dict[str, Any]]:
        """
        get sections of profile with at least thresh entries
        """
        raise NotImplementedError

    @abstractmethod
    async def add_history(
        self,
        group_id: str = "NA",
        user_id: str = "NA",
        session_id: str = "NA",
        messages: str = "NA",
    ):
        raise NotImplementedError

    @abstractmethod
    async def delete_history(
        self,
        group_id: str = "NA",
        user_id: str = "NA",
        session_id: str = "NA",
        start_time=0,
        end_time=0,
    ):
        raise NotImplementedError

    @abstractmethod
    async def get_last_history_messages(
        self,
        group_id: str = "NA",
        user_id: str = "NA",
        session_id: str = "NA",
        message_num=0,
    ) -> list[tuple[int, str]]:
        raise NotImplementedError

    @abstractmethod
    async def get_history_message(
        self,
        group_id: str = "NA",
        user_id: str = "NA",
        session_id: str = "NA",
        start_time=0,
        end_time=0,
    ) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    async def purge_history(
        self,
        group_id: str = "NA",
        user_id: str = "NA",
        session_id: str = "NA",
        start_time=0,
    ):
        raise NotImplementedError
