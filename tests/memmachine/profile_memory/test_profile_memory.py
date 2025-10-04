"""Unit tests for the profile_memory module."""

import time

import pytest

from memmachine.profile_memory.profile_memory import (
    ProfileUpdateTracker,
    ProfileUpdateTrackerManager,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def profile_tracker():
    return ProfileUpdateTracker("a", message_limit=2, time_limit_sec=0.1)


def test_profile_tracker_expires(profile_tracker):
    assert not profile_tracker.should_update()
    profile_tracker.mark_update()
    assert not profile_tracker.should_update()
    time.sleep(0.15)
    assert profile_tracker.should_update()


def test_profile_tracker_message_limit(profile_tracker):
    assert not profile_tracker.should_update()
    profile_tracker.mark_update()
    assert not profile_tracker.should_update()
    profile_tracker.mark_update()
    assert profile_tracker.should_update()
    profile_tracker.reset()
    assert not profile_tracker.should_update()


@pytest.fixture
def profile_update_tracker_manager():
    return ProfileUpdateTrackerManager(message_limit=2, time_limit_sec=0.1)


async def test_profile_update_tracker_manager_with_message_limit(
    profile_update_tracker_manager,
):
    users = await profile_update_tracker_manager.get_users_to_update()
    assert users == []

    for user in ["a", "b", "a", "a"]:
        await profile_update_tracker_manager.mark_update(user)

    users = await profile_update_tracker_manager.get_users_to_update()
    assert set(users) == {"a"}

    for user in ["b", "a"]:
        await profile_update_tracker_manager.mark_update(user)
    users = await profile_update_tracker_manager.get_users_to_update()
    assert set(users) == {"b"}


async def test_profile_update_tracker_manager_with_time_limit(
    profile_update_tracker_manager,
):
    users = await profile_update_tracker_manager.get_users_to_update()
    assert users == []

    await profile_update_tracker_manager.mark_update("a")
    await profile_update_tracker_manager.mark_update("b")
    users = await profile_update_tracker_manager.get_users_to_update()
    assert users == []

    time.sleep(0.15)
    users = await profile_update_tracker_manager.get_users_to_update()
    assert set(users) == {"a", "b"}
