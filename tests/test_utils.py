"""Tests for small helpers in src.utils."""

from src import utils
from src.config import settings


def test_is_dev_env_true(monkeypatch):
    monkeypatch.setattr(settings, "ENV", "dev")
    assert utils.is_dev_env() is True


def test_is_dev_env_false(monkeypatch):
    monkeypatch.setattr(settings, "ENV", "prod")
    assert utils.is_dev_env() is False
