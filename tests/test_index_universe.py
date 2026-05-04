"""Tests for the static NIFTY universe loader."""
from __future__ import annotations

from pathlib import Path

import pytest

from stock_knight.data.index_universe import available_indices, load_index_members

REPO_ROOT = Path(__file__).resolve().parents[1]
LISTS_DIR = REPO_ROOT / "data" / "universe_lists"


def test_available_indices_matches_files() -> None:
    names = available_indices()
    assert set(names) == {"nifty50", "nifty100", "nifty150", "nifty500"}


@pytest.mark.parametrize(
    "index_name,min_count",
    [
        ("nifty50", 50),
        ("nifty100", 100),
        ("nifty150", 100),
        ("nifty500", 200),
    ],
)
def test_membership_lists_have_expected_size(index_name: str, min_count: int) -> None:
    members = load_index_members(index_name, lists_dir=LISTS_DIR)
    assert len(members) >= min_count
    assert len(set(members)) == len(members), "members should be deduplicated"


def test_unknown_index_raises() -> None:
    with pytest.raises(ValueError, match="Unknown index"):
        load_index_members("nifty1000")


def test_name_normalization() -> None:
    a = load_index_members("nifty50", lists_dir=LISTS_DIR)
    b = load_index_members("NIFTY 50", lists_dir=LISTS_DIR)
    c = load_index_members("Nifty-50", lists_dir=LISTS_DIR)
    assert a == b == c
