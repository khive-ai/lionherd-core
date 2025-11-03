# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import pytest

from lionherd_core.libs.string_handlers._string_similarity import SimilarityAlgo
from lionherd_core.ln._fuzzy_match import fuzzy_match_keys

# ============================================================================
# Input Validation Tests - Lines 43, 45, 47
# ============================================================================


def test_fuzzy_match_keys_non_dict_input():
    """Test TypeError when first argument is not a dictionary - line 43"""
    with pytest.raises(TypeError, match="First argument must be a dictionary"):
        fuzzy_match_keys("not a dict", ["key1"])


def test_fuzzy_match_keys_none_keys():
    """Test TypeError when keys argument is None - line 45"""
    with pytest.raises(TypeError, match="Keys argument cannot be None"):
        fuzzy_match_keys({"key1": "value1"}, None)


def test_fuzzy_match_keys_invalid_threshold_below():
    """Test ValueError for similarity_threshold below 0.0 - line 47"""
    with pytest.raises(ValueError, match=r"similarity_threshold must be between 0.0 and 1.0"):
        fuzzy_match_keys({"key1": "value1"}, ["key1"], similarity_threshold=-0.1)


def test_fuzzy_match_keys_invalid_threshold_above():
    """Test ValueError for similarity_threshold above 1.0 - line 47"""
    with pytest.raises(ValueError, match=r"similarity_threshold must be between 0.0 and 1.0"):
        fuzzy_match_keys({"key1": "value1"}, ["key1"], similarity_threshold=1.5)


# ============================================================================
# Similarity Algorithm Tests - Lines 61, 64, 67
# ============================================================================


def test_fuzzy_match_keys_similarity_algo_enum():
    """Test with SimilarityAlgo enum - line 61"""
    d = {"naem": "John", "age": 30}
    keys = ["name", "age"]
    result = fuzzy_match_keys(
        d,
        keys,
        similarity_algo=SimilarityAlgo.JARO_WINKLER,
        fuzzy_match=True,
    )
    assert "name" in result
    assert result["name"] == "John"
    assert result["age"] == 30


def test_fuzzy_match_keys_unknown_algorithm():
    """Test ValueError for unknown similarity algorithm string - line 64"""
    with pytest.raises(ValueError, match="Unknown similarity algorithm"):
        fuzzy_match_keys(
            {"key1": "value1"},
            ["key1"],
            similarity_algo="invalid_algorithm_xyz",
        )


def test_fuzzy_match_keys_custom_function():
    """Test with custom similarity function - line 67"""

    def custom_similarity(s1: str, s2: str) -> float:
        """Custom similarity: 1.0 if equal, 0.5 if similar length, else 0.0"""
        if s1 == s2:
            return 1.0
        if abs(len(s1) - len(s2)) <= 2:
            return 0.5
        return 0.0

    d = {"naem": "John", "agee": 30}
    keys = ["name", "age"]
    result = fuzzy_match_keys(
        d,
        keys,
        similarity_algo=custom_similarity,
        fuzzy_match=True,
        similarity_threshold=0.4,
    )
    # Custom function should match based on length similarity
    assert len(result) >= 2


# ============================================================================
# Handle Unmatched Tests - Lines 100, 124
# ============================================================================


def test_fuzzy_match_keys_ignore_unmatched_no_fuzzy_match():
    """Test handle_unmatched='ignore' with unmatched keys during fuzzy match - line 100"""
    d = {"exact_match": "value1", "no_match_xyz": "value2"}
    keys = ["exact_match", "expected_key"]
    result = fuzzy_match_keys(
        d,
        keys,
        fuzzy_match=True,
        handle_unmatched="ignore",
        similarity_threshold=0.95,  # High threshold to prevent fuzzy matching
    )
    # exact_match should be in result
    assert result["exact_match"] == "value1"
    # no_match_xyz should also be in result (line 100)
    assert result["no_match_xyz"] == "value2"


def test_fuzzy_match_keys_fill_with_unmatched_input():
    """Test handle_unmatched='fill' keeps unmatched input keys - line 124"""
    d = {"key1": "value1", "extra_key": "extra_value"}
    keys = ["key1", "key2"]  # key2 is expected but missing
    result = fuzzy_match_keys(
        d,
        keys,
        handle_unmatched="fill",
        fill_value="default",
        fuzzy_match=False,  # Disable fuzzy matching for clarity
    )
    # key1 should match exactly
    assert result["key1"] == "value1"
    # key2 should be filled with default
    assert result["key2"] == "default"
    # extra_key should be kept (line 124)
    assert result["extra_key"] == "extra_value"


# ============================================================================
# Edge Case Tests - Lines 83, 149
# ============================================================================


def test_fuzzy_match_keys_more_input_than_expected():
    """Test break when more input keys than expected during fuzzy match - line 83"""
    # More input keys than expected keys - should trigger break at line 83
    d = {"key1": "v1", "key2": "v2", "key3": "v3", "key4": "v4"}
    keys = ["key1", "key2"]  # Only 2 expected keys, but 4 input keys
    result = fuzzy_match_keys(
        d,
        keys,
        fuzzy_match=True,
        handle_unmatched="ignore",
    )
    # Should match the expected keys
    assert result["key1"] == "v1"
    assert result["key2"] == "v2"
    # Unmatched input keys should be ignored and included
    assert result["key3"] == "v3"
    assert result["key4"] == "v4"


def test_fuzzy_match_keys_params_call():
    """Test FuzzyMatchKeysParams __call__ method - line 149"""
    from lionherd_core.ln._fuzzy_match import FuzzyMatchKeysParams

    params = FuzzyMatchKeysParams()
    d = {"naem": "John", "age": 30}
    keys = ["name", "age"]
    result = params(d, keys)
    assert result["name"] == "John"
    assert result["age"] == 30


# ============================================================================
# Additional Integration Tests for Completeness
# ============================================================================


def test_fuzzy_match_keys_basic_exact_match():
    """Test basic exact key matching"""
    d = {"name": "John", "age": 30}
    keys = ["name", "age"]
    result = fuzzy_match_keys(d, keys)
    assert result == {"name": "John", "age": 30}


def test_fuzzy_match_keys_basic_fuzzy_match():
    """Test basic fuzzy key matching"""
    d = {"naem": "John", "age": 30}
    keys = ["name", "age"]
    result = fuzzy_match_keys(d, keys, fuzzy_match=True)
    assert result["name"] == "John"
    assert result["age"] == 30


def test_fuzzy_match_keys_empty_keys():
    """Test with empty keys list returns copy of dict"""
    d = {"key1": "value1", "key2": "value2"}
    result = fuzzy_match_keys(d, [])
    assert result == d
    assert result is not d  # Should be a copy


def test_fuzzy_match_keys_handle_unmatched_raise():
    """Test handle_unmatched='raise' raises ValueError"""
    d = {"key1": "value1", "extra_key": "value2"}
    keys = ["key1"]
    with pytest.raises(ValueError, match="Unmatched keys found"):
        fuzzy_match_keys(d, keys, handle_unmatched="raise", fuzzy_match=False)


def test_fuzzy_match_keys_handle_unmatched_force():
    """Test handle_unmatched='force' fills missing and removes unmatched"""
    d = {"key1": "value1", "extra_key": "value2"}
    keys = ["key1", "key2"]
    result = fuzzy_match_keys(
        d,
        keys,
        handle_unmatched="force",
        fill_value="default",
        fuzzy_match=False,
    )
    assert result["key1"] == "value1"
    assert result["key2"] == "default"
    assert "extra_key" not in result


def test_fuzzy_match_keys_strict_mode():
    """Test strict mode raises when expected keys missing"""
    d = {"key1": "value1"}
    keys = ["key1", "key2"]
    with pytest.raises(ValueError, match="Missing required keys"):
        fuzzy_match_keys(d, keys, strict=True, fuzzy_match=False)


def test_fuzzy_match_keys_fill_mapping():
    """Test fill_mapping provides custom defaults"""
    d = {"key1": "value1"}
    keys = ["key1", "key2", "key3"]
    result = fuzzy_match_keys(
        d,
        keys,
        handle_unmatched="fill",
        fill_mapping={"key2": "custom_value"},
        fill_value="default",
        fuzzy_match=False,
    )
    assert result["key1"] == "value1"
    assert result["key2"] == "custom_value"
    assert result["key3"] == "default"
