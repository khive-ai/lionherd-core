# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from lionherd_core.lndl.prompt import LNDL_SYSTEM_PROMPT, get_lndl_system_prompt


def test_get_lndl_system_prompt():
    """Test get_lndl_system_prompt returns stripped constant."""
    result = get_lndl_system_prompt()

    assert result == LNDL_SYSTEM_PROMPT.strip()
    assert "LNDL - Structured Output" in result
    assert "OUT{" in result
