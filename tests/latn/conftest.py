"""Explicit host-neutral configuration for LATN core tests."""

from contextlib import ExitStack

import pytest

from latn.lexer.lexicon import use_lexicon
from latn.lexer.sp_policy import use_sp_policy
from latn.lexer.spatial_policy import use_spatial_policy
from latn.lexer.vp_policy import use_vp_policy

from tests.latn.support import (
    TestSPPolicy,
    TestSpatialPolicy,
    TestVPPolicy,
    build_test_lexicon,
)


@pytest.fixture
def neutral_latn():
    """Activate only neutral test vocabulary and mock host policies."""
    lexicon = build_test_lexicon()
    with ExitStack() as stack:
        stack.enter_context(use_lexicon(lexicon))
        stack.enter_context(use_vp_policy(TestVPPolicy()))
        stack.enter_context(use_sp_policy(TestSPPolicy()))
        stack.enter_context(use_spatial_policy(TestSpatialPolicy()))
        yield lexicon
