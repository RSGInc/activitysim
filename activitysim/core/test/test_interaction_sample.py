# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from activitysim.core import interaction_sample, workflow


@pytest.fixture
def state() -> workflow.State:
    state = workflow.State().default_settings()
    state.settings.check_for_variability = False
    return state


def test_interaction_sample_parity(state):
    # Run interaction_sample with and without explicit error terms and check that results are similar.

    num_choosers = 100_000
    num_alts = 100
    sample_size = 10

    # Create random choosers and alternatives
    rng = np.random.default_rng(42)
    choosers = pd.DataFrame(
        {"chooser_attr": rng.random(num_choosers)},
        index=pd.Index(range(num_choosers), name="person_id"),
    )

    alternatives = pd.DataFrame(
        {"alt_attr": rng.random(num_alts)},
        index=pd.Index(range(num_alts), name="alt_id"),
    )

    # Simple spec: utility = chooser_attr * alt_attr
    spec = pd.DataFrame(
        {"coefficient": [1.0]},
        index=pd.Index(["chooser_attr * alt_attr"], name="Expression"),
    )

    # Run _without_ explicit error terms
    state.settings.use_explicit_error_terms = False
    state.rng().set_base_seed(42)
    state.rng().add_channel("person_id", choosers)
    state.rng().begin_step("test_step_mnl")

    choices_mnl = interaction_sample.interaction_sample(
        state,
        choosers,
        alternatives,
        spec,
        sample_size=sample_size,
        alt_col_name="alt_id",
    )

    # Run _with_ explicit error terms
    state.init_state()  # reset the state to rerun with same seed
    state.settings.use_explicit_error_terms = True
    state.rng().set_base_seed(42)
    state.rng().add_channel("person_id", choosers)
    state.rng().begin_step("test_step_explicit")

    choices_explicit = interaction_sample.interaction_sample(
        state,
        choosers,
        alternatives,
        spec,
        sample_size=sample_size,
        alt_col_name="alt_id",
    )

    assert "alt_id" in choices_mnl.columns
    assert "alt_id" in choices_explicit.columns
    assert not choices_mnl["alt_id"].isna().any()
    assert not choices_explicit["alt_id"].isna().any()
    assert choices_mnl["alt_id"].isin(alternatives.index).all()
    assert choices_explicit["alt_id"].isin(alternatives.index).all()

    # In interaction_sample, choices_explicit and choices_mnl are DataFrames with sampled alternatives.
    # The statistics of chosen alternatives should be similar.
    mnl_counts = choices_mnl["alt_id"].value_counts(normalize=True).sort_index()
    explicit_counts = (
        choices_explicit["alt_id"].value_counts(normalize=True).sort_index()
    )

    # Check top choices overlap significantly or shares are close
    all_alts = set(mnl_counts.index) | set(explicit_counts.index)
    for alt in all_alts:
        share_mnl = mnl_counts.get(alt, 0)
        share_explicit = explicit_counts.get(alt, 0)
        diff = abs(share_mnl - share_explicit)
        assert diff < 0.05, (
            f"Large discrepancy at alt {alt}: "
            f"mnl={share_mnl:.4f}, explicit={share_explicit:.4f}, diff={diff:.4f}"
        )


def test_interaction_sample_eet_unavailable_alternatives(state):
    # Test that EET handles unavailable alternatives in sampling
    num_choosers = 100
    num_alts = 10
    sample_size = 2
    choosers = pd.DataFrame(
        {"chooser_attr": np.ones(num_choosers)},
        index=pd.Index(range(num_choosers), name="person_id"),
    )

    # Alt 0-4 are attractive, Alt 5-9 are "unavailable"
    alternatives = pd.DataFrame(
        {"alt_attr": [10.0] * 5 + [-1000.0] * 5},
        index=pd.Index(range(num_alts), name="alt_id"),
    )

    spec = pd.DataFrame(
        {"coefficient": [1.0]},
        index=pd.Index(["alt_attr"], name="Expression"),
    )

    # Run with EET
    state.settings.use_explicit_error_terms = True
    state.rng().set_base_seed(42)
    state.rng().add_channel("person_id", choosers)
    state.rng().begin_step("test_unavailable_eet")

    choices_eet = interaction_sample.interaction_sample(
        state,
        choosers,
        alternatives,
        spec,
        sample_size=sample_size,
        alt_col_name="alt_id",
    )

    # Sampled alternatives should only be from Alt 0-4
    assert choices_eet["alt_id"].isin([0, 1, 2, 3, 4]).all()
    assert not choices_eet["alt_id"].isin([5, 6, 7, 8, 9]).any()
