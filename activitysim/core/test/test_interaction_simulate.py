# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from activitysim.core import interaction_simulate, workflow


@pytest.fixture
def state() -> workflow.State:
    state = workflow.State().default_settings()
    state.settings.check_for_variability = False
    return state


def test_interaction_simulate_explicit_error_terms_parity(state):
    # Run interaction_simulate with and without explicit error terms and check that results are similar.

    # Set up a simple case: 10000 choosers, 5 alternatives for better statistical convergence
    num_choosers = 100_000
    num_alts = 10
    sample_size = num_alts

    # Create random choosers and alternatives
    np.random.seed(42)
    choosers = pd.DataFrame(
        {"chooser_attr": np.random.rand(num_choosers)},
        index=pd.Index(range(num_choosers), name="person_id"),
    )

    alternatives = pd.DataFrame(
        {"alt_attr": np.random.rand(num_alts)},
        index=pd.Index(range(num_alts), name="alt_id"),
    )

    # Simple spec: utility = chooser_attr * alt_attr
    spec = pd.DataFrame(
        {"coefficient": [1.0]},
        index=pd.Index(["chooser_attr * alt_attr"], name="Expression"),
    )

    # Run _without_ explicit error terms
    state.settings.use_explicit_error_terms = False
    state.rng().set_base_seed(42)  # Set seed BEFORE adding channels or steps
    state.rng().add_channel("person_id", choosers)
    state.rng().begin_step("test_step_mnl")
    
    choices_mnl = interaction_simulate.interaction_simulate(
        state,
        choosers,
        alternatives,
        spec,
        sample_size=sample_size,
    )

    # Run _with_ explicit error terms
    state.init_state() # reset the state to rerun with same seed
    state.settings.use_explicit_error_terms = True
    state.rng().set_base_seed(42)
    state.rng().add_channel("person_id", choosers)
    state.rng().begin_step("test_step_explicit")

    choices_explicit = interaction_simulate.interaction_simulate(
        state,
        choosers,
        alternatives,
        spec,
        sample_size=sample_size,
    )

    mnl_counts = choices_mnl.value_counts(normalize=True).sort_index()
    explicit_counts = choices_explicit.value_counts(normalize=True).sort_index()

    # Check that they aren't wildly different (e.g., within 1% share for each alt)
    for alt in alternatives.index:
        share_mnl = mnl_counts.get(alt, 0)
        share_explicit = explicit_counts.get(alt, 0)
        assert abs(share_mnl - share_explicit) < 0.01, f"Large discrepancy at alt {alt}: {share_mnl} vs {share_explicit}"