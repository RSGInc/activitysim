# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import os.path
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest

from activitysim.core import chunk, simulate, workflow


@pytest.fixture
def data_dir():
    return Path(__file__).parent / "data"


@pytest.fixture
def spec_name(data_dir):
    return "sample_spec.csv"


@pytest.fixture
def state(data_dir) -> workflow.State:
    state = workflow.State().default_settings()
    state.initialize_filesystem(working_dir=Path(__file__).parent, data_dir=(data_dir,))
    return state


@pytest.fixture
def spec(state, spec_name):
    return state.filesystem.read_model_spec(file_name=spec_name)


@pytest.fixture
def data(data_dir):
    return pd.read_csv(os.path.join(data_dir, "data.csv"))


def test_read_model_spec(
    state, spec_name
):  # NOTE: this tests code not directly related to simulate
    spec = state.filesystem.read_model_spec(file_name=spec_name)

    assert len(spec) == 4
    assert spec.index.name == "Expression"
    assert list(spec.columns) == ["alt0", "alt1"]
    npt.assert_array_equal(spec.values, [[1.1, 11], [2.2, 22], [3.3, 33], [4.4, 44]])


def test_eval_variables(state, spec, data):
    result = simulate.eval_variables(state, spec.index, data)

    expected_result = [
        [1, 0, 4, 1],
        [0, 1, 4, 1],
        [0, 1, 5, 1],
    ]
    expected = pd.DataFrame(expected_result, index=data.index, columns=spec.index)

    # type-cast to match the expected result dtypes
    expected[expected.columns[0]] = expected[expected.columns[0]].astype(np.int8)
    expected[expected.columns[1]] = expected[expected.columns[1]].astype(np.int8)
    expected[expected.columns[2]] = expected[expected.columns[2]].astype(np.int64)
    expected[expected.columns[3]] = expected[expected.columns[3]].astype(int)

    print("\nexpected\n%s" % expected.dtypes)
    print("\nresult\n%s" % result.dtypes)

    pdt.assert_frame_equal(result, expected, check_names=False)


def test_simple_simulate(state, data, spec):
    state.settings.check_for_variability = False

    choices = simulate.simple_simulate(state, choosers=data, spec=spec, nest_spec=None)
    expected = pd.Series([1, 1, 1], index=data.index)
    pdt.assert_series_equal(choices, expected, check_dtype=False)


def test_simple_simulate_chunked(state, data, spec):
    state.settings.check_for_variability = False
    state.settings.chunk_size = 2
    choices = simulate.simple_simulate(
        state,
        choosers=data,
        spec=spec,
        nest_spec=None,
    )
    expected = pd.Series([1, 1, 1], index=data.index)
    pdt.assert_series_equal(choices, expected, check_dtype=False)


def test_eval_mnl_eet(state):
    # Check that the same counts are returned by eval_mnl when using EET and when not.

    num_choosers = 100_000

    np.random.seed(42)
    data2 = pd.DataFrame(
        {
            "chooser_attr": np.random.rand(num_choosers),
        },
        index=pd.Index(range(num_choosers), name="person_id"),
    )

    spec2 = pd.DataFrame(
        {"alt0": [1.0], "alt1": [2.0]},
        index=pd.Index(["chooser_attr"], name="Expression"),
    )

    # Set up a state with EET enabled
    state.settings.use_explicit_error_terms = True
    state.rng().set_base_seed(42)
    state.rng().add_channel("person_id", data2)
    state.rng().begin_step("test_step_mnl")

    chunk_sizer = chunk.ChunkSizer(state, "", "", num_choosers)

    # run eval_mnl with EET enabled
    choices_eet = simulate.eval_mnl(
        state=state,
        choosers=data2,
        spec=spec2,
        locals_d=None,
        custom_chooser=None,
        estimator=None,
        chunk_sizer=chunk_sizer,
    )

    # Reset the state, without EET enabled
    state.settings.use_explicit_error_terms = False

    state.rng().end_step("test_step_mnl")
    state.rng().begin_step("test_step_mnl")

    choices_mnl = simulate.eval_mnl(
        state=state,
        choosers=data2,
        spec=spec2,
        locals_d=None,
        custom_chooser=None,
        estimator=None,
        chunk_sizer=chunk_sizer,
    )

    # Compare counts
    mnl_counts = choices_mnl.value_counts(normalize=True)
    explicit_counts = choices_eet.value_counts(normalize=True)
    assert np.allclose(mnl_counts, explicit_counts, atol=0.01)


def test_eval_nl_eet(state):
    # Check that the same counts are returned by eval_nl when using EET and when not.

    num_choosers = 100_000

    np.random.seed(42)
    data2 = pd.DataFrame(
        {
            "chooser_attr": np.random.rand(num_choosers),
        },
        index=pd.Index(range(num_choosers), name="person_id"),
    )

    spec2 = pd.DataFrame(
        {"alt1": [2.0], "alt0.0": [0.5], "alt0.1": [0.2]},
        index=pd.Index(["chooser_attr"], name="Expression"),
    )

    nest_spec = {
        "name": "root",
        "coefficient": 1.0,
        "alternatives": [
            {"name": "alt0", "coefficient": 0.5, "alternatives": ["alt0.0", "alt0.1"]},
            "alt1",
        ],
    }

    # Set up a state with EET enabled
    state.settings.use_explicit_error_terms = True
    state.rng().set_base_seed(42)
    state.rng().add_channel("person_id", data2)
    state.rng().begin_step("test_step_mnl")

    chunk_sizer = chunk.ChunkSizer(state, "", "", num_choosers)

    # run eval_nl with EET enabled
    choices_eet = simulate.eval_nl(
        state=state,
        choosers=data2,
        spec=spec2,
        nest_spec=nest_spec,
        locals_d={},
        custom_chooser=None,
        estimator=None,
        trace_label="test",
        chunk_sizer=chunk_sizer,
    )

    # Reset the state, without EET enabled
    state.settings.use_explicit_error_terms = False

    state.rng().end_step("test_step_mnl")
    state.rng().begin_step("test_step_mnl")

    choices_mnl = simulate.eval_nl(
        state=state,
        choosers=data2,
        spec=spec2,
        nest_spec=nest_spec,
        locals_d={},
        custom_chooser=None,
        trace_label="test",
        estimator=None,
        chunk_sizer=chunk_sizer,
    )

    # Compare counts
    mnl_counts = choices_mnl.value_counts(normalize=True)
    explicit_counts = choices_eet.value_counts(normalize=True)
    assert np.allclose(mnl_counts, explicit_counts, atol=0.01)
