# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import os.path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from activitysim.core import logit, workflow
from activitysim.core.exceptions import InvalidTravelError
from activitysim.core.simulate import eval_variables


@pytest.fixture(scope="module")
def data_dir():
    return os.path.join(os.path.dirname(__file__), "data")


# this is lifted straight from urbansim's test_mnl.py
@pytest.fixture(
    scope="module",
    params=[
        (
            "fish.csv",
            "fish_choosers.csv",
            pd.DataFrame(
                [[-0.02047652], [0.95309824]], index=["price", "catch"], columns=["Alt"]
            ),
            pd.DataFrame(
                [
                    [0.2849598, 0.2742482, 0.1605457, 0.2802463],
                    [0.1498991, 0.4542377, 0.2600969, 0.1357664],
                ],
                columns=["beach", "boat", "charter", "pier"],
            ),
        )
    ],
)
def test_data(request):
    data, choosers, spec, probabilities = request.param
    return {
        "data": data,
        "choosers": choosers,
        "spec": spec,
        "probabilities": probabilities,
    }


@pytest.fixture
def choosers(test_data, data_dir):
    filen = os.path.join(data_dir, test_data["choosers"])
    return pd.read_csv(filen)


@pytest.fixture
def spec(test_data):
    return test_data["spec"]


@pytest.fixture
def utilities(choosers, spec, test_data):
    state = workflow.State().default_settings()
    vars = eval_variables(state, spec.index, choosers)
    utils = vars.dot(spec).astype("float")
    return pd.DataFrame(
        utils.values.reshape(test_data["probabilities"].shape),
        columns=test_data["probabilities"].columns,
    )


def test_validate_utils_replaces_unavailable_values():
    state = workflow.State().default_settings()
    utils = pd.DataFrame([[0.0, logit.UTIL_MIN - 1.0], [1.0, 2.0]])

    validated = logit.validate_utils(state, utils, allow_zero_probs=False)

    assert validated.iloc[0, 0] == pytest.approx(0.0)
    assert validated.iloc[0, 1] == pytest.approx(logit.UTIL_UNAVAILABLE)
    assert validated.iloc[1, 0] == pytest.approx(1.0)
    assert validated.iloc[1, 1] == pytest.approx(2.0)


def test_validate_utils_raises_when_all_unavailable():
    state = workflow.State().default_settings()
    utils = pd.DataFrame([[logit.UTIL_MIN - 1.0, logit.UTIL_MIN - 2.0]])

    with pytest.raises(InvalidTravelError) as excinfo:
        logit.validate_utils(state, utils, allow_zero_probs=False)

    assert "all probabilities are zero" in str(excinfo.value)


def test_validate_utils_allows_zero_probs():
    state = workflow.State().default_settings()
    utils = pd.DataFrame([[logit.UTIL_MIN - 1.0, logit.UTIL_MIN - 2.0]])

    validated = logit.validate_utils(state, utils, allow_zero_probs=True)

    assert (validated.iloc[0] == logit.UTIL_UNAVAILABLE).all()


def test_validate_utils_does_not_mutate_input():
    state = workflow.State().default_settings()
    utils = pd.DataFrame([[0.0, logit.UTIL_MIN - 1.0], [1.0, 2.0]])
    original = utils.copy()

    _ = logit.validate_utils(state, utils, allow_zero_probs=False)

    pdt.assert_frame_equal(utils, original)


def test_utils_to_probs_logsums_with_overflow_protection():
    state = workflow.State().default_settings()
    utils = pd.DataFrame(
        [[1000.0, 1001.0, 999.0], [-1000.0, -1001.0, -999.0]],
        columns=["a", "b", "c"],
    )
    original_utils = utils.copy()

    probs, logsums = logit.utils_to_probs(
        state,
        utils,
        trace_label=None,
        overflow_protection=True,
        return_logsums=True,
    )

    utils_np = original_utils.to_numpy()
    row_max = utils_np.max(axis=1, keepdims=True)
    exp_shifted = np.exp(utils_np - row_max)
    expected_probs = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)
    expected_logsums = pd.Series(
        np.log(exp_shifted.sum(axis=1)) + row_max.squeeze(),
        index=utils.index,
    )

    pdt.assert_frame_equal(
        probs,
        pd.DataFrame(expected_probs, index=utils.index, columns=utils.columns),
        rtol=1.0e-7,
        atol=0.0,
    )
    pdt.assert_series_equal(logsums, expected_logsums, rtol=1.0e-7, atol=0.0)


def test_utils_to_probs_warns_on_zero_probs_overflow():
    state = workflow.State().default_settings()
    utils = pd.DataFrame(
        [[logit.UTIL_MIN - 1.0, logit.UTIL_MIN - 2.0], [0.0, 0.0]],
        columns=["a", "b"],
    )

    with pytest.warns(UserWarning, match="cannot set overflow_protection"):
        probs = logit.utils_to_probs(
            state,
            utils,
            trace_label=None,
            allow_zero_probs=True,
            overflow_protection=True,
        )

    assert (probs.iloc[0] == 0.0).all()
    assert probs.iloc[1].sum() == pytest.approx(1.0)
    assert probs.iloc[1].iloc[0] == pytest.approx(0.5)
    assert probs.iloc[1].iloc[1] == pytest.approx(0.5)


def test_utils_to_probs_raises_on_float32_zero_probs_overflow():
    state = workflow.State().default_settings()
    utils = pd.DataFrame(np.array([[90.0, 0.0]], dtype=np.float32))

    with pytest.raises(ValueError, match="cannot prevent expected overflow"):
        logit.utils_to_probs(
            state,
            utils,
            trace_label=None,
            allow_zero_probs=True,
            overflow_protection=True,
        )


def test_utils_to_probs_does_not_mutate_input():
    state = workflow.State().default_settings()
    utils = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["a", "b"])
    original = utils.copy()

    _ = logit.utils_to_probs(state, utils, trace_label=None)

    pdt.assert_frame_equal(utils, original)


def test_utils_to_probs(utilities, test_data):
    state = workflow.State().default_settings()
    probs = logit.utils_to_probs(state, utilities, trace_label=None)
    pdt.assert_frame_equal(probs, test_data["probabilities"])


def test_utils_to_probs_raises():
    state = workflow.State().default_settings()
    idx = pd.Index(name="household_id", data=[1])
    with pytest.raises(RuntimeError) as excinfo:
        logit.utils_to_probs(
            state,
            pd.DataFrame([[1, 2, np.inf, 3]], index=idx),
            trace_label=None,
            overflow_protection=False,
        )
    assert "infinite exponentiated utilities" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        logit.utils_to_probs(
            state,
            pd.DataFrame([[1, 2, 9999, 3]], index=idx),
            trace_label=None,
            overflow_protection=False,
        )
    assert "infinite exponentiated utilities" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        logit.utils_to_probs(
            state,
            pd.DataFrame([[-999, -999, -999, -999]], index=idx),
            trace_label=None,
            overflow_protection=False,
        )
    assert "all probabilities are zero" in str(excinfo.value)

    # test that overflow protection works
    z = logit.utils_to_probs(
        state,
        pd.DataFrame([[1, 2, 9999, 3]], index=idx),
        trace_label=None,
        overflow_protection=True,
    )
    assert np.asarray(z).ravel() == pytest.approx(np.asarray([0.0, 0.0, 1.0, 0.0]))


def test_make_choices_only_one():
    state = workflow.State().default_settings()
    probs = pd.DataFrame(
        [[1, 0, 0], [0, 1, 0]], columns=["a", "b", "c"], index=["x", "y"]
    )
    choices, rands = logit.make_choices(state, probs)

    pdt.assert_series_equal(
        choices, pd.Series([0, 1], index=["x", "y"]), check_dtype=False
    )


def test_make_choices_matches_random_draws():
    class DummyRNG:
        def random_for_df(self, df, n=1):
            assert n == 1
            return np.array([[0.05], [0.6], [0.95]])

    class DummyState:
        @staticmethod
        def get_rn_generator():
            return DummyRNG()

    state = DummyState()
    probs = pd.DataFrame(
        [[0.1, 0.2, 0.7], [0.4, 0.4, 0.2], [0.05, 0.9, 0.05]],
        index=["a", "b", "c"],
        columns=["x", "y", "z"],
    )
    choices, rands = logit.make_choices(state, probs)

    expected_rands = np.array([0.05, 0.6, 0.95])
    expected_choices = np.array([0, 1, 1])

    pdt.assert_series_equal(
        rands,
        pd.Series(expected_rands, index=probs.index),
        check_names=False,
    )
    pdt.assert_series_equal(
        choices,
        pd.Series(expected_choices, index=probs.index),
        check_dtype=False,
    )


def test_add_ev1_random():
    class DummyRNG:
        def gumbel_for_df(self, df, n):
            # Deterministic, non-constant draws make it easy to verify
            # correct per-row/per-column addition behavior.
            row_component = df.index.to_numpy(dtype=float).reshape(-1, 1) / 10.0
            col_component = np.arange(n, dtype=float).reshape(1, -1)
            return row_component + col_component

    rng = DummyRNG()

    class DummyState:
        @staticmethod
        def get_rn_generator():
            return rng

    utilities = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=[10, 11],
        columns=["a", "b"],
    )

    randomized = logit.add_ev1_random(DummyState(), utilities)

    expected = pd.DataFrame(
        [[2.0, 4.0], [4.1, 6.1]],
        index=[10, 11],
        columns=["a", "b"],
    )

    # check that the random component was added correctly, and that the original utilities were not mutated
    pdt.assert_frame_equal(randomized, expected)
    pdt.assert_index_equal(randomized.index, utilities.index)
    pdt.assert_index_equal(randomized.columns, utilities.columns)
    pdt.assert_frame_equal(
        utilities,
        pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0]],
            index=[10, 11],
            columns=["a", "b"],
        ),
    )


def test_group_nest_names_by_level():
    nest_spec = {
        "name": "root",
        "coefficient": 1.0,
        "alternatives": [
            {"name": "motorized", "coefficient": 0.7, "alternatives": ["car", "bus"]},
            "walk",
        ],
    }

    grouped = logit.group_nest_names_by_level(nest_spec)

    assert grouped == {1: ["root"], 2: ["motorized", "walk"], 3: ["car", "bus"]}


def test_choose_from_tree_selects_leaf():
    nest_utils = pd.Series(
        {
            "motorized": 2.0,
            "walk": 1.0,
            "car": 5.0,
            "bus": 3.0,
        }
    )
    all_alternatives = {"walk", "car", "bus"}
    logit_nest_groups = {1: ["root"], 2: ["motorized", "walk"], 3: ["car", "bus"]}
    nest_alternatives_by_name = {
        "root": ["motorized", "walk"],
        "motorized": ["car", "bus"],
    }

    choice = logit.choose_from_tree(
        nest_utils, all_alternatives, logit_nest_groups, nest_alternatives_by_name
    )

    assert choice == "car"


def test_choose_from_tree_raises_on_missing_leaf():
    nest_utils = pd.Series({"motorized": 2.0, "walk": 1.0})
    all_alternatives = {"car", "bus"}
    logit_nest_groups = {1: ["root"], 2: ["motorized", "walk"]}
    nest_alternatives_by_name = {
        "root": ["motorized", "walk"],
        "motorized": ["car", "bus"],
    }

    with pytest.raises(ValueError, match="no alternative found"):
        logit.choose_from_tree(
            nest_utils, all_alternatives, logit_nest_groups, nest_alternatives_by_name
        )


def test_make_choices_eet_mnl(monkeypatch):
    def fake_add_ev1_random(_state, _df):
        return pd.DataFrame(
            [[1.0, 3.0], [4.0, 2.0]],
            index=[100, 101],
            columns=["a", "b"],
        )

    monkeypatch.setattr(logit, "add_ev1_random", fake_add_ev1_random)

    choices = logit.make_choices_explicit_error_term_mnl(
        workflow.State().default_settings(),
        pd.DataFrame([[0.0, 0.0], [0.0, 0.0]], index=[100, 101], columns=["a", "b"]),
        trace_label=None,
    )

    pdt.assert_series_equal(choices, pd.Series([1, 0], index=[100, 101]))


def test_make_choices_eet_nl(monkeypatch):
    def fake_add_ev1_random(_state, _df):
        return pd.DataFrame(
            [[5.0, 1.0, 4.0, 2.0], [3.0, 4.0, 1.0, 2.0]],
            index=[10, 11],
            columns=["motorized", "walk", "car", "bus"],
        )

    monkeypatch.setattr(logit, "add_ev1_random", fake_add_ev1_random)

    nest_spec = {
        "name": "root",
        "coefficient": 1.0,
        "alternatives": [
            {"name": "motorized", "coefficient": 0.7, "alternatives": ["car", "bus"]},
            "walk",
        ],
    }
    alt_order_array = np.array(["walk", "car", "bus"])

    choices = logit.make_choices_explicit_error_term_nl(
        workflow.State().default_settings(),
        pd.DataFrame(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            index=[10, 11],
            columns=["motorized", "walk", "car", "bus"],
        ),
        alt_order_array,
        nest_spec,
        trace_label=None,
    )

    pdt.assert_series_equal(choices, pd.Series([1, 0], index=[10, 11]))


def test_make_choices_utility_based_sets_zero_rands(monkeypatch):
    def fake_add_ev1_random(_state, df):
        return pd.DataFrame(
            [[2.0, 1.0], [0.5, 2.5]],
            index=df.index,
            columns=df.columns,
        )

    monkeypatch.setattr(logit, "add_ev1_random", fake_add_ev1_random)

    utilities = pd.DataFrame([[3.0, 2.0], [1.0, 4.0]], index=[11, 12])
    choices, rands = logit.make_choices_utility_based(
        workflow.State().default_settings(),
        utilities,
        name_mapping=np.array(["a", "b"]),
        nest_spec=None,
        trace_label=None,
    )

    expected_choices = pd.Series([0, 1], index=[11, 12])
    pdt.assert_series_equal(choices, expected_choices)
    pdt.assert_series_equal(rands, pd.Series([0, 0], index=[11, 12]))


def test_make_choices_vs_eet_same_distribution():
    """With many draws, make_choices (probability-based) and
    make_choices_explicit_error_term_mnl (EET) should produce roughly the
    same empirical choice-frequency distribution for the same utilities."""
    n_draws = 100_000
    utils_values = [5.0, 6.0, 7.0, 8.0, 9.0]
    n_alts = len(utils_values)
    columns = ["a", "b", "c", "d", "e"]

    utils = pd.DataFrame([utils_values] * n_draws, columns=columns)

    # Probability-based (Monte Carlo) path — independent RNG
    mc_rng = np.random.default_rng(42)

    class MCDummyRNG:
        def random_for_df(self, df, n=1):
            return mc_rng.random((len(df), n))

    class MCDummyState:
        @staticmethod
        def get_rn_generator():
            return MCDummyRNG()

    probs = logit.utils_to_probs(
        MCDummyState(), utils, trace_label=None, overflow_protection=True
    )
    choices_mc, _ = logit.make_choices(MCDummyState(), probs, trace_label=None)

    # Explicit-error-term (EET) path — independent RNG
    eet_rng = np.random.default_rng(123)

    class EETDummyRNG:
        def gumbel_for_df(self, df, n):
            return eet_rng.gumbel(size=(len(df), n))

    class EETDummyState:
        @staticmethod
        def get_rn_generator():
            return EETDummyRNG()

    choices_eet = logit.make_choices_explicit_error_term_mnl(
        EETDummyState(), utils, trace_label=None
    )

    mc_fracs = np.bincount(choices_mc.values.astype(int), minlength=n_alts) / n_draws
    eet_fracs = np.bincount(choices_eet.values.astype(int), minlength=n_alts) / n_draws

    np.testing.assert_allclose(mc_fracs, eet_fracs, atol=0.005)


@pytest.fixture(scope="module")
def interaction_choosers():
    return pd.DataFrame({"attr": ["a", "b", "c", "b"]}, index=["w", "x", "y", "z"])


@pytest.fixture(scope="module")
def interaction_alts():
    return pd.DataFrame({"prop": [10, 20, 30, 40]}, index=[1, 2, 3, 4])


def test_interaction_dataset_no_sample(interaction_choosers, interaction_alts):
    expected = pd.DataFrame(
        {
            "attr": ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["b"] * 4,
            "prop": [10, 20, 30, 40] * 4,
        },
        index=[1, 2, 3, 4] * 4,
    )

    interacted = logit.interaction_dataset(
        workflow.State().default_settings(), interaction_choosers, interaction_alts
    )

    interacted, expected = interacted.align(expected, axis=1)
    pdt.assert_frame_equal(interacted, expected)


def test_interaction_dataset_sampled(interaction_choosers, interaction_alts):
    class DummyRNG:
        def choice_for_df(self, df, a, size, replace=False):
            return np.array([2, 3, 0, 2, 3, 0, 1, 0])

    class DummyState:
        @staticmethod
        def get_rn_generator():
            return DummyRNG()

    expected = pd.DataFrame(
        {
            "attr": ["a"] * 2 + ["b"] * 2 + ["c"] * 2 + ["b"] * 2,
            "prop": [30, 40, 10, 30, 40, 10, 20, 10],
        },
        index=[3, 4, 1, 3, 4, 1, 2, 1],
    )

    interacted = logit.interaction_dataset(
        DummyState(),
        interaction_choosers,
        interaction_alts,
        sample_size=2,
    )

    interacted, expected = interacted.align(expected, axis=1)
    pdt.assert_frame_equal(interacted, expected)