"""Check that estimation uses the same tour-purpose segments as simulation."""

from importlib import import_module
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("larch")

mode_choice = import_module("activitysim.estimation.larch.mode_choice")


@pytest.fixture
def mode_choice_bundle(monkeypatch):
    choosers = pd.DataFrame(
        {
            "tour_type": ["school", "school", "work"],
            "tour_purpose": ["school", "univ", "work"],
            "is_university": [False, True, False],
            "override_choice": ["WALK", "DRIVEALONE", "WALK"],
            "util_asc": [1.0, 1.0, 1.0],
        },
        index=pd.Index([101, 205, 309], name="tour_id"),
    )
    purposes = ["school", "univ", "work", "atwork"]
    data = SimpleNamespace(
        chooser_data=choosers,
        coefficients=pd.DataFrame(
            {"value": [0.0] * 4, "constrain": ["F"] * 4},
            index=pd.Index([f"asc_{p}" for p in purposes], name="coefficient_name"),
        ),
        coef_template=pd.DataFrame(
            {p: [f"asc_{p}"] for p in purposes},
            index=pd.Index(["asc"], name="coefficient_name"),
        ),
        spec=pd.DataFrame(
            {
                "Label": ["util_asc"],
                "Description": ["constant"],
                "Expression": ["1"],
                "WALK": ["asc"],
                "DRIVEALONE": ["asc"],
            }
        ),
        settings={"NESTS": {}},
        alt_names=["WALK", "DRIVEALONE"],
        alt_codes=[1, 2],
        alt_names_to_codes={"WALK": 1, "DRIVEALONE": 2},
        alt_codes_to_names={1: "WALK", 2: "DRIVEALONE"},
    )
    monkeypatch.setattr(mode_choice, "simple_simulate_data", lambda **kwargs: data)
    monkeypatch.setattr(
        mode_choice,
        "construct_availability",
        lambda model, choosers, codes: pd.DataFrame(
            1, index=choosers.index, columns=codes
        ),
    )
    return data


def case_ids(group):
    return {model.title: list(model.datatree.dc.caseids()) for model in group}


@pytest.mark.parametrize("name", ["tour_mode_choice", "custom_tour_bundle"])
def test_tour_mode_choice_separates_school_and_university(mode_choice_bundle, name):
    group = mode_choice.tour_mode_choice_model(name=name)
    assert case_ids(group) == {"school": [101], "univ": [205], "work": [309]}


def test_trip_mode_choice_keeps_tour_type_segmentation(mode_choice_bundle):
    # A trip bundle need not have the tour-only derived segmentation column.
    mode_choice_bundle.chooser_data.drop(columns="tour_purpose", inplace=True)
    group = mode_choice.trip_mode_choice_model()
    assert case_ids(group) == {"school": [101, 205], "univ": [], "work": [309]}


def test_atwork_mode_choice_keeps_all_cases(mode_choice_bundle):
    mode_choice_bundle.chooser_data.drop(columns="tour_purpose", inplace=True)
    group = mode_choice.atwork_subtour_mode_choice_model()
    assert case_ids(group) == {"atwork": [101, 205, 309]}
