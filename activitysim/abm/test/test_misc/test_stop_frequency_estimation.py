"""Verify that estimation uses survey trips rather than modeled escort trips.

These tests run the component entry points with small in-memory tables. Expensive
simulation, skim, and file operations are mocked so the tests isolate trip
selection, survey overrides, and registration of the resulting trips.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from activitysim.abm.models import (
    stop_frequency,
    trip_destination,
    trip_mode_choice,
    trip_purpose,
    trip_scheduling,
)
from activitysim.abm.models.util import canonical_ids, trip


@pytest.fixture
def trip_data(monkeypatch):
    """Build a shared household, tour alternatives, and survey-backed estimator.

    The household has one work tour and one school tour. Survey choices specify
    one outbound stop per tour, whereas the simulation mocks below choose none.
    This difference makes it possible to detect a missing survey override.
    """
    # Keep canonical trip IDs deterministic without reading model configuration.
    monkeypatch.setattr(canonical_ids, "determine_max_trips_per_leg", lambda state: 4)
    state = MagicMock()
    state.settings = SimpleNamespace(
        trace_hh_id=None,
        multiprocess=False,
        downcast_int=False,
        downcast_float=False,
        chunk_size=0,
        testing_fail_trip_destination=False,
        skip_failed_choices=False,
    )
    # Use a real dictionary behind the mocked state so table reads and writes
    # behave consistently and the tests can inspect what each component saved.
    tables = {}
    state.add_table.side_effect = tables.__setitem__
    state.get_dataframe.side_effect = lambda name, **kwargs: tables[name]
    state.is_table.side_effect = tables.__contains__
    tours = pd.DataFrame(
        {
            "person_id": [11, 12],
            "household_id": [1, 1],
            "origin": [100, 100],
            "destination": [200, 300],
        },
        index=pd.Index([10, 20], name="tour_id"),
    )
    # Match the component inputs: primary_purpose is available on tours_merged
    # but is not yet attached to the underlying tours table.
    merged = tours.assign(primary_purpose=["work", "school"])
    alts = pd.DataFrame(
        {"out": [0, 1], "in": [0, 0]},
        index=["0out_0in", "1out_0in"],
    )
    # Exercise the real trip initializer: each survey tour has two outbound
    # trips and one inbound trip, for six trips across the household.
    survey = trip.initialize_from_tours(
        state, merged.assign(stop_frequency="1out_0in"), alts
    )
    # Most trips have no escort metadata. Deliberately omit the temporary
    # school_escort_trip_id, which is not required in survey trip data.
    survey["escort_participants"] = pd.NA
    survey["school_escort_direction"] = pd.NA
    survey.loc[survey.index[0], "escort_participants"] = "12"
    survey.loc[survey.index[0], "school_escort_direction"] = "outbound"
    survey["purpose"] = ["escort", "work", "home", "escort", "school", "home"]
    survey["depart"] = [7, 8, 17, 7, 8, 15]
    survey["destination"] = [150, 200, 100, 150, 300, 100]
    survey["trip_mode"] = "SURVEY_MODE"

    estimator = MagicMock()

    def survey_values(values, table_name, column_names):
        """Return the requested observed values in the modeled rows' ID order."""
        source = (
            merged.assign(stop_frequency="1out_0in")
            if table_name == "tours"
            else survey
        )
        return source.loc[values.index, column_names]

    # Components use both the estimator and the shared estimation manager to
    # retrieve survey values; make both paths read the same in-memory records.
    estimator.get_survey_values.side_effect = survey_values
    monkeypatch.setattr(
        stop_frequency.estimation.manager, "get_survey_values", survey_values
    )
    monkeypatch.setattr(
        stop_frequency.estimation.manager, "get_survey_table", lambda **kwargs: survey
    )
    # Table-annotation expressions are configuration-dependent and outside the
    # behavior under test. All components reference this shared expressions module.
    monkeypatch.setattr(stop_frequency.expressions, "annotate_tables", MagicMock())
    return SimpleNamespace(
        state=state,
        tables=tables,
        tours=tours,
        merged=merged,
        alts=alts,
        survey=survey,
        estimator=estimator,
    )


@pytest.mark.parametrize(
    "estimating,school_escorting,survey_problem",
    [
        # Cover simulation and estimation with the escort model disabled,
        # enabled without escort trips, and enabled with escort trips.
        (estimating, school_escorting, None)
        for estimating in [False, True]
        for school_escorting in ["absent", "empty", "nonempty"]
    ]
    + [
        # Corrupt one aspect of an otherwise valid survey per negative case.
        (True, "nonempty", problem)
        for problem in [
            "duplicate",
            "missing_trip",
            "extra_trip",
            "wrong_person",
            "wrong_tour",
            "wrong_household",
            "wrong_outbound",
            "invalid_direction",
            "missing_metadata",
        ]
    ],
)
def test_stop_frequency_survey_metadata(
    trip_data, monkeypatch, estimating, school_escorting, survey_problem
):
    """Check trip creation, escort metadata alignment, and invalid-survey failures.

    Valid estimation runs must use observed stop counts and metadata. Simulation
    must retain its modeled choices and call the school-escort merge when enabled.
    Invalid survey data must fail before trips are published or registered.
    """
    data = trip_data
    if school_escorting != "absent":
        data.tables["school_escort_trips"] = (
            data.survey.iloc[:1].copy()
            if school_escorting == "nonempty"
            else data.survey.iloc[:0].copy()
        )
    monkeypatch.setattr(
        stop_frequency.estimation.manager,
        "begin_estimation",
        lambda *args, **kwargs: data.estimator if estimating else None,
    )
    # Always simulate the first alternative (zero stops). Only an active
    # estimator should replace that result with the observed one-stop choice.
    spec = pd.DataFrame([[0, 0]], columns=data.alts.index)
    data.state.filesystem.read_model_spec.return_value = spec
    monkeypatch.setattr(
        stop_frequency.simulate, "eval_coefficients", lambda state, spec, *args: spec
    )
    monkeypatch.setattr(
        stop_frequency.simulate,
        "simple_simulate",
        lambda state, choosers, **kwargs: pd.Series(0, index=choosers.index),
    )
    # Test whether the simulation-only merge is dispatched, not its internals.
    # This mock leaves the newly initialized trips unchanged for inspection.
    merge = MagicMock()
    monkeypatch.setattr(
        stop_frequency.school_escort_tours_trips,
        "merge_school_escort_trips_into_pipeline",
        merge,
    )
    # Include an unused university segment to exercise empty-segment skipping
    # after the two populated segments have finished estimation.
    settings = stop_frequency.StopFrequencySettings(
        SPEC_SEGMENTS=[
            {
                "primary_purpose": purpose,
                "SPEC": "spec.csv",
                "COEFFICIENTS": "coefficients.csv",
            }
            for purpose in ["work", "school", "univ"]
        ]
    )
    # Extra households are legitimate when modeling a sample. Shuffle the survey
    # as well, so correct metadata alignment must use trip IDs, not row positions.
    other_household = data.survey.iloc[:1].copy()
    other_household.index = pd.Index([999], name="trip_id")
    other_household["household_id"] = 99
    survey = pd.concat([data.survey, other_household]).sample(frac=1, random_state=0)
    if school_escorting == "absent":
        # Runs without the escort model must not require escort survey columns.
        survey = survey.drop(columns=["escort_participants", "school_escort_direction"])
    expected_error = AssertionError
    first_trip_id = data.survey.index[0]
    if survey_problem == "duplicate":
        survey = pd.concat([survey, data.survey.iloc[:1]])
        expected_error = pd.errors.MergeError
    elif survey_problem == "missing_trip":
        survey = survey.drop(index=first_trip_id)
    elif survey_problem == "extra_trip":
        extra_trip = data.survey.iloc[:1].copy()
        extra_trip.index = pd.Index([998], name="trip_id")
        survey = pd.concat([survey, extra_trip])
    elif survey_problem in ["wrong_person", "wrong_tour", "wrong_household"]:
        column = {
            "wrong_person": "person_id",
            "wrong_tour": "tour_id",
            "wrong_household": "household_id",
        }[survey_problem]
        # Corrupt both lookup paths: get_survey_table reads this shuffled copy,
        # while the fixture's get_survey_values closure reads data.survey.
        survey.loc[first_trip_id, column] = 999
        data.survey.loc[first_trip_id, column] = 999
    elif survey_problem == "wrong_outbound":
        survey.loc[first_trip_id, "outbound"] = False
        data.survey.loc[first_trip_id, "outbound"] = False
    elif survey_problem == "invalid_direction":
        survey.loc[first_trip_id, "school_escort_direction"] = "invalid"
    elif survey_problem == "missing_metadata":
        survey = survey.drop(columns="escort_participants")
        expected_error = KeyError
    monkeypatch.setattr(
        stop_frequency.estimation.manager, "get_survey_table", lambda **kwargs: survey
    )

    # Use the same component invocation for successful runs and expected errors.
    with pytest.raises(expected_error) if survey_problem else nullcontext():
        stop_frequency.stop_frequency(
            data.state,
            data.tours,
            data.merged,
            data.alts,
            network_los=None,
            model_settings=settings,
        )
    if survey_problem:
        # A failed validation must not leave a usable-looking trips table or RNG
        # channel behind, even if the tours table was already updated.
        assert "trips" not in data.tables
        data.state.get_rn_generator().add_channel.assert_not_called()
        return

    # Two no-stop tours produce four trips; the observed outbound stops add two.
    result = data.tables["trips"]
    assert len(result) == (6 if estimating else 4)
    assert result.index.is_unique
    assert result.index.name == "trip_id"
    assert set(data.tables["tours"].stop_frequency) == {
        "1out_0in" if estimating else "0out_0in"
    }
    if estimating and school_escorting != "absent":
        for column in ["escort_participants", "school_escort_direction"]:
            # Compare values independently of categorical storage and the
            # different null representations used by pandas.
            pd.testing.assert_series_equal(
                result[column].astype(object).fillna(""),
                data.survey[column].astype(object).fillna(""),
            )
        assert isinstance(result.school_escort_direction.dtype, pd.CategoricalDtype)
    else:
        assert "escort_participants" not in result
    # Only simulation dispatches the escort merge. Both modes must register the
    # trips they publish for tracing and reproducible random-number generation.
    assert merge.call_count == int(not estimating and school_escorting != "absent")
    data.state.tracing.register_traceable_table.assert_called_once_with("trips", result)
    data.state.get_rn_generator().add_channel.assert_called_once_with("trips", result)


@pytest.mark.parametrize("estimating", [False, True])
@pytest.mark.parametrize("component", ["purpose", "destination", "scheduling", "mode"])
@pytest.mark.parametrize("school_escorting", ["absent", "empty", "nonempty"])
def test_trip_models_keep_survey_school_escort_trips(
    trip_data, monkeypatch, estimating, component, school_escorting
):
    """Keep all survey trips authoritative in each downstream estimation step.

    Conflicting modeled escort values expose accidental exclusions or overwrites.
    Companion simulation cases verify that modeled escort handling still runs.
    """
    data = trip_data
    # Remove purpose and departure time so those components must populate them.
    # Destination remains initialized, as it would after stop frequency.
    trips = data.survey.drop(
        columns=["purpose", "depart", "escort_participants", "school_escort_direction"]
    ).copy()
    # A provisional modeled ID collides with a valid survey ID. It must not be
    # used to exclude or overwrite that survey trip during estimation.
    modeled_escort = data.survey.iloc[:1].copy()
    modeled_escort["purpose"] = "shopping"
    modeled_escort["destination"] = 999
    modeled_escort["depart"] = 23
    if school_escorting == "empty":
        modeled_escort = modeled_escort.iloc[:0]
    if school_escorting != "absent":
        data.tables["school_escort_trips"] = modeled_escort
    monkeypatch.setattr(
        stop_frequency.estimation.manager,
        "begin_estimation",
        lambda *args, **kwargs: data.estimator if estimating else None,
    )
    # Record which trips actually reach each mocked model, not just which trips
    # appear in its final output. Dropped-and-restored rows could hide exclusions.
    captured = []

    if component == "purpose":
        settings = MagicMock()
        monkeypatch.setattr(
            trip_purpose.TripPurposeSettings,
            "read_settings_file",
            lambda *args: settings,
        )

        def run_purpose(state, trips, *args, **kwargs):
            """Record purpose choosers and return choices unlike the survey."""
            captured.extend(trips.index)
            return pd.Series("shopping", index=trips.index)

        monkeypatch.setattr(trip_purpose, "run_trip_purpose", run_purpose)
        trip_purpose.trip_purpose(data.state, trips)
        column = "purpose"
    elif component == "destination":
        settings = SimpleNamespace(
            fail_some_trips_for_testing=False,
            COEFFICIENTS="coefficients.csv",
            SPEC="spec.csv",
            ALT_DEST_COL_NAME="destination",
            CLEANUP=False,
        )
        data.tables["land_use"] = pd.DataFrame()

        def run_destination(state, trips, *args, estimator, **kwargs):
            """Emulate destination choices, including the runner's survey override."""
            captured.extend(trips.index)
            trips = trips.copy()
            trips["destination"] = (
                data.survey.loc[trips.index, "destination"] if estimator else 400
            )
            return trips, None

        monkeypatch.setattr(trip_destination, "run_trip_destination", run_destination)
        trip_destination.trip_destination(
            data.state, trips, data.merged, model_settings=settings
        )
        column = "destination"
    elif component == "scheduling":
        # One successful scheduling pass is enough to test the wrapper's survey
        # override; probability files and retry behavior are unrelated here.
        settings = SimpleNamespace(
            PROBS_SPEC="probs.csv",
            FAILFIX=trip_scheduling.FAILFIX_DROP_AND_CLEANUP,
            MAX_ITERATIONS=1,
            scheduling_mode="absolute",
        )
        monkeypatch.setattr(
            trip_scheduling.pd, "read_csv", lambda *args, **kwargs: None
        )

        def set_tour_hour(trips, tours):
            """Supply the timing columns needed to export estimation choosers."""
            for column in ["earliest", "latest", "tour_hour"]:
                trips[column] = 1

        monkeypatch.setattr(trip_scheduling, "set_tour_hour", set_tour_hour)
        # Process all trips in one chunk without allocating real chunk resources.
        monkeypatch.setattr(
            trip_scheduling.chunk,
            "adaptive_chunked_choosers_by_chunk_id",
            lambda state, trips, *args: [(0, trips, "test", None)],
        )
        monkeypatch.setattr(
            trip_scheduling.chunk, "chunk_log", lambda *args: nullcontext()
        )

        def run_scheduling(state, trips, *args, **kwargs):
            """Record scheduling choosers and propose a non-survey departure time."""
            captured.extend(trips.index)
            return pd.Series(12, index=trips.index)

        monkeypatch.setattr(trip_scheduling, "run_trip_scheduling", run_scheduling)
        trip_scheduling.trip_scheduling(
            data.state, trips, data.merged, model_settings=settings
        )
        column = "depart"
    else:
        # Mode choice requires departure times and tour modes from earlier steps.
        # Its current-model check must also recognize this as the actual trip
        # mode component, rather than a logsum calculation called by another step.
        trips["depart"] = data.survey.depart
        trips["tour_mode"] = "SIMULATED_MODE"
        data.state.current_model_name = "trip_mode_choice"
        if school_escorting != "absent":
            data.tables["school_escort_tours"] = pd.DataFrame()
        settings = trip_mode_choice.TripModeChoiceSettings(
            SPEC="spec.csv", TOURS_MERGED_CHOOSER_COLUMNS=[]
        )
        # Supply only the skim/time-period interface used to prepare choosers;
        # no network data or utility evaluation is needed for this regression.
        network_los = MagicMock()
        network_los.skim_time_periods.period_minutes = 60
        network_los.skim_time_period_label.return_value = pd.Series(
            "AM", index=trips.index
        )
        network_los.get_default_skim_dict().map_time_periods_from_series.return_value = (
            None
        )
        data.state.filesystem.get_segment_coefficients.return_value = {}
        monkeypatch.setattr(
            trip_mode_choice.simulate,
            "eval_coefficients",
            lambda state, spec, *args: spec,
        )
        monkeypatch.setattr(
            trip_mode_choice.simulate, "eval_nest_coefficients", lambda *args: None
        )
        monkeypatch.setattr(
            trip_mode_choice.simulate, "set_skim_wrapper_targets", MagicMock()
        )
        monkeypatch.setattr(
            trip_mode_choice.expressions, "annotate_preprocessors", MagicMock()
        )

        def mode_choice(state, choosers, **kwargs):
            """Record mode choosers and propose a mode distinct from survey modes."""
            captured.extend(choosers.index)
            return pd.DataFrame({"trip_mode": "SIMULATED_MODE"}, index=choosers.index)

        monkeypatch.setattr(trip_mode_choice, "mode_choice_simulate", mode_choice)

        def force_mode_match(state, trips):
            """Make the simulation-only chauffeur-mode overwrite easy to detect."""
            trips.loc[modeled_escort.index, "trip_mode"] = "MODELED_MATCH"
            return trips

        force_match = MagicMock(side_effect=force_mode_match)
        monkeypatch.setattr(
            trip_mode_choice.school_escort_tours_trips,
            "force_escortee_trip_modes_to_match_chauffeur",
            force_match,
        )
        trip_mode_choice.trip_mode_choice(
            data.state, trips, network_los, model_settings=settings
        )
        assert force_match.call_count == int(
            not estimating and school_escorting != "absent"
        )
        column = "trip_mode"

    # Purpose, destination, and scheduling exclude modeled escort trips only in
    # simulation. Mode choice includes them, then optionally forces a mode match.
    excludes_escort = (
        not estimating and component != "mode" and school_escorting == "nonempty"
    )
    expected_index = trips.index[1:] if excludes_escort else trips.index
    # Segment processing can change chooser order; membership must be identical.
    pd.testing.assert_index_equal(
        pd.Index(captured, name="trip_id").sort_values(), expected_index.sort_values()
    )
    # Published trips must retain their original ordering in every case. During
    # estimation, verify the complete result column, not just the escort row.
    result = data.tables["trips"]
    pd.testing.assert_index_equal(result.index, trips.index)
    if estimating:
        pd.testing.assert_series_equal(result[column], data.survey[column])
    elif component != "mode" and school_escorting == "nonempty":
        assert (
            result.loc[modeled_escort.index[0], column]
            == modeled_escort[column].iloc[0]
        )
    elif component == "mode" and school_escorting == "nonempty":
        assert result.loc[modeled_escort.index[0], column] == "MODELED_MATCH"
