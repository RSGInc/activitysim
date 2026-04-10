"""
Tests confirming the SCHEDULE_ID in run_trip_scheduling_choice is not chunk-sensitive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from activitysim.abm.models import trip_scheduling_choice as tsc


def _make_two_way_stop_tours(tour_ids, duration=4):
    """Return a minimal tours DataFrame where every tour has stops on both legs."""
    n = len(tour_ids)
    return pd.DataFrame(
        {
            tsc.TOUR_DURATION_COLUMN: [duration] * n,
            tsc.NUM_OB_STOPS: [1] * n,
            tsc.NUM_IB_STOPS: [1] * n,
            tsc.HAS_OB_STOPS: [True] * n,
            tsc.HAS_IB_STOPS: [True] * n,
        },
        index=pd.Index(tour_ids, name="tour_id"),
    )


def test_schedule_ids_shift_with_different_co_chunked_tours():
    """
    Confirm that SCHEDULE_IDs assigned to a given tour's alternatives do not change
    depending on which other tours are present in the same chunk.

    generate_schedule_alternatives numbers alternatives sequentially starting at 1
    across the full set of input tours.  A tour processed alongside tours with lower
    IDs will therefore have its alternatives numbered beginning at a higher offset
    than if it were processed alone.
    """
    tours_both = _make_two_way_stop_tours([0, 1], duration=4)
    tours_solo = _make_two_way_stop_tours([1], duration=4)

    alts_both = tsc.generate_schedule_alternatives(tours_both)
    alts_solo = tsc.generate_schedule_alternatives(tours_solo)

    ids_with_tour0 = alts_both.loc[alts_both.index == 1, tsc.SCHEDULE_ID].values
    ids_without_tour0 = alts_solo.loc[alts_solo.index == 1, tsc.SCHEDULE_ID].values

    # Same number of schedule alternatives for tour 1 regardless of co-tours
    assert len(ids_with_tour0) == len(ids_without_tour0), (
        "Tour 1 should have the same number of alternatives whether processed "
        "alone or together with tour 0."
    )

    # and the IDs themselves don't differ
    assert np.array_equal(
        ids_with_tour0, ids_without_tour0
    ), "SCHEDULE_IDs for tour 1 changed when tour 0 was added to the chunk."


def test_shifted_schedule_ids_produce_same_gumbel_draws():
    """
    Confirm that the SCHEDULE_ID shift documented in
    test_schedule_ids_shift_with_different_co_chunked_tours translates directly
    into different Gumbel error terms under the AltsContext indexing scheme.

    add_ev1_random generates a dense array of random numbers with length
    alt_info.n_alts_to_cover_max_id, then selects per-alternative values via
    np.take_along_axis indexed by the SCHEDULE_IDs.  When those IDs change, the
    selected values change too — meaning a tour can receive different error terms
    (and make a different choice) solely because of who else is in its chunk.
    """
    tours_both = _make_two_way_stop_tours([0, 1], duration=4)
    tours_solo = _make_two_way_stop_tours([1], duration=4)

    alts_both = tsc.generate_schedule_alternatives(tours_both)
    alts_solo = tsc.generate_schedule_alternatives(tours_solo)

    ids_with_tour0 = alts_both.loc[alts_both.index == 1, tsc.SCHEDULE_ID].values
    ids_without_tour0 = alts_solo.loc[alts_solo.index == 1, tsc.SCHEDULE_ID].values

    # Reproduce the dense random draw that add_ev1_random would make for tour 1.
    # Use a fixed seed to make the test deterministic.
    max_alt_id_both = int(alts_both[tsc.SCHEDULE_ID].max())
    rng = np.random.RandomState(42)
    # n_alts_to_cover_max_id = max_alt_id + 1  (see AltsContext.__post_init__)
    rands_dense = rng.gumbel(size=max_alt_id_both + 1)

    gumbel_with_tour0 = rands_dense[ids_with_tour0]

    # For the solo run, the dense array is shorter; regenerate from the same seed
    max_alt_id_solo = int(alts_solo[tsc.SCHEDULE_ID].max())
    rng2 = np.random.RandomState(42)
    rands_dense_solo = rng2.gumbel(size=max_alt_id_solo + 1)

    gumbel_without_tour0 = rands_dense_solo[ids_without_tour0]

    assert np.array_equal(gumbel_with_tour0, gumbel_without_tour0), (
        "Gumbel draws for tour 1's alternatives should not differ when SCHEDULE_IDs "
        "are shifted by the presence of tour 0."
    )
