from __future__ import annotations

# ActivitySim
# See full license in LICENSE.txt.
import importlib.resources
import os
import subprocess
import sys

import pandas as pd
import pandas.testing as pdt

from activitysim.core.test import assert_frame_substantively_equal


def _test_arc(recode=False, sharrow=False, eet=False):
    def example_path(dirname):
        resource = os.path.join("examples", "prototype_arc", dirname)
        return str(importlib.resources.files("activitysim").joinpath(resource))

    def test_path(dirname):
        return os.path.join(os.path.dirname(__file__), dirname)

    def regress():
        if sharrow:
            # sharrow results in tiny changes (one trip moving one time period earlier)
            regress_trips_df = pd.read_csv(
                test_path(f"regress/final_trips{'_eet' if eet else ''}_sh.csv")
            )
        else:
            regress_trips_df = pd.read_csv(
                test_path(f"regress/final_trips{'_eet' if eet else ''}.csv")
            )
        final_trips_df = pd.read_csv(test_path("output/final_trips.csv"))

        # person_id,household_id,tour_id,primary_purpose,trip_num,outbound,trip_count,purpose,
        # destination,origin,destination_logsum,depart,trip_mode,mode_choice_logsum
        # compare_cols = []
        assert_frame_substantively_equal(final_trips_df, regress_trips_df)

    file_path = os.path.join(os.path.dirname(__file__), "simulation.py")

    test_configs = []
    if eet:
        test_configs.extend(["-c", test_path("configs_eet")])

    if recode:
        test_configs.extend(["-c", test_path("configs_recode")])
    elif sharrow:
        test_configs.extend(["-c", test_path("configs_sharrow")])
    else:
        test_configs.extend(["-c", test_path("configs")])

    run_args = [
        *test_configs,
        "-c",
        example_path("configs"),
        "-d",
        example_path("data"),
        "-o",
        test_path("output"),
    ]

    if os.environ.get("GITHUB_ACTIONS") == "true":
        subprocess.run(["coverage", "run", "-a", file_path] + run_args, check=True)
    else:
        subprocess.run([sys.executable, file_path] + run_args, check=True)

    regress()


def test_arc():
    _test_arc()


def test_arc_eet():
    _test_arc(eet=True)


def test_arc_recode():
    _test_arc(recode=True)


def test_arc_sharrow():
    _test_arc(sharrow=True)


if __name__ == "__main__":
    _test_arc()
    _test_arc(eet=True)
    _test_arc(recode=True)
    _test_arc(sharrow=True)
