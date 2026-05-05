"""Benchmark ActivitySim RNG behavior in simplified location-choice scenarios.

This script times three real ActivitySim choice paths against synthetic but
model-shaped inputs:

1. Non-EET legacy: utilities are converted to probabilities and choices are
     made with the legacy per-chooser uniform random draw path.
2. EET legacy_dense: utilities receive EV1 error terms using the legacy dense
     explicit-error-term path.
3. EET keyed_hash: utilities receive EV1 error terms using the keyed hash path.

Two scenario families are supported:

- one_zone: a dense chooser x zone matrix where every chooser sees every zone
- two_zone: a simplified TAZ-to-MAZ structure where each chooser sees sampled
    MAZ alternatives nested under sampled TAZs

Jargon used in this file:

- chooser: the decision-maker, represented as one row in the benchmark input
- zone: the destination alternative being chosen among
- TAZ: a larger traffic analysis zone used in the repo's two-zone geography
- MAZ: a smaller micro analysis zone nested inside a TAZ
- EET: explicit error terms, meaning Gumbel-distributed random utility shocks
- legacy_dense: the current dense EET approach that draws enough random values
    to cover the full alternative-id span and then gathers sampled alternatives
- keyed_hash: a stateless EET approach that generates values directly from the
    chooser seed, alternative id, and offset without drawing the full dense span
- zone_id_span: the largest alternative id that the dense EET path has to cover;
    in sparse-id cases this can be much larger than the number of sampled alts

The goal is to benchmark ActivitySim's real choice code paths while keeping the
rest of the location-choice pipeline out of the measurement.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activitysim.core import logit, workflow
from activitysim.core.logit import AltsContext


# Filesystem settings.
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output" / "location_choice_rng_benchmark"

# RNG settings shared by every timed run.
# BASE_SEED controls the ActivitySim random stream state.
# INPUT_SEED controls synthetic benchmark input generation so scenario inputs are
# stable between script runs.
BASE_SEED = 0
INPUT_SEED = 987654
STEP_NAME = "rng_benchmark_step"
CHANNEL_NAME = "benchmark_choosers"
CHOOSER_INDEX_NAME = "chooser_id"

# Scenario-family switches. Keep these at the top of the script so the benchmark
# shape can be changed without expanding the command-line interface.
ENABLE_ONE_ZONE = True
ENABLE_TWO_ZONE = True

# Two-zone settings.
# We intentionally space MAZ ids apart so the legacy dense EET path has to pay
# for a larger alternative-id span than the number of sampled alternatives.
TWO_ZONE_MAZ_ID_STRIDE = 1

# Plot settings. These are globals by design so the script only needs the
# fast/full profile switch at runtime.
PLOT_DPI = 150
PLOT_FIGSIZE = (12.0, 4.8)


@dataclass(frozen=True)
class ProfileSettings:
    """Top-level sweep sizes for a benchmark profile.

    The `fast` profile is meant for quick iteration and smoke validation.
    The `full` profile is intended for a more stable runtime comparison.
    """

    repeat: int
    one_zone_fixed_choosers: int
    one_zone_fixed_zones: int
    one_zone_chooser_counts: tuple[int, ...]
    one_zone_zone_counts: tuple[int, ...]
    two_zone_fixed_choosers: int
    two_zone_fixed_taz_count: int
    two_zone_chooser_counts: tuple[int, ...]
    two_zone_taz_counts: tuple[int, ...]
    two_zone_mazs_per_taz: int
    two_zone_sampled_taz_count: int
    two_zone_sampled_mazs_per_taz: int


FAST_PROFILE = ProfileSettings(
    # Keep the fast profile small enough to run interactively while still
    # producing a useful chooser sweep and geography sweep.
    repeat=2,
    one_zone_fixed_choosers=1_000,
    one_zone_fixed_zones=200,
    one_zone_chooser_counts=(250, 1_000),
    one_zone_zone_counts=(50, 200, 800),
    two_zone_fixed_choosers=1_000,
    two_zone_fixed_taz_count=100,
    two_zone_chooser_counts=(250, 1_000),
    two_zone_taz_counts=(25, 100, 400),
    two_zone_mazs_per_taz=4,
    two_zone_sampled_taz_count=4,
    two_zone_sampled_mazs_per_taz=1,
)

FULL_PROFILE = ProfileSettings(
    # The full profile increases chooser and geography sizes to make the slope
    # of each RNG path more visible in the output plots.
    repeat=4,
    one_zone_fixed_choosers=1_000,
    one_zone_fixed_zones=6_000,
    one_zone_chooser_counts=(500, 1_000, 2_000),
    one_zone_zone_counts=(1_000, 3_000, 6_000),
    two_zone_fixed_choosers=1_000,
    two_zone_fixed_taz_count=3_000,
    two_zone_chooser_counts=(500, 1_000, 2_000),
    two_zone_taz_counts=(1_000, 3_000, 6_000),
    two_zone_mazs_per_taz=4,
    two_zone_sampled_taz_count=30,
    two_zone_sampled_mazs_per_taz=1,
)


@dataclass(frozen=True)
class ScenarioSpec:
    """A single benchmark scenario.

    `zone_count` always means the total number of zones represented by the
    geography. `utility_alt_count` means the number of alternatives actually
    present in the utility matrix passed into the choice function.
    """

    family: str
    sweep: str
    name: str
    chooser_count: int
    zone_count: int
    utility_alt_count: int
    zone_id_span: int
    taz_count: int | None = None
    sampled_taz_count: int | None = None
    sampled_mazs_per_taz: int | None = None


@dataclass
class BenchmarkInputs:
    """Prebuilt inputs for one benchmark scenario."""

    chooser_df: pd.DataFrame
    utilities: pd.DataFrame
    alt_info: AltsContext
    alt_nrs_df: pd.DataFrame
    utility_alt_count: int


@dataclass(frozen=True)
class BenchmarkResult:
    """A timed result row written to console, CSV, and plots."""

    family: str
    sweep: str
    scenario: str
    variant: str
    chooser_count: int
    zone_count: int
    utility_alt_count: int
    zone_id_span: int
    repeat: int
    mean_seconds: float
    std_seconds: float
    utility_cells_per_sec: float
    taz_count: int | None = None
    sampled_taz_count: int | None = None
    sampled_mazs_per_taz: int | None = None


# Human-readable labels used in output tables and plot legends.
VARIANT_LABELS = {
    "non_eet_legacy": "Non-EET legacy",
    "eet_legacy_dense": "EET legacy_dense",
    "eet_keyed_hash": "EET keyed_hash",
}


def parse_args() -> argparse.Namespace:
    """Parse the intentionally minimal CLI.

    Everything except profile selection lives in the global settings above.
    """

    parser = argparse.ArgumentParser(
        description="Benchmark ActivitySim location-choice RNG paths using direct logit calls."
    )
    parser.add_argument(
        "--profile",
        choices=["fast", "full"],
        default="fast",
        help="Benchmark profile size.",
    )
    return parser.parse_args()


def get_profile(profile_name: str) -> ProfileSettings:
    """Resolve the requested profile name to a concrete settings block."""

    if profile_name == "fast":
        return FAST_PROFILE
    if profile_name == "full":
        return FULL_PROFILE
    raise ValueError("profile must be 'fast' or 'full'")


def build_scenarios(profile: ProfileSettings) -> list[ScenarioSpec]:
    """Construct the scenario list for the selected profile.

    The one-zone family varies chooser count and total zone count directly.
    The two-zone family varies chooser count and TAZ count, while MAZ count and
    sparse MAZ ids are derived from the top-level settings.
    """

    scenarios: list[ScenarioSpec] = []

    if ENABLE_ONE_ZONE:
        for chooser_count in profile.one_zone_chooser_counts:
            scenarios.append(
                ScenarioSpec(
                    family="one_zone",
                    sweep="chooser_sweep",
                    name=f"one_zone_choosers_{chooser_count}",
                    chooser_count=chooser_count,
                    zone_count=profile.one_zone_fixed_zones,
                    utility_alt_count=profile.one_zone_fixed_zones,
                    zone_id_span=profile.one_zone_fixed_zones,
                )
            )
        for zone_count in profile.one_zone_zone_counts:
            scenarios.append(
                ScenarioSpec(
                    family="one_zone",
                    sweep="zone_sweep",
                    name=f"one_zone_zones_{zone_count}",
                    chooser_count=profile.one_zone_fixed_choosers,
                    zone_count=zone_count,
                    utility_alt_count=zone_count,
                    zone_id_span=zone_count,
                )
            )

    if ENABLE_TWO_ZONE:
        fixed_sampled_taz = min(
            profile.two_zone_sampled_taz_count, profile.two_zone_fixed_taz_count
        )
        fixed_zone_count = (
            profile.two_zone_fixed_taz_count * profile.two_zone_mazs_per_taz
        )
        fixed_span = 1 + (fixed_zone_count - 1) * TWO_ZONE_MAZ_ID_STRIDE

        for chooser_count in profile.two_zone_chooser_counts:
            scenarios.append(
                ScenarioSpec(
                    family="two_zone",
                    sweep="chooser_sweep",
                    name=f"two_zone_choosers_{chooser_count}",
                    chooser_count=chooser_count,
                    zone_count=fixed_zone_count,
                    utility_alt_count=(
                        fixed_sampled_taz * profile.two_zone_sampled_mazs_per_taz
                    ),
                    zone_id_span=fixed_span,
                    taz_count=profile.two_zone_fixed_taz_count,
                    sampled_taz_count=fixed_sampled_taz,
                    sampled_mazs_per_taz=profile.two_zone_sampled_mazs_per_taz,
                )
            )

        for taz_count in profile.two_zone_taz_counts:
            sampled_taz_count = min(profile.two_zone_sampled_taz_count, taz_count)
            zone_count = taz_count * profile.two_zone_mazs_per_taz
            zone_id_span = 1 + (zone_count - 1) * TWO_ZONE_MAZ_ID_STRIDE
            scenarios.append(
                ScenarioSpec(
                    family="two_zone",
                    sweep="zone_sweep",
                    name=f"two_zone_tazs_{taz_count}",
                    chooser_count=profile.two_zone_fixed_choosers,
                    zone_count=zone_count,
                    utility_alt_count=(
                        sampled_taz_count * profile.two_zone_sampled_mazs_per_taz
                    ),
                    zone_id_span=zone_id_span,
                    taz_count=taz_count,
                    sampled_taz_count=sampled_taz_count,
                    sampled_mazs_per_taz=profile.two_zone_sampled_mazs_per_taz,
                )
            )

    return scenarios


def chooser_index(chooser_count: int) -> pd.Index:
    """Create the canonical chooser index used by the RNG channel."""

    return pd.Index(np.arange(1, chooser_count + 1), name=CHOOSER_INDEX_NAME)


def chooser_frame(chooser_count: int) -> pd.DataFrame:
    """Create the minimal chooser frame needed to register an RNG channel.

    The benchmark does not need chooser attributes, but using one dummy column
    avoids the empty-domain warning emitted by the RNG channel helper.
    """

    return pd.DataFrame(
        {"_benchmark_channel": np.zeros(chooser_count, dtype=np.int8)},
        index=chooser_index(chooser_count),
    )


def build_one_zone_inputs(spec: ScenarioSpec) -> BenchmarkInputs:
    """Build dense chooser x zone utilities for the one-zone benchmark family."""

    chooser_ids = np.arange(1, spec.chooser_count + 1, dtype=np.float64)[:, np.newaxis]
    zone_ids = np.arange(1, spec.zone_count + 1, dtype=np.float64)[np.newaxis, :]

    # The utility surface is deterministic and mildly structured so the choice
    # functions do real work without needing any external model inputs.
    utilities = (
        0.0015 * chooser_ids
        + 0.006 * zone_ids
        + 0.18 * np.sin(chooser_ids * 0.013 + zone_ids * 0.041)
        + 0.04 * np.cos(zone_ids * 0.17)
    )

    chooser_df = chooser_frame(spec.chooser_count)
    utility_columns = pd.Index(range(spec.utility_alt_count), name="alt_position")
    utility_df = pd.DataFrame(
        utilities, index=chooser_df.index, columns=utility_columns
    )

    actual_zone_ids = np.arange(1, spec.zone_count + 1, dtype=np.int64)
    alt_nrs = np.broadcast_to(
        actual_zone_ids, (spec.chooser_count, spec.zone_count)
    ).copy()
    alt_nrs_df = pd.DataFrame(alt_nrs, index=chooser_df.index, columns=utility_columns)

    return BenchmarkInputs(
        chooser_df=chooser_df,
        utilities=utility_df,
        alt_info=AltsContext.from_num_alts(spec.zone_count, zero_based=False),
        alt_nrs_df=alt_nrs_df,
        utility_alt_count=spec.utility_alt_count,
    )


def build_two_zone_inputs(spec: ScenarioSpec) -> BenchmarkInputs:
    """Build sampled TAZ-to-MAZ style utilities for the two-zone family.

    The utility matrix only contains sampled MAZ alternatives, but `alt_nrs_df`
    holds their global MAZ ids so the EET legacy path still pays for the full
    MAZ id span while keyed_hash only touches sampled MAZ alternatives.
    """

    assert spec.taz_count is not None
    assert spec.sampled_taz_count is not None
    assert spec.sampled_mazs_per_taz is not None

    chooser_df = chooser_frame(spec.chooser_count)
    chooser_ids = chooser_df.index.to_numpy(dtype=np.int64)

    mazes_per_taz = spec.zone_count // spec.taz_count
    all_maz_ids = np.arange(
        1,
        1 + spec.zone_count * TWO_ZONE_MAZ_ID_STRIDE,
        TWO_ZONE_MAZ_ID_STRIDE,
        dtype=np.int64,
    )
    maz_matrix = all_maz_ids.reshape(spec.taz_count, mazes_per_taz)
    taz_ids = np.arange(1, spec.taz_count + 1, dtype=np.int64)

    # Seed the synthetic geography sampling from a separate input seed so the
    # benchmarked RNG paths are not coupled to data-generation randomness.
    scenario_seed = (
        INPUT_SEED
        + spec.chooser_count * 31
        + spec.zone_count * 17
        + spec.taz_count * 13
        + spec.utility_alt_count
    )
    rng = np.random.default_rng(scenario_seed)

    sampled_alt_ids = np.empty(
        (spec.chooser_count, spec.utility_alt_count), dtype=np.int64
    )
    sampled_taz_ids = np.empty_like(sampled_alt_ids)

    # Each chooser samples a set of TAZs, then a small number of MAZs within
    # each chosen TAZ. This approximates the shape of a production two-zone
    # location-choice problem without needing the full location_choice pipeline.
    for row in range(spec.chooser_count):
        chosen_taz_positions = rng.choice(
            spec.taz_count, size=spec.sampled_taz_count, replace=False
        )
        chosen_taz_positions.sort()
        cursor = 0
        for taz_position in chosen_taz_positions:
            local_maz = rng.choice(
                maz_matrix[taz_position],
                size=spec.sampled_mazs_per_taz,
                replace=False,
            )
            local_maz.sort()
            next_cursor = cursor + spec.sampled_mazs_per_taz
            sampled_alt_ids[row, cursor:next_cursor] = local_maz
            sampled_taz_ids[row, cursor:next_cursor] = taz_ids[taz_position]
            cursor = next_cursor

    chooser_component = chooser_ids.astype(np.float64)[:, np.newaxis]
    maz_component = sampled_alt_ids.astype(np.float64)
    taz_component = sampled_taz_ids.astype(np.float64)
    utilities = (
        0.0012 * chooser_component
        + 0.00008 * maz_component
        + 0.02 * taz_component
        + 0.16 * np.sin(chooser_component * 0.009 + taz_component * 0.37)
        + 0.03 * np.cos(maz_component * 0.005)
    )

    utility_columns = pd.Index(range(spec.utility_alt_count), name="alt_position")
    utility_df = pd.DataFrame(
        utilities, index=chooser_df.index, columns=utility_columns
    )
    alt_nrs_df = pd.DataFrame(
        sampled_alt_ids, index=chooser_df.index, columns=utility_columns
    )

    return BenchmarkInputs(
        chooser_df=chooser_df,
        utilities=utility_df,
        alt_info=AltsContext.from_series(pd.Index(all_maz_ids)),
        alt_nrs_df=alt_nrs_df,
        utility_alt_count=spec.utility_alt_count,
    )


def build_inputs(spec: ScenarioSpec) -> BenchmarkInputs:
    """Dispatch to the correct input builder for the scenario family."""

    if spec.family == "one_zone":
        return build_one_zone_inputs(spec)
    if spec.family == "two_zone":
        return build_two_zone_inputs(spec)
    raise ValueError(f"Unknown scenario family '{spec.family}'")


def make_state(
    chooser_df: pd.DataFrame, eet_error_term_rng: str | None
) -> workflow.State:
    """Create a fresh ActivitySim state for one timed run.

    A fresh state keeps RNG offsets from leaking across variants or repeats.
    """

    state = workflow.State().default_settings()
    if eet_error_term_rng is not None:
        state.settings.eet_error_term_rng = eet_error_term_rng

    rng = state.get_rn_generator()
    rng.set_base_seed(BASE_SEED)
    rng.begin_step(STEP_NAME)
    rng.add_channel(CHANNEL_NAME, chooser_df)
    return state


def run_non_eet_once(inputs: BenchmarkInputs) -> pd.Series:
    """Run the non-EET path once and return the chosen alternative positions."""

    state = make_state(inputs.chooser_df, eet_error_term_rng=None)
    probs = logit.utils_to_probs(state, inputs.utilities, trace_label=None)
    choices, _ = logit.make_choices(state, probs, trace_label=None)
    state.get_rn_generator().end_step(STEP_NAME)
    return choices


def run_eet_once(inputs: BenchmarkInputs, eet_error_term_rng: str) -> pd.Series:
    """Run one EET path once and return the chosen alternative positions."""

    state = make_state(inputs.chooser_df, eet_error_term_rng=eet_error_term_rng)
    choices, _ = logit.make_choices_utility_based(
        state,
        inputs.utilities,
        name_mapping=inputs.utilities.columns.to_numpy(),
        trace_label=None,
        alts_context=inputs.alt_info,
        alt_nrs_df=inputs.alt_nrs_df,
    )
    state.get_rn_generator().end_step(STEP_NAME)
    return choices


def benchmark_variant(
    spec: ScenarioSpec, inputs: BenchmarkInputs, variant: str, repeat: int
) -> BenchmarkResult:
    """Warm up, time, and validate one variant for one scenario.

    The initial untimed run serves two purposes: it verifies reproducibility for
    later repeats and removes first-call cold start effects from the timing.
    """

    durations: list[float] = []
    if variant == "non_eet_legacy":
        reference_choices = run_non_eet_once(inputs)
    elif variant == "eet_legacy_dense":
        reference_choices = run_eet_once(inputs, "legacy_dense")
    elif variant == "eet_keyed_hash":
        reference_choices = run_eet_once(inputs, "keyed_hash")
    else:
        raise ValueError(f"Unknown benchmark variant '{variant}'")

    for _ in range(repeat):
        gc.collect()
        t0 = time.perf_counter_ns()
        if variant == "non_eet_legacy":
            choices = run_non_eet_once(inputs)
        elif variant == "eet_legacy_dense":
            choices = run_eet_once(inputs, "legacy_dense")
        elif variant == "eet_keyed_hash":
            choices = run_eet_once(inputs, "keyed_hash")
        else:
            raise ValueError(f"Unknown benchmark variant '{variant}'")
        t1 = time.perf_counter_ns()

        if not choices.equals(reference_choices):
            raise AssertionError(
                f"Benchmark variant '{variant}' produced non-repeatable choices for {spec.name}"
            )

        durations.append((t1 - t0) / 1e9)

    arr = np.asarray(durations, dtype=np.float64)
    mean_seconds = float(arr.mean())
    std_seconds = float(arr.std(ddof=0))
    utility_cells = spec.chooser_count * inputs.utility_alt_count

    return BenchmarkResult(
        family=spec.family,
        sweep=spec.sweep,
        scenario=spec.name,
        variant=variant,
        chooser_count=spec.chooser_count,
        zone_count=spec.zone_count,
        utility_alt_count=spec.utility_alt_count,
        zone_id_span=spec.zone_id_span,
        repeat=repeat,
        mean_seconds=mean_seconds,
        std_seconds=std_seconds,
        utility_cells_per_sec=(
            0.0 if mean_seconds <= 0 else utility_cells / mean_seconds
        ),
        taz_count=spec.taz_count,
        sampled_taz_count=spec.sampled_taz_count,
        sampled_mazs_per_taz=spec.sampled_mazs_per_taz,
    )


def results_to_dataframe(results: list[BenchmarkResult]) -> pd.DataFrame:
    """Convert result rows to a DataFrame for reporting and plotting."""

    return pd.DataFrame(asdict(result) for result in results)


def write_results_csv(df: pd.DataFrame, profile_name: str, output_dir: Path) -> Path:
    """Write the benchmark results table to CSV."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"location_choice_rng_benchmark_{profile_name}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _safe_import_matplotlib():
    """Import matplotlib lazily so the benchmark only pays for plotting at the end."""

    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "matplotlib is required to generate benchmark plots. Install it in the benchmark environment."
        ) from exc


def plot_family_runtime(df: pd.DataFrame, family: str, output_dir: Path) -> list[Path]:
    """Create summary runtime plots for one scenario family."""

    plt = _safe_import_matplotlib()
    saved_paths: list[Path] = []

    family_df = df[df["family"] == family].copy()
    if family_df.empty:
        return saved_paths

    fig, axes = plt.subplots(1, 2, figsize=PLOT_FIGSIZE)
    sweep_specs = [
        ("chooser_sweep", "chooser_count", "Chooser count"),
        (
            "zone_sweep",
            "zone_count",
            "Zone count" if family == "one_zone" else "Total MAZ count",
        ),
    ]

    for ax, (sweep, x_field, x_label) in zip(axes, sweep_specs):
        subset = family_df[family_df["sweep"] == sweep].copy()
        if subset.empty:
            ax.set_visible(False)
            continue

        for variant in VARIANT_LABELS:
            variant_rows = subset[subset["variant"] == variant].sort_values(x_field)
            ax.plot(
                variant_rows[x_field],
                variant_rows["mean_seconds"],
                marker="o",
                linewidth=2.0,
                markersize=5,
                label=VARIANT_LABELS[variant],
            )

        ax.set_title(sweep.replace("_", " ").title())
        ax.set_xlabel(x_label)
        ax.set_ylabel("Mean runtime (s)")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        # Reserve a separate band below the suptitle for the figure-level legend
        # so it does not overlap with the title text.
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.93),
            ncol=3,
            frameon=False,
        )
    fig.suptitle(f"{family.replace('_', ' ').title()} runtime", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.82])

    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = output_dir / f"{family}_runtime_summary.png"
    fig.savefig(runtime_path, dpi=PLOT_DPI)
    plt.close(fig)
    saved_paths.append(runtime_path)

    if (
        family == "two_zone"
        and (family_df["zone_id_span"] != family_df["zone_count"]).any()
    ):
        subset = family_df[family_df["sweep"] == "zone_sweep"].copy()
        if not subset.empty:
            fig, ax = plt.subplots(figsize=(6.2, 4.8))
            for variant in VARIANT_LABELS:
                variant_rows = subset[subset["variant"] == variant].sort_values(
                    "zone_id_span"
                )
                ax.plot(
                    variant_rows["zone_id_span"],
                    variant_rows["mean_seconds"],
                    marker="o",
                    linewidth=2.0,
                    markersize=5,
                    label=VARIANT_LABELS[variant],
                )
            ax.set_title("Two Zone runtime vs MAZ id span")
            ax.set_xlabel("MAZ id span")
            ax.set_ylabel("Mean runtime (s)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            fig.tight_layout()
            span_path = output_dir / "two_zone_runtime_vs_maz_id_span.png"
            fig.savefig(span_path, dpi=PLOT_DPI)
            plt.close(fig)
            saved_paths.append(span_path)

    return saved_paths


def write_plots(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Write all runtime plots for the completed benchmark run."""

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for old_plot in plot_dir.glob("*.png"):
        old_plot.unlink()

    saved_paths: list[Path] = []
    for family in ["one_zone", "two_zone"]:
        saved_paths.extend(plot_family_runtime(df, family, plot_dir))
    return saved_paths


def print_results(df: pd.DataFrame) -> None:
    """Print a compact tabular summary to the console."""

    header = (
        f"{'Family':10} {'Sweep':13} {'Variant':18} {'Choosers':>9} {'Zones':>9} "
        f"{'UtilAlts':>9} {'ZoneSpan':>10} {'Mean(s)':>10} {'Std(s)':>10} {'Cells/s':>12}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in df.sort_values(
        ["family", "sweep", "variant", "chooser_count", "zone_count"]
    ).itertuples(index=False):
        print(
            f"{row.family:10} {row.sweep:13} {VARIANT_LABELS[row.variant]:18} {row.chooser_count:9d} {row.zone_count:9d} "
            f"{row.utility_alt_count:9d} {row.zone_id_span:10d} {row.mean_seconds:10.6f} {row.std_seconds:10.6f} "
            f"{row.utility_cells_per_sec:12.1f}"
        )


def main() -> None:
    """Run the benchmark for the selected profile and write its outputs."""

    args = parse_args()
    profile = get_profile(args.profile)
    scenarios = build_scenarios(profile)

    results: list[BenchmarkResult] = []
    for spec in scenarios:
        inputs = build_inputs(spec)
        for variant in VARIANT_LABELS:
            print(
                f"Running scenario '{spec.name}' variant '{variant}' with {inputs.chooser_df.shape[0]} choosers and "
                f"{inputs.utilities.shape[1]} utility alternatives..."
            )
            results.append(benchmark_variant(spec, inputs, variant, profile.repeat))

    print("\nBenchmark complete. Processing results...")
    df = results_to_dataframe(results)
    print_results(df)

    output_dir = OUTPUT_DIR / args.profile
    csv_path = write_results_csv(df, args.profile, output_dir)
    plot_paths = write_plots(df, output_dir)

    print(f"\nWrote CSV: {csv_path}")
    for plot_path in plot_paths:
        print(f"Wrote plot: {plot_path}")


if __name__ == "__main__":
    main()
