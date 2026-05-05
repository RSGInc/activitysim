from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from benchmark_rng import (
    RNGCandidate,
    activitysim_row_seeds,
    available_candidates,
    summarize_times,
    timed_repeats,
)


DEFAULT_DRAW_COUNT = 10_000
DEFAULT_OFFSET = 128
DEFAULT_MAZ_ALT_ID = 1_023
DEFAULT_EET_ALT_COUNT = 1_000
DEFAULT_OUTPUT_DIRNAME = "rng_runtime_breakdown"
FAST_REPEAT = 1
FULL_REPEAT = 3

_MASK_64 = np.uint64((1 << 64) - 1)
_HASH_GOLDEN_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_HASH_MUL1 = np.uint64(0xBF58476D1CE4E5B9)
_HASH_MUL2 = np.uint64(0x94D049BB133111EB)
_HASH_ALT_TAG = np.uint64(0x51D7348C2F9A3B17)


@dataclass(frozen=True)
class ScenarioSpec:
    key: str
    name: str
    label: str
    description: str
    chooser_count: int
    sample_size: int
    use_reseed: bool
    use_offset: bool
    use_sparse_maz: bool

    @property
    def useful_values(self) -> int:
        return self.chooser_count * self.sample_size


@dataclass(frozen=True)
class EngineSpec:
    key: str
    label: str
    engine_family: str
    drop_in_compatible: bool
    algorithm: str
    candidate: RNGCandidate | None = None
    applicable_scenarios: tuple[str, ...] | None = None


@dataclass(frozen=True)
class RuntimeResult:
    scenario: str
    scenario_label: str
    engine: str
    engine_family: str
    drop_in_compatible: bool
    chooser_count: int
    sample_size: int
    useful_values: int
    generated_values: int
    waste_factor: float
    repeat: int
    mean_seconds: float
    std_seconds: float
    useful_values_per_sec: float
    generated_values_per_sec: float


def _format_int(value: int) -> str:
    return f"{value:,}"


def _engine_family_title(engine_family: str) -> str:
    if engine_family == "drop_in_dense":
        return "Drop-in Dense"
    if engine_family == "structural_eet":
        return "Structural EET"
    return engine_family.replace("_", " ").title()


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Draw-timing benchmark for dense, offset, sparse, and EET-like RNG conditions."
    )
    parser.add_argument(
        "--profile",
        choices=["fast", "full"],
        default="full",
        help="Repeat profile to use for timings.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="Override repeat count for each scenario.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "output" / DEFAULT_OUTPUT_DIRNAME,
        help="Directory for CSV output and charts.",
    )
    return parser.parse_args()


def scenario_specs() -> list[ScenarioSpec]:
    chooser_count_text = _format_int(DEFAULT_DRAW_COUNT)
    eet_alt_count_text = _format_int(DEFAULT_EET_ALT_COUNT)
    return [
        ScenarioSpec(
            key="bulk_draw",
            name=f"bulk_draw_{DEFAULT_DRAW_COUNT}",
            label=f"Draw {chooser_count_text} values",
            description=(
                f"One continuous stream draws {chooser_count_text} uniform values in a single call."
            ),
            chooser_count=DEFAULT_DRAW_COUNT,
            sample_size=1,
            use_reseed=False,
            use_offset=False,
            use_sparse_maz=False,
        ),
        ScenarioSpec(
            key="activitysim_reseed",
            name="activitysim_reseed",
            label="ActivitySim reseed per chooser",
            description=(
                f"{chooser_count_text} chooser rows, one draw per chooser, reseeded using ActivitySim-style row seeds."
            ),
            chooser_count=DEFAULT_DRAW_COUNT,
            sample_size=1,
            use_reseed=True,
            use_offset=False,
            use_sparse_maz=False,
        ),
        ScenarioSpec(
            key="offset_only",
            name="offset_only",
            label="Offset without reseed",
            description=(
                f"One continuous stream consumes an offset and then draws {chooser_count_text} uniform values."
            ),
            chooser_count=DEFAULT_DRAW_COUNT,
            sample_size=1,
            use_reseed=False,
            use_offset=True,
            use_sparse_maz=False,
        ),
        ScenarioSpec(
            key="reseed_plus_offset",
            name="reseed_plus_offset",
            label="Reseed plus offset",
            description=(
                f"{chooser_count_text} chooser rows, one draw per chooser, with ActivitySim-style reseeding and a nonzero offset."
            ),
            chooser_count=DEFAULT_DRAW_COUNT,
            sample_size=1,
            use_reseed=True,
            use_offset=True,
            use_sparse_maz=False,
        ),
        ScenarioSpec(
            key="reseed_offset_sparse_maz",
            name="reseed_offset_sparse_maz",
            label="Reseed + offset + sparse MAZ id",
            description=(
                f"{chooser_count_text} chooser rows, one sparse MAZ id per chooser, with dense coverage to MAZ id {DEFAULT_MAZ_ALT_ID}."
            ),
            chooser_count=DEFAULT_DRAW_COUNT,
            sample_size=1,
            use_reseed=True,
            use_offset=True,
            use_sparse_maz=True,
        ),
        ScenarioSpec(
            key="reseed_offset_sparse_maz_eet",
            name=f"reseed_offset_sparse_maz_{DEFAULT_EET_ALT_COUNT}_alts",
            label=f"Reseed + offset + sparse MAZ ids + {eet_alt_count_text} alts",
            description=(
                f"{chooser_count_text} chooser rows with {eet_alt_count_text} sparse sampled alternatives per chooser."
            ),
            chooser_count=DEFAULT_DRAW_COUNT,
            sample_size=DEFAULT_EET_ALT_COUNT,
            use_reseed=True,
            use_offset=True,
            use_sparse_maz=True,
        ),
    ]


def _candidate_engine_key(candidate_name: str) -> str:
    return candidate_name.lower().replace(" ", "_")


def available_engines() -> list[EngineSpec]:
    candidate_map = available_candidates()
    dense_candidates = [
        ("RandomState", "Dense gather | RandomState", True),
        ("GeneratorPCG64", "Dense gather | PCG64", True),
        ("GeneratorSFC64", "Dense gather | SFC64", True),
        ("GeneratorPhilox", "Dense gather | Philox", True),
        ("GeneratorMT19937", "Dense gather | MT19937", True),
        ("PhiloxAdvance", "Dense gather | Philox advance", True),
        ("VectorizedChooserHash", "Dense gather | Vectorized hash", False),
    ]
    engines = [
        EngineSpec(
            key=f"dense_{_candidate_engine_key(candidate_name)}",
            label=label,
            engine_family="drop_in_dense",
            drop_in_compatible=drop_in_compatible,
            algorithm="dense",
            candidate=candidate_map[candidate_name],
        )
        for candidate_name, label, drop_in_compatible in dense_candidates
    ]
    engines.append(
        EngineSpec(
            key="keyed_hash_direct",
            label="Direct sampled hash",
            engine_family="structural_eet",
            drop_in_compatible=False,
            algorithm="keyed_hash",
            applicable_scenarios=(
                "reseed_offset_sparse_maz",
                "reseed_offset_sparse_maz_eet",
            ),
        )
    )
    return engines


def _build_row_seeds(chooser_count: int) -> np.ndarray:
    return activitysim_row_seeds(
        rows=chooser_count,
        base_seed=12345,
        channel_name="persons",
        step_name="rng_runtime_breakdown",
    )


def _dense_lookup(draws: np.ndarray, positions: np.ndarray) -> np.ndarray:
    row_index = np.arange(draws.shape[0])[:, np.newaxis]
    return np.asarray(draws[row_index, positions])


def _sample_alt_ids(max_alt_id: int, alt_count: int) -> np.ndarray:
    start = max(1, max_alt_id - alt_count + 1)
    return np.arange(start, max_alt_id + 1, dtype=np.int64)


def _sampled_alt_matrix(scenario: ScenarioSpec, maz_alt_id: int) -> np.ndarray:
    if scenario.sample_size == 1:
        return np.full((scenario.chooser_count, 1), maz_alt_id, dtype=np.int64)
    alt_ids = _sample_alt_ids(maz_alt_id, scenario.sample_size)
    return np.broadcast_to(alt_ids, (scenario.chooser_count, alt_ids.shape[0]))


def _splitmix64(values: np.ndarray) -> np.ndarray:
    values = (values + _HASH_GOLDEN_GAMMA) & _MASK_64
    values = ((values ^ (values >> np.uint64(30))) * _HASH_MUL1) & _MASK_64
    values = ((values ^ (values >> np.uint64(27))) * _HASH_MUL2) & _MASK_64
    return values ^ (values >> np.uint64(31))


def _keyed_hash_uniforms(
    row_seeds: np.ndarray, sampled_alt_ids: np.ndarray, offset: int
) -> np.ndarray:
    row_state = row_seeds.astype(np.uint64)[:, np.newaxis]
    alt_state = sampled_alt_ids.astype(np.uint64)
    mixed = (
        row_state * _HASH_MUL1
        + alt_state * _HASH_GOLDEN_GAMMA
        + _HASH_ALT_TAG
        + np.uint64((offset * int(_HASH_MUL2)) & ((1 << 64) - 1))
    ) & _MASK_64
    hashed = _splitmix64(mixed)
    return ((hashed >> np.uint64(11)).astype(np.float64)) * (1.0 / (1 << 53))


def _generated_values(
    engine: EngineSpec, scenario: ScenarioSpec, maz_alt_id: int
) -> int:
    if engine.algorithm == "dense" and scenario.use_sparse_maz:
        return scenario.chooser_count * (maz_alt_id + 1)
    return scenario.useful_values


def _supports_scenario(engine: EngineSpec, scenario: ScenarioSpec) -> bool:
    if engine.applicable_scenarios is None:
        return True
    return scenario.key in engine.applicable_scenarios


def make_workload(
    engine: EngineSpec,
    scenario: ScenarioSpec,
    offset: int,
    maz_alt_id: int,
) -> Callable[[], object]:
    effective_offset = offset if scenario.use_offset else 0
    single_seed = np.asarray([12345], dtype=np.uint32)

    if engine.algorithm == "dense":
        assert engine.candidate is not None
        if scenario.use_sparse_maz:
            row_seeds = _build_row_seeds(scenario.chooser_count)
            sampled_alt_ids = _sampled_alt_matrix(scenario, maz_alt_id)

            def dense_sparse() -> np.ndarray:
                draws = engine.candidate.draw_uniform(
                    row_seeds,
                    n=maz_alt_id + 1,
                    offset=effective_offset,
                )
                return _dense_lookup(draws, sampled_alt_ids)

            return dense_sparse

        if scenario.use_reseed:
            row_seeds = _build_row_seeds(scenario.chooser_count)
            return lambda: engine.candidate.draw_uniform(
                row_seeds,
                n=scenario.sample_size,
                offset=effective_offset,
            )

        return lambda: engine.candidate.draw_uniform(
            single_seed,
            n=scenario.useful_values,
            offset=effective_offset,
        )

    if engine.algorithm == "keyed_hash":
        row_seeds = _build_row_seeds(scenario.chooser_count)
        sampled_alt_ids = _sampled_alt_matrix(scenario, maz_alt_id)
        return lambda: _keyed_hash_uniforms(
            row_seeds, sampled_alt_ids, effective_offset
        )

    raise ValueError(f"Unknown engine algorithm '{engine.algorithm}'")


def benchmark_engine(
    scenario: ScenarioSpec,
    engine: EngineSpec,
    offset: int,
    maz_alt_id: int,
    repeat: int,
) -> RuntimeResult:
    workload = make_workload(engine, scenario, offset=offset, maz_alt_id=maz_alt_id)
    times, _ = timed_repeats(workload, repeat)
    mean_seconds, std_seconds = summarize_times(times)
    useful_values = scenario.useful_values
    generated_values = _generated_values(engine, scenario, maz_alt_id)
    return RuntimeResult(
        scenario=scenario.name,
        scenario_label=scenario.label,
        engine=engine.label,
        engine_family=engine.engine_family,
        drop_in_compatible=engine.drop_in_compatible,
        chooser_count=scenario.chooser_count,
        sample_size=scenario.sample_size,
        useful_values=useful_values,
        generated_values=generated_values,
        waste_factor=(generated_values / useful_values),
        repeat=repeat,
        mean_seconds=mean_seconds,
        std_seconds=std_seconds,
        useful_values_per_sec=(
            0.0 if mean_seconds <= 0 else useful_values / mean_seconds
        ),
        generated_values_per_sec=(
            0.0 if mean_seconds <= 0 else generated_values / mean_seconds
        ),
    )


def write_results_csv(results: list[RuntimeResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "rng_runtime_breakdown_results.csv"
    with csv_path.open("w", newline="", encoding="utf8") as file_obj:
        writer = csv.DictWriter(
            file_obj, fieldnames=list(RuntimeResult.__dataclass_fields__.keys())
        )
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))
    return csv_path


def _sanitize_filename(value: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in value)


def write_bar_charts(
    results: list[RuntimeResult],
    scenarios: list[ScenarioSpec],
    output_dir: Path,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for old_plot in plot_dir.glob("*.png"):
        old_plot.unlink()

    saved_paths: list[Path] = []
    for engine_family in ["drop_in_dense", "structural_eet"]:
        family_dir = plot_dir / engine_family
        family_dir.mkdir(parents=True, exist_ok=True)
        for old_plot in family_dir.glob("*.png"):
            old_plot.unlink()

        for scenario_index, scenario in enumerate(scenarios, start=1):
            rows = [
                row
                for row in results
                if row.scenario == scenario.name and row.engine_family == engine_family
            ]
            if not rows:
                continue

            fig, ax = plt.subplots(figsize=(11, 6))
            x = np.arange(len(rows), dtype=np.float64)
            means = [row.mean_seconds for row in rows]
            stds = [row.std_seconds for row in rows]
            labels = [row.engine for row in rows]
            ax.bar(x, means, yerr=stds, capsize=4, color="#4C6EF5")
            ax.set_xticks(x, labels, rotation=30, ha="right")
            ax.set_ylabel("mean runtime (seconds)")
            ax.set_title(f"{_engine_family_title(engine_family)} | {scenario.label}")
            ax.grid(axis="y", alpha=0.3)

            reference_row = rows[0]
            detail_text = (
                f"choosers={_format_int(reference_row.chooser_count)}; "
                f"sample_size={_format_int(reference_row.sample_size)}; "
                f"useful={_format_int(reference_row.useful_values)}; "
                f"generated={_format_int(reference_row.generated_values)}; "
                f"waste={reference_row.waste_factor:.1f}"
            )
            ax.text(
                0.01,
                0.98,
                scenario.description + "\n" + detail_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
            )
            fig.tight_layout()

            save_path = (
                family_dir
                / f"{scenario_index:02d}_{_sanitize_filename(scenario.name)}.png"
            )
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            saved_paths.append(save_path)

    return saved_paths


def print_results(results: list[RuntimeResult]) -> None:
    header = (
        f"{'Scenario':38} {'Family':16} {'Engine':28} {'Choosers':>9} {'Sample':>8} "
        f"{'Useful':>12} {'Generated':>12} {'Waste':>8} {'Mean(s)':>10} {'Useful/s':>12}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row.scenario_label:38} {_engine_family_title(row.engine_family):16} {row.engine[:28]:28} "
            f"{row.chooser_count:9d} {row.sample_size:8d} {row.useful_values:12d} {row.generated_values:12d} "
            f"{row.waste_factor:8.2f} {row.mean_seconds:10.6f} {row.useful_values_per_sec:12.1f}"
        )


def resolve_repeat(profile: str, override: int | None) -> int:
    repeat = FAST_REPEAT if profile == "fast" else FULL_REPEAT
    if override is not None:
        repeat = int(override)
    if repeat <= 0:
        raise ValueError("repeat must be positive")
    return repeat


def validate_args(args: argparse.Namespace) -> None:
    if DEFAULT_DRAW_COUNT <= 0:
        raise ValueError("DEFAULT_DRAW_COUNT must be positive")
    if DEFAULT_OFFSET < 0:
        raise ValueError("DEFAULT_OFFSET must be nonnegative")
    if DEFAULT_MAZ_ALT_ID < 1:
        raise ValueError("DEFAULT_MAZ_ALT_ID must be at least 1")
    if DEFAULT_EET_ALT_COUNT <= 0:
        raise ValueError("DEFAULT_EET_ALT_COUNT must be positive")
    if DEFAULT_EET_ALT_COUNT > DEFAULT_MAZ_ALT_ID:
        raise ValueError(
            "DEFAULT_EET_ALT_COUNT must be less than or equal to DEFAULT_MAZ_ALT_ID"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    repeat = resolve_repeat(args.profile, args.repeat)
    scenarios = scenario_specs()
    engines = available_engines()

    results: list[RuntimeResult] = []
    for scenario in scenarios:
        for engine in engines:
            if not _supports_scenario(engine, scenario):
                continue
            print(
                f"Benchmarking {engine.label} for {scenario.name} "
                f"(choosers={scenario.chooser_count}, sample_size={scenario.sample_size}, offset={DEFAULT_OFFSET}, maz_alt_id={DEFAULT_MAZ_ALT_ID})",
                flush=True,
            )
            results.append(
                benchmark_engine(
                    scenario,
                    engine,
                    offset=DEFAULT_OFFSET,
                    maz_alt_id=DEFAULT_MAZ_ALT_ID,
                    repeat=repeat,
                )
            )

    print_results(results)
    csv_path = write_results_csv(results, args.output_dir)
    plot_paths = write_bar_charts(results, scenarios, args.output_dir)

    print(f"\nWrote runtime CSV: {csv_path}")
    if plot_paths:
        print(f"Wrote {len(plot_paths)} bar charts under: {args.output_dir / 'plots'}")
    else:
        print("No bar charts were written because matplotlib is unavailable.")


if __name__ == "__main__":
    main()
