from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

_MAX_SEED = 1 << 32
_SEED_MASK = 0xFFFFFFFF
_MASK_64 = np.uint64((1 << 64) - 1)


@dataclass(frozen=True)
class ScenarioSpec:
    suite: str
    name: str
    chooser_count: int
    sample_size: int
    actual_alt_count: int
    max_alt_id: int
    overlap_ratio: float
    offset: int = 0

    @property
    def sparsity_ratio(self) -> float:
        return self.max_alt_id / self.actual_alt_count


@dataclass
class ScenarioInputs:
    actual_alt_ids: np.ndarray
    row_seeds: np.ndarray
    chooser_ids: np.ndarray
    sampled_alt_ids_a: np.ndarray
    sampled_alt_ids_b: np.ndarray
    dense_positions_a: np.ndarray
    dense_positions_b: np.ndarray
    utilities_a: np.ndarray
    utilities_b: np.ndarray


@dataclass(frozen=True)
class InvarianceResult:
    engine: str
    strategy_label: str
    offset: int
    batch_invariant: bool
    sampled_set_invariant: bool
    offset_changes_values: bool
    shared_pairs_checked: int
    message: str


@dataclass(frozen=True)
class KernelResult:
    suite: str
    scenario: str
    kernel: str
    engine: str
    strategy_label: str
    chooser_count: int
    sample_size: int
    actual_alt_count: int
    max_alt_id: int
    sparsity_ratio: float
    overlap_ratio: float
    offset: int
    generated_shocks_per_chooser: int
    useful_shocks_per_chooser: int
    waste_factor: float
    repeat: int
    mean_seconds: float
    std_seconds: float
    peak_memory_mb: float
    useful_shocks_per_sec: float
    generated_shocks_per_sec: float


def hash32(text: str) -> int:
    digest = hashlib.md5(text.encode("utf8")).hexdigest()
    return int(digest, base=16) & _SEED_MASK


def activitysim_row_seeds(
    rows: int,
    base_seed: int,
    channel_name: str,
    step_name: str,
    index_start: int = 1,
) -> np.ndarray:
    channel_seed = hash32(channel_name)
    step_seed = hash32(step_name)
    idx = np.arange(index_start, index_start + rows, dtype=np.uint64)
    seeds = (base_seed + channel_seed + step_seed + idx) % _MAX_SEED
    return seeds.astype(np.uint32)


def _uniforms_to_gumbel(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-12, 1 - 1e-12)
    return -np.log(-np.log(clipped))


def _timed_repeats(func: Callable[[], object], repeat: int) -> tuple[list[float], int]:
    durations: list[float] = []
    for _ in range(repeat):
        gc.collect()
        t0 = time.perf_counter_ns()
        _ = func()
        t1 = time.perf_counter_ns()
        durations.append((t1 - t0) / 1e9)

    gc.collect()
    tracemalloc.start()
    _ = func()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return durations, peak


class ShockEngine:
    name = "base"
    strategy_label = "baseline"

    def shocks(
        self,
        row_seeds: np.ndarray,
        sampled_alt_ids: np.ndarray,
        dense_positions: np.ndarray,
        actual_alt_ids: np.ndarray,
        max_alt_id: int,
        offset: int = 0,
    ) -> np.ndarray:
        raise NotImplementedError

    def generated_shocks_per_chooser(
        self,
        sampled_alt_ids: np.ndarray,
        actual_alt_ids: np.ndarray,
        max_alt_id: int,
    ) -> int:
        raise NotImplementedError


class CurrentDenseEngine(ShockEngine):
    name = "current_dense"
    strategy_label = "Dense max-id gather"

    def shocks(
        self,
        row_seeds: np.ndarray,
        sampled_alt_ids: np.ndarray,
        dense_positions: np.ndarray,
        actual_alt_ids: np.ndarray,
        max_alt_id: int,
        offset: int = 0,
    ) -> np.ndarray:
        dense_count = max_alt_id + 1
        prng = np.random.RandomState()
        out = np.empty(sampled_alt_ids.shape, dtype=np.float64)
        for row_idx, seed in enumerate(row_seeds):
            prng.seed(int(seed))
            if offset:
                prng.rand(offset)
            dense_draws = _uniforms_to_gumbel(prng.rand(dense_count))
            out[row_idx] = dense_draws[sampled_alt_ids[row_idx]]
        return out

    def generated_shocks_per_chooser(
        self,
        sampled_alt_ids: np.ndarray,
        actual_alt_ids: np.ndarray,
        max_alt_id: int,
    ) -> int:
        return max_alt_id + 1


class CompressedDenseEngine(ShockEngine):
    name = "compressed_dense"
    strategy_label = "Compressed alt remap"

    def shocks(
        self,
        row_seeds: np.ndarray,
        sampled_alt_ids: np.ndarray,
        dense_positions: np.ndarray,
        actual_alt_ids: np.ndarray,
        max_alt_id: int,
        offset: int = 0,
    ) -> np.ndarray:
        dense_count = len(actual_alt_ids)
        prng = np.random.RandomState()
        out = np.empty(sampled_alt_ids.shape, dtype=np.float64)
        for row_idx, seed in enumerate(row_seeds):
            prng.seed(int(seed))
            if offset:
                prng.rand(offset)
            dense_draws = _uniforms_to_gumbel(prng.rand(dense_count))
            out[row_idx] = dense_draws[dense_positions[row_idx]]
        return out

    def generated_shocks_per_chooser(
        self,
        sampled_alt_ids: np.ndarray,
        actual_alt_ids: np.ndarray,
        max_alt_id: int,
    ) -> int:
        return len(actual_alt_ids)


class KeyedHashEngine(ShockEngine):
    name = "keyed_hash"
    strategy_label = "Keyed chooser-alt hash"

    _GOLDEN_GAMMA = np.uint64(0x9E3779B97F4A7C15)
    _MUL1 = np.uint64(0xBF58476D1CE4E5B9)
    _MUL2 = np.uint64(0x94D049BB133111EB)
    _STAGE_TAG = np.uint64(0x51D7348C2F9A3B17)

    @classmethod
    def _splitmix64(cls, values: np.ndarray) -> np.ndarray:
        values = (values + cls._GOLDEN_GAMMA) & _MASK_64
        values = ((values ^ (values >> np.uint64(30))) * cls._MUL1) & _MASK_64
        values = ((values ^ (values >> np.uint64(27))) * cls._MUL2) & _MASK_64
        return values ^ (values >> np.uint64(31))

    def shocks(
        self,
        row_seeds: np.ndarray,
        sampled_alt_ids: np.ndarray,
        dense_positions: np.ndarray,
        actual_alt_ids: np.ndarray,
        max_alt_id: int,
        offset: int = 0,
    ) -> np.ndarray:
        row_state = row_seeds.astype(np.uint64)[:, np.newaxis]
        alt_state = sampled_alt_ids.astype(np.uint64)
        mixed = (
            row_state * self._MUL1
            + alt_state * self._GOLDEN_GAMMA
            + self._STAGE_TAG
            + np.uint64((offset * int(self._MUL2)) & ((1 << 64) - 1))
        ) & _MASK_64
        hashed = self._splitmix64(mixed)
        uniforms = ((hashed >> np.uint64(11)).astype(np.float64)) * (1.0 / (1 << 53))
        return _uniforms_to_gumbel(uniforms)

    def generated_shocks_per_chooser(
        self,
        sampled_alt_ids: np.ndarray,
        actual_alt_ids: np.ndarray,
        max_alt_id: int,
    ) -> int:
        return int(sampled_alt_ids.shape[1])


def _pair_seed(row_seed: int, alt_id: int, stream_tag: int = 0) -> int:
    mask = (1 << 64) - 1
    mixed = (
        int(np.uint32(row_seed)) * 0x9E3779B97F4A7C15
        + int(alt_id)
        + 0xBF58476D1CE4E5B9
        + int(stream_tag)
    ) & mask
    mixed ^= mixed >> 30
    mixed = (mixed * 0xBF58476D1CE4E5B9) & mask
    mixed ^= mixed >> 27
    mixed = (mixed * 0x94D049BB133111EB) & mask
    mixed ^= mixed >> 31
    return mixed


class KeyedBitGeneratorEngine(ShockEngine):
    bitgen_cls: type[np.random.BitGenerator]
    use_advance: bool

    def __init__(
        self,
        bitgen_cls: type[np.random.BitGenerator],
        name: str,
        strategy_label: str,
        use_advance: bool,
    ) -> None:
        self.bitgen_cls = bitgen_cls
        self.name = name
        self.strategy_label = strategy_label
        self.use_advance = use_advance

    def _draw_one(self, pair_seed: int, offset: int) -> float:
        bitgen = self.bitgen_cls(pair_seed)
        rng = np.random.Generator(bitgen)
        if offset:
            if self.use_advance and hasattr(bitgen, "advance"):
                bitgen.advance(offset)
            else:
                rng.random(offset)
        return float(rng.random())

    def shocks(
        self,
        row_seeds: np.ndarray,
        sampled_alt_ids: np.ndarray,
        dense_positions: np.ndarray,
        actual_alt_ids: np.ndarray,
        max_alt_id: int,
        offset: int = 0,
    ) -> np.ndarray:
        out = np.empty(sampled_alt_ids.shape, dtype=np.float64)
        for row_idx, row_seed in enumerate(row_seeds):
            for col_idx, alt_id in enumerate(sampled_alt_ids[row_idx]):
                pair_seed = _pair_seed(
                    int(row_seed), int(alt_id), stream_tag=0x51D7348C
                )
                out[row_idx, col_idx] = self._draw_one(pair_seed, offset)
        return _uniforms_to_gumbel(out)

    def generated_shocks_per_chooser(
        self,
        sampled_alt_ids: np.ndarray,
        actual_alt_ids: np.ndarray,
        max_alt_id: int,
    ) -> int:
        return int(sampled_alt_ids.shape[1])


def available_engines() -> list[ShockEngine]:
    return [
        CurrentDenseEngine(),
        CompressedDenseEngine(),
        KeyedHashEngine(),
        KeyedBitGeneratorEngine(
            np.random.Philox,
            "keyed_philox",
            "Keyed chooser-alt Philox",
            use_advance=True,
        ),
        KeyedBitGeneratorEngine(
            np.random.PCG64,
            "keyed_pcg64",
            "Keyed chooser-alt PCG64",
            use_advance=False,
        ),
    ]


def _make_actual_alt_ids(actual_alt_count: int, max_alt_id: int) -> np.ndarray:
    if max_alt_id < actual_alt_count:
        raise ValueError(
            f"max_alt_id={max_alt_id} must be >= actual_alt_count={actual_alt_count}"
        )
    if actual_alt_count == 1:
        return np.asarray([max_alt_id], dtype=np.int64)
    step = max(1, (max_alt_id - 1) // (actual_alt_count - 1))
    alt_ids = 1 + step * np.arange(actual_alt_count, dtype=np.int64)
    alt_ids[-1] = max_alt_id
    return alt_ids


def _sample_without_replacement(
    rng: np.random.Generator, actual_alt_ids: np.ndarray, sample_size: int
) -> np.ndarray:
    if sample_size > len(actual_alt_ids):
        raise ValueError(
            f"sample_size={sample_size} exceeds actual_alt_count={len(actual_alt_ids)}"
        )
    return np.asarray(
        rng.choice(actual_alt_ids, size=sample_size, replace=False), dtype=np.int64
    )


def build_scenario_inputs(spec: ScenarioSpec) -> ScenarioInputs:
    actual_alt_ids = _make_actual_alt_ids(spec.actual_alt_count, spec.max_alt_id)
    chooser_ids = np.arange(spec.chooser_count, dtype=np.int64)
    row_seeds = activitysim_row_seeds(
        rows=spec.chooser_count,
        base_seed=12345,
        channel_name="persons",
        step_name=f"eet_keyed_maz_test::{spec.name}",
    )
    rng = np.random.Generator(np.random.PCG64(hash32(spec.name) + 999))
    shared_count = int(round(spec.sample_size * spec.overlap_ratio))
    shared_count = max(0, min(shared_count, spec.sample_size))

    sampled_alt_ids_a = np.empty((spec.chooser_count, spec.sample_size), dtype=np.int64)
    sampled_alt_ids_b = np.empty((spec.chooser_count, spec.sample_size), dtype=np.int64)

    for chooser_idx in range(spec.chooser_count):
        sample_a = _sample_without_replacement(rng, actual_alt_ids, spec.sample_size)
        rng.shuffle(sample_a)
        sampled_alt_ids_a[chooser_idx] = sample_a

        sample_b = np.empty(spec.sample_size, dtype=np.int64)
        if shared_count:
            sample_b[:shared_count] = sample_a[:shared_count]
        if shared_count < spec.sample_size:
            remaining = np.setdiff1d(
                actual_alt_ids, sample_a[:shared_count], assume_unique=True
            )
            new_count = spec.sample_size - shared_count
            sample_b[shared_count:] = _sample_without_replacement(
                rng, remaining, new_count
            )
        rng.shuffle(sample_b)
        sampled_alt_ids_b[chooser_idx] = sample_b

    dense_positions_a = np.searchsorted(actual_alt_ids, sampled_alt_ids_a)
    dense_positions_b = np.searchsorted(actual_alt_ids, sampled_alt_ids_b)

    chooser_term = 0.025 * np.log1p(chooser_ids).reshape(-1, 1)
    utilities_a = (
        0.15 * np.sin((sampled_alt_ids_a % 97 + 1) / 9.0)
        + 0.05 * np.log1p(sampled_alt_ids_a)
        + chooser_term
    )
    utilities_b = (
        0.15 * np.sin((sampled_alt_ids_b % 97 + 1) / 9.0)
        + 0.05 * np.log1p(sampled_alt_ids_b)
        + chooser_term
    )

    return ScenarioInputs(
        actual_alt_ids=actual_alt_ids,
        row_seeds=row_seeds,
        chooser_ids=chooser_ids,
        sampled_alt_ids_a=sampled_alt_ids_a,
        sampled_alt_ids_b=sampled_alt_ids_b,
        dense_positions_a=dense_positions_a,
        dense_positions_b=dense_positions_b,
        utilities_a=utilities_a,
        utilities_b=utilities_b,
    )


def _shared_pair_match_count(
    alt_ids_a: np.ndarray,
    shocks_a: np.ndarray,
    alt_ids_b: np.ndarray,
    shocks_b: np.ndarray,
) -> tuple[int, int]:
    checked = 0
    matched = 0
    for row_idx in range(alt_ids_a.shape[0]):
        map_a = {
            int(alt): float(shock)
            for alt, shock in zip(alt_ids_a[row_idx], shocks_a[row_idx])
        }
        map_b = {
            int(alt): float(shock)
            for alt, shock in zip(alt_ids_b[row_idx], shocks_b[row_idx])
        }
        shared = set(map_a).intersection(map_b)
        checked += len(shared)
        matched += sum(map_a[alt] == map_b[alt] for alt in shared)
    return matched, checked


def run_invariance_check(
    engine: ShockEngine,
    inputs: ScenarioInputs,
    spec: ScenarioSpec,
    offset: int,
) -> InvarianceResult:
    shocks_a = engine.shocks(
        inputs.row_seeds,
        inputs.sampled_alt_ids_a,
        inputs.dense_positions_a,
        inputs.actual_alt_ids,
        spec.max_alt_id,
        offset=offset,
    )
    shocks_b = engine.shocks(
        inputs.row_seeds,
        inputs.sampled_alt_ids_b,
        inputs.dense_positions_b,
        inputs.actual_alt_ids,
        spec.max_alt_id,
        offset=offset,
    )
    matched, checked = _shared_pair_match_count(
        inputs.sampled_alt_ids_a,
        shocks_a,
        inputs.sampled_alt_ids_b,
        shocks_b,
    )

    reversed_seeds = inputs.row_seeds[::-1]
    reversed_alt_ids = inputs.sampled_alt_ids_a[::-1]
    reversed_dense = inputs.dense_positions_a[::-1]
    reversed_shocks = engine.shocks(
        reversed_seeds,
        reversed_alt_ids,
        reversed_dense,
        inputs.actual_alt_ids,
        spec.max_alt_id,
        offset=offset,
    )
    batch_invariant = np.array_equal(reversed_shocks[::-1], shocks_a)
    sampled_set_invariant = checked == matched
    baseline_shocks = engine.shocks(
        inputs.row_seeds,
        inputs.sampled_alt_ids_a,
        inputs.dense_positions_a,
        inputs.actual_alt_ids,
        spec.max_alt_id,
        offset=0,
    )
    offset_changes_values = (
        True if offset == 0 else not np.array_equal(baseline_shocks, shocks_a)
    )
    message = (
        f"offset={offset}; shared_pairs={checked}; batch_invariant={batch_invariant}; "
        f"sampled_set_invariant={sampled_set_invariant}; offset_changes_values={offset_changes_values}"
    )
    return InvarianceResult(
        engine=engine.name,
        strategy_label=engine.strategy_label,
        offset=offset,
        batch_invariant=batch_invariant,
        sampled_set_invariant=sampled_set_invariant,
        offset_changes_values=offset_changes_values,
        shared_pairs_checked=checked,
        message=message,
    )


def _summarize_times(times: list[float]) -> tuple[float, float]:
    arr = np.asarray(times, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def _make_kernel_result(
    spec: ScenarioSpec,
    engine: ShockEngine,
    kernel: str,
    times: list[float],
    peak_bytes: int,
) -> KernelResult:
    mean_seconds, std_seconds = _summarize_times(times)
    useful = spec.chooser_count * spec.sample_size
    generated = spec.chooser_count * engine.generated_shocks_per_chooser(
        np.empty((1, spec.sample_size), dtype=np.int64),
        np.empty(spec.actual_alt_count, dtype=np.int64),
        spec.max_alt_id,
    )
    return KernelResult(
        suite=spec.suite,
        scenario=spec.name,
        kernel=kernel,
        engine=engine.name,
        strategy_label=engine.strategy_label,
        chooser_count=spec.chooser_count,
        sample_size=spec.sample_size,
        actual_alt_count=spec.actual_alt_count,
        max_alt_id=spec.max_alt_id,
        sparsity_ratio=spec.sparsity_ratio,
        overlap_ratio=spec.overlap_ratio,
        offset=spec.offset,
        generated_shocks_per_chooser=generated // spec.chooser_count,
        useful_shocks_per_chooser=spec.sample_size,
        waste_factor=(generated / useful),
        repeat=len(times),
        mean_seconds=mean_seconds,
        std_seconds=std_seconds,
        peak_memory_mb=peak_bytes / (1024 * 1024),
        useful_shocks_per_sec=(0.0 if mean_seconds <= 0 else useful / mean_seconds),
        generated_shocks_per_sec=(
            0.0 if mean_seconds <= 0 else generated / mean_seconds
        ),
    )


def benchmark_engine(
    engine: ShockEngine,
    spec: ScenarioSpec,
    inputs: ScenarioInputs,
    repeat: int,
) -> list[KernelResult]:
    def shock_lookup() -> np.ndarray:
        return engine.shocks(
            inputs.row_seeds,
            inputs.sampled_alt_ids_a,
            inputs.dense_positions_a,
            inputs.actual_alt_ids,
            spec.max_alt_id,
            offset=spec.offset,
        )

    def final_choice() -> np.ndarray:
        shocks = shock_lookup()
        return np.argmax(inputs.utilities_a + shocks, axis=1)

    lookup_times, lookup_peak = _timed_repeats(shock_lookup, repeat)
    choice_times, choice_peak = _timed_repeats(final_choice, repeat)

    return [
        _make_kernel_result(spec, engine, "shock_lookup", lookup_times, lookup_peak),
        _make_kernel_result(spec, engine, "final_choice", choice_times, choice_peak),
    ]


def _markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_results_csv(results: list[KernelResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "eet_keyed_maz_test_results.csv"
    with csv_path.open("w", newline="", encoding="utf8") as file_obj:
        writer = csv.DictWriter(
            file_obj, fieldnames=list(KernelResult.__dataclass_fields__.keys())
        )
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    return csv_path


def write_invariance_csv(results: list[InvarianceResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "eet_keyed_maz_test_invariance.csv"
    with csv_path.open("w", newline="", encoding="utf8") as file_obj:
        writer = csv.DictWriter(
            file_obj, fieldnames=list(InvarianceResult.__dataclass_fields__.keys())
        )
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    return csv_path


def write_summary_markdown(
    results: list[KernelResult],
    invariance_results: list[InvarianceResult],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "eet_keyed_maz_test_summary.md"
    df = pd.DataFrame(asdict(result) for result in results)
    inv_df = pd.DataFrame(asdict(result) for result in invariance_results)

    grouped = (
        df.groupby(["suite", "kernel", "strategy_label", "offset"], as_index=False)[
            ["mean_seconds", "peak_memory_mb", "waste_factor", "useful_shocks_per_sec"]
        ]
        .mean()
        .round(4)
    )

    speedup_rows: list[dict[str, object]] = []
    for suite in sorted(df["suite"].unique()):
        for kernel in sorted(df["kernel"].unique()):
            subset = df[(df["suite"] == suite) & (df["kernel"] == kernel)].copy()
            if subset.empty:
                continue
            baseline = subset[subset["engine"] == "current_dense"][
                ["scenario", "mean_seconds"]
            ].rename(columns={"mean_seconds": "baseline_seconds"})
            merged = subset.merge(baseline, on="scenario", how="left")
            merged["speedup_vs_current"] = (
                merged["baseline_seconds"] / merged["mean_seconds"]
            )
            summary = (
                merged.groupby(["strategy_label", "offset"], as_index=False)[
                    "speedup_vs_current"
                ]
                .mean()
                .round(3)
            )
            for row in summary.itertuples(index=False):
                speedup_rows.append(
                    {
                        "suite": suite,
                        "kernel": kernel,
                        "strategy_label": row.strategy_label,
                        "offset": row.offset,
                        "avg_speedup_vs_current": row.speedup_vs_current,
                    }
                )
    speedup_df = pd.DataFrame(speedup_rows)

    lines = [
        "# EET keyed MAZ benchmark summary",
        "",
        "This output compares several shock-generation strategies on a simplified, off-model version of the explicit-error-term final MAZ choice problem:",
        "",
        "- Dense max-id gather",
        "- Compressed alternative remap",
        "- Keyed chooser-alt hash",
        "- Keyed chooser-alt Philox",
        "- Keyed chooser-alt PCG64",
        "",
        "The benchmark now includes sparsity, chooser-count, and offset sweeps.",
        "",
        "The kernels reported below isolate what matters for the decision:",
        "",
        "- `shock_lookup`: generate or recover the EV1 shock for each sampled MAZ",
        "- `final_choice`: add those shocks to deterministic utilities and take the argmax",
        "",
        "Offset sweeps show how each strategy behaves when the random stream position needs to advance before sampled MAZ shocks are consumed.",
        "",
        "## Average metrics by suite, kernel, and strategy",
        "",
        _markdown_table(grouped),
        "",
        "## Average speedup versus current dense baseline",
        "",
        _markdown_table(speedup_df),
        "",
        "## Invariance checks",
        "",
        _markdown_table(inv_df),
        "",
        "## Reading the metrics",
        "",
        "- `waste_factor` is generated shocks divided by useful shocks. Lower is better.",
        "- `useful_shocks_per_sec` is the most comparable throughput metric across strategies.",
        "- The `offset` column is the number of prior draws consumed before the tested shock lookup.",
        "- A passing invariance result means shared chooser-MAZ pairs keep the same shock even when the sampled set changes, batch order does not matter, and nonzero offsets actually change the draw state.",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf8")
    return summary_path


def _safe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def _plot_metric(
    df: pd.DataFrame,
    suite: str,
    kernel: str,
    x_field: str,
    y_field: str,
    output_dir: Path,
) -> Path | None:
    plt = _safe_import_matplotlib()
    if plt is None:
        return None
    subset = df[(df["suite"] == suite) & (df["kernel"] == kernel)].copy()
    if subset.empty:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for strategy_label in subset["strategy_label"].drop_duplicates().tolist():
        option_rows = subset[subset["strategy_label"] == strategy_label].sort_values(
            x_field
        )
        ax.plot(
            option_rows[x_field],
            option_rows[y_field],
            marker="o",
            linewidth=2.0,
            markersize=5,
            label=strategy_label,
        )

    title_metric = y_field.replace("_", " ")
    title_x = x_field.replace("_", " ")
    ax.set_title(
        f"{suite.replace('_', ' ').title()} | {kernel.replace('_', ' ')} | {title_metric}"
    )
    ax.set_xlabel(title_x)
    ax.set_ylabel(title_metric)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    save_path = output_dir / f"{suite}__{kernel}__{y_field}__vs_{x_field}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def _plot_speedup(
    df: pd.DataFrame, suite: str, kernel: str, x_field: str, output_dir: Path
) -> Path | None:
    plt = _safe_import_matplotlib()
    if plt is None:
        return None
    subset = df[(df["suite"] == suite) & (df["kernel"] == kernel)].copy()
    baseline = subset[subset["engine"] == "current_dense"][
        [x_field, "mean_seconds"]
    ].rename(columns={"mean_seconds": "baseline_seconds"})
    merged = subset.merge(baseline, on=x_field, how="left")
    merged["speedup_vs_current"] = merged["baseline_seconds"] / merged["mean_seconds"]
    merged = merged[merged["engine"] != "current_dense"]
    if merged.empty:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for strategy_label in merged["strategy_label"].drop_duplicates().tolist():
        option_rows = merged[merged["strategy_label"] == strategy_label].sort_values(
            x_field
        )
        ax.plot(
            option_rows[x_field],
            option_rows["speedup_vs_current"],
            marker="o",
            linewidth=2.0,
            markersize=5,
            label=strategy_label,
        )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title(
        f"{suite.replace('_', ' ').title()} | {kernel.replace('_', ' ')} | speedup vs current dense"
    )
    ax.set_xlabel(x_field.replace("_", " "))
    ax.set_ylabel("speedup vs current dense")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    save_path = output_dir / f"{suite}__{kernel}__speedup_vs_current__vs_{x_field}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def write_plots(results: list[KernelResult], output_dir: Path) -> list[Path]:
    df = pd.DataFrame(asdict(result) for result in results)
    plot_dir = output_dir / "plots"
    plot_specs = [
        ("sparsity_sweep", "shock_lookup", "sparsity_ratio", "mean_seconds"),
        ("sparsity_sweep", "final_choice", "sparsity_ratio", "mean_seconds"),
        ("sparsity_sweep", "shock_lookup", "sparsity_ratio", "peak_memory_mb"),
        ("sparsity_sweep", "final_choice", "sparsity_ratio", "useful_shocks_per_sec"),
        ("sparsity_sweep", "shock_lookup", "sparsity_ratio", "waste_factor"),
        ("chooser_sweep", "final_choice", "chooser_count", "mean_seconds"),
        ("chooser_sweep", "final_choice", "chooser_count", "peak_memory_mb"),
        ("offset_sweep", "shock_lookup", "offset", "mean_seconds"),
        ("offset_sweep", "final_choice", "offset", "mean_seconds"),
        ("offset_sweep", "shock_lookup", "offset", "useful_shocks_per_sec"),
    ]
    saved_paths: list[Path] = []
    for suite, kernel, x_field, y_field in plot_specs:
        save_path = _plot_metric(df, suite, kernel, x_field, y_field, plot_dir)
        if save_path is not None:
            saved_paths.append(save_path)
    for suite, kernel, x_field in [
        ("sparsity_sweep", "shock_lookup", "sparsity_ratio"),
        ("sparsity_sweep", "final_choice", "sparsity_ratio"),
        ("chooser_sweep", "final_choice", "chooser_count"),
        ("offset_sweep", "shock_lookup", "offset"),
        ("offset_sweep", "final_choice", "offset"),
    ]:
        save_path = _plot_speedup(df, suite, kernel, x_field, plot_dir)
        if save_path is not None:
            saved_paths.append(save_path)
    return saved_paths


def make_profile(profile: str) -> tuple[list[ScenarioSpec], int]:
    profile = profile.lower().strip()
    if profile == "fast":
        repeat = 1
        sparsity_ratios = [1, 4, 16, 64]
        chooser_counts = [100, 400, 800]
        offsets = [0, 1, 8, 32]
        sample_size = 12
        actual_alt_count = 96
        overlap_ratio = 0.5
    elif profile == "full":
        repeat = 3
        sparsity_ratios = [1, 4, 16, 64, 128]
        chooser_counts = [250, 1000, 2000]
        offsets = [0, 1, 8, 32, 128]
        sample_size = 16
        actual_alt_count = 128
        overlap_ratio = 0.5
    else:
        raise ValueError("profile must be 'fast' or 'full'")

    scenarios: list[ScenarioSpec] = []
    fixed_choosers = chooser_counts[1]
    fixed_ratio = sparsity_ratios[-1]
    for ratio in sparsity_ratios:
        scenarios.append(
            ScenarioSpec(
                suite="sparsity_sweep",
                name=f"sparsity_r{ratio}",
                chooser_count=fixed_choosers,
                sample_size=sample_size,
                actual_alt_count=actual_alt_count,
                max_alt_id=actual_alt_count * ratio,
                overlap_ratio=overlap_ratio,
            )
        )
    for chooser_count in chooser_counts:
        scenarios.append(
            ScenarioSpec(
                suite="chooser_sweep",
                name=f"choosers_{chooser_count}",
                chooser_count=chooser_count,
                sample_size=sample_size,
                actual_alt_count=actual_alt_count,
                max_alt_id=actual_alt_count * fixed_ratio,
                overlap_ratio=overlap_ratio,
            )
        )
    for offset in offsets:
        scenarios.append(
            ScenarioSpec(
                suite="offset_sweep",
                name=f"offset_{offset}",
                chooser_count=fixed_choosers,
                sample_size=sample_size,
                actual_alt_count=actual_alt_count,
                max_alt_id=actual_alt_count * fixed_ratio,
                overlap_ratio=overlap_ratio,
                offset=offset,
            )
        )
    return scenarios, repeat


def print_results(
    results: list[KernelResult], invariance_results: list[InvarianceResult]
) -> None:
    header = (
        f"{'Scenario':16} {'Kernel':13} {'Strategy':28} {'Offset':>8} {'Choosers':>9} {'Sparse':>8} "
        f"{'Mean(s)':>10} {'PeakMB':>9} {'Waste':>8} {'Useful/s':>12}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row.scenario:16} {row.kernel:13} {row.strategy_label:28} {row.offset:8d} {row.chooser_count:9d} "
            f"{row.sparsity_ratio:8.1f} {row.mean_seconds:10.6f} {row.peak_memory_mb:9.2f} "
            f"{row.waste_factor:8.2f} {row.useful_shocks_per_sec:12.1f}"
        )
    print("\nInvariance checks")
    print("-" * 80)
    for result in invariance_results:
        status = (
            "PASS"
            if result.batch_invariant
            and result.sampled_set_invariant
            and result.offset_changes_values
            else "FAIL"
        )
        print(f"{result.strategy_label:28} {status:4} {result.message}")


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Small off-model benchmark for EET MAZ shock generation strategies."
    )
    parser.add_argument(
        "--profile",
        choices=["fast", "full"],
        default="fast",
        help="Benchmark profile size.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="Override repeat count for each timed kernel.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "output" / "eet_keyed_maz_test",
        help="Directory for CSV, markdown summary, and plots.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation even if matplotlib is installed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios, repeat = make_profile(args.profile)
    if args.repeat is not None:
        repeat = int(args.repeat)
    engines = available_engines()
    output_dir = Path(args.output_dir)

    invariance_spec = ScenarioSpec(
        suite="invariance",
        name="invariance_reference",
        chooser_count=32,
        sample_size=12,
        actual_alt_count=96,
        max_alt_id=96 * 64,
        overlap_ratio=0.5,
    )
    invariance_inputs = build_scenario_inputs(invariance_spec)
    invariance_results: list[InvarianceResult] = []
    for offset in [0, 16]:
        invariance_results.extend(
            run_invariance_check(
                engine, invariance_inputs, invariance_spec, offset=offset
            )
            for engine in engines
        )

    results: list[KernelResult] = []
    for spec in scenarios:
        inputs = build_scenario_inputs(spec)
        for engine in engines:
            results.extend(benchmark_engine(engine, spec, inputs, repeat))

    print_results(results, invariance_results)

    results_csv = write_results_csv(results, output_dir)
    invariance_csv = write_invariance_csv(invariance_results, output_dir)
    summary_md = write_summary_markdown(results, invariance_results, output_dir)
    plot_paths: list[Path] = []
    if not args.skip_plots:
        plot_paths = write_plots(results, output_dir)

    print(f"\nWrote metrics CSV: {results_csv}")
    print(f"Wrote invariance CSV: {invariance_csv}")
    print(f"Wrote summary markdown: {summary_md}")
    if plot_paths:
        print(f"Wrote {len(plot_paths)} plot files under: {output_dir / 'plots'}")
    elif args.skip_plots:
        print("Skipped plot generation by request.")
    else:
        print("No plots were created because matplotlib is unavailable.")


if __name__ == "__main__":
    main()
