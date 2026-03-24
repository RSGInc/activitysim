from __future__ import annotations

import csv
import gc
import hashlib
import math
import time
import tracemalloc
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

_MAX_SEED = 1 << 32
_SEED_MASK = 0xFFFFFFFF


RUN_SETTINGS = {
    # Switch between benchmark profiles: "fast" or "full".
    "run_profile": "fast",
    # Run mode: "benchmark", "seed_unit_test", or "both".
    "run_mode": "both",
    # Global controls
    "repeat": 5,
    "candidates": "all",  # or list like ["RandomState", "GeneratorPCG64"]
    "csv": "output/benchmark_rng_results.csv",
    "run_repro_check": True,
    "run_quality_check": True,
    "quality_sample": 100000,
    # Plot controls
    "plots": {
        "enabled": True,
        "output_dir": "output",
        "metric": "mean_seconds",
    },
}


RUN_PROFILES = {
    "fast": {
        "reseed_sweep": {
            "reseeds": [1000, 10000, 50000],
            "draws_per_reseed": 10,
        },
        "draw_sweep": {
            "reseeds": 10000,
            "draws_per_reseed": [1, 10, 100, 1000, 5000],
        },
        "repeat": 2,
        "run_repro_check": False,
        "run_quality_check": False,
        "quality_sample": 10000,
    },
    "full": {
        "reseed_sweep": {
            "reseeds": [1000, 10000, 100000, 500000],
            "draws_per_reseed": 30,
        },
        "draw_sweep": {
            "reseeds": 10000,
            "draws_per_reseed": [1, 10, 100, 1000, 10000, 30000, 70000],
        },
        "repeat": 2,
        "run_repro_check": True,
        "run_quality_check": True,
        "quality_sample": 10000,
    },
}


@dataclass
class BenchmarkResult:
    scenario: str
    candidate: str
    rows: int
    draws_per_reseed: int
    draws: int
    repeat: int
    mean_seconds: float
    std_seconds: float
    throughput_draws_per_sec: float
    peak_memory_bytes: int
    notes: str = ""


@dataclass
class SeedUnitTestResult:
    candidate: str
    passed: bool
    message: str


def hash32(text: str) -> int:
    data = text.encode("utf8")
    digest = hashlib.md5(data).hexdigest()
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


def timed_repeats(func: Callable[[], object], repeat: int) -> tuple[list[float], int]:
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

    return durations, int(peak)


class RNGCandidate:
    name = "base"

    def draw_uniform(self, seeds: np.ndarray, n: int, offset: int = 0) -> np.ndarray:
        raise NotImplementedError

    def draw_normal(
        self,
        seeds: np.ndarray,
        mu: float,
        sigma: float,
        size: int | None = None,
        offset: int = 0,
        lognormal: bool = False,
    ) -> np.ndarray:
        raise NotImplementedError

    def draw_choice(
        self,
        seeds: np.ndarray,
        a: np.ndarray,
        size: int,
        replace: bool,
        offset: int = 0,
    ) -> np.ndarray:
        raise NotImplementedError

    def supports_choice(self, replace: bool) -> bool:
        return True

    def instance_factory(self, seed: int):
        raise NotImplementedError


class LegacyRandomStateCandidate(RNGCandidate):
    name = "RandomState"

    def _generators(self, seeds: np.ndarray, offset: int = 0):
        prng = np.random.RandomState()
        for seed in seeds:
            prng.seed(int(seed))
            if offset:
                prng.rand(offset)
            yield prng

    def draw_uniform(self, seeds: np.ndarray, n: int, offset: int = 0) -> np.ndarray:
        prng = np.random.RandomState()
        out = np.empty((len(seeds), n), dtype=np.float64)
        for i, seed in enumerate(seeds):
            prng.seed(int(seed))
            if offset:
                prng.rand(offset)
            out[i] = prng.rand(n)
        return out

    def draw_normal(
        self,
        seeds: np.ndarray,
        mu: float,
        sigma: float,
        size: int | None = None,
        offset: int = 0,
        lognormal: bool = False,
    ) -> np.ndarray:
        generators = self._generators(seeds, offset=offset)
        if lognormal:
            rands = np.asanyarray(
                [prng.lognormal(mean=mu, sigma=sigma, size=size) for prng in generators]
            )
        else:
            rands = np.asanyarray(
                [prng.normal(loc=mu, scale=sigma, size=size) for prng in generators]
            )
        return rands

    def draw_choice(
        self,
        seeds: np.ndarray,
        a: np.ndarray,
        size: int,
        replace: bool,
        offset: int = 0,
    ) -> np.ndarray:
        generators = self._generators(seeds, offset=offset)
        return np.concatenate(
            tuple(prng.choice(a, size=size, replace=replace) for prng in generators)
        )

    def instance_factory(self, seed: int):
        return np.random.RandomState(seed)


class GeneratorBitGenCandidate(RNGCandidate):
    name = "Generator"
    bitgen_cls: type[np.random.BitGenerator]

    def __init__(self, bitgen_cls: type[np.random.BitGenerator], name: str):
        self.bitgen_cls = bitgen_cls
        self.name = name

    def _make_rng(self, seed: int) -> np.random.Generator:
        return np.random.Generator(self.bitgen_cls(seed))

    def _generators(self, seeds: np.ndarray, offset: int = 0):
        for seed in seeds:
            rng = self._make_rng(int(seed))
            if offset:
                rng.random(offset)
            yield rng

    def draw_uniform(self, seeds: np.ndarray, n: int, offset: int = 0) -> np.ndarray:
        out = np.empty((len(seeds), n), dtype=np.float64)
        for i, seed in enumerate(seeds):
            rng = self._make_rng(int(seed))
            if offset:
                rng.random(offset)
            out[i] = rng.random(n)
        return out

    def draw_normal(
        self,
        seeds: np.ndarray,
        mu: float,
        sigma: float,
        size: int | None = None,
        offset: int = 0,
        lognormal: bool = False,
    ) -> np.ndarray:
        generators = self._generators(seeds, offset=offset)
        if lognormal:
            rands = np.asanyarray(
                [rng.lognormal(mean=mu, sigma=sigma, size=size) for rng in generators]
            )
        else:
            rands = np.asanyarray(
                [rng.normal(loc=mu, scale=sigma, size=size) for rng in generators]
            )
        return rands

    def draw_choice(
        self,
        seeds: np.ndarray,
        a: np.ndarray,
        size: int,
        replace: bool,
        offset: int = 0,
    ) -> np.ndarray:
        generators = self._generators(seeds, offset=offset)
        return np.concatenate(
            tuple(rng.choice(a, size=size, replace=replace) for rng in generators)
        )

    def instance_factory(self, seed: int):
        return self._make_rng(seed)


class PhiloxAdvanceCandidate(GeneratorBitGenCandidate):
    name = "PhiloxAdvance"

    def __init__(self):
        super().__init__(np.random.Philox, "PhiloxAdvance")

    def _make_bitgen(self, seed: int) -> np.random.Philox:
        return np.random.Philox(seed)

    def _generators(self, seeds: np.ndarray, offset: int = 0):
        for seed in seeds:
            bitgen = self._make_bitgen(int(seed))
            if offset:
                bitgen.advance(offset)
            yield np.random.Generator(bitgen)

    def draw_uniform(self, seeds: np.ndarray, n: int, offset: int = 0) -> np.ndarray:
        out = np.empty((len(seeds), n), dtype=np.float64)
        for i, seed in enumerate(seeds):
            bitgen = self._make_bitgen(int(seed))
            if offset:
                bitgen.advance(offset)
            rng = np.random.Generator(bitgen)
            out[i] = rng.random(n)
        return out

    def draw_normal(
        self,
        seeds: np.ndarray,
        mu: float,
        sigma: float,
        size: int | None = None,
        offset: int = 0,
        lognormal: bool = False,
    ) -> np.ndarray:
        generators = self._generators(seeds, offset=offset)
        if lognormal:
            rands = np.asanyarray(
                [rng.lognormal(mean=mu, sigma=sigma, size=size) for rng in generators]
            )
        else:
            rands = np.asanyarray(
                [rng.normal(loc=mu, scale=sigma, size=size) for rng in generators]
            )
        return rands

    def draw_choice(
        self,
        seeds: np.ndarray,
        a: np.ndarray,
        size: int,
        replace: bool,
        offset: int = 0,
    ) -> np.ndarray:
        generators = self._generators(seeds, offset=offset)
        return np.concatenate(
            tuple(rng.choice(a, size=size, replace=replace) for rng in generators)
        )


class VectorizedChooserHashCandidate(RNGCandidate):
    name = "VectorizedChooserHash"

    # Stateless per-seed mixing keeps chooser draws independent of batch membership/order
    # while still vectorizing across all choosers for speed.
    _MASK = np.uint64((1 << 64) - 1)
    _GOLDEN_GAMMA = np.uint64(0x9E3779B97F4A7C15)
    _MUL1 = np.uint64(0xBF58476D1CE4E5B9)
    _MUL2 = np.uint64(0x94D049BB133111EB)

    def _splitmix64_next(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state = (state + self._GOLDEN_GAMMA) & self._MASK
        z = state.copy()
        z = ((z ^ (z >> np.uint64(30))) * self._MUL1) & self._MASK
        z = ((z ^ (z >> np.uint64(27))) * self._MUL2) & self._MASK
        z = z ^ (z >> np.uint64(31))
        return state, z

    def _uniform_matrix(self, seeds: np.ndarray, n: int, offset: int = 0) -> np.ndarray:
        state = seeds.astype(np.uint64) + np.uint64(offset)
        out = np.empty((len(seeds), n), dtype=np.float64)
        for j in range(n):
            state, z = self._splitmix64_next(state)
            out[:, j] = ((z >> np.uint64(11)).astype(np.float64)) * (1.0 / (1 << 53))
        return out

    def draw_uniform(self, seeds: np.ndarray, n: int, offset: int = 0) -> np.ndarray:
        return self._uniform_matrix(seeds, n=n, offset=offset)

    def draw_normal(
        self,
        seeds: np.ndarray,
        mu: float,
        sigma: float,
        size: int | None = None,
        offset: int = 0,
        lognormal: bool = False,
    ) -> np.ndarray:
        one_size = 1 if size is None else int(size)
        pairs = one_size if one_size % 2 == 0 else one_size + 1
        uniforms = self._uniform_matrix(seeds, n=pairs, offset=offset)
        u1 = np.clip(uniforms[:, 0::2], 1e-12, 1 - 1e-12)
        u2 = uniforms[:, 1::2]
        z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2 * np.pi * u2)
        values = mu + sigma * z0[:, :one_size]
        if lognormal:
            values = np.exp(values)
        return values

    def draw_choice(
        self,
        seeds: np.ndarray,
        a: np.ndarray,
        size: int,
        replace: bool,
        offset: int = 0,
    ) -> np.ndarray:
        draw_size = int(size)

        if draw_size <= 0:
            return np.empty((len(seeds), 0), dtype=a.dtype)

        if not replace and draw_size > len(a):
            raise ValueError(
                f"Cannot take a sample larger than population when replace=False: {draw_size} > {len(a)}"
            )

        uniforms = self._uniform_matrix(seeds, n=draw_size, offset=offset)

        if replace:
            idx = np.floor(uniforms * len(a)).astype(np.int64)
            idx = np.clip(idx, 0, len(a) - 1)
            return a[idx]

        # Deterministic without-replacement per row: draw one key per alternative,
        # then take the alternatives with the lowest keys.
        keys = self._uniform_matrix(seeds, n=len(a), offset=offset)
        picks = np.argsort(keys, axis=1)[:, :draw_size]
        return a[picks]

    def instance_factory(self, seed: int):
        return np.random.Generator(np.random.PCG64(seed))


def available_candidates() -> dict[str, RNGCandidate]:
    return {
        "RandomState": LegacyRandomStateCandidate(),
        "GeneratorPCG64": GeneratorBitGenCandidate(np.random.PCG64, "GeneratorPCG64"),
        "GeneratorSFC64": GeneratorBitGenCandidate(np.random.SFC64, "GeneratorSFC64"),
        "GeneratorPhilox": GeneratorBitGenCandidate(
            np.random.Philox, "GeneratorPhilox"
        ),
        "GeneratorMT19937": GeneratorBitGenCandidate(
            np.random.MT19937, "GeneratorMT19937"
        ),
        "PhiloxAdvance": PhiloxAdvanceCandidate(),
        "VectorizedChooserHash": VectorizedChooserHashCandidate(),
    }


def summarize_times(times: list[float]) -> tuple[float, float]:
    arr = np.asarray(times, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def make_result(
    scenario: str,
    candidate: RNGCandidate,
    rows: int,
    draws_per_reseed: int,
    draws: int,
    repeat: int,
    times: list[float],
    peak_memory_bytes: int,
    notes: str = "",
) -> BenchmarkResult:
    mean_seconds, std_seconds = summarize_times(times)
    throughput = 0.0 if mean_seconds <= 0 else draws / mean_seconds
    return BenchmarkResult(
        scenario=scenario,
        candidate=candidate.name,
        rows=rows,
        draws_per_reseed=draws_per_reseed,
        draws=draws,
        repeat=repeat,
        mean_seconds=mean_seconds,
        std_seconds=std_seconds,
        throughput_draws_per_sec=throughput,
        peak_memory_bytes=peak_memory_bytes,
        notes=notes,
    )


def benchmark_seed_generation(rows: int, repeat: int) -> BenchmarkResult:
    def workload():
        return activitysim_row_seeds(
            rows=rows,
            base_seed=12345,
            channel_name="persons",
            step_name="trip_mode_choice",
        )

    times, peak = timed_repeats(workload, repeat)
    draws = rows
    return make_result(
        scenario="A0_seed_formula",
        candidate=LegacyRandomStateCandidate(),
        rows=rows,
        draws_per_reseed=1,
        draws=draws,
        repeat=repeat,
        times=times,
        peak_memory_bytes=peak,
        notes="Vectorized seed generation only",
    )


def benchmark_scenario_uniform(
    candidate: RNGCandidate,
    rows: int,
    n_draws: int,
    repeat: int,
) -> BenchmarkResult:
    seeds = activitysim_row_seeds(rows, 12345, "persons", "logit_make_choices")

    def workload():
        return candidate.draw_uniform(seeds=seeds, n=n_draws, offset=0)

    times, peak = timed_repeats(workload, repeat)
    return make_result(
        scenario=f"A_uniform_n{n_draws}",
        candidate=candidate,
        rows=rows,
        draws_per_reseed=n_draws,
        draws=rows * n_draws,
        repeat=repeat,
        times=times,
        peak_memory_bytes=peak,
    )


def benchmark_scenario_normal(
    candidate: RNGCandidate,
    rows: int,
    size: int,
    repeat: int,
) -> BenchmarkResult:
    seeds = activitysim_row_seeds(rows, 12345, "households", "annotate_households")

    def workload():
        return candidate.draw_normal(
            seeds=seeds,
            mu=0.5,
            sigma=1.25,
            size=size,
            offset=0,
            lognormal=False,
        )

    times, peak = timed_repeats(workload, repeat)
    return make_result(
        scenario=f"C_normal_size{size}",
        candidate=candidate,
        rows=rows,
        draws_per_reseed=size,
        draws=rows * size,
        repeat=repeat,
        times=times,
        peak_memory_bytes=peak,
    )


def benchmark_scenario_choice(
    candidate: RNGCandidate,
    rows: int,
    a_size: int,
    size: int,
    replace: bool,
    repeat: int,
) -> BenchmarkResult:
    if not candidate.supports_choice(replace=replace):
        return BenchmarkResult(
            scenario=f"D_choice_size{size}_replace{replace}",
            candidate=candidate.name,
            rows=rows,
            draws_per_reseed=size,
            draws=0,
            repeat=repeat,
            mean_seconds=math.nan,
            std_seconds=math.nan,
            throughput_draws_per_sec=math.nan,
            peak_memory_bytes=0,
            notes="Skipped: candidate does not support requested choice mode",
        )

    seeds = activitysim_row_seeds(rows, 12345, "tours", "interaction_sample")
    a = np.arange(a_size, dtype=np.int64)

    def workload():
        return candidate.draw_choice(
            seeds=seeds,
            a=a,
            size=size,
            replace=replace,
            offset=0,
        )

    times, peak = timed_repeats(workload, repeat)
    return make_result(
        scenario=f"D_choice_size{size}_replace{replace}",
        candidate=candidate,
        rows=rows,
        draws_per_reseed=size,
        draws=rows * size,
        repeat=repeat,
        times=times,
        peak_memory_bytes=peak,
    )


def benchmark_scenario_fast_forward(
    candidate: RNGCandidate,
    rows: int,
    offset: int,
    repeat: int,
) -> BenchmarkResult:
    seeds = activitysim_row_seeds(rows, 12345, "trips", "trip_destination")

    def workload():
        return candidate.draw_uniform(seeds=seeds, n=1, offset=offset)

    times, peak = timed_repeats(workload, repeat)
    return make_result(
        scenario=f"E_fast_forward_offset{offset}",
        candidate=candidate,
        rows=rows,
        draws_per_reseed=1,
        draws=rows,
        repeat=repeat,
        times=times,
        peak_memory_bytes=peak,
    )


def benchmark_memory_instances(
    candidate: RNGCandidate, count: int, repeat: int
) -> BenchmarkResult:
    def workload():
        objs = [candidate.instance_factory(i + 1) for i in range(count)]
        return objs

    times, peak = timed_repeats(workload, repeat)
    return make_result(
        scenario="F_instance_memory",
        candidate=candidate,
        rows=count,
        draws_per_reseed=1,
        draws=count,
        repeat=repeat,
        times=times,
        peak_memory_bytes=peak,
        notes="Object instantiation memory stress",
    )


def benchmark_location_choice_reseed_sweep(
    candidate: RNGCandidate,
    reseeds: int,
    draws_per_reseed: int,
    repeat: int,
    offset: int = 0,
) -> BenchmarkResult:
    seeds = activitysim_row_seeds(reseeds, 12345, "persons", "logit_make_choices")

    def workload():
        return candidate.draw_uniform(seeds=seeds, n=draws_per_reseed, offset=offset)

    times, peak = timed_repeats(workload, repeat)
    return make_result(
        scenario="LC_reseed_sweep",
        candidate=candidate,
        rows=reseeds,
        draws_per_reseed=draws_per_reseed,
        draws=reseeds * draws_per_reseed,
        repeat=repeat,
        times=times,
        peak_memory_bytes=peak,
        notes=f"fixed_draws_per_reseed={draws_per_reseed}; offset={offset}",
    )


def benchmark_location_choice_draw_sweep(
    candidate: RNGCandidate,
    reseeds: int,
    draws_per_reseed: int,
    repeat: int,
    offset: int = 0,
) -> BenchmarkResult:
    seeds = activitysim_row_seeds(reseeds, 12345, "persons", "logit_make_choices")

    def workload():
        return candidate.draw_uniform(seeds=seeds, n=draws_per_reseed, offset=offset)

    times, peak = timed_repeats(workload, repeat)
    return make_result(
        scenario="LC_draw_sweep",
        candidate=candidate,
        rows=reseeds,
        draws_per_reseed=draws_per_reseed,
        draws=reseeds * draws_per_reseed,
        repeat=repeat,
        times=times,
        peak_memory_bytes=peak,
        notes=f"fixed_reseeds={reseeds}; offset={offset}",
    )


def ks_one_sample_uniform(values: np.ndarray) -> tuple[float, float]:
    x = np.sort(values)
    n = x.size
    if n == 0:
        return math.nan, math.nan

    i = np.arange(1, n + 1)
    d_plus = np.max(i / n - x)
    d_minus = np.max(x - (i - 1) / n)
    d = float(max(d_plus, d_minus))

    en = math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)
    lam = en * d
    # Asymptotic Kolmogorov distribution approximation.
    p = 0.0
    for k in range(1, 101):
        p += ((-1) ** (k - 1)) * math.exp(-2.0 * (k * k) * (lam * lam))
    p = float(min(max(2.0 * p, 0.0), 1.0))
    return d, p


def reproducibility_check(
    candidate: RNGCandidate, rows: int = 256, n_draws: int = 8
) -> tuple[bool, str]:
    seeds = activitysim_row_seeds(rows, 12345, "persons", "repro_test")
    a = candidate.draw_uniform(seeds, n=n_draws, offset=5)
    b = candidate.draw_uniform(seeds, n=n_draws, offset=5)

    same_seed_ok = np.array_equal(a, b)

    alt_seeds = activitysim_row_seeds(rows, 12346, "persons", "repro_test")
    c = candidate.draw_uniform(alt_seeds, n=n_draws, offset=5)
    different_seed_differs = not np.array_equal(a, c)

    ok = same_seed_ok and different_seed_differs
    message = f"same_seed_equal={same_seed_ok}; different_seed_differs={different_seed_differs}"
    return ok, message


def quality_check_uniform(
    candidate: RNGCandidate, n: int = 10_000
) -> tuple[float, float, float, float, float, float]:
    seeds = activitysim_row_seeds(n, 7, "persons", "quality_test")
    values = candidate.draw_uniform(seeds, n=1).reshape(-1)
    mean = float(np.mean(values))
    std = float(np.std(values))
    min_v = float(np.min(values))
    max_v = float(np.max(values))
    d, p = ks_one_sample_uniform(values)
    return mean, std, min_v, max_v, d, p


def seed_invariance_unit_test(
    candidate: RNGCandidate,
    draws_per_seed: int = 16,
    offset: int = 7,
) -> SeedUnitTestResult:
    """
    Small unit test for chooser-id invariance.

    For a fixed chooser seed, draws should be identical regardless of which
    other chooser seeds are present in the batch or their order.
    """

    target_seed = np.uint32(987654321)

    solo = np.asarray([target_seed], dtype=np.uint32)
    batch_a = np.asarray(
        [111111111, target_seed, 222222222, 333333333], dtype=np.uint32
    )
    batch_b = np.asarray(
        [333333333, 222222222, target_seed, 111111111], dtype=np.uint32
    )

    solo_draw = candidate.draw_uniform(solo, n=draws_per_seed, offset=offset)[0]

    idx_a = int(np.where(batch_a == target_seed)[0][0])
    idx_b = int(np.where(batch_b == target_seed)[0][0])

    batch_a_draw = candidate.draw_uniform(batch_a, n=draws_per_seed, offset=offset)[
        idx_a
    ]
    batch_b_draw = candidate.draw_uniform(batch_b, n=draws_per_seed, offset=offset)[
        idx_b
    ]

    same_a = np.array_equal(solo_draw, batch_a_draw)
    same_b = np.array_equal(solo_draw, batch_b_draw)
    passed = bool(same_a and same_b)

    if passed:
        message = (
            "PASS: target chooser draws are invariant to batch membership and order"
        )
    else:
        message = (
            "FAIL: target chooser draws changed with batch membership/order "
            f"(same_vs_batch_a={same_a}, same_vs_batch_b={same_b})"
        )

    return SeedUnitTestResult(candidate=candidate.name, passed=passed, message=message)


def run_seed_invariance_unit_tests(
    settings: dict, candidates: list[RNGCandidate]
) -> bool:
    draws_per_seed = int(settings.get("seed_unit_test_draws", 16))
    offset = int(settings.get("seed_unit_test_offset", 7))

    print("\nSeed invariance unit tests")
    print("-" * 80)
    print(f"Test config: draws_per_seed={draws_per_seed}, offset={offset}")

    all_passed = True
    for candidate in candidates:
        result = seed_invariance_unit_test(
            candidate,
            draws_per_seed=draws_per_seed,
            offset=offset,
        )
        status = "PASS" if result.passed else "FAIL"
        print(f"{result.candidate:22} {status:4} {result.message}")
        all_passed = all_passed and result.passed

    return all_passed


def print_results(results: list[BenchmarkResult]) -> None:
    header = (
        f"{'Scenario':30} {'Candidate':22} {'Reseeds':>9} {'Draws/Seed':>12} {'TotalDraws':>12} {'Mean(s)':>10} "
        f"{'Std(s)':>10} {'Draws/s':>14} {'Peak MB':>10}"
    )
    print("\n" + header)
    print("-" * len(header))

    for r in results:
        peak_mb = r.peak_memory_bytes / (1024 * 1024)
        mean_str = (
            f"{r.mean_seconds:10.6f}" if np.isfinite(r.mean_seconds) else f"{'nan':>10}"
        )
        std_str = (
            f"{r.std_seconds:10.6f}" if np.isfinite(r.std_seconds) else f"{'nan':>10}"
        )
        tput_str = (
            f"{r.throughput_draws_per_sec:14.1f}"
            if np.isfinite(r.throughput_draws_per_sec)
            else f"{'nan':>14}"
        )
        print(
            f"{r.scenario:30} {r.candidate:22} {r.rows:9d} {r.draws_per_reseed:12d} {r.draws:12d} "
            f"{mean_str} {std_str} {tput_str} {peak_mb:10.2f}"
        )
        if r.notes:
            print(f"{'':30} {'':22} Notes: {r.notes}")


def write_csv(results: list[BenchmarkResult], csv_path: str) -> None:
    fields = [
        "scenario",
        "candidate",
        "rows",
        "draws_per_reseed",
        "draws",
        "repeat",
        "mean_seconds",
        "std_seconds",
        "throughput_draws_per_sec",
        "peak_memory_bytes",
        "notes",
    ]
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow(row.__dict__)


def _sanitize_filename(value: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in value)


def _metric_label(metric: str) -> str:
    labels = {
        "mean_seconds": "Mean Time (seconds)",
        "std_seconds": "Std Dev Time (seconds)",
        "throughput_draws_per_sec": "Throughput (draws/second)",
        "peak_memory_bytes": "Peak Memory (bytes)",
    }
    return labels.get(metric, metric)


def _plot_single_sweep(
    results: list[BenchmarkResult],
    scenario: str,
    x_field: str,
    metric: str,
    output_dir: str,
    title: str,
    xlabel: str,
) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nSkipping plots: matplotlib is not installed in this environment.")
        return None

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario_rows = [r for r in results if r.scenario == scenario]
    if not scenario_rows:
        return None

    candidates = sorted({r.candidate for r in scenario_rows})
    fig, ax = plt.subplots(figsize=(10, 6))
    drew_any = False

    for candidate in candidates:
        candidate_rows = [r for r in scenario_rows if r.candidate == candidate]
        points: list[tuple[int, float]] = []
        for row in candidate_rows:
            y = getattr(row, metric)
            if not np.isfinite(y):
                continue
            x = getattr(row, x_field)
            points.append((int(x), float(y)))

        points.sort(key=lambda p: p[0])
        if len(points) < 2:
            continue

        xs, ys = zip(*points)
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=4, label=candidate)
        drew_any = True

    if not drew_any:
        plt.close(fig)
        return None

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(_metric_label(metric))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    file_name = f"{_sanitize_filename(scenario)}__{metric}__vs_{x_field}.png"
    save_path = out_dir / file_name
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    return str(save_path)


def plot_location_choice_sweeps(
    results: list[BenchmarkResult],
    output_dir: str,
    metric: str = "mean_seconds",
) -> list[str]:
    saved_paths: list[str] = []

    reseed_rows = [r for r in results if r.scenario == "LC_reseed_sweep"]
    fixed_draws = (
        sorted({int(r.draws_per_reseed) for r in reseed_rows}) if reseed_rows else []
    )
    fixed_draws_text = (
        str(fixed_draws[0])
        if len(fixed_draws) == 1
        else ",".join(map(str, fixed_draws))
    )

    reseed_plot = _plot_single_sweep(
        results=results,
        scenario="LC_reseed_sweep",
        x_field="rows",
        metric=metric,
        output_dir=output_dir,
        title=f"Runtime vs Number of Reseeds (Fixed draws per reseed = {fixed_draws_text})",
        xlabel="Number of Reseeds",
    )
    if reseed_plot:
        saved_paths.append(reseed_plot)

    draw_rows = [r for r in results if r.scenario == "LC_draw_sweep"]
    fixed_reseeds = sorted({int(r.rows) for r in draw_rows}) if draw_rows else []
    fixed_reseeds_text = (
        str(fixed_reseeds[0])
        if len(fixed_reseeds) == 1
        else ",".join(map(str, fixed_reseeds))
    )

    draw_plot = _plot_single_sweep(
        results=results,
        scenario="LC_draw_sweep",
        x_field="draws_per_reseed",
        metric=metric,
        output_dir=output_dir,
        title=f"Runtime vs Draws per Reseed (Fixed reseeds = {fixed_reseeds_text})",
        xlabel="Draws per Reseed",
    )
    if draw_plot:
        saved_paths.append(draw_plot)

    return saved_paths


def select_candidates(settings: dict) -> list[RNGCandidate]:
    candidate_map = available_candidates()
    requested = settings.get("candidates", "all")

    if isinstance(requested, str) and requested.lower() == "all":
        return list(candidate_map.values())

    if not isinstance(requested, list):
        raise ValueError("RUN_SETTINGS['candidates'] must be 'all' or a list of names")

    missing = [name for name in requested if name not in candidate_map]
    if missing:
        raise ValueError(
            f"Unknown candidate names in RUN_SETTINGS['candidates']: {missing}"
        )

    return [candidate_map[name] for name in requested]


def _deep_update(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def resolve_run_settings(settings: dict) -> dict:
    resolved = deepcopy(settings)
    profile = str(resolved.get("run_profile", "full")).strip().lower()

    if profile not in RUN_PROFILES:
        valid = ", ".join(sorted(RUN_PROFILES.keys()))
        raise ValueError(f"Unknown run_profile '{profile}'. Valid options: {valid}")

    return _deep_update(resolved, RUN_PROFILES[profile])


def main() -> None:
    settings = resolve_run_settings(RUN_SETTINGS)
    print(f"Using run profile: {settings.get('run_profile', 'full')}")
    run_mode = str(settings.get("run_mode", "benchmark")).strip().lower()
    valid_run_modes = {"benchmark", "seed_unit_test", "both"}
    if run_mode not in valid_run_modes:
        valid = ", ".join(sorted(valid_run_modes))
        raise ValueError(f"Unknown run_mode '{run_mode}'. Valid options: {valid}")

    print(f"Using run mode: {run_mode}")
    candidates = select_candidates(settings)
    repeat = int(settings.get("repeat", 5))

    if run_mode in {"seed_unit_test", "both"}:
        all_passed = run_seed_invariance_unit_tests(settings, candidates)
        if run_mode == "seed_unit_test":
            if not all_passed:
                raise SystemExit(1)
            return

    results: list[BenchmarkResult] = []

    reseed_sweep = settings.get("reseed_sweep", {})
    if reseed_sweep.get("enabled", True):
        draws_fixed = int(reseed_sweep.get("draws_per_reseed", 10))
        offset = int(reseed_sweep.get("offset", 0))
        for reseeds in [int(v) for v in reseed_sweep.get("reseeds", [])]:
            for candidate in candidates:
                print(
                    f"Testing candidate: {candidate.name} [reseed_sweep reseeds={reseeds}, draws_per_reseed={draws_fixed}]",
                    flush=True,
                )
                results.append(
                    benchmark_location_choice_reseed_sweep(
                        candidate=candidate,
                        reseeds=reseeds,
                        draws_per_reseed=draws_fixed,
                        repeat=repeat,
                        offset=offset,
                    )
                )

    draw_sweep = settings.get("draw_sweep", {})
    if draw_sweep.get("enabled", True):
        reseeds_fixed = int(draw_sweep.get("reseeds", 100))
        offset = int(draw_sweep.get("offset", 0))
        for draws_per_reseed in [
            int(v) for v in draw_sweep.get("draws_per_reseed", [])
        ]:
            for candidate in candidates:
                print(
                    f"Testing candidate: {candidate.name} [draw_sweep reseeds={reseeds_fixed}, draws_per_reseed={draws_per_reseed}]",
                    flush=True,
                )
                results.append(
                    benchmark_location_choice_draw_sweep(
                        candidate=candidate,
                        reseeds=reseeds_fixed,
                        draws_per_reseed=draws_per_reseed,
                        repeat=repeat,
                        offset=offset,
                    )
                )

    print_results(results)

    csv_path = str(settings.get("csv", "")).strip()
    if csv_path:
        write_csv(results, csv_path)
        print(f"\nWrote benchmark results to: {csv_path}")

    plot_settings = settings.get("plots", {})
    if plot_settings.get("enabled", False):
        plot_files = plot_location_choice_sweeps(
            results=results,
            output_dir=str(plot_settings.get("output_dir", "benchmark_rng_plots")),
            metric=str(plot_settings.get("metric", "mean_seconds")),
        )
        if plot_files:
            print(
                f"\nWrote {len(plot_files)} plot files to: {plot_settings.get('output_dir', 'benchmark_rng_plots')}"
            )
        else:
            print(
                "\nNo plots were created (insufficient x-axis variation or plotting unavailable)."
            )

    if bool(settings.get("run_repro_check", True)):
        print("\nReproducibility checks")
        print("-" * 80)
        for candidate in candidates:
            ok, msg = reproducibility_check(candidate)
            print(f"{candidate.name:22} {msg}")

    if bool(settings.get("run_quality_check", True)):
        print("\nUniform quality sanity checks")
        print("-" * 80)
        print(
            f"{'Candidate':22} {'mean':>10} {'std':>10} {'min':>10} {'max':>10} {'KS_D':>10} {'KS_p':>10}"
        )
        for candidate in candidates:
            mean, std, min_v, max_v, d, p = quality_check_uniform(
                candidate,
                n=int(settings.get("quality_sample", 100000)),
            )
            print(
                f"{candidate.name:22} {mean:10.6f} {std:10.6f} {min_v:10.6f} {max_v:10.6f} {d:10.6f} {p:10.6f}"
            )


if __name__ == "__main__":
    main()
