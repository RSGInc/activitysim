from __future__ import annotations

import csv
import gc
import hashlib
import math
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

_MAX_SEED = 1 << 32
_SEED_MASK = 0xFFFFFFFF


WORKFLOW_DEFS = {
    "prob_sample_maz_prob_choice": {
        "sample_method": "prob",
        "final_method": "prob",
        "label": "prob sample + MAZ + prob choice",
    },
    "utility_sample_maz_prob_choice": {
        "sample_method": "utility",
        "final_method": "prob",
        "label": "utility sample + MAZ + prob choice",
    },
    "utility_sample_maz_dense_eet": {
        "sample_method": "utility",
        "final_method": "dense_eet",
        "label": "utility sample + MAZ + dense EET",
    },
}


RUN_SETTINGS = {
    "run_profile": "fast",
    "run_mode": "both",
    "repeat": 3,
    "candidates": "all",
    "csv": "output/benchmark_rng_results.csv",
    "run_repro_check": True,
    "run_quality_check": True,
    "run_workflow_check": True,
    "quality_sample": 100000,
    "plots": {
        "enabled": True,
        "output_dir": "output",
        "metric": "mean_seconds",
    },
    "destination_choice": {
        "workflow_variants": list(WORKFLOW_DEFS.keys()),
        "draw_structures": ["batched", "serial", "ragged"],
        "maz_shape": "uniform",
        "serial_dense_eet_block": 128,
        "chooser_sweep": {
            "enabled": True,
            "chooser_counts": [250, 1000],
            "taz_count": 128,
            "sample_size": 12,
            "max_maz_count": 24,
            "dense_alt_count": 4096,
        },
        "sample_sweep": {
            "enabled": True,
            "chooser_count": 1000,
            "taz_count": 128,
            "sample_sizes": [8, 12, 20],
            "max_maz_count": 24,
            "dense_alt_count": 4096,
        },
        "maz_sweep": {
            "enabled": True,
            "chooser_count": 1000,
            "taz_count": 128,
            "sample_size": 12,
            "max_maz_counts": [8, 16, 24],
            "dense_alt_count": 4096,
        },
    },
}


RUN_PROFILES = {
    "fast": {
        "run_mode": "benchmark",
        "repeat": 1,
        "run_repro_check": False,
        "run_quality_check": False,
        "run_workflow_check": False,
        "quality_sample": 5000,
        "destination_choice": {
            "workflow_variants": ["prob_sample_maz_prob_choice"],
            "draw_structures": ["batched", "ragged"],
            "serial_dense_eet_block": 128,
            "chooser_sweep": {
                "enabled": True,
                "chooser_counts": [128, 256],
                "taz_count": 48,
                "sample_size": 4,
                "max_maz_count": 6,
                "dense_alt_count": 288,
            },
            "sample_sweep": {
                "enabled": True,
                "chooser_count": 256,
                "taz_count": 48,
                "sample_sizes": [4],
                "max_maz_count": 6,
                "dense_alt_count": 288,
            },
            "maz_sweep": {
                "enabled": True,
                "chooser_count": 256,
                "taz_count": 48,
                "sample_size": 4,
                "max_maz_counts": [6],
                "dense_alt_count": 288,
            },
        },
    },
    "full": {
        "repeat": 3,
        "run_repro_check": True,
        "run_quality_check": True,
        "quality_sample": 50000,
        "destination_choice": {
            "serial_dense_eet_block": 256,
            "chooser_sweep": {
                "chooser_counts": [500, 2000, 5000],
                "taz_count": 256,
                "sample_size": 16,
                "max_maz_count": 32,
                "dense_alt_count": 8192,
            },
            "sample_sweep": {
                "chooser_count": 2000,
                "taz_count": 256,
                "sample_sizes": [8, 16, 24],
                "max_maz_count": 32,
                "dense_alt_count": 8192,
            },
            "maz_sweep": {
                "chooser_count": 2000,
                "taz_count": 256,
                "sample_size": 16,
                "max_maz_counts": [12, 24, 40],
                "dense_alt_count": 8192,
            },
        },
    },
}


@dataclass(frozen=True)
class DestinationChoiceSpec:
    scenario: str
    x_field: str
    workflow: str
    draw_structure: str
    chooser_count: int
    taz_count: int
    sample_size: int
    max_maz_count: int
    dense_alt_count: int
    maz_shape: str = "ragged"
    serial_dense_eet_block: int = 128


@dataclass
class DestinationChoiceContext:
    taz_prob_cdf: np.ndarray
    taz_sample_utilities: np.ndarray
    maz_counts: np.ndarray
    maz_weights_padded: np.ndarray
    maz_id_lookup: np.ndarray
    maz_weight_lists: list[np.ndarray]
    maz_id_lists: list[np.ndarray]
    dense_alt_utilities: np.ndarray


@dataclass
class WorkflowOutput:
    sampled_taz: np.ndarray
    sampled_maz: np.ndarray
    final_choices: np.ndarray
    end_offset: int


@dataclass
class BenchmarkResult:
    scenario: str
    x_field: str
    workflow: str
    draw_structure: str
    candidate: str
    chooser_count: int
    taz_count: int
    sample_size: int
    max_maz_count: int
    dense_alt_count: int
    end_offset: int
    total_draws: int
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


@dataclass
class WorkflowUnitTestResult:
    candidate: str
    workflow: str
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


def _uniforms_to_gumbel(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-12, 1 - 1e-12)
    return -np.log(-np.log(clipped))


class RNGCandidate:
    name = "base"

    def draw_uniform(self, seeds: np.ndarray, n: int, offset: int = 0) -> np.ndarray:
        raise NotImplementedError

    def draw_gumbel(self, seeds: np.ndarray, n: int, offset: int = 0) -> np.ndarray:
        raise NotImplementedError

    def instance_factory(self, seed: int):
        raise NotImplementedError


class LegacyRandomStateCandidate(RNGCandidate):
    name = "RandomState"

    def draw_uniform(self, seeds: np.ndarray, n: int, offset: int = 0) -> np.ndarray:
        prng = np.random.RandomState()
        out = np.empty((len(seeds), n), dtype=np.float64)
        for i, seed in enumerate(seeds):
            prng.seed(int(seed))
            if offset:
                prng.rand(offset)
            out[i] = prng.rand(n)
        return out

    def draw_gumbel(self, seeds: np.ndarray, n: int, offset: int = 0) -> np.ndarray:
        return _uniforms_to_gumbel(self.draw_uniform(seeds, n=n, offset=offset))

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

    def draw_uniform(self, seeds: np.ndarray, n: int, offset: int = 0) -> np.ndarray:
        out = np.empty((len(seeds), n), dtype=np.float64)
        for i, seed in enumerate(seeds):
            rng = self._make_rng(int(seed))
            if offset:
                rng.random(offset)
            out[i] = rng.random(n)
        return out

    def draw_gumbel(self, seeds: np.ndarray, n: int, offset: int = 0) -> np.ndarray:
        return _uniforms_to_gumbel(self.draw_uniform(seeds, n=n, offset=offset))

    def instance_factory(self, seed: int):
        return self._make_rng(seed)


class PhiloxAdvanceCandidate(GeneratorBitGenCandidate):
    name = "PhiloxAdvance"

    def __init__(self):
        super().__init__(np.random.Philox, "PhiloxAdvance")

    def _make_bitgen(self, seed: int) -> np.random.Philox:
        return np.random.Philox(seed)

    def draw_uniform(self, seeds: np.ndarray, n: int, offset: int = 0) -> np.ndarray:
        out = np.empty((len(seeds), n), dtype=np.float64)
        for i, seed in enumerate(seeds):
            bitgen = self._make_bitgen(int(seed))
            if offset:
                bitgen.advance(offset)
            out[i] = np.random.Generator(bitgen).random(n)
        return out

    def draw_gumbel(self, seeds: np.ndarray, n: int, offset: int = 0) -> np.ndarray:
        return _uniforms_to_gumbel(self.draw_uniform(seeds, n=n, offset=offset))


class VectorizedChooserHashCandidate(RNGCandidate):
    name = "VectorizedChooserHash"

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
        offset_state = np.uint64(
            (int(offset) * int(self._GOLDEN_GAMMA)) & ((1 << 64) - 1)
        )
        state = (seeds.astype(np.uint64) + offset_state) & self._MASK
        out = np.empty((len(seeds), n), dtype=np.float64)
        for j in range(n):
            state, z = self._splitmix64_next(state)
            out[:, j] = ((z >> np.uint64(11)).astype(np.float64)) * (1.0 / (1 << 53))
        return out

    def draw_uniform(self, seeds: np.ndarray, n: int, offset: int = 0) -> np.ndarray:
        return self._uniform_matrix(seeds, n=n, offset=offset)

    def draw_gumbel(self, seeds: np.ndarray, n: int, offset: int = 0) -> np.ndarray:
        return _uniforms_to_gumbel(self._uniform_matrix(seeds, n=n, offset=offset))

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


def _deep_copy_dict(value: dict) -> dict:
    out: dict = {}
    for key, item in value.items():
        out[key] = _deep_copy_dict(item) if isinstance(item, dict) else item
    return out


def _deep_update(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def resolve_run_settings(settings: dict) -> dict:
    resolved = {k: v for k, v in settings.items()}
    for key, value in settings.items():
        if isinstance(value, dict):
            resolved[key] = _deep_copy_dict(value)
    profile = str(resolved.get("run_profile", "full")).strip().lower()
    if profile not in RUN_PROFILES:
        valid = ", ".join(sorted(RUN_PROFILES.keys()))
        raise ValueError(f"Unknown run_profile '{profile}'. Valid options: {valid}")
    return _deep_update(resolved, RUN_PROFILES[profile])


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


def _build_maz_counts(taz_count: int, max_maz_count: int, maz_shape: str) -> np.ndarray:
    if max_maz_count <= 0:
        raise ValueError("max_maz_count must be positive")
    if maz_shape == "uniform":
        return np.full(taz_count, max_maz_count, dtype=np.int64)
    if maz_shape != "ragged":
        raise ValueError(f"Unknown maz_shape '{maz_shape}'")
    counts = 1 + ((np.arange(taz_count, dtype=np.int64) * 7 + 3) % max_maz_count)
    return counts.astype(np.int64)


def build_destination_context(spec: DestinationChoiceSpec) -> DestinationChoiceContext:
    taz_idx = np.arange(spec.taz_count, dtype=np.float64)
    taz_weights = 1.0 + ((taz_idx % 11.0) + 1.0)
    taz_weights *= 1.0 + 0.15 * np.sin((taz_idx + 1.0) / 7.0)
    taz_weights = np.clip(taz_weights, 1e-6, None)
    taz_probs = taz_weights / taz_weights.sum()
    taz_prob_cdf = np.cumsum(taz_probs)
    taz_prob_cdf[-1] = 1.0

    taz_sample_utilities = np.log(taz_probs + 1e-12) * 0.75 + 0.05 * np.cos(
        taz_idx / 5.0
    )

    maz_counts = _build_maz_counts(spec.taz_count, spec.max_maz_count, spec.maz_shape)
    total_maz_ids = int(maz_counts.sum())
    if total_maz_ids > spec.dense_alt_count:
        raise ValueError(
            "dense_alt_count must cover all concrete MAZ ids in the benchmark: "
            f"need at least {total_maz_ids}, got {spec.dense_alt_count}"
        )

    maz_weights_padded = np.zeros(
        (spec.taz_count, spec.max_maz_count), dtype=np.float64
    )
    maz_id_lookup = np.full((spec.taz_count, spec.max_maz_count), -1, dtype=np.int64)
    maz_weight_lists: list[np.ndarray] = []
    maz_id_lists: list[np.ndarray] = []

    next_maz_id = 0
    for taz in range(spec.taz_count):
        count = int(maz_counts[taz])
        ids = np.arange(next_maz_id, next_maz_id + count, dtype=np.int64)
        next_maz_id += count
        weight_idx = np.arange(count, dtype=np.float64)
        weights = 1.0 + ((weight_idx + taz) % 5.0)
        weights *= 1.0 + 0.05 * np.cos((weight_idx + 1.0) * (taz + 1.0) / 11.0)
        weights = np.clip(weights, 1e-6, None)
        weights /= weights.sum()
        maz_weights_padded[taz, :count] = weights
        maz_id_lookup[taz, :count] = ids
        maz_weight_lists.append(weights.astype(np.float64))
        maz_id_lists.append(ids)

    dense_idx = np.arange(spec.dense_alt_count, dtype=np.float64)
    dense_alt_utilities = 0.15 * np.sin((dense_idx + 1.0) / 13.0) + 0.05 * np.log1p(
        dense_idx
    )

    return DestinationChoiceContext(
        taz_prob_cdf=taz_prob_cdf,
        taz_sample_utilities=taz_sample_utilities.astype(np.float64),
        maz_counts=maz_counts,
        maz_weights_padded=maz_weights_padded,
        maz_id_lookup=maz_id_lookup,
        maz_weight_lists=maz_weight_lists,
        maz_id_lists=maz_id_lists,
        dense_alt_utilities=dense_alt_utilities.astype(np.float64),
    )


def _sample_from_cdf(cdf: np.ndarray, draws: np.ndarray) -> np.ndarray:
    picked = np.searchsorted(cdf, draws.reshape(-1), side="right")
    picked = np.clip(picked, 0, len(cdf) - 1)
    return picked.reshape(draws.shape)


def _prob_sample_stage(
    candidate: RNGCandidate,
    seeds: np.ndarray,
    context: DestinationChoiceContext,
    sample_size: int,
    draw_structure: str,
    offset: int,
) -> tuple[np.ndarray, int]:
    chooser_count = len(seeds)
    sampled_taz = np.empty((chooser_count, sample_size), dtype=np.int64)
    if draw_structure == "batched":
        draws = candidate.draw_uniform(seeds, n=sample_size, offset=offset)
        sampled_taz[:] = _sample_from_cdf(context.taz_prob_cdf, draws)
        offset += sample_size
        return sampled_taz, offset

    for sample_idx in range(sample_size):
        draws = candidate.draw_uniform(seeds, n=1, offset=offset).reshape(-1)
        sampled_taz[:, sample_idx] = _sample_from_cdf(context.taz_prob_cdf, draws)
        offset += 1
    return sampled_taz, offset


def _utility_sample_stage(
    candidate: RNGCandidate,
    seeds: np.ndarray,
    context: DestinationChoiceContext,
    taz_count: int,
    sample_size: int,
    draw_structure: str,
    offset: int,
) -> tuple[np.ndarray, int]:
    chooser_count = len(seeds)
    sampled_taz = np.empty((chooser_count, sample_size), dtype=np.int64)
    if draw_structure == "batched":
        draws = candidate.draw_gumbel(seeds, n=taz_count * sample_size, offset=offset)
        shaped = draws.reshape(chooser_count, taz_count, sample_size)
        sampled_taz[:] = np.argmax(
            shaped + context.taz_sample_utilities[np.newaxis, :, np.newaxis],
            axis=1,
        )
        offset += taz_count * sample_size
        return sampled_taz, offset

    best_scores = np.full((chooser_count, sample_size), -np.inf, dtype=np.float64)
    sampled_taz.fill(-1)
    for alt_idx in range(taz_count):
        draws = candidate.draw_gumbel(seeds, n=sample_size, offset=offset)
        scores = draws + context.taz_sample_utilities[alt_idx]
        better = scores > best_scores
        best_scores[better] = scores[better]
        sampled_taz[better] = alt_idx
        offset += sample_size
    return sampled_taz, offset


def _choose_maz_dense(
    candidate: RNGCandidate,
    seeds: np.ndarray,
    context: DestinationChoiceContext,
    sampled_taz: np.ndarray,
    draw_structure: str,
    offset: int,
) -> tuple[np.ndarray, int]:
    chooser_count, sample_size = sampled_taz.shape
    sampled_maz = np.empty((chooser_count, sample_size), dtype=np.int64)
    if draw_structure == "batched":
        flat_taz = sampled_taz.reshape(-1)
        max_selected = int(context.maz_counts[flat_taz].max())
        weights = context.maz_weights_padded[flat_taz, :max_selected]
        cdf = np.cumsum(weights, axis=1)
        draws = candidate.draw_uniform(seeds, n=sample_size, offset=offset).reshape(
            -1, 1
        )
        positions = np.argmax((cdf - draws) > 0.0, axis=1)
        sampled_maz[:] = context.maz_id_lookup[flat_taz, positions].reshape(
            chooser_count, sample_size
        )
        offset += sample_size
        return sampled_maz, offset

    for sample_idx in range(sample_size):
        taz_col = sampled_taz[:, sample_idx]
        max_selected = int(context.maz_counts[taz_col].max())
        weights = context.maz_weights_padded[taz_col, :max_selected]
        cdf = np.cumsum(weights, axis=1)
        draws = candidate.draw_uniform(seeds, n=1, offset=offset).reshape(-1, 1)
        positions = np.argmax((cdf - draws) > 0.0, axis=1)
        sampled_maz[:, sample_idx] = context.maz_id_lookup[taz_col, positions]
        offset += 1
    return sampled_maz, offset


def _choose_maz_ragged(
    candidate: RNGCandidate,
    seeds: np.ndarray,
    context: DestinationChoiceContext,
    sampled_taz: np.ndarray,
    offset: int,
) -> tuple[np.ndarray, int]:
    chooser_count, sample_size = sampled_taz.shape
    sampled_maz = np.empty((chooser_count, sample_size), dtype=np.int64)
    for sample_idx in range(sample_size):
        taz_col = sampled_taz[:, sample_idx]
        draws = candidate.draw_uniform(seeds, n=1, offset=offset).reshape(-1)
        offset += 1
        for taz in np.unique(taz_col):
            mask = taz_col == taz
            weights = context.maz_weight_lists[int(taz)]
            ids = context.maz_id_lists[int(taz)]
            cdf = np.cumsum(weights)
            positions = np.searchsorted(cdf, draws[mask], side="right")
            positions = np.clip(positions, 0, len(ids) - 1)
            sampled_maz[mask, sample_idx] = ids[positions]
    return sampled_maz, offset


def _final_probability_choice(
    candidate: RNGCandidate,
    seeds: np.ndarray,
    context: DestinationChoiceContext,
    sampled_maz: np.ndarray,
    offset: int,
) -> tuple[np.ndarray, int]:
    weights = np.exp(context.dense_alt_utilities[sampled_maz])
    weights += 0.05 * (np.arange(sampled_maz.shape[1], dtype=np.float64) + 1.0)
    probs = weights / weights.sum(axis=1, keepdims=True)
    draws = candidate.draw_uniform(seeds, n=1, offset=offset).reshape(-1, 1)
    positions = np.argmax((np.cumsum(probs, axis=1) - draws) > 0.0, axis=1)
    choices = sampled_maz[np.arange(sampled_maz.shape[0]), positions]
    return choices, offset + 1


def _final_dense_eet_choice(
    candidate: RNGCandidate,
    seeds: np.ndarray,
    context: DestinationChoiceContext,
    sampled_maz: np.ndarray,
    draw_structure: str,
    dense_alt_count: int,
    dense_eet_block: int,
    offset: int,
) -> tuple[np.ndarray, int]:
    chooser_count, sample_size = sampled_maz.shape
    base_utils = context.dense_alt_utilities[sampled_maz]

    if draw_structure == "batched":
        draws = candidate.draw_gumbel(seeds, n=dense_alt_count, offset=offset)
        selected_noise = np.take_along_axis(draws, sampled_maz, axis=1)
        positions = np.argmax(base_utils + selected_noise, axis=1)
        choices = sampled_maz[np.arange(chooser_count), positions]
        return choices, offset + dense_alt_count

    block = max(1, min(int(dense_eet_block), dense_alt_count))
    selected_noise = np.empty((chooser_count, sample_size), dtype=np.float64)
    next_offset = offset
    for block_start in range(0, dense_alt_count, block):
        block_end = min(block_start + block, dense_alt_count)
        block_size = block_end - block_start
        draws = candidate.draw_gumbel(seeds, n=block_size, offset=next_offset)
        next_offset += block_size
        for col in range(sample_size):
            ids = sampled_maz[:, col]
            mask = (ids >= block_start) & (ids < block_end)
            if mask.any():
                selected_noise[mask, col] = draws[mask, ids[mask] - block_start]

    positions = np.argmax(base_utils + selected_noise, axis=1)
    choices = sampled_maz[np.arange(chooser_count), positions]
    return choices, next_offset


def execute_destination_workflow(
    candidate: RNGCandidate,
    spec: DestinationChoiceSpec,
    context: DestinationChoiceContext,
    seeds: np.ndarray,
) -> WorkflowOutput:
    workflow_def = WORKFLOW_DEFS[spec.workflow]
    offset = 0

    if workflow_def["sample_method"] == "prob":
        sampled_taz, offset = _prob_sample_stage(
            candidate,
            seeds,
            context,
            sample_size=spec.sample_size,
            draw_structure=spec.draw_structure,
            offset=offset,
        )
    else:
        sampled_taz, offset = _utility_sample_stage(
            candidate,
            seeds,
            context,
            taz_count=spec.taz_count,
            sample_size=spec.sample_size,
            draw_structure=spec.draw_structure,
            offset=offset,
        )

    if spec.draw_structure == "ragged":
        sampled_maz, offset = _choose_maz_ragged(
            candidate,
            seeds,
            context,
            sampled_taz=sampled_taz,
            offset=offset,
        )
    else:
        sampled_maz, offset = _choose_maz_dense(
            candidate,
            seeds,
            context,
            sampled_taz=sampled_taz,
            draw_structure=spec.draw_structure,
            offset=offset,
        )

    if workflow_def["final_method"] == "prob":
        final_choices, offset = _final_probability_choice(
            candidate,
            seeds,
            context,
            sampled_maz=sampled_maz,
            offset=offset,
        )
    else:
        final_choices, offset = _final_dense_eet_choice(
            candidate,
            seeds,
            context,
            sampled_maz=sampled_maz,
            draw_structure=spec.draw_structure,
            dense_alt_count=spec.dense_alt_count,
            dense_eet_block=spec.serial_dense_eet_block,
            offset=offset,
        )

    return WorkflowOutput(
        sampled_taz=sampled_taz,
        sampled_maz=sampled_maz,
        final_choices=final_choices,
        end_offset=offset,
    )


def workflow_stage_draws(spec: DestinationChoiceSpec) -> dict[str, int]:
    workflow_def = WORKFLOW_DEFS[spec.workflow]
    sample_draws = spec.sample_size
    if workflow_def["sample_method"] == "utility":
        sample_draws *= spec.taz_count
    final_draws = 1 if workflow_def["final_method"] == "prob" else spec.dense_alt_count
    return {
        "sample": sample_draws,
        "maz": spec.sample_size,
        "final": final_draws,
    }


def _scenario_notes(spec: DestinationChoiceSpec) -> str:
    stage_draws = workflow_stage_draws(spec)
    return (
        f"maz_shape={spec.maz_shape}; draws_per_chooser="
        f"sample:{stage_draws['sample']},maz:{stage_draws['maz']},final:{stage_draws['final']}; "
        f"end_offset={sum(stage_draws.values())}; structure={spec.draw_structure}; dense_eet_block={spec.serial_dense_eet_block}"
    )


def make_result(
    spec: DestinationChoiceSpec,
    candidate: RNGCandidate,
    repeat: int,
    times: list[float],
    peak_memory_bytes: int,
) -> BenchmarkResult:
    mean_seconds, std_seconds = summarize_times(times)
    end_offset = sum(workflow_stage_draws(spec).values())
    total_draws = spec.chooser_count * end_offset
    throughput = 0.0 if mean_seconds <= 0 else total_draws / mean_seconds
    return BenchmarkResult(
        scenario=spec.scenario,
        x_field=spec.x_field,
        workflow=spec.workflow,
        draw_structure=spec.draw_structure,
        candidate=candidate.name,
        chooser_count=spec.chooser_count,
        taz_count=spec.taz_count,
        sample_size=spec.sample_size,
        max_maz_count=spec.max_maz_count,
        dense_alt_count=spec.dense_alt_count,
        end_offset=end_offset,
        total_draws=total_draws,
        repeat=repeat,
        mean_seconds=mean_seconds,
        std_seconds=std_seconds,
        throughput_draws_per_sec=throughput,
        peak_memory_bytes=peak_memory_bytes,
        notes=_scenario_notes(spec),
    )


def benchmark_destination_choice(
    candidate: RNGCandidate,
    spec: DestinationChoiceSpec,
    repeat: int,
) -> BenchmarkResult:
    context = build_destination_context(spec)
    seeds = activitysim_row_seeds(
        spec.chooser_count,
        12345,
        "persons",
        f"destination_choice__{spec.workflow}",
    )

    def workload():
        return execute_destination_workflow(candidate, spec, context, seeds)

    times, peak = timed_repeats(workload, repeat)
    return make_result(spec, candidate, repeat, times, peak)


def _workflow_signature(output: WorkflowOutput, row_idx: int) -> tuple:
    return (
        tuple(output.sampled_taz[row_idx].tolist()),
        tuple(output.sampled_maz[row_idx].tolist()),
        int(output.final_choices[row_idx]),
        int(output.end_offset),
    )


def workflow_invariance_unit_test(
    candidate: RNGCandidate,
    workflow: str,
    draw_structure: str,
    serial_dense_eet_block: int,
) -> WorkflowUnitTestResult:
    target_seed = np.uint32(987654321)
    solo = np.asarray([target_seed], dtype=np.uint32)
    batch_a = np.asarray(
        [111111111, target_seed, 222222222, 333333333], dtype=np.uint32
    )
    batch_b = np.asarray(
        [333333333, 222222222, target_seed, 111111111], dtype=np.uint32
    )
    spec = DestinationChoiceSpec(
        scenario="workflow_check",
        x_field="chooser_count",
        workflow=workflow,
        draw_structure=draw_structure,
        chooser_count=len(batch_a),
        taz_count=24,
        sample_size=4,
        max_maz_count=6,
        dense_alt_count=128,
        serial_dense_eet_block=serial_dense_eet_block,
    )
    context = build_destination_context(spec)

    solo_spec = DestinationChoiceSpec(
        scenario=spec.scenario,
        x_field=spec.x_field,
        workflow=spec.workflow,
        draw_structure=spec.draw_structure,
        chooser_count=1,
        taz_count=spec.taz_count,
        sample_size=spec.sample_size,
        max_maz_count=spec.max_maz_count,
        dense_alt_count=spec.dense_alt_count,
        serial_dense_eet_block=spec.serial_dense_eet_block,
    )
    solo_context = build_destination_context(solo_spec)

    solo_out = execute_destination_workflow(candidate, solo_spec, solo_context, solo)
    batch_a_out = execute_destination_workflow(candidate, spec, context, batch_a)
    batch_b_out = execute_destination_workflow(candidate, spec, context, batch_b)

    idx_a = int(np.where(batch_a == target_seed)[0][0])
    idx_b = int(np.where(batch_b == target_seed)[0][0])

    signature_solo = _workflow_signature(solo_out, 0)
    signature_a = _workflow_signature(batch_a_out, idx_a)
    signature_b = _workflow_signature(batch_b_out, idx_b)

    passed = signature_solo == signature_a == signature_b
    if passed:
        message = f"{draw_structure}: invariant to batch membership/order"
    else:
        message = f"{draw_structure}: batch membership/order changed staged draws"
    return WorkflowUnitTestResult(
        candidate=candidate.name,
        workflow=workflow,
        passed=passed,
        message=message,
    )


def workflow_structure_equivalence_unit_test(
    candidate: RNGCandidate,
    workflow: str,
    serial_dense_eet_block: int,
) -> WorkflowUnitTestResult:
    base_kwargs = {
        "scenario": "workflow_equivalence",
        "x_field": "chooser_count",
        "workflow": workflow,
        "chooser_count": 6,
        "taz_count": 32,
        "sample_size": 5,
        "max_maz_count": 7,
        "dense_alt_count": 192,
        "serial_dense_eet_block": serial_dense_eet_block,
    }
    seeds = activitysim_row_seeds(6, 12345, "persons", f"workflow_eq__{workflow}")
    outputs: dict[str, WorkflowOutput] = {}
    for draw_structure in ["batched", "serial", "ragged"]:
        spec = DestinationChoiceSpec(draw_structure=draw_structure, **base_kwargs)
        context = build_destination_context(spec)
        outputs[draw_structure] = execute_destination_workflow(
            candidate, spec, context, seeds
        )

    baseline = outputs["batched"]
    passed = True
    for output in outputs.values():
        if not np.array_equal(output.sampled_taz, baseline.sampled_taz):
            passed = False
        if not np.array_equal(output.sampled_maz, baseline.sampled_maz):
            passed = False
        if not np.array_equal(output.final_choices, baseline.final_choices):
            passed = False
        if output.end_offset != baseline.end_offset:
            passed = False

    message = (
        "draw structures agree on staged results"
        if passed
        else "draw structures diverged"
    )
    return WorkflowUnitTestResult(
        candidate=candidate.name,
        workflow=workflow,
        passed=passed,
        message=message,
    )


def run_workflow_unit_tests(settings: dict, candidates: list[RNGCandidate]) -> bool:
    dc = settings.get("destination_choice", {})
    serial_dense_eet_block = int(dc.get("serial_dense_eet_block", 128))
    workflows = list(dc.get("workflow_variants", list(WORKFLOW_DEFS.keys())))
    draw_structures = list(dc.get("draw_structures", ["batched", "serial", "ragged"]))
    print("\nDestination workflow unit tests")
    print("-" * 80)
    all_passed = True
    for candidate in candidates:
        for workflow in workflows:
            for draw_structure in draw_structures:
                invariance_result = workflow_invariance_unit_test(
                    candidate,
                    workflow,
                    draw_structure,
                    serial_dense_eet_block=serial_dense_eet_block,
                )
                status = "PASS" if invariance_result.passed else "FAIL"
                print(
                    f"{candidate.name:22} {workflow:30} {status:4} {invariance_result.message}"
                )
                all_passed = all_passed and invariance_result.passed

            equivalence_result = workflow_structure_equivalence_unit_test(
                candidate,
                workflow,
                serial_dense_eet_block=serial_dense_eet_block,
            )
            status = "PASS" if equivalence_result.passed else "FAIL"
            print(
                f"{candidate.name:22} {workflow:30} {status:4} {equivalence_result.message}"
            )
            all_passed = all_passed and equivalence_result.passed
    return all_passed


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
            candidate, draws_per_seed=draws_per_seed, offset=offset
        )
        status = "PASS" if result.passed else "FAIL"
        print(f"{result.candidate:22} {status:4} {result.message}")
        all_passed = all_passed and result.passed
    return all_passed


def iter_destination_choice_specs(settings: dict) -> list[DestinationChoiceSpec]:
    dc = settings.get("destination_choice", {})
    workflows = list(dc.get("workflow_variants", list(WORKFLOW_DEFS.keys())))
    draw_structures = list(dc.get("draw_structures", ["batched", "serial", "ragged"]))
    maz_shape = str(dc.get("maz_shape", "ragged"))
    serial_dense_eet_block = int(dc.get("serial_dense_eet_block", 128))

    specs: list[DestinationChoiceSpec] = []
    chooser_sweep = dc.get("chooser_sweep", {})
    if chooser_sweep.get("enabled", True):
        for chooser_count in [int(v) for v in chooser_sweep.get("chooser_counts", [])]:
            for workflow in workflows:
                for draw_structure in draw_structures:
                    specs.append(
                        DestinationChoiceSpec(
                            scenario=f"DC_chooser_sweep__{workflow}",
                            x_field="chooser_count",
                            workflow=workflow,
                            draw_structure=draw_structure,
                            chooser_count=chooser_count,
                            taz_count=int(chooser_sweep.get("taz_count", 128)),
                            sample_size=int(chooser_sweep.get("sample_size", 12)),
                            max_maz_count=int(chooser_sweep.get("max_maz_count", 24)),
                            dense_alt_count=int(
                                chooser_sweep.get("dense_alt_count", 4096)
                            ),
                            maz_shape=maz_shape,
                            serial_dense_eet_block=serial_dense_eet_block,
                        )
                    )

    sample_sweep = dc.get("sample_sweep", {})
    if sample_sweep.get("enabled", True):
        for sample_size in [int(v) for v in sample_sweep.get("sample_sizes", [])]:
            for workflow in workflows:
                for draw_structure in draw_structures:
                    specs.append(
                        DestinationChoiceSpec(
                            scenario=f"DC_sample_sweep__{workflow}",
                            x_field="sample_size",
                            workflow=workflow,
                            draw_structure=draw_structure,
                            chooser_count=int(sample_sweep.get("chooser_count", 1000)),
                            taz_count=int(sample_sweep.get("taz_count", 128)),
                            sample_size=sample_size,
                            max_maz_count=int(sample_sweep.get("max_maz_count", 24)),
                            dense_alt_count=int(
                                sample_sweep.get("dense_alt_count", 4096)
                            ),
                            maz_shape=maz_shape,
                            serial_dense_eet_block=serial_dense_eet_block,
                        )
                    )

    maz_sweep = dc.get("maz_sweep", {})
    if maz_sweep.get("enabled", True):
        for max_maz_count in [int(v) for v in maz_sweep.get("max_maz_counts", [])]:
            for workflow in workflows:
                for draw_structure in draw_structures:
                    specs.append(
                        DestinationChoiceSpec(
                            scenario=f"DC_maz_sweep__{workflow}",
                            x_field="max_maz_count",
                            workflow=workflow,
                            draw_structure=draw_structure,
                            chooser_count=int(maz_sweep.get("chooser_count", 1000)),
                            taz_count=int(maz_sweep.get("taz_count", 128)),
                            sample_size=int(maz_sweep.get("sample_size", 12)),
                            max_maz_count=max_maz_count,
                            dense_alt_count=int(maz_sweep.get("dense_alt_count", 4096)),
                            maz_shape=maz_shape,
                            serial_dense_eet_block=serial_dense_eet_block,
                        )
                    )

    return specs


def print_results(results: list[BenchmarkResult]) -> None:
    header = (
        f"{'Scenario':30} {'Structure':10} {'Candidate':22} {'Choosers':>9} {'Sample':>8} {'MaxMAZ':>8} "
        f"{'EndOff':>8} {'Draws':>12} {'Mean(s)':>10} {'Draws/s':>14} {'Peak MB':>10}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        peak_mb = r.peak_memory_bytes / (1024 * 1024)
        print(
            f"{r.scenario:30} {r.draw_structure:10} {r.candidate:22} {r.chooser_count:9d} {r.sample_size:8d} {r.max_maz_count:8d} "
            f"{r.end_offset:8d} {r.total_draws:12d} {r.mean_seconds:10.6f} {r.throughput_draws_per_sec:14.1f} {peak_mb:10.2f}"
        )
        print(f"{'':30} {'':10} {'':22} Notes: {r.notes}")


def write_csv(results: list[BenchmarkResult], csv_path: str) -> None:
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf8") as file_obj:
        writer = csv.DictWriter(
            file_obj, fieldnames=list(BenchmarkResult.__dataclass_fields__.keys())
        )
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


def _scenario_title(scenario: str) -> str:
    sweep, workflow = scenario.split("__", maxsplit=1)
    sweep_text = sweep.replace("DC_", "").replace("_", " ").title()
    workflow_text = WORKFLOW_DEFS[workflow]["label"]
    return f"{sweep_text}: {workflow_text}"


def _plot_single_sweep(
    results: list[BenchmarkResult],
    scenario: str,
    x_field: str,
    metric: str,
    output_dir: str,
) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nSkipping plots: matplotlib is not installed in this environment.")
        return None

    scenario_rows = [r for r in results if r.scenario == scenario]
    if not scenario_rows:
        return None

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    drew_any = False
    labels = sorted({f"{r.candidate} [{r.draw_structure}]" for r in scenario_rows})
    for label in labels:
        candidate_name, _, structure = label.partition(" [")
        structure = structure.rstrip("]")
        candidate_rows = [
            r
            for r in scenario_rows
            if r.candidate == candidate_name and r.draw_structure == structure
        ]
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
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=4, label=label)
        drew_any = True

    if not drew_any:
        plt.close(fig)
        return None

    ax.set_title(_scenario_title(scenario))
    ax.set_xlabel(x_field.replace("_", " ").title())
    ax.set_ylabel(_metric_label(metric))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    save_path = out_dir / f"{_sanitize_filename(scenario)}__{metric}__vs_{x_field}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return str(save_path)


def plot_destination_choice_sweeps(
    results: list[BenchmarkResult],
    output_dir: str,
    metric: str = "mean_seconds",
) -> list[str]:
    saved_paths: list[str] = []
    for scenario in sorted({r.scenario for r in results}):
        scenario_rows = [r for r in results if r.scenario == scenario]
        if not scenario_rows:
            continue
        x_field = scenario_rows[0].x_field
        save_path = _plot_single_sweep(
            results=results,
            scenario=scenario,
            x_field=x_field,
            metric=metric,
            output_dir=output_dir,
        )
        if save_path:
            saved_paths.append(save_path)
    return saved_paths


def main() -> None:
    settings = resolve_run_settings(RUN_SETTINGS)
    print(f"Using run profile: {settings.get('run_profile', 'full')}")
    run_mode = str(settings.get("run_mode", "benchmark")).strip().lower()
    valid_run_modes = {
        "benchmark",
        "seed_unit_test",
        "workflow_unit_test",
        "checks",
        "both",
    }
    if run_mode not in valid_run_modes:
        valid = ", ".join(sorted(valid_run_modes))
        raise ValueError(f"Unknown run_mode '{run_mode}'. Valid options: {valid}")

    print(f"Using run mode: {run_mode}")
    candidates = select_candidates(settings)
    repeat = int(settings.get("repeat", 3))

    seed_tests_passed = True
    if run_mode in {"seed_unit_test", "checks", "both"}:
        seed_tests_passed = run_seed_invariance_unit_tests(settings, candidates)
        if run_mode == "seed_unit_test":
            if not seed_tests_passed:
                raise SystemExit(1)
            return

    workflow_tests_passed = True
    if bool(settings.get("run_workflow_check", True)) and run_mode in {
        "workflow_unit_test",
        "checks",
        "both",
    }:
        workflow_tests_passed = run_workflow_unit_tests(settings, candidates)
        if run_mode == "workflow_unit_test":
            if not workflow_tests_passed:
                raise SystemExit(1)
            return

    if run_mode == "checks":
        if not (seed_tests_passed and workflow_tests_passed):
            raise SystemExit(1)
        return

    results: list[BenchmarkResult] = []
    for spec in iter_destination_choice_specs(settings):
        for candidate in candidates:
            print(
                f"Testing candidate: {candidate.name} [{spec.scenario}, structure={spec.draw_structure}, choosers={spec.chooser_count}, sample={spec.sample_size}, max_maz={spec.max_maz_count}]",
                flush=True,
            )
            results.append(benchmark_destination_choice(candidate, spec, repeat))

    print_results(results)

    csv_path = str(settings.get("csv", "")).strip()
    if csv_path:
        write_csv(results, csv_path)
        print(f"\nWrote benchmark results to: {csv_path}")

    plot_settings = settings.get("plots", {})
    if plot_settings.get("enabled", False):
        plot_files = plot_destination_choice_sweeps(
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
            _ok, msg = reproducibility_check(candidate)
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
