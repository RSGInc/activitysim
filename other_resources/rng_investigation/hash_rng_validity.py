from __future__ import annotations

import argparse
import csv
import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

_MAX_SEED = 1 << 32
_SEED_MASK = 0xFFFFFFFF
_MASK_64 = np.uint64((1 << 64) - 1)


@dataclass(frozen=True)
class UniformTestSpec:
    name: str
    seed_count: int
    draws_per_seed: int
    offset: int


@dataclass(frozen=True)
class SparseAltSpec:
    name: str
    seed_count: int
    alt_count: int
    max_alt_id: int
    offset: int


@dataclass(frozen=True)
class ChoiceScenario:
    name: str
    chooser_count: int
    replications: int
    offset: int
    utilities: tuple[float, ...]
    sparse_max_alt_id: int


@dataclass(frozen=True)
class InvarianceSpec:
    chooser_count: int
    sample_size: int
    actual_alt_count: int
    max_alt_id: int
    overlap_ratio: float
    offset: int


@dataclass(frozen=True)
class UniformResult:
    engine: str
    test_name: str
    sample_count: int
    offset: int
    mean: float
    mean_z: float
    variance: float
    variance_pct_error: float
    ks_d: float
    ks_pvalue: float
    max_bin_z: float
    lag1_corr: float
    adjacent_seed_corr: float


@dataclass(frozen=True)
class SparseAltResult:
    engine: str
    test_name: str
    seed_count: int
    alt_count: int
    max_alt_id: int
    offset: int
    alt_mean_id_corr: float
    max_alt_mean_z: float
    alt_mean_std: float
    adjacent_alt_corr: float
    matrix_mean: float
    matrix_variance: float


@dataclass(frozen=True)
class ChoiceLayoutResult:
    engine: str
    scenario: str
    layout: str
    chooser_count: int
    replications: int
    alt_count: int
    offset: int
    max_alt_id: int
    max_abs_error: float
    rmse_error: float
    mean_tv_distance: float
    worst_replication_max_abs_error: float


@dataclass(frozen=True)
class ChoiceDriftResult:
    engine: str
    scenario: str
    chooser_count: int
    replications: int
    alt_count: int
    offset: int
    mean_abs_dense_sparse_drift: float
    max_abs_dense_sparse_drift: float


@dataclass(frozen=True)
class InvarianceResult:
    engine: str
    chooser_count: int
    sample_size: int
    actual_alt_count: int
    max_alt_id: int
    offset: int
    batch_invariant: bool
    sampled_set_invariant: bool
    offset_changes_values: bool
    shared_pairs_checked: int


def hash32(text: str) -> int:
    digest = hashlib.md5(text.encode("utf8")).hexdigest()
    return int(digest, 16) & _SEED_MASK


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


def ks_one_sample_uniform(values: np.ndarray) -> tuple[float, float]:
    x = np.sort(values)
    n = x.size
    if n == 0:
        return math.nan, math.nan
    cdf = np.arange(1, n + 1) / n
    d_plus = np.max(cdf - x)
    d_minus = np.max(x - np.arange(0, n) / n)
    d = float(max(d_plus, d_minus))
    en = math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)
    lam = en * d
    p = 0.0
    for k in range(1, 101):
        p += ((-1) ** (k - 1)) * math.exp(-2.0 * (k * k) * (lam * lam))
    p = max(0.0, min(1.0, 2.0 * p))
    return d, p


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return math.nan
    a_std = float(np.std(a))
    b_std = float(np.std(b))
    if a_std == 0.0 or b_std == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()


def _make_sparse_alt_ids(alt_count: int, max_alt_id: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(1234567 + max_alt_id + alt_count))
    values = np.sort(
        rng.choice(
            np.arange(1, max_alt_id + 1, dtype=np.int64), size=alt_count, replace=False
        )
    )
    return values.astype(np.int64)


def _build_sampled_sets(
    spec: InvarianceSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    actual_alt_ids = _make_sparse_alt_ids(spec.actual_alt_count, spec.max_alt_id)
    rng = np.random.Generator(np.random.PCG64(24681357 + spec.max_alt_id))
    shared_count = int(round(spec.sample_size * spec.overlap_ratio))
    shared_count = max(0, min(shared_count, spec.sample_size))
    sampled_a = np.empty((spec.chooser_count, spec.sample_size), dtype=np.int64)
    sampled_b = np.empty((spec.chooser_count, spec.sample_size), dtype=np.int64)

    for row in range(spec.chooser_count):
        sample_a = np.asarray(
            rng.choice(actual_alt_ids, size=spec.sample_size, replace=False),
            dtype=np.int64,
        )
        rng.shuffle(sample_a)
        sampled_a[row] = sample_a

        sample_b = np.empty(spec.sample_size, dtype=np.int64)
        if shared_count:
            sample_b[:shared_count] = sample_a[:shared_count]
        if shared_count < spec.sample_size:
            remaining = np.setdiff1d(
                actual_alt_ids, sample_a[:shared_count], assume_unique=True
            )
            sample_b[shared_count:] = np.asarray(
                rng.choice(
                    remaining, size=spec.sample_size - shared_count, replace=False
                ),
                dtype=np.int64,
            )
        rng.shuffle(sample_b)
        sampled_b[row] = sample_b

    dense_pos_a = np.searchsorted(actual_alt_ids, sampled_a)
    dense_pos_b = np.searchsorted(actual_alt_ids, sampled_b)
    return actual_alt_ids, sampled_a, sampled_b, dense_pos_a, dense_pos_b


class RNGEngine:
    name = "base"

    def draw_uniform_matrix(
        self, seeds: np.ndarray, n: int, offset: int = 0
    ) -> np.ndarray:
        raise NotImplementedError

    def draw_uniform_for_alt_ids(
        self,
        seeds: np.ndarray,
        alt_ids: np.ndarray,
        max_alt_id: int,
        offset: int = 0,
    ) -> np.ndarray:
        raise NotImplementedError

    def draw_gumbel_matrix(
        self, seeds: np.ndarray, n: int, offset: int = 0
    ) -> np.ndarray:
        return _uniforms_to_gumbel(self.draw_uniform_matrix(seeds, n=n, offset=offset))

    def draw_gumbel_for_alt_ids(
        self,
        seeds: np.ndarray,
        alt_ids: np.ndarray,
        max_alt_id: int,
        offset: int = 0,
    ) -> np.ndarray:
        return _uniforms_to_gumbel(
            self.draw_uniform_for_alt_ids(
                seeds, alt_ids=alt_ids, max_alt_id=max_alt_id, offset=offset
            )
        )


class RandomStateEngine(RNGEngine):
    name = "RandomState"

    def draw_uniform_matrix(
        self, seeds: np.ndarray, n: int, offset: int = 0
    ) -> np.ndarray:
        prng = np.random.RandomState()
        out = np.empty((len(seeds), n), dtype=np.float64)
        for row_idx, seed in enumerate(seeds):
            prng.seed(int(seed))
            if offset:
                prng.rand(offset)
            out[row_idx] = prng.rand(n)
        return out

    def draw_uniform_for_alt_ids(
        self,
        seeds: np.ndarray,
        alt_ids: np.ndarray,
        max_alt_id: int,
        offset: int = 0,
    ) -> np.ndarray:
        prng = np.random.RandomState()
        dense_count = int(max_alt_id) + 1
        out = np.empty(alt_ids.shape, dtype=np.float64)
        for row_idx, seed in enumerate(seeds):
            prng.seed(int(seed))
            if offset:
                prng.rand(offset)
            dense_draws = prng.rand(dense_count)
            out[row_idx] = dense_draws[alt_ids[row_idx]]
        return out


class KeyedHashEngine(RNGEngine):
    name = "KeyedHash"

    _GOLDEN_GAMMA = np.uint64(0x9E3779B97F4A7C15)
    _MUL1 = np.uint64(0xBF58476D1CE4E5B9)
    _MUL2 = np.uint64(0x94D049BB133111EB)
    _SEQ_TAG = np.uint64(0xA24BAED4963EE407)
    _ALT_TAG = np.uint64(0x51D7348C2F9A3B17)

    @classmethod
    def _splitmix64_next(cls, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state = (state + cls._GOLDEN_GAMMA) & _MASK_64
        z = state
        z = ((z ^ (z >> np.uint64(30))) * cls._MUL1) & _MASK_64
        z = ((z ^ (z >> np.uint64(27))) * cls._MUL2) & _MASK_64
        z = z ^ (z >> np.uint64(31))
        return state, z

    @staticmethod
    def _to_uniform(z: np.ndarray) -> np.ndarray:
        return ((z >> np.uint64(11)).astype(np.float64)) * (1.0 / (1 << 53))

    def draw_uniform_matrix(
        self, seeds: np.ndarray, n: int, offset: int = 0
    ) -> np.ndarray:
        out = np.empty((len(seeds), n), dtype=np.float64)
        offset_state = np.uint64(
            (int(offset) * int(self._GOLDEN_GAMMA)) & ((1 << 64) - 1)
        )
        state = (seeds.astype(np.uint64) + self._SEQ_TAG + offset_state) & _MASK_64
        for col in range(n):
            state, z = self._splitmix64_next(state)
            out[:, col] = self._to_uniform(z)
        return out

    def draw_uniform_for_alt_ids(
        self,
        seeds: np.ndarray,
        alt_ids: np.ndarray,
        max_alt_id: int,
        offset: int = 0,
    ) -> np.ndarray:
        row_state = seeds.astype(np.uint64)[:, np.newaxis]
        alt_state = alt_ids.astype(np.uint64)
        mixed = (
            row_state * self._MUL1
            + alt_state * self._GOLDEN_GAMMA
            + self._ALT_TAG
            + np.uint64((int(offset) * int(self._MUL2)) & ((1 << 64) - 1))
        ) & _MASK_64
        _, z = self._splitmix64_next(mixed)
        return self._to_uniform(z)


def available_engines() -> list[RNGEngine]:
    return [RandomStateEngine(), KeyedHashEngine()]


def run_uniform_test(engine: RNGEngine, spec: UniformTestSpec) -> UniformResult:
    seeds = activitysim_row_seeds(
        spec.seed_count,
        12345,
        "persons",
        f"hash_rng_validity_uniform::{spec.name}",
    )
    values = engine.draw_uniform_matrix(
        seeds, n=spec.draws_per_seed, offset=spec.offset
    )
    flat = values.reshape(-1)
    expected_mean = 0.5
    expected_variance = 1.0 / 12.0
    mean = float(np.mean(flat))
    variance = float(np.var(flat))
    mean_se = math.sqrt(expected_variance / flat.size)
    mean_z = 0.0 if mean_se == 0.0 else abs(mean - expected_mean) / mean_se
    variance_pct_error = 100.0 * (variance - expected_variance) / expected_variance
    ks_d, ks_pvalue = ks_one_sample_uniform(flat)
    counts, _ = np.histogram(flat, bins=32, range=(0.0, 1.0))
    expected_count = flat.size / 32.0
    bin_var = expected_count * (1.0 - 1.0 / 32.0)
    max_bin_z = (
        0.0
        if bin_var <= 0.0
        else float(np.max(np.abs(counts - expected_count) / math.sqrt(bin_var)))
    )
    lag1_corr = _safe_corr(values[:, :-1].reshape(-1), values[:, 1:].reshape(-1))
    adjacent_seed_corr = _safe_corr(
        values[:-1, :].reshape(-1), values[1:, :].reshape(-1)
    )
    return UniformResult(
        engine=engine.name,
        test_name=spec.name,
        sample_count=int(flat.size),
        offset=spec.offset,
        mean=mean,
        mean_z=float(mean_z),
        variance=variance,
        variance_pct_error=float(variance_pct_error),
        ks_d=float(ks_d),
        ks_pvalue=float(ks_pvalue),
        max_bin_z=float(max_bin_z),
        lag1_corr=float(lag1_corr),
        adjacent_seed_corr=float(adjacent_seed_corr),
    )


def run_sparse_alt_test(engine: RNGEngine, spec: SparseAltSpec) -> SparseAltResult:
    seeds = activitysim_row_seeds(
        spec.seed_count,
        12345,
        "persons",
        f"hash_rng_validity_sparse::{spec.name}",
    )
    alt_ids = _make_sparse_alt_ids(spec.alt_count, spec.max_alt_id)
    tiled_alt_ids = np.repeat(alt_ids[np.newaxis, :], spec.seed_count, axis=0)
    values = engine.draw_uniform_for_alt_ids(
        seeds, alt_ids=tiled_alt_ids, max_alt_id=spec.max_alt_id, offset=spec.offset
    )
    alt_means = values.mean(axis=0)
    expected_variance = 1.0 / 12.0
    mean_se = math.sqrt(expected_variance / spec.seed_count)
    max_alt_mean_z = (
        0.0 if mean_se == 0.0 else float(np.max(np.abs(alt_means - 0.5) / mean_se))
    )
    return SparseAltResult(
        engine=engine.name,
        test_name=spec.name,
        seed_count=spec.seed_count,
        alt_count=spec.alt_count,
        max_alt_id=spec.max_alt_id,
        offset=spec.offset,
        alt_mean_id_corr=_safe_corr(alt_ids.astype(np.float64), alt_means),
        max_alt_mean_z=max_alt_mean_z,
        alt_mean_std=float(np.std(alt_means, ddof=0)),
        adjacent_alt_corr=_safe_corr(
            values[:, :-1].reshape(-1), values[:, 1:].reshape(-1)
        ),
        matrix_mean=float(np.mean(values)),
        matrix_variance=float(np.var(values)),
    )


def _simulate_choice_shares(
    engine: RNGEngine,
    scenario: ChoiceScenario,
    alt_ids: np.ndarray,
    max_alt_id: int,
    layout_name: str,
) -> tuple[ChoiceLayoutResult, np.ndarray]:
    utilities = np.asarray(scenario.utilities, dtype=np.float64)
    expected = _softmax(utilities)
    shares = np.empty((scenario.replications, len(utilities)), dtype=np.float64)
    max_errors: list[float] = []
    tv_distances: list[float] = []

    for rep in range(scenario.replications):
        seeds = activitysim_row_seeds(
            scenario.chooser_count,
            12345,
            "persons",
            f"hash_rng_validity_choice::{scenario.name}::{layout_name}::{rep}",
        )
        tiled_alt_ids = np.repeat(
            alt_ids[np.newaxis, :], scenario.chooser_count, axis=0
        )
        shocks = engine.draw_gumbel_for_alt_ids(
            seeds, alt_ids=tiled_alt_ids, max_alt_id=max_alt_id, offset=scenario.offset
        )
        choices = np.argmax(utilities[np.newaxis, :] + shocks, axis=1)
        empirical = (
            np.bincount(choices, minlength=len(utilities)).astype(np.float64)
            / scenario.chooser_count
        )
        shares[rep] = empirical
        max_errors.append(float(np.max(np.abs(empirical - expected))))
        tv_distances.append(float(0.5 * np.sum(np.abs(empirical - expected))))

    mean_shares = shares.mean(axis=0)
    diff = mean_shares - expected
    result = ChoiceLayoutResult(
        engine=engine.name,
        scenario=scenario.name,
        layout=layout_name,
        chooser_count=scenario.chooser_count,
        replications=scenario.replications,
        alt_count=len(utilities),
        offset=scenario.offset,
        max_alt_id=max_alt_id,
        max_abs_error=float(np.max(np.abs(diff))),
        rmse_error=float(np.sqrt(np.mean(diff * diff))),
        mean_tv_distance=float(np.mean(tv_distances)),
        worst_replication_max_abs_error=float(np.max(max_errors)),
    )
    return result, shares


def run_choice_scenario(
    engine: RNGEngine,
    scenario: ChoiceScenario,
) -> tuple[list[ChoiceLayoutResult], ChoiceDriftResult]:
    dense_alt_ids = np.arange(len(scenario.utilities), dtype=np.int64)
    sparse_alt_ids = _make_sparse_alt_ids(
        len(scenario.utilities), scenario.sparse_max_alt_id
    )
    dense_result, dense_shares = _simulate_choice_shares(
        engine,
        scenario,
        alt_ids=dense_alt_ids,
        max_alt_id=int(dense_alt_ids[-1]),
        layout_name="dense_ids",
    )
    sparse_result, sparse_shares = _simulate_choice_shares(
        engine,
        scenario,
        alt_ids=sparse_alt_ids,
        max_alt_id=int(sparse_alt_ids[-1]),
        layout_name="sparse_ids",
    )
    share_drift = np.abs(dense_shares.mean(axis=0) - sparse_shares.mean(axis=0))
    drift_result = ChoiceDriftResult(
        engine=engine.name,
        scenario=scenario.name,
        chooser_count=scenario.chooser_count,
        replications=scenario.replications,
        alt_count=len(scenario.utilities),
        offset=scenario.offset,
        mean_abs_dense_sparse_drift=float(np.mean(share_drift)),
        max_abs_dense_sparse_drift=float(np.max(share_drift)),
    )
    return [dense_result, sparse_result], drift_result


def run_invariance_test(engine: RNGEngine, spec: InvarianceSpec) -> InvarianceResult:
    seeds = activitysim_row_seeds(
        spec.chooser_count,
        12345,
        "persons",
        "hash_rng_validity_invariance",
    )
    actual_alt_ids, sampled_a, sampled_b, _, _ = _build_sampled_sets(spec)
    shocks_a = engine.draw_uniform_for_alt_ids(
        seeds, sampled_a, max_alt_id=spec.max_alt_id, offset=spec.offset
    )
    shocks_b = engine.draw_uniform_for_alt_ids(
        seeds, sampled_b, max_alt_id=spec.max_alt_id, offset=spec.offset
    )

    reversed_shocks = engine.draw_uniform_for_alt_ids(
        seeds[::-1],
        sampled_a[::-1],
        max_alt_id=spec.max_alt_id,
        offset=spec.offset,
    )
    batch_invariant = np.array_equal(reversed_shocks[::-1], shocks_a)

    checked = 0
    matched = 0
    for row in range(spec.chooser_count):
        map_a = {
            int(alt): float(val) for alt, val in zip(sampled_a[row], shocks_a[row])
        }
        map_b = {
            int(alt): float(val) for alt, val in zip(sampled_b[row], shocks_b[row])
        }
        shared = set(map_a).intersection(map_b)
        checked += len(shared)
        matched += sum(map_a[alt] == map_b[alt] for alt in shared)
    sampled_set_invariant = checked == matched

    if spec.offset == 0:
        offset_changes_values = True
    else:
        baseline = engine.draw_uniform_for_alt_ids(
            seeds, sampled_a, max_alt_id=spec.max_alt_id, offset=0
        )
        offset_changes_values = not np.array_equal(baseline, shocks_a)

    return InvarianceResult(
        engine=engine.name,
        chooser_count=spec.chooser_count,
        sample_size=spec.sample_size,
        actual_alt_count=spec.actual_alt_count,
        max_alt_id=spec.max_alt_id,
        offset=spec.offset,
        batch_invariant=bool(batch_invariant),
        sampled_set_invariant=bool(sampled_set_invariant),
        offset_changes_values=bool(offset_changes_values),
        shared_pairs_checked=int(checked),
    )


def profile_settings(
    profile: str,
) -> tuple[
    list[UniformTestSpec],
    list[SparseAltSpec],
    list[ChoiceScenario],
    list[InvarianceSpec],
]:
    normalized = profile.strip().lower()
    if normalized == "fast":
        uniform_specs = [
            UniformTestSpec(
                "uniform_offset_0", seed_count=1024, draws_per_seed=32, offset=0
            ),
            UniformTestSpec(
                "uniform_offset_257", seed_count=1024, draws_per_seed=32, offset=257
            ),
        ]
        sparse_specs = [
            SparseAltSpec(
                "sparse_alt_offset_0",
                seed_count=1024,
                alt_count=24,
                max_alt_id=1024,
                offset=0,
            ),
            SparseAltSpec(
                "sparse_alt_offset_257",
                seed_count=1024,
                alt_count=24,
                max_alt_id=1024,
                offset=257,
            ),
        ]
        choice_scenarios = [
            ChoiceScenario(
                "balanced_4",
                chooser_count=5000,
                replications=4,
                offset=0,
                utilities=(0.0, 0.0, 0.0, 0.0),
                sparse_max_alt_id=512,
            ),
            ChoiceScenario(
                "skewed_4",
                chooser_count=5000,
                replications=4,
                offset=0,
                utilities=(0.0, 0.4, 1.0, -0.6),
                sparse_max_alt_id=512,
            ),
            ChoiceScenario(
                "long_tail_8",
                chooser_count=6000,
                replications=4,
                offset=17,
                utilities=(-1.0, -0.5, -0.1, 0.0, 0.2, 0.4, 0.8, 1.1),
                sparse_max_alt_id=1024,
            ),
        ]
        invariance_specs = [
            InvarianceSpec(
                chooser_count=24,
                sample_size=8,
                actual_alt_count=32,
                max_alt_id=512,
                overlap_ratio=0.5,
                offset=0,
            ),
            InvarianceSpec(
                chooser_count=24,
                sample_size=8,
                actual_alt_count=32,
                max_alt_id=512,
                overlap_ratio=0.5,
                offset=19,
            ),
        ]
    elif normalized == "full":
        uniform_specs = [
            UniformTestSpec(
                "uniform_offset_0", seed_count=4096, draws_per_seed=64, offset=0
            ),
            UniformTestSpec(
                "uniform_offset_257", seed_count=4096, draws_per_seed=64, offset=257
            ),
        ]
        sparse_specs = [
            SparseAltSpec(
                "sparse_alt_offset_0",
                seed_count=4096,
                alt_count=48,
                max_alt_id=4096,
                offset=0,
            ),
            SparseAltSpec(
                "sparse_alt_offset_257",
                seed_count=4096,
                alt_count=48,
                max_alt_id=4096,
                offset=257,
            ),
        ]
        choice_scenarios = [
            ChoiceScenario(
                "balanced_4",
                chooser_count=12000,
                replications=8,
                offset=0,
                utilities=(0.0, 0.0, 0.0, 0.0),
                sparse_max_alt_id=1024,
            ),
            ChoiceScenario(
                "skewed_4",
                chooser_count=12000,
                replications=8,
                offset=0,
                utilities=(0.0, 0.4, 1.0, -0.6),
                sparse_max_alt_id=1024,
            ),
            ChoiceScenario(
                "long_tail_8",
                chooser_count=14000,
                replications=8,
                offset=17,
                utilities=(-1.0, -0.5, -0.1, 0.0, 0.2, 0.4, 0.8, 1.1),
                sparse_max_alt_id=2048,
            ),
        ]
        invariance_specs = [
            InvarianceSpec(
                chooser_count=64,
                sample_size=12,
                actual_alt_count=64,
                max_alt_id=2048,
                overlap_ratio=0.5,
                offset=0,
            ),
            InvarianceSpec(
                chooser_count=64,
                sample_size=12,
                actual_alt_count=64,
                max_alt_id=2048,
                overlap_ratio=0.5,
                offset=19,
            ),
        ]
    else:
        raise ValueError("profile must be 'fast' or 'full'")
    return uniform_specs, sparse_specs, choice_scenarios, invariance_specs


def _write_csv(rows: list[object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _markdown_table_from_rows(rows: list[dict[str, object]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _format_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _format_int(value: int) -> str:
    return f"{value:,}"


def _build_report(
    uniform_results: list[UniformResult],
    sparse_results: list[SparseAltResult],
    choice_layout_results: list[ChoiceLayoutResult],
    drift_results: list[ChoiceDriftResult],
    invariance_results: list[InvarianceResult],
    uniform_specs: list[UniformTestSpec],
    sparse_specs: list[SparseAltSpec],
    choice_scenarios: list[ChoiceScenario],
    invariance_specs: list[InvarianceSpec],
    profile: str,
) -> str:
    uniform_by_engine = {
        row.engine: [item for item in uniform_results if item.engine == row.engine]
        for row in uniform_results
    }
    sparse_by_engine = {
        row.engine: [item for item in sparse_results if item.engine == row.engine]
        for row in sparse_results
    }
    choice_by_engine = {
        row.engine: [
            item for item in choice_layout_results if item.engine == row.engine
        ]
        for row in choice_layout_results
    }
    drift_by_engine = {
        row.engine: [item for item in drift_results if item.engine == row.engine]
        for row in drift_results
    }

    def baseline_limit(values: list[float], floor: float, scale: float = 1.5) -> float:
        return max(floor, scale * max(abs(value) for value in values))

    hash_uniform = uniform_by_engine["KeyedHash"]
    baseline_uniform = uniform_by_engine["RandomState"]
    hash_sparse = sparse_by_engine["KeyedHash"]
    baseline_sparse = sparse_by_engine["RandomState"]
    hash_choice = choice_by_engine["KeyedHash"]
    baseline_choice = choice_by_engine["RandomState"]
    hash_drift = drift_by_engine["KeyedHash"]
    baseline_drift = drift_by_engine["RandomState"]
    hash_invariance = [row for row in invariance_results if row.engine == "KeyedHash"]

    invariance_ok = all(
        row.batch_invariant and row.sampled_set_invariant and row.offset_changes_values
        for row in hash_invariance
    )

    uniform_lag_limit = baseline_limit(
        [row.lag1_corr for row in baseline_uniform], floor=0.015
    )
    uniform_adjacent_seed_limit = baseline_limit(
        [row.adjacent_seed_corr for row in baseline_uniform],
        floor=0.015,
    )
    uniform_ok = all(
        row.mean_z < 4.5
        and abs(row.variance_pct_error) < 3.0
        and row.ks_d < 0.01
        and row.max_bin_z < 4.5
        and abs(row.lag1_corr) <= uniform_lag_limit
        and abs(row.adjacent_seed_corr) <= uniform_adjacent_seed_limit
        for row in hash_uniform
    )

    sparse_corr_limit = baseline_limit(
        [row.alt_mean_id_corr for row in baseline_sparse], floor=0.20
    )
    sparse_adjacent_alt_limit = baseline_limit(
        [row.adjacent_alt_corr for row in baseline_sparse],
        floor=0.015,
    )
    sparse_ok = all(
        abs(row.alt_mean_id_corr) <= sparse_corr_limit
        and row.max_alt_mean_z < 4.5
        and abs(row.adjacent_alt_corr) <= sparse_adjacent_alt_limit
        for row in hash_sparse
    )

    baseline_choice_error_limit = baseline_limit(
        [row.max_abs_error for row in baseline_choice], floor=0.02
    )
    baseline_choice_tv_limit = baseline_limit(
        [row.mean_tv_distance for row in baseline_choice], floor=0.03
    )
    baseline_drift_limit = baseline_limit(
        [row.max_abs_dense_sparse_drift for row in baseline_drift],
        floor=0.015,
    )
    choice_ok = all(
        row.max_abs_error <= baseline_choice_error_limit
        and row.mean_tv_distance <= baseline_choice_tv_limit
        for row in hash_choice
    ) and all(
        row.max_abs_dense_sparse_drift <= baseline_drift_limit for row in hash_drift
    )

    verdict = (
        "Yes" if invariance_ok and uniform_ok and sparse_ok and choice_ok else "No"
    )
    summary_sentence = (
        "The keyed hash looks statistically valid enough for the ActivitySim use cases exercised here."
        if verdict == "Yes"
        else "The keyed hash does not clear the bar set by these ActivitySim-oriented checks."
    )

    uniform_rows = []
    for row in uniform_results:
        uniform_rows.append(
            {
                "engine": row.engine,
                "test": row.test_name,
                "mean_z": _format_float(row.mean_z, 2),
                "ks_d": _format_float(row.ks_d, 4),
                "max_bin_z": _format_float(row.max_bin_z, 2),
                "lag1_corr": _format_float(row.lag1_corr, 4),
                "adjacent_seed_corr": _format_float(row.adjacent_seed_corr, 4),
            }
        )
    sparse_rows = []
    for row in sparse_results:
        sparse_rows.append(
            {
                "engine": row.engine,
                "test": row.test_name,
                "alt_mean_id_corr": _format_float(row.alt_mean_id_corr, 4),
                "max_alt_mean_z": _format_float(row.max_alt_mean_z, 2),
                "adjacent_alt_corr": _format_float(row.adjacent_alt_corr, 4),
            }
        )
    choice_rows = []
    for row in choice_layout_results:
        choice_rows.append(
            {
                "engine": row.engine,
                "scenario": row.scenario,
                "layout": row.layout,
                "max_abs_error": _format_float(row.max_abs_error, 4),
                "mean_tv_distance": _format_float(row.mean_tv_distance, 4),
                "worst_replication_max_abs_error": _format_float(
                    row.worst_replication_max_abs_error, 4
                ),
            }
        )
    drift_rows = []
    for row in drift_results:
        drift_rows.append(
            {
                "engine": row.engine,
                "scenario": row.scenario,
                "mean_abs_dense_sparse_drift": _format_float(
                    row.mean_abs_dense_sparse_drift, 4
                ),
                "max_abs_dense_sparse_drift": _format_float(
                    row.max_abs_dense_sparse_drift, 4
                ),
            }
        )
    invariance_rows = []
    for row in invariance_results:
        invariance_rows.append(
            {
                "engine": row.engine,
                "offset": row.offset,
                "batch_invariant": str(row.batch_invariant),
                "sampled_set_invariant": str(row.sampled_set_invariant),
                "offset_changes_values": str(row.offset_changes_values),
                "shared_pairs_checked": str(row.shared_pairs_checked),
            }
        )

    hash_vs_baseline_choice = []
    baseline_choice_map = {
        (row.scenario, row.layout): row for row in choice_by_engine["RandomState"]
    }
    for row in hash_choice:
        baseline = baseline_choice_map[(row.scenario, row.layout)]
        hash_vs_baseline_choice.append(
            f"{row.scenario} {row.layout}: hash max error {_format_float(row.max_abs_error, 4)} vs RandomState {_format_float(baseline.max_abs_error, 4)}"
        )

    expected_uniform_variance = 1.0 / 12.0
    expected_alt_mean_std = math.sqrt(
        expected_uniform_variance / sparse_results[0].seed_count
    )
    uniform_spec_map = {spec.name: spec for spec in uniform_specs}
    sparse_spec_map = {spec.name: spec for spec in sparse_specs}
    choice_spec_map = {spec.name: spec for spec in choice_scenarios}
    invariance_spec_map = {spec.offset: spec for spec in invariance_specs}

    uniform_count_rows = []
    for spec in uniform_specs:
        returned_draws = spec.seed_count * spec.draws_per_seed
        uniform_count_rows.extend(
            [
                {
                    "test": spec.name,
                    "engine": "RandomState",
                    "returned_values": _format_int(returned_draws),
                    "raw_rng_draws": _format_int(
                        spec.seed_count * (spec.draws_per_seed + spec.offset)
                    ),
                    "notes": "offset burns are consumed as real draws before the retained values are returned",
                },
                {
                    "test": spec.name,
                    "engine": "KeyedHash",
                    "returned_values": _format_int(returned_draws),
                    "raw_rng_draws": _format_int(returned_draws),
                    "notes": "offset is mixed into the stateless key, so no extra burn draws are needed",
                },
            ]
        )

    sparse_count_rows = []
    for spec in sparse_specs:
        returned_draws = spec.seed_count * spec.alt_count
        sparse_count_rows.extend(
            [
                {
                    "test": spec.name,
                    "engine": "RandomState",
                    "returned_values": _format_int(returned_draws),
                    "raw_rng_draws": _format_int(
                        spec.seed_count * (spec.max_alt_id + 1 + spec.offset)
                    ),
                    "notes": "the dense gather path draws the full id span, then gathers the requested sparse ids back out",
                },
                {
                    "test": spec.name,
                    "engine": "KeyedHash",
                    "returned_values": _format_int(returned_draws),
                    "raw_rng_draws": _format_int(returned_draws),
                    "notes": "only the requested chooser-alt pairs are generated",
                },
            ]
        )

    choice_count_rows = []
    for row in choice_layout_results:
        spec = choice_spec_map[row.scenario]
        returned_draws = spec.chooser_count * spec.replications * row.alt_count
        if row.engine == "RandomState":
            raw_rng_draws = (
                spec.chooser_count
                * spec.replications
                * (row.max_alt_id + 1 + row.offset)
            )
            note = "RandomState burns the offset and, for sparse layouts, draws the full covered id range"
        else:
            raw_rng_draws = returned_draws
            note = "KeyedHash generates exactly one value per chooser-alt pair used in the choice simulation"
        choice_count_rows.append(
            {
                "scenario": row.scenario,
                "layout": row.layout,
                "engine": row.engine,
                "returned_values": _format_int(returned_draws),
                "raw_rng_draws": _format_int(raw_rng_draws),
                "notes": note,
            }
        )

    invariance_count_rows = []
    for row in invariance_results:
        spec = invariance_spec_map[row.offset]
        pair_count = spec.chooser_count * spec.sample_size
        returned_draws = pair_count * (4 if row.offset else 3)
        if row.engine == "RandomState":
            raw_rng_draws = 0
            raw_rng_draws += spec.chooser_count * (spec.max_alt_id + 1 + row.offset)
            raw_rng_draws += spec.chooser_count * (spec.max_alt_id + 1 + row.offset)
            raw_rng_draws += spec.chooser_count * (spec.max_alt_id + 1 + row.offset)
            if row.offset:
                raw_rng_draws += spec.chooser_count * (spec.max_alt_id + 1)
            note = "three sampled-matrix checks are always run; the nonzero-offset case adds one extra baseline-at-offset-0 comparison"
        else:
            raw_rng_draws = returned_draws
            note = "the same logical comparisons are run, but each comparison only generates the requested chooser-alt pairs"
        invariance_count_rows.append(
            {
                "offset": str(row.offset),
                "engine": row.engine,
                "returned_values": _format_int(returned_draws),
                "raw_rng_draws": _format_int(raw_rng_draws),
                "notes": note,
            }
        )

    lines = [
        "# Hash RNG Statistical Validity Report",
        "",
        f"Profile: `{profile}`",
        "",
        "## Bottom line",
        "",
        f"Verdict: **{verdict}**.",
        "",
        summary_sentence,
        "",
        "This report compares the keyed hash approach against the current ActivitySim-style NumPy `RandomState` approach on four questions that matter for ActivitySim:",
        "",
        "1. Do the raw uniform draws still look like uniform draws?",
        "2. Do sparse alternative ids introduce visible bias or correlation?",
        "3. Do MNL-style choice shares come out right with EV1 shocks?",
        "4. Does the keyed approach preserve the invariance ActivitySim relies on when offsets and sampled choice sets change?",
        "",
        "## What the results say",
        "",
        f"Invariance checks: {'passed' if invariance_ok else 'failed'} for the keyed hash in all tested cases.",
        f"Uniformity and simple dependence checks: {'passed' if uniform_ok else 'failed'} for the keyed hash under both offsets.",
        f"Sparse-id bias checks: {'passed' if sparse_ok else 'failed'} for the keyed hash under both offsets.",
        f"Choice-share recovery and dense-vs-sparse stability checks: {'passed' if choice_ok else 'failed'} for the keyed hash.",
        "",
        "Choice error comparison against RandomState:",
        "",
    ]
    lines.extend(f"- {item}" for item in hash_vs_baseline_choice)
    lines.extend(
        [
            "",
            "## Metric guide",
            "",
            "This section explains each metric that appears either in the tables below or in the CSV outputs. For each one, the important questions are the same: why we use it, how it is computed, what a large or small value means for ActivitySim, and how many random values were used to estimate it.",
            "",
            "A useful distinction in the count explanations below is this:",
            "",
            "- `returned values` means the random values that directly feed the metric calculation.",
            "- `raw RNG draws` means the total number of random values the engine had to generate or consume internally to produce those returned values. These can differ a lot when an engine burns offsets or draws a dense id span and then gathers sparse ids back out.",
            "",
            "### Uniform test metrics",
            "",
            f"These metrics ask whether the raw `U(0, 1)` draws look like draws from a good uniform source before they are turned into EV1 shocks. In this `{profile}` profile, each uniform test row is based on `{_format_int(uniform_specs[0].seed_count)} chooser seeds x {_format_int(uniform_specs[0].draws_per_seed)} retained draws = {_format_int(uniform_specs[0].seed_count * uniform_specs[0].draws_per_seed)}` returned values per engine. In the offset-257 case, RandomState also burns `{_format_int(uniform_specs[1].seed_count * uniform_specs[1].offset)}` extra values internally, so its raw draw count there is `{_format_int(uniform_specs[1].seed_count * (uniform_specs[1].draws_per_seed + uniform_specs[1].offset))}` while KeyedHash still uses `{_format_int(uniform_specs[1].seed_count * uniform_specs[1].draws_per_seed)}`.",
            "",
            "`mean`:",
            "The overall average of all sampled uniform values. It is included in the CSV because the simplest possible failure mode is a generator that is systematically too high or too low. It is computed as the arithmetic average of all sampled draws. For a good uniform generator, it should be close to `0.5`. Small deviations are normal; the question is whether the deviation is large relative to the sample size.",
            "",
            "`mean_z`:",
            "This is the mean error converted into standard-error units, so it is easier to judge across different sample sizes. It is computed as `abs(mean - 0.5) / sqrt((1/12) / N)`, where `N` is the number of sampled uniforms and `1/12` is the theoretical variance of `U(0, 1)`. Near zero is ideal. Values in the low single digits are usually fine at these sample sizes. Large values would mean the generator is persistently shifted away from `0.5`.",
            "",
            "`variance` and `variance_pct_error`:",
            "These check whether the spread of the uniform draws is correct. `variance` is the sample variance of all draws. `variance_pct_error` compares that value to the theoretical target `1/12` using `(observed - expected) / expected * 100`. A value near zero means the spread is right. A strongly negative value would mean the generator is too concentrated. A strongly positive value would mean the generator is too dispersed.",
            "",
            "`ks_d` and `ks_pvalue`:",
            "These come from the one-sample Kolmogorov-Smirnov test against the uniform distribution. `ks_d` is the maximum vertical gap between the empirical cumulative distribution function of the sampled draws and the ideal uniform CDF. Smaller is better. `ks_pvalue` is a rough significance score for that gap; smaller values mean the observed gap would be less plausible under an ideal uniform generator. The report tables show `ks_d` because it is the direct effect size. The CSV also includes `ks_pvalue` as supporting context.",
            "",
            "`max_bin_z`:",
            "This is a coarse histogram-based sanity check. The interval `[0, 1]` is divided into 32 equal bins, the draw count in each bin is compared to its expected count, and the largest standardized deviation is reported. It is useful because some bad generators show local pileups or holes that are easy to see in bins even when the global mean looks fine. Near zero is ideal. Large values mean at least one part of `[0, 1]` is over- or under-filled.",
            "",
            "`lag1_corr`:",
            "This measures the correlation between neighboring draws within the same chooser stream. It is computed by pairing draw `j` with draw `j + 1` and computing the Pearson correlation over all such pairs. It is used because ActivitySim consumes sequences of draws inside each chooser stream. A value near zero means consecutive draws behave independently enough for this use. A materially positive or negative value would suggest serial structure leaking into model decisions.",
            "",
            "`adjacent_seed_corr`:",
            "This measures the correlation between the same draw positions for neighboring chooser seeds. It is computed by comparing row `i` and row `i + 1` at matching draw positions. It matters because ActivitySim relies on row-seed separation: nearby chooser ids should not inherit similar streams just because their seeds are numerically close. Near zero is what we want.",
            "",
            "### Sparse-alternative metrics",
            "",
            f"These metrics ask a different question: when the generator is queried by sparse alternative ids instead of a dense `0, 1, 2, ...` sequence, do those ids themselves introduce bias or dependence? That is directly relevant to sampled-alternative work in ActivitySim. In this `{profile}` profile, each sparse-id test returns `{_format_int(sparse_specs[0].seed_count)} x {_format_int(sparse_specs[0].alt_count)} = {_format_int(sparse_specs[0].seed_count * sparse_specs[0].alt_count)}` values per engine. But the raw draw counts differ sharply: at offset 0, RandomState must consume `{_format_int(sparse_specs[0].seed_count * (sparse_specs[0].max_alt_id + 1))}` values to cover the dense id span, and at offset 257 it consumes `{_format_int(sparse_specs[1].seed_count * (sparse_specs[1].max_alt_id + 1 + sparse_specs[1].offset))}`. KeyedHash generates only the `{_format_int(sparse_specs[0].seed_count * sparse_specs[0].alt_count)}` requested chooser-alt pairs in either case.",
            "",
            "`alt_mean_id_corr`:",
            "For each alternative id, the test computes the mean draw across all chooser seeds. It then computes the correlation between those per-alt means and the numeric alternative ids. This is used to detect whether larger or smaller ids systematically receive different random values. Near zero is the target. A strong positive or negative value would mean id magnitude is leaking into the distribution, which would be unacceptable for sparse-id lookup.",
            "",
            "`max_alt_mean_z`:",
            "This checks the worst alternative-specific mean error after standardizing by the expected sampling noise. For each alternative, the test computes the mean draw across all seeds, compares it with `0.5`, and divides by `sqrt((1/12) / seed_count)`. The reported value is the maximum absolute z-score over all alternatives. This matters because a generator can have the right global mean while still favoring a few specific alternative ids. Smaller is better.",
            "",
            f"`alt_mean_std`:",
            f"This is the standard deviation across the alternative-specific mean draws. It is a compact way to summarize how much those alternative means vary from one alt id to another. With `seed_count = {sparse_results[0].seed_count}` in this profile, the natural noise scale is about `{_format_float(expected_alt_mean_std, 4)}`. Values around that scale are normal; much larger values would suggest some alternatives behave differently from others for reasons other than sampling noise.",
            "",
            "`adjacent_alt_corr`:",
            "This measures correlation between neighboring alternative columns inside the sampled matrix. It is computed by correlating column `k` with column `k + 1` over all chooser rows. It is useful because a keyed generator could accidentally create local structure in nearby alternative ids. Near zero means adjacent ids are not moving together in an obvious way.",
            "",
            "`matrix_mean` and `matrix_variance`:",
            "These are the same basic checks as `mean` and `variance`, but applied specifically to the sparse-id lookup matrix. They exist in the CSV as a sanity check that the sparse-id code path still behaves like a uniform source overall, not just alt-by-alt.",
            "",
            "### Choice-recovery metrics",
            "",
            "These metrics move one step closer to actual model behavior. Instead of looking only at raw uniforms, they convert those draws into Gumbel shocks, add them to deterministic utilities, and check whether empirical MNL choice shares match the theoretical softmax shares. The number of random values here depends on the scenario and layout. Each replication draws one shock per chooser-alternative pair that is evaluated by the engine. For example, in this profile, `balanced_4` and `skewed_4` each use `12,000 x 8 x 4 = 384,000` returned values per layout, while `long_tail_8` uses `14,000 x 8 x 8 = 896,000`. In dense layouts those counts are also the raw draw counts for both engines when offset is zero. In sparse layouts, and in any layout with nonzero offset, RandomState consumes more raw values because it must burn the offset and cover the full dense span before gathering the specific alternatives back out.",
            "",
            "`max_abs_error`:",
            "For each scenario and layout, the expected choice probability of each alternative is computed from the deterministic utilities using the softmax formula. The simulation is then run repeatedly, the empirical shares are averaged across replications, and the largest absolute gap between empirical and expected share is reported. This is one of the most important metrics because it answers the direct question: if we use this RNG in an EV1-based choice model, do the resulting shares come out right? Smaller is better.",
            "",
            "`rmse_error`:",
            "This is the root mean squared error between the average empirical share vector and the expected share vector. It summarizes all alternative-level share errors into one number instead of only reporting the worst case. It is useful because a generator could have one moderate outlier or many small errors; RMSE helps distinguish those patterns. Smaller is better.",
            "",
            "`mean_tv_distance`:",
            "This is the mean total variation distance across replications. For one replication, it is `0.5 * sum(abs(empirical_share - expected_share))` over all alternatives. It represents the amount of probability mass that would need to be moved to turn the empirical distribution into the expected one. It is easy to interpret: `0` is perfect, and larger values mean the simulated chooser distribution is farther from the theoretical target.",
            "",
            "`worst_replication_max_abs_error`:",
            "This reports the single worst alternative-share error seen in any one replication, not just after averaging replications together. It is included because averaging can hide unstable runs. A generator with acceptable average behavior but occasional very bad runs would still be concerning. Smaller is better.",
            "",
            "### Dense-versus-sparse drift metrics",
            "",
            "These metrics compare the same choice scenario under two id layouts: a dense layout such as `0, 1, 2, ...` and a sparse layout with the same utilities but much larger id values. The point is to test whether id encoding alone changes behavior.",
            "",
            "`mean_abs_dense_sparse_drift`:",
            "After averaging shares across replications for the dense and sparse layouts, this metric takes the absolute difference alternative by alternative and averages those differences. It answers: on average, how much does the final share move when only the id layout changes? Smaller is better.",
            "",
            "`max_abs_dense_sparse_drift`:",
            "This is the largest single alternative-level drift between the dense and sparse layouts. It is the strictest layout-invariance check in the choice layer. Small values mean the generator is not materially sensitive to whether ids are dense or sparse. Larger values would indicate that id encoding itself can move modeled shares.",
            "",
            "### Invariance metrics",
            "",
            f"These are boolean checks rather than scalar diagnostics. They test the invariance properties ActivitySim depends on when rows are reordered, sampled choice sets differ, or offsets advance within a stream. In this profile each sampled matrix contains `{_format_int(invariance_specs[0].chooser_count)} x {_format_int(invariance_specs[0].sample_size)} = {_format_int(invariance_specs[0].chooser_count * invariance_specs[0].sample_size)}` chooser-alt pairs. The offset-0 invariance case runs three such comparisons, so it uses `{_format_int(3 * invariance_specs[0].chooser_count * invariance_specs[0].sample_size)}` returned values for KeyedHash. The nonzero-offset case adds one extra baseline comparison, so it uses `{_format_int(4 * invariance_specs[1].chooser_count * invariance_specs[1].sample_size)}` returned values. RandomState consumes many more raw values internally because each comparison covers the dense id span to protect invariance.",
            "",
            "`batch_invariant`:",
            "The same chooser and alternative pair should get the same random value even if the chooser rows are processed in a different order or with different neighbors. The test reverses the chooser batch and checks for exact equality. `True` is required.",
            "",
            "`sampled_set_invariant`:",
            "The same chooser-alternative pair should keep the same random value even if the surrounding sampled choice set changes. The test builds two partially overlapping sampled sets and checks exact equality on all shared pairs. `True` is required.",
            "",
            "`offset_changes_values`:",
            "Offsets are part of ActivitySim's stream bookkeeping. This metric checks that using a nonzero offset actually changes the generated values relative to offset `0`. If it were `False`, the generator would be ignoring the stream position, which would break step-to-step semantics. `True` is required.",
            "",
            "`shared_pairs_checked`:",
            "This is just the number of chooser-alternative pairs that were available for the sampled-set invariance comparison. Larger values mean the invariance check exercised more shared pairs. It is not a quality score by itself; it is context for how much overlap the test covered.",
            "",
            "## How the verdict is formed",
            "",
            "The report does not treat RandomState as mathematically perfect. It uses RandomState as the practical comparison baseline because that is the behavior ActivitySim already accepts today.",
            "",
            "The keyed hash is marked acceptable only if it satisfies all invariance checks and stays within reasonable bounds on the statistical diagnostics. Some bounds are fixed absolute checks and some are baseline-relative checks:",
            "",
            "- Uniform checks require `mean_z < 4.5`, `abs(variance_pct_error) < 3`, `ks_d < 0.01`, and `max_bin_z < 4.5`. The serial-correlation metrics are compared to a baseline-derived limit: `max(0.015, 1.5 * worst RandomState absolute value)`.",
            "- Sparse-id checks require `max_alt_mean_z < 4.5`. The correlation metrics are compared against baseline-derived limits: `alt_mean_id_corr` uses `max(0.20, 1.5 * worst RandomState absolute value)` and `adjacent_alt_corr` uses `max(0.015, 1.5 * worst RandomState absolute value)`.",
            "- Choice-share checks compare the keyed-hash errors against the scale of the RandomState errors. The limits are `max(0.02, 1.5 * worst RandomState max_abs_error)`, `max(0.03, 1.5 * worst RandomState mean_tv_distance)`, and `max(0.015, 1.5 * worst RandomState max_abs_dense_sparse_drift)`.",
            "",
            "Why those specific numbers were chosen:",
            "",
            "- `mean_z < 4.5` and `max_alt_mean_z < 4.5`: after standardizing by sampling noise, a well-behaved generator should usually land within a few standard errors of the target mean. The cutoff `4.5` is intentionally looser than a textbook `3-sigma` rule because this report runs several checks and we do not want to fail a practically acceptable generator for ordinary Monte Carlo variation. But `4.5` is still strict enough that exceeding it would be hard to dismiss as random fluctuation alone.",
            "- `max_bin_z < 4.5`: same logic as the z-score limits above, but applied to histogram bins. A 4.5-sigma bin excess is too large to call routine bin noise, yet still generous enough to avoid overreacting to a single busy bin.",
            "- `abs(variance_pct_error) < 3`: with the draw counts used here, the sampling error in variance should be well below 3% for a healthy generator. The 3% band is therefore a practical tolerance, not a tight theoretical one. It is meant to catch real spread distortions, not harmless noise.",
            "- `ks_d < 0.01`: a KS gap of 0.01 means the empirical CDF is off from the ideal uniform CDF by as much as one percentage point somewhere on `[0, 1]`. At these sample sizes that is already a visibly large distortion. The observed baselines are much smaller, so `0.01` is a generous outer limit.",
            "- The `1.5 x RandomState` rule: baseline-relative metrics use a multiplier of `1.5` because the hash does not need to beat RandomState to be acceptable, but it should stay in the same order of magnitude. Fifty percent headroom leaves room for normal simulation noise and implementation differences without allowing behavior that is materially worse than today's production baseline.",
            "- The floor `0.015` for correlation-style checks: even if RandomState happens to produce correlations extremely close to zero in one run, we do not want the acceptance band to collapse to an unrealistically tiny value. A 1.5% correlation is still small in practical model terms, so this floor keeps the check stable while remaining meaningfully strict.",
            "- The floor `0.20` for `alt_mean_id_corr`: this sparse-id correlation is measured over a relatively small number of alternatives, so it is noisier than the raw-uniform correlations. A limit of 0.20 still represents only a moderate association; larger values would start to look like real id-linked structure rather than chance.",
            "- The floor `0.02` for `max_abs_error`: this means the averaged choice-share error should not exceed about two percentage points for any single alternative unless RandomState itself is already worse. That is a practical bound on what would count as a visible share distortion.",
            "- The floor `0.03` for `mean_tv_distance`: total variation distance is the share of probability mass that would need to move to match the theoretical distribution. Allowing up to 3% keeps the tolerance small enough to matter while not overfitting to Monte Carlo noise.",
            "- The floor `0.015` for `max_abs_dense_sparse_drift`: changing only the id layout should not move an alternative's average share by much. A 1.5 percentage point maximum drift is a practical upper bound for saying that dense versus sparse ids are behaving the same for ActivitySim purposes.",
            "",
            "Interpreting the outcome therefore means asking two things at once: are the keyed-hash metrics small in absolute terms, and are they still in the same operational range as RandomState? The report answers `Yes` only when both are true.",
            "",
            "## Draw counts behind the metrics",
            "",
            "### Uniform tests",
            "",
            _markdown_table_from_rows(
                uniform_count_rows,
                ["test", "engine", "returned_values", "raw_rng_draws", "notes"],
            ),
            "",
            "### Sparse-id tests",
            "",
            _markdown_table_from_rows(
                sparse_count_rows,
                ["test", "engine", "returned_values", "raw_rng_draws", "notes"],
            ),
            "",
            "### Choice-recovery tests",
            "",
            _markdown_table_from_rows(
                choice_count_rows,
                [
                    "scenario",
                    "layout",
                    "engine",
                    "returned_values",
                    "raw_rng_draws",
                    "notes",
                ],
            ),
            "",
            "### Invariance tests",
            "",
            _markdown_table_from_rows(
                invariance_count_rows,
                ["offset", "engine", "returned_values", "raw_rng_draws", "notes"],
            ),
            "",
            "## Uniform tests",
            "",
            _markdown_table_from_rows(
                uniform_rows,
                [
                    "engine",
                    "test",
                    "mean_z",
                    "ks_d",
                    "max_bin_z",
                    "lag1_corr",
                    "adjacent_seed_corr",
                ],
            ),
            "",
            "## Sparse-id tests",
            "",
            _markdown_table_from_rows(
                sparse_rows,
                [
                    "engine",
                    "test",
                    "alt_mean_id_corr",
                    "max_alt_mean_z",
                    "adjacent_alt_corr",
                ],
            ),
            "",
            "## Choice-share recovery",
            "",
            _markdown_table_from_rows(
                choice_rows,
                [
                    "engine",
                    "scenario",
                    "layout",
                    "max_abs_error",
                    "mean_tv_distance",
                    "worst_replication_max_abs_error",
                ],
            ),
            "",
            "## Dense vs sparse drift",
            "",
            _markdown_table_from_rows(
                drift_rows,
                [
                    "engine",
                    "scenario",
                    "mean_abs_dense_sparse_drift",
                    "max_abs_dense_sparse_drift",
                ],
            ),
            "",
            "## Invariance checks",
            "",
            _markdown_table_from_rows(
                invariance_rows,
                [
                    "engine",
                    "offset",
                    "batch_invariant",
                    "sampled_set_invariant",
                    "offset_changes_values",
                    "shared_pairs_checked",
                ],
            ),
            "",
            "## Interpretation for a non-RNG specialist",
            "",
            "The current RandomState implementation is the benchmark because it is what ActivitySim uses today. The keyed hash does not need to be mathematically perfect in the abstract; it needs to avoid introducing visible artifacts in the kinds of random draws ActivitySim actually uses.",
            "",
            "The report treats RandomState as the comparison scale, not as a theoretical ideal. In other words, the hash only fails if it looks materially worse than the behavior ActivitySim already accepts today.",
            "",
            "That makes the verdict practical rather than abstract: the keyed hash clears the bar only if it stays in the same range as RandomState on the signals that matter for ActivitySim, namely uniformity, simple dependence, sparse-id behavior, choice-share recovery, and invariance.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Compare keyed-hash statistical behavior against ActivitySim's current RandomState RNG."
    )
    parser.add_argument("--profile", choices=["fast", "full"], default="fast")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "output" / "hash_rng_validity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uniform_specs, sparse_specs, choice_scenarios, invariance_specs = profile_settings(
        args.profile
    )
    engines = available_engines()

    uniform_results: list[UniformResult] = []
    sparse_results: list[SparseAltResult] = []
    choice_layout_results: list[ChoiceLayoutResult] = []
    drift_results: list[ChoiceDriftResult] = []
    invariance_results: list[InvarianceResult] = []

    for engine in engines:
        for spec in uniform_specs:
            uniform_results.append(run_uniform_test(engine, spec))
        for spec in sparse_specs:
            sparse_results.append(run_sparse_alt_test(engine, spec))
        for scenario in choice_scenarios:
            scenario_results, drift_result = run_choice_scenario(engine, scenario)
            choice_layout_results.extend(scenario_results)
            drift_results.append(drift_result)
        for spec in invariance_specs:
            invariance_results.append(run_invariance_test(engine, spec))

    output_dir = Path(args.output_dir)
    _write_csv(uniform_results, output_dir / "uniform_results.csv")
    _write_csv(sparse_results, output_dir / "sparse_alt_results.csv")
    _write_csv(choice_layout_results, output_dir / "choice_layout_results.csv")
    _write_csv(drift_results, output_dir / "choice_drift_results.csv")
    _write_csv(invariance_results, output_dir / "invariance_results.csv")

    report_text = _build_report(
        uniform_results,
        sparse_results,
        choice_layout_results,
        drift_results,
        invariance_results,
        uniform_specs,
        sparse_specs,
        choice_scenarios,
        invariance_specs,
        profile=args.profile,
    )
    report_path = output_dir / "hash_rng_validity_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf8")

    print(f"Wrote {output_dir / 'uniform_results.csv'}")
    print(f"Wrote {output_dir / 'sparse_alt_results.csv'}")
    print(f"Wrote {output_dir / 'choice_layout_results.csv'}")
    print(f"Wrote {output_dir / 'choice_drift_results.csv'}")
    print(f"Wrote {output_dir / 'invariance_results.csv'}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
