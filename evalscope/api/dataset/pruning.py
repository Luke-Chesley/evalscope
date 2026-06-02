# Copyright (c) Alibaba, Inc. and its affiliates.
"""Reusable dataset pruning helpers.

This module operates on EvalScope ``DatasetDict`` objects after normal benchmark
loading, filtering, and sample conversion. Benchmark adapters can provide a
feature builder while sharing the same pruning configuration and selection
algorithm.
"""

import hashlib
import math
import re
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from evalscope.api.dataset.dataset import DatasetDict, MemoryDataset, Sample
from evalscope.api.messages import messages_to_markdown
from evalscope.utils.logger import get_logger

logger = get_logger()

PRUNING_EXTRA_PARAMS = {
    'pruning_strategy': {
        'type': 'str',
        'description': 'Pruning strategy name. Supported: fixed_indices, metadata_coreset, calibrated_outputs.',
        'value': 'metadata_coreset'
    },
    'prune_ratio': {
        'type': 'float | null',
        'description': 'Fraction of samples to keep after normal benchmark loading and filtering.',
        'value': None
    },
    'prune_k': {
        'type': 'int | null',
        'description': 'Exact number of samples to keep after normal benchmark loading and filtering.',
        'value': None
    },
    'subset_indices': {
        'type': 'list[int] | str | null',
        'description': 'Optional fixed sample ids to evaluate after normal loading and filtering.',
        'value': None
    },
    'calibration_dir': {
        'type': 'str | null',
        'description': 'Directory containing raw prediction/review artifacts used by calibrated_outputs.',
        'value': None
    },
    'calibration_embedding_model': {
        'type': 'str | null',
        'description': 'SentenceTransformer model used by calibrated_outputs semantic features.',
        'value': None
    },
    'calibration_stratum': {
        'type': 'str | null',
        'description': 'Stratification mode for calibrated_outputs; benchmark-specific defaults are used when null.',
        'value': None
    },
    'score_weight': {
        'type': 'float | null',
        'description': 'Optional calibrated_outputs weight for per-model score pattern features.',
        'value': None
    },
    'discrimination_weight': {
        'type': 'float | null',
        'description': 'Optional calibrated_outputs weight for item discrimination features.',
        'value': None
    },
    'structure_weight': {
        'type': 'float | null',
        'description': 'Optional calibrated_outputs weight for length/structure features.',
        'value': None
    },
    'semantic_weight': {
        'type': 'float | null',
        'description': 'Optional calibrated_outputs weight for embedding features.',
        'value': None
    },
    'keyword_weight': {
        'type': 'float | null',
        'description': 'Optional calibrated_outputs weight for keyword features when a benchmark supports them.',
        'value': None
    },
    'random_state': {
        'type': 'int | null',
        'description': 'Random seed used by calibrated_outputs clustering.',
        'value': None
    },
    'use_embeddings': {
        'type': 'bool | null',
        'description': 'Whether calibrated_outputs includes SentenceTransformer semantic embeddings.',
        'value': None
    },
    'embedding_text_mode': {
        'type': 'str | null',
        'description': 'Text source used for calibrated semantic embeddings; benchmark-specific default when null.',
        'value': None
    },
    'available_id_policy': {
        'type': 'str | null',
        'description': "How to handle calibration ids absent from the loaded dataset: 'filter' or 'error'.",
        'value': None
    },
    'calibration_artifact_name': {
        'type': 'str | null',
        'description': 'Optional raw artifact prefix override for calibrated_outputs.',
        'value': None
    },
}


class PrunedBenchmarkMixin:
    """Reusable pruning hook for DefaultDataAdapter-compatible benchmarks."""

    def _init_pruning(self) -> None:
        extra_params = self.extra_params
        self.pruning_strategy = extra_params.get('pruning_strategy', 'metadata_coreset')
        self.prune_ratio = extra_params.get('prune_ratio')
        self.prune_k = extra_params.get('prune_k')
        self.subset_indices = parse_index_list(extra_params.get('subset_indices'))
        self.calibration_dir = extra_params.get('calibration_dir')

    def _load_pruned_dataset(self) -> Tuple[DatasetDict, Optional[DatasetDict]]:
        test_dataset, fewshot_dataset = super().load()
        if self.subset_indices is not None or self.pruning_strategy == 'fixed_indices':
            if self.subset_indices is None:
                raise ValueError('fixed_indices pruning requires subset_indices.')
            return apply_index_subset(test_dataset, self.subset_indices, repeats=self.repeats), fewshot_dataset

        if self.pruning_strategy == 'calibrated_outputs':
            selected_indices = self.calibrated_subset_indices(test_dataset)
            return apply_index_subset(test_dataset, selected_indices, repeats=self.repeats), fewshot_dataset

        if self.pruning_strategy != 'metadata_coreset':
            raise ValueError(
                f'Unsupported pruning_strategy={self.pruning_strategy!r}; '
                'supported strategies are fixed_indices, metadata_coreset, and calibrated_outputs.'
            )

        return apply_coreset_pruning_by_ratio(
            test_dataset,
            prune_k=self.prune_k,
            prune_ratio=self.prune_ratio,
            feature_builder=self.pruning_feature_builder,
            repeats=self.repeats,
        ), fewshot_dataset

    def pruning_feature_builder(self, sample: Sample) -> Dict[str, Any]:
        """Return benchmark-specific pruning features."""
        return generic_text_feature_builder(sample)

    def calibrated_subset_indices(self, dataset_dict: DatasetDict) -> List[int]:
        """Return sample ids selected from external calibration artifacts."""
        raise NotImplementedError(
            f'{self.__class__.__name__} does not implement calibrated_outputs pruning.'
        )

    def _calibrated_backend_indices(self, benchmark_name: str, dataset_dict: DatasetDict) -> List[int]:
        from evalscope.api.dataset.calibrated_pruning import select_calibrated_pruned_indices

        config = dict(self.extra_params)
        config['prune_k'] = self.prune_k
        config['prune_ratio'] = self.prune_ratio
        config['calibration_dir'] = self.calibration_dir
        return select_calibrated_pruned_indices(benchmark_name, dataset_dict, config)

    def _resolve_calibrated_k(self, sample_count: int) -> int:
        from evalscope.api.dataset.calibrated_pruning import resolve_prune_k

        return resolve_prune_k(sample_count, self.prune_k, self.prune_ratio)

    def _source_sample_count(self, dataset_dict: DatasetDict) -> int:
        return sum(len(dataset) for _, dataset in dataset_dict.items())

    def _source_ids(self, dataset_dict: DatasetDict) -> List[int]:
        ids = []
        for _, dataset in dataset_dict.items():
            ids.extend(sample.id for sample in dataset)
        return ids

    def _extra_float(self, key: str, default: float) -> float:
        value = self.extra_params.get(key)
        return default if value is None else float(value)

    def _extra_int(self, key: str, default: int) -> int:
        value = self.extra_params.get(key)
        return default if value is None else int(value)

    def _extra_str(self, key: str, default: str) -> str:
        value = self.extra_params.get(key)
        return default if value is None else str(value)


def sample_text(sample: Sample) -> str:
    """Return text usable for prompt-derived features."""
    if isinstance(sample.input, str):
        return sample.input
    return messages_to_markdown(sample.input)


def parse_index_list(value: Optional[Any]) -> Optional[List[int]]:
    """Parse fixed sample indices from a list or comma-separated string."""
    if value is None:
        return None
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(',') if part.strip()]
    if isinstance(value, Iterable):
        return [int(item) for item in value]
    raise TypeError(f'Unsupported subset_indices value: {value!r}')


def apply_index_subset(dataset_dict: DatasetDict, indices: Sequence[int], repeats: int = 1) -> DatasetDict:
    """Keep samples whose current EvalScope sample id is in ``indices``."""
    selected = set(indices)
    pruned = {}
    for subset, dataset in dataset_dict.items():
        samples = []
        for sample in dataset:
            if sample.id not in selected:
                continue
            sample.metadata = dict(sample.metadata or {})
            sample.metadata.setdefault('source_id', sample.id)
            samples.append(sample)
        memory_dataset = MemoryDataset(samples=samples, name=dataset.name, location=dataset.location)
        memory_dataset.reindex(group_size=repeats)
        pruned[subset] = memory_dataset
        logger.info(f'Applied fixed index pruning to {subset}: kept {len(samples)} / {len(dataset)} samples.')
    return DatasetDict(pruned)


def apply_coreset_pruning(
    dataset_dict: DatasetDict,
    k: int,
    feature_builder: Callable[[Sample], Dict[str, Any]],
    repeats: int = 1,
) -> DatasetDict:
    """Select a deterministic metadata coreset for every subset in a DatasetDict."""
    if k <= 0:
        raise ValueError('prune_k must be positive.')

    pruned = {}
    for subset, dataset in dataset_dict.items():
        samples = list(dataset)
        selected_positions = select_diverse_samples(samples, k=k, feature_builder=feature_builder)
        selected = [samples[pos] for pos in selected_positions]
        memory_dataset = MemoryDataset(samples=selected, name=dataset.name, location=dataset.location)
        memory_dataset.reindex(group_size=repeats)
        pruned[subset] = memory_dataset
        logger.info(f'Applied coreset pruning to {subset}: kept {len(selected)} / {len(samples)} samples.')
    return DatasetDict(pruned)


def apply_coreset_pruning_by_ratio(
    dataset_dict: DatasetDict,
    prune_k: Optional[int],
    prune_ratio: Optional[float],
    feature_builder: Callable[[Sample], Dict[str, Any]],
    repeats: int = 1,
) -> DatasetDict:
    """Select a coreset using either exact K or keep ratio."""
    pruned = {}
    for subset, dataset in dataset_dict.items():
        k = _resolve_subset_k(len(dataset), prune_k=prune_k, prune_ratio=prune_ratio)
        subset_dict = apply_coreset_pruning(
            DatasetDict({subset: dataset}),
            k=k,
            feature_builder=feature_builder,
            repeats=repeats,
        )
        pruned[subset] = subset_dict[subset]
    return DatasetDict(pruned)


def generic_text_feature_builder(sample: Sample) -> Dict[str, Any]:
    """Build generic prompt-text pruning features for benchmarks without a custom extractor."""
    text = sample_text(sample)
    return {
        'stratum': 'default',
        'vector': [
            _scaled_log(len(text)),
            _scaled_log(_rough_token_count(text)),
            *_hashed_text_vector(text, dimensions=64),
        ],
    }


def select_diverse_samples(
    samples: Sequence[Sample],
    k: int,
    feature_builder: Callable[[Sample], Dict[str, Any]],
) -> List[int]:
    """Coverage-first selection with proportional strata and farthest-first medoids."""
    if k >= len(samples):
        return list(range(len(samples)))

    features = [feature_builder(sample) for sample in samples]
    strata = [str(feature.get('stratum', 'default')) for feature in features]
    vectors = [feature.get('vector', []) for feature in features]

    allocations = _allocate_by_stratum(strata, k)
    selected = []
    for stratum, stratum_k in allocations.items():
        positions = [idx for idx, value in enumerate(strata) if value == stratum]
        selected.extend(_select_farthest_first(positions, vectors, stratum_k))

    return sorted(dict.fromkeys(selected))


def _allocate_by_stratum(strata: Sequence[str], k: int) -> Dict[str, int]:
    counts = Counter(strata)
    allocations = {stratum: 0 for stratum in counts}
    remaining = k

    for stratum in sorted(counts, key=lambda item: counts[item], reverse=True):
        if remaining <= 0:
            break
        allocations[stratum] = 1
        remaining -= 1

    total = len(strata)
    while remaining > 0:
        candidates = [stratum for stratum, count in counts.items() if allocations[stratum] < count]
        if not candidates:
            break
        best = max(candidates, key=lambda stratum: counts[stratum] / total - allocations[stratum] / k)
        allocations[best] += 1
        remaining -= 1

    return {stratum: allocation for stratum, allocation in allocations.items() if allocation > 0}


def _resolve_subset_k(length: int, prune_k: Optional[int], prune_ratio: Optional[float]) -> int:
    if prune_k is not None:
        return max(1, min(length, int(prune_k)))
    if prune_ratio is None:
        raise ValueError('metadata_coreset pruning requires prune_k or prune_ratio.')
    ratio = float(prune_ratio)
    if ratio <= 0 or ratio > 1:
        raise ValueError('prune_ratio must be in (0, 1].')
    return max(1, min(length, math.ceil(length * ratio)))


def _select_farthest_first(positions: Sequence[int], vectors: Sequence[Sequence[float]], k: int) -> List[int]:
    if k >= len(positions):
        return list(positions)

    centroid = _centroid([vectors[pos] for pos in positions])
    first = min(positions, key=lambda pos: (_distance(vectors[pos], centroid), pos))
    selected = [first]

    while len(selected) < k:
        remaining = [pos for pos in positions if pos not in selected]
        next_pos = max(
            remaining,
            key=lambda pos: (min(_distance(vectors[pos], vectors[chosen]) for chosen in selected), -pos),
        )
        selected.append(next_pos)

    return selected


def _centroid(vectors: Sequence[Sequence[float]]) -> List[float]:
    if not vectors:
        return []
    width = len(vectors[0])
    return [sum(vector[idx] for vector in vectors) / len(vectors) for idx in range(width)]


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _scaled_log(value: float) -> float:
    return math.log1p(max(float(value), 0.0))


def _rough_token_count(text: str) -> int:
    return max(1, len(re.findall(r'\S+', text)))


def _hashed_text_vector(text: str, dimensions: int) -> List[float]:
    buckets = [0.0] * dimensions
    tokens = re.findall(r'[A-Za-z_][A-Za-z_0-9]+|\d+', text.lower())
    for token in tokens:
        digest = hashlib.md5(token.encode('utf-8')).digest()
        bucket = digest[0] % dimensions
        sign = 1.0 if digest[1] % 2 == 0 else -1.0
        buckets[bucket] += sign
    norm = math.sqrt(sum(value * value for value in buckets)) or 1.0
    return [value / norm for value in buckets]
