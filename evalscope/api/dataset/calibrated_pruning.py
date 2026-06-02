# Copyright (c) Alibaba, Inc. and its affiliates.
"""Configurable calibrated pruning from EvalScope prediction/review artifacts."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from glob import glob
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from evalscope.api.dataset.dataset import DatasetDict
from evalscope.utils.logger import get_logger

logger = get_logger()


@dataclass(frozen=True)
class CalibratedBenchmarkSpec:
    """Benchmark-specific defaults for the shared calibrated pruning backend."""

    artifact_benchmark_name: str
    default_stratum: str
    default_embedding_text_mode: str
    default_score_weight: float
    default_discrimination_weight: float
    default_structure_weight: float
    default_semantic_weight: float
    default_keyword_weight: float = 0.0
    metadata_resolver: Optional[Callable[[DatasetDict], Dict[str, Dict[int, Any]]]] = None


def _aa_lcr_metadata(dataset_dict: DatasetDict) -> Dict[str, Dict[int, Any]]:
    domains = {}
    for _, dataset in dataset_dict.items():
        for sample in dataset:
            domains[sample.id] = (sample.metadata or {}).get('document_category') or 'unknown'
    return {'domain': domains}


def _no_metadata(_: DatasetDict) -> Dict[str, Dict[int, Any]]:
    return {}


BENCHMARK_SPECS: Dict[str, CalibratedBenchmarkSpec] = {
    'aa_lcr': CalibratedBenchmarkSpec(
        artifact_benchmark_name='aa_lcr',
        default_stratum='domain',
        default_embedding_text_mode='question',
        default_score_weight=0.55,
        default_discrimination_weight=0.25,
        default_structure_weight=1.0,
        default_semantic_weight=0.45,
        metadata_resolver=_aa_lcr_metadata,
    ),
    'live_code_bench': CalibratedBenchmarkSpec(
        artifact_benchmark_name='live_code_bench_v5',
        default_stratum='score_pattern',
        default_embedding_text_mode='reasoning_code',
        default_score_weight=0.45,
        default_discrimination_weight=0.20,
        default_structure_weight=0.85,
        default_keyword_weight=0.65,
        default_semantic_weight=0.40,
        metadata_resolver=_no_metadata,
    ),
}


def select_calibrated_pruned_indices(
    benchmark_name: str,
    dataset_dict: DatasetDict,
    config: Dict[str, Any],
    benchmark_spec: Optional[CalibratedBenchmarkSpec] = None,
) -> List[int]:
    """Select pruned sample ids using one calibrated backend across benchmarks."""
    spec = benchmark_spec or BENCHMARK_SPECS[benchmark_name]
    calibration_dir = config.get('calibration_dir')
    if not calibration_dir:
        raise ValueError('calibrated_outputs pruning requires extra_params.calibration_dir.')

    sample_count = _source_sample_count(dataset_dict)
    k = resolve_prune_k(sample_count, config.get('prune_k'), config.get('prune_ratio'))
    available_ids = _source_ids(dataset_dict)
    artifact_name = config.get('calibration_artifact_name') or spec.artifact_benchmark_name
    df_raw = load_benchmark_artifacts(str(artifact_name), str(calibration_dir))

    available_id_policy = str(config.get('available_id_policy') or 'filter')
    if available_id_policy not in {'filter', 'error'}:
        raise ValueError("available_id_policy must be either 'filter' or 'error'.")
    if benchmark_name == 'live_code_bench':
        df_raw = _align_calibration_rows(df_raw, available_ids, available_id_policy)

    metadata = spec.metadata_resolver(dataset_dict) if spec.metadata_resolver else {}
    features = build_calibrated_features(
        benchmark_name=benchmark_name,
        df_raw=df_raw,
        metadata=metadata,
        embedding_model=str(config.get('calibration_embedding_model') or 'all-MiniLM-L6-v2'),
        embedding_text_mode=str(config.get('embedding_text_mode') or spec.default_embedding_text_mode),
        use_embeddings=_as_bool(config.get('use_embeddings'), default=True),
    )
    stratum_mode = str(config.get('calibration_stratum') or spec.default_stratum)
    features = prepare_calibrated_strata(benchmark_name, features, stratum_mode)
    features['selection_vector'] = build_selection_vectors(
        features,
        benchmark_name=benchmark_name,
        score_weight=_as_float(config.get('score_weight'), spec.default_score_weight),
        discrimination_weight=_as_float(
            config.get('discrimination_weight'),
            spec.default_discrimination_weight,
        ),
        structure_weight=_as_float(config.get('structure_weight'), spec.default_structure_weight),
        keyword_weight=_as_float(config.get('keyword_weight'), spec.default_keyword_weight),
        semantic_weight=_as_float(config.get('semantic_weight'), spec.default_semantic_weight),
    )

    selected = _select_stratified_medoids(
        features,
        k,
        'selection_stratum',
        'selection_vector',
        random_state=_as_int(config.get('random_state'), 42),
    )
    logger.info(f'{benchmark_name} calibrated_outputs selected ids: {selected}')
    return selected


def resolve_prune_k(sample_count: int, prune_k: Optional[Any], prune_ratio: Optional[Any]) -> int:
    if sample_count <= 0:
        raise ValueError('Cannot prune an empty dataset.')
    if prune_k is not None:
        return max(1, min(sample_count, int(prune_k)))
    if prune_ratio is None:
        raise ValueError('calibrated_outputs pruning requires prune_k or prune_ratio.')
    ratio = float(prune_ratio)
    if ratio <= 0 or ratio > 1:
        raise ValueError('prune_ratio must be in (0, 1].')
    return max(1, min(sample_count, int(np.ceil(sample_count * ratio))))


def load_benchmark_artifacts(artifact_name: str, calibration_dir: str) -> pd.DataFrame:
    """Load prediction/review JSONL artifacts into one row per original sample."""
    pred_pattern = os.path.join(calibration_dir, 'predictions', f'{artifact_name}__*.jsonl')
    pred_files = sorted(glob(pred_pattern))
    if not pred_files:
        raise FileNotFoundError(f'No prediction files found for {artifact_name!r} in {calibration_dir}')

    model_dfs = []
    for pred_path in pred_files:
        filename = os.path.basename(pred_path)
        model_name = filename.replace(f'{artifact_name}__', '').replace('.jsonl', '')
        review_path = os.path.join(calibration_dir, 'reviews', filename)
        if not os.path.exists(review_path):
            logger.warning(f'Skipping calibration model {model_name}; review file missing: {review_path}')
            continue

        with open(pred_path, encoding='utf-8') as f:
            pred_by_idx = {row['index']: row for row in (json.loads(line) for line in f)}
        with open(review_path, encoding='utf-8') as f:
            review_by_idx = {row['index']: row for row in (json.loads(line) for line in f)}

        rows = [
            _flatten_record(pred_by_idx.get(index, {}), review_by_idx.get(index, {}))
            for index in sorted(set(pred_by_idx) | set(review_by_idx))
        ]
        model_dfs.append((model_name, pd.DataFrame(rows).set_index('index')))

    if not model_dfs:
        raise ValueError(f'No valid prediction/review pairs found for {artifact_name!r} in {calibration_dir}')

    all_indices = pd.Index([])
    for _, df in model_dfs:
        all_indices = all_indices.union(df.index)

    combined = pd.DataFrame(index=all_indices)
    common_written = False
    for model_name, df in model_dfs:
        if not common_written:
            for col in ['question', 'data_source_urls']:
                if col in df.columns:
                    combined[col] = df[col]
            common_written = True

        for col in [
            'score_value',
            'model_reasoning',
            'model_response',
            'mo_usage_input_tokens',
            'extracted_prediction',
        ]:
            if col in df.columns:
                combined[f'{col}_{model_name}'] = df[col]

    return combined.sort_index()


def build_calibrated_features(
    *,
    benchmark_name: str,
    df_raw: pd.DataFrame,
    metadata: Dict[str, Dict[int, Any]],
    embedding_model: str,
    embedding_text_mode: str,
    use_embeddings: bool,
) -> pd.DataFrame:
    features = _base_score_features(df_raw)
    if benchmark_name == 'aa_lcr':
        features['domain'] = pd.Series({
            idx: metadata.get('domain', {}).get(idx, 'unknown') for idx in df_raw.index
        })
        features['mean_input_context_length'] = _mean_prefixed(df_raw, 'mo_usage_input_tokens_')
        features['mean_reasoning_length'] = _mean_text_length(df_raw, 'model_reasoning_')
    elif benchmark_name == 'live_code_bench':
        success_sum = features[_score_feature_cols(features)].sum(axis=1)
        features['difficulty'] = success_sum.apply(
            lambda score: 'Easy' if score == 3 else 'Hard' if score == 0 else 'Medium'
        )
        features['mean_input_context_length'] = _mean_prefixed(df_raw, 'mo_usage_input_tokens_')
        features['mean_reasoning_length'] = _mean_text_length(df_raw, 'model_reasoning_')
        features['mean_output_code_length'] = _mean_text_length(df_raw, 'model_response_')
        features['mean_extracted_code_length'] = _mean_text_length(df_raw, 'extracted_prediction_')
        features['mean_code_lines'] = _mean_line_count(df_raw, 'extracted_prediction_')
        _add_lcb_keyword_features(features, df_raw)
    else:
        raise ValueError(f'Unsupported calibrated benchmark: {benchmark_name}')

    for col in [col for col in features.columns if col.startswith('mean_')]:
        features[f'scaled_{col.removeprefix("mean_")}'] = _min_max(features[col])

    if use_embeddings:
        texts = _embedding_text(df_raw, embedding_text_mode)
        features['norm_sem_embedding'] = list(_normalized_embeddings(texts, embedding_model))
    else:
        features['norm_sem_embedding'] = [np.zeros(0, dtype=float) for _ in range(len(features))]
    return features


def prepare_calibrated_strata(benchmark_name: str, features: pd.DataFrame, mode: str) -> pd.DataFrame:
    features = features.copy()
    if benchmark_name == 'aa_lcr':
        if mode == 'domain':
            features['selection_stratum'] = features['domain'].astype(str)
        elif mode == 'domain_score_pattern':
            features['selection_stratum'] = features[['domain', 'score_pattern']].astype(str).agg(' | '.join, axis=1)
        else:
            raise ValueError(f'Unknown AA-LCR stratum mode: {mode}')
    elif benchmark_name == 'live_code_bench':
        if mode == 'difficulty':
            features['selection_stratum'] = features['difficulty'].astype(str)
        elif mode == 'score_pattern':
            features['selection_stratum'] = features['score_pattern'].astype(str)
        elif mode == 'collapsed_score_pattern':
            counts = features['score_pattern'].value_counts()
            features['selection_stratum'] = features['score_pattern'].where(
                features['score_pattern'].map(counts) >= 8,
                'rare',
            )
        else:
            raise ValueError(f'Unknown LiveCodeBench stratum mode: {mode}')
    else:
        raise ValueError(f'Unsupported calibrated benchmark: {benchmark_name}')
    return features


def build_selection_vectors(
    features: pd.DataFrame,
    *,
    benchmark_name: str,
    score_weight: float,
    discrimination_weight: float,
    structure_weight: float,
    keyword_weight: float,
    semantic_weight: float,
) -> List[np.ndarray]:
    score_cols = _score_feature_cols(features)
    if benchmark_name == 'aa_lcr':
        structure_cols = ['scaled_input_context_length', 'scaled_reasoning_length']
        keyword_cols: List[str] = []
    elif benchmark_name == 'live_code_bench':
        structure_cols = [
            'scaled_input_context_length',
            'scaled_reasoning_length',
            'scaled_output_code_length',
            'scaled_extracted_code_length',
            'scaled_code_lines',
        ]
        keyword_cols = list(_LCB_KEYWORD_PATTERNS) + [
            'uses_recursion',
            'uses_class_solution',
            'uses_stdin_solve',
        ]
    else:
        raise ValueError(f'Unsupported calibrated benchmark: {benchmark_name}')

    vectors = []
    for _, row in features.iterrows():
        parts = [
            row[score_cols].to_numpy(dtype=float) * score_weight,
            np.array([float(row['discrimination']) * discrimination_weight]),
            row[structure_cols].to_numpy(dtype=float) * structure_weight,
        ]
        if keyword_cols:
            parts.append(row[keyword_cols].to_numpy(dtype=float) * keyword_weight)
        embedding = np.asarray(row['norm_sem_embedding'], dtype=float)
        if len(embedding):
            parts.append(embedding * semantic_weight)
        vectors.append(np.concatenate(parts))
    return vectors


def select_lcr_calibrated_indices(
    calibration_dir: str,
    domains_by_id: Dict[int, str],
    k: int,
    *,
    embedding_model: str = 'all-MiniLM-L6-v2',
    stratum_mode: str = 'domain',
    score_weight: float = 0.55,
    discrimination_weight: float = 0.25,
    structure_weight: float = 1.0,
    semantic_weight: float = 0.45,
    random_state: int = 42,
) -> List[int]:
    """Backward-compatible wrapper around the shared backend."""
    dataset_dict = _FakeDatasetDict(domains_by_id)
    return select_calibrated_pruned_indices(
        'aa_lcr',
        dataset_dict,  # type: ignore[arg-type]
        {
            'calibration_dir': calibration_dir,
            'prune_k': k,
            'calibration_embedding_model': embedding_model,
            'calibration_stratum': stratum_mode,
            'score_weight': score_weight,
            'discrimination_weight': discrimination_weight,
            'structure_weight': structure_weight,
            'semantic_weight': semantic_weight,
            'random_state': random_state,
        },
    )


def select_lcb_calibrated_indices(
    calibration_dir: str,
    k: int,
    *,
    available_ids: Optional[List[int]] = None,
    embedding_model: str = 'all-MiniLM-L6-v2',
    stratum_mode: str = 'score_pattern',
    score_weight: float = 0.45,
    discrimination_weight: float = 0.20,
    structure_weight: float = 0.85,
    keyword_weight: float = 0.65,
    semantic_weight: float = 0.40,
    random_state: int = 42,
) -> List[int]:
    """Backward-compatible wrapper around the shared backend."""
    dataset_dict = _FakeDatasetDict({idx: 'available' for idx in available_ids or []})
    return select_calibrated_pruned_indices(
        'live_code_bench',
        dataset_dict,  # type: ignore[arg-type]
        {
            'calibration_dir': calibration_dir,
            'prune_k': k,
            'calibration_embedding_model': embedding_model,
            'calibration_stratum': stratum_mode,
            'score_weight': score_weight,
            'discrimination_weight': discrimination_weight,
            'structure_weight': structure_weight,
            'keyword_weight': keyword_weight,
            'semantic_weight': semantic_weight,
            'random_state': random_state,
            'available_id_policy': 'filter' if available_ids is not None else 'error',
        },
    )


def _align_calibration_rows(df_raw: pd.DataFrame, available_ids: Sequence[int], policy: str) -> pd.DataFrame:
    available = pd.Index(available_ids)
    missing = df_raw.index.difference(available)
    if policy == 'error' and len(missing):
        raise ValueError(
            'Calibration artifacts contain ids that are not present in the loaded EvalScope dataset: '
            f'{missing.tolist()[:20]}{"..." if len(missing) > 20 else ""}'
        )
    aligned = df_raw.loc[df_raw.index.intersection(available)]
    if aligned.empty:
        raise ValueError('No calibration rows match the loaded EvalScope dataset ids.')
    return aligned


def _source_sample_count(dataset_dict: DatasetDict) -> int:
    return sum(len(dataset) for _, dataset in dataset_dict.items())


def _source_ids(dataset_dict: DatasetDict) -> List[int]:
    ids = []
    for _, dataset in dataset_dict.items():
        ids.extend(sample.id for sample in dataset)
    return ids


def _flatten_record(pred: Dict[str, Any], review: Dict[str, Any]) -> Dict[str, Any]:
    flat = {'index': pred.get('index'), 'question': _nested(pred, ['metadata', 'question'])}
    model_output = pred.get('model_output') or {}
    flat['mo_usage_input_tokens'] = _nested(model_output, ['usage', 'input_tokens'])
    reasoning_parts = []
    text_parts = []
    content = _nested(model_output, ['choices', 0, 'message', 'content'])
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get('type') == 'reasoning':
                reasoning_parts.append(part.get('reasoning') or '')
            elif part.get('type') == 'text':
                text_parts.append(part.get('text') or '')
    elif isinstance(content, str):
        text_parts.append(content)
    flat['model_reasoning'] = '\n'.join(reasoning_parts)
    flat['model_response'] = '\n'.join(text_parts)
    score_obj = ((review.get('sample_score') or {}).get('score') or {})
    main_score_name = score_obj.get('main_score_name')
    flat['score_value'] = (score_obj.get('value') or {}).get(main_score_name) if main_score_name else None
    flat['extracted_prediction'] = score_obj.get('extracted_prediction')
    return flat


def _nested(value: Any, keys: List[Any], default: Any = None) -> Any:
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        elif isinstance(value, list) and isinstance(key, int) and len(value) > key:
            value = value[key]
        else:
            return default
    return value


def _base_score_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df_raw.index)
    score_cols = [col for col in df_raw.columns if col.startswith('score_value_')]
    preferred = [
        'score_value_kimi-k2.5',
        'score_value_minimax-m2.5',
        'score_value_gpt-oss-120b',
    ]
    score_cols = [col for col in preferred if col in score_cols] + sorted(
        col for col in score_cols if col not in preferred
    )
    if not score_cols:
        raise ValueError('No score_value_* columns were found in calibration artifacts.')

    for position, col in enumerate(score_cols):
        model_name = col.removeprefix('score_value_')
        stable_name = {
            'kimi-k2.5': 'score_kimi',
            'minimax-m2.5': 'score_minimax',
            'gpt-oss-120b': 'score_gpt',
        }.get(model_name, f'score_model_{position}')
        features[stable_name] = df_raw[col].fillna(0.0).astype(float)

    score_features = _score_feature_cols(features)
    features['score_pattern'] = features[score_features].apply(
        lambda row: ''.join(str(int(value > 0)) for value in row),
        axis=1,
    )
    abilities = features[score_features].mean().to_numpy(dtype=float)
    disc = []
    for _, row in features.iterrows():
        scores = row[score_features].to_numpy(dtype=float)
        if np.std(scores) == 0:
            disc.append(0.0)
        else:
            corr = np.corrcoef(scores, abilities)[0, 1]
            disc.append(0.0 if np.isnan(corr) else float(corr))
    features['discrimination'] = disc
    return features


def _score_feature_cols(features: pd.DataFrame) -> List[str]:
    preferred = ['score_kimi', 'score_minimax', 'score_gpt']
    return [col for col in preferred if col in features] + sorted(
        col for col in features.columns if col.startswith('score_model_')
    )


def _normalized_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            'calibrated_outputs with use_embeddings=true requires sentence-transformers. '
            'Set use_embeddings=false or install sentence-transformers.'
        ) from exc

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.where(norms == 0, 1e-9, norms)


def _embedding_text(df_raw: pd.DataFrame, mode: str) -> List[str]:
    if mode == 'question':
        return df_raw.get('question', pd.Series('', index=df_raw.index)).fillna('').astype(str).tolist()
    if mode == 'reasoning_code':
        return (
            _joined_prefixed(df_raw, 'model_reasoning_') + '\n' + _joined_prefixed(df_raw, 'extracted_prediction_')
        ).fillna('').astype(str).tolist()
    if mode == 'response_code':
        return (
            _joined_prefixed(df_raw, 'model_response_') + '\n' + _joined_prefixed(df_raw, 'extracted_prediction_')
        ).fillna('').astype(str).tolist()
    raise ValueError(f'Unknown embedding_text_mode: {mode}')


def _add_lcb_keyword_features(features: pd.DataFrame, df_raw: pd.DataFrame) -> None:
    code_text = _joined_prefixed(df_raw, 'extracted_prediction_')
    reasoning_text = _joined_prefixed(df_raw, 'model_reasoning_')
    response_text = _joined_prefixed(df_raw, 'model_response_')
    lower_text = (reasoning_text + '\n' + response_text + '\n' + code_text).str.lower()
    for feature, pattern in _LCB_KEYWORD_PATTERNS.items():
        features[feature] = lower_text.apply(lambda text, pat=pattern: int(bool(re.search(pat, text))))
    features['uses_recursion'] = lower_text.apply(
        lambda text: int(bool(re.search(r'\b(recursion|recursive|dfs|sys\.setrecursionlimit)\b', text)))
    )
    features['uses_class_solution'] = code_text.apply(lambda text: int('class Solution' in text))
    features['uses_stdin_solve'] = code_text.apply(lambda text: int(('def solve' in text) or ('sys.stdin' in text)))


def _mean_prefixed(df: pd.DataFrame, prefix: str) -> pd.Series:
    cols = [col for col in df.columns if col.startswith(prefix)]
    return df[cols].mean(axis=1) if cols else pd.Series(0.0, index=df.index)


def _mean_text_length(df: pd.DataFrame, prefix: str) -> pd.Series:
    cols = [col for col in df.columns if col.startswith(prefix)]
    if not cols:
        return pd.Series(0.0, index=df.index)
    return pd.DataFrame({col: df[col].fillna('').astype(str).apply(len) for col in cols}).mean(axis=1)


def _mean_line_count(df: pd.DataFrame, prefix: str) -> pd.Series:
    cols = [col for col in df.columns if col.startswith(prefix)]
    if not cols:
        return pd.Series(0.0, index=df.index)
    return pd.DataFrame({
        col: df[col].fillna('').astype(str).apply(lambda text: len(text.splitlines())) for col in cols
    }).mean(axis=1)


def _joined_prefixed(df: pd.DataFrame, prefix: str) -> pd.Series:
    cols = [col for col in df.columns if col.startswith(prefix)]
    if not cols:
        return pd.Series('', index=df.index)
    return df[cols].fillna('').astype(str).agg('\n'.join, axis=1)


def _min_max(series: pd.Series) -> pd.Series:
    s_min = series.min()
    s_max = series.max()
    if s_max == s_min:
        return pd.Series(0.0, index=series.index)
    return (series - s_min) / (s_max - s_min)


def _select_stratified_medoids(
    features: pd.DataFrame,
    k: int,
    stratify_col: str,
    vector_col: str,
    *,
    random_state: int,
) -> List[int]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min

    if k <= 0:
        return []
    if k >= len(features):
        return sorted(features.index.tolist())

    counts = features[stratify_col].value_counts()
    active = [cat for cat, count in counts.items() if count > 0]
    allocations = {cat: 0 for cat in active}
    remaining = k
    for cat in active:
        allocations[cat] = 1
        remaining -= 1
        if remaining == 0:
            break
    while remaining > 0:
        candidates = [cat for cat in active if allocations[cat] < counts[cat]]
        if not candidates:
            break
        best = max(candidates, key=lambda cat: counts[cat] / len(features) - allocations[cat] / k)
        allocations[best] += 1
        remaining -= 1

    selected = []
    for cat, cat_k in allocations.items():
        if cat_k <= 0:
            continue
        subset = features[features[stratify_col] == cat]
        if len(subset) <= cat_k:
            selected.extend(subset.index.tolist())
            continue
        matrix = np.stack(subset[vector_col].values)
        kmeans = KMeans(n_clusters=cat_k, random_state=random_state, n_init='auto')
        kmeans.fit(matrix)
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, matrix)
        subset_indices = subset.index.tolist()
        selected.extend(subset_indices[idx] for idx in closest)
    return sorted(dict.fromkeys(selected))


def _as_float(value: Optional[Any], default: float) -> float:
    return default if value is None else float(value)


def _as_int(value: Optional[Any], default: int) -> int:
    return default if value is None else int(value)


def _as_bool(value: Optional[Any], default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {'1', 'true', 'yes', 'y'}
    return bool(value)


class _FakeSample:
    def __init__(self, idx: int, domain: str):
        self.id = idx
        self.metadata = {'document_category': domain}


class _FakeDataset(list):
    pass


class _FakeDatasetDict(dict):
    def __init__(self, domains_by_id: Dict[int, str]):
        super().__init__({'default': _FakeDataset(_FakeSample(idx, domain) for idx, domain in domains_by_id.items())})


_LCB_KEYWORD_PATTERNS = {
    'kw_graph': r'\b(graph|tree|node|edge|dfs|bfs|dijkstra|shortest|floyd|warshall|lca)\b',
    'kw_dp': r'\b(dp|dynamic programming|memo|state transition)\b',
    'kw_search': r'\b(binary search|bisect|lower_bound|upper_bound|two pointers|sliding window)\b',
    'kw_greedy_sort': r'\b(greedy|sort|sorted|heapq|priority queue)\b',
    'kw_math': r'\b(gcd|lcm|modulo|prime|combin|factorial|matrix|geometry)\b',
    'kw_bitmask': r'\b(bitmask|bit set|mask|subset)\b',
    'kw_string': r'\b(string|substring|subsequence|prefix|suffix|trie|regex)\b',
    'kw_data_structures': r'\b(deque|counter|defaultdict|set|dict|heapq|fenwick|segment tree|union find|disjoint)\b',
}
