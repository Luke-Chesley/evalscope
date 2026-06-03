from typing import Any

from evalscope.api.dataset import DatasetDict, MemoryDataset, Sample
from evalscope.api.dataset.pruning import (
    BENCHMARK_SPECS,
    apply_coreset_pruning,
    apply_coreset_pruning_by_ratio,
    apply_index_subset,
    build_calibrated_features,
    build_selection_vectors,
    parse_index_list,
    prepare_calibrated_strata,
    resolve_prune_k,
    select_diverse_samples,
)
from evalscope.api.messages import ChatMessageUser
from evalscope.benchmarks.aa_lcr.aa_lcr_adapter import aa_lcr_feature_builder
from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import live_code_bench_feature_builder


def _sample(text: str, idx: int, **metadata: Any) -> Sample:
    return Sample(input=[ChatMessageUser(content=text)], id=idx, group_id=idx, metadata=metadata)


def test_parse_index_list_accepts_comma_separated_ids() -> None:
    assert parse_index_list('1, 3,5') == [1, 3, 5]


def test_fixed_index_subset_reindexes_selected_samples() -> None:
    dataset = MemoryDataset([_sample(f'question {idx}', idx) for idx in range(5)], name='demo')
    pruned = apply_index_subset(DatasetDict({'default': dataset}), [1, 3])

    samples = list(pruned['default'])
    assert len(samples) == 2
    assert [sample.id for sample in samples] == [0, 1]
    assert [sample.group_id for sample in samples] == [0, 1]


def test_diverse_selection_preserves_strata() -> None:
    samples = [
        _sample('company document short', 0, document_category='company', input_tokens=100),
        _sample('company document long', 1, document_category='company', input_tokens=10000),
        _sample('legal document short', 2, document_category='legal', input_tokens=100),
        _sample('legal document long', 3, document_category='legal', input_tokens=10000),
    ]

    selected = select_diverse_samples(samples, k=2, feature_builder=aa_lcr_feature_builder)
    selected_categories = {samples[idx].metadata['document_category'] for idx in selected}

    assert selected_categories == {'company', 'legal'}


def test_coreset_pruning_uses_lcb_prompt_features() -> None:
    samples = [
        _sample('Given a graph, find the shortest path.', 0, question_content='Given a graph, find the shortest path.'),
        _sample('Reverse a string.', 1, question_content='Reverse a string.'),
        _sample(
            'Use dynamic programming over a bitmask.',
            2,
            question_content='Use dynamic programming over a bitmask.',
        ),
    ]
    dataset = MemoryDataset(samples, name='lcb')

    pruned = apply_coreset_pruning(
        DatasetDict({'v5': dataset}),
        k=2,
        feature_builder=live_code_bench_feature_builder,
    )

    assert len(pruned['v5']) == 2
    assert [sample.id for sample in pruned['v5']] == [0, 1]


def test_coreset_pruning_ratio_resolves_subset_size() -> None:
    samples = [_sample(f'question {idx}', idx) for idx in range(10)]
    dataset = MemoryDataset(samples, name='ratio-demo')

    pruned = apply_coreset_pruning_by_ratio(
        DatasetDict({'default': dataset}),
        prune_k=None,
        prune_ratio=0.3,
        feature_builder=aa_lcr_feature_builder,
    )

    assert len(pruned['default']) == 3


def test_calibrated_prune_k_resolution_prefers_exact_k() -> None:
    assert resolve_prune_k(sample_count=100, prune_k=25, prune_ratio=0.5) == 25


def test_calibrated_prune_k_resolution_uses_ratio() -> None:
    assert resolve_prune_k(sample_count=100, prune_k=None, prune_ratio=0.4) == 40


def test_calibrated_backend_builds_lcr_vector_without_embeddings() -> None:
    import pandas as pd

    raw = pd.DataFrame(
        {
            'score_value_kimi-k2.5': [1.0, 0.0],
            'score_value_minimax-m2.5': [1.0, 1.0],
            'score_value_gpt-oss-120b': [0.0, 0.0],
            'question': ['question one', 'question two'],
            'mo_usage_input_tokens_kimi-k2.5': [10, 20],
            'model_reasoning_kimi-k2.5': ['short', 'longer'],
        },
        index=[0, 1],
    )
    features = build_calibrated_features(
        benchmark_name='aa_lcr',
        df_raw=raw,
        metadata={'domain': {0: 'Legal', 1: 'Academia'}},
        embedding_model='all-MiniLM-L6-v2',
        embedding_text_mode=BENCHMARK_SPECS['aa_lcr'].default_embedding_text_mode,
        use_embeddings=False,
    )
    features = prepare_calibrated_strata('aa_lcr', features, 'domain')
    vectors = build_selection_vectors(
        features,
        benchmark_name='aa_lcr',
        score_weight=0.55,
        discrimination_weight=0.25,
        structure_weight=1.0,
        keyword_weight=0.0,
        semantic_weight=0.45,
    )

    assert features['selection_stratum'].tolist() == ['Legal', 'Academia']
    assert len(vectors) == 2
