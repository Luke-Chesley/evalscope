import base64
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw

from evalscope.api.dataset import DatasetDict, MemoryDataset, Sample
from evalscope.api.messages import ChatMessageUser, ContentImage, ContentText
import evalscope.benchmarks.mmmu.visual_pruning as visual_pruning
from evalscope.benchmarks.mmmu.visual_pruning import (
    apply_mmmu_visual_pruning,
    extract_visual_features_from_sample,
    score_visual_stress,
    select_stress_diverse_probe,
)


def _image_uri(size: tuple[int, int] = (64, 64), color: tuple[int, int, int] = (255, 255, 255)) -> str:
    image = Image.new('RGB', size, color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, size[0] - 8, size[1] - 8), outline=(0, 0, 0), width=2)
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    payload = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{payload}'


def _sample(idx: int, image_type: Any, *, subset: str = 'Accounting') -> Sample:
    return Sample(
        input=[
            ChatMessageUser(
                content=[
                    ContentText(text='Answer the question from the image.'),
                    ContentImage(image=_image_uri()),
                ]
            )
        ],
        target='A',
        id=idx,
        group_id=idx,
        metadata={
            'id': f'{subset}-{idx}',
            'question_type': 'multiple-choice',
            'subfield': 'demo',
            'img_type': image_type,
            'topic_difficulty': 'medium',
        },
    )


def _dataset_dict() -> DatasetDict:
    return DatasetDict({
        'Accounting': MemoryDataset([_sample(0, 'Tables', subset='Accounting')], name='Accounting'),
        'Agriculture': MemoryDataset([_sample(0, 'Landscapes', subset='Agriculture')], name='Agriculture'),
    })


def test_visual_features_extract_base64_image_stats_without_optional_backends() -> None:
    row = extract_visual_features_from_sample(
        _sample(0, "['Tables']"),
        subset='Accounting',
        source_id=0,
        ocr_engine=None,
        clip_encoder=None,
    )

    assert row['num_images'] == 1
    assert row['image_decode_error_count'] == 0
    assert row['max_width'] == 64
    assert row['max_height'] == 64
    assert row['ocr_available'] is False
    assert row['clip_available'] is False
    assert row['image_type_label'] == 'Tables'

    scored = score_visual_stress(pd.DataFrame([row]))
    assert scored.loc[0, 'visual_stress_score'] > 0


def test_visual_pruning_is_subject_aware_when_sample_ids_repeat() -> None:
    pruned = apply_mmmu_visual_pruning(
        _dataset_dict(),
        {
            'prune_k': 1,
            'candidate_pool_size': 2,
            'run_ocr': False,
            'use_clip': False,
        },
    )

    assert len(pruned['Accounting']) == 1
    assert len(pruned['Agriculture']) == 0
    assert sum(len(dataset) for _, dataset in pruned.items()) == 1
    selected = list(pruned['Accounting'])[0]
    assert selected.id == 0
    assert selected.metadata['mmmu_visual_source_subset'] == 'Accounting'
    assert selected.metadata['mmmu_visual_source_id'] == 0


def test_stress_diverse_selection_uses_clip_novelty_after_top_stress_sample() -> None:
    scored_df = pd.DataFrame(
        {
            'subset': ['A', 'A', 'B'],
            'source_id': [0, 1, 0],
            'img_type': ['Tables', 'Landscapes', 'Diagrams'],
            'image_type_label': ['Tables', 'Landscapes', 'Diagrams'],
            'topic_difficulty': ['hard', 'easy', 'medium'],
            'visual_stress_score': [0.90, 0.85, 0.80],
            'clip_embedding': [
                np.array([1.0, 0.0]),
                np.array([0.0, 1.0]),
                np.array([1.0, 0.0]),
            ],
        }
    )

    selected = select_stress_diverse_probe(
        scored_df,
        k=2,
        candidate_pool_size=3,
        stress_weight=0.70,
        diversity_weight=0.25,
        coverage_weight=0.05,
    )

    assert selected[['subset', 'source_id']].values.tolist() == [['A', 0], ['A', 1]]
    assert selected.iloc[1]['selection_novelty'] == 1.0


def test_ocr_dependency_failure_is_not_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_ocr(_: Any) -> Any:
        raise RuntimeError('missing tesseract')

    monkeypatch.setattr(visual_pruning, '_require_pytesseract', fail_ocr)

    with pytest.raises(RuntimeError, match='missing tesseract'):
        apply_mmmu_visual_pruning(
            _dataset_dict(),
            {
                'prune_k': 1,
                'run_ocr': True,
                'use_clip': False,
            },
        )


def test_clip_dependency_failure_is_not_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingClipEncoder:

        def __init__(self, _: str):
            raise RuntimeError('missing clip')

    monkeypatch.setattr(visual_pruning, 'ClipImageEncoder', FailingClipEncoder)

    with pytest.raises(RuntimeError, match='missing clip'):
        apply_mmmu_visual_pruning(
            _dataset_dict(),
            {
                'prune_k': 1,
                'run_ocr': False,
                'use_clip': True,
            },
        )


def test_ocr_and_clip_disabled_still_runs_with_image_metadata() -> None:
    pruned = apply_mmmu_visual_pruning(
        _dataset_dict(),
        {
            'prune_k': 1,
            'candidate_pool_size': 1,
            'run_ocr': False,
            'use_clip': False,
        },
    )

    assert sum(len(dataset) for _, dataset in pruned.items()) == 1
