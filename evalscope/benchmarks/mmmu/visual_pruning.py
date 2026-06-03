"""Visual-stress pruning for MMMU.

This backend selects a small MMMU probe that stresses image encoding, then
keeps enough visual diversity to avoid producing a narrow OCR-only subset.
"""

from __future__ import annotations

import ast
import base64
import copy
import html
import json
import math
import os
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from evalscope.api.dataset import DatasetDict, MemoryDataset, Sample
from evalscope.api.messages import ContentImage
from evalscope.utils.logger import get_logger

logger = get_logger()

DEFAULT_PRUNE_K = 120
DEFAULT_CANDIDATE_POOL_SIZE = 600
DEFAULT_CLIP_MODEL = 'openai/clip-vit-base-patch32'

MMMU_VISUAL_PRUNING_EXTRA_PARAMS = {
    'pruning_strategy': {
        'type': 'str',
        'description': 'Pruning strategy name. Supported for MMMU: visual_stress, fixed_indices, metadata_coreset.',
        'value': 'visual_stress'
    },
    'prune_k': {
        'type': 'int | null',
        'description': 'Exact number of MMMU validation samples to keep for the visual probe.',
        'value': DEFAULT_PRUNE_K
    },
    'candidate_pool_size': {
        'type': 'int',
        'description': 'Number of highest visual-stress samples considered by the diversity selector.',
        'value': DEFAULT_CANDIDATE_POOL_SIZE
    },
    'run_ocr': {
        'type': 'bool',
        'description': 'Run pytesseract OCR while extracting visual stress features.',
        'value': True
    },
    'use_clip': {
        'type': 'bool',
        'description': 'Use CLIP image embeddings for visual diversity.',
        'value': True
    },
    'clip_model': {
        'type': 'str',
        'description': 'Hugging Face CLIP model used for image diversity embeddings.',
        'value': DEFAULT_CLIP_MODEL
    },
    'tesseract_cmd': {
        'type': 'str | null',
        'description': 'Optional path to the system tesseract executable.',
        'value': None
    },
    'stress_weight': {
        'type': 'float',
        'description': 'Greedy selection weight for visual stress score.',
        'value': 0.70
    },
    'diversity_weight': {
        'type': 'float',
        'description': 'Greedy selection weight for CLIP novelty.',
        'value': 0.25
    },
    'coverage_weight': {
        'type': 'float',
        'description': 'Greedy selection weight for new image-type and subject coverage.',
        'value': 0.05
    },
    'save_pruning_report': {
        'type': 'bool',
        'description': 'Write selected samples and coverage diagnostics before evaluation.',
        'value': False
    },
    'pruning_output_dir': {
        'type': 'str | null',
        'description': 'Directory for optional visual pruning report artifacts.',
        'value': None
    },
    'include_html_preview': {
        'type': 'bool',
        'description': 'Include base64 image previews in an optional HTML report.',
        'value': False
    },
}

IMAGE_TYPE_STRESS_WEIGHTS = {
    'tables': 0.95,
    'table': 0.95,
    'plots and charts': 0.95,
    'charts': 0.95,
    'chart': 0.95,
    'plot': 0.95,
    'technical blueprints': 0.95,
    'technical blueprint': 0.95,
    'blueprint': 0.95,
    'diagrams': 0.90,
    'diagram': 0.90,
    'chemical structures': 0.90,
    'chemical structure': 0.90,
    'mathematical notations': 0.90,
    'mathematical notation': 0.90,
    'medical images': 0.85,
    'microscopic images': 0.85,
    'pathological images': 0.85,
    'body scans': 0.85,
    'body scan': 0.85,
    'maps': 0.80,
    'trees and graphs': 0.80,
    'tree': 0.80,
    'graph': 0.80,
    'screenshots': 0.75,
    'screenshot': 0.75,
    'dna sequences': 0.75,
    'dna': 0.75,
    'icons': 0.70,
    'icon': 0.70,
    'comics': 0.60,
    'comic': 0.60,
    'photographs': 0.55,
    'photograph': 0.55,
    'photo': 0.55,
    'paintings': 0.45,
    'painting': 0.45,
    'portraits': 0.35,
    'portrait': 0.35,
    'landscapes': 0.30,
    'landscape': 0.30,
}

STRESS_COMPONENT_WEIGHTS = {
    'stress_ocr_density': 0.22,
    'stress_ocr_volume': 0.14,
    'stress_small_text': 0.18,
    'stress_numeric_symbolic': 0.16,
    'stress_layout_complexity': 0.18,
    'stress_multi_image': 0.04,
    'stress_image_type': 0.08,
}


@dataclass(frozen=True)
class MmmuVisualPruningConfig:
    """Runtime config for MMMU visual-stress pruning."""

    prune_k: Optional[int] = DEFAULT_PRUNE_K
    prune_ratio: Optional[float] = None
    candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE
    run_ocr: bool = True
    use_clip: bool = True
    clip_model: str = DEFAULT_CLIP_MODEL
    tesseract_cmd: Optional[str] = None
    stress_weight: float = 0.70
    diversity_weight: float = 0.25
    coverage_weight: float = 0.05
    save_pruning_report: bool = False
    pruning_output_dir: Optional[str] = None
    include_html_preview: bool = False

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> 'MmmuVisualPruningConfig':
        return cls(
            prune_k=_optional_int(params.get('prune_k'), DEFAULT_PRUNE_K),
            prune_ratio=_optional_float(params.get('prune_ratio')),
            candidate_pool_size=int(params.get('candidate_pool_size') or DEFAULT_CANDIDATE_POOL_SIZE),
            run_ocr=_as_bool(params.get('run_ocr'), True),
            use_clip=_as_bool(params.get('use_clip'), True),
            clip_model=str(params.get('clip_model') or DEFAULT_CLIP_MODEL),
            tesseract_cmd=_optional_str(params.get('tesseract_cmd')),
            stress_weight=float(params.get('stress_weight') if params.get('stress_weight') is not None else 0.70),
            diversity_weight=float(
                params.get('diversity_weight') if params.get('diversity_weight') is not None else 0.25
            ),
            coverage_weight=float(
                params.get('coverage_weight') if params.get('coverage_weight') is not None else 0.05
            ),
            save_pruning_report=_as_bool(params.get('save_pruning_report'), False),
            pruning_output_dir=_optional_str(params.get('pruning_output_dir')),
            include_html_preview=_as_bool(params.get('include_html_preview'), False),
        )


class OcrEngine:
    """Thin wrapper around pytesseract with an explicit binary check."""

    def __init__(self, tesseract_cmd: Optional[str] = None):
        self.pytesseract = _require_pytesseract(tesseract_cmd)

    def extract(self, image: Image.Image) -> Dict[str, Any]:
        text = self.pytesseract.image_to_string(image)
        data = self.pytesseract.image_to_data(image, output_type=self.pytesseract.Output.DICT)
        words, heights = _ocr_words_and_heights(data)
        return _ocr_feature_dict(text=text, words=words, heights=heights, image_height=image.height)


class ClipImageEncoder:
    """CLIP image encoder used only for selection-time diversity."""

    def __init__(self, model_name: str):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except Exception as exc:
            raise ImportError(
                "MMMU visual pruning with use_clip=true requires torch and transformers. "
                "Install the optional extra with `pip install 'evalscope[mmmu_visual_pruning]'`."
            ) from exc

        try:
            self.torch = torch
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name)
            self.model.eval()
            self.model_name = model_name
        except Exception as exc:
            raise RuntimeError(f'Failed to load CLIP model {model_name!r} for MMMU visual pruning: {exc}') from exc

    def encode(self, images: Sequence[Image.Image]) -> Optional[np.ndarray]:
        if not images:
            return None

        rgb_images = [image.convert('RGB') for image in images]
        with self.torch.no_grad():
            inputs = self.processor(images=rgb_images, return_tensors='pt')
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            embedding = features.mean(dim=0)
            embedding = embedding / embedding.norm().clamp(min=1e-12)
        return embedding.detach().cpu().numpy().astype(float)


def apply_mmmu_visual_pruning(
    dataset_dict: DatasetDict,
    params: Dict[str, Any],
    *,
    repeats: int = 1,
) -> DatasetDict:
    """Apply MMMU visual-stress pruning and return a reindexed DatasetDict."""
    config = MmmuVisualPruningConfig.from_params(params)
    sample_count = _sample_count(dataset_dict)
    k = _resolve_visual_k(sample_count, config.prune_k, config.prune_ratio)

    ocr_engine = OcrEngine(config.tesseract_cmd) if config.run_ocr else None
    clip_encoder = ClipImageEncoder(config.clip_model) if config.use_clip else None

    feature_df = build_visual_feature_frame(
        dataset_dict,
        ocr_engine=ocr_engine,
        clip_encoder=clip_encoder,
        include_image_data=config.include_html_preview,
    )
    scored_df = score_visual_stress(feature_df)
    selected_df = select_stress_diverse_probe(
        scored_df,
        k=k,
        candidate_pool_size=config.candidate_pool_size,
        stress_weight=config.stress_weight,
        diversity_weight=config.diversity_weight,
        coverage_weight=config.coverage_weight,
    )

    if config.save_pruning_report:
        write_visual_pruning_report(
            scored_df=scored_df,
            selected_df=selected_df,
            output_dir=config.pruning_output_dir,
            include_html_preview=config.include_html_preview,
        )

    logger.info(f'MMMU visual_stress selected {len(selected_df)} / {sample_count} samples.')
    return apply_subject_aware_selection(dataset_dict, selected_df, repeats=repeats)


def build_visual_feature_frame(
    dataset_dict: DatasetDict,
    *,
    ocr_engine: Optional[OcrEngine] = None,
    clip_encoder: Optional[ClipImageEncoder] = None,
    include_image_data: bool = False,
) -> pd.DataFrame:
    """Extract MMMU visual features from every sample in the loaded DatasetDict."""
    rows = []
    for subset, dataset in dataset_dict.items():
        for position, sample in enumerate(dataset):
            source_id = _sample_source_id(sample, position)
            rows.append(
                extract_visual_features_from_sample(
                    sample,
                    subset=subset,
                    source_id=source_id,
                    ocr_engine=ocr_engine,
                    clip_encoder=clip_encoder,
                    include_image_data=include_image_data,
                )
            )
    if not rows:
        raise ValueError('Cannot run MMMU visual pruning on an empty dataset.')
    return pd.DataFrame(rows)


def extract_visual_features_from_sample(
    sample: Sample,
    *,
    subset: str,
    source_id: int,
    ocr_engine: Optional[OcrEngine] = None,
    clip_encoder: Optional[ClipImageEncoder] = None,
    include_image_data: bool = False,
) -> Dict[str, Any]:
    """Extract image, OCR, CLIP, and metadata features for one MMMU sample."""
    image_values = extract_image_values(sample)
    decoded_images = []
    image_decode_error_count = 0
    total_image_bytes = 0
    image_stats = []

    for image_value in image_values:
        try:
            raw_bytes, image = _decode_image_value(image_value)
            total_image_bytes += len(raw_bytes)
            decoded_images.append(image)
            image_stats.append(_image_feature_dict(image))
        except Exception:
            image_decode_error_count += 1

    metadata = sample.metadata or {}
    image_types = parse_image_types(metadata.get('img_type'))
    row = {
        'subset': subset,
        'source_id': source_id,
        'dataset_id': metadata.get('id'),
        'question_type': metadata.get('question_type'),
        'subfield': metadata.get('subfield'),
        'img_type': metadata.get('img_type'),
        'image_type_label': ' | '.join(image_types),
        'image_type_stress': max((_image_type_stress(image_type) for image_type in image_types), default=0.50),
        'topic_difficulty': metadata.get('topic_difficulty'),
        'num_images': len(image_values),
        'image_decode_error_count': image_decode_error_count,
        'total_image_bytes': total_image_bytes,
        **_aggregate_image_stats(image_stats),
    }

    row.update(_aggregate_ocr_features(decoded_images, ocr_engine))
    if clip_encoder is None:
        row.update({
            'clip_available': False,
            'clip_model': None,
            'clip_embedding_dim': None,
            'clip_embedding': None,
        })
    else:
        embedding = clip_encoder.encode(decoded_images)
        row.update({
            'clip_available': embedding is not None,
            'clip_model': clip_encoder.model_name,
            'clip_embedding_dim': int(len(embedding)) if embedding is not None else None,
            'clip_embedding': embedding,
        })

    if include_image_data:
        row['image_data_uris'] = image_values

    return row


def extract_image_values(sample: Sample) -> List[str]:
    """Return base64 image payloads from an EvalScope sample."""
    if isinstance(sample.input, str):
        return []

    image_values = []
    for message in sample.input:
        content = message.content
        if isinstance(content, str):
            continue
        for item in content:
            if isinstance(item, ContentImage) or getattr(item, 'type', None) == 'image':
                image_value = getattr(item, 'image', None)
                if image_value:
                    image_values.append(str(image_value))
    return image_values


def score_visual_stress(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Add notebook-equivalent visual stress component scores."""
    scored = feature_df.copy()
    scored['stress_ocr_density'] = _rank_signal(scored.get('ocr_words_per_megapixel'))
    scored['stress_ocr_volume'] = _rank_signal(scored.get('ocr_word_count'))
    scored['stress_small_text'] = _rank_signal(_small_text_signal(scored.get('ocr_median_box_height_ratio')))
    scored['stress_numeric_symbolic'] = (
        0.60 * _rank_signal(scored.get('ocr_number_count'))
        + 0.40 * _rank_signal(scored.get('ocr_equation_symbol_count'))
    )
    scored['stress_layout_complexity'] = (
        0.40 * _rank_signal(scored.get('ocr_box_count'))
        + 0.30 * _rank_signal(scored.get('mean_edge_density'))
        + 0.20 * _rank_signal(scored.get('mean_gray_entropy'))
        + 0.10 * _rank_signal(scored.get('mean_contrast'))
    )
    scored['stress_multi_image'] = _rank_signal(scored.get('num_images'))
    scored['stress_image_type'] = pd.to_numeric(scored.get('image_type_stress'), errors='coerce').fillna(0.50)
    scored['visual_stress_score'] = sum(
        scored[column] * weight for column, weight in STRESS_COMPONENT_WEIGHTS.items()
    )
    return scored


def select_stress_diverse_probe(
    scored_df: pd.DataFrame,
    *,
    k: int,
    candidate_pool_size: int,
    stress_weight: float,
    diversity_weight: float,
    coverage_weight: float,
) -> pd.DataFrame:
    """Greedily select high-stress samples while preserving visual diversity."""
    if k <= 0:
        raise ValueError('prune_k must be positive.')
    if k >= len(scored_df):
        selected = scored_df.copy()
        selected['selection_order'] = range(len(selected))
        selected['selection_score'] = selected['visual_stress_score']
        selected['selection_novelty'] = 0.0
        selected['selection_coverage_bonus'] = 0.0
        return selected

    pool_size = max(k, min(len(scored_df), int(candidate_pool_size)))
    candidates = scored_df.sort_values(
        ['visual_stress_score', 'subset', 'source_id'],
        ascending=[False, True, True],
    ).head(pool_size).copy()
    candidates['_candidate_rank'] = range(len(candidates))

    selected_indices = []
    selected_image_types = set()
    selected_subjects = set()

    while len(selected_indices) < k:
        best_idx = None
        best_key = None
        for idx, row in candidates.drop(index=selected_indices).iterrows():
            novelty = _clip_novelty(row.get('clip_embedding'), candidates.loc[selected_indices])
            coverage_bonus = _coverage_bonus(row, selected_image_types, selected_subjects)
            selection_score = (
                stress_weight * float(row['visual_stress_score'])
                + diversity_weight * novelty
                + coverage_weight * coverage_bonus
            )
            key = (
                selection_score,
                float(row['visual_stress_score']),
                novelty,
                coverage_bonus,
                -int(row['_candidate_rank']),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_idx = idx

        if best_idx is None:
            break

        selected_indices.append(best_idx)
        best_row = candidates.loc[best_idx]
        selected_image_types.update(parse_image_types(best_row.get('img_type')))
        selected_subjects.add(str(best_row.get('subset')))

    selected = candidates.loc[selected_indices].copy()
    selected['selection_order'] = range(len(selected))
    selected['selection_novelty'] = [
        _clip_novelty(row.get('clip_embedding'), selected.iloc[:order]) for order, (_, row) in enumerate(selected.iterrows())
    ]
    selected['selection_coverage_bonus'] = [
        _coverage_bonus_for_order(row, selected.iloc[:order]) for order, (_, row) in enumerate(selected.iterrows())
    ]
    selected['selection_score'] = (
        stress_weight * selected['visual_stress_score']
        + diversity_weight * selected['selection_novelty']
        + coverage_weight * selected['selection_coverage_bonus']
    )
    return selected.drop(columns=['_candidate_rank'])


def apply_subject_aware_selection(
    dataset_dict: DatasetDict,
    selected_df: pd.DataFrame,
    *,
    repeats: int = 1,
) -> DatasetDict:
    """Keep selected samples by `(subset, source_id)` and then reindex per subset."""
    selected_keys = {(str(row['subset']), int(row['source_id'])) for _, row in selected_df.iterrows()}
    selected_metadata = {
        (str(row['subset']), int(row['source_id'])): row.to_dict() for _, row in selected_df.iterrows()
    }

    pruned = {}
    for subset, dataset in dataset_dict.items():
        samples = []
        for position, sample in enumerate(dataset):
            source_id = _sample_source_id(sample, position)
            key = (str(subset), int(source_id))
            if key not in selected_keys:
                continue
            selected_sample = copy.deepcopy(sample)
            selected_sample.metadata = dict(selected_sample.metadata or {})
            selected_sample.metadata.setdefault('source_id', source_id)
            _attach_selection_metadata(selected_sample, selected_metadata[key])
            samples.append(selected_sample)

        memory_dataset = MemoryDataset(samples=samples, name=dataset.name, location=dataset.location)
        memory_dataset.reindex(group_size=repeats)
        pruned[subset] = memory_dataset
        logger.info(f'Applied MMMU visual pruning to {subset}: kept {len(samples)} / {len(dataset)} samples.')

    return DatasetDict(pruned)


def write_visual_pruning_report(
    *,
    scored_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    output_dir: Optional[str],
    include_html_preview: bool,
) -> None:
    """Write optional MMMU visual pruning artifacts."""
    target_dir = Path(output_dir or os.path.join('outputs', 'mmmu_visual_pruning'))
    target_dir.mkdir(parents=True, exist_ok=True)

    selected_export = _report_export_frame(selected_df)
    selected_export.to_csv(target_dir / 'selected_mmmu_visual_probe.csv', index=False)
    (target_dir / 'selected_mmmu_visual_probe.json').write_text(
        json.dumps(_json_records(selected_export), indent=2),
        encoding='utf-8',
    )
    (target_dir / 'coverage_summary.json').write_text(
        json.dumps(_coverage_summary(scored_df, selected_df), indent=2),
        encoding='utf-8',
    )
    if include_html_preview:
        (target_dir / 'selected_mmmu_visual_probe.html').write_text(
            _html_preview(selected_df),
            encoding='utf-8',
        )

    logger.info(f'MMMU visual pruning report written to {target_dir}')


def parse_image_types(value: Any) -> List[str]:
    """Normalize MMMU image type metadata into a list of strings."""
    if value is None:
        return ['unknown']
    if isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items or ['unknown']
    text = str(value).strip()
    if not text:
        return ['unknown']
    if text.startswith('['):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                items = [str(item).strip() for item in parsed if str(item).strip()]
                return items or ['unknown']
        except Exception:
            pass
    if ',' in text:
        items = [item.strip().strip("'\"") for item in text.split(',') if item.strip()]
        return items or ['unknown']
    return [text.strip("'\"")]


def _attach_selection_metadata(sample: Sample, row: Dict[str, Any]) -> None:
    for key in [
        'selection_order',
        'selection_score',
        'selection_novelty',
        'selection_coverage_bonus',
        'visual_stress_score',
        'stress_ocr_density',
        'stress_ocr_volume',
        'stress_small_text',
        'stress_numeric_symbolic',
        'stress_layout_complexity',
        'stress_multi_image',
        'stress_image_type',
    ]:
        sample.metadata[f'mmmu_visual_{key}'] = _json_value(row.get(key))
    sample.metadata['mmmu_visual_source_subset'] = row.get('subset')
    sample.metadata['mmmu_visual_source_id'] = _json_value(row.get('source_id'))


def _require_pytesseract(tesseract_cmd: Optional[str]) -> Any:
    try:
        import pytesseract
    except Exception as exc:
        raise ImportError(
            "MMMU visual pruning with run_ocr=true requires pytesseract. "
            "Install the optional extra with `pip install 'evalscope[mmmu_visual_pruning]'`."
        ) from exc

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        raise RuntimeError(
            'MMMU visual pruning with run_ocr=true requires the system tesseract executable. '
            'Install tesseract separately or pass tesseract_cmd.'
        ) from exc
    return pytesseract


def _decode_image_value(image_value: str) -> Tuple[bytes, Image.Image]:
    if image_value.startswith('http://') or image_value.startswith('https://'):
        raise ValueError('MMMU visual pruning expects embedded base64 images, not remote URLs.')
    payload = image_value.split(',', 1)[1] if ',' in image_value else image_value
    raw_bytes = base64.b64decode(payload)
    image = Image.open(BytesIO(raw_bytes)).convert('RGB')
    return raw_bytes, image


def _image_feature_dict(image: Image.Image) -> Dict[str, float]:
    gray = np.asarray(image.convert('L'), dtype=np.float32)
    gradient_y, gradient_x = np.gradient(gray)
    edge_magnitude = np.hypot(gradient_x, gradient_y)
    threshold = max(10.0, float(edge_magnitude.mean() + edge_magnitude.std()))
    histogram, _ = np.histogram(gray, bins=256, range=(0, 255), density=False)
    probabilities = histogram.astype(float) / max(1, int(histogram.sum()))
    probabilities = probabilities[probabilities > 0]
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    return {
        'width': float(image.width),
        'height': float(image.height),
        'pixels': float(image.width * image.height),
        'aspect_ratio': float(image.width / image.height) if image.height else 0.0,
        'brightness': float(gray.mean()) if gray.size else 0.0,
        'contrast': float(gray.std()) if gray.size else 0.0,
        'edge_density': float((edge_magnitude > threshold).mean()) if edge_magnitude.size else 0.0,
        'edge_variance': float(edge_magnitude.var()) if edge_magnitude.size else 0.0,
        'gray_entropy': entropy,
    }


def _aggregate_image_stats(stats: Sequence[Dict[str, float]]) -> Dict[str, Optional[float]]:
    if not stats:
        return {
            'max_width': None,
            'max_height': None,
            'max_pixels': None,
            'mean_aspect_ratio': None,
            'mean_brightness': None,
            'mean_contrast': None,
            'mean_edge_density': None,
            'mean_edge_variance': None,
            'mean_gray_entropy': None,
        }
    return {
        'max_width': max(item['width'] for item in stats),
        'max_height': max(item['height'] for item in stats),
        'max_pixels': max(item['pixels'] for item in stats),
        'mean_aspect_ratio': _mean(item['aspect_ratio'] for item in stats),
        'mean_brightness': _mean(item['brightness'] for item in stats),
        'mean_contrast': _mean(item['contrast'] for item in stats),
        'mean_edge_density': _mean(item['edge_density'] for item in stats),
        'mean_edge_variance': _mean(item['edge_variance'] for item in stats),
        'mean_gray_entropy': _mean(item['gray_entropy'] for item in stats),
    }


def _aggregate_ocr_features(
    images: Sequence[Image.Image],
    ocr_engine: Optional[OcrEngine],
) -> Dict[str, Any]:
    if ocr_engine is None:
        return {
            'ocr_available': False,
            'ocr_backend': None,
            'ocr_char_count': 0,
            'ocr_word_count': 0,
            'ocr_line_count': 0,
            'ocr_number_count': 0,
            'ocr_equation_symbol_count': 0,
            'ocr_box_count': 0,
            'ocr_min_box_height': None,
            'ocr_median_box_height': None,
            'ocr_median_box_height_ratio': None,
            'ocr_small_text_share': None,
            'ocr_words_per_megapixel': None,
        }

    per_image = [ocr_engine.extract(image) for image in images]
    heights = [height for item in per_image for height in item['ocr_box_heights']]
    total_pixels = sum(image.width * image.height for image in images)
    word_count = sum(int(item['ocr_word_count']) for item in per_image)
    return {
        'ocr_available': True,
        'ocr_backend': 'pytesseract',
        'ocr_char_count': sum(int(item['ocr_char_count']) for item in per_image),
        'ocr_word_count': word_count,
        'ocr_line_count': sum(int(item['ocr_line_count']) for item in per_image),
        'ocr_number_count': sum(int(item['ocr_number_count']) for item in per_image),
        'ocr_equation_symbol_count': sum(int(item['ocr_equation_symbol_count']) for item in per_image),
        'ocr_box_count': len(heights),
        'ocr_min_box_height': min(heights) if heights else None,
        'ocr_median_box_height': float(np.median(heights)) if heights else None,
        'ocr_median_box_height_ratio': _median_box_height_ratio(heights, images),
        'ocr_small_text_share': _small_text_share(heights, images),
        'ocr_words_per_megapixel': word_count / max(total_pixels / 1_000_000.0, 1e-9) if total_pixels else None,
    }


def _ocr_feature_dict(text: str, words: Sequence[str], heights: Sequence[int], image_height: int) -> Dict[str, Any]:
    stripped_lines = [line for line in text.splitlines() if line.strip()]
    return {
        'ocr_char_count': len(text),
        'ocr_word_count': len(words),
        'ocr_line_count': len(stripped_lines),
        'ocr_number_count': len(re.findall(r'\b\d+(?:\.\d+)?\b', text)),
        'ocr_equation_symbol_count': len(re.findall(r'[=+\-*/^<>≤≥∑∫√π∞≈≠×÷]', text)),
        'ocr_box_heights': [int(height) for height in heights if int(height) > 0],
        'ocr_image_height': image_height,
    }


def _ocr_words_and_heights(data: Dict[str, Sequence[Any]]) -> Tuple[List[str], List[int]]:
    texts = data.get('text', [])
    heights = data.get('height', [])
    confs = data.get('conf', [])
    words = []
    word_heights = []
    for idx, text in enumerate(texts):
        word = str(text).strip()
        if not word:
            continue
        conf = _safe_float(confs[idx] if idx < len(confs) else None, default=0.0)
        if conf < 0:
            continue
        words.append(word)
        word_heights.append(int(_safe_float(heights[idx] if idx < len(heights) else 0, default=0.0)))
    return words, word_heights


def _median_box_height_ratio(heights: Sequence[int], images: Sequence[Image.Image]) -> Optional[float]:
    if not heights or not images:
        return None
    max_height = max(image.height for image in images)
    return float(np.median(heights) / max(max_height, 1))


def _small_text_share(heights: Sequence[int], images: Sequence[Image.Image]) -> Optional[float]:
    if not heights or not images:
        return None
    max_height = max(image.height for image in images)
    ratios = np.asarray(heights, dtype=float) / max(max_height, 1)
    return float((ratios < 0.035).mean())


def _rank_signal(values: Optional[pd.Series]) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    series = pd.to_numeric(values, errors='coerce').fillna(0.0).astype(float)
    if series.empty:
        return series
    if float(series.max()) == float(series.min()):
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return series.rank(method='average', pct=True)


def _small_text_signal(values: Optional[pd.Series]) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    ratios = pd.to_numeric(values, errors='coerce')
    signal = 1.0 - ratios.clip(lower=0.0, upper=1.0)
    return signal.fillna(0.0)


def _clip_novelty(embedding: Any, selected_rows: pd.DataFrame) -> float:
    current = _as_embedding(embedding)
    if current is None:
        return 0.0
    selected_embeddings = [_as_embedding(value) for value in selected_rows.get('clip_embedding', [])]
    selected_embeddings = [value for value in selected_embeddings if value is not None]
    if not selected_embeddings:
        return 1.0
    similarities = [float(np.dot(current, selected)) for selected in selected_embeddings]
    return max(0.0, 1.0 - max(similarities))


def _coverage_bonus(row: pd.Series, selected_image_types: set[str], selected_subjects: set[str]) -> float:
    image_types = set(parse_image_types(row.get('img_type')))
    image_bonus = 0.70 if not image_types.issubset(selected_image_types) else 0.0
    subject_bonus = 0.30 if str(row.get('subset')) not in selected_subjects else 0.0
    return image_bonus + subject_bonus


def _coverage_bonus_for_order(row: pd.Series, selected_rows: pd.DataFrame) -> float:
    selected_image_types = set()
    selected_subjects = set()
    for _, selected_row in selected_rows.iterrows():
        selected_image_types.update(parse_image_types(selected_row.get('img_type')))
        selected_subjects.add(str(selected_row.get('subset')))
    return _coverage_bonus(row, selected_image_types, selected_subjects)


def _as_embedding(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    embedding = np.asarray(value, dtype=float)
    if embedding.size == 0:
        return None
    norm = np.linalg.norm(embedding)
    if norm <= 0:
        return None
    return embedding / norm


def _sample_count(dataset_dict: DatasetDict) -> int:
    return sum(len(dataset) for _, dataset in dataset_dict.items())


def _resolve_visual_k(sample_count: int, prune_k: Optional[int], prune_ratio: Optional[float]) -> int:
    if sample_count <= 0:
        raise ValueError('Cannot prune an empty dataset.')
    if prune_k is not None:
        return max(1, min(sample_count, int(prune_k)))
    if prune_ratio is None:
        raise ValueError('visual_stress pruning requires prune_k or prune_ratio.')
    ratio = float(prune_ratio)
    if ratio <= 0 or ratio > 1:
        raise ValueError('prune_ratio must be in (0, 1].')
    return max(1, min(sample_count, int(math.ceil(sample_count * ratio))))


def _sample_source_id(sample: Sample, position: int) -> int:
    return int(sample.id if sample.id is not None else position)


def _image_type_stress(image_type: str) -> float:
    normalized = re.sub(r'\s+', ' ', str(image_type).strip().lower())
    return IMAGE_TYPE_STRESS_WEIGHTS.get(normalized, 0.50)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    return bool(value)


def _optional_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    return int(value)


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _report_export_frame(df: pd.DataFrame) -> pd.DataFrame:
    excluded = {'clip_embedding', 'image_data_uris'}
    columns = [column for column in df.columns if column not in excluded]
    return df[columns].copy()


def _json_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return [{key: _json_value(value) for key, value in row.items()} for row in df.to_dict(orient='records')]


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _coverage_summary(scored_df: pd.DataFrame, selected_df: pd.DataFrame) -> Dict[str, Any]:
    summary = {
        'full_sample_count': int(len(scored_df)),
        'selected_sample_count': int(len(selected_df)),
        'full_subject_count': int(scored_df['subset'].nunique()),
        'selected_subject_count': int(selected_df['subset'].nunique()),
        'full_image_type_count': int(scored_df['image_type_label'].nunique()),
        'selected_image_type_count': int(selected_df['image_type_label'].nunique()),
        'mean_full_visual_stress': float(scored_df['visual_stress_score'].mean()),
        'mean_selected_visual_stress': float(selected_df['visual_stress_score'].mean()),
        'full_subject_distribution': _string_value_counts(scored_df['subset']),
        'selected_subject_distribution': _string_value_counts(selected_df['subset']),
        'full_image_type_distribution': _string_value_counts(scored_df['image_type_label']),
        'selected_image_type_distribution': _string_value_counts(selected_df['image_type_label']),
        'full_topic_difficulty_distribution': _string_value_counts(scored_df['topic_difficulty']),
        'selected_topic_difficulty_distribution': _string_value_counts(selected_df['topic_difficulty']),
        'missing_subjects': sorted(set(scored_df['subset'].astype(str)) - set(selected_df['subset'].astype(str))),
        'missing_image_types': sorted(
            set(scored_df['image_type_label'].astype(str)) - set(selected_df['image_type_label'].astype(str))
        ),
    }
    summary.update(_clip_cluster_coverage(scored_df, selected_df))
    return summary


def _clip_cluster_coverage(scored_df: pd.DataFrame, selected_df: pd.DataFrame) -> Dict[str, Any]:
    embeddings = []
    row_indices = []
    for idx, value in scored_df.get('clip_embedding', pd.Series(dtype=object)).items():
        embedding = _as_embedding(value)
        if embedding is None:
            continue
        embeddings.append(embedding)
        row_indices.append(idx)

    if len(embeddings) < 2:
        return {}

    try:
        from sklearn.cluster import KMeans
    except Exception:
        return {}

    cluster_count = min(12, len(embeddings))
    labels = KMeans(n_clusters=cluster_count, random_state=0, n_init=10).fit_predict(np.vstack(embeddings))
    labels_by_index = {idx: int(label) for idx, label in zip(row_indices, labels)}
    full_labels = pd.Series(labels_by_index)
    selected_labels = pd.Series({
        idx: labels_by_index[idx] for idx in selected_df.index if idx in labels_by_index
    })
    if selected_labels.empty:
        return {
            'clip_cluster_count': cluster_count,
            'full_clip_cluster_distribution': _string_value_counts(full_labels),
            'selected_clip_cluster_distribution': {},
            'missing_clip_clusters': sorted(str(label) for label in set(full_labels.astype(str))),
        }
    return {
        'clip_cluster_count': cluster_count,
        'full_clip_cluster_distribution': _string_value_counts(full_labels),
        'selected_clip_cluster_distribution': _string_value_counts(selected_labels),
        'missing_clip_clusters': sorted(set(full_labels.astype(str)) - set(selected_labels.astype(str))),
    }


def _string_value_counts(series: pd.Series) -> Dict[str, int]:
    return {str(key): int(value) for key, value in series.fillna('unknown').astype(str).value_counts().items()}


def _html_preview(selected_df: pd.DataFrame) -> str:
    rows = []
    for _, row in selected_df.sort_values('selection_order').iterrows():
        image_html = ''.join(
            f'<img src="{html.escape(uri)}" style="max-width:220px;max-height:220px;margin:4px;border:1px solid #ddd">'
            for uri in row.get('image_data_uris', []) or []
        )
        rows.append(
            '<tr>'
            f'<td>{html.escape(str(row.get("selection_order")))}</td>'
            f'<td>{html.escape(str(row.get("subset")))}</td>'
            f'<td>{html.escape(str(row.get("source_id")))}</td>'
            f'<td>{html.escape(str(row.get("image_type_label")))}</td>'
            f'<td>{float(row.get("visual_stress_score", 0.0)):.3f}</td>'
            f'<td>{image_html}</td>'
            '</tr>'
        )
    return (
        '<!doctype html><html><head><meta charset="utf-8"><title>MMMU Visual Probe</title></head>'
        '<body><table border="1" cellspacing="0" cellpadding="6">'
        '<thead><tr><th>Order</th><th>Subject</th><th>Source ID</th><th>Image Type</th>'
        '<th>Stress</th><th>Images</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></body></html>'
    )
