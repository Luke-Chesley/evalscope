# Copyright (c) Alibaba, Inc. and its affiliates.
"""CLI command for loading benchmark datasets without running model inference."""

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List

from evalscope.cli.base import CLICommand
from evalscope.config import TaskConfig
from evalscope.utils.logger import get_logger

logger = get_logger()


class DatasetCMD(CLICommand):
    name = 'dataset'

    def __init__(self, args: Namespace):
        self.args = args

    @staticmethod
    def define_args(parsers: ArgumentParser):
        parser = parsers.add_parser(
            DatasetCMD.name,
            help='Load and inspect a benchmark dataset without running model inference.',
        )
        parser.add_argument('--datasets', type=str, nargs='+', required=True, help='Benchmark dataset name(s).')
        parser.add_argument('--dataset-args', type=json.loads, default='{}', help='Dataset args as a JSON string.')
        parser.add_argument('--dataset-dir', help='Dataset cache directory.')
        parser.add_argument('--dataset-hub', help='Dataset hub, e.g. modelscope, huggingface, local.')
        parser.add_argument('--limit', type=float, default=None, help='Max samples to load per subset before pruning.')
        parser.add_argument('--repeats', type=int, default=1, help='Number of times to repeat loaded samples.')
        parser.add_argument(
            '--dump-jsonl',
            type=str,
            default=None,
            help='Optional path to write loaded sample ids/metadata.',
        )
        parser.add_argument(
            '--preview-chars',
            type=int,
            default=160,
            help='Characters of input preview to include when dumping JSONL.',
        )
        parser.set_defaults(func=DatasetCMD)

    def execute(self):
        from evalscope.api.messages import messages_to_markdown
        from evalscope.api.registry import get_benchmark

        task_config_kwargs = {
            'datasets': self.args.datasets,
            'dataset_args': self.args.dataset_args,
            'limit': self.args.limit,
            'repeats': self.args.repeats,
        }
        if self.args.dataset_dir is not None:
            task_config_kwargs['dataset_dir'] = self.args.dataset_dir
        if self.args.dataset_hub is not None:
            task_config_kwargs['dataset_hub'] = self.args.dataset_hub
        task_config = TaskConfig(**task_config_kwargs)

        rows: List[Dict[str, Any]] = []
        total = 0
        for dataset_name in task_config.datasets:
            benchmark = get_benchmark(dataset_name, task_config)
            dataset_dict = benchmark.load_dataset()
            print(f'\nDataset: {dataset_name}')
            for subset, dataset in dataset_dict.items():
                count = len(dataset)
                total += count
                ids = [sample.id for sample in dataset]
                print(f'  subset={subset} count={count} ids={ids[:10]}{"..." if len(ids) > 10 else ""}')

                if self.args.dump_jsonl:
                    for sample in dataset:
                        input_text = (
                            sample.input if isinstance(sample.input, str) else messages_to_markdown(sample.input)
                        )
                        rows.append(
                            {
                                'dataset': dataset_name,
                                'subset': subset,
                                'id': sample.id,
                                'group_id': sample.group_id,
                                'target': sample.target,
                                'metadata': sample.metadata,
                                'input_preview': input_text[:self.args.preview_chars],
                            }
                        )

        print(f'\nTotal loaded samples: {total}')

        if self.args.dump_jsonl:
            path = Path(self.args.dump_jsonl)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('w', encoding='utf-8') as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
            logger.info(f'Wrote dataset inspection rows to {path}')
