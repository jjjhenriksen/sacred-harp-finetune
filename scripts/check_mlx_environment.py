#!/usr/bin/env python3
"""Check installed native runtime and CLI entry points without loading models."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_environment() -> dict:
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise RuntimeError('Use native arm64 Python on Apple Silicon macOS for this runtime')
    if sys.version_info < (3, 11) or int(platform.mac_ver()[0].split('.')[0]) < 14:
        raise RuntimeError('Use Python >=3.11 and macOS >=14 for the pinned MLX runtime')
    expected = {'mlx': '0.32.3', 'mlx-lm': '0.32.0'}
    versions = {name: importlib.metadata.version(name) for name in expected}
    if versions != expected:
        raise RuntimeError(f'Install requirements-mlx.txt; expected {expected}, found {versions}')
    # Fail closed for any Hub access even if a future import/help entry changes.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    with tempfile.TemporaryDirectory(prefix='sacred-harp-import-check-') as cache:
        os.environ['HF_HOME'] = cache
        imports = ['mlx.core', 'mlx_lm', 'mlx_lm.lora', 'mlx_lm.sample_utils', 'datasets']
        for name in imports:
            importlib.import_module(name)
        source = ROOT / 'openclaw_capability' / 'sacred_harp_openclaw_server.py'
        spec = importlib.util.spec_from_file_location('sacred_harp_install_check', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cli_commands = [
            [sys.executable, '-m', 'mlx_lm.lora', '--help'],
            [sys.executable, '-m', 'mlx_lm', 'lora', '--help'],
            [sys.executable, '-m', 'mlx_lm.generate', '--help'],
            [sys.executable, str(ROOT / 'prepare_dataset.py'), '--help'],
            [sys.executable, str(source), '--help'],
        ]
        checks = []
        for command in cli_commands:
            result = subprocess.run(command, check=True, capture_output=True, text=True,
                                    env=os.environ.copy(), timeout=60, cwd=ROOT)
            if 'usage:' not in result.stdout.lower():
                raise RuntimeError(f'CLI did not print help: {command}')
            checks.append({'command': [Path(command[0]).name, *command[1:]], 'exit_code': result.returncode, 'help': True})
        cache_files = list(Path(cache).rglob('*'))
        if any(path.is_file() and path.stat().st_size for path in cache_files):
            raise RuntimeError('Unexpected cache files during import/help-only check')
    return {'python': platform.python_version(), 'macos': platform.mac_ver()[0],
            'architecture': platform.machine(), 'versions': versions,
            'imports': [*imports, source.name], 'cli_checks': checks,
            'hub_offline': True, 'model_cache_files': 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional JSON evidence output')
    args = parser.parse_args()
    report = json.dumps(check_environment(), indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    print(report, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
