"""Import provider code with dependencies that forbid model loading/generation."""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch


def load_provider():
    mlx = ModuleType('mlx_lm')
    mlx.load = Mock(side_effect=AssertionError('Model load is forbidden in portable fixtures'))
    mlx.generate = Mock(side_effect=AssertionError('Model generation is forbidden in portable fixtures'))
    sampling = ModuleType('mlx_lm.sample_utils')
    sampling.make_sampler = Mock(side_effect=AssertionError('Native sampler is forbidden in portable fixtures'))
    source = Path(__file__).resolve().parents[1] / 'openclaw_capability' / 'sacred_harp_openclaw_server.py'
    spec = importlib.util.spec_from_file_location('portable_sacred_harp_provider', source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'mlx_lm': mlx, 'mlx_lm.sample_utils': sampling}):
        spec.loader.exec_module(module)
    return module
