"""Prompt construction and native JSON compatibility require no device runtime."""
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_offline(tmp_path, source):
    # A fresh process also isolates native top-level package names from parent
    # repository tests. Fail any accidental device-package or LLM-client import.
    bootstrap = f"""
import importlib
import importlib.abc
from pathlib import Path
import sys
sys.path.insert(0, {str(ROOT)!r})
class NoRuntimeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'torch', 'triton', 'torch_npu', 'openai'}}:
            raise AssertionError('Prompt generation imported ' + fullname)
sys.meta_path.insert(0, NoRuntimeImports())
"""
    result = subprocess.run([sys.executable, "-c", bootstrap + source], cwd=tmp_path,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("language", ["triton_metax", "maca"])
def test_strategy_registers_independently_and_reads_target_from_any_cwd(tmp_path, language):
    run_offline(tmp_path, f"""
importlib.import_module('prompt_generators.{language}_add_shot')
from prompt_generators.prompt_registry import PROMPT_REGISTRY
from dataset import dataset
assert set(PROMPT_REGISTRY) == {{{language!r}}}
strategy = PROMPT_REGISTRY[{language!r}]['add_shot']
prompt = strategy.generate('relu')
root = Path({str(ROOT)!r})
reference = root / 'reference' / dataset['relu']['category'] / 'relu.py'
example = root / 'reference' / dataset['add_bias_broadcast']['category'] / 'add_bias_broadcast.py'
assert reference.read_text().strip() in prompt
assert example.read_text().strip() in prompt
assert 'problem `relu`' in prompt
if {language!r} == 'triton_metax':
    assert (root / 'examples/metax/add_bias_triton.py').read_text().strip() in prompt
    assert 'NVIDIA PTX' in prompt and 'ModelNew' in prompt
""")


def test_maca_prompt_embeds_a_complete_native_parseable_project(tmp_path):
    run_offline(tmp_path, r"""
import json
import re
from backends.maca_compiler import parse_project
from prompt_generators.maca_add_shot import MacaAddShotPromptStrategy
from prompt_generators.metax_prompt_utils import ROOT
from scripts.maca_project_to_json import package_project
prompt = MacaAddShotPromptStrategy().generate('add_bias_broadcast')
blocks = re.findall(r'```json\n(.*?)\n```', prompt, re.DOTALL)
assert len(blocks) == 1
assert json.loads(blocks[0]) == package_project(ROOT / 'examples/metax/add_bias_maca')
project = parse_project(blocks[0])
assert project.model_path == Path('ModelNew.py')
assert project.model_name == 'ModelNew'
assert project.kernels == [Path('kernel.cpp')]
assert project.bindings == [Path('binding.cpp')]
assert project.module_name == 'maca_add_bias'
model = project.sources[project.model_path]
assert '_maca_extension' in model and '_maca_module_name' in model
assert 'current_stream(x.device).cuda_stream' in model
compile(model, 'ModelNew.py', 'exec')
""")


def test_generate_prompt_discovers_both_strategies_without_client_calls(tmp_path):
    run_offline(tmp_path, """
import generate_and_write
def no_client(*args, **kwargs):
    raise AssertionError('Offline prompt generation requested an LLM client')
generate_and_write.get_client = no_client
from prompt_generators.prompt_registry import PROMPT_REGISTRY
assert not PROMPT_REGISTRY
for language in ('triton_metax', 'maca'):
    prompt = generate_and_write.generate_prompt(language, 'add_shot', 'add_bias_broadcast')
    assert 'problem `add_bias_broadcast`' in prompt
assert set(PROMPT_REGISTRY) == {'triton_metax', 'maca'}
""")
