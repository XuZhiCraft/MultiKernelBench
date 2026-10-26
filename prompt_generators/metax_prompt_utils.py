"""Read MetaX prompt examples without importing accelerator or LLM packages."""
from pathlib import Path

from dataset import dataset


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_OP = "add_bias_broadcast"


def reference_source(op):
    category = dataset[op]["category"]
    return (ROOT / "reference" / category / f"{op}.py").read_text(encoding="utf-8")


def code_block(source, language):
    return f"```{language}\n{source.rstrip()}\n```"
