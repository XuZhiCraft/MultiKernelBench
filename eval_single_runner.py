import argparse
import json
import math
import os
from pathlib import Path

from dataset import dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one generated response for one MultiKernelBench task."
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_path",
        required=True,
        help="Path to the generated response text file.",
    )
    parser.add_argument(
        "-o",
        "--op",
        dest="op",
        required=True,
        help="Task/operator name, such as relu or 1_logical_and.",
    )
    parser.add_argument(
        "-l",
        "--language",
        dest="language",
        required=True,
        help="Backend name, such as cuda, ascendc_direct_launch, triton_metax, or maca.",
    )
    parser.add_argument(
        "-r",
        "--result",
        dest="result_path",
        required=True,
        help="Path where the JSON result should be written.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation for the result file. Defaults to 2.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    # Dynamic libraries must be selected at interpreter startup, before the
    # evaluation helpers import torch. Existing platforms do not re-exec.
    from utils.metax_environment import METAX_LANGUAGES, prepare_process
    input_path = Path(args.input_path)
    result_path = Path(args.result_path)
    metax = args.language in METAX_LANGUAGES

    if result_path.parent.exists() and not result_path.parent.is_dir():
        raise FileExistsError(
            f"Result parent path exists but is not a directory: {result_path.parent}"
        )

    if metax:
        if input_path.resolve() == result_path.resolve():
            raise ValueError("Input and result paths must be different")
        # A failed new run cannot leave a previous successful MetaX result.
        result_path.unlink(missing_ok=True)
    try:
        prepare_process(args.language)
        if args.op not in dataset:
            raise ValueError(f"Unknown op: {args.op}")
        if not input_path.is_file():
            raise FileNotFoundError(f"Generated response file not found: {input_path}")
        from utils.evaluation_utils import eval_single

        response_txt = input_path.read_text(encoding="utf-8")
        result = eval_single(response_txt, args.op, args.language)
    except Exception as error:
        if not metax:
            raise
        result = {"compiled": False, "correctness": False, "performance": None,
                  "hardware": None, "cheating": False,
                  "error": f"{type(error).__name__}: {error}"}

    result_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=args.indent, ensure_ascii=False) + "\n"
    if not metax:
        # Retain the existing CLI's result format and exit semantics.
        result_path.write_text(text, encoding="utf-8")
        return
    temporary = result_path.with_name(f".{result_path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(result_path)
    finally:
        temporary.unlink(missing_ok=True)
    performance = result.get("performance") or {}
    mean = performance.get("mean")
    passed = (result.get("compiled") is True and result.get("correctness") is True
              and not result.get("cheating") and type(mean) in (float, int)
              and math.isfinite(mean) and mean > 0
              and type(performance.get("num_trials")) is int and performance["num_trials"] > 0)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
