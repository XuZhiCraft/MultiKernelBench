"""Measure PyTorch references, optionally selecting one operator."""
import argparse
import importlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import statistics

from config import project_root_path
from dataset import dataset


def run_op(op, language, return_dict):
    backend = None
    try:
        from backends.backend_registry import BACKEND_REGISTRY
        from config import seed_num
        from utils.correctness import set_seed
        from utils.utils import get_ref_src_path

        if language not in BACKEND_REGISTRY:
            importlib.import_module(f"backends.{language}_backend")
        backend = BACKEND_REGISTRY.get(language)
        if backend is None:
            raise ValueError(f"Unsupported language/platform: {language}")
        reference = Path(get_ref_src_path(op)).resolve()
        backend.context = {"__name__": "__baseline__", "__file__": str(reference)}
        set_seed(seed_num)
        exec(compile(reference.read_text(encoding="utf-8"), str(reference), "exec"),
             backend.context)
        elapsed_times = backend.time_execution("Model")
        if not elapsed_times or any(not math.isfinite(value) or value <= 0 for value in elapsed_times):
            raise ValueError("Reference timing must contain finite, positive samples")
        rounded = lambda value: float(f"{value:.3g}")
        return_dict[op] = {
            "mean": rounded(statistics.mean(elapsed_times)),
            "std": rounded(statistics.pstdev(elapsed_times)),
            "min": rounded(min(elapsed_times)),
            "max": rounded(max(elapsed_times)),
            "num_trials": len(elapsed_times),
            "device": backend.get_hardware_name(),
        }
    except Exception as error:
        print(f"[ERROR] {op}: {error}", flush=True)
        return_dict[op] = "not supported"
    finally:
        if backend is not None:
            try:
                backend.cleanup()
            except Exception as error:
                print(f"[WARNING] {op} cleanup: {error}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", default="cuda", help="Device backend used to time PyTorch Model.")
    parser.add_argument("--op", help="Collect only this operator, e.g. add_bias_broadcast.")
    parser.add_argument("--output", type=Path, help="Write the baseline JSON to this path.")
    args = parser.parse_args(argv)
    if args.op is not None and args.op not in dataset:
        parser.error(f"Unknown operator: {args.op}")

    # Match the standalone evaluator: prepare MACA before importing device packages.
    from utils.metax_environment import prepare_process

    prepare_process(args.language)
    operators = [args.op] if args.op is not None else list(dataset)
    result = {}
    device_name = "unknown_device"
    for op in operators:
        print(f"[INFO] {op}", flush=True)
        with mp.Manager() as manager:
            return_dict = manager.dict()
            process = mp.Process(target=run_op, args=(op, args.language, return_dict))
            process.start()
            try:
                process.join(timeout=120)
                if process.is_alive():
                    print(f"[TIMEOUT] {op}", flush=True)
                    result[op] = "timeout"
                else:
                    result[op] = return_dict.get(op, "not supported")
            finally:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join()
                process.close()
        if isinstance(result[op], dict):
            device_name = result[op]["device"]

    suffix = f"_{args.op}" if args.op is not None else ""
    output = args.output or (Path(project_root_path) / "baselines" /
                             f"{args.language}_{device_name}{suffix}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] Write to {output}", flush=True)
    return 0 if all(isinstance(item, dict) for item in result.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
