#!/usr/bin/env python3
"""Generate a standalone MadGraph library from a process JSON specification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import subprocess

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from madgraph_process_integrand import load_process_spec
from setup_madgraph_tth_benchmark import compile_libraries, inspect_library, resolve_executable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mg5", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    spec = load_process_spec(args.config)
    output = (args.output or Path(spec["standalone"])).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    mg5 = resolve_executable(args.mg5)
    output.parent.mkdir(parents=True, exist_ok=True)
    commands = (
        "set automatic_html_opening False\n"
        f"import model {spec.get('model', 'sm')}\n"
        f"generate {spec['process']}\n"
        f"output standalone {output} --prefix=int\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".mg5", delete=False) as handle:
        handle.write(commands)
        command_card = Path(handle.name)
    try:
        subprocess.run([mg5, str(command_card)], check=True)
    finally:
        command_card.unlink(missing_ok=True)

    metadata = {**spec, "config": str(args.config.resolve())}
    if not args.no_compile:
        libraries = compile_libraries(output)
        metadata["libraries"] = [str(item.resolve()) for item in libraries]
        metadata["madgraph_api"] = [inspect_library(output, item) for item in libraries]
    (output / "conditional_benchmark.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Generated {spec['name']} in {output}")


if __name__ == "__main__":
    main()
