#!/usr/bin/env python3
"""Train the established conditional copula workflow for a configured process."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import run_adaptive_integrand_structure_search as baseline
from madgraph_process_integrand import ConfiguredMadGraphIntegrand, load_process_spec


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--alpha-s", type=float, default=0.118)
    parser.add_argument("--me-chunk", type=int, default=512)
    process_args, remaining = parser.parse_known_args()

    spec = load_process_spec(process_args.config)
    evaluator = ConfiguredMadGraphIntegrand(spec, PROJECT, process_args.alpha_s)
    condition = spec["condition"]
    minimum = float(condition["minimum"])
    maximum = float(condition["maximum"])
    if minimum <= evaluator.phase_space.minimum_energy:
        raise ValueError(
            f"Condition minimum must exceed threshold "
            f"{evaluator.phase_space.minimum_energy:g} GeV"
        )
    benchmark = f"madgraph_{spec['name']}_{evaluator.dimension}"

    def sample_conditions(n, device):
        z = torch.rand(n, 1, device=device)
        return torch.cat((z, z * z, torch.log1p(z) / np.log(2.0)), dim=1)

    def target_log_integrand(name, unit, cond):
        if name != benchmark:
            raise ValueError(f"Unexpected benchmark {name!r}")
        z = cond[:, 0].detach().cpu().double().numpy()
        energy = minimum + (maximum - minimum) * z
        unit_numpy = unit.detach().cpu().double().numpy()
        pieces = []
        for start in range(0, len(unit_numpy), process_args.me_chunk):
            stop = min(start + process_args.me_chunk, len(unit_numpy))
            pieces.append(evaluator.log_integrand(unit_numpy[start:stop], energy[start:stop]))
        return torch.as_tensor(np.concatenate(pieces), device=unit.device, dtype=unit.dtype)

    baseline.SUPPORTED_BENCHMARKS = {benchmark}
    baseline.BENCHMARK_DIMENSIONS = {benchmark: evaluator.dimension}
    baseline.sample_conditions = sample_conditions
    baseline.target_log_integrand = target_log_integrand

    filtered = []
    skip = False
    for value in remaining:
        if skip:
            skip = False
            continue
        if value in ("--benchmarks", "--max-rounds"):
            skip = True
            continue
        if value.startswith(("--benchmarks=", "--max-rounds=")):
            continue
        if value in ("--compile-kernels", "--compile-model"):
            continue
        filtered.append(value)
    if not any(item == "--output" or item.startswith("--output=") for item in filtered):
        filtered.extend(["--output", f"results/{spec['name']}_conditional_copula"])
    filtered.extend([
        "--benchmarks", benchmark, "--max-rounds", "0",
        "--no-compile-kernels", "--no-compile-model",
    ])
    sys.argv = [sys.argv[0], *filtered]
    print(f"Configured process       = {spec['name']}")
    print(f"MadGraph process         = {spec['process']}")
    print(f"dimension                = {evaluator.dimension}")
    print(f"sqrt(s_hat) range        = [{minimum:g}, {maximum:g}] GeV")
    baseline.main()


if __name__ == "__main__":
    main()
