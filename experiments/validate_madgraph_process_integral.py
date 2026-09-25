#!/usr/bin/env python3
"""Held-out NF and randomized-Sobol validation for a configured process."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp
from scipy.stats import qmc

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import run_adaptive_integrand_structure_search as adaptive
from madgraph_process_integrand import ConfiguredMadGraphIntegrand, load_process_spec


GEV2_TO_PB = 3.893793721e8
COND_DIM = 3


def condition_features(z: float) -> np.ndarray:
    return np.array([z, z * z, math.log1p(z) / math.log(2.0)], dtype=np.float32)


def stable_mean(log_values: np.ndarray) -> float:
    finite = np.isfinite(log_values)
    if not np.any(finite):
        return 0.0
    return math.exp(float(logsumexp(log_values[finite]) - math.log(len(log_values))))


def load_model(checkpoint_path: Path, dimension: int, device: torch.device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    adaptive.DIM = dimension
    model = adaptive.make_real_model(checkpoint["graph"], None, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def sobol_replicates(evaluator, energy: float, args):
    estimates = []
    rows = []
    points = 1 << args.sobol_power
    for replicate in range(args.sobol_replicates):
        started = time.perf_counter()
        engine = qmc.Sobol(
            d=evaluator.dimension, scramble=True,
            seed=args.seed + int(round(energy)) * 100 + replicate,
        )
        unit = engine.random_base2(args.sobol_power)
        energies = np.full(points, energy, dtype=np.float64)
        estimate = stable_mean(evaluator.log_integrand(unit, energies))
        estimates.append(estimate)
        rows.append({
            "energy_gev": energy, "method": "sobol_qmc", "replicate": replicate,
            "evaluations": points, "estimate_gev2": estimate,
            "estimate_pb": estimate * GEV2_TO_PB,
            "seconds": time.perf_counter() - started,
        })
    values = np.asarray(estimates)
    mean = float(values.mean())
    stderr = float(values.std(ddof=1) / math.sqrt(len(values)))
    return rows, {
        "energy_gev": energy, "method": "sobol_qmc",
        "estimate_gev2": mean, "stderr_gev2": stderr,
        "estimate_pb": mean * GEV2_TO_PB, "stderr_pb": stderr * GEV2_TO_PB,
        "relative_stderr": stderr / abs(mean), "normalized_ess": math.nan,
    }


@torch.no_grad()
def nf_estimate(model, evaluator, energy: float, minimum: float, maximum: float, args, device):
    z = (energy - minimum) / (maximum - minimum)
    feature = torch.as_tensor(condition_features(z), device=device).reshape(1, COND_DIM)
    log_weights = []
    remaining = args.nf_evals
    started = time.perf_counter()
    while remaining:
        count = min(remaining, args.gpu_chunk)
        cond = feature.expand(count, COND_DIM)
        unit, logq = adaptive.sample_from_proposal(
            proposal_model=model, n=count, cond=cond, device=device,
            uniform_fraction=args.uniform_fraction,
        )
        unit_numpy = unit.detach().cpu().double().numpy()
        energies = np.full(count, energy, dtype=np.float64)
        logf = evaluator.log_integrand(unit_numpy, energies)
        log_weights.append(logf - logq.detach().cpu().double().numpy())
        remaining -= count
    logw = np.concatenate(log_weights)
    log_sum = float(logsumexp(logw))
    log_sum2 = float(logsumexp(2.0 * logw))
    count = len(logw)
    estimate = math.exp(log_sum - math.log(count))
    second = math.exp(log_sum2 - math.log(count))
    variance = max(0.0, second - estimate * estimate)
    stderr = math.sqrt(variance / count)
    ess = math.exp(2.0 * log_sum - math.log(count) - log_sum2)
    return {
        "energy_gev": energy, "method": "conditional_nf", "replicate": 0,
        "evaluations": count, "estimate_gev2": estimate,
        "stderr_gev2": stderr, "estimate_pb": estimate * GEV2_TO_PB,
        "stderr_pb": stderr * GEV2_TO_PB,
        "relative_stderr": stderr / abs(estimate), "normalized_ess": ess,
        "seconds": time.perf_counter() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--energies", type=float, nargs="+", default=None)
    parser.add_argument("--sobol-power", type=int, default=18)
    parser.add_argument("--sobol-replicates", type=int, default=8)
    parser.add_argument("--nf-evals", type=int, default=500_000)
    parser.add_argument("--gpu-chunk", type=int, default=10_000)
    parser.add_argument("--uniform-fraction", type=float, default=0.10)
    parser.add_argument("--alpha-s", type=float, default=0.118)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    spec = load_process_spec(args.config)
    evaluator = ConfiguredMadGraphIntegrand(spec, PROJECT, args.alpha_s)
    condition = spec["condition"]
    minimum, maximum = float(condition["minimum"]), float(condition["maximum"])
    energies = args.energies or [float(item) for item in condition["held_out"]]
    if any(item < minimum or item > maximum for item in energies):
        raise ValueError("Every validation energy must lie inside the condition range")
    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, evaluator.dimension, device)
    expected = f"madgraph_{spec['name']}_{evaluator.dimension}"
    if checkpoint.get("benchmark") != expected:
        raise ValueError(
            f"Checkpoint benchmark {checkpoint.get('benchmark')!r}; expected {expected!r}"
        )

    detail_rows, summary_rows = [], []
    for energy in energies:
        sobol_rows, sobol_summary = sobol_replicates(evaluator, energy, args)
        nf = nf_estimate(model, evaluator, energy, minimum, maximum, args, device)
        detail_rows.extend(sobol_rows)
        detail_rows.append(nf)
        summary_rows.extend((sobol_summary, nf))
        combined = math.hypot(sobol_summary["stderr_pb"], nf["stderr_pb"])
        pull = (nf["estimate_pb"] - sobol_summary["estimate_pb"]) / combined
        print(
            f"{energy:g} GeV: NF={nf['estimate_pb']:.8g} +/- {nf['stderr_pb']:.2g} pb "
            f"ESS={nf['normalized_ess']:.5f}; Sobol={sobol_summary['estimate_pb']:.8g} "
            f"+/- {sobol_summary['stderr_pb']:.2g} pb; pull={pull:.2f}", flush=True,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(detail_rows).to_csv(args.output / "replicate_results.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(args.output / "summary.csv", index=False)
    config = vars(args).copy()
    config.update({key: str(value) for key, value in config.items() if isinstance(value, Path)})
    config["resolved_energies"] = energies
    (args.output / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
