"""Evaluator — the truth gate of the optimization loop.

Pipeline (per (template, point)):
  materialize -> compile (LayerOracle) -> correctness oracle (vs baseline)
  -> measure harness (warmup + N runs + noise floor) -> MeasureSample
"""

from .cpu_runner import CpuRunner
from .vk_runner import VkRunner
from .correctness_oracle import CorrectnessOracle
from .measure_harness import MeasureHarness
from .evaluator import Evaluator

__all__ = ["CpuRunner", "VkRunner", "CorrectnessOracle", "MeasureHarness", "Evaluator"]
