"""
common/build_record.py

The schema at the center of the whole harness: every quantized build gets
tagged with what it IS (technique, bit-width, group size) and what it's
VALID for (hardware generation, workload). The build matrix is just a list
of these; "which build should I serve" is a filter over this list.

Design note: "valid" is not a single global flag. A build can be a strong
INT4 result on Hopper (native fast INT4 GEMM paths) but the *same* weights
might not be worth deploying on Ada if Ada's kernel stack makes INT8 faster
per-token than INT4 for this shape. So validity is computed per (hardware,
latency_budget_ms) query, not baked in at build time.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import time


SUPPORTED_HARDWARE = ["T4", "ADA", "HOPPER", "BLACKWELL"]


@dataclass
class QualityMetrics:
  perplexity: Optional[float] = None
  downstream_accuracy: Optional[float] = None
  # relative degradation vs the fp16 baseline on the SAME metric, filled in by the harness
  perplexity_delta_pct: Optional[float] = None
  downstream_delta_pct: Optional[float] = None


@dataclass
class PerfMetrics:
  tokens_per_sec: Optional[float] = None
  ttft_ms: Optional[float] = None            # time to first token
  peak_memory_gb: Optional[float] = None
  weight_memory_gb: Optional[float] = None


@dataclass
class BuildRecord:
  build_id: str
  model_name: str
  technique: str                # "fp16" | "int8_rtn" | "int4_rtn" | "int4_gptq" | "smoothquant_int8" | "int4_2to4_sparse"
  bits: int
  group_size: Optional[int]
  hardware_tag: str             # one of SUPPORTED_HARDWARE, the GPU family this build was benchmarked on
  workload_tag: str             # e.g. "tts_short_utterance" | "llm_chat_turn" | "general"
  quality: QualityMetrics = field(default_factory=QualityMetrics)
  perf: PerfMetrics = field(default_factory=PerfMetrics)
  passed_quality_gate: Optional[bool] = None
  quality_gate_reason: str = ""
  created_at: float = field(default_factory=time.time)
  notes: str = ""

  def to_dict(self):
    d = asdict(self)
    return d


class BuildMatrix:
  """A collection of BuildRecords plus the query logic: 'given hardware + a
  latency budget, which build should I serve'."""

  def __init__(self):
    self.records = []

  def add(self, record: BuildRecord):
    self.records.append(record)

  def apply_quality_gate(self, record: BuildRecord, max_ppl_delta_pct=5.0, max_downstream_delta_pct=2.0):
    reasons = []
    ok = True
    if record.quality.perplexity_delta_pct is not None and record.quality.perplexity_delta_pct > max_ppl_delta_pct:
      ok = False
      reasons.append(f"ppl delta {record.quality.perplexity_delta_pct:.2f}% > {max_ppl_delta_pct}%")
    if record.quality.downstream_delta_pct is not None and record.quality.downstream_delta_pct > max_downstream_delta_pct:
      ok = False
      reasons.append(f"downstream delta {record.quality.downstream_delta_pct:.2f}% > {max_downstream_delta_pct}%")
    record.passed_quality_gate = ok
    record.quality_gate_reason = "; ".join(reasons) if reasons else "passed"
    return ok

  def best_build_for(self, hardware_tag, latency_budget_ms=None, workload_tag=None, require_gate_pass=True):
    """query: which build should we serve, given hardware and a latency budget.
    picks the build with the smallest bit-width / footprint that still clears
    the quality gate and the latency budget, on this specific hardware."""
    candidates = [r for r in self.records if r.hardware_tag == hardware_tag]
    if workload_tag is not None:
      candidates = [r for r in candidates if r.workload_tag == workload_tag]
    if require_gate_pass:
      candidates = [r for r in candidates if r.passed_quality_gate]
    if latency_budget_ms is not None:
      candidates = [r for r in candidates if r.perf.ttft_ms is not None and r.perf.ttft_ms <= latency_budget_ms]
    if not candidates:
      return None
    # prefer highest tok/s among the ones that clear the bar
    candidates.sort(key=lambda r: (r.perf.tokens_per_sec or 0), reverse=True)
    return candidates[0]

  def compare_ranking_across_hardware(self, hw_a, hw_b, workload_tag=None):
    """research question: does the best build actually differ by hardware?
    returns the ranked technique order on each, so you can eyeball whether
    they agree."""
    def ranked(hw):
      cands = [r for r in self.records if r.hardware_tag == hw and r.passed_quality_gate]
      if workload_tag is not None:
        cands = [r for r in cands if r.workload_tag == workload_tag]
      cands.sort(key=lambda r: (r.perf.tokens_per_sec or 0), reverse=True)
      return [r.technique for r in cands]
    return {hw_a: ranked(hw_a), hw_b: ranked(hw_b)}

  def to_json(self, path):
    with open(path, "w") as f:
      json.dump([r.to_dict() for r in self.records], f, indent=2, default=str)

  def summary_table(self):
    rows = []
    header = f"{'build_id':<22}{'technique':<18}{'hw':<10}{'tok/s':>10}{'mem_gb':>10}{'ppl_delta%':>12}{'gate':>8}"
    rows.append(header)
    rows.append("-" * len(header))
    for r in self.records:
      tok_s = f"{r.perf.tokens_per_sec:.1f}" if r.perf.tokens_per_sec else "-"
      mem = f"{r.perf.peak_memory_gb:.2f}" if r.perf.peak_memory_gb else "-"
      ppl_d = f"{r.quality.perplexity_delta_pct:.2f}" if r.quality.perplexity_delta_pct is not None else "-"
      gate = "pass" if r.passed_quality_gate else ("fail" if r.passed_quality_gate is False else "-")
      rows.append(f"{r.build_id:<22}{r.technique:<18}{r.hardware_tag:<10}{tok_s:>10}{mem:>10}{ppl_d:>12}{gate:>8}")
    return "\n".join(rows)
