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
  perplexity:Optional[float] = None
  downstreamAccuracy:Optional[float] = None
  perplexityDeltaPct:Optional[float] = None
  downstreamDeltaPct:Optional[float] = None


@dataclass
class PerfMetrics:
  tokensPerSec:Optional[float] = None
  ttftMs:Optional[float] = None
  peakMemoryGb:Optional[float] = None
  weightMemoryGb:Optional[float] = None


@dataclass
class BuildRecord:
  buildId:str
  modelName:str
  technique:str
  bits:int
  groupSize:Optional[int]
  hardwareTag:str
  workloadTag:str
  quality:QualityMetrics = field(default_factory=QualityMetrics)
  perf:PerfMetrics = field(default_factory=PerfMetrics)
  passedQualityGate:Optional[bool] = None
  qualityGateReason:str = ""
  createdAt:float = field(default_factory=time.time)
  notes:str = ""

  def toDict(self):
    return asdict(self)


class BuildMatrix:
  """a collection of BuildRecords plus the query logic: 'given hardware + a
  latency budget, which build should I serve'."""

  def __init__(self):
    self.records = []

  def add(self, record):
    self.records.append(record)

  def applyQualityGate(self, record, maxPplDeltaPct=5.0, maxDownstreamDeltaPct=2.0):
    reasons = []
    ok = True
    if record.quality.perplexityDeltaPct is not None and record.quality.perplexityDeltaPct>maxPplDeltaPct:
      ok = False
      reasons.append(f"ppl delta {record.quality.perplexityDeltaPct:.2f}% > {maxPplDeltaPct}%")
    if record.quality.downstreamDeltaPct is not None and record.quality.downstreamDeltaPct>maxDownstreamDeltaPct:
      ok = False
      reasons.append(f"downstream delta {record.quality.downstreamDeltaPct:.2f}% > {maxDownstreamDeltaPct}%")
    record.passedQualityGate = ok
    record.qualityGateReason = "; ".join(reasons) if reasons else "passed"
    return ok

  def bestBuildFor(self, hardwareTag, latencyBudgetMs=None, workloadTag=None, requireGatePass=True):
    """query: which build should we serve, given hardware and a latency budget.
    picks the build with the highest tok/s among those that clear the
    quality gate and the latency budget, on this specific hardware."""
    candidates = [r for r in self.records if r.hardwareTag==hardwareTag]
    if workloadTag is not None:
      candidates = [r for r in candidates if r.workloadTag==workloadTag]
    if requireGatePass:
      candidates = [r for r in candidates if r.passedQualityGate]
    if latencyBudgetMs is not None:
      candidates = [r for r in candidates if r.perf.ttftMs is not None and r.perf.ttftMs<=latencyBudgetMs]
    if not candidates:
      return None
    candidates.sort(key=lambda r:(r.perf.tokensPerSec or 0), reverse=True)
    return candidates[0]

  def compareRankingAcrossHardware(self, hwA, hwB, workloadTag=None):
    """research question: does the best build actually differ by hardware?
    returns the ranked technique order on each, so you can eyeball whether
    they agree."""
    def ranked(hw):
      cands = [r for r in self.records if r.hardwareTag==hw and r.passedQualityGate]
      if workloadTag is not None:
        cands = [r for r in cands if r.workloadTag==workloadTag]
      cands.sort(key=lambda r:(r.perf.tokensPerSec or 0), reverse=True)
      return [r.technique for r in cands]
    return {hwA:ranked(hwA), hwB:ranked(hwB)}

  def toJson(self, path):
    with open(path, "w") as f:
      json.dump([r.toDict() for r in self.records], f, indent=2, default=str)

  def summaryTable(self):
    rows = []
    header = f"{'build_id':<22}{'technique':<18}{'hw':<10}{'tok/s':>10}{'mem_gb':>10}{'ppl_delta%':>12}{'gate':>8}"
    rows.append(header)
    rows.append("-"*len(header))
    for r in self.records:
      tokS = f"{r.perf.tokensPerSec:.1f}" if r.perf.tokensPerSec else "-"
      mem = f"{r.perf.peakMemoryGb:.2f}" if r.perf.peakMemoryGb else "-"
      pplD = f"{r.quality.perplexityDeltaPct:.2f}" if r.quality.perplexityDeltaPct is not None else "-"
      gate = "pass" if r.passedQualityGate else ("fail" if r.passedQualityGate is False else "-")
      rows.append(f"{r.buildId:<22}{r.technique:<18}{r.hardwareTag:<10}{tokS:>10}{mem:>10}{pplD:>12}{gate:>8}")
    return "\n".join(rows)
