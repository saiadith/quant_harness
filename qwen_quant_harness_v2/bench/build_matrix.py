"""
bench/build_matrix.py

Orchestrates the whole pipeline: for each (technique, bits) combination,
produce a quantized copy of the model, run it through the quality gate
(perplexity + downstream accuracy vs the fp16 baseline) and the latency
benchmark, and emit a BuildRecord. This is the "build matrix" itself -
run it once per hardware tag (once on a T4 session, once on whatever else
you can get access to) and merge the resulting JSON files to answer the
"does the best build differ by hardware" question.
"""

import copy
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from common.build_record import BuildRecord, BuildMatrix, QualityMetrics, PerfMetrics
from common.calibration import collectCalibrationStats, defaultCalibrationTexts
from quantizers.rtn import quantizeModelWeightsRtn
from quantizers.gptq_lite import quantizeModelWeightsGptq
from eval.perplexity import computePerplexity, defaultEvalText
from eval.downstream_task import evaluateMultipleChoice, defaultEvalSet
from bench.latency import benchmarkGeneration, modelWeightMemoryGb, detectHardwareTag


TECHNIQUES = {
  "fp16":              dict(quantizeFn=None, bits=16),
  "int8_rtn":          dict(quantizeFn="rtn", bits=8, groupSize=128),
  "int4_rtn":          dict(quantizeFn="rtn", bits=4, groupSize=128),
  "int4_gptq":         dict(quantizeFn="gptq", bits=4, groupSize=128),
  "int8_gptq":         dict(quantizeFn="gptq", bits=8, groupSize=128),
}


def buildAndEvaluate(baseModel, tokenizer, techniqueName, device="cuda", workloadTag="general",
                      evalText=None, evalSet=None, calibTexts=None,
                      baselinePpl=None, baselineAcc=None, maxGenTokens=64):
  """produces ONE build record for `techniqueName`. `baseModel` is copied,
  not mutated, so you can call this repeatedly against the same fp16
  checkpoint. pass baselinePpl/baselineAcc (from the fp16 build) so the
  quality gate can compute a relative delta rather than an absolute cutoff."""
  cfg = TECHNIQUES[techniqueName]
  evalText = evalText or defaultEvalText()
  evalSet = evalSet or defaultEvalSet()
  calibTexts = calibTexts or defaultCalibrationTexts()

  model = copy.deepcopy(baseModel).to(device)

  if cfg["quantizeFn"]=="rtn":
    quantizeModelWeightsRtn(model, bits=cfg["bits"], groupSize=cfg["groupSize"])
  elif cfg["quantizeFn"]=="gptq":
    calibStats = collectCalibrationStats(model, tokenizer, calibTexts, device=device)
    quantizeModelWeightsGptq(model, calibStats, bits=cfg["bits"], groupSize=cfg["groupSize"])
  # fp16: no-op, this IS the baseline

  ppl = computePerplexity(model, tokenizer, evalText, device=device)
  acc = evaluateMultipleChoice(model, tokenizer, evalSet, device=device)
  perf = benchmarkGeneration(model, tokenizer, "the weather today is", device=device, maxNewTokens=maxGenTokens)

  quality = QualityMetrics(perplexity=ppl, downstreamAccuracy=acc)
  if baselinePpl is not None:
    quality.perplexityDeltaPct = 100.0*(ppl-baselinePpl)/baselinePpl
  if baselineAcc is not None and baselineAcc>0:
    quality.downstreamDeltaPct = 100.0*(baselineAcc-acc)/baselineAcc   # positive = accuracy dropped

  record = BuildRecord(
    buildId=f"{techniqueName}_{detectHardwareTag()}",
    modelName=getattr(baseModel.config, "name_or_path", "unknown"),
    technique=techniqueName,
    bits=cfg["bits"],
    groupSize=cfg.get("groupSize"),
    hardwareTag=detectHardwareTag(),
    workloadTag=workloadTag,
    quality=quality,
    perf=PerfMetrics(
      tokensPerSec=perf["tokensPerSec"],
      ttftMs=perf["ttftMs"],
      peakMemoryGb=perf["peakMemoryGb"],
      weightMemoryGb=modelWeightMemoryGb(model),
    ),
  )

  del model
  if device.startswith("cuda"):
    import torch
    torch.cuda.empty_cache()

  return record


def runFullBuildMatrix(baseModel, tokenizer, device="cuda", workloadTag="general", techniques=None):
  """runs every technique in `techniques` (default: all of TECHNIQUES),
  computes the fp16 baseline first so every other build's quality delta is
  relative to it, applies the quality gate, and returns a populated
  BuildMatrix."""
  techniques = techniques or list(TECHNIQUES.keys())
  matrix = BuildMatrix()

  assert "fp16" in techniques, "fp16 baseline is required to compute quality deltas for everything else"
  baselineRecord = buildAndEvaluate(baseModel, tokenizer, "fp16", device=device, workloadTag=workloadTag)
  matrix.applyQualityGate(baselineRecord)
  matrix.add(baselineRecord)

  for name in techniques:
    if name=="fp16":
      continue
    record = buildAndEvaluate(
      baseModel, tokenizer, name, device=device, workloadTag=workloadTag,
      baselinePpl=baselineRecord.quality.perplexity,
      baselineAcc=baselineRecord.quality.downstreamAccuracy,
    )
    matrix.applyQualityGate(record)
    matrix.add(record)

  return matrix
