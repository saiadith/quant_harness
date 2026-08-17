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
import gc
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from common.build_record import BuildRecord, BuildMatrix, QualityMetrics, PerfMetrics
from common.calibration import collect_calibration_stats, default_calibration_texts
from quantizers.rtn import quantize_model_weights_rtn_
from quantizers.gptq_lite import quantize_model_weights_gptq_
from eval.perplexity import compute_perplexity, default_eval_text
from eval.downstream_task import evaluate_multiple_choice, default_eval_set
from bench.latency import benchmark_generation, model_weight_memory_gb, detect_hardware_tag


TECHNIQUES = {
  "fp16":              dict(quantize_fn=None, bits=16),
  "int8_rtn":          dict(quantize_fn="rtn", bits=8, group_size=128),
  "int4_rtn":          dict(quantize_fn="rtn", bits=4, group_size=128),
  "int4_gptq":         dict(quantize_fn="gptq", bits=4, group_size=128),
  "int8_gptq":         dict(quantize_fn="gptq", bits=8, group_size=128),
}


def build_and_evaluate(base_model, tokenizer, technique_name, device="cuda", workload_tag="general",
                        eval_text=None, eval_set=None, calib_texts=None,
                        baseline_ppl=None, baseline_acc=None, max_gen_tokens=64):
  """produces ONE build record for `technique_name`. `base_model` is copied,
  not mutated, so you can call this repeatedly against the same fp16
  checkpoint. pass baseline_ppl/baseline_acc (from the fp16 build) so the
  quality gate can compute a relative delta rather than an absolute cutoff."""
  cfg = TECHNIQUES[technique_name]
  eval_text = eval_text or default_eval_text()
  eval_set = eval_set or default_eval_set()
  calib_texts = calib_texts or default_calibration_texts()

  model = copy.deepcopy(base_model).to(device)

  if cfg["quantize_fn"] == "rtn":
    quantize_model_weights_rtn_(model, bits=cfg["bits"], group_size=cfg["group_size"])
  elif cfg["quantize_fn"] == "gptq":
    calib_stats = collect_calibration_stats(model, tokenizer, calib_texts, device=device)
    quantize_model_weights_gptq_(model, calib_stats, bits=cfg["bits"], group_size=cfg["group_size"])
  # fp16: no-op, this IS the baseline

  ppl = compute_perplexity(model, tokenizer, eval_text, device=device)
  acc = evaluate_multiple_choice(model, tokenizer, eval_set, device=device)
  perf = benchmark_generation(model, tokenizer, "the weather today is", device=device, max_new_tokens=max_gen_tokens)

  quality = QualityMetrics(perplexity=ppl, downstream_accuracy=acc)
  if baseline_ppl is not None:
    quality.perplexity_delta_pct = 100.0 * (ppl - baseline_ppl) / baseline_ppl
  if baseline_acc is not None and baseline_acc > 0:
    quality.downstream_delta_pct = 100.0 * (baseline_acc - acc) / baseline_acc   # positive = accuracy dropped

  record = BuildRecord(
    build_id=f"{technique_name}_{detect_hardware_tag()}",
    model_name=getattr(base_model.config, "name_or_path", "unknown"),
    technique=technique_name,
    bits=cfg["bits"],
    group_size=cfg.get("group_size"),
    hardware_tag=detect_hardware_tag(),
    workload_tag=workload_tag,
    quality=quality,
    perf=PerfMetrics(
      tokens_per_sec=perf["tokens_per_sec"],
      ttft_ms=perf["ttft_ms"],
      peak_memory_gb=perf["peak_memory_gb"],
      weight_memory_gb=model_weight_memory_gb(model),
    ),
  )

  del model
  gc.collect()
  if device.startswith("cuda"):
    import torch
    torch.cuda.empty_cache()

  return record


def run_full_build_matrix(base_model, tokenizer, device="cuda", workload_tag="general", techniques=None):
  """runs every technique in `techniques` (default: all of TECHNIQUES),
  computes the fp16 baseline first so every other build's quality delta is
  relative to it, applies the quality gate, and returns a populated
  BuildMatrix."""
  techniques = techniques or list(TECHNIQUES.keys())
  matrix = BuildMatrix()

  assert "fp16" in techniques, "fp16 baseline is required to compute quality deltas for everything else"
  baseline_record = build_and_evaluate(base_model, tokenizer, "fp16", device=device, workload_tag=workload_tag)
  matrix.apply_quality_gate(baseline_record)
  matrix.add(baseline_record)
  gc.collect()
  if device.startswith("cuda"):
    import torch
    torch.cuda.empty_cache()

  for name in techniques:
    if name == "fp16":
      continue
    record = build_and_evaluate(
      base_model, tokenizer, name, device=device, workload_tag=workload_tag,
      baseline_ppl=baseline_record.quality.perplexity,
      baseline_acc=baseline_record.quality.downstream_accuracy,
    )
    matrix.apply_quality_gate(record)
    matrix.add(record)
    gc.collect()
    if device.startswith("cuda"):
      import torch
      torch.cuda.empty_cache()

  return matrix
