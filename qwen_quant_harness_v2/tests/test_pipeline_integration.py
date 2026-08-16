"""
tests/test_pipeline_integration.py

No internet access to huggingface.co in this sandbox, so this can't
download real Qwen weights. Instead: build a tiny randomly-initialized
llama-architecture model (transformers can construct this from a Config
with zero network access) plus a minimal hand-rolled character-level
tokenizer that satisfies the small interface our eval/bench code expects
(__call__ -> .input_ids, .eos_token_id, .decode). This validates that
calibration hooking, rtn/gptq quantization, perplexity, downstream scoring,
latency benchmarking, and the build-matrix orchestrator all wire together
correctly end to end - the plumbing, not the model quality - before ever
touching a real checkpoint on kaggle.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from common.calibration import collectCalibrationStats, defaultCalibrationTexts
from quantizers.rtn import quantizeModelWeightsRtn
from quantizers.gptq_lite import quantizeModelWeightsGptq
from eval.perplexity import computePerplexity
from eval.downstream_task import evaluateMultipleChoice
from bench.latency import benchmarkGeneration, modelWeightMemoryGb
from bench.build_matrix import buildAndEvaluate, runFullBuildMatrix


class TokenizerOutput(dict):
  """dict subclass so `model(**enc)` unpacking works, plus attribute access
  and a no-op `.to(device)` to match the real hf BatchEncoding interface."""

  def __getattr__(self, name):
    try:
      return self[name]
    except KeyError:
      raise AttributeError(name)

  def to(self, device):
    return TokenizerOutput({k:v.to(device) for k, v in self.items()})


class CharTokenizer:
  """minimal stand-in exposing just enough of the hf tokenizer interface
  for our eval/bench code to run - no network access needed."""

  def __init__(self, textCorpus):
    chars = sorted(set("".join(textCorpus))|set(" abcdefghijklmnopqrstuvwxyz.,"))
    self.stoi = {c:i for i, c in enumerate(chars)}
    self.itos = {i:c for i, c in enumerate(chars)}
    self.vocabSize = len(chars)
    self.eos_token_id = 0

  def __call__(self, text, return_tensors=None, truncation=False, max_length=None):
    ids = [self.stoi.get(c, 0) for c in text]
    if max_length:
      ids = ids[:max_length]
    return TokenizerOutput(input_ids=torch.tensor([ids]))

  def decode(self, ids):
    return "".join(self.itos.get(int(i), "") for i in ids)


def makeTinyModelAndTokenizer():
  corpus = defaultCalibrationTexts()
  tok = CharTokenizer(corpus)
  config = LlamaConfig(
    vocab_size=tok.vocabSize, max_position_embeddings=128, hidden_size=32,
    intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
    bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
  )
  model = LlamaForCausalLM(config)
  model.config.name_or_path = "tiny-test-llama"
  return model, tok


def test_calibrationAndRtnQuantizeRun():
  model, tok = makeTinyModelAndTokenizer()
  stats = collectCalibrationStats(model, tok, defaultCalibrationTexts()[:4], device="cpu", maxLength=64)
  assert len(stats)>0, "should have hooked at least one linear layer"
  n = quantizeModelWeightsRtn(model, bits=4, groupSize=16)
  assert n>0
  print("test_calibrationAndRtnQuantizeRun: pass, quantized", n, "layers")


def test_gptqQuantizeRunsOnRealModelShapes():
  model, tok = makeTinyModelAndTokenizer()
  stats = collectCalibrationStats(model, tok, defaultCalibrationTexts(), device="cpu", maxLength=64)
  n = quantizeModelWeightsGptq(model, stats, bits=4, groupSize=16)
  assert n>0
  for p in model.parameters():
    assert torch.isfinite(p).all(), "gptq quantization produced non-finite weights"
  print("test_gptqQuantizeRunsOnRealModelShapes: pass, quantized", n, "layers")


def test_perplexityRuns():
  model, tok = makeTinyModelAndTokenizer()
  ppl = computePerplexity(model, tok, "the quick brown fox jumps over the lazy dog. "*10, device="cpu", maxLength=64, stride=32)
  assert ppl>0 and torch.isfinite(torch.tensor(ppl))
  print("test_perplexityRuns: pass, ppl=", ppl)


def test_downstreamEvalRuns():
  model, tok = makeTinyModelAndTokenizer()
  evalSet = [
    {"prompt":"the sky is ", "choices":["blue.", "loud."], "answer_idx":0},
    {"prompt":"the dog is ", "choices":["barking.", "swimming."], "answer_idx":0},
  ]
  acc = evaluateMultipleChoice(model, tok, evalSet, device="cpu")
  assert 0.0<=acc<=1.0
  print("test_downstreamEvalRuns: pass, acc=", acc)


def test_latencyBenchmarkRunsOnCpu():
  model, tok = makeTinyModelAndTokenizer()
  perf = benchmarkGeneration(model, tok, "hello world", device="cpu", maxNewTokens=4, nWarmup=1, nRuns=1)
  assert perf["tokensPerSec"]>0
  print("test_latencyBenchmarkRunsOnCpu: pass,", perf)
  mem = modelWeightMemoryGb(model)
  assert mem>0
  print("  modelWeightMemoryGb:", mem)


def test_fullBuildMatrixRunsEndToEnd():
  model, tok = makeTinyModelAndTokenizer()
  evalSet = [{"prompt":"the sky is ", "choices":["blue.", "loud."], "answer_idx":0}]
  matrix = runFullBuildMatrix(
    model, tok, device="cpu", workloadTag="smoke_test",
    techniques=["fp16", "int8_rtn", "int4_rtn", "int4_gptq"],
  )
  assert len(matrix.records)==4
  print(matrix.summaryTable())
  best = matrix.bestBuildFor("CPU")
  print("best build:", best.buildId if best else None)
  print("test_fullBuildMatrixRunsEndToEnd: pass")


if __name__=="__main__":
  test_calibrationAndRtnQuantizeRun()
  test_gptqQuantizeRunsOnRealModelShapes()
  test_perplexityRuns()
  test_downstreamEvalRuns()
  test_latencyBenchmarkRunsOnCpu()
  test_fullBuildMatrixRunsEndToEnd()
  print("\nall integration tests passed")
