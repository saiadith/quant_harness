"""
tests/test_pipeline_integration.py

No internet access to huggingface.co in this sandbox, so this can't
download real Qwen weights. Instead: build a tiny randomly-initialized
gpt2-architecture model (transformers can construct this from a Config
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

from common.calibration import collect_calibration_stats, default_calibration_texts
from quantizers.rtn import quantize_model_weights_rtn_
from quantizers.gptq_lite import quantize_model_weights_gptq_
from eval.perplexity import compute_perplexity
from eval.downstream_task import evaluate_multiple_choice
from bench.latency import benchmark_generation, model_weight_memory_gb
from bench.build_matrix import build_and_evaluate, run_full_build_matrix


class TokenizerOutput(dict):
  """dict subclass so `model(**enc)` unpacking works, plus attribute access
  and a no-op `.to(device)` to match the real hf BatchEncoding interface."""

  def __getattr__(self, name):
    try:
      return self[name]
    except KeyError:
      raise AttributeError(name)

  def to(self, device):
    return TokenizerOutput({k: v.to(device) for k, v in self.items()})


class CharTokenizer:
  """minimal stand-in exposing just enough of the hf tokenizer interface
  for our eval/bench code to run - no network access needed."""

  def __init__(self, text_corpus):
    chars = sorted(set("".join(text_corpus)) | set(" abcdefghijklmnopqrstuvwxyz.,"))
    self.stoi = {c: i for i, c in enumerate(chars)}
    self.itos = {i: c for i, c in enumerate(chars)}
    self.vocab_size = len(chars)
    self.eos_token_id = 0

  def __call__(self, text, return_tensors=None, truncation=False, max_length=None):
    ids = [self.stoi.get(c, 0) for c in text]
    if max_length:
      ids = ids[:max_length]
    return TokenizerOutput(input_ids=torch.tensor([ids]))

  def decode(self, ids):
    return "".join(self.itos.get(int(i), "") for i in ids)


def make_tiny_model_and_tokenizer():
  corpus = default_calibration_texts()
  tok = CharTokenizer(corpus)
  config = LlamaConfig(
    vocab_size=tok.vocab_size, max_position_embeddings=128, hidden_size=32,
    intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
    bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
  )
  model = LlamaForCausalLM(config)
  model.config.name_or_path = "tiny-test-llama"
  return model, tok


def test_calibration_and_rtn_quantize_run():
  model, tok = make_tiny_model_and_tokenizer()
  stats = collect_calibration_stats(model, tok, default_calibration_texts()[:4], device="cpu", max_length=64)
  assert len(stats) > 0, "should have hooked at least one linear layer"
  n = quantize_model_weights_rtn_(model, bits=4, group_size=16)
  assert n > 0
  print("test_calibration_and_rtn_quantize_run: pass, quantized", n, "layers")


def test_gptq_quantize_runs_on_real_model_shapes():
  model, tok = make_tiny_model_and_tokenizer()
  stats = collect_calibration_stats(model, tok, default_calibration_texts(), device="cpu", max_length=64)
  n = quantize_model_weights_gptq_(model, stats, bits=4, group_size=16)
  assert n > 0
  for p in model.parameters():
    assert torch.isfinite(p).all(), "gptq quantization produced non-finite weights"
  print("test_gptq_quantize_runs_on_real_model_shapes: pass, quantized", n, "layers")


def test_perplexity_runs():
  model, tok = make_tiny_model_and_tokenizer()
  ppl = compute_perplexity(model, tok, "the quick brown fox jumps over the lazy dog. " * 10, device="cpu", max_length=64, stride=32)
  assert ppl > 0 and torch.isfinite(torch.tensor(ppl))
  print("test_perplexity_runs: pass, ppl=", ppl)


def test_downstream_eval_runs():
  model, tok = make_tiny_model_and_tokenizer()
  eval_set = [
    {"prompt": "the sky is ", "choices": ["blue.", "loud."], "answer_idx": 0},
    {"prompt": "the dog is ", "choices": ["barking.", "swimming."], "answer_idx": 0},
  ]
  acc = evaluate_multiple_choice(model, tok, eval_set, device="cpu")
  assert 0.0 <= acc <= 1.0
  print("test_downstream_eval_runs: pass, acc=", acc)


def test_latency_benchmark_runs_on_cpu():
  model, tok = make_tiny_model_and_tokenizer()
  perf = benchmark_generation(model, tok, "hello world", device="cpu", max_new_tokens=4, n_warmup=1, n_runs=1)
  assert perf["tokens_per_sec"] > 0
  print("test_latency_benchmark_runs_on_cpu: pass,", perf)
  mem = model_weight_memory_gb(model)
  assert mem > 0
  print("  model_weight_memory_gb:", mem)


def test_full_build_matrix_runs_end_to_end():
  model, tok = make_tiny_model_and_tokenizer()
  eval_set = [{"prompt": "the sky is ", "choices": ["blue.", "loud."], "answer_idx": 0}]
  matrix = run_full_build_matrix(
    model, tok, device="cpu", workload_tag="smoke_test",
    techniques=["fp16", "int8_rtn", "int4_rtn", "int4_gptq"],
  )
  assert len(matrix.records) == 4
  print(matrix.summary_table())
  best = matrix.best_build_for("CPU")
  print("best build:", best.build_id if best else None)
  print("test_full_build_matrix_runs_end_to_end: pass")


if __name__ == "__main__":
  test_calibration_and_rtn_quantize_run()
  test_gptq_quantize_runs_on_real_model_shapes()
  test_perplexity_runs()
  test_downstream_eval_runs()
  test_latency_benchmark_runs_on_cpu()
  test_full_build_matrix_runs_end_to_end()
  print("\nall integration tests passed")
