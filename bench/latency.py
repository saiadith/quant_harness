"""
bench/latency.py

Tokens/sec, time-to-first-token, and peak memory for a given model. Written
to be hardware-generic (just torch.cuda calls) so the same function runs
unmodified on a T4, Ada, Hopper, or Blackwell card - only the RESULTS
differ, which is exactly the point of tagging every build with the
hardware it was measured on.
"""

import time
import torch


@torch.no_grad()
def benchmark_generation(model, tokenizer, prompt, device="cuda", max_new_tokens=64, n_warmup=2, n_runs=5):
  model.eval()
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

  if device.startswith("cuda"):
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

  for _ in range(n_warmup):
    model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)

  if device.startswith("cuda"):
    torch.cuda.synchronize()

  ttft_samples = []
  total_time = 0.0
  total_new_tokens = 0

  for _ in range(n_runs):
    start = time.perf_counter()
    # time to first token: generate exactly one token
    _ = model.generate(input_ids, max_new_tokens=1, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    if device.startswith("cuda"):
      torch.cuda.synchronize()
    ttft_samples.append((time.perf_counter() - start) * 1000)

    start = time.perf_counter()
    out = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    if device.startswith("cuda"):
      torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    n_generated = out.shape[1] - input_ids.shape[1]
    total_time += elapsed
    total_new_tokens += n_generated

  tokens_per_sec = total_new_tokens / total_time if total_time > 0 else 0.0
  ttft_ms = sum(ttft_samples) / len(ttft_samples)

  peak_memory_gb = None
  if device.startswith("cuda"):
    peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

  return dict(tokens_per_sec=tokens_per_sec, ttft_ms=ttft_ms, peak_memory_gb=peak_memory_gb)


def model_weight_memory_gb(model):
  total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
  return total_bytes / (1024 ** 3)


def detect_hardware_tag():
  """maps the visible cuda device name to one of common.build_record.SUPPORTED_HARDWARE.
  extend the substring map as you test on more cards."""
  if not torch.cuda.is_available():
    return "CPU"
  name = torch.cuda.get_device_name(0).upper()
  if "T4" in name:
    return "T4"
  if "H100" in name or "H200" in name or "HOPPER" in name:
    return "HOPPER"
  if "BLACKWELL" in name or "B100" in name or "B200" in name or "RTX 6000 PRO" in name:
    return "BLACKWELL"
  if "A100" in name or "L4" in name or "RTX 4" in name or "ADA" in name:
    return "ADA"
  return name
