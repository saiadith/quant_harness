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
def benchmarkGeneration(model, tokenizer, prompt, device="cuda", maxNewTokens=64, nWarmup=2, nRuns=5):
  model.eval()
  inputIds = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

  if device.startswith("cuda"):
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

  for _ in range(nWarmup):
    model.generate(inputIds, max_new_tokens=maxNewTokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)

  if device.startswith("cuda"):
    torch.cuda.synchronize()

  ttftSamples = []
  totalTime = 0.0
  totalNewTokens = 0

  for _ in range(nRuns):
    start = time.perf_counter()
    # time to first token: generate exactly one token
    _ = model.generate(inputIds, max_new_tokens=1, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    if device.startswith("cuda"):
      torch.cuda.synchronize()
    ttftSamples.append((time.perf_counter()-start)*1000)

    start = time.perf_counter()
    out = model.generate(inputIds, max_new_tokens=maxNewTokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    if device.startswith("cuda"):
      torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    nGenerated = out.shape[1]-inputIds.shape[1]
    totalTime += elapsed
    totalNewTokens += nGenerated

  tokensPerSec = totalNewTokens/totalTime if totalTime>0 else 0.0
  ttftMs = sum(ttftSamples)/len(ttftSamples)

  peakMemoryGb = None
  if device.startswith("cuda"):
    peakMemoryGb = torch.cuda.max_memory_allocated(device)/(1024**3)

  return dict(tokensPerSec=tokensPerSec, ttftMs=ttftMs, peakMemoryGb=peakMemoryGb)


def modelWeightMemoryGb(model):
  totalBytes = sum(p.numel()*p.element_size() for p in model.parameters())
  return totalBytes/(1024**3)


def detectHardwareTag():
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
