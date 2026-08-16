"""
quantizers/rtn.py

Round-to-nearest (RTN) weight-only quantization. The simplest possible
scheme, and the baseline every fancier technique (gptq, smoothquant) has to
beat to justify its extra complexity.

Per-group affine quantization: split each output row's weights into groups
of `group_size` along the input dimension, compute a scale (and zero-point,
if asymmetric) per group, round to the nearest representable level, clamp,
then dequantize back to float for simulated ("fake") quantization.

This whole module works in "fake quant" mode: weights stay float32/16 in
memory but pass through quantize->dequantize, so downstream code (the model
forward pass) doesn't need custom int4 kernels to measure QUALITY impact.
Real deployment would pack these into actual int4/int8 storage using
something like NVIDIA Model-Optimizer or llm-compressor's kernels - that's
a systems/kernels problem, this module is about getting the quantization
MATH right first.
"""

import torch


def _quant_params_per_group(w_group, bits, symmetric=True):
  """w_group: (..., group_size). returns (scale, zero_point) broadcastable over group_size."""
  qmax = 2 ** (bits - 1) - 1
  qmin = -2 ** (bits - 1)
  if symmetric:
    max_abs = w_group.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = max_abs / qmax
    zero_point = torch.zeros_like(scale)
  else:
    qmax_u = 2 ** bits - 1
    w_max = w_group.amax(dim=-1, keepdim=True)
    w_min = w_group.amin(dim=-1, keepdim=True)
    scale = (w_max - w_min).clamp(min=1e-8) / qmax_u
    zero_point = torch.round(-w_min / scale)
    qmax, qmin = qmax_u, 0
  return scale, zero_point, qmin, qmax


def fake_quantize_rtn(weight, bits=4, group_size=128, symmetric=True):
  """weight: (out_features, in_features). returns dequantized weight of the
  same shape (fake-quantized), plus the scale/zero_point tensors for
  inspection or for packing into a real low-bit format later."""
  out_features, in_features = weight.shape
  gs = group_size if group_size > 0 else in_features
  pad = (gs - in_features % gs) % gs
  w = weight
  if pad > 0:
    w = torch.nn.functional.pad(w, (0, pad))
  w_grouped = w.reshape(out_features, -1, gs)

  scale, zero_point, qmin, qmax = _quant_params_per_group(w_grouped, bits, symmetric)

  q = torch.round(w_grouped / scale + zero_point).clamp(qmin, qmax)
  deq = (q - zero_point) * scale

  deq = deq.reshape(out_features, -1)
  if pad > 0:
    deq = deq[:, :in_features]

  return deq, scale, zero_point


def quantize_model_weights_rtn_(model, bits=4, group_size=128, symmetric=True, target_module_types=None):
  """in-place: replaces every nn.Linear's weight with its rtn fake-quantized
  version. skips lm_head / embed_tokens by default since those are usually
  kept higher precision (huge quality sensitivity per parameter touched)."""
  import torch.nn as nn
  target_module_types = target_module_types or (nn.Linear,)
  skip_name_substrings = ("lm_head", "embed_tokens")

  n_quantized = 0
  for name, module in model.named_modules():
    if isinstance(module, target_module_types):
      if any(s in name for s in skip_name_substrings):
        continue
      with torch.no_grad():
        deq, scale, zp = fake_quantize_rtn(module.weight.data.float(), bits=bits, group_size=group_size, symmetric=symmetric)
        module.weight.data.copy_(deq.to(module.weight.dtype))
      n_quantized += 1
  return n_quantized
