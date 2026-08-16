"""
quantizers/rtn.py

Round-to-nearest (RTN) weight-only quantization. The simplest possible
scheme, and the baseline every fancier technique (gptq, smoothquant) has to
beat to justify its extra complexity.

Per-group affine quantization: split each output row's weights into groups
of `groupSize` along the input dimension, compute a scale (and zero-point,
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


def quantParamsPerGroup(wGroup, bits, symmetric=True):
  """wGroup: (..., group_size). returns (scale, zero_point) broadcastable over group_size."""
  qmax = 2**(bits-1)-1
  qmin = -2**(bits-1)
  if symmetric:
    maxAbs = wGroup.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = maxAbs/qmax
    zeroPoint = torch.zeros_like(scale)
  else:
    qmaxU = 2**bits-1
    wMax = wGroup.amax(dim=-1, keepdim=True)
    wMin = wGroup.amin(dim=-1, keepdim=True)
    scale = (wMax-wMin).clamp(min=1e-8)/qmaxU
    zeroPoint = torch.round(-wMin/scale)
    qmax, qmin = qmaxU, 0
  return scale, zeroPoint, qmin, qmax


def fakeQuantizeRtn(weight, bits=4, groupSize=128, symmetric=True):
  """weight: (out_features, in_features). returns dequantized weight of the
  same shape (fake-quantized), plus the scale/zero_point tensors for
  inspection or for packing into a real low-bit format later."""
  outFeatures, inFeatures = weight.shape
  gs = groupSize if groupSize>0 else inFeatures
  pad = (gs-inFeatures%gs)%gs
  w = weight
  if pad>0:
    w = torch.nn.functional.pad(w, (0, pad))
  wGrouped = w.reshape(outFeatures, -1, gs)

  scale, zeroPoint, qmin, qmax = quantParamsPerGroup(wGrouped, bits, symmetric)

  q = torch.round(wGrouped/scale+zeroPoint).clamp(qmin, qmax)
  deq = (q-zeroPoint)*scale

  deq = deq.reshape(outFeatures, -1)
  if pad>0:
    deq = deq[:, :inFeatures]

  return deq, scale, zeroPoint


def quantizeModelWeightsRtn(model, bits=4, groupSize=128, symmetric=True, targetModuleTypes=None):
  """in-place: replaces every nn.Linear's weight with its rtn fake-quantized
  version. skips lm_head / embed_tokens by default since those are usually
  kept higher precision (huge quality sensitivity per parameter touched)."""
  import torch.nn as nn
  targetModuleTypes = targetModuleTypes or (nn.Linear,)
  skipNameSubstrings = ("lm_head", "embed_tokens")

  nQuantized = 0
  for name, module in model.named_modules():
    if isinstance(module, targetModuleTypes):
      if any(s in name for s in skipNameSubstrings):
        continue
      with torch.no_grad():
        deq, scale, zp = fakeQuantizeRtn(module.weight.data.float(), bits=bits, groupSize=groupSize, symmetric=symmetric)
        module.weight.data.copy_(deq.to(module.weight.dtype))
      nQuantized += 1
  return nQuantized
