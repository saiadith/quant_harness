"""
quantizers/smoothquant.py

Simplified SmoothQuant (Xiao et al. 2022) + an LLM.int8()-style outlier
channel fallback, since the two problems are closely related: activations
in transformer layers develop a small number of channels with much larger
magnitude than the rest, and those outlier channels are what make
per-tensor activation quantization fall apart (a few huge values force a
large scale, crushing everything else to near-zero resolution). Weights,
by contrast, are much more uniform.

SmoothQuant's move: migrate quantization difficulty from activations to
weights with a per-channel scale, since weights can absorb it more easily.
For input channel j:

    s_j = max(|X_j|)^alpha / max(|W_j|)^(1 - alpha)

then X'_j = X_j / s_j  and  W'_j = W_j * s_j  (same matmul, since
(X/s) @ (W*s)^T recovers X @ W^T along that channel). alpha=0.5 splits the
difficulty evenly; alpha->1 pushes almost everything onto the weights.

Outlier channel handling (LLM.int8()-style, on top of smoothing): even
after smoothing, a handful of channels can remain large enough to still
dominate a per-tensor int8 range. those channels are pulled out and kept
in fp16, computed separately, and added back - a small, cheap "escape
hatch" for the channels smoothing alone doesn't fully tame.
"""

import torch


def computeSmoothingScale(weightAbsMax, actAbsMax, alpha=0.5, eps=1e-5):
  """weightAbsMax, actAbsMax: (in_features,). returns per-channel scale s."""
  return (actAbsMax.clamp(min=eps)**alpha)/(weightAbsMax.clamp(min=eps)**(1-alpha))


def detectOutlierChannels(actAbsMax, k=None, ratioThreshold=6.0):
  """flags channels whose activation magnitude is far above the median - the
  ones a per-tensor int8 range would otherwise be dominated by. either take
  the top-k, or anything more than `ratioThreshold` times the median."""
  median = actAbsMax.median()
  if k is not None:
    _, idx = torch.topk(actAbsMax, k)
    return idx
  return torch.nonzero(actAbsMax>ratioThreshold*median, as_tuple=True)[0]


def quantizeActivationsInt8Dynamic(x, outlierIdx=None):
  """x: (..., in_features). per-token (row-wise) dynamic int8 fake-quant,
  with outlier channels excluded from the int8 range and passed through
  in full precision, then added back after dequantization."""
  origShape = x.shape
  xFlat = x.reshape(-1, origShape[-1])

  if outlierIdx is not None and len(outlierIdx)>0:
    mask = torch.zeros(xFlat.shape[-1], dtype=torch.bool, device=x.device)
    mask[outlierIdx] = True
    xOutlier = xFlat*mask
    xNormal = xFlat*(~mask)
  else:
    xOutlier = torch.zeros_like(xFlat)
    xNormal = xFlat

  rowAbsMax = xNormal.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
  qmax = 127
  scale = rowAbsMax/qmax
  q = torch.round(xNormal/scale).clamp(-qmax, qmax)
  deqNormal = q*scale

  deq = deqNormal+xOutlier   # outliers stay exact (fp) - they never touched the int8 path
  return deq.reshape(origShape)


def applySmoothquantToLinear(weight, actAbsMax, bits=8, groupSize=128, alpha=0.5, outlierK=None, outlierRatio=6.0):
  """
  weight: (out_features, in_features)
  actAbsMax: (in_features,) from calibration
  returns: (smoothed_and_quantized_weight, smoothing_scale, outlier_idx)
    smoothing_scale must be applied to ACTIVATIONS at inference: x' = x / s
    before feeding this weight (the weight already has *s baked in).
  """
  from quantizers.rtn import fakeQuantizeRtn

  weightAbsMax = weight.abs().amax(dim=0)   # (in_features,) per input-channel max over all output rows
  s = computeSmoothingScale(weightAbsMax, actAbsMax, alpha=alpha)

  wSmoothed = weight*s.unsqueeze(0)        # broadcast across out_features

  outlierIdx = detectOutlierChannels(actAbsMax, k=outlierK, ratioThreshold=outlierRatio)

  wQuantized, scale, zp = fakeQuantizeRtn(wSmoothed, bits=bits, groupSize=groupSize, symmetric=True)

  return wQuantized, s, outlierIdx


def quantizeModelWeightsSmoothquant(model, calibStats, bits=8, groupSize=128, alpha=0.5, outlierK=None, outlierRatio=6.0, targetModuleTypes=None):
  """
  in-place: for every linear layer with calibration stats, bakes the
  smoothing scale into the weight and fake-quantizes it. NOTE: this
  simulates the weight side only. to fully simulate SmoothQuant's activation
  int8 path too (not just its effect on weight quantizability), wrap the
  module's forward to divide its input by `s` and call
  quantizeActivationsInt8Dynamic - see notebooks/ for a worked example
  using forward hooks, since that requires touching the live forward pass
  rather than just the stored weight.
  """
  import torch.nn as nn
  targetModuleTypes = targetModuleTypes or (nn.Linear,)
  skipNameSubstrings = ("lm_head", "embed_tokens")

  info = {}
  for name, module in model.named_modules():
    if isinstance(module, targetModuleTypes):
      if any(s in name for s in skipNameSubstrings):
        continue
      if name not in calibStats:
        continue
      actAbsMax = calibStats[name].absMax
      with torch.no_grad():
        wQ, s, outlierIdx = applySmoothquantToLinear(
          module.weight.data.float(), actAbsMax, bits=bits, groupSize=groupSize,
          alpha=alpha, outlierK=outlierK, outlierRatio=outlierRatio,
        )
        module.weight.data.copy_(wQ.to(module.weight.dtype))
      info[name] = dict(smoothingScale=s, outlierIdx=outlierIdx)
  return info
