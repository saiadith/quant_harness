"""
quantizers/gptq_lite.py

A from-scratch, simplified reimplementation of GPTQ (Frantar et al. 2022).
The core idea RTN is missing: when you quantize one weight column, you
introduce an error. GPTQ propagates that error to the NOT-YET-quantized
columns of the same row, weighted by how sensitive the layer's OUTPUT is
to each column (captured by the calibration Hessian H = X^T X). Columns
whose direction the layer barely uses (low curvature) absorb the error;
columns the layer output is sensitive to are corrected for it. RTN treats
every column as equally important - this is the whole reason GPTQ tends to
beat RTN at the same bit-width.

Algorithm (non-blocked, full-precision-inverse version - the paper's fast
implementation uses one Cholesky factorization instead of blockwise
re-inversion, which is a straightforward optimization left as an exercise;
this version keeps the math visible column-by-column instead):

  1. dampen H's diagonal for numerical stability, invert it
  2. take the Cholesky factor U of H^-1 (upper triangular, U^T U = H^-1)
  3. for each input-dim column i, left to right:
       - quantize column i with RTN (using the current, error-corrected weights)
       - compute the residual error, scaled by 1 / U[i, i]
       - subtract that error's projection onto every NOT-YET-quantized
         column, using row i of U as the weighting
"""

import torch
from quantizers.rtn import quantParamsPerGroup


def gptqQuantizeLayer(weight, hessian, bits=4, groupSize=128, symmetric=True, damping=0.01):
  """
  weight: (out_features, in_features) float tensor, the layer to quantize
  hessian: (in_features, in_features) float tensor from calibration (see
    common.calibration.ActivationStats.hessian)
  returns: dequantized (fake-quantized) weight of the same shape
  """
  w = weight.clone().float()
  outFeatures, inFeatures = w.shape

  h = hessian.clone().float()
  diag = torch.diag(h)
  dead = diag==0
  if dead.any():
    # observation: columns with zero calibration activation give a singular hessian row -
    # treat them as "don't care", inverse-variance goes to a neutral default
    h[dead, dead] = 1.0
  damp = damping*diag.mean()
  h = h+damp*torch.eye(inFeatures, device=h.device, dtype=h.dtype)

  hinv = torch.inverse(h)
  hinvChol = torch.linalg.cholesky(hinv).T

  q = torch.zeros_like(w)
  scale = zeroPoint = None
  qmin = qmax = None

  for col in range(inFeatures):
    if col%groupSize==0:
      groupEnd = min(col+groupSize, inFeatures)
      wGroup = w[:, col:groupEnd].unsqueeze(1)
      scaleG, zpG, qmin, qmax = quantParamsPerGroup(wGroup, bits, symmetric)
      scale, zeroPoint = scaleG.squeeze(1), zpG.squeeze(1)

    wCol = w[:, col:col+1]
    qLevel = torch.round(wCol/scale+zeroPoint).clamp(qmin, qmax)
    deqCol = (qLevel-zeroPoint)*scale
    q[:, col:col+1] = deqCol

    denom = hinvChol[col, col]
    if denom.abs()<1e-8:
      denom = torch.tensor(1e-8)
    err = (wCol-deqCol)/denom

    if col+1<inFeatures:
      # observation: this is the whole trick - spread this column's rounding
      # error onto the still-unquantized columns, weighted by hessian curvature
      w[:, col+1:] = w[:, col+1:]-err@hinvChol[col:col+1, col+1:]

  return q


def quantizeModelWeightsGptq(model, calibStats, bits=4, groupSize=128, symmetric=True, damping=0.01, targetModuleTypes=None):
  """in-place gptq quantization of every linear layer for which we collected
  calibration stats. layers without stats (not exercised by calibration
  text) silently fall back to skip - log a warning in real use."""
  import torch.nn as nn
  targetModuleTypes = targetModuleTypes or (nn.Linear,)
  skipNameSubstrings = ("lm_head", "embed_tokens")

  nQuantized = 0
  for name, module in model.named_modules():
    if isinstance(module, targetModuleTypes):
      if any(s in name for s in skipNameSubstrings):
        continue
      if name not in calibStats:
        continue
      h = calibStats[name].hessian(damping=damping)
      with torch.no_grad():
        deq = gptqQuantizeLayer(module.weight.data.float(), h, bits=bits, groupSize=groupSize, symmetric=symmetric, damping=damping)
        module.weight.data.copy_(deq.to(module.weight.dtype))
      nQuantized += 1
  return nQuantized
