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

Note: quantize_model_weights_gptq_ below runs this on CPU. The calibration
Hessian is CPU-resident (see common/calibration.py) to avoid an OOM from
keeping every layer's Hessian on the GPU at once, so the weight being
quantized has to move to CPU too for the two to interoperate. This makes
GPTQ noticeably slower than doing the same math on GPU - a fine trade for
correctness on a 16GB T4, but worth knowing if a full build matrix run
feels like it's taking a while.
"""

import torch
from quantizers.rtn import _quant_params_per_group


def gptq_quantize_layer(weight, hessian, bits=4, group_size=128, symmetric=True, damping=0.01):
  """
  weight: (out_features, in_features) float tensor, the layer to quantize
  hessian: (in_features, in_features) float tensor from calibration (see
    common.calibration.ActivationStats.hessian)
  returns: dequantized (fake-quantized) weight of the same shape
  """
  W = weight.clone().float()
  out_features, in_features = W.shape

  H = hessian.clone().float()
  diag = torch.diag(H)
  dead = diag == 0
  if dead.any():
    # observation: columns with zero calibration activation give a singular hessian row -
    # treat them as "don't care", inverse-variance goes to a neutral default
    H[dead, dead] = 1.0
  damp = damping * diag.mean()
  H = H + damp * torch.eye(in_features, device=H.device, dtype=H.dtype)

  Hinv = torch.inverse(H)
  # upper-triangular U with U^T @ U == Hinv
  Hinv_chol = torch.linalg.cholesky(Hinv).T

  Q = torch.zeros_like(W)
  scale = zero_point = None
  qmin = qmax = None

  for col in range(in_features):
    if col % group_size == 0:
      group_end = min(col + group_size, in_features)
      w_group = W[:, col:group_end].unsqueeze(1)   # (out, 1, group_size)
      scale_g, zp_g, qmin, qmax = _quant_params_per_group(w_group, bits, symmetric)
      scale, zero_point = scale_g.squeeze(1), zp_g.squeeze(1)   # (out, 1)

    w_col = W[:, col:col + 1]
    q_level = torch.round(w_col / scale + zero_point).clamp(qmin, qmax)
    deq_col = (q_level - zero_point) * scale
    Q[:, col:col + 1] = deq_col

    denom = Hinv_chol[col, col]
    if denom.abs() < 1e-8:
      denom = torch.tensor(1e-8)
    err = (w_col - deq_col) / denom

    if col + 1 < in_features:
      # observation: this is the whole trick - spread this column's rounding
      # error onto the still-unquantized columns, weighted by hessian curvature
      W[:, col + 1:] = W[:, col + 1:] - err @ Hinv_chol[col:col + 1, col + 1:]

  return Q


def quantize_model_weights_gptq_(model, calib_stats, bits=4, group_size=128, symmetric=True, damping=0.01, target_module_types=None):
  """in-place gptq quantization of every linear layer for which we collected
  calibration stats. layers without stats (not exercised by calibration
  text) silently fall back to skip - log a warning in real use.

  weight is moved to cpu before quantizing since the hessian is cpu-resident
  (see the OOM fix in common/calibration.py) - both operands need to be on
  the same device for the matmuls in gptq_quantize_layer to work."""
  import torch.nn as nn
  target_module_types = target_module_types or (nn.Linear,)
  skip_name_substrings = ("lm_head", "embed_tokens")

  n_quantized = 0
  for name, module in model.named_modules():
    if isinstance(module, target_module_types):
      if any(s in name for s in skip_name_substrings):
        continue
      if name not in calib_stats:
        continue
      H = calib_stats[name].hessian(damping=damping)
      with torch.no_grad():
        orig_device = module.weight.data.device
        w_cpu = module.weight.data.float().cpu()
        deq = gptq_quantize_layer(w_cpu, H, bits=bits, group_size=group_size, symmetric=symmetric, damping=damping)
        module.weight.data.copy_(deq.to(orig_device).to(module.weight.dtype))
      n_quantized += 1
  return n_quantized
