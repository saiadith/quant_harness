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


def compute_smoothing_scale(weight_absmax, act_absmax, alpha=0.5, eps=1e-5):
  """weight_absmax, act_absmax: (in_features,). returns per-channel scale s."""
  return (act_absmax.clamp(min=eps) ** alpha) / (weight_absmax.clamp(min=eps) ** (1 - alpha))


def detect_outlier_channels(act_absmax, k=None, ratio_threshold=6.0):
  """flags channels whose activation magnitude is far above the median - the
  ones a per-tensor int8 range would otherwise be dominated by. either take
  the top-k, or anything more than `ratio_threshold` times the median."""
  median = act_absmax.median()
  if k is not None:
    _, idx = torch.topk(act_absmax, k)
    return idx
  return torch.nonzero(act_absmax > ratio_threshold * median, as_tuple=True)[0]


def quantize_activations_int8_dynamic(x, outlier_idx=None):
  """x: (..., in_features). per-token (row-wise) dynamic int8 fake-quant,
  with outlier channels excluded from the int8 range and passed through
  in full precision, then added back after dequantization."""
  orig_shape = x.shape
  x_flat = x.reshape(-1, orig_shape[-1])

  if outlier_idx is not None and len(outlier_idx) > 0:
    mask = torch.zeros(x_flat.shape[-1], dtype=torch.bool, device=x.device)
    mask[outlier_idx] = True
    x_outlier = x_flat * mask
    x_normal = x_flat * (~mask)
  else:
    x_outlier = torch.zeros_like(x_flat)
    x_normal = x_flat

  row_absmax = x_normal.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
  qmax = 127
  scale = row_absmax / qmax
  q = torch.round(x_normal / scale).clamp(-qmax, qmax)
  deq_normal = q * scale

  deq = deq_normal + x_outlier   # outliers stay exact (fp) - they never touched the int8 path
  return deq.reshape(orig_shape)


def apply_smoothquant_to_linear(weight, act_absmax, bits=8, group_size=128, alpha=0.5, outlier_k=None, outlier_ratio=6.0):
  """
  weight: (out_features, in_features)
  act_absmax: (in_features,) from calibration
  returns: (smoothed_and_quantized_weight, smoothing_scale, outlier_idx)
    smoothing_scale must be applied to ACTIVATIONS at inference: x' = x / s
    before feeding this weight (the weight already has *s baked in).
  """
  from quantizers.rtn import fake_quantize_rtn

  weight_absmax = weight.abs().amax(dim=0)   # (in_features,) per input-channel max over all output rows
  s = compute_smoothing_scale(weight_absmax, act_absmax, alpha=alpha)

  w_smoothed = weight * s.unsqueeze(0)        # broadcast across out_features

  outlier_idx = detect_outlier_channels(act_absmax, k=outlier_k, ratio_threshold=outlier_ratio)

  w_quantized, scale, zp = fake_quantize_rtn(w_smoothed, bits=bits, group_size=group_size, symmetric=True)

  return w_quantized, s, outlier_idx


def quantize_model_weights_smoothquant_(model, calib_stats, bits=8, group_size=128, alpha=0.5, outlier_k=None, outlier_ratio=6.0, target_module_types=None):
  """
  in-place: for every linear layer with calibration stats, bakes the
  smoothing scale into the weight and fake-quantizes it. NOTE: this
  simulates the weight side only. to fully simulate SmoothQuant's activation
  int8 path too (not just its effect on weight quantizability), wrap the
  module's forward to divide its input by `s` and call
  quantize_activations_int8_dynamic - see notebooks/ for a worked example
  using forward hooks, since that requires touching the live forward pass
  rather than just the stored weight.
  """
  import torch.nn as nn
  target_module_types = target_module_types or (nn.Linear,)
  skip_name_substrings = ("lm_head", "embed_tokens")

  info = {}
  for name, module in model.named_modules():
    if isinstance(module, target_module_types):
      if any(s in name for s in skip_name_substrings):
        continue
      if name not in calib_stats:
        continue
      act_absmax = calib_stats[name].abs_max
      with torch.no_grad():
        w_q, s, outlier_idx = apply_smoothquant_to_linear(
          module.weight.data.float(), act_absmax, bits=bits, group_size=group_size,
          alpha=alpha, outlier_k=outlier_k, outlier_ratio=outlier_ratio,
        )
        module.weight.data.copy_(w_q.to(module.weight.dtype))
      info[name] = dict(smoothing_scale=s, outlier_idx=outlier_idx)
  return info
