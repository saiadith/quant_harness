"""
common/calibration.py

Collects per-channel activation statistics (running abs-max, and full
activation samples for GPTQ's Hessian) by hooking every nn.Linear during a
few forward passes over calibration text. Every quantization technique
downstream (RTN doesn't need this, GPTQ and SmoothQuant do) consumes
whatever this collects.
"""

import torch
import torch.nn as nn


class ActivationStats:
  def __init__(self):
    self.abs_max = None          # (in_features,) running per-channel abs max, for smoothquant
    self.n_samples = 0
    self.sum_xtx = None          # (in_features, in_features) running sum of X^T X, for gptq hessian
    self.n_rows = 0

  def update(self, x_flat):
    # move off gpu before accumulating - the hessian for every linear layer
    # in the model otherwise stays resident on the gpu at the same time,
    # which is what actually causes an oom on a t4, not the model weights
    # themselves. cpu has far more headroom for this.
    x_flat = x_flat.detach().float().cpu()

    batch_max = x_flat.abs().amax(dim=0)
    if self.abs_max is None:
      self.abs_max = batch_max.clone()
    else:
      self.abs_max = torch.maximum(self.abs_max, batch_max)
    self.n_samples += x_flat.shape[0]

    xtx = x_flat.T @ x_flat
    if self.sum_xtx is None:
      self.sum_xtx = xtx.clone()
    else:
      self.sum_xtx += xtx
    self.n_rows += x_flat.shape[0]

  def hessian(self, damping=0.01):
    """gptq-style hessian approximation: H = 2 * X^T X / n, with diagonal damping
    for numerical stability during inversion. lives on cpu, since update() now
    accumulates there."""
    h = 2.0 * self.sum_xtx / max(self.n_rows, 1)
    mean_diag = torch.diag(h).mean()
    damp = damping * mean_diag
    h = h + damp * torch.eye(h.shape[0], device=h.device, dtype=h.dtype)
    return h


def collect_calibration_stats(model, tokenizer, calib_texts, device="cpu", max_length=512, target_module_types=(nn.Linear,)):
  """runs calibration text through the model once, collecting per-linear-layer
  ActivationStats via forward hooks. returns {module_name: ActivationStats}.
  the forward pass itself still runs on `device` (fast) - only the persistent
  per-layer stats get moved to cpu inside ActivationStats.update()."""
  stats = {}
  handles = []

  def make_hook(name):
    def hook(module, inputs, output):
      x = inputs[0].detach()
      x_flat = x.reshape(-1, x.shape[-1]).to(torch.float32)
      if name not in stats:
        stats[name] = ActivationStats()
      stats[name].update(x_flat)
    return hook

  for name, module in model.named_modules():
    if isinstance(module, target_module_types):
      handles.append(module.register_forward_hook(make_hook(name)))

  model.eval()
  with torch.no_grad():
    for text in calib_texts:
      enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
      model(**enc)

  for h in handles:
    h.remove()

  return stats


def default_calibration_texts():
  """small, self-contained calibration set so the harness runs with zero
  external dataset dependency. swap for c4/wikitext samples on kaggle for
  a more representative calibration distribution."""
  return [
    "the quarterly earnings report showed a significant increase in revenue across all business units.",
    "to reset your password, click the link below and follow the instructions provided.",
    "photosynthesis is the process by which plants convert sunlight into chemical energy.",
    "the customer service representative apologized for the delay and offered a refund.",
    "in the year 1969, astronauts first landed on the surface of the moon.",
    "please confirm your appointment for tomorrow at 3pm by replying to this message.",
    "the recipe calls for two cups of flour, one teaspoon of salt, and three eggs.",
    "our flight has been delayed due to weather conditions at the departure airport.",
    "the committee will review the proposal and provide feedback within two weeks.",
    "machine learning models require large amounts of labeled data to train effectively.",
    "thank you for calling support, how can i help you with your account today.",
    "the museum's new exhibit features artifacts from ancient civilizations around the world.",
  ]
