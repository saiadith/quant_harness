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
    self.absMax = None
    self.nSamples = 0
    self.sumXtx = None
    self.nRows = 0

  def update(self, xFlat):
    batchMax = xFlat.abs().amax(dim=0)
    if self.absMax is None:
      self.absMax = batchMax.clone()
    else:
      self.absMax = torch.maximum(self.absMax, batchMax)
    self.nSamples += xFlat.shape[0]

    xtx = xFlat.T@xFlat
    if self.sumXtx is None:
      self.sumXtx = xtx.clone()
    else:
      self.sumXtx += xtx
    self.nRows += xFlat.shape[0]

  def hessian(self, damping=0.01):
    """gptq-style hessian approximation: h = 2 * x^t x / n, with diagonal damping
    for numerical stability during inversion."""
    h = 2.0*self.sumXtx/max(self.nRows, 1)
    meanDiag = torch.diag(h).mean()
    damp = damping*meanDiag
    h = h+damp*torch.eye(h.shape[0], device=h.device, dtype=h.dtype)
    return h


def collectCalibrationStats(model, tokenizer, calibTexts, device="cpu", maxLength=512, targetModuleTypes=(nn.Linear,)):
  """runs calibration text through the model once, collecting per-linear-layer
  ActivationStats via forward hooks. returns {module_name: ActivationStats}."""
  stats = {}
  handles = []

  def makeHook(name):
    def hook(module, inputs, output):
      x = inputs[0].detach()
      xFlat = x.reshape(-1, x.shape[-1]).to(torch.float32)
      if name not in stats:
        stats[name] = ActivationStats()
      stats[name].update(xFlat)
    return hook

  for name, module in model.named_modules():
    if isinstance(module, targetModuleTypes):
      handles.append(module.register_forward_hook(makeHook(name)))

  model.eval()
  with torch.no_grad():
    for text in calibTexts:
      enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=maxLength).to(device)
      model(**enc)

  for h in handles:
    h.remove()

  return stats


def defaultCalibrationTexts():
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
