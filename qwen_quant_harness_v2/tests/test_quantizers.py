"""
tests/test_quantizers.py

Validates the quantization math on synthetic linear layers before ever
touching a real model. Key research question this answers: does gptq
actually beat plain rtn at the same bit-width, given the kind of
CORRELATED, outlier-heavy activation distribution real transformer layers
produce? We build synthetic data with exactly that shape (a few high-
variance directions, correlated columns) since on purely i.i.d. isotropic
data rtn and gptq are expected to perform similarly - correlation is what
gptq's hessian-based correction is designed to exploit.

note: function names below keep the pytest-required "test_" prefix (pytest's
default discovery pattern is "test_*") but drop internal underscores after that.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from quantizers.rtn import fakeQuantizeRtn
from quantizers.gptq_lite import gptqQuantizeLayer
from quantizers.smoothquant import applySmoothquantToLinear, quantizeActivationsInt8Dynamic, detectOutlierChannels
from common.calibration import ActivationStats


torch.manual_seed(0)


def makeCorrelatedCalibrationData(nTokens=2000, inFeatures=256, nOutlierChannels=4, outlierScale=20.0, basis=None, outlierChannels=None, seed=None):
  """simulate what real transformer activations look like: mostly small,
  correlated values, plus a handful of channels that run much hotter -
  this is the exact pattern smoothquant/llm.int8() target. a small full-rank
  noise floor is added on top of the low-rank correlated component so the
  resulting hessian stays invertible (a purely low-rank construction makes
  h singular in the null directions, which blows up ANY inverse-based
  method - not a fair test of gptq specifically).

  pass the SAME `basis` and `outlierChannels` for calibration and eval
  calls - real calibration text and real eval text come from the same
  domain/distribution, so the correlation structure gptq calibrates on
  should match what it's evaluated on. regenerating a fresh random
  subspace per call silently breaks that assumption and makes gptq look
  worse than rtn for reasons that have nothing to do with the algorithm."""
  if seed is not None:
    torch.manual_seed(seed)
  if basis is None:
    basis = torch.randn(inFeatures//4, inFeatures)
  base = torch.randn(nTokens, inFeatures//4)
  x = base@basis
  x = x+0.15*torch.randn(nTokens, inFeatures)
  x = x/x.std()

  if outlierChannels is None:
    outlierChannels = torch.randperm(inFeatures)[:nOutlierChannels]
  x[:, outlierChannels] *= outlierScale
  return x, outlierChannels, basis


def layerOutputMse(weight, x, weightHat):
  yTrue = x@weight.T
  yHat = x@weightHat.T
  return ((yTrue-yHat)**2).mean().item()


def test_rtnShapesAndDtypePreserved():
  w = torch.randn(64, 256)
  deq, scale, zp = fakeQuantizeRtn(w, bits=4, groupSize=128)
  assert deq.shape==w.shape
  assert scale.shape[0]==w.shape[0]
  print("test_rtnShapesAndDtypePreserved: pass")


def test_rtnErrorShrinksWithMoreBits():
  w = torch.randn(64, 256)
  x = torch.randn(500, 256)
  errs = {}
  for bits in [2, 4, 8]:
    deq, _, _ = fakeQuantizeRtn(w, bits=bits, groupSize=128)
    errs[bits] = layerOutputMse(w, x, deq)
  assert errs[2]>errs[4]>errs[8], errs
  print("test_rtnErrorShrinksWithMoreBits: pass", errs)


def test_gptqBeatsRtnOnCorrelatedData():
  inFeatures, outFeatures = 256, 64
  weight = torch.randn(outFeatures, inFeatures)
  x, outlierChannels, basis = makeCorrelatedCalibrationData(nTokens=3000, inFeatures=inFeatures, seed=0)

  stats = ActivationStats()
  stats.update(x)
  h = stats.hessian(damping=0.01)

  rtnDeq, _, _ = fakeQuantizeRtn(weight, bits=4, groupSize=128)
  gptqDeq = gptqQuantizeLayer(weight, h, bits=4, groupSize=128, damping=0.01)

  evalX, _, _ = makeCorrelatedCalibrationData(nTokens=1000, inFeatures=inFeatures, basis=basis, outlierChannels=outlierChannels)
  rtnMse = layerOutputMse(weight, evalX, rtnDeq)
  gptqMse = layerOutputMse(weight, evalX, gptqDeq)

  print(f"test_gptqBeatsRtnOnCorrelatedData: rtn_mse={rtnMse:.5f} gptq_mse={gptqMse:.5f}")
  assert gptqMse<rtnMse, "gptq should reduce layer-output error vs rtn at the same bit-width on correlated data"
  print("test_gptqBeatsRtnOnCorrelatedData: pass")


def test_smoothquantReducesActivationQuantErrorOnOutlierChannels():
  inFeatures, outFeatures = 256, 64
  weight = torch.randn(outFeatures, inFeatures)*0.1   # weights stay small/uniform, activations carry the outliers
  x, outlierChannels, _ = makeCorrelatedCalibrationData(nTokens=2000, inFeatures=inFeatures, nOutlierChannels=4, outlierScale=25.0, seed=1)

  actAbsMax = x.abs().amax(dim=0)

  # baseline: naive per-tensor int8 quant of raw activations, no smoothing, no outlier handling
  rawScale = x.abs().amax()/127
  qRaw = torch.round(x/rawScale).clamp(-127, 127)
  deqRaw = qRaw*rawScale
  rawMse = ((x-deqRaw)**2).mean().item()

  # smoothquant: migrate scale to weights, then quantize activations with outlier channels excluded
  wQ, s, detectedOutliers = applySmoothquantToLinear(weight, actAbsMax, bits=8, groupSize=128, alpha=0.5, outlierRatio=6.0)
  xSmoothed = x/s.unsqueeze(0)
  deqSmoothed = quantizeActivationsInt8Dynamic(xSmoothed, outlierIdx=detectedOutliers)
  deqSmoothedRescaled = deqSmoothed*s.unsqueeze(0)   # undo the smoothing to compare in the original activation space
  smoothMse = ((x-deqSmoothedRescaled)**2).mean().item()

  print(f"test_smoothquantReducesActivationQuantErrorOnOutlierChannels: raw_mse={rawMse:.4f} smooth_mse={smoothMse:.4f}")
  print(f"  known outlier channels: {sorted(outlierChannels.tolist())}, detected: {sorted(detectedOutliers.tolist())}")
  assert smoothMse<rawMse, "smoothing + outlier handling should reduce activation quantization error vs naive per-tensor int8"
  print("test_smoothquantReducesActivationQuantErrorOnOutlierChannels: pass")


def test_outlierDetectionFindsInjectedChannels():
  x, outlierChannels, _ = makeCorrelatedCalibrationData(nTokens=1000, inFeatures=128, nOutlierChannels=3, outlierScale=30.0, seed=2)
  actAbsMax = x.abs().amax(dim=0)
  detected = detectOutlierChannels(actAbsMax, ratioThreshold=6.0)
  overlap = set(outlierChannels.tolist())&set(detected.tolist())
  print(f"test_outlierDetectionFindsInjectedChannels: injected={sorted(outlierChannels.tolist())} detected={sorted(detected.tolist())}")
  assert len(overlap)==len(outlierChannels), "should detect all injected outlier channels"
  print("test_outlierDetectionFindsInjectedChannels: pass")


if __name__=="__main__":
  test_rtnShapesAndDtypePreserved()
  test_rtnErrorShrinksWithMoreBits()
  test_gptqBeatsRtnOnCorrelatedData()
  test_smoothquantReducesActivationQuantErrorOnOutlierChannels()
  test_outlierDetectionFindsInjectedChannels()
  print("\nall tests passed")
