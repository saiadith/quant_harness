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
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from quantizers.rtn import fake_quantize_rtn
from quantizers.gptq_lite import gptq_quantize_layer
from quantizers.smoothquant import apply_smoothquant_to_linear, quantize_activations_int8_dynamic, detect_outlier_channels
from common.calibration import ActivationStats


torch.manual_seed(0)


def make_correlated_calibration_data(n_tokens=2000, in_features=256, n_outlier_channels=4, outlier_scale=20.0, basis=None, outlier_channels=None, seed=None):
  """simulate what real transformer activations look like: mostly small,
  correlated values, plus a handful of channels that run much hotter -
  this is the exact pattern smoothquant/llm.int8() target. a small full-rank
  noise floor is added on top of the low-rank correlated component so the
  resulting hessian stays invertible (a purely low-rank construction makes
  H singular in the null directions, which blows up ANY inverse-based
  method - not a fair test of gptq specifically).

  pass the SAME `basis` and `outlier_channels` for calibration and eval
  calls - real calibration text and real eval text come from the same
  domain/distribution, so the correlation structure gptq calibrates on
  should match what it's evaluated on. regenerating a fresh random
  subspace per call (the original version of this function) silently
  breaks that assumption and makes gptq look worse than rtn for reasons
  that have nothing to do with the algorithm - a good reminder that
  calibration/eval distribution mismatch is a real failure mode to
  watch for later with actual text data too."""
  if seed is not None:
    torch.manual_seed(seed)
  if basis is None:
    basis = torch.randn(in_features // 4, in_features)
  base = torch.randn(n_tokens, in_features // 4)
  x = base @ basis
  x = x + 0.15 * torch.randn(n_tokens, in_features)
  x = x / x.std()

  if outlier_channels is None:
    outlier_channels = torch.randperm(in_features)[:n_outlier_channels]
  x[:, outlier_channels] *= outlier_scale
  return x, outlier_channels, basis


def layer_output_mse(weight, x, weight_hat):
  y_true = x @ weight.T
  y_hat = x @ weight_hat.T
  return ((y_true - y_hat) ** 2).mean().item()


def test_rtn_shapes_and_dtype_preserved():
  w = torch.randn(64, 256)
  deq, scale, zp = fake_quantize_rtn(w, bits=4, group_size=128)
  assert deq.shape == w.shape
  assert scale.shape[0] == w.shape[0]
  print("test_rtn_shapes_and_dtype_preserved: pass")


def test_rtn_error_shrinks_with_more_bits():
  w = torch.randn(64, 256)
  x = torch.randn(500, 256)
  errs = {}
  for bits in [2, 4, 8]:
    deq, _, _ = fake_quantize_rtn(w, bits=bits, group_size=128)
    errs[bits] = layer_output_mse(w, x, deq)
  assert errs[2] > errs[4] > errs[8], errs
  print("test_rtn_error_shrinks_with_more_bits: pass", errs)


def test_gptq_beats_rtn_on_correlated_data():
  in_features, out_features = 256, 64
  weight = torch.randn(out_features, in_features)
  x, outlier_channels, basis = make_correlated_calibration_data(n_tokens=3000, in_features=in_features, seed=0)

  stats = ActivationStats()
  stats.update(x)
  H = stats.hessian(damping=0.01)

  rtn_deq, _, _ = fake_quantize_rtn(weight, bits=4, group_size=128)
  gptq_deq = gptq_quantize_layer(weight, H, bits=4, group_size=128, damping=0.01)

  eval_x, _, _ = make_correlated_calibration_data(n_tokens=1000, in_features=in_features, basis=basis, outlier_channels=outlier_channels)
  rtn_mse = layer_output_mse(weight, eval_x, rtn_deq)
  gptq_mse = layer_output_mse(weight, eval_x, gptq_deq)

  print(f"test_gptq_beats_rtn_on_correlated_data: rtn_mse={rtn_mse:.5f} gptq_mse={gptq_mse:.5f}")
  assert gptq_mse < rtn_mse, "gptq should reduce layer-output error vs rtn at the same bit-width on correlated data"
  print("test_gptq_beats_rtn_on_correlated_data: pass")


def test_smoothquant_reduces_activation_quant_error_on_outlier_channels():
  in_features, out_features = 256, 64
  weight = torch.randn(out_features, in_features) * 0.1   # weights stay small/uniform, activations carry the outliers
  x, outlier_channels, _ = make_correlated_calibration_data(n_tokens=2000, in_features=in_features, n_outlier_channels=4, outlier_scale=25.0, seed=1)

  act_absmax = x.abs().amax(dim=0)

  # baseline: naive per-tensor int8 quant of raw activations, no smoothing, no outlier handling
  raw_scale = x.abs().amax() / 127
  q_raw = torch.round(x / raw_scale).clamp(-127, 127)
  deq_raw = q_raw * raw_scale
  raw_mse = ((x - deq_raw) ** 2).mean().item()

  # smoothquant: migrate scale to weights, then quantize activations with outlier channels excluded
  w_q, s, detected_outliers = apply_smoothquant_to_linear(weight, act_absmax, bits=8, group_size=128, alpha=0.5, outlier_ratio=6.0)
  x_smoothed = x / s.unsqueeze(0)
  deq_smoothed = quantize_activations_int8_dynamic(x_smoothed, outlier_idx=detected_outliers)
  deq_smoothed_rescaled = deq_smoothed * s.unsqueeze(0)   # undo the smoothing to compare in the original activation space
  smooth_mse = ((x - deq_smoothed_rescaled) ** 2).mean().item()

  print(f"test_smoothquant_reduces_activation_quant_error_on_outlier_channels: raw_mse={raw_mse:.4f} smooth_mse={smooth_mse:.4f}")
  print(f"  known outlier channels: {sorted(outlier_channels.tolist())}, detected: {sorted(detected_outliers.tolist())}")
  assert smooth_mse < raw_mse, "smoothing + outlier handling should reduce activation quantization error vs naive per-tensor int8"
  print("test_smoothquant_reduces_activation_quant_error_on_outlier_channels: pass")


def test_outlier_detection_finds_injected_channels():
  x, outlier_channels, _ = make_correlated_calibration_data(n_tokens=1000, in_features=128, n_outlier_channels=3, outlier_scale=30.0, seed=2)
  act_absmax = x.abs().amax(dim=0)
  detected = detect_outlier_channels(act_absmax, ratio_threshold=6.0)
  overlap = set(outlier_channels.tolist()) & set(detected.tolist())
  print(f"test_outlier_detection_finds_injected_channels: injected={sorted(outlier_channels.tolist())} detected={sorted(detected.tolist())}")
  assert len(overlap) == len(outlier_channels), "should detect all injected outlier channels"
  print("test_outlier_detection_finds_injected_channels: pass")


if __name__ == "__main__":
  test_rtn_shapes_and_dtype_preserved()
  test_rtn_error_shrinks_with_more_bits()
  test_gptq_beats_rtn_on_correlated_data()
  test_smoothquant_reduces_activation_quant_error_on_outlier_channels()
  test_outlier_detection_finds_injected_channels()
  print("\nall tests passed")
