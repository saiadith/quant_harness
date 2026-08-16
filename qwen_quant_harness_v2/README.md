# Qwen Quantization Build Matrix

A build-matrix quantization harness for small LLMs, built around the same
question the target role's internship JD poses: given the hardware and a
latency budget, which build are we allowed to serve? Every quantization
technique is implemented from scratch (no bitsandbytes/AutoGPTQ import for
the actual math — RTN, GPTQ's Hessian-based error correction, and
SmoothQuant's activation smoothing + outlier handling are all hand-written
and unit-tested against synthetic data before ever touching a real model).

## Code style

2-space indent, no spaces around commas or `=` (assignment, keyword args,
dataclass fields). All functions we define ourselves use camelCase with no
underscores. The two exceptions:

- **framework-required names** — `forward`, `__init__`, `__call__`,
  `__getattr__`, and similar dunder/protocol methods keep the name PyTorch
  or Python itself requires
- **pytest discovery** — test functions keep the `test_` prefix (pytest's
  default `python_functions = test_*` pattern needs the literal
  underscore) but drop underscores after that, e.g. `test_gptqBeatsRtnOnCorrelatedData`

## Repo structure

```
common/
  build_record.py    - the BuildRecord/BuildMatrix schema: what a build IS
                        and what hardware/workload it's valid for, plus the
                        "which build should I serve" query logic
  calibration.py      - forward-hook based collection of per-channel
                        activation stats + the GPTQ Hessian
quantizers/
  rtn.py               - round-to-nearest weight-only quantization (the baseline)
  gptq_lite.py         - Hessian-based sequential quantization with error
                        propagation (simplified GPTQ, Frantar et al. 2022)
  smoothquant.py       - activation smoothing + outlier channel handling
                        (SmoothQuant, Xiao et al. 2022, + LLM.int8()-style
                        outlier fallback)
eval/
  perplexity.py        - sliding-window perplexity on held-out text
  downstream_task.py   - multiple-choice accuracy via log-likelihood scoring
bench/
  latency.py           - tok/s, time-to-first-token, peak memory
  build_matrix.py       - orchestrates quantize -> eval -> benchmark into
                        BuildRecords across every technique
tests/
  test_quantizers.py            - validates the quantization MATH on
                                  synthetic linear layers (no model/GPU
                                  needed) - confirms GPTQ beats RTN on
                                  correlated data, SmoothQuant kills
                                  outlier-channel quant error, etc.
  test_pipeline_integration.py  - validates the WIRING (calibration hooks
                                  -> quantize -> eval -> bench -> build
                                  matrix) against a tiny random Llama-arch
                                  model, no internet access needed
notebooks/
  run_build_matrix.ipynb  - the kaggle notebook: loads a real Qwen2.5
                            checkpoint and runs the full pipeline on GPU
```

Module/file names stay snake_case (Python's own convention for filenames);
only the functions and methods defined inside them are camelCase.

## Run the tests first

```bash
python3 tests/test_quantizers.py            # ~seconds, cpu, no model download
python3 tests/test_pipeline_integration.py   # ~seconds, cpu, no model download
```

Both suites are designed to catch bugs in the quantization math and the
pipeline wiring *before* you spend GPU time on Kaggle. `test_quantizers.py`
in particular validates the actual research claims: GPTQ reduces
layer-output error vs RTN at the same bit-width on correlated (realistic)
activation data, and SmoothQuant + outlier handling nearly eliminates the
activation-quantization error that outlier channels would otherwise cause.

## Run the real thing

Open `notebooks/run_build_matrix.ipynb` on Kaggle with a T4 x2 GPU
accelerator. It installs `transformers`, loads `Qwen/Qwen2.5-1.5B` (or
swap to `0.5B` for faster iteration), and runs the full build matrix:
fp16 baseline, int8/int4 RTN, int8/int4 GPTQ, plus a separate SmoothQuant
pass with activation quantization wired in via forward hooks.

## Why 1.5B over 0.5B on a T4

0.5B fits and runs easily, but quantization degradation and activation
outliers are both weaker at that scale — the harness would "work" but the
numbers wouldn't tell you much. 1.5B (~3GB in fp16) still fits comfortably
on a single 16GB T4 with room for calibration/eval overhead, and shows
much more realistic quantization behavior. Keep 0.5B configured as a fast
dev loop while you're debugging the harness itself.

## Design notes / where this is intentionally simplified

- **RTN and GPTQ run in "fake quant" mode**: weights are quantized then
  immediately dequantized back to float, rather than packed into real
  int4/int8 storage with a custom GEMM kernel. This isolates the
  *quality* question (does this technique preserve model behavior) from
  the *systems* question (does this technique actually run faster on
  real hardware) — the latter is exactly what `llm-compressor` /
  NVIDIA Model-Optimizer solve with real kernels, and is the natural next
  layer to add once the quantization math itself is validated.
- **GPTQ here is a straightforward per-column loop**, not the paper's
  blockwise-Cholesky-batched implementation — correct, but O(d^2) per
  layer in Python loops rather than vectorized blocks. Fine for 1.5B-scale
  layers; would need blocking for much larger models.
- **SmoothQuant's activation path needs a forward hook** to fully
  simulate (see the notebook's dedicated cell) since `quantizeModelWeightsSmoothquant`
  only touches the stored weights — the activation-side quantization has
  to happen live during the forward pass.
- **The quality gate defaults are placeholders** (5% perplexity delta, 2%
  downstream accuracy delta) — tune these against what your actual speech
  pipeline's downstream WER/UTMOS sensitivity looks like once you have one.

## Open research questions this harness is built to help answer

1. **Does the best build actually differ by hardware?** Run the notebook
   on two different GPU generations, merge the resulting
   `build_matrix_*.json` files, and call
   `BuildMatrix.compareRankingAcrossHardware()`.
2. **Is speculative decoding lossless for audio?** Not implemented here —
   see the "research question 2" cell in the notebook for how the
   `BuildRecord` schema is meant to extend to this, and [3] in the JD's
   reading list for the audio-specific framing of the problem.
