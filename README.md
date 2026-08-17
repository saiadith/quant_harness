# qwen quantization build matrix

i built this to prep for a role that wanted exactly this: given the hardware and a latency budget, which quantized build am i actually allowed to serve? so instead of just calling bitsandbytes or AutoGPTQ, i wrote the actual quantization math myself - round-to-nearest (rtn), gptq's hessian-based error correction, and smoothquant's activation smoothing + outlier handling - and unit tested all of it on synthetic data before ever touching a real model.

## what's in here

```
common/
  build_record.py    - the schema for a "build": what it is (technique, bits,
                        group size) and what hardware/workload it's valid for,
                        plus the query logic for "which build should i serve"
  calibration.py      - hooks every linear layer during calibration text and
                        collects per-channel activation stats + the gptq hessian
quantizers/
  rtn.py               - round-to-nearest, the baseline everything else has to beat
  gptq_lite.py         - my simplified gptq (frantar et al 2022) - sequential,
                        hessian-weighted error correction, no autogptq import
  smoothquant.py       - activation smoothing + outlier channel fallback
                        (xiao et al 2022, plus an llm.int8()-style escape hatch)
eval/
  perplexity.py        - sliding-window perplexity on held-out text
  downstream_task.py   - multiple-choice accuracy via log-likelihood scoring
bench/
  latency.py           - tok/s, time to first token, peak memory
  build_matrix.py       - runs quantize -> eval -> benchmark across every
                        technique and spits out a BuildRecord for each
tests/
  test_quantizers.py            - checks the actual math on synthetic linear
                                  layers, no gpu needed. confirms gptq really
                                  does beat rtn on correlated data, and
                                  smoothquant really does kill outlier error
  test_pipeline_integration.py  - checks the wiring end to end against a tiny
                                  random model, no internet needed
notebooks/
  run_build_matrix.ipynb  - the actual kaggle notebook, loads real qwen2.5
                            and runs the whole thing on gpu
```

## run the tests first, seriously

```bash
python3 tests/test_quantizers.py
python3 tests/test_pipeline_integration.py
```

both take a few seconds, run on cpu, no model download needed. i'd catch bugs here way before wasting gpu time on kaggle - i actually did catch a real bug this way (my synthetic eval set was accidentally sampling a different random subspace than calibration, which made gptq look worse than rtn for reasons that had nothing to do with the algorithm).

## running it for real

open `notebooks/run_build_matrix.ipynb` on kaggle with a t4x2 gpu turned on. it installs transformers, loads `Qwen/Qwen2.5-1.5B` (swap to the 0.5b checkpoint if you just want a faster loop while debugging), and runs the whole build matrix - fp16 baseline, int8/int4 rtn, int8/int4 gptq - plus a separate smoothquant pass with the activation quantization wired in via forward hooks.

## why 1.5b and not 0.5b

0.5b runs fine on a t4 but it's small enough that quantization barely hurts it, so the numbers don't really tell you anything. 1.5b (about 3gb in fp16) still fits comfortably on a 16gb t4 with room to spare, and actually shows realistic degradation and outlier behavior. i kept 0.5b around as a fast dev loop while poking at the harness itself.

## a real bug i hit and how i fixed it

first real run on kaggle, gptq calibration blew up with a cuda oom. turns out the calibration hook was building a full hessian (in_features x in_features) for every single linear layer in the model and keeping all of them on the gpu at once - for qwen's mlp layers that's thousands of dimensions per hessian, times a couple dozen layers, and it just didn't fit alongside the model weights.

fix: the hessian accumulation now happens on cpu instead. the forward pass itself still runs on gpu (that part's fast), but the stats that actually need to stick around get moved off gpu immediately. costs a bit of speed on the gptq quantization step itself since it now does its per-column math on cpu, but it doesn't crash anymore, which i'll take.

## stuff i simplified on purpose

- **rtn and gptq run in "fake quant" mode** - i quantize then immediately dequantize back to float, instead of packing into real int4/int8 storage with a custom gemm kernel. this isolates "does this preserve quality" from "does this actually run faster on real hardware," which is really a separate systems problem (that's what llm-compressor / nvidia model-optimizer solve with real kernels)
- **my gptq is a plain per-column python loop**, not the paper's blocked cholesky implementation. correct, but not vectorized, so it's slower than it could be - fine at 1.5b scale, would need blocking for anything bigger
- **smoothquant's activation side needs a forward hook** to fully simulate - the weight-quantizing function only touches the stored weights, the activation quantization has to happen live during the forward pass (see the dedicated cell in the notebook)
- **the quality gate thresholds are placeholders** (5% ppl delta, 2% downstream delta) - i'd tune these against real wer/utmos sensitivity if i had an actual speech pipeline to test against

## open questions i'm still poking at

1. **does the best build actually depend on the hardware?** run the notebook on two different gpu generations, merge the two `build_matrix_*.json` files, call `BuildMatrix.compare_ranking_across_hardware()` and see if they agree
2. **is speculative decoding lossless for audio?** haven't built this part yet - see the relevant notebook cell for how i'm thinking about extending the BuildRecord schema for it
