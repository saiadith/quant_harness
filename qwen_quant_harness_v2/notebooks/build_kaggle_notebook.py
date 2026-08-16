import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
  cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
  cells.append(nbf.v4.new_code_cell(src.strip("\n")))

md("""# qwen quantization build matrix — kaggle t4x2

Runs the harness in `../` against a real Qwen2.5 checkpoint: RTN and GPTQ weight quantization, SmoothQuant activation smoothing, a perplexity + downstream-task quality gate, and latency/memory benchmarking — producing one `BuildRecord` per technique.

**Runtime**: Kaggle notebook settings -> Accelerator -> GPU T4 x2. This harness only needs one GPU; the second is handy for running a second hardware/precision comparison in parallel later, or for the speculative-decoding draft+verifier pair in a follow-up notebook.

Set `MODEL_NAME` below to switch between the 0.5B (fast iteration) and 1.5B (more representative quantization results) checkpoints.""")

code("""
!pip install -q transformers accelerate
""")

code("""
import sys
sys.path.append("..")

import torch
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
""")

md("""## load model + tokenizer

`MODEL_NAME = "Qwen/Qwen2.5-1.5B"` is the primary target (see the writeup for why 1.5B shows more realistic quantization effects than 0.5B on a T4). Switch to `"Qwen/Qwen2.5-0.5B"` for a fast dev loop while debugging the harness itself.""")

code("""
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
baseModel = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)
baseModel.eval()

print(baseModel.config.num_hidden_layers, "layers,", sum(p.numel() for p in baseModel.parameters())/1e6, "M params")
""")

md("""## sanity check: generation works before we start quantizing anything""")

code("""
prompt = "the best way to reduce latency in a real-time voice agent is"
inputIds = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
out = baseModel.generate(inputIds, max_new_tokens=40, do_sample=False, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(out[0], skip_special_tokens=True))
""")

md("""## run the full build matrix

fp16 baseline, int8/int4 rtn, int8/int4 gptq. each build is quantized from a fresh copy of the fp16 checkpoint (`buildAndEvaluate` deep-copies `baseModel`), evaluated for perplexity + downstream accuracy relative to the fp16 baseline, and benchmarked for tok/s, time-to-first-token, and peak memory. this takes a while on a T4 — GPTQ's calibration forward passes plus the per-column Hessian correction are the slow part; expect several minutes for 1.5B.""")

code("""
from bench.build_matrix import runFullBuildMatrix

matrix = runFullBuildMatrix(
    baseModel, tokenizer, device=DEVICE, workloadTag="general",
    techniques=["fp16", "int8_rtn", "int4_rtn", "int4_gptq", "int8_gptq"],
)

print(matrix.summaryTable())
""")

code("""
matrix.toJson("build_matrix_t4.json")
print("saved build_matrix_t4.json")
""")

md("""## smoothquant pass, separately

smoothquant modifies the activation path as well as the weights (see `quantizers/smoothquant.py`'s docstring — the module quantizes the weight side directly, and wrapping the live forward pass to also fake-quantize activations needs a forward hook rather than the copy-and-mutate pattern `buildAndEvaluate` uses for rtn/gptq). This cell shows that wiring explicitly with hooks, on the attention/mlp projection layers only (the layers where activation outliers actually show up).""")

code("""
import copy
import torch.nn as nn
from common.calibration import collectCalibrationStats, defaultCalibrationTexts
from quantizers.smoothquant import quantizeModelWeightsSmoothquant, quantizeActivationsInt8Dynamic

sqModel = copy.deepcopy(baseModel)
calibStats = collectCalibrationStats(sqModel, tokenizer, defaultCalibrationTexts(), device=DEVICE)
sqInfo = quantizeModelWeightsSmoothquant(sqModel, calibStats, bits=8, groupSize=128, alpha=0.5, outlierRatio=6.0)

print(f"smoothquant applied to {len(sqInfo)} layers")

# wrap each smoothed layer's forward to also fake-quantize its activations,
# dividing by the same smoothing scale baked into the weight
def makeSmoothquantHook(scale, outlierIdx):
  def hook(module, inputs):
    x = inputs[0]
    xSmoothed = x/scale.to(x.device)
    xQuantized = quantizeActivationsInt8Dynamic(xSmoothed, outlierIdx=outlierIdx)
    xRescaled = xQuantized*scale.to(x.device)
    return (xRescaled,)+inputs[1:]
  return hook

handles = []
for name, module in sqModel.named_modules():
  if name in sqInfo:
    h = module.register_forward_pre_hook(makeSmoothquantHook(sqInfo[name]["smoothingScale"], sqInfo[name]["outlierIdx"]))
    handles.append(h)

from eval.perplexity import computePerplexity, defaultEvalText
from eval.downstream_task import evaluateMultipleChoice, defaultEvalSet

sqPpl = computePerplexity(sqModel, tokenizer, defaultEvalText(), device=DEVICE)
sqAcc = evaluateMultipleChoice(sqModel, tokenizer, defaultEvalSet(), device=DEVICE)
print("smoothquant int8 (weights + activations): ppl=", sqPpl, "acc=", sqAcc)

for h in handles:
  h.remove()
""")

md("""## research question 1: does the best build actually differ by hardware?

run this same notebook on a second hardware tag (kaggle only gives T4 here — for a second point, an A10/L4/A100 on colab or a cloud spot instance works) and merge the two `build_matrix_*.json` files below. If the two hardware tags agree on the ranking, that's evidence the model-side quantization choice is fairly hardware-independent for this model size; if they disagree, it's evidence the "best build" genuinely depends on the serving GPU, which is the premise the whole build-matrix idea rests on.""")

code("""
import json
from common.build_record import BuildMatrix, BuildRecord, QualityMetrics, PerfMetrics

def loadBuildMatrixJson(path):
  m = BuildMatrix()
  with open(path) as f:
    records = json.load(f)
  for r in records:
    br = BuildRecord(
      buildId=r["buildId"], modelName=r["modelName"], technique=r["technique"], bits=r["bits"],
      groupSize=r["groupSize"], hardwareTag=r["hardwareTag"], workloadTag=r["workloadTag"],
      quality=QualityMetrics(**r["quality"]), perf=PerfMetrics(**r["perf"]),
      passedQualityGate=r["passedQualityGate"], qualityGateReason=r["qualityGateReason"],
    )
    m.add(br)
  return m

# once you have a second hardware tag's json, e.g. build_matrix_a100.json:
# combined = BuildMatrix()
# combined.records = loadBuildMatrixJson("build_matrix_t4.json").records + loadBuildMatrixJson("build_matrix_a100.json").records
# print(combined.compareRankingAcrossHardware("T4", "ADA"))
""")

md("""## research question 2: is speculative decoding lossless for audio?

Out of scope for this notebook (needs a draft/verifier pair and a speech-token vocabulary), but the harness's `BuildRecord` schema already has room for it: a speculative-decoding build would report a *higher* `tokensPerSec` at the *same* `perplexity`/`downstreamAccuracy` as its non-speculative counterpart if and only if the acceptance criterion is exact (lossless). The natural follow-up experiment is comparing greedy-equivalent vs relaxed acceptance thresholds and watching exactly where `perplexityDeltaPct` stops being ~0 — see [3] in the JD's reading list (ICASSP 2026, "Principled Coarse-Grained Acceptance for Speculative Decoding in Speech") for the audio-specific framing of this question.""")

md("""## next steps

- swap `defaultCalibrationTexts()` / `defaultEvalText()` / `defaultEvalSet()` for real calibration data (a slice of c4/wikitext) and a real downstream benchmark (arc-easy, piqa) via `datasets.load_dataset` once running on kaggle with network access
- extend `TECHNIQUES` in `bench/build_matrix.py` with an `nvfp4`/2:4-sparse entry once you've validated the sparsity path (`quantizers/` doesn't have a sparsity module yet — natural phase-4 addition)
- try the same build matrix against `Qwen/Qwen2.5-0.5B` for a fast-iteration comparison, and see whether the 0.5B ranking agrees with 1.5B's - a second flavor of the "does the best build depend on X" question, this time X = model size instead of hardware""")

nb["cells"] = cells
with open("/home/claude/qwen_quant_harness_v2/notebooks/run_build_matrix.ipynb", "w") as f:
  nbf.write(nb, f)
print("notebook written, cells:", len(cells))
