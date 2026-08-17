import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
  cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
  cells.append(nbf.v4.new_code_cell(src.strip("\n")))

md("""# qwen quantization build matrix — kaggle t4x2

runs the harness in `../` against a real Qwen2.5 checkpoint: rtn and gptq weight quantization, smoothquant activation smoothing, a perplexity + downstream-task quality gate, and latency/memory benchmarking.

**runtime**: kaggle notebook settings -> accelerator -> gpu t4 x2, internet on (needed for the model download).

set `MODEL_NAME` below to switch between the 0.5b (fast iteration) and 1.5b (more representative quantization results) checkpoints.

note on gptq speed: gptq's calibration hessian is kept on cpu (fixes a cuda oom that otherwise hits on a 16gb t4 - see the readme), which makes the gptq quantization step slower than a pure-gpu version would be. if a full run feels like it's taking forever, drop `int8_gptq` from the techniques list below and just run `int4_gptq` first.""")

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

md("""## load model + tokenizer""")

code("""
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)
base_model.eval()

print(base_model.config.num_hidden_layers, "layers,", sum(p.numel() for p in base_model.parameters())/1e6, "M params")
""")

md("""## sanity check: generation works before we start quantizing anything""")

code("""
prompt = "the best way to reduce latency in a real-time voice agent is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
out = base_model.generate(input_ids, max_new_tokens=40, do_sample=False, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(out[0], skip_special_tokens=True))
""")

md("""## run the full build matrix

fp16 baseline, int8/int4 rtn, int8/int4 gptq. this can take a while on a t4 - gptq's calibration + per-column hessian correction is the slow part.""")

code("""
from bench.build_matrix import run_full_build_matrix

matrix = run_full_build_matrix(
    base_model, tokenizer, device=DEVICE, workload_tag="general",
    techniques=["fp16", "int8_rtn", "int4_rtn", "int4_gptq", "int8_gptq"],
)

print(matrix.summary_table())
""")

code("""
matrix.to_json("build_matrix_t4.json")
print("saved build_matrix_t4.json")
""")

md("""## smoothquant pass, separately

smoothquant modifies the activation path as well as the weights - the weight-quantizing function only touches stored weights, so activation quantization needs a forward hook to fully simulate.""")

code("""
import copy
import torch.nn as nn
from common.calibration import collect_calibration_stats, default_calibration_texts
from quantizers.smoothquant import quantize_model_weights_smoothquant_, quantize_activations_int8_dynamic

sq_model = copy.deepcopy(base_model)
calib_stats = collect_calibration_stats(sq_model, tokenizer, default_calibration_texts(), device=DEVICE)
sq_info = quantize_model_weights_smoothquant_(sq_model, calib_stats, bits=8, group_size=128, alpha=0.5, outlier_ratio=6.0)

print(f"smoothquant applied to {len(sq_info)} layers")

def make_smoothquant_hook(scale, outlier_idx):
  def hook(module, inputs):
    x = inputs[0]
    x_smoothed = x / scale.to(x.device)
    x_quantized = quantize_activations_int8_dynamic(x_smoothed, outlier_idx=outlier_idx)
    x_rescaled = x_quantized * scale.to(x.device)
    return (x_rescaled,) + inputs[1:]
  return hook

handles = []
for name, module in sq_model.named_modules():
  if name in sq_info:
    h = module.register_forward_pre_hook(make_smoothquant_hook(sq_info[name]["smoothing_scale"], sq_info[name]["outlier_idx"]))
    handles.append(h)

from eval.perplexity import compute_perplexity, default_eval_text
from eval.downstream_task import evaluate_multiple_choice, default_eval_set

sq_ppl = compute_perplexity(sq_model, tokenizer, default_eval_text(), device=DEVICE)
sq_acc = evaluate_multiple_choice(sq_model, tokenizer, default_eval_set(), device=DEVICE)
print("smoothquant int8 (weights + activations): ppl=", sq_ppl, "acc=", sq_acc)

for h in handles:
  h.remove()
""")

md("""## does the best build actually differ by hardware?

run this same notebook on a second gpu generation, merge the two `build_matrix_*.json` files below, and compare rankings.""")

code("""
import json
from common.build_record import BuildMatrix, BuildRecord, QualityMetrics, PerfMetrics

def load_build_matrix_json(path):
  m = BuildMatrix()
  with open(path) as f:
    records = json.load(f)
  for r in records:
    br = BuildRecord(
      build_id=r["build_id"], model_name=r["model_name"], technique=r["technique"], bits=r["bits"],
      group_size=r["group_size"], hardware_tag=r["hardware_tag"], workload_tag=r["workload_tag"],
      quality=QualityMetrics(**r["quality"]), perf=PerfMetrics(**r["perf"]),
      passed_quality_gate=r["passed_quality_gate"], quality_gate_reason=r["quality_gate_reason"],
    )
    m.add(br)
  return m

# once you have a second hardware tag's json, e.g. build_matrix_a100.json:
# combined = BuildMatrix()
# combined.records = load_build_matrix_json("build_matrix_t4.json").records + load_build_matrix_json("build_matrix_a100.json").records
# print(combined.compare_ranking_across_hardware("T4", "ADA"))
""")

md("""## next steps

- swap `default_calibration_texts()` / `default_eval_text()` / `default_eval_set()` for real calibration data and a real downstream benchmark via `datasets.load_dataset`
- add a sparsity module (2:4 pruning) once the quantization side feels solid
- speculative decoding for audio is still an open question - see the readme""")

nb["cells"] = cells
with open("/home/claude/quant_harness_fixed/notebooks/run_build_matrix.ipynb", "w") as f:
  nbf.write(nb, f)
print("notebook written, cells:", len(cells))
