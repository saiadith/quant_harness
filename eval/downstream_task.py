"""
eval/downstream_task.py

Multiple-choice accuracy via log-likelihood scoring (the lm-eval-harness
approach): for each question, score every candidate answer by the model's
summed log-probability of that answer's tokens conditioned on the
question, and pick the highest-scoring one. This is closer to how these
models are actually graded in the literature than exact-match generation,
and it's cheap - no sampling, one forward pass per candidate.

Ships with a small self-contained set so the pipeline runs with zero
external dataset dependency; swap `default_eval_set()` for a real slice of
a downloaded dataset (e.g. arc-easy, piqa) on kaggle for a meaningful
number - the point here is a correctly-implemented SCORING harness, not a
large enough eval set to draw real conclusions from.
"""

import torch


@torch.no_grad()
def score_candidate(model, tokenizer, prompt, candidate, device="cpu"):
  """returns the average log-probability of `candidate`'s tokens given
  `prompt` as context. averaging (not summing) avoids penalizing longer
  candidates just for being longer."""
  full_text = prompt + candidate
  prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
  full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)

  n_prompt_tokens = prompt_ids.shape[1]
  n_candidate_tokens = full_ids.shape[1] - n_prompt_tokens
  if n_candidate_tokens <= 0:
    return float("-inf")

  out = model(full_ids)
  logits = out.logits[0]                      # (seq_len, vocab)
  log_probs = torch.log_softmax(logits.float(), dim=-1)

  # logits at position i predict token i+1
  candidate_positions = range(n_prompt_tokens - 1, full_ids.shape[1] - 1)
  candidate_target_ids = full_ids[0, n_prompt_tokens:]

  total_logprob = 0.0
  for pos, target_id in zip(candidate_positions, candidate_target_ids):
    total_logprob += log_probs[pos, target_id].item()

  return total_logprob / n_candidate_tokens


@torch.no_grad()
def evaluate_multiple_choice(model, tokenizer, eval_set, device="cpu"):
  """eval_set: list of {"prompt": str, "choices": [str, ...], "answer_idx": int}
  returns accuracy over the set."""
  correct = 0
  for item in eval_set:
    scores = [score_candidate(model, tokenizer, item["prompt"], choice, device=device) for choice in item["choices"]]
    pred = int(torch.tensor(scores).argmax())
    if pred == item["answer_idx"]:
      correct += 1
  return correct / len(eval_set)


def default_eval_set():
  """tiny hand-written multiple-choice set (common-sense + factual), enough
  to sanity-check the scoring pipeline end to end before swapping in a real
  benchmark subset on kaggle."""
  return [
    {
      "prompt": "the capital of france is",
      "choices": [" paris.", " berlin.", " madrid.", " rome."],
      "answer_idx": 0,
    },
    {
      "prompt": "water freezes at zero degrees",
      "choices": [" celsius.", " fahrenheit.", " kelvin.", " newton."],
      "answer_idx": 0,
    },
    {
      "prompt": "if you drop a glass on a hard floor, it will most likely",
      "choices": [" shatter.", " bounce gently.", " float.", " melt."],
      "answer_idx": 0,
    },
    {
      "prompt": "the sun rises in the",
      "choices": [" east.", " west.", " north.", " south."],
      "answer_idx": 0,
    },
    {
      "prompt": "to boil water for tea, you would typically use a",
      "choices": [" kettle.", " hammer.", " umbrella.", " pillow."],
      "answer_idx": 0,
    },
    {
      "prompt": "the largest planet in our solar system is",
      "choices": [" jupiter.", " mars.", " mercury.", " venus."],
      "answer_idx": 0,
    },
  ]
