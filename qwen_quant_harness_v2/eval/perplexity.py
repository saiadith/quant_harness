"""
eval/perplexity.py

Standard sliding-window perplexity: chunk held-out text into overlapping
windows, only score the non-overlapping "new" tokens per window (so early
tokens in a window, which have little context, don't unfairly inflate the
loss), and exponentiate the average negative log-likelihood.
"""

import torch


@torch.no_grad()
def computePerplexity(model, tokenizer, text, device="cpu", maxLength=1024, stride=512):
  encodings = tokenizer(text, return_tensors="pt")
  inputIds = encodings.input_ids.to(device)
  seqLen = inputIds.shape[1]

  nlls = []
  nTokens = 0
  prevEnd = 0

  for begin in range(0, seqLen, stride):
    end = min(begin+maxLength, seqLen)
    trgLen = end-prevEnd   # only score tokens we haven't scored yet
    ids = inputIds[:, begin:end]
    target = ids.clone()
    target[:, :-trgLen] = -100   # mask out the "already scored" context tokens

    out = model(ids, labels=target)
    nValid = (target!=-100).sum().item()
    if nValid>0:
      nlls.append(out.loss.float()*nValid)
      nTokens += nValid

    prevEnd = end
    if end==seqLen:
      break

  avgNll = torch.stack(nlls).sum()/max(nTokens, 1)
  return torch.exp(avgNll).item()


def defaultEvalText():
  """small, self-contained held-out text for a quick perplexity check.
  swap for a wikitext-2/c4 held-out split on kaggle for a real number -
  this is enough to sanity check the pipeline runs end to end."""
  return (
    "the history of computing spans several centuries, beginning with mechanical calculators "
    "and evolving through electromechanical devices before the invention of the electronic "
    "digital computer in the mid twentieth century. early computers occupied entire rooms and "
    "relied on vacuum tubes, which were later replaced by transistors and then integrated "
    "circuits, each generation dramatically reducing size while increasing processing power. "
    "the personal computer revolution of the nineteen seventies and eighties brought computing "
    "into homes and small businesses for the first time, fundamentally changing how people "
    "worked, communicated, and accessed information. the subsequent rise of the internet "
    "connected these individual machines into a global network, enabling instant communication "
    "and access to vast amounts of information regardless of physical location. mobile "
    "computing further extended this accessibility, putting devices far more powerful than "
    "early room-sized computers into the pockets of billions of people worldwide."
  )
