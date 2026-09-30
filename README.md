# mlxtune

**Fine-tune LLMs on your Mac with one command.** Built on [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm), with defaults that actually fit in 8 GB.

```bash
pip install mlxtune
mlxtune train --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --data my_data.jsonl
mlxtune chat adapters/run
```

[![CI](https://github.com/Dolmaa24/mlxtune/actions/workflows/ci.yml/badge.svg)](https://github.com/Dolmaa24/mlxtune/actions)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

## Why

mlx-lm already ships a capable LoRA trainer. What it doesn't do is tell you *what will fit on your Mac*, accept the dataset you already have, or get you from "adapter trained" to "model I can talk to" without reading three READMEs. mlxtune fills that gap:

- **Memory-aware defaults.** `batch_size`, `max_seq_length`, `num_layers` and gradient checkpointing default to `auto` and resolve from your RAM. Before training starts you get an estimated peak (calibrated on real runs), and after it finishes you get the measured one.
- **`mlxtune models`** — a curated list of 4-bit models with a yes / tight / no verdict for *this* machine.
- **Any common dataset format** — `messages`, alpaca, ShareGPT, prompt/completion, plain text; from `.jsonl` / `.json` / `.csv` / `.txt`, an mlx-lm style directory, or a Hub dataset id. Converted to what mlx-lm expects, with bad rows dropped and counted rather than crashing.
- **`mlxtune validate`** — see the exact text the model will train on and how long it is, before you spend the time.
- **One path to a usable model** — `train` → `eval` → `chat` → `fuse` → `export` (GGUF for any llama.cpp-supported architecture, plus an Ollama Modelfile).
- **`mlxtune eval`** answers "did it help?" — the same validation loss mlx-lm trains against, tuned vs base, with sample generations side by side.
- **Reproducible** — every run directory gets the resolved config, metrics with peak memory and throughput, and a README.
  It also holds `data/train.jsonl` and `data/valid.jsonl`, the exact converted split the run used — handy for `resume`, but it means **sharing or uploading an adapter directory shares your training data**. Delete `data/` first if that matters.

Measured on an M2 with 8 GB: Qwen2.5-0.5B-4bit trains at ~12 it/s with 0.5 GB peak on short examples; Qwen2.5-1.5B-4bit on 300 rows of `yahma/alpaca-cleaned` (up to 645 tokens) ran at 1.8 it/s with **2.1 GB peak**, val loss 1.31 → 1.08, held-out perplexity 3.70 → 2.93 vs base. 3B fits with the same settings.

## Install

> **Not on PyPI yet.** `pip install mlxtune` starts working when v0.1.0 is published. Until then install from GitHub:
> ```bash
> pip install "mlxtune @ git+https://github.com/Dolmaa24/mlxtune"
> ```


Apple Silicon Mac, macOS 13.5+, Python 3.10+.

```bash
pip install mlxtune              # or: uv pip install mlxtune
pip install "mlxtune[hub]"       # + load datasets from the Hugging Face Hub by id
```

From source:

```bash
git clone https://github.com/Dolmaa24/mlxtune && cd mlxtune
pip install -e ".[dev]"
```

## Quickstart

```bash
mlxtune check                 # your chip, RAM, and the defaults mlxtune will use
mlxtune models                # which models fit, with estimated peak memory

# check your data: format detection, token stats, one rendered example
mlxtune validate --data examples/pirate.jsonl

# train (adapters land in adapters/pirate; the tiny example takes ~30 s of compute)
mlxtune train --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --data examples/pirate.jsonl \
    --output adapters/pirate --set train.epochs=3 --set train.lr=1e-4

mlxtune eval adapters/pirate                       # held-out loss + samples, tuned vs base
mlxtune chat adapters/pirate                       # talk to it
mlxtune fuse adapters/pirate --output pirate-model # standalone MLX model
```

Prefer a config file for anything you'll run twice:

```bash
mlxtune init config.yaml
mlxtune train config.yaml --set lora.rank=16
```

Ready-made configs in [`configs/`](configs/):

| config | model | est. peak | for |
|---|---|---|---|
| `smoke.yaml` | Qwen2.5-0.5B | ~1.4 GB | any Mac, proves the pipeline in a minute |
| `8gb-qwen2.5-1.5b.yaml` | Qwen2.5-1.5B | ~3 GB | 8 GB Macs |
| `16gb-qwen2.5-7b.yaml` | Qwen2.5-7B | ~11 GB | 16 GB Macs |
| `32gb-llama-3.1-8b-dora.yaml` | Llama-3.1-8B, DoRA | ~17 GB | 32 GB Macs |

## What fits?

`mlxtune models` prints this for your machine. Estimates use the tier's default settings and are calibrated against measured runs (±30 % for models much larger than the reference):

| RAM | defaults (batch × seq, layers) | comfortable | tight |
|---|---|---|---|
| 8 GB | 1 × 1024, 8 | up to 3B (Qwen2.5-3B, Llama-3.2-3B, Phi-3.5-mini) | — |
| 16 GB | 1 × 2048, 16 | up to 3B | 7–8B (Qwen2.5-7B, Llama-3.1-8B) |
| 32 GB | 2 × 2048, 16 | 7–8B | 14B |
| 64 GB+ | 4 × 2048, 16 | 14B | 32B |

Two things that matter more than you'd expect:

1. **Vocabulary size.** The logits and their gradient cost ~1.2 GB per 1 000 tokens per step for a 150k-vocab model (Qwen), ~1.0 GB for Llama 3, ~0.25 GB for Phi. This is why a 0.5B model still peaks at 2 GB with 1k-token examples, and why `max_seq_length × batch_size` is the first knob to turn.
2. **`num_layers`.** Only the top N transformer layers get adapters and gradients. 8 is plenty for style/format tuning; 16 for more; `-1` for all.

If a run prints a peak memory well under your RAM, raise `batch_size` (faster) or `max_seq_length` (longer examples). If it swaps or crashes, lower them.

## Data formats

The first row decides the converter. All conversational formats become `messages`, and the model's own chat template is applied, so training and inference see the same prompt format.

| format | row shape | notes |
|---|---|---|
| `messages` | `{"messages": [{"role": "user", "content": "…"}, {"role": "assistant", "content": "…"}]}` | canonical; multi-turn ok |
| `alpaca` | `{"instruction": "…", "input": "…", "output": "…"}` | also `question`/`answer`, `prompt`/`response`, optional `system` |
| `sharegpt` | `{"conversations": [{"from": "human", "value": "…"}, {"from": "gpt", "value": "…"}]}` | |
| `prompt_completion` | `{"prompt": "…", "completion": "…"}` | no chat template applied |
| `text` | `{"text": "…"}` | plain continued pre-training; `mask_prompt` is ignored |

A directory containing `train.jsonl` (and optionally `valid.jsonl`) in mlx-lm's own layout is used as-is, so existing mlx-lm datasets work unchanged.

`train.mask_prompt: true` (the default) computes loss only on assistant turns / completions, which is what you want for chat tuning.

## Configuration reference

```yaml
model: mlx-community/Qwen2.5-1.5B-Instruct-4bit   # mlx-community repo, any HF model (converted on load), or local path

data:
  path: data/train.jsonl
  format: auto                  # auto | messages | alpaca | sharegpt | prompt_completion | text
  eval_fraction: 0.05           # ignored if the directory has its own valid.jsonl
  max_samples: null
  system_prompt: null
  shuffle_seed: 42
  text_field: text              # column names, when auto-detection can't find them
  prompt_field: prompt
  completion_field: completion

lora:
  type: lora                    # lora | dora | full
  rank: 8
  scale: 20.0                   # MLX uses a direct scale, not alpha/rank
  dropout: 0.0
  num_layers: auto              # layers (from the top) that get adapters; -1 = all

train:
  output: adapters/run
  epochs: 1
  iters: null                   # overrides epochs when set
  batch_size: auto
  max_seq_length: auto
  grad_checkpoint: auto
  grad_accumulation_steps: 1
  lr: 1.0e-5                    # mlx-lm's default; 1e-4 for small datasets / bigger shifts
  optimizer: adam               # adam | adamw | sgd | adafactor | muon
  warmup_steps: 0               # > 0 enables warmup + cosine decay
  mask_prompt: true
  steps_per_report: 10
  steps_per_eval: null          # null = 4 evals per run
  val_batches: 25
  save_every: null              # null = every eval
  seed: 0
  resume: null                  # path to adapters.safetensors
```

Everything is overridable with `--set section.key=value`; `--model`, `--data`, `--output` are shortcuts.

## Commands

| command | what it does |
|---|---|
| `mlxtune check` | chip, RAM, tier and the auto defaults |
| `mlxtune models [--all]` | curated models with fit verdicts and memory estimates |
| `mlxtune init [path]` | commented starter config |
| `mlxtune validate` | convert the dataset, show drop reasons, token stats, rendered example |
| `mlxtune train` | train; `--dry-run` converts data and prints the plan only |
| `mlxtune chat <path>` | interactive chat with an adapter dir, fused dir, or Hub id |
| `mlxtune eval <path>` | held-out loss / perplexity + sample generations, tuned vs base; data taken from the run's `mlxtune.yaml` unless `--data` is given |
| `mlxtune fuse <adapter>` | merge into a standalone model; `--dequantize` for export; `--gguf out.gguf` for llama-family |
| `mlxtune export <fused>` | GGUF via llama.cpp's converter (any architecture it supports) + `--ollama` Modelfile |
| `mlxtune info <run dir>` | metrics from a finished run |

## Export to Ollama / llama.cpp

Two routes. For any architecture llama.cpp supports (Qwen, Phi, Gemma, Llama, Mistral…), fuse in full precision and run llama.cpp's converter through `mlxtune export`:

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
pip install "mlxtune[gguf]"                                   # torch + sentencepiece + protobuf, which the converter imports
mlxtune fuse adapters/run --output fused --dequantize
mlxtune export fused --quant q8_0 --ollama                    # finds ~/llama.cpp automatically (or --llama-cpp PATH)
ollama create my-model -f fused/Modelfile && ollama run my-model
```

This route was verified end-to-end on an M2 with a Qwen2.5-0.5B adapter: the Ollama model answered in the fine-tuned style. `export` refuses a fused directory that still holds 4-bit MLX weights and tells you to re-fuse with `--dequantize`. Don't install llama.cpp's own pinned `requirements-convert_hf_to_gguf.txt` into the mlxtune environment; if you want their pinned versions, put them in `~/llama.cpp/.venv` and mlxtune will use that interpreter.

For `llama`, `mistral` and `mixtral` architectures only, mlx-lm can also write GGUF itself, with no llama.cpp checkout:

```bash
mlxtune fuse adapters/run --output fused --gguf model.gguf --ollama
```

## Python API

```python
from mlxtune.config import RunConfig
from mlxtune.train import run
from mlxtune.inference import load_for_inference, stream_reply

out = run(RunConfig.from_dict({"model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit", "data": "data.jsonl"}))
model, tok = load_for_inference(str(out))
print("".join(stream_reply(model, tok, [{"role": "user", "content": "hello"}])))
```

## FAQ

**It's memorising my examples / answering everything the same way.** Small dataset + several epochs + `lr: 1e-4` will do that (the bundled pirate example does it on purpose — ask it the capital of Japan). Use more varied data, fewer epochs, or drop back to `lr: 1e-5`.

**Loss barely moves.** `lr: 1e-5` with `scale: 20` is conservative. Try `lr: 1e-4`, `rank: 16`, or more `num_layers`.

**Can I use a non-mlx-community model?** Yes — mlx-lm converts Hugging Face models on load, but un-quantised weights use 4× the memory of 4-bit ones. Quantise first with `python -m mlx_lm convert --hf-path <repo> -q`.

**Does this work on Intel Macs / Linux / Windows?** No; MLX is Apple Silicon only. For NVIDIA GPUs and Colab, see the sibling project [tunekit](../tunekit).

## Contributing

Hardware reports are the most useful contribution: run something, then open a [hardware report](.github/ISSUE_TEMPLATE/hardware_report.md) with the estimate vs the measured peak. That data tightens the estimator and tier defaults for everyone. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. Models and datasets keep their own licenses.
