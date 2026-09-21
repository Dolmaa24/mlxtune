# Contributing to mlxtune

The goal is that someone with a Mac and a dataset gets a working fine-tune with one command and no surprises. Every change should make that more true.

## Setup

```bash
git clone https://github.com/Dolmaa24/mlxtune && cd mlxtune
python -m venv .venv && source .venv/bin/activate      # or: uv venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Checks

```bash
ruff check src tests && ruff format src tests
pytest -m "not integration"      # fast, offline
pytest -m integration            # downloads Qwen2.5-0.5B-4bit (~300 MB), trains 3 iters (~10 s)
```

CI runs all of these on GitHub's M-series macOS runners.

## Layout

```
src/mlxtune/
  hardware.py   Mac detection, RAM tiers + defaults, curated model list, memory estimator
  config.py     pydantic schema, 'auto' resolution, --set overrides
  data.py       any format -> mlx-lm's train/valid.jsonl
  train.py      maps config onto mlx_lm.lora args, runs train_model, records metrics
  inference.py  chat, fuse, tokenizer-only loading, Ollama Modelfile
  cli.py        typer commands
configs/        example configs per RAM tier
examples/       tiny datasets in each supported format
tests/          unit tests + one integration test
```

## The most useful contribution: hardware reports

The memory estimator (`hardware.estimate_train_gb`) is calibrated on one machine and one small model. Every measured run on a different Mac or model makes it better. Open a [hardware report issue](.github/ISSUE_TEMPLATE/hardware_report.md) with your `mlxtune check`, the model, settings, and the `peak_memory_gb` from `mlxtune info`. If you're comfortable with it, adjust the constants in `hardware.py` and add the measurement to `tests/test_config_hardware.py::test_estimator_tracks_measurements`.

## Other good first contributions

- A model added to `MODELS` in `hardware.py` — needs the repo to exist, the safetensors size, and the vocab size (`config.json` → `vocab_size`).
- A new dataset format in `data.py` with tests.
- A config for a machine/model combo you've actually run, with the measured peak in the header comment.

## Roadmap (help wanted)

- [x] `mlxtune eval` — held-out loss / perplexity + sample generations, tuned vs base
- [ ] Auto-tune: try a few `(batch_size, max_seq_length)` combos for 5 iters each and pick the largest that fits
- [ ] Preference tuning (DPO/ORPO) as mlx-lm grows support
- [x] GGUF export for non-llama architectures via llama.cpp's converter (`mlxtune export`)

## Style

Python 3.10+, type hints, `ruff` clean. Errors users can hit should say what to change. Heavy imports (`mlx_lm`) stay inside functions so `mlxtune --help` is instant. Don't add a dependency for something 20 lines can do.
