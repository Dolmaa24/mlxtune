# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/); versions follow [SemVer](https://semver.org/).

## [0.1.0] - Unreleased

First release.

- `mlxtune check / models / init / validate / train / eval / chat / fuse / export / info`.
- RAM-tier defaults (`auto` batch size, sequence length, LoRA layers, gradient checkpointing) resolved at train time.
- Peak-memory estimator calibrated on measured runs (M2, 8 GB); `mlxtune models` shows which curated 4-bit models fit.
- Dataset auto-detection and conversion to mlx-lm's `train/valid.jsonl` layout.
- `eval`: mlx-lm's own validation loss (assistant tokens only with `mask_prompt`), sample generations, tuned vs base.
- `export`: GGUF for any llama.cpp-supported architecture via its converter (after `fuse --dequantize`), plus an Ollama Modelfile; verified end-to-end through `ollama run` with a Qwen2.5 adapter.
- Workaround for `mlx_lm fuse` failing with a Hub repo id on recent huggingface_hub (`IncompleteSnapshotError`).
- `mlx-lm` pinned `<1.0` (mlxtune uses `train_model`, `CONFIG_DEFAULTS`, `load_dataset` from its internals).
- Example configs, unit + integration tests, macOS CI (push, PR, weekly against latest deps), Dependabot, pre-commit.
