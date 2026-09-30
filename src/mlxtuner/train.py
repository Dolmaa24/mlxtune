"""Drive mlx-lm's LoRA trainer from a resolved RunConfig, capturing metrics as we go."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rich.console import Console

from . import __version__
from .config import RunConfig
from .data import DataError, Prepared, prepare
from .hardware import Machine, detect, estimate_train_gb, fits

console = Console()


class MetricsCallback:
    """Collects what mlx-lm reports so we can write it to mlxtuner.json."""

    def __init__(self) -> None:
        self.train: list[dict[str, Any]] = []
        self.val: list[dict[str, Any]] = []

    def on_train_loss_report(self, info: dict) -> None:
        self.train.append(dict(info))

    def on_val_loss_report(self, info: dict) -> None:
        self.val.append(dict(info))

    @property
    def peak_gb(self) -> float | None:
        return max((r.get("peak_memory", 0) for r in self.train), default=None)


def build_args(cfg: RunConfig, data: Prepared, n_train: int) -> SimpleNamespace:
    """Map our config onto the argparse namespace mlx_lm.lora expects."""
    from mlx_lm.lora import CONFIG_DEFAULTS

    t, lo = cfg.train, cfg.lora
    batch_size = int(t.batch_size)
    steps_per_epoch = max(1, math.ceil(n_train / batch_size))
    iters = t.iters or max(1, int(steps_per_epoch * t.epochs))
    steps_per_eval = t.steps_per_eval or max(1, iters // 4)
    save_every = t.save_every or steps_per_eval

    args = SimpleNamespace(**CONFIG_DEFAULTS)
    args.model = cfg.model
    args.train = True
    args.test = False
    args.data = str(data.dir)
    args.adapter_path = t.output
    args.fine_tune_type = lo.type
    args.num_layers = int(lo.num_layers)
    args.lora_parameters = {"rank": lo.rank, "dropout": lo.dropout, "scale": lo.scale}
    args.batch_size = batch_size
    args.iters = iters
    args.learning_rate = t.lr
    args.optimizer = t.optimizer
    args.max_seq_length = int(t.max_seq_length)
    args.grad_checkpoint = bool(t.grad_checkpoint)
    args.grad_accumulation_steps = t.grad_accumulation_steps
    args.steps_per_report = t.steps_per_report
    args.steps_per_eval = steps_per_eval
    args.save_every = save_every
    args.val_batches = t.val_batches
    args.seed = t.seed
    args.mask_prompt = t.mask_prompt and data.kind != "text"
    args.resume_adapter_file = t.resume
    args.report_to = None
    if t.warmup_steps > 0:
        args.lr_schedule = {
            "name": "cosine_decay",
            "warmup": t.warmup_steps,
            "warmup_init": 0.0,
            "arguments": [t.lr, iters, t.lr * 0.1],
        }
    return args


def run(cfg: RunConfig, dry_run: bool = False, machine: Machine | None = None) -> Path:
    t0 = time.time()
    machine = machine or detect()
    cfg = cfg.resolved(machine)
    out = Path(cfg.train.output)
    out.mkdir(parents=True, exist_ok=True)

    console.rule("[bold]mlxtuner train")
    console.print(
        f"[dim]mlxtuner {__version__}[/]  {machine.chip}, {machine.ram_gb:g} GB (tier {machine.tier})"
    )

    # -- data ----------------------------------------------------------------
    try:
        data = prepare(cfg.data, out / "data")
    except DataError as e:
        raise SystemExit(f"data error: {e}") from None
    console.print(
        f"data: {data.n_train:,} train / {data.n_valid:,} valid  "
        f"[dim]({data.source_format} -> {data.kind}, dropped {data.dropped})[/]"
    )
    for reason, n in sorted(data.drop_reasons.items(), key=lambda kv: -kv[1])[:3]:
        console.print(f"  [yellow]dropped {n}[/]: {reason}")

    # -- plan ----------------------------------------------------------------
    args = build_args(cfg, data, data.n_train)
    console.print(
        f"model: {cfg.model}   {cfg.lora.type} rank={cfg.lora.rank} scale={cfg.lora.scale:g} "
        f"layers={args.num_layers}"
    )
    console.print(
        f"schedule: {args.iters} iters  batch={args.batch_size}  seq={args.max_seq_length}  "
        f"lr={args.learning_rate:g}  grad_checkpoint={'on' if args.grad_checkpoint else 'off'}  "
        f"mask_prompt={'on' if args.mask_prompt else 'off'}"
    )
    est = _estimate(cfg, args)
    if est is not None:
        verdict = fits(machine, est)
        colour = {"yes": "green", "tight": "yellow", "no": "red"}[verdict]
        console.print(
            f"memory: ~{est} GB estimated peak  [{colour}]{verdict}[/] for {machine.ram_gb:g} GB"
        )
        if verdict == "no":
            console.print(
                "[red]This will probably swap or crash.[/] Lower train.max_seq_length, train.batch_size or "
                "lora.num_layers, or pick a smaller model (`mlxtuner models`)."
            )
    cfg.to_yaml(out / "mlxtuner.yaml")
    if dry_run:
        console.print("[green]dry run OK[/]  (data converted, nothing trained)")
        return out

    # -- train ---------------------------------------------------------------
    import mlx.core as mx
    from mlx_lm.lora import train_model
    from mlx_lm.tuner.datasets import load_dataset
    from mlx_lm.utils import load

    console.print(f"[dim]loading {cfg.model} ...[/]")
    model, tokenizer = load(cfg.model, tokenizer_config={"trust_remote_code": True})
    if data.kind == "messages" and getattr(tokenizer, "chat_template", None) is None:
        raise SystemExit(
            f"{cfg.model} has no chat template; use an -Instruct model for conversational data"
        )
    train_set, valid_set, _ = load_dataset(args, tokenizer)
    if args.num_layers == -1 or args.num_layers > len(model.layers):
        args.num_layers = len(model.layers)

    mx.reset_peak_memory()
    metrics = MetricsCallback()
    train_model(args, model, train_set, valid_set, metrics)

    # -- save metadata -------------------------------------------------------
    last_train = metrics.train[-1] if metrics.train else {}
    last_val = metrics.val[-1] if metrics.val else {}
    meta = {
        "mlxtuner_version": __version__,
        "model": cfg.model,
        "machine": {"chip": machine.chip, "ram_gb": machine.ram_gb},
        "kind": data.kind,
        "train_examples": data.n_train,
        "valid_examples": data.n_valid,
        "iters": args.iters,
        "final_train_loss": last_train.get("train_loss"),
        "final_val_loss": last_val.get("val_loss"),
        "peak_memory_gb": round(metrics.peak_gb, 2) if metrics.peak_gb else None,
        "tokens_per_second": round(last_train.get("tokens_per_second", 0), 1),
        "wall_time_s": round(time.time() - t0, 1),
        "history": {"train": metrics.train, "val": metrics.val},
    }
    (out / "mlxtuner.json").write_text(json.dumps(meta, indent=2, default=str))
    _write_readme(out, cfg, meta)

    tl, vl = meta["final_train_loss"], meta["final_val_loss"]
    console.print(
        f"[green]done[/] in {meta['wall_time_s']}s   train_loss={tl:.3f}"
        + (f"  val_loss={vl:.3f}" if vl else "")
        + (f"  peak_mem={meta['peak_memory_gb']} GB" if meta["peak_memory_gb"] else "")
    )
    console.print(f"adapter saved to [bold]{out}[/]   try it:  mlxtuner chat {out}")
    return out


def _estimate(cfg: RunConfig, args: SimpleNamespace) -> float | None:
    from .hardware import MODELS

    rec = next((m for m in MODELS if m.repo == cfg.model), None)
    if rec is None:
        return None
    return estimate_train_gb(
        rec.weights_gb,
        args.batch_size,
        args.max_seq_length,
        args.num_layers,
        args.grad_checkpoint,
        rec.vocab_k,
    )


def _write_readme(out: Path, cfg: RunConfig, meta: dict[str, Any]) -> None:
    tl, vl = meta["final_train_loss"], meta["final_val_loss"]
    lines = [
        f"# LoRA adapter for `{cfg.model}`",
        "",
        f"Trained with [mlxtuner](https://github.com/Dolmaa24/mlxtuner) {meta['mlxtuner_version']} "
        f"on {meta['machine']['chip']} ({meta['machine']['ram_gb']:g} GB).",
        "",
        "| | |",
        "|---|---|",
        f"| Base model | `{cfg.model}` |",
        f"| Method | {cfg.lora.type} rank={cfg.lora.rank} scale={cfg.lora.scale:g} layers={cfg.lora.num_layers} |",
        f"| Train / valid examples | {meta['train_examples']:,} / {meta['valid_examples']:,} |",
        f"| Iters | {meta['iters']} (batch {cfg.train.batch_size}, seq {cfg.train.max_seq_length}) |",
        f"| Learning rate | {cfg.train.lr:g} |",
        f"| Final train loss | {tl:.3f} |" if tl is not None else "",
        f"| Final val loss | {vl:.3f} |" if vl is not None else "",
        f"| Peak memory | {meta['peak_memory_gb']} GB |" if meta["peak_memory_gb"] else "",
        "",
        "## Use",
        "",
        "```bash",
        f"mlxtuner chat {cfg.train.output}",
        f"mlxtuner fuse {cfg.train.output} --output fused-model   # standalone model",
        "```",
        "",
        "Re-run with `mlxtuner train mlxtuner.yaml`.",
    ]
    (out / "README.md").write_text("\n".join(x for x in lines if x is not None) + "\n")
