"""mlxtune command line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import EXAMPLE_CONFIG, RunConfig

app = typer.Typer(
    name="mlxtune",
    help="Fine-tune LLMs on Apple Silicon with one command.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)
console = Console()

SetOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--set", "-s", help="Override a config value, e.g. --set train.lr=1e-4", show_default=False
    ),
]


def _version(value: bool) -> None:
    if value:
        console.print(f"mlxtune {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[
        bool, typer.Option("--version", "-V", callback=_version, is_eager=True, help="Show version")
    ] = False,
) -> None:
    pass


def _load_config(
    config: Path | None,
    model: str | None,
    data: str | None,
    output: str | None,
    overrides: list[str] | None,
) -> RunConfig:
    overrides = list(overrides or [])
    if model:
        overrides.append(f"model={model}")
    if data:
        overrides.append(f"data.path={data}")
    if output:
        overrides.append(f"train.output={output}")
    if config:
        return RunConfig.from_yaml(config, overrides)
    if not (model and data):
        raise typer.BadParameter("Pass a config file, or both --model and --data.")
    return RunConfig.from_dict({"model": model, "data": {"path": data}}, overrides)


# ---------------------------------------------------------------------------


@app.command()
def check() -> None:
    """Show this Mac's chip and RAM, and the training defaults mlxtune will use for it."""
    from .hardware import TIER_DEFAULTS, detect

    m = detect()
    console.print(f"mlxtune {__version__}")
    console.print(f"chip: {m.chip}")
    console.print(f"ram:  {m.ram_gb:g} GB  -> tier {m.tier}")
    if not m.apple_silicon:
        console.print("[red]This is not an Apple Silicon Mac; MLX will not run here.[/]")
        raise typer.Exit(1)
    d = TIER_DEFAULTS[m.tier]
    console.print(
        f"defaults: batch_size={d.batch_size}  max_seq_length={d.max_seq_length}  "
        f"num_layers={d.num_layers}  grad_checkpoint={'on' if d.grad_checkpoint else 'off'}"
    )
    console.print("see what fits:  mlxtune models")


@app.command()
def models(
    all_: Annotated[
        bool, typer.Option("--all", help="Include models that don't fit this Mac")
    ] = False,
) -> None:
    """List recommended 4-bit models and whether each fits this Mac for LoRA training."""
    from .hardware import MODELS, TIER_DEFAULTS, detect, estimate_train_gb, fits

    m = detect()
    d = TIER_DEFAULTS[m.tier]
    table = Table(
        title=f"LoRA training on {m.chip}, {m.ram_gb:g} GB\n"
        f"(batch {d.batch_size}, seq {d.max_seq_length}, {d.num_layers} layers, grad checkpoint {'on' if d.grad_checkpoint else 'off'})",
        caption="all repos are under mlx-community/",
    )
    table.add_column("model", no_wrap=True)
    table.add_column("params", justify="right", no_wrap=True)
    table.add_column("weights", justify="right", no_wrap=True)
    table.add_column("est. peak", justify="right", no_wrap=True)
    table.add_column("fits", no_wrap=True)
    table.add_column("note", style="dim")
    colour = {"yes": "green", "tight": "yellow", "no": "red"}
    for rec in MODELS:
        est = estimate_train_gb(
            rec.weights_gb,
            d.batch_size,
            d.max_seq_length,
            d.num_layers,
            d.grad_checkpoint,
            rec.vocab_k,
        )
        verdict = fits(m, est)
        if verdict == "no" and not all_:
            continue
        table.add_row(
            rec.repo.removeprefix("mlx-community/"), f"{rec.params_b:g}B", f"{rec.weights_gb:.1f} GB",
            f"~{est:g} GB", f"[{colour[verdict]}]{verdict}[/]", rec.note,
        )  # fmt: skip
    console.print(table)
    if not all_:
        console.print(
            "[dim]--all shows models that don't fit. Estimates are ±30 %; peak memory is reported after each run.[/]"
        )


@app.command()
def init(
    path: Annotated[Path, typer.Argument(help="Where to write the starter config")] = Path(
        "config.yaml"
    ),
    force: Annotated[bool, typer.Option("--force", "-f")] = False,
) -> None:
    """Write a commented starter config."""
    if path.exists() and not force:
        console.print(f"[red]{path} exists[/] (use --force to overwrite)")
        raise typer.Exit(1)
    path.write_text(EXAMPLE_CONFIG)
    console.print(f"[green]wrote {path}[/]  next:  edit it, then  mlxtune train {path}")


@app.command()
def validate(
    config: Annotated[
        Path | None, typer.Argument(help="Run config (optional if --data is given)")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Model id (for tokenizer stats)")
    ] = None,
    data: Annotated[str | None, typer.Option("--data", "-d", help="Dataset path or Hub id")] = None,
    overrides: SetOpt = None,
    show: Annotated[int, typer.Option(help="Print this many rendered examples")] = 1,
) -> None:
    """Convert a dataset without training: format detection, drop reasons, token-length stats."""
    import tempfile

    from .data import DataError, prepare, render, token_stats
    from .hardware import detect

    model = model or "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    cfg = _load_config(config, model, data, None, overrides).resolved(detect())
    with tempfile.TemporaryDirectory() as tmp:
        try:
            p = prepare(cfg.data, Path(tmp) / "data")
        except DataError as e:
            console.print(f"[red]data error:[/] {e}")
            raise typer.Exit(1) from None
        console.print(f"format: [bold]{p.source_format}[/] -> {p.kind}")
        console.print(f"rows: {p.n_train:,} train / {p.n_valid:,} valid / {p.dropped} dropped")
        for reason, n in sorted(p.drop_reasons.items(), key=lambda kv: -kv[1])[:5]:
            console.print(f"  [yellow]{n:>6}[/]  {reason}")

        from .inference import load_tokenizer_only

        tok = load_tokenizer_only(cfg.model)
        if p.kind == "messages" and getattr(tok, "chat_template", None) is None:
            console.print(f"[red]{cfg.model} has no chat template; use an -Instruct model[/]")
            raise typer.Exit(1)
        with open(p.dir / "train.jsonl") as f:
            rows = [json.loads(line) for line in f]
        st = token_stats(tok, rows, int(cfg.train.max_seq_length))
        table = Table(title=f"token lengths (sampled {st['sampled']:,})")
        for k in ("min", "p50", "p90", "max", "mean"):
            table.add_column(k, justify="right")
        table.add_row(*(str(st[k]) for k in ("min", "p50", "p90", "max", "mean")))
        console.print(table)
        if st["over_max"]:
            console.print(
                f"[yellow]{st['over_max']} of {st['sampled']} rows exceed max_seq_length={cfg.train.max_seq_length} "
                "and will be truncated.[/]"
            )
        for i in range(min(show, len(rows))):
            console.rule(f"[dim]example {i}")
            console.print(render(tok, rows[i]), markup=False, highlight=False)
    console.print("[green]dataset OK[/]")


@app.command()
def train(
    config: Annotated[
        Path | None, typer.Argument(help="YAML run config (see `mlxtune init`)")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Model id or local path")
    ] = None,
    data: Annotated[str | None, typer.Option("--data", "-d", help="Dataset path or Hub id")] = None,
    output: Annotated[
        str | None, typer.Option("--output", "-o", help="Adapter output directory")
    ] = None,
    overrides: SetOpt = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Convert data and print the plan, train nothing")
    ] = False,
) -> None:
    """Fine-tune a model with LoRA. Use a config file, or --model + --data for auto defaults."""
    from .train import run

    run(_load_config(config, model, data, output, overrides), dry_run=dry_run)


@app.command()
def chat(
    path: Annotated[
        str, typer.Argument(help="Adapter dir from `mlxtune train`, fused model dir, or Hub id")
    ],
    system: Annotated[str | None, typer.Option("--system", help="System prompt")] = None,
    max_tokens: Annotated[int, typer.Option(help="Max tokens per reply")] = 512,
    temperature: Annotated[float, typer.Option(help="0 = greedy")] = 0.7,
) -> None:
    """Chat interactively with a fine-tuned model."""
    from .inference import chat_loop

    chat_loop(path, system=system, max_tokens=max_tokens, temperature=temperature)


@app.command()
def fuse(
    adapter: Annotated[str, typer.Argument(help="Adapter dir from `mlxtune train`")],
    output: Annotated[
        str, typer.Option("--output", "-o", help="Where to save the fused model")
    ] = "fused-model",
    dequantize: Annotated[
        bool, typer.Option("--dequantize", help="Save in full precision (needed for GGUF)")
    ] = False,
    gguf: Annotated[
        str | None,
        typer.Option(
            "--gguf", help="Also export a GGUF file to this path (llama/mistral/mixtral only)"
        ),
    ] = None,
    ollama: Annotated[
        bool, typer.Option("--ollama", help="Write an Ollama Modelfile next to the GGUF")
    ] = False,
    system: Annotated[str | None, typer.Option(help="System prompt for the Modelfile")] = None,
) -> None:
    """Merge the adapter into the base model -> standalone MLX model (and optionally GGUF / Ollama)."""
    from .inference import fuse as _fuse
    from .inference import write_ollama_modelfile

    if gguf and not dequantize:
        console.print("[dim]--gguf implies --dequantize[/]")
        dequantize = True
    _fuse(adapter, output, dequantize=dequantize, gguf=gguf)
    if ollama:
        if not gguf:
            console.print("[red]--ollama needs --gguf <path>[/]")
            raise typer.Exit(1)
        write_ollama_modelfile(gguf, system=system)


@app.command()
def info(path: Annotated[str, typer.Argument(help="A run output directory")]) -> None:
    """Summarise a finished run (mlxtune.json)."""
    meta_path = Path(path) / "mlxtune.json"
    if not meta_path.exists():
        console.print(f"[red]{meta_path} not found[/]")
        raise typer.Exit(1)
    meta = json.loads(meta_path.read_text())
    meta.pop("history", None)
    console.print_json(json.dumps(meta))


if __name__ == "__main__":
    app()
