"""Downloads Qwen2.5-0.5B-4bit (~300 MB) and trains 3 iters. Apple Silicon only.

Run with:  pytest -m integration
"""

import json
from pathlib import Path

import pytest

from mlxtuner.config import RunConfig
from mlxtuner.train import run

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.mark.integration
def test_train_chat_fuse(tmp_path):
    cfg = RunConfig.from_dict(
        {
            "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            "data": {"path": str(EXAMPLES / "pirate.jsonl"), "max_samples": 12, "eval_fraction": 0},
            "lora": {"rank": 4, "num_layers": 4},
            "train": {
                "output": str(tmp_path / "run"),
                "iters": 3,
                "batch_size": 1,
                "max_seq_length": 256,
                "steps_per_report": 1,
            },
        }
    )
    out = run(cfg)
    assert (out / "adapters.safetensors").exists()
    meta = json.loads((out / "mlxtuner.json").read_text())
    assert meta["peak_memory_gb"] and meta["final_train_loss"] is not None

    from mlxtuner.inference import fuse, load_for_inference, stream_reply

    model, tok = load_for_inference(str(out))
    reply = "".join(
        stream_reply(model, tok, [{"role": "user", "content": "hi"}], max_tokens=4, temperature=0)
    )
    assert isinstance(reply, str)

    from mlxtuner.eval import run_eval

    r = run_eval(
        str(out),
        cfg.data,
        max_seq_length=256,
        compare_base=True,
        n_samples=1,
        max_examples=4,
        max_tokens=4,
    )
    assert r.tuned_loss > 0 and r.base_loss is not None
    assert len(r.samples) == 1 and "base" in r.samples[0] and "tuned" in r.samples[0]

    fused = fuse(str(out), str(tmp_path / "fused"))
    assert (fused / "config.json").exists() and not (fused / "adapters.safetensors").exists()
