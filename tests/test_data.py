import json

import pytest

from mlxtune.config import DataConfig
from mlxtune.data import DataError, convert_row, detect_format, load_rows, prepare, to_messages


def _cfg(**kw):
    return DataConfig(path="unused", **kw)


@pytest.mark.parametrize(
    "row,expected",
    [
        ({"messages": [{"role": "user", "content": "hi"}]}, "messages"),
        ({"conversations": [{"from": "human", "value": "hi"}]}, "sharegpt"),
        ({"instruction": "a", "output": "b"}, "alpaca"),
        ({"question": "a", "answer": "b"}, "alpaca"),
        ({"prompt": "a", "completion": "b"}, "prompt_completion"),
        ({"text": "hello"}, "text"),
    ],
)
def test_detect_format(row, expected):
    assert detect_format(row, _cfg()) == expected


def test_detect_errors():
    with pytest.raises(DataError, match="preference"):
        detect_format({"prompt": "a", "chosen": "b", "rejected": "c"})
    with pytest.raises(DataError, match="Could not detect"):
        detect_format({"foo": 1})


def test_alpaca_to_messages_with_input_and_system():
    msgs = to_messages(
        {"system": "s", "instruction": "Sum", "input": "1+1", "output": "2"}, "alpaca", None
    )
    assert msgs == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "Sum\n\n1+1"},
        {"role": "assistant", "content": "2"},
    ]


def test_sharegpt_roles_and_system_prompt():
    row = {"conversations": [{"from": "human", "value": "h"}, {"from": "gpt", "value": "g"}]}
    msgs = to_messages(row, "sharegpt", "SYS")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[0]["content"] == "SYS"


def test_messages_flatten_parts():
    row = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": "yo"},
        ]
    }
    assert to_messages(row, "messages", None)[0]["content"] == "hi"


def test_validation_rules():
    with pytest.raises(DataError, match="no assistant"):
        to_messages({"messages": [{"role": "user", "content": "x"}]}, "messages", None)
    with pytest.raises(DataError, match="end with an assistant"):
        to_messages(
            {"messages": [{"role": "assistant", "content": "x"}, {"role": "user", "content": "y"}]},
            "messages",
            None,
        )
    with pytest.raises(DataError, match="Unknown sharegpt speaker"):
        to_messages({"conversations": [{"from": "alien", "value": "x"}]}, "sharegpt", None)


def test_convert_row_text_and_prompt_completion():
    assert convert_row({"body": "t"}, "text", _cfg(text_field="body")) == {"text": "t"}
    out = convert_row(
        {"q": "a", "a": "b"},
        "prompt_completion",
        _cfg(prompt_field="q", completion_field="a", system_prompt="S"),
    )
    assert out == {"prompt": "S\n\na", "completion": "b"}
    with pytest.raises(DataError, match="empty completion"):
        convert_row({"prompt": "a", "completion": " "}, "prompt_completion", _cfg())


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_load_rows_file_dir_and_mlx_layout(tmp_path):
    rows = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(6)]
    _write_jsonl(tmp_path / "a.jsonl", rows[:3])
    (tmp_path / "b.csv").write_text("instruction,output\nx,y\n")
    assert len(load_rows(str(tmp_path / "a.jsonl"))[0]) == 3
    assert len(load_rows(str(tmp_path))[0]) == 4  # jsonl + csv merged

    mlx = tmp_path / "mlx"
    mlx.mkdir()
    _write_jsonl(mlx / "train.jsonl", rows[:4])
    _write_jsonl(mlx / "valid.jsonl", rows[4:])
    train, valid = load_rows(str(mlx))
    assert len(train) == 4 and len(valid) == 2


def test_load_rows_json_wrapper(tmp_path):
    (tmp_path / "d.json").write_text(json.dumps({"data": [{"text": "a"}, {"text": "b"}]}))
    assert len(load_rows(str(tmp_path / "d.json"))[0]) == 2


def test_prepare_writes_mlx_layout_and_splits(tmp_path):
    src = tmp_path / "d.jsonl"
    _write_jsonl(src, [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(40)])
    p = prepare(DataConfig(path=str(src), eval_fraction=0.1), tmp_path / "work")
    assert p.kind == "messages" and p.source_format == "alpaca"
    assert p.n_train == 36 and p.n_valid == 4
    assert (p.dir / "train.jsonl").exists() and (p.dir / "valid.jsonl").exists()
    first = json.loads((p.dir / "train.jsonl").read_text().splitlines()[0])
    assert set(first) == {"messages"}


def test_prepare_no_valid_file_when_empty(tmp_path):
    src = tmp_path / "d.jsonl"
    _write_jsonl(src, [{"text": f"t{i}"} for i in range(5)])
    p = prepare(DataConfig(path=str(src)), tmp_path / "work")
    assert p.n_valid == 0
    assert not (p.dir / "valid.jsonl").exists()  # mlx-lm chokes on an empty valid.jsonl


def test_prepare_drops_bad_rows(tmp_path):
    src = tmp_path / "d.jsonl"
    _write_jsonl(src, [{"instruction": "a", "output": "b"}, {"instruction": "c", "output": ""}])
    p = prepare(DataConfig(path=str(src), eval_fraction=0), tmp_path / "work")
    assert p.n_train == 1 and p.dropped == 1


def test_missing_local_file_says_so_not_a_hub_error():
    with pytest.raises(DataError, match="No such file"):
        load_rows("data/train.jsonl")
