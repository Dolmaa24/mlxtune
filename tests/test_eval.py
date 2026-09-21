from mlxtune.eval import _prompt_messages, _reference


def test_prompt_and_reference_messages():
    row = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
    assert _prompt_messages(row) == [{"role": "user", "content": "q"}]
    assert _reference(row) == "a"


def test_prompt_and_reference_completion_and_text():
    assert _prompt_messages({"prompt": "p", "completion": "c"}) == [
        {"role": "user", "content": "p"}
    ]
    assert _reference({"prompt": "p", "completion": "c"}) == "c"
    assert _prompt_messages({"text": "t"}) is None
