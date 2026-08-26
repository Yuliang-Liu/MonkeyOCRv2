"""Exercise parsing orchestration with deterministic model responses."""

from __future__ import annotations

from PIL import Image

from core_runner import (
    ALL_PROMPT,
    _build_page_tasks,
    _format_block_fields,
    get_layout,
)


class _FakeModel:
    def __init__(self):
        self.calls = []

    def batch_inference(self, images, questions, **kwargs):
        self.calls.append((len(images), list(questions), kwargs))
        outputs = []
        for question in questions:
            if question == ALL_PROMPT["LAYOUT"]:
                outputs.append('[{"bbox": [0, 0, 1000, 1000], "label": "Text"}]')
            else:
                outputs.append("fake content")
        return outputs


def test_fake_model_layout_to_formatted_blocks():
    model = _FakeModel()
    images = [Image.new("RGB", (32, 16), "white")]
    layouts = get_layout(model, images)
    tasks = _build_page_tasks(0, images[0], layouts[0])
    outputs = model.batch_inference(
        [task["image"] for task in tasks],
        [task["question"] for task in tasks],
        max_tokens=5000,
    )

    assert layouts == [[{"bbox": [0, 0, 32, 16], "label": "Text"}]]
    assert outputs == ["fake content"]
    assert _format_block_fields(tasks[0], outputs[0], "doc", [0], True, None) == {
        "content": "fake content",
    }
    assert model.calls[0][0] == 1
    assert model.calls[0][2]["max_tokens"] == 4096
    assert model.calls[1][0] == 1
