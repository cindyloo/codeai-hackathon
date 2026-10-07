"""fal.ai image client: request shape, response handling, error translation.

No fal call is made — the client is replaced with a stub returning canned
results.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

import config
import fal_images


def data_uri(data: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


class StubClient:
    def __init__(self, *results, queue_updates=()):
        self.results = list(results)
        self.calls = []
        self.queue_updates = list(queue_updates)

    def subscribe(self, model_id, arguments, **kwargs):
        self.calls.append((model_id, arguments))
        on_queue_update = kwargs.get("on_queue_update")
        for status in self.queue_updates:
            if on_queue_update:
                on_queue_update(status)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def stub(monkeypatch):
    def install(*results):
        client = StubClient(*results)
        monkeypatch.setattr(fal_images, "_get_client", lambda: client)
        return client
    return install


def edited(data: bytes, mime: str = "image/png", **extra):
    return {"images": [{"url": data_uri(data, mime), "content_type": mime}],
            "seed": 42, "has_nsfw_concepts": [False], **extra}


def test_render_sends_the_drawing_and_returns_the_image(stub, monkeypatch):
    monkeypatch.setattr(config, "FAL_RENDER_MODEL", "render-model")
    client = stub(edited(b"out", "image/jpeg"))

    result = fal_images.render_sketch(b"\x89PNG drawing", "a robot",
                                      style="animated",
                                      negative_prompt="blurry")

    assert result == {"image_bytes": b"out", "output_format": "jpeg",
                      "seed": 42}
    model_id, arguments = client.calls[0]
    assert model_id == "render-model"
    assert arguments["image_url"] == data_uri(b"\x89PNG drawing")
    assert "a robot" in arguments["prompt"]
    assert "Avoid: blurry" in arguments["prompt"]


def test_lines_style_traces_the_drawing_with_the_controlnet_model(
        stub, monkeypatch):
    monkeypatch.setattr(config, "FAL_LINES_MODEL", "lines-model")
    client = stub(edited(b"out"))

    result = fal_images.render_sketch(b"\x89PNG drawing", "a dog",
                                      style="lines", negative_prompt="blurry")

    assert result["image_bytes"] == b"out"
    model_id, arguments = client.calls[0]
    assert model_id == "lines-model"
    assert arguments["control_lora_image_url"] == data_uri(b"\x89PNG drawing")
    assert "image_url" not in arguments
    assert arguments["prompt"].startswith("a dog.")
    assert "Avoid" not in arguments["prompt"]


def test_animated_style_edits_with_kontext_leading_with_the_subject(
        stub, monkeypatch):
    monkeypatch.setattr(config, "FAL_RENDER_MODEL", "render-model")
    client = stub(edited(b"out"))

    fal_images.render_sketch(b"x", "a dog", style="animated")

    model_id, arguments = client.calls[0]
    instruction = arguments["prompt"]
    assert model_id == "render-model"
    assert arguments["image_url"] == data_uri(b"x")
    assert instruction.index("a dog") < 40
    assert "Front view" in instruction
    assert fal_images.FULL_BODY_HINT in instruction
    assert "three-quarter view" in instruction.split("Avoid:")[1]


def test_slow_start_is_reported_once_while_still_queued(monkeypatch):
    import fal_client

    clock = iter([0.0, 1.0, 6.0, 9.0])
    monkeypatch.setattr(fal_images, "time",
                        SimpleNamespace(monotonic=lambda: next(clock)))
    queued = fal_client.Queued(position=0)
    client = StubClient(edited(b"out"), queue_updates=[queued, queued, queued])
    monkeypatch.setattr(fal_images, "_get_client", lambda: client)
    calls = []

    fal_images.render_sketch(b"x", "a dog",
                             on_slow_start=lambda: calls.append(1))

    assert calls == [1]


def test_a_quick_start_is_not_reported(monkeypatch):
    import fal_client

    clock = iter([0.0, 1.0])
    monkeypatch.setattr(fal_images, "time",
                        SimpleNamespace(monotonic=lambda: next(clock)))
    client = StubClient(edited(b"out"),
                        queue_updates=[fal_client.Queued(position=0),
                                       fal_client.InProgress(logs=None)])
    monkeypatch.setattr(fal_images, "_get_client", lambda: client)
    calls = []

    fal_images.render_sketch(b"x", "a dog",
                             on_slow_start=lambda: calls.append(1))

    assert calls == []


def test_unknown_style_is_refused(stub):
    stub(edited(b"out"))
    with pytest.raises(ValueError):
        fal_images.render_sketch(b"x", "a dog", style="photoreal")


def test_flagged_output_is_a_422(stub):
    stub(edited(b"black", has_nsfw_concepts=[True]))
    with pytest.raises(fal_images.FalError) as excinfo:
        fal_images.render_sketch(b"x", "robot")
    assert excinfo.value.status == 422


def test_no_images_is_a_502(stub):
    stub({"images": [], "seed": 1, "has_nsfw_concepts": []})
    with pytest.raises(fal_images.FalError) as excinfo:
        fal_images.render_sketch(b"x", "robot")
    assert excinfo.value.status == 502


def test_tpose_poses_then_removes_the_background(stub, monkeypatch):
    monkeypatch.setattr(config, "FAL_TPOSE_MODEL", "pose-model")
    monkeypatch.setattr(config, "FAL_BG_REMOVAL_MODEL", "bg-model")
    client = stub(edited(b"posed"), {"image": {"url": data_uri(b"cutout")}})

    result = fal_images.tpose_transform(b"drawing")

    assert result == {"image_bytes": b"cutout", "output_format": "png"}
    assert [model for model, _ in client.calls] == ["pose-model", "bg-model"]
    assert "T-pose" in client.calls[0][1]["prompt"]
    assert client.calls[1][1]["image_url"] == data_uri(b"posed")


def test_api_errors_keep_their_detail_off_the_message(stub):
    fal_client = pytest.importorskip("fal_client")
    httpx = pytest.importorskip("httpx")
    secret = "key 1234abcd:secret is not valid"
    request = httpx.Request("POST", "https://queue.fal.run/x")
    stub(fal_client.FalClientHTTPError(
        message=secret, status_code=401, response_headers={},
        response=httpx.Response(401, request=request)))

    with pytest.raises(fal_images.FalError) as excinfo:
        fal_images.render_sketch(b"x", "robot")

    assert excinfo.value.status == 503
    assert secret not in excinfo.value.message


def test_missing_key_fails_closed(monkeypatch):
    monkeypatch.setattr(fal_images, "_client", None)
    monkeypatch.setattr(config, "FAL_KEY", "")
    with pytest.raises(fal_images.FalError) as excinfo:
        fal_images._get_client()
    assert excinfo.value.status == 503
