"""fal.ai image client: request shape, response handling, error translation.

No fal call is made — the client is replaced with a stub returning canned
results.
"""

from __future__ import annotations

import base64

import pytest

import config
import fal_images


def data_uri(data: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


class StubClient:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def subscribe(self, model_id, arguments, **kwargs):
        self.calls.append((model_id, arguments))
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
                                      negative_prompt="blurry")

    assert result == {"image_bytes": b"out", "output_format": "jpeg",
                      "seed": 42}
    model_id, arguments = client.calls[0]
    assert model_id == "render-model"
    assert arguments["image_url"] == data_uri(b"\x89PNG drawing")
    assert "a robot" in arguments["prompt"]
    assert "Avoid: blurry" in arguments["prompt"]


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
