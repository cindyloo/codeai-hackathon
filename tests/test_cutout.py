"""Upload a photo: the subject comes back on transparency, colours kept."""

from __future__ import annotations

import io
from urllib.error import URLError

import pytest

import app as app_module
import config
import cutout
import fal_images

PNG = b"\x89PNG\r\n\x1a\nfake"


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as test_client:
        yield test_client


def post_photo(client, data=b"jpeg bytes"):
    return client.post("/api/cutouts",
                       data={"image": (io.BytesIO(data), "toy.jpg")})


def test_returns_seg_servers_cutout(client, monkeypatch):
    monkeypatch.setattr(config, "RIGGING_SERVICE_URL", "http://seg.test")
    monkeypatch.setattr(cutout, "_seg_server", lambda data: PNG)

    response = post_photo(client)

    assert response.status_code == 200
    assert response.mimetype == "image/png"
    assert response.data == PNG


def test_falls_back_to_fal_when_seg_server_is_down(client, monkeypatch):
    def down(data):
        raise URLError("connection refused")

    monkeypatch.setattr(config, "RIGGING_SERVICE_URL", "http://seg.test")
    monkeypatch.setattr(config, "FAL_KEY", "test-key")
    monkeypatch.setattr(cutout, "_seg_server", down)
    monkeypatch.setattr(fal_images, "remove_background", lambda data: PNG)

    assert post_photo(client).data == PNG


def test_says_so_when_nothing_can_cut_it_out(client, monkeypatch):
    monkeypatch.setattr(config, "RIGGING_SERVICE_URL", "")
    monkeypatch.setattr(config, "FAL_KEY", "")

    response = post_photo(client)

    assert response.status_code == 502
    assert "background" in response.get_json()["error"]


def test_rejects_an_empty_upload(client):
    assert post_photo(client, b"").status_code == 400
