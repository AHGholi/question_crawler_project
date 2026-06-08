# tests/crawler/test_downloader.py
import io
from unittest.mock import MagicMock
import pytest

# tests/crawler/test_downloader.py
import crawler.downloader as downloader


def test_download_file_writes_html(monkeypatch, tmp_path):
    fake_resp = MagicMock()
    fake_resp.__enter__.return_value = fake_resp
    fake_resp.iter_content.return_value = [b"<html>ok</html>"]
    fake_resp.headers = {"Content-Type": "text/html"}
    fake_resp.raise_for_status.return_value = None

    monkeypatch.setattr(downloader.requests, "get", MagicMock(return_value=fake_resp))

    target = tmp_path / "page.html"
    result = downloader.download_file("https://example.com", str(target))

    assert target.read_text() == "<html>ok</html>"
    assert result == str(target)
