# tests/api_clients/test_google_search.py
import pytest
import crawler.api_clients.google_search as client

def test_google_search_raises_without_credentials(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CSE_ID", raising=False)

    with pytest.raises(ValueError):
        client.google_search("test")
