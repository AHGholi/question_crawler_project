# tests/crawler/test_search.py
import crawler.search as search

class DummyResponse:
    def __init__(self, link):
        self.link = link

def test_run_search_filters_and_downloads(monkeypatch, tmp_path):
    search.ALLOW_DOMAINS = ["example.com"]
    monkeypatch.setenv("DOWNLOAD_DIR", str(tmp_path))

    monkeypatch.setattr(
        "crawler.search.google_search",
        lambda query, num_results=10: [{"link": "https://example.com/page", "title": "OK", "snippet": "..."},]
    )
    monkeypatch.setattr(
        "crawler.search.download_file",
        lambda url, path: path
    )

    results = search.run_search("test", max_results=1)

    assert len(results) == 1
    assert results[0]["download"]["status"] == "ok"
