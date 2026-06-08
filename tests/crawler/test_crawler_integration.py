import os
from pathlib import Path

import pytest
import yaml

from crawler.search import run_search


@pytest.mark.integration
@pytest.mark.network
def test_crawler_can_download_real_files(tmp_path: Path):
    """
    Integration test:
    - Runs the real crawler
    - Downloads real pages
    - Verifies files exist and are non-empty
    """

    # --- Arrange ---
    query = query = "machine learning"
    max_results = 5

    # Load credentials from config.yaml
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "config.yaml"

    assert config_path.exists(), "config.yaml not found at project root"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    api_key = config["google"]["api_key"]
    cse_id = config["google"].get("cse_id") or config["google"].get("cx")

    assert api_key, "Missing google.api_key in config.yaml"
    assert cse_id, "Missing google.cse_id (or cx) in config.yaml"

    # Inject credentials into environment (what crawler actually uses)
    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ["GOOGLE_CSE_ID"] = cse_id

    # Override download directory
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()

    os.environ["CRAWLER__DOWNLOAD_DIR"] = str(download_dir)

    # --- Act ---
    results = run_search(query=query, max_results=max_results)

    # --- Assert: search results ---
    assert isinstance(results, list)
    assert len(results) > 0, "Crawler returned no search results"

    # --- Assert: at least one successful download ---
    successful = [
        r for r in results
        if r.get("download", {}).get("status") == "ok"
    ]

    assert successful, "No files were successfully downloaded"

    # --- Assert: downloaded files exist and are non-empty ---
    for item in successful:
        path = item["download"]["path"]
        assert path is not None

        file_path = Path(path)
        assert file_path.exists(), f"File does not exist: {file_path}"
        assert file_path.stat().st_size > 0, f"File is empty: {file_path}"

        assert file_path.suffix == ".html"
