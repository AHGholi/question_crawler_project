# tests/conftest.py
import pytest
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

@pytest.fixture
def temp_output_dir(tmp_path: Path):
    return tmp_path / "downloads"

@pytest.fixture
def sample_google_item():
    return {
        "title": "Sample Title",
        "link": "https://example.com/page",
        "snippet": "Example snippet"
    }

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
