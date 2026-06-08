# utils/config.py
from __future__ import annotations
import os
from functools import lru_cache
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

DEFAULT_CONFIG_PATH = "config.yaml"


class ConfigError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    load_dotenv()  # ensures .env is read once
    cfg_path = path or os.getenv("CONFIG_FILE", DEFAULT_CONFIG_PATH)

    if not os.path.exists(cfg_path):
        raise ConfigError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    # Optional: perform validation / fill defaults
    data.setdefault("crawler", {})
    data.setdefault("google", {})
    return data


def get_setting(*keys: str, default: Any = None) -> Any:
    """
    Access nested config keys: get_setting("crawler", "crawl_delay_seconds").
    """
    data = load_config()
    cursor: Any = data
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor
