from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_credentials(path: str | Path | None) -> dict[str, Any]:
    """
    Load credentials from a YAML file.

    Example YAML:
      zerodha:
        api_key: "..."
        access_token: "..."
      upstox:
        access_token: "..."
      angelone:
        api_key: "..."
        client_code: "..."
        password: "..."
        totp: "..."
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    data = yaml.safe_load(p.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("credentials file must be a YAML mapping")
    return data


def save_credentials(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(data, sort_keys=False))

