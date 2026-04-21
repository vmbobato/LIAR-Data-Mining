from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    for _, rel_path in cfg["paths"].items():
        Path(rel_path).mkdir(parents=True, exist_ok=True)
