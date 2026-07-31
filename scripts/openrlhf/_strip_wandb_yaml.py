#!/usr/bin/env python3
"""Write a copy of an OpenRLHF YAML config with logger.wandb removed."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: _strip_wandb_yaml.py SRC.yaml DST.yaml")
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    cfg = yaml.safe_load(src.read_text()) or {}
    logger = cfg.get("logger") or {}
    logger.pop("wandb", None)
    cfg["logger"] = logger
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False))


if __name__ == "__main__":
    main()
