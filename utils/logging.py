# -*- coding: utf-8 -*-
# --------------------------------------------
# 文件描述: 日志打印工具
# --------------------------------------------

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


_LOG_CONFIGURED = False
_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"
_ENV_LEVEL_KEY = "AGENTIC_RL_LOG_LEVEL"


class _MaxLevelFilter(logging.Filter):
    def __init__(self, level: int) -> None:
        super().__init__()
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.level


def _parse_level(level: str | int | None = None) -> int:
    env_level = level if level is not None else os.getenv(_ENV_LEVEL_KEY, "INFO")

    if isinstance(env_level, int):
        return env_level

    level_name = str(env_level).upper()
    return getattr(logging, level_name, logging.INFO)


def setup_logging(level: str | int | None = None) -> logging.Logger:
    global _LOG_CONFIGURED

    if _LOG_CONFIGURED:
        return logging.getLogger()

    root = logging.getLogger()
    root.setLevel(_parse_level(level))

    formatter = logging.Formatter(fmt=_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)

    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(_MaxLevelFilter(logging.INFO))
    stdout_handler.setFormatter(formatter)
    root.addHandler(stdout_handler)

    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    root.addHandler(stderr_handler)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _LOG_CONFIGURED = True
    return root


def get_logger(name: str | None = None) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)