"""Shared configuration loading and resolution helpers.

All stage modules read the same ``config.toml`` contract. This module avoids
duplicated TOML parsing and path resolution logic.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from .paths import DEFAULT_RAW_DATA_DIR, PROJECT_ROOT


def load_config() -> dict[str, Any]:
    """Load and return the project configuration dictionary.

    Returns
    -------
    dict[str, Any]
        Parsed ``config.toml`` contents.
    """
    config_path = PROJECT_ROOT / "config.toml"
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def configured_symbols(config: dict[str, Any]) -> list[str]:
    """Return the full symbol list for pipeline stages.

    Parameters
    ----------
    config : dict[str, Any]
        Parsed project configuration.

    Returns
    -------
    list[str]
        Concatenation of primary symbols and optional robustness symbols.
    """
    data_section = config["data"]
    primary_symbols = list(data_section.get("symbols", []))
    robustness_symbols = list(data_section.get("robustness_symbols", []))
    return primary_symbols + robustness_symbols


def resolve_raw_cache_dir(config: dict[str, Any]) -> Path:
    """Resolve raw cache directory from config with project-local default.

    Parameters
    ----------
    config : dict[str, Any]
        Parsed project configuration.

    Returns
    -------
    Path
        Absolute path to stage-1 raw parquet cache.
    """
    cache_section = config.get("cache", {})
    raw_cache_dir = cache_section.get("raw_cache_dir", str(DEFAULT_RAW_DATA_DIR))

    path = Path(raw_cache_dir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def resolve_metadata_suffix(config: dict[str, Any]) -> str:
    """Read metadata sidecar suffix for raw cache files."""
    cache_section = config.get("cache", {})
    return str(cache_section.get("metadata_suffix", ".metadata.json"))


def resolve_cache_version(config: dict[str, Any]) -> int:
    """Read expected raw-cache contract version."""
    cache_section = config.get("cache", {})
    return int(cache_section.get("version", 1))


def resolve_options_shard_rows(config: dict[str, Any]) -> int:
    """Read maximum rows per options parquet shard."""
    cache_section = config.get("cache", {})
    return int(cache_section.get("options_shard_rows", 2_500_000))
