"""
Configuration management for GeoSuite.

Provides utilities for loading and managing configuration from YAML/JSON files.
"""

from .manager import ConfigManager, load_config, get_config, set_config

__all__ = [
    "ConfigManager",
    "load_config",
    "get_config",
    "set_config",
]

