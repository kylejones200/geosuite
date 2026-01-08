"""
Configuration manager for GeoSuite.

Handles loading and managing configuration from YAML/JSON files.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages configuration for GeoSuite.
    
    Supports loading from YAML/JSON files and programmatic configuration.
    
    Example:
        >>> from geosuite.config import ConfigManager
        >>> config = ConfigManager()
        >>> config.load_from_file("config.yaml")
        >>> value = config.get("petro.archie.a", default=1.0)
        >>> config.set("petro.archie.m", 2.0)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration manager.
        
        Parameters
        ----------
        config : dict, optional
            Initial configuration dictionary
        """
        self._config: Dict[str, Any] = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_from_file(self, file_path: Union[str, Path]) -> 'ConfigManager':
        """
        Load configuration from a YAML or JSON file.
        
        Parameters
        ----------
        file_path : str or Path
            Path to configuration file
            
        Returns
        -------
        ConfigManager
            Self for method chaining
            
        Raises
        ------
        FileNotFoundError
            If file doesn't exist
        ValueError
            If file format is not supported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        loaders = {
            '.yaml': self._load_yaml,
            '.yml': self._load_yaml,
            '.json': self._load_json,
        }
        
        loader = loaders.get(suffix)
        if loader is None:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported formats: {list(loaders.keys())}"
            )
        
        with open(file_path, 'r') as f:
            loaded_config = loader(f)
        
        self._config = self._merge_config(self._config, loaded_config)
        self.logger.info(f"Loaded configuration from {file_path}")
        
        return self
    
    def _load_yaml(self, file_handle) -> Dict[str, Any]:
        """Load YAML configuration."""
        try:
            return yaml.safe_load(file_handle) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")
    
    def _load_json(self, file_handle) -> Dict[str, Any]:
        """Load JSON configuration."""
        try:
            return json.load(file_handle) or {}
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
    
    def _merge_config(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two configuration dictionaries.
        
        Parameters
        ----------
        base : dict
            Base configuration
        update : dict
            Configuration to merge in
            
        Returns
        -------
        dict
            Merged configuration
        """
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Parameters
        ----------
        key : str
            Configuration key (supports dot notation, e.g., "petro.archie.a")
        default : any, optional
            Default value if key not found
            
        Returns
        -------
        any
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> 'ConfigManager':
        """
        Set configuration value using dot notation.
        
        Parameters
        ----------
        key : str
            Configuration key (supports dot notation)
        value : any
            Value to set
            
        Returns
        -------
        ConfigManager
            Self for method chaining
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get full configuration as dictionary.
        
        Returns
        -------
        dict
            Full configuration dictionary
        """
        return self._config.copy()
    
    def save_to_file(self, file_path: Union[str, Path], format: str = 'yaml') -> 'ConfigManager':
        """
        Save configuration to a file.
        
        Parameters
        ----------
        file_path : str or Path
            Path to save configuration
        format : str, default 'yaml'
            File format ('yaml' or 'json')
            
        Returns
        -------
        ConfigManager
            Self for method chaining
        """
        file_path = Path(file_path)
        
        savers = {
            'yaml': self._save_yaml,
            'json': self._save_json,
        }
        
        saver = savers.get(format.lower())
        if saver is None:
            raise ValueError(f"Unsupported format: {format}. Choose: {list(savers.keys())}")
        
        with open(file_path, 'w') as f:
            saver(f, self._config)
        
        self.logger.info(f"Saved configuration to {file_path}")
        return self
    
    def _save_yaml(self, file_handle, config: Dict[str, Any]):
        """Save configuration as YAML."""
        yaml.dump(config, file_handle, default_flow_style=False, sort_keys=False)
    
    def _save_json(self, file_handle, config: Dict[str, Any]):
        """Save configuration as JSON."""
        json.dump(config, file_handle, indent=2, sort_keys=False)


_global_config: Optional[ConfigManager] = None


def load_config(file_path: Union[str, Path]) -> ConfigManager:
    """
    Load configuration from file and set as global config.
    
    Parameters
    ----------
    file_path : str or Path
        Path to configuration file
        
    Returns
    -------
    ConfigManager
        Configuration manager instance
    """
    global _global_config
    _global_config = ConfigManager().load_from_file(file_path)
    return _global_config


def get_config(key: str, default: Any = None, config: Optional[ConfigManager] = None) -> Any:
    """
    Get value from global configuration or provided config manager.
    
    Parameters
    ----------
    key : str
        Configuration key (supports dot notation)
    default : any, optional
        Default value if key not found
    config : ConfigManager, optional
        Config manager to use. If None, uses global config.
        
    Returns
    -------
    any
        Configuration value or default
    """
    if config is not None:
        return config.get(key, default)
    
    global _global_config
    if _global_config is None:
        _global_config = ConfigManager()
    return _global_config.get(key, default)


def set_config(key: str, value: Any) -> None:
    """
    Set value in global configuration.
    
    Parameters
    ----------
    key : str
        Configuration key (supports dot notation)
    value : any
        Value to set
    """
    global _global_config
    if _global_config is None:
        _global_config = ConfigManager()
    _global_config.set(key, value)

