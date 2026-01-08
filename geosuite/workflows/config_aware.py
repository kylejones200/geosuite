"""
Config-aware function decorators and utilities.

Allows functions to automatically read parameters from configuration
when not explicitly provided.
"""

import functools
import logging
from typing import Any, Callable, Dict, Optional, Union
from ..config import get_config, ConfigManager

logger = logging.getLogger(__name__)


def get_config_value(
    key: str,
    default: Any = None,
    config: Optional[ConfigManager] = None
) -> Any:
    """
    Get a configuration value by key.
    
    Parameters
    ----------
    key : str
        Configuration key (supports dot notation, e.g., "petro.archie.m")
    default : any, optional
        Default value if key not found
    config : ConfigManager, optional
        Config manager instance. If None, uses global config.
        
    Returns
    -------
    any
        Configuration value or default
    """
    if config is not None:
        return config.get(key, default)
    return get_config(key, default)


def config_aware(
    config_key_prefix: str,
    parameter_map: Optional[Dict[str, str]] = None
):
    """
    Decorator to make a function config-aware.
    
    Automatically loads parameters from config when not provided.
    
    Parameters
    ----------
    config_key_prefix : str
        Prefix for config keys (e.g., "petro.archie" for archie functions)
    parameter_map : dict, optional
        Map of function parameter names to config keys.
        If None, assumes parameter name matches config key.
        Example: {"rw": "rw", "m": "m"} or {"rw": "water_resistivity"}
        
    Example
    -------
    >>> @config_aware("petro.archie")
    ... def calculate_water_saturation(phi, rt, rw=None, m=None, n=None):
    ...     rw = rw or get_config("petro.archie.rw", 0.05)
    ...     m = m or get_config("petro.archie.m", 2.0)
    ...     ...
    
    Or with explicit mapping:
    
    >>> @config_aware("petro.archie", {"rw": "rw", "cement_exp": "m"})
    ... def calculate_sw(phi, rt, rw=None, cement_exp=None):
    ...     ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # If config parameter is passed, use it; otherwise use global
            config = kwargs.pop('config', None)
            
            # Get function signature to find default parameters
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            
            # For each parameter that's None or has a default, try config
            for param_name, param in sig.parameters.items():
                # Skip if already provided
                if param_name in bound.arguments and bound.arguments[param_name] is not None:
                    continue
                    
                # Skip positional-only or special parameters
                if param_name in ('self', 'cls', 'config'):
                    continue
                
                # Determine config key
                if parameter_map and param_name in parameter_map:
                    config_key = f"{config_key_prefix}.{parameter_map[param_name]}"
                else:
                    config_key = f"{config_key_prefix}.{param_name}"
                
                # Try to get from config
                config_value = get_config_value(config_key, config=config)
                
                if config_value is not None:
                    kwargs[param_name] = config_value
                    logger.debug(
                        f"Loaded {param_name}={config_value} from config key {config_key}"
                    )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


class ConfigAwareFunction:
    """
    Wrapper class for making functions config-aware.
    
    Useful when you can't modify the function directly.
    """
    
    def __init__(
        self,
        func: Callable,
        config_key_prefix: str,
        parameter_map: Optional[Dict[str, str]] = None
    ):
        """
        Initialize config-aware function wrapper.
        
        Parameters
        ----------
        func : callable
            Function to wrap
        config_key_prefix : str
            Prefix for config keys
        parameter_map : dict, optional
            Map of function parameter names to config keys
        """
        self.func = func
        self.config_key_prefix = config_key_prefix
        self.parameter_map = parameter_map or {}
        
    def __call__(self, *args, **kwargs):
        """Call function with config-aware parameter loading."""
        config = kwargs.pop('config', None)
        
        import inspect
        sig = inspect.signature(self.func)
        
        # For missing parameters, try to load from config
        for param_name, param in sig.parameters.items():
            if param_name in kwargs and kwargs[param_name] is not None:
                continue
            if param_name in ('self', 'cls', 'config'):
                continue
            
            # Determine config key
            if param_name in self.parameter_map:
                config_key = f"{self.config_key_prefix}.{self.parameter_map[param_name]}"
            else:
                config_key = f"{self.config_key_prefix}.{param_name}"
            
            # Try config
            config_value = get_config_value(config_key, config=config)
            if config_value is not None:
                kwargs[param_name] = config_value
        
        return self.func(*args, **kwargs)
    
    def __getattr__(self, name):
        """Delegate other attributes to wrapped function."""
        return getattr(self.func, name)

