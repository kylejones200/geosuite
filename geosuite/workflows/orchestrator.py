"""
Workflow orchestrator for executing config-driven workflows.

Supports YAML/JSON workflow definitions with steps, dependencies, and parameters.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import yaml
import json
import pandas as pd
import numpy as np

from ..config import ConfigManager, load_config, get_config
from ..data import load_demo_well_logs

logger = logging.getLogger(__name__)


# Registry of available workflow steps
STEP_REGISTRY: Dict[str, Callable] = {}


def register_step(name: str, func: Callable):
    """Register a function as a workflow step."""
    STEP_REGISTRY[name] = func
    logger.debug(f"Registered workflow step: {name}")


def _add_to_dataframe(df: pd.DataFrame, columns: dict) -> pd.DataFrame:
    """Helper function to add calculated columns to DataFrame."""
    df_copy = df.copy()
    for col_name, values in columns.items():
        if isinstance(values, np.ndarray):
            df_copy[col_name] = values
        else:
            df_copy[col_name] = values
    return df_copy


def _register_default_steps():
    """Register default workflow steps."""
    # Data loading
    try:
        from ..io.las_loader import load_las_file
        register_step("load_las", load_las_file)
    except ImportError:
        pass  # lasio not available
    register_step("load_demo_data", load_demo_well_logs)
    register_step("add_to_dataframe", _add_to_dataframe)
    
    # Petrophysics
    from ..petro import (
        calculate_water_saturation,
        calculate_porosity_from_density,
        calculate_formation_factor,
        pickett_plot,
        buckles_plot
    )
    register_step("calculate_water_saturation", calculate_water_saturation)
    register_step("calculate_porosity", calculate_porosity_from_density)
    register_step("calculate_formation_factor", calculate_formation_factor)
    register_step("pickett_plot", pickett_plot)
    register_step("buckles_plot", buckles_plot)
    
    # Geomechanics
    from ..geomech import (
        calculate_overburden_stress,
        calculate_hydrostatic_pressure,
        calculate_pore_pressure_eaton,
        stress_polygon_limits,
        plot_stress_polygon
    )
    register_step("calculate_overburden_stress", calculate_overburden_stress)
    register_step("calculate_hydrostatic_pressure", calculate_hydrostatic_pressure)
    register_step("calculate_pore_pressure", calculate_pore_pressure_eaton)
    register_step("stress_polygon_limits", stress_polygon_limits)
    register_step("plot_stress_polygon", plot_stress_polygon)
    
    # Stratigraphy
    from ..stratigraphy import (
        preprocess_log,
        detect_pelt,
        detect_bayesian_online
    )
    register_step("preprocess_log", preprocess_log)
    register_step("detect_pelt", detect_pelt)
    register_step("detect_bayesian_online", detect_bayesian_online)
    
    # Machine Learning
    from ..ml import train_facies_classifier
    register_step("train_facies_classifier", train_facies_classifier)
    
    # Plotting
    from ..plotting import create_strip_chart
    register_step("create_strip_chart", create_strip_chart)


# Initialize default steps
_register_default_steps()


class WorkflowOrchestrator:
    """
    Orchestrator for executing config-driven workflows.
    
    Loads workflow definitions from YAML/JSON and executes steps
    in order, with dependency resolution and config-aware parameters.
    """
    
    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        working_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize workflow orchestrator.
        
        Parameters
        ----------
        config : ConfigManager, optional
            Configuration manager. If None, uses global config.
        working_dir : str or Path, optional
            Working directory for relative paths in workflow
        """
        self.config = config
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.results: Dict[str, Any] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_workflow_file(
        self,
        file_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Load workflow definition from file.
        
        Parameters
        ----------
        file_path : str or Path
            Path to YAML or JSON workflow file
            
        Returns
        -------
        dict
            Workflow definition
            
        Raises
        ------
        FileNotFoundError
            If file doesn't exist
        ValueError
            If file format is unsupported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Workflow file not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        with open(file_path, 'r') as f:
            if suffix in ('.yaml', '.yml'):
                workflow = yaml.safe_load(f)
            elif suffix == '.json':
                workflow = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported workflow file format: {suffix}. "
                    "Use .yaml, .yml, or .json"
                )
        
        self.logger.info(f"Loaded workflow from {file_path}")
        return workflow
    
    def _resolve_dependencies(
        self,
        steps: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Resolve step dependencies and return ordered execution list.
        
        Parameters
        ----------
        steps : list
            List of step definitions
            
        Returns
        -------
        list
            Steps in execution order
        """
        # Simple dependency resolution (can be enhanced)
        # For now, just execute in order
        return steps
    
    def _resolve_parameter(
        self,
        value: Any,
        step_name: str
    ) -> Any:
        """
        Resolve parameter value, supporting references to previous steps.
        
        Parameters
        ----------
        value : any
            Parameter value (may be string reference like "${step_name.output}")
        step_name : str
            Current step name
            
        Returns
        -------
        any
            Resolved value
        """
        if isinstance(value, str):
            # Check for step reference: ${step_name.output} or ${step_name}
            if value.startswith("${") and value.endswith("}"):
                ref = value[2:-1]
                
                # Check for config reference first
                if ref.startswith("config."):
                    config_key = ref[7:]  # Remove "config."
                    from .config_aware import get_config_value
                    return get_config_value(config_key, config=self.config)
                
                # Otherwise it's a step reference
                if '.' in ref:
                    step_ref, attr = ref.split('.', 1)
                else:
                    step_ref = ref
                    attr = 'output'
                
                if step_ref in self.results:
                    result = self.results[step_ref]
                    if attr == 'output':
                        return result
                    elif isinstance(result, pd.DataFrame):
                        # DataFrame column access
                        if attr in result.columns:
                            return result[attr].values if hasattr(result[attr], 'values') else result[attr]
                        else:
                            raise ValueError(
                                f"Column '{attr}' not found in step '{step_ref}' output. "
                                f"Available columns: {list(result.columns)}"
                            )
                    elif isinstance(result, dict) and attr in result:
                        return result[attr]  # Dictionary key
                    elif isinstance(result, np.ndarray):
                        # For numpy arrays, try to return as-is
                        return result
                    elif hasattr(result, attr):
                        return getattr(result, attr)
                    else:
                        raise ValueError(
                            f"Reference ${value} not found in step '{step_ref}'. "
                            f"Result type: {type(result)}"
                        )
                else:
                    raise ValueError(
                        f"Step '{step_ref}' not found in results (referenced by ${value})"
                    )
        
        return value
    
    def _resolve_parameters(
        self,
        params: Dict[str, Any],
        step_name: str
    ) -> Dict[str, Any]:
        """Resolve all parameters in a dictionary."""
        resolved = {}
        for key, value in params.items():
            if isinstance(value, dict):
                resolved[key] = self._resolve_parameters(value, step_name)
            elif isinstance(value, list):
                resolved[key] = [
                    self._resolve_parameter(item, step_name)
                    for item in value
                ]
            else:
                resolved[key] = self._resolve_parameter(value, step_name)
        return resolved
    
    def _execute_step(
        self,
        step: Dict[str, Any],
        step_index: int
    ) -> Any:
        """
        Execute a single workflow step.
        
        Parameters
        ----------
        step : dict
            Step definition
        step_index : int
            Step index (for logging)
            
        Returns
        -------
        any
            Step result
        """
        step_name = step.get('name') or step.get('step') or f"step_{step_index}"
        step_type = step.get('type') or step.get('function')
        
        if not step_type:
            raise ValueError(f"Step {step_name} missing 'type' or 'function' field")
        
        self.logger.info(f"Executing step {step_index + 1}: {step_name} ({step_type})")
        
        # Get function from registry
        func = STEP_REGISTRY.get(step_type)
        if func is None:
            raise ValueError(
                f"Unknown step type: {step_type}. "
                f"Available: {list(STEP_REGISTRY.keys())}"
            )
        
        # Get parameters
        params = step.get('params', step.get('parameters', {}))
        
        # Resolve parameters (references, config values)
        params = self._resolve_parameters(params, step_name)
        
        # Add config if function accepts it (check signature)
        import inspect
        sig = inspect.signature(func)
        if 'config' in sig.parameters:
            params['config'] = self.config
        
        # Note: DataFrame column references are resolved in _resolve_parameter
        # This section is kept for any additional post-processing if needed
        
        # Execute function
        try:
            result = func(**params)
            self.results[step_name] = result
            self.logger.info(f"✓ Step {step_name} completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"✗ Step {step_name} failed: {e}")
            raise
    
    def execute(
        self,
        workflow: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a workflow definition.
        
        Parameters
        ----------
        workflow : dict
            Workflow definition with 'steps' list
            
        Returns
        -------
        dict
            Results from all steps
        """
        self.logger.info("Starting workflow execution")
        
        # Load config if specified
        config_file = workflow.get('config')
        if config_file:
            config_path = Path(config_file)
            # Try as absolute path first, then relative to working dir
            if not config_path.is_absolute():
                config_path = self.working_dir / config_path
            
            # Also try relative to geosuite package if not found
            if not config_path.exists():
                from ..config import default_config
                import os
                package_dir = Path(__file__).parent.parent
                alt_path = package_dir / "config" / config_path.name
                if alt_path.exists():
                    config_path = alt_path
            
            if config_path.exists():
                self.config = load_config(config_path)
                self.logger.info(f"Loaded config from {config_path}")
            else:
                self.logger.warning(f"Config file not found: {config_file}, using defaults")
                # Load default config
                default_config_path = Path(__file__).parent.parent / "config" / "default_config.yaml"
                if default_config_path.exists() and self.config is None:
                    self.config = load_config(default_config_path)
        
        # Get steps
        steps = workflow.get('steps', [])
        if not steps:
            raise ValueError("Workflow must contain 'steps' list")
        
        # Resolve dependencies
        ordered_steps = self._resolve_dependencies(steps)
        
        # Execute each step
        for i, step in enumerate(ordered_steps):
            try:
                self._execute_step(step, i)
            except Exception as e:
                self.logger.error(f"Workflow failed at step {i + 1}: {e}")
                if workflow.get('stop_on_error', True):
                    raise
        
        self.logger.info(f"Workflow completed successfully ({len(steps)} steps)")
        return self.results
    
    def execute_file(
        self,
        file_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Load and execute workflow from file.
        
        Parameters
        ----------
        file_path : str or Path
            Path to workflow file
            
        Returns
        -------
        dict
            Results from all steps
        """
        workflow = self.load_workflow_file(file_path)
        return self.execute(workflow)


def run_workflow(
    workflow_file: Union[str, Path],
    config: Optional[ConfigManager] = None,
    working_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Convenience function to run a workflow from a file.
    
    Parameters
    ----------
    workflow_file : str or Path
        Path to workflow YAML/JSON file
    config : ConfigManager, optional
        Configuration manager
    working_dir : str or Path, optional
        Working directory
        
    Returns
    -------
    dict
        Results from all steps
        
    Example
    -------
    >>> from geosuite.workflows import run_workflow
    >>> results = run_workflow("my_workflow.yaml")
    """
    orchestrator = WorkflowOrchestrator(config=config, working_dir=working_dir)
    return orchestrator.execute_file(workflow_file)


def load_workflow(
    file_path: Union[str, Path]
) -> Dict[str, Any]:
    """
    Load workflow definition without executing.
    
    Parameters
    ----------
    file_path : str or Path
        Path to workflow file
        
    Returns
    -------
    dict
        Workflow definition
    """
    orchestrator = WorkflowOrchestrator()
    return orchestrator.load_workflow_file(file_path)

