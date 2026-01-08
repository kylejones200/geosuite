"""
Workflow orchestration for GeoSuite.

Provides config-driven workflows that can be defined in YAML/JSON
and executed programmatically or via CLI.
"""

from .orchestrator import WorkflowOrchestrator, run_workflow, load_workflow
from .config_aware import (
    config_aware,
    get_config_value,
    ConfigAwareFunction
)

__all__ = [
    "WorkflowOrchestrator",
    "run_workflow",
    "load_workflow",
    "config_aware",
    "get_config_value",
    "ConfigAwareFunction",
]

