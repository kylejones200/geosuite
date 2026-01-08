# GeoSuite Config-Driven Workflows

GeoSuite now supports **full config-driven workflows** where all modules can be orchestrated easily through YAML/JSON configuration files.

## ✅ What's Been Implemented

### 1. Config-Aware Functions

All major functions now automatically read parameters from configuration:

**Petrophysics:**
- `calculate_water_saturation()` - Reads `petro.archie.*` from config
- `calculate_porosity_from_density()` - Reads `petro.density.*` from config
- `calculate_formation_factor()` - Reads `petro.archie.*` from config
- `pickett_plot()` - Reads `petro.archie.*` from config

**Geomechanics:**
- `calculate_overburden_stress()` - Reads `geomech.default.g` from config
- `calculate_hydrostatic_pressure()` - Reads `geomech.default.*` from config

### 2. Workflow Orchestrator

Complete workflow execution system:
- Loads workflow definitions from YAML/JSON
- Executes steps in order with dependency resolution
- Supports step output references: `${step_name.output}`
- Supports DataFrame column access: `${step_name.column}`
- Supports config references: `${config.key}`
- Automatic config loading and parameter injection

### 3. Configuration Management

- `ConfigManager` class for loading YAML/JSON configs
- Default config file with all module settings
- Global config access via `get_config()`, `set_config()`
- Hierarchical config keys with dot notation

### 4. CLI Integration

New command: `geosuite-workflow`
- Run workflows from command line
- Validate workflows with `--dry-run`
- Support custom config files

## Usage Examples

### Example 1: Config-Aware Function Calls

```python
from geosuite.petro import calculate_water_saturation
from geosuite.config import load_config

# Load config
config = load_config("my_config.yaml")

# Function automatically uses config for missing parameters
sw = calculate_water_saturation(
    phi=porosity_array,
    rt=resistivity_array,
    # rw, m, n, a read from config.petro.archie.*
    config=config
)
```

### Example 2: Workflow from YAML

```yaml
# workflow.yaml
name: "Petrophysical Analysis"

steps:
  - name: load_data
    type: load_demo_data
  
  - name: porosity
    type: calculate_porosity
    params:
      rhob: "${load_data.output.RHOB}"
      # rho_matrix, rho_fluid from config
  
  - name: saturation
    type: calculate_water_saturation
    params:
      phi: "${porosity.output}"
      rt: "${load_data.output.RT}"
      # rw, m, n, a from config
```

Run with:
```bash
geosuite-workflow workflow.yaml
```

Or in Python:
```python
from geosuite.workflows import run_workflow
results = run_workflow("workflow.yaml")
```

### Example 3: Custom Configuration

```yaml
# custom_config.yaml
petro:
  archie:
    m: 2.2  # Custom cementation exponent
    n: 2.0
    rw: 0.06  # Custom water resistivity

geomech:
  default:
    rho_water: 1.05  # Brine density
```

```python
from geosuite.config import load_config
from geosuite.workflows import run_workflow

config = load_config("custom_config.yaml")
results = run_workflow("workflow.yaml", config=config)
```

## Key Features

### ✅ Automatic Config Reading
Functions check config when parameters are `None` or not provided

### ✅ Workflow Chaining
Steps can reference outputs from previous steps

### ✅ DataFrame Integration
Easily access DataFrame columns: `${step.output.COLUMN}`

### ✅ Flexible Parameters
Override config values explicitly when needed

### ✅ Easy Extension
Register custom functions as workflow steps

## Files Created/Modified

**New Modules:**
- `geosuite/workflows/` - Workflow orchestration system
  - `orchestrator.py` - Workflow execution engine
  - `config_aware.py` - Config-aware function utilities
  - `__init__.py` - Module exports

**Updated Functions:**
- `geosuite/petro/calculations.py` - Added config support
- `geosuite/petro/pickett.py` - Added config support
- `geosuite/geomech/pressures.py` - Added config support
- `geosuite/config/manager.py` - Enhanced get_config()

**Workflow Examples:**
- `examples/workflows/petrophysical_workflow.yaml`
- `examples/workflows/geomechanical_workflow.yaml`
- `examples/workflows/complete_analysis_workflow.yaml`
- `examples/workflows/simple_example.yaml`
- `examples/workflows/README.md` - Documentation

**CLI:**
- `geosuite/cli/workflow.py` - Workflow CLI command
- Updated `pyproject.toml` - Added CLI entry point

## Configuration Structure

See `geosuite/config/default_config.yaml` for complete structure:

```yaml
petro:
  archie:
    a: 1.0
    m: 2.0
    n: 2.0
    rw: 0.05
  density:
    rho_matrix: 2.65
    rho_fluid: 1.0

geomech:
  default:
    rho_water: 1.03
    g: 9.81
  pore_pressure:
    eaton:
      exponent: 3.0
```

## Next Steps

The system is ready to use! You can:
1. Create custom workflows in YAML
2. Use config-aware functions in your code
3. Build complex pipelines easily
4. Share workflows as configuration files

For more examples, see `examples/workflows/` directory.

