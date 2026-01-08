# GeoSuite Config-Driven Workflows - Complete Guide

## Overview

GeoSuite now supports **full config-driven workflows** where:
- **Functions automatically read from config** when parameters aren't provided
- **Workflows are defined in YAML/JSON** files
- **Steps chain together** using reference syntax
- **Easy orchestration** via CLI or Python API

## Quick Start

### 1. Basic Usage

```python
from geosuite.workflows import run_workflow

# Run a workflow
results = run_workflow("examples/workflows/petrophysical_workflow.yaml")

# Access results
porosity = results['calculate_porosity']
saturation = results['calculate_saturation']
```

### 2. Command Line

```bash
# Run workflow
geosuite-workflow examples/workflows/petrophysical_workflow.yaml

# With custom config
geosuite-workflow workflow.yaml --config my_config.yaml

# Validate workflow
geosuite-workflow workflow.yaml --dry-run
```

## Config-Aware Functions

All major GeoSuite functions now support config:

```python
from geosuite.petro import calculate_water_saturation
from geosuite.config import load_config

# Load config
config = load_config("config.yaml")

# Function automatically reads from config if params not provided
sw = calculate_water_saturation(
    phi=porosity_array,
    rt=resistivity_array,
    # rw, m, n, a will be read from config.petro.archie.*
    config=config  # Pass config explicitly
)

# Or use global config (no config parameter needed)
from geosuite.config import set_config
set_config("petro.archie.rw", 0.05)
sw = calculate_water_saturation(phi=porosity_array, rt=resistivity_array)
```

## Workflow Definition

### Basic Structure

```yaml
name: "Workflow Name"
description: "What this workflow does"

# Optional: Path to config file
config: "path/to/config.yaml"

steps:
  - name: step_1
    type: function_name
    params:
      param1: value1
      param2: "${previous_step.output}"
```

### Parameter References

**Step Outputs:**
- `${step_name.output}` - Full output from step
- `${step_name.column}` - DataFrame column from step output

**Config Values:**
- `${config.key}` - Read from configuration

**Example:**
```yaml
steps:
  - name: load_data
    type: load_demo_data
  
  - name: porosity
    type: calculate_porosity
    params:
      rhob: "${load_data.output.RHOB}"  # DataFrame column
  
  - name: saturation
    type: calculate_water_saturation
    params:
      phi: "${porosity.output}"  # Array output
      rt: "${load_data.output.RT}"
      # rw, m, n, a read from config automatically
```

## Configuration File Structure

Default config: `geosuite/config/default_config.yaml`

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

## Available Step Types

### Data Loading
- `load_las`: Load LAS file
- `load_demo_data`: Load demo well log data

### Petrophysics
- `calculate_porosity`: Porosity from density (config: `petro.density.*`)
- `calculate_water_saturation`: Water saturation (config: `petro.archie.*`)
- `calculate_formation_factor`: Formation factor
- `pickett_plot`: Pickett plot (config: `petro.archie.*`)
- `buckles_plot`: Buckles plot

### Geomechanics
- `calculate_overburden_stress`: Vertical stress (config: `geomech.default.g`)
- `calculate_hydrostatic_pressure`: Hydrostatic (config: `geomech.default.*`)
- `calculate_pore_pressure`: Pore pressure (config: `geomech.pore_pressure.*`)
- `stress_polygon_limits`: Stress polygon
- `plot_stress_polygon`: Visualization

### Stratigraphy
- `preprocess_log`: Log preprocessing
- `detect_pelt`: PELT change-point detection
- `detect_bayesian_online`: Bayesian detection

### Machine Learning
- `train_facies_classifier`: Train classifier (config: `ml.default.*`)

### Visualization
- `create_strip_chart`: Well log display

## Examples

See `examples/workflows/` for complete examples:
- `simple_example.yaml` - Minimal example
- `petrophysical_workflow.yaml` - Full petrophysical analysis
- `geomechanical_workflow.yaml` - Geomechanical modeling
- `complete_analysis_workflow.yaml` - Integrated analysis

## Extending Workflows

Register custom functions:

```python
from geosuite.workflows import register_step

def my_function(param1, param2):
    # Your code
    return result

register_step("my_function", my_function)
```

## Best Practices

1. **Use config for standard values** - Let functions read defaults from config
2. **Override when needed** - Specify parameters explicitly when they differ
3. **Chain steps** - Use `${step.output}` to build pipelines
4. **Name clearly** - Descriptive step names help debugging
5. **Document workflows** - Add descriptions to each step

## Architecture

- **ConfigManager**: Loads and manages YAML/JSON configs
- **WorkflowOrchestrator**: Executes workflow definitions
- **Config-aware functions**: Auto-read from config when parameters missing
- **Step registry**: Maps step types to functions
- **Parameter resolution**: Handles references and config lookups

