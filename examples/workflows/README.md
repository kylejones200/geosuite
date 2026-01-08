# GeoSuite Config-Driven Workflows

This directory contains example workflows demonstrating config-driven orchestration with GeoSuite.

## Overview

GeoSuite workflows allow you to define complete analysis pipelines in YAML/JSON files. The workflow system:

- **Reads from config**: Functions automatically load parameters from configuration files
- **Chains steps**: Outputs from one step can be used as inputs to the next
- **Supports references**: Use `${step_name.output}` or `${step_name.column}` to reference previous results
- **Config integration**: Use `${config.key}` to reference configuration values

## Running Workflows

### Command Line

```bash
# Run a workflow
geosuite-workflow petrophysical_workflow.yaml

# With custom config
geosuite-workflow workflow.yaml --config custom_config.yaml

# Dry run (validate only)
geosuite-workflow workflow.yaml --dry-run

# Verbose output
geosuite-workflow workflow.yaml --verbose
```

### Python API

```python
from geosuite.workflows import run_workflow

# Run workflow from file
results = run_workflow("petrophysical_workflow.yaml")

# Access results
porosity = results['calculate_porosity']
saturation = results['calculate_saturation']
```

## Workflow Format

Workflows are defined in YAML or JSON:

```yaml
# Optional: Path to config file
config: "path/to/config.yaml"

# Workflow metadata
name: "My Workflow"
description: "Description of what this workflow does"

# Steps (executed in order)
steps:
  - name: step_name
    type: function_name
    description: "What this step does"
    params:
      param1: value1
      param2: "${previous_step.output}"
      param3: "${config.key}"
```

## Parameter References

### Step Outputs

Reference outputs from previous steps:

```yaml
params:
  df: "${load_data.output}"  # Full output from previous step
  phi: "${calculate_porosity.output}"  # Specific output
  rhob: "${load_data.output.RHOB}"  # DataFrame column
```

### Configuration Values

Reference config values:

```yaml
params:
  m: "${config.petro.archie.m}"  # Read from config
  rw: 0.05  # Or override with explicit value
```

### Config-Aware Functions

Most GeoSuite functions automatically read parameters from config when not provided:

```yaml
# In workflow - parameters can be omitted to use config defaults
- name: calculate_saturation
  type: calculate_water_saturation
  params:
    phi: "${porosity.output}"
    rt: "${load_data.output.RT}"
    # rw, m, n, a will be read from config automatically
```

## Example Workflows

### 1. Petrophysical Analysis (`petrophysical_workflow.yaml`)

Complete petrophysical workflow:
- Load data
- Calculate porosity from density
- Calculate water saturation (Archie equation)
- Create Pickett and Buckles plots
- Generate strip chart

### 2. Geomechanical Analysis (`geomechanical_workflow.yaml`)

Geomechanical model building:
- Calculate overburden stress
- Estimate pore pressure
- Stress polygon analysis
- Visualization

### 3. Complete Analysis (`complete_analysis_workflow.yaml`)

Integrated workflow combining:
- Petrophysics
- Geomechanics
- Stratigraphy (change-point detection)

## Configuration

Default configuration is in `geosuite/config/default_config.yaml`. Key sections:

- `petro.archie.*`: Archie equation parameters (a, m, n, rw)
- `geomech.default.*`: Geomechanical defaults (g, rho_water, etc.)
- `ml.default.*`: ML model defaults
- `stratigraphy.*`: Change-point detection settings

## Available Step Types

Common workflow steps (see `WorkflowOrchestrator` for complete list):

**Data Loading:**
- `load_las`: Load LAS file
- `load_demo_data`: Load demo well log data

**Petrophysics:**
- `calculate_porosity`: Porosity from density
- `calculate_water_saturation`: Water saturation (Archie)
- `calculate_formation_factor`: Formation factor
- `pickett_plot`: Pickett plot
- `buckles_plot`: Buckles plot

**Geomechanics:**
- `calculate_overburden_stress`: Vertical stress
- `calculate_hydrostatic_pressure`: Hydrostatic pressure
- `calculate_pore_pressure`: Pore pressure (Eaton)
- `stress_polygon_limits`: Stress polygon analysis
- `plot_stress_polygon`: Stress polygon visualization

**Stratigraphy:**
- `preprocess_log`: Log preprocessing
- `detect_pelt`: PELT change-point detection
- `detect_bayesian_online`: Bayesian change-point detection

**Machine Learning:**
- `train_facies_classifier`: Train facies classifier

**Visualization:**
- `create_strip_chart`: Well log strip chart

## Best Practices

1. **Use config for defaults**: Let functions read from config for standard parameters
2. **Override when needed**: Explicitly specify parameters that differ from config
3. **Name steps clearly**: Use descriptive step names for easier debugging
4. **Chain steps**: Use step references to build pipelines
5. **Handle errors**: Set `stop_on_error: false` to continue despite failures

## Extending Workflows

Register custom functions as workflow steps:

```python
from geosuite.workflows import register_step

def my_custom_function(param1, param2):
    # Your function
    return result

register_step("my_function", my_custom_function)
```

Then use in workflows:

```yaml
- name: custom_step
  type: my_function
  params:
    param1: value1
    param2: value2
```

