# GeoSuite API Standards

This document defines the standard function signatures and conventions for GeoSuite.

## Type Hints

### Input Types
- **Array inputs**: Always use `Union[np.ndarray, pd.Series]` for array-like inputs
- **Scalar inputs**: Use specific types (`float`, `int`, `str`, `bool`)
- **Optional inputs**: Use `Optional[Type]` with `None` as default

### Return Types
- **Calculation functions**: Always return `np.ndarray` (convert pd.Series to np.ndarray)
- **Plotting functions**: Return `matplotlib.figure.Figure`
- **DataFrame functions**: Return `pd.DataFrame`

## Parameter Naming

### Standard Names
- **Porosity**: `phi` (fraction, 0-1)
- **Water saturation**: `sw` (fraction, 0-1)
- **Oil saturation**: `so` (fraction, 0-1)
- **Gas saturation**: `sg` (fraction, 0-1)
- **Resistivity**: `rt` (ohm-m) for true resistivity, `rw` for water resistivity
- **Bulk density**: `rhob` (g/cc)
- **Matrix density**: `rho_matrix` (g/cc)
- **Fluid density**: `rho_fluid` (g/cc)
- **Water density**: `rho_water` (g/cc)
- **Depth**: `depth` (meters)
- **Overburden stress**: `sv` (MPa)
- **Pore pressure**: `pp` (MPa)
- **Hydrostatic pressure**: `ph` (MPa)
- **Gravitational acceleration**: `g` (m/s²), default 9.81
- **Archie parameters**: `a` (tortuosity), `m` (cementation), `n` (saturation), `rw` (water resistivity)

### Default Values
- `a = 1.0` (Archie tortuosity)
- `m = 2.0` (Archie cementation exponent)
- `n = 2.0` (Archie saturation exponent)
- `rw = 0.05` (formation water resistivity, ohm-m)
- `g = 9.81` (gravitational acceleration, m/s²)
- `rho_water = 1.03` (water density, g/cc)
- `rho_matrix = 2.65` (matrix density, g/cc)
- `rho_fluid = 1.0` (fluid density, g/cc)

## Function Naming

### Calculation Functions
- Prefix: `calculate_` (e.g., `calculate_water_saturation`)
- Verb + noun pattern
- Clear, descriptive names

### Plotting Functions
- Suffix: `_plot` (e.g., `pickett_plot`)
- Or descriptive name (e.g., `neutron_density_crossplot`)

## Function Signature Pattern

### Calculation Functions
```python
def calculate_property(
    primary_input: Union[np.ndarray, pd.Series],
    secondary_input: Optional[Union[np.ndarray, pd.Series]] = None,
    parameter1: float = default_value,
    parameter2: float = default_value,
    **kwargs
) -> np.ndarray:
    """
    Brief description.
    
    Formula or equation if applicable.
    
    Args:
        primary_input: Description (units)
        secondary_input: Description (units), optional
        parameter1: Description (units), default default_value
        parameter2: Description (units), default default_value
        
    Returns:
        Description (units) as numpy array
        
    Example:
        >>> result = calculate_property(input_array)
    """
    # Convert inputs to numpy arrays
    primary_input = np.asarray(primary_input, dtype=float)
    if secondary_input is not None:
        secondary_input = np.asarray(secondary_input, dtype=float)
    
    # Validation
    if len(primary_input) == 0:
        raise ValueError("primary_input must not be empty")
    
    # Calculation
    result = ...
    
    # Return as numpy array
    return result
```

### Plotting Functions
```python
def plot_name(
    df: pd.DataFrame,
    x_col: str = 'default_x',
    y_col: str = 'default_y',
    color_by: Optional[str] = None,
    title: str = 'Default Title',
    figsize: Tuple[float, float] = (8, 6)
) -> Figure:
    """
    Brief description.
    
    Args:
        df: DataFrame with data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        color_by: Optional column to color by
        title: Plot title
        figsize: Figure size (width, height) in inches
        
    Returns:
        Matplotlib Figure object
    """
    ...
    return fig
```

## Error Handling

- Always validate input arrays are not empty
- Always validate array lengths match when multiple arrays required
- Use `np.asarray()` to convert inputs
- Use `np.clip()` for physical bounds when appropriate
- Use `np.where()` or `np.nan` for invalid inputs rather than raising errors immediately

## Logging

- Use module-level logger: `logger = logging.getLogger(__name__)`
- Log warnings for invalid inputs (e.g., values outside physical bounds)
- Log info for significant operations (optional)

