# GeoSuite Plotting with signalplot

All plots in GeoSuite use [signalplot](https://github.com/kylejones200/signalplot) for consistent, minimalist styling that produces publication-quality figures.

## Overview

signalplot provides:
- **Minimalist styling**: Clean, professional appearance
- **Consistent aesthetics**: All plots share the same visual language
- **Publication-ready**: High-quality figures suitable for papers and presentations
- **Automatic spine handling**: Clean borders and grid lines

## Plotting Modules

### Core Plotting (`geosuite.plotting`)

**Strip Charts (Well Logs)**
- `create_strip_chart()`: Multi-track well log visualization
- `create_facies_log_plot()`: Well logs with facies track
- `add_log_track()`: Add individual log tracks
- `add_facies_track()`: Add facies visualization

**Example:**
```python
from geosuite.plotting import create_strip_chart

fig = create_strip_chart(
    df,
    log_cols=['GR', 'RHOB', 'NPHI', 'RT'],
    facies_col='Facies',
    title='Well Log Analysis'
)
```

### Petrophysics Plots (`geosuite.petro`)

**Pickett Plot**
- `pickett_plot()`: Porosity-resistivity crossplot for water saturation analysis

**Buckles Plot**
- `buckles_plot()`: Bulk Volume Water vs Porosity for reservoir quality

**Lithology Crossplots**
- `neutron_density_crossplot()`: Neutron-density crossplot for lithology identification

**Example:**
```python
from geosuite.petro import pickett_plot, buckles_plot

# Pickett plot for saturation analysis
fig1 = pickett_plot(df, porosity_col='NPHI', resistivity_col='RT')

# Buckles plot for reservoir quality
fig2 = buckles_plot(df, porosity_col='PHIND', sw_col='SW')
```

### Geomechanics Plots (`geosuite.geomech`)

**Stress Polygon**
- `plot_stress_polygon()`: Visualize allowable stress regimes

**Pressure Profiles**
- `plot_pressure_profile()`: Depth profiles of pressure and stress
- `plot_mud_weight_profile()`: Equivalent mud weight profiles

**Example:**
```python
from geosuite.geomech import plot_stress_polygon, plot_pressure_profile

# Stress polygon
fig1 = plot_stress_polygon(depths, sv, pp, shmin)

# Pressure profile
fig2 = plot_pressure_profile(df, pressure_cols=['Sv', 'Ph', 'Pp'])
```

### Machine Learning Plots (`geosuite.ml`)

**Confusion Matrix**
- `plot_confusion_matrix()`: Heatmap visualization of classification results

**Example:**
```python
from geosuite.ml import plot_confusion_matrix

fig = plot_confusion_matrix(
    cm, 
    labels=['Sand', 'Shale', 'Carbonate'],
    normalize=True
)
```

## signalplot Integration

All plotting modules automatically apply signalplot styling:

```python
import signalplot

# Applied automatically in each module
signalplot.apply()
```

This ensures:
- Consistent color schemes
- Clean spines (top and right removed by default)
- Professional typography
- Minimalist grid styling
- Publication-ready appearance

## Customization

While signalplot provides excellent defaults, you can customize plots using standard matplotlib commands:

```python
from geosuite.plotting import create_strip_chart
import matplotlib.pyplot as plt

fig = create_strip_chart(df)

# Customize after creation
ax = fig.axes[0]
ax.set_xlim(0, 200)  # Custom axis limits
ax.grid(True, alpha=0.3)  # Add grid

plt.savefig('custom_plot.png', dpi=300, bbox_inches='tight')
```

## Export Options

All plots return matplotlib Figure objects, so you can use standard matplotlib export methods:

```python
fig = pickett_plot(df)

# Save as PNG
fig.savefig('pickett_plot.png', dpi=300, bbox_inches='tight')

# Save as PDF
fig.savefig('pickett_plot.pdf', bbox_inches='tight')

# Save as SVG
fig.savefig('pickett_plot.svg', bbox_inches='tight')
```

## References

- [signalplot GitHub](https://github.com/kylejones200/signalplot) - The styling library
- [GeoSuite Documentation](https://geosuite.readthedocs.io) - Full API documentation

