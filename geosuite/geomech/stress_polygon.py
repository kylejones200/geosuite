"""
Stress polygon analysis for faulting regime determination.

All plots use signalplot for consistent, minimalist styling.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import signalplot
from typing import Tuple, Dict
from matplotlib.figure import Figure

# Apply signalplot style globally for this module
signalplot.apply()


def stress_polygon_limits(
    sv: float,
    pp: float,
    shmin: float = None,
    mu: float = 0.6,
    cohesion: float = 0.0
) -> Dict[str, Tuple[float, float]]:
    """
    Calculate stress polygon limits for given depth.
    
    Returns the allowable ranges for SHmax based on faulting theory.
    
    Args:
        sv: Vertical stress (MPa)
        pp: Pore pressure (MPa)
        shmin: Minimum horizontal stress (MPa), optional
        mu: Coefficient of friction (typically 0.6-1.0)
        cohesion: Cohesion (MPa), typically 0
        
    Returns:
        Dictionary with stress limits for each regime
    """
    # Mohr-Coulomb failure criterion
    q = np.sqrt(mu**2 + 1) + mu
    
    # Effective stresses
    sv_eff = sv - pp
    
    # Normal faulting: Sv > SHmax > Shmin
    # SHmax_max = Sv
    # SHmax_min = (Sv - C) / q + Pp
    nf_max = sv
    nf_min = (sv_eff - cohesion) / q + pp
    
    # Strike-slip faulting: SHmax > Sv > Shmin
    # SHmax_max = q * (Sv - Pp) + C + Pp
    # SHmax_min = Sv
    ss_min = sv
    ss_max = q * sv_eff + cohesion + pp
    
    # Reverse/Thrust faulting: SHmax > Shmin > Sv
    # SHmax_min = q * (Sv - Pp) + C + Pp
    # If Shmin is known: SHmax_max = q * (Shmin - Pp) + C + Pp
    rf_min = q * sv_eff + cohesion + pp
    if shmin is not None:
        shmin_eff = shmin - pp
        rf_max = q * shmin_eff + cohesion + pp
    else:
        rf_max = None
    
    return {
        'normal': (nf_min, nf_max),
        'strike_slip': (ss_min, ss_max),
        'reverse': (rf_min, rf_max)
    }


def plot_stress_polygon(
    depths: np.ndarray,
    sv: np.ndarray,
    pp: np.ndarray,
    shmin: np.ndarray = None,
    mu: float = 0.6,
    cohesion: float = 0.0,
    title: str = 'Stress Polygon',
    figsize: Tuple[float, float] = (6, 8)
) -> Figure:
    """
    Create stress polygon plot showing allowable stress regimes.
    
    Args:
        depths: Depth array (meters)
        sv: Vertical stress array (MPa)
        pp: Pore pressure array (MPa)
        shmin: Minimum horizontal stress array (MPa), optional
        mu: Coefficient of friction
        cohesion: Cohesion (MPa)
        title: Plot title
        figsize: Figure size (width, height) in inches
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate bounds
    q = np.sqrt(mu**2 + 1) + mu
    sv_eff = sv - pp
    
    # Normal faulting upper bound (SHmax = Sv)
    nf_upper = sv
    
    # Normal faulting lower bound
    nf_lower = (sv_eff - cohesion) / q + pp
    
    # Strike-slip upper bound
    ss_upper = q * sv_eff + cohesion + pp
    
    # Add normal faulting zone
    ax.fill_betweenx(depths, nf_lower, nf_upper, 
                      alpha=0.3, color='orange', label='Normal Faulting')
    ax.plot(nf_upper, depths, 'orange', linewidth=1.5, alpha=0.7)
    ax.plot(nf_lower, depths, 'orange', linewidth=1.5, alpha=0.7)
    
    # Add strike-slip zone
    ax.fill_betweenx(depths, sv, ss_upper,
                      alpha=0.3, color='blue', label='Strike-Slip')
    ax.plot(sv, depths, 'blue', linewidth=1.5, alpha=0.7)
    ax.plot(ss_upper, depths, 'blue', linewidth=1.5, alpha=0.7)
    
    # Add reverse faulting zone if Shmin is provided
    if shmin is not None:
        shmin_eff = shmin - pp
        rf_upper = q * shmin_eff + cohesion + pp
        
        ax.fill_betweenx(depths, ss_upper, rf_upper,
                          alpha=0.3, color='red', label='Reverse Faulting')
        ax.plot(rf_upper, depths, 'red', linewidth=1.5, alpha=0.7)
    
    # Add Sv line
    ax.plot(sv, depths, 'k--', linewidth=1.5, label='Sv')
    
    # Add Pp line
    ax.plot(pp, depths, 'k:', linewidth=1.5, label='Pp')
    
    # Add Shmin line if provided
    if shmin is not None:
        ax.plot(shmin, depths, 'k-.', linewidth=1.5, label='Shmin')
    
    # Labels and title
    ax.set_xlabel('Stress (MPa)', fontsize=11)
    ax.set_ylabel('Depth (m)', fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    
    # Invert y-axis for depth
    ax.invert_yaxis()
    
    # Legend
    ax.legend(frameon=False, fontsize=9, loc='best')
    
    # Add parameter text
    param_text = f'μ={mu}, C={cohesion} MPa'
    ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


def determine_stress_regime(
    sv: float,
    shmin: float,
    shmax: float,
    tolerance: float = 0.1
) -> str:
    """
    Determine stress regime from principal stresses.
    
    Args:
        sv: Vertical stress (MPa)
        shmin: Minimum horizontal stress (MPa)
        shmax: Maximum horizontal stress (MPa)
        tolerance: Tolerance for numerical comparison (MPa)
        
    Returns:
        Stress regime: 'normal', 'strike_slip', 'reverse', or 'unknown'
    """
    s1, s2, s3 = sorted([sv, shmin, shmax], reverse=True)
    
    # Map vertical stress position to faulting regime
    regime_map = [
        (np.isclose(sv, s1, atol=tolerance), 'normal'),       # Sv ≈ S1 (largest)
        (np.isclose(sv, s3, atol=tolerance), 'reverse'),      # Sv ≈ S3 (smallest)
        (np.isclose(sv, s2, atol=tolerance), 'strike_slip'),  # Sv ≈ S2 (intermediate)
    ]
    
    return next((regime for condition, regime in regime_map if condition), 'unknown')
