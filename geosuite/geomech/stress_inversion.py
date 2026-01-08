"""
Stress inversion tools for determining stress magnitudes from wellbore failure observations.

Uses breakout and drilling-induced fracture (DIF) data to constrain stress magnitudes.
"""

import numpy as np
import pandas as pd
import logging
from typing import Union, Optional, Dict, Tuple, List
from scipy.optimize import minimize, differential_evolution

logger = logging.getLogger(__name__)


def invert_stress_from_breakout(
    breakout_width: Union[float, np.ndarray],
    breakout_azimuth: Union[float, np.ndarray],
    depth: Union[float, np.ndarray],
    sv: Union[float, np.ndarray],
    pp: Union[float, np.ndarray],
    ucs: float = 50.0,
    poisson: float = 0.25,
    wellbore_azimuth: float = 0.0,
    wellbore_inclination: float = 0.0,
    method: str = 'optimization'
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Invert stress magnitudes from breakout observations.
    
    Uses breakout width and azimuth to constrain SHmax and Shmin.
    
    Args:
        breakout_width: Breakout width in degrees
        breakout_azimuth: Breakout azimuth in degrees (from north)
        depth: Depth (m)
        sv: Vertical stress (MPa)
        pp: Pore pressure (MPa)
        ucs: Unconfined compressive strength (MPa)
        poisson: Poisson's ratio
        wellbore_azimuth: Wellbore azimuth in degrees
        wellbore_inclination: Wellbore inclination in degrees
        method: Inversion method ('optimization' or 'analytical')
        
    Returns:
        Dictionary with estimated SHmax, Shmin, and stress ratio
    """
    breakout_width = np.asarray(breakout_width, dtype=float)
    breakout_azimuth = np.asarray(breakout_azimuth, dtype=float)
    depth = np.asarray(depth, dtype=float)
    sv = np.asarray(sv, dtype=float)
    pp = np.asarray(pp, dtype=float)
    
    if method == 'optimization':
        return _invert_stress_optimization(
            breakout_width, breakout_azimuth, depth, sv, pp, ucs, poisson,
            wellbore_azimuth, wellbore_inclination
        )
    else:
        return _invert_stress_analytical(
            breakout_width, depth, sv, pp, ucs, poisson
        )


def _invert_stress_optimization(
    breakout_width: np.ndarray,
    breakout_azimuth: np.ndarray,
    depth: np.ndarray,
    sv: np.ndarray,
    pp: np.ndarray,
    ucs: float,
    poisson: float,
    wellbore_azimuth: float,
    wellbore_inclination: float
) -> Dict[str, np.ndarray]:
    """Invert stress using optimization approach."""
    results = []
    
    for i in range(len(breakout_width)):
        def objective(params):
            shmax, shmin = params
            shmax_eff = shmax - pp[i]
            shmin_eff = shmin - pp[i]
            sv_eff = sv[i] - pp[i]
            
            # Simplified breakout width model
            # Breakout occurs when tangential stress exceeds UCS
            stress_diff = shmax_eff - shmin_eff
            predicted_width = np.degrees(2 * np.arcsin(np.clip(ucs / (2 * stress_diff), -1, 1)))
            
            # Minimize difference between observed and predicted
            width_error = (predicted_width - breakout_width[i]) ** 2
            
            # Add constraint penalties
            penalty = 0.0
            if shmax < shmin:
                penalty += 1000
            if shmax > sv[i] * 1.5:
                penalty += 1000
            if shmin < pp[i]:
                penalty += 1000
            
            return width_error + penalty
        
        # Initial guess: typical stress ratios
        initial_guess = [sv[i] * 1.1, sv[i] * 0.7]
        bounds = [(pp[i], sv[i] * 1.5), (pp[i], sv[i])]
        
        try:
            result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
            if result.success:
                shmax_est, shmin_est = result.x
            else:
                shmax_est, shmin_est = initial_guess
        except:
            shmax_est, shmin_est = initial_guess
        
        stress_ratio = (shmax_est - pp[i]) / (sv[i] - pp[i]) if (sv[i] - pp[i]) > 0 else 1.0
        
        results.append({
            'shmax': shmax_est,
            'shmin': shmin_est,
            'stress_ratio': stress_ratio
        })
    
    return {
        'shmax': np.array([r['shmax'] for r in results]),
        'shmin': np.array([r['shmin'] for r in results]),
        'stress_ratio': np.array([r['stress_ratio'] for r in results])
    }


def _invert_stress_analytical(
    breakout_width: np.ndarray,
    depth: np.ndarray,
    sv: np.ndarray,
    pp: np.ndarray,
    ucs: float,
    poisson: float
) -> Dict[str, np.ndarray]:
    """Invert stress using analytical approach (simplified)."""
    sv_eff = sv - pp
    
    # Simplified model: breakout width relates to stress difference
    # Assuming breakout occurs when tangential stress = UCS
    stress_diff = ucs / (2 * np.sin(np.radians(breakout_width / 2)))
    stress_diff = np.clip(stress_diff, 0, sv_eff)
    
    # Estimate Shmin from Poisson's ratio
    shmin = pp + poisson / (1 - poisson) * sv_eff
    
    # Estimate SHmax from stress difference
    shmax = shmin + stress_diff
    
    # Ensure physical constraints
    shmax = np.clip(shmax, pp, sv * 1.5)
    shmin = np.clip(shmin, pp, sv)
    
    stress_ratio = (shmax - pp) / sv_eff
    stress_ratio = np.clip(stress_ratio, 0.5, 2.0)
    
    return {
        'shmax': shmax,
        'shmin': shmin,
        'stress_ratio': stress_ratio
    }


def invert_stress_from_dif(
    dif_azimuth: Union[float, np.ndarray],
    depth: Union[float, np.ndarray],
    sv: Union[float, np.ndarray],
    pp: Union[float, np.ndarray],
    tensile_strength: float = 5.0,
    poisson: float = 0.25
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Invert stress magnitudes from drilling-induced fracture (DIF) observations.
    
    DIFs form perpendicular to Shmin direction.
    
    Args:
        dif_azimuth: DIF azimuth in degrees (from north)
        depth: Depth (m)
        sv: Vertical stress (MPa)
        pp: Pore pressure (MPa)
        tensile_strength: Tensile strength (MPa)
        poisson: Poisson's ratio
        
    Returns:
        Dictionary with estimated SHmax, Shmin, and stress ratio
    """
    dif_azimuth = np.asarray(dif_azimuth, dtype=float)
    depth = np.asarray(depth, dtype=float)
    sv = np.asarray(sv, dtype=float)
    pp = np.asarray(pp, dtype=float)
    
    sv_eff = sv - pp
    
    # DIF forms when wellbore pressure exceeds minimum stress + tensile strength
    # Shmin is approximately perpendicular to DIF azimuth
    # Simplified: estimate Shmin from vertical stress
    shmin = pp + poisson / (1 - poisson) * sv_eff
    
    # SHmax is constrained by DIF formation
    # DIF forms when: Pw > Shmin + T0
    # This gives lower bound on Shmin
    shmin_lower = pp + tensile_strength * 0.5
    
    shmin = np.maximum(shmin, shmin_lower)
    shmin = np.clip(shmin, pp, sv)
    
    # Estimate SHmax (typically 1.1-1.3x Shmin for normal faulting)
    shmax = shmin * 1.2
    shmax = np.clip(shmax, sv, sv * 1.5)
    
    stress_ratio = (shmax - pp) / sv_eff
    stress_ratio = np.clip(stress_ratio, 0.5, 2.0)
    
    return {
        'shmax': shmax,
        'shmin': shmin,
        'stress_ratio': stress_ratio,
        'shmin_azimuth': (dif_azimuth + 90) % 360
    }


def invert_stress_combined(
    depth: np.ndarray,
    sv: np.ndarray,
    pp: np.ndarray,
    breakout_data: Optional[Dict[str, np.ndarray]] = None,
    dif_data: Optional[Dict[str, np.ndarray]] = None,
    ucs: float = 50.0,
    tensile_strength: float = 5.0,
    poisson: float = 0.25
) -> Dict[str, np.ndarray]:
    """
    Invert stress magnitudes from combined breakout and DIF observations.
    
    Combines constraints from both failure types for more robust estimates.
    
    Args:
        breakout_data: Dictionary with 'width' and 'azimuth' keys
        dif_data: Dictionary with 'azimuth' key
        depth: Depth array (m)
        sv: Vertical stress array (MPa)
        pp: Pore pressure array (MPa)
        ucs: Unconfined compressive strength (MPa)
        tensile_strength: Tensile strength (MPa)
        poisson: Poisson's ratio
        
    Returns:
        Dictionary with estimated SHmax, Shmin, stress ratio, and confidence
    """
    depth = np.asarray(depth, dtype=float)
    sv = np.asarray(sv, dtype=float)
    pp = np.asarray(pp, dtype=float)
    
    results_breakout = None
    results_dif = None
    
    if breakout_data is not None:
        results_breakout = _invert_stress_analytical(
            breakout_data['width'],
            depth,
            sv,
            pp,
            ucs,
            poisson
        )
    
    if dif_data is not None:
        results_dif = invert_stress_from_dif(
            dif_data['azimuth'],
            depth,
            sv,
            pp,
            tensile_strength,
            poisson
        )
    
    # Combine results
    if results_breakout is not None and results_dif is not None:
        # Weighted average (breakout typically more reliable)
        weight_breakout = 0.7
        weight_dif = 0.3
        
        shmax = weight_breakout * results_breakout['shmax'] + weight_dif * results_dif['shmax']
        shmin = weight_breakout * results_breakout['shmin'] + weight_dif * results_dif['shmin']
        
        confidence = 'high'
    elif results_breakout is not None:
        shmax = results_breakout['shmax']
        shmin = results_breakout['shmin']
        confidence = 'medium'
    elif results_dif is not None:
        shmax = results_dif['shmax']
        shmin = results_dif['shmin']
        confidence = 'low'
    else:
        raise ValueError("At least one of breakout_data or dif_data must be provided")
    
    sv_eff = sv - pp
    stress_ratio = (shmax - pp) / sv_eff
    stress_ratio = np.clip(stress_ratio, 0.5, 2.0)
    
    return {
        'shmax': shmax,
        'shmin': shmin,
        'stress_ratio': stress_ratio,
        'confidence': confidence
    }

