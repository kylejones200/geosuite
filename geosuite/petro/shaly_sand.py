"""
Shaly sand water saturation models.

Provides alternatives to Archie's equation for shaly sand formations
where clay content affects resistivity measurements.
"""
import logging
from typing import Union, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_water_saturation_simandoux(
    phi: Union[np.ndarray, pd.Series],
    rt: Union[np.ndarray, pd.Series],
    rsh: Union[np.ndarray, pd.Series],
    vsh: Union[np.ndarray, pd.Series],
    rw: float = 0.05,
    m: float = 2.0,
    n: float = 2.0,
    a: float = 1.0
) -> np.ndarray:
    """
    Calculate water saturation using Simandoux equation for shaly sands.
    
    The Simandoux equation accounts for clay conductivity in shaly formations:
    Sw = sqrt((a * Rw) / (phi^m * (1/Rt - Vsh/Rsh)))
    
    Parameters
    ----------
    phi : np.ndarray or pd.Series
        Porosity (fraction)
    rt : np.ndarray or pd.Series
        True resistivity (ohm-m)
    rsh : np.ndarray or pd.Series
        Shale resistivity (ohm-m)
    vsh : np.ndarray or pd.Series
        Shale volume fraction (fraction)
    rw : float, default 0.05
        Formation water resistivity (ohm-m)
    m : float, default 2.0
        Cementation exponent
    n : float, default 2.0
        Saturation exponent (typically 2.0 for Simandoux)
    a : float, default 1.0
        Tortuosity factor
        
    Returns
    -------
    np.ndarray
        Water saturation (fraction)
        
    Raises
    ------
    ValueError
        If input arrays have mismatched lengths or invalid values
    """
    # Convert to numpy arrays
    phi = np.asarray(phi, dtype=float)
    rt = np.asarray(rt, dtype=float)
    rsh = np.asarray(rsh, dtype=float)
    vsh = np.asarray(vsh, dtype=float)
    
    # Validate inputs
    if len(phi) != len(rt) or len(phi) != len(rsh) or len(phi) != len(vsh):
        raise ValueError(
            "All input arrays must have the same length. "
            f"Got phi: {len(phi)}, rt: {len(rt)}, rsh: {len(rsh)}, vsh: {len(vsh)}"
        )
    
    if np.any(phi <= 0):
        logger.warning("Found non-positive porosity values, will result in NaN")
    if np.any(rt <= 0):
        logger.warning("Found non-positive resistivity values, will result in NaN")
    if np.any(rsh <= 0):
        logger.warning("Found non-positive shale resistivity values, will result in NaN")
    if np.any((vsh < 0) | (vsh > 1)):
        logger.warning("Found shale volume outside [0, 1], clipping to valid range")
        vsh = np.clip(vsh, 0, 1)
    
    logger.debug(
        f"Calculating water saturation using Simandoux equation "
        f"for {len(phi)} samples"
    )
    
    # Simandoux equation
    # Sw = sqrt((a * Rw) / (phi^m * (1/Rt - Vsh/Rsh)))
    with np.errstate(divide='ignore', invalid='ignore'):
        # Calculate clay conductivity term
        clay_conductivity = vsh / rsh
        
        # Calculate sand conductivity
        sand_conductivity = 1.0 / rt
        
        # Net conductivity (sand minus clay)
        net_conductivity = sand_conductivity - clay_conductivity
        
        # Avoid division by zero
        net_conductivity = np.where(net_conductivity <= 0, np.nan, net_conductivity)
        
        # Calculate water saturation
        numerator = a * rw
        denominator = (phi ** m) * net_conductivity
        sw = np.sqrt(numerator / denominator)
    
    # Clip to valid range
    sw = np.clip(sw, 0, 1)
    
    # Count NaN values
    nan_count = np.isnan(sw).sum()
    if nan_count > 0:
        logger.warning(f"Generated {nan_count} NaN values in Simandoux calculation")
    
    return sw


def calculate_water_saturation_indonesia(
    phi: Union[np.ndarray, pd.Series],
    rt: Union[np.ndarray, pd.Series],
    rsh: Union[np.ndarray, pd.Series],
    vsh: Union[np.ndarray, pd.Series],
    rw: float = 0.05,
    m: float = 2.0,
    n: float = 2.0,
    a: float = 1.0
) -> np.ndarray:
    """
    Calculate water saturation using Indonesia equation for shaly sands.
    
    The Indonesia equation is an improved version of Simandoux that better
    handles high shale volumes:
    Sw = [sqrt((a * Rw) / (phi^m * Rt)) + sqrt(Vsh * Rw / Rsh))]^(-2/n)
    
    Parameters
    ----------
    phi : np.ndarray or pd.Series
        Porosity (fraction)
    rt : np.ndarray or pd.Series
        True resistivity (ohm-m)
    rsh : np.ndarray or pd.Series
        Shale resistivity (ohm-m)
    vsh : np.ndarray or pd.Series
        Shale volume fraction (fraction)
    rw : float, default 0.05
        Formation water resistivity (ohm-m)
    m : float, default 2.0
        Cementation exponent
    n : float, default 2.0
        Saturation exponent
    a : float, default 1.0
        Tortuosity factor
        
    Returns
    -------
    np.ndarray
        Water saturation (fraction)
        
    Raises
    ------
    ValueError
        If input arrays have mismatched lengths or invalid values
    """
    # Convert to numpy arrays
    phi = np.asarray(phi, dtype=float)
    rt = np.asarray(rt, dtype=float)
    rsh = np.asarray(rsh, dtype=float)
    vsh = np.asarray(vsh, dtype=float)
    
    # Validate inputs
    lengths = [len(phi), len(rt), len(rsh), len(vsh)]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"All input arrays must have the same length. "
            f"Got lengths: phi={lengths[0]}, rt={lengths[1]}, rsh={lengths[2]}, vsh={lengths[3]}"
        )
    
    # Vectorized validation checks
    validation_checks = {
        'non-positive porosity': np.any(phi <= 0),
        'non-positive resistivity': np.any(rt <= 0),
        'non-positive shale resistivity': np.any(rsh <= 0),
        'shale volume out of range': np.any((vsh < 0) | (vsh > 1)),
    }
    
    for check_name, check_result in validation_checks.items():
        if check_result:
            logger.warning(f"Found {check_name}, may result in invalid values")
    
    vsh = np.clip(vsh, 0, 1)
    
    logger.debug(
        f"Calculating water saturation using Indonesia equation "
        f"for {len(phi)} samples"
    )
    
    # Indonesia equation
    # Sw = [sqrt((a * Rw) / (phi^m * Rt)) + sqrt(Vsh * Rw / Rsh))]^(-2/n)
    with np.errstate(divide='ignore', invalid='ignore'):
        # Archie term
        archie_term = np.sqrt((a * rw) / ((phi ** m) * rt))
        
        # Shale term
        shale_term = np.sqrt(vsh * rw / rsh)
        
        # Combined term
        combined = archie_term + shale_term
        
        # Avoid division by zero
        combined = np.where(combined <= 0, np.nan, combined)
        
        # Calculate water saturation
        sw = np.power(combined, -2.0 / n)
    
    # Clip to valid range
    sw = np.clip(sw, 0, 1)
    
    # Count NaN values
    nan_count = np.isnan(sw).sum()
    if nan_count > 0:
        logger.warning(f"Generated {nan_count} NaN values in Indonesia calculation")
    
    return sw


def calculate_water_saturation_waxman_smits(
    phi: Union[np.ndarray, pd.Series],
    rt: Union[np.ndarray, pd.Series],
    cec: Union[np.ndarray, pd.Series],
    rw: float = 0.05,
    m: float = 2.0,
    n: float = 2.0,
    a: float = 1.0,
    b: Optional[float] = None,
    temperature: float = 25.0
) -> np.ndarray:
    """
    Calculate water saturation using Waxman-Smits equation for shaly sands.
    
    The Waxman-Smits model accounts for clay cation exchange capacity (CEC)
    and is more physically-based than Simandoux:
    Sw = [sqrt((a * Rw) / (phi^m * Rt)) + B * Qv * Rw]^(-2/n)
    
    where Qv is the cation exchange capacity per unit pore volume.
    
    Parameters
    ----------
    phi : np.ndarray or pd.Series
        Porosity (fraction)
    rt : np.ndarray or pd.Series
        True resistivity (ohm-m)
    cec : np.ndarray or pd.Series
        Cation exchange capacity (meq/100g)
    rw : float, default 0.05
        Formation water resistivity (ohm-m)
    m : float, default 2.0
        Cementation exponent
    n : float, default 2.0
        Saturation exponent
    a : float, default 1.0
        Tortuosity factor
    b : float, optional
        Equivalent counterion conductance (mho/m per meq/ml). 
        If None, calculated from temperature.
    temperature : float, default 25.0
        Formation temperature (°C) for B calculation if b is None
        
    Returns
    -------
    np.ndarray
        Water saturation (fraction)
        
    Raises
    ------
    ValueError
        If input arrays have mismatched lengths or invalid values
    """
    # Convert to numpy arrays
    phi = np.asarray(phi, dtype=float)
    rt = np.asarray(rt, dtype=float)
    cec = np.asarray(cec, dtype=float)
    
    # Validate inputs
    lengths = [len(phi), len(rt), len(cec)]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"All input arrays must have the same length. "
            f"Got lengths: phi={lengths[0]}, rt={lengths[1]}, cec={lengths[2]}"
        )
    
    # Vectorized validation checks
    validation_checks = {
        'non-positive porosity': np.any(phi <= 0),
        'non-positive resistivity': np.any(rt <= 0),
        'negative CEC': np.any(cec < 0),
    }
    
    for check_name, check_result in validation_checks.items():
        if check_result:
            logger.warning(f"Found {check_name}, may result in invalid values")
    
    logger.debug(
        f"Calculating water saturation using Waxman-Smits equation "
        f"for {len(phi)} samples"
    )
    
    # Calculate B (equivalent counterion conductance) if not provided
    if b is None:
        # B = 4.6 * (1 - 0.6 * exp(-0.77 / Rw)) at 25°C
        # Temperature correction: B(T) = B(25) * (1 + 0.02 * (T - 25))
        b_25 = 4.6 * (1 - 0.6 * np.exp(-0.77 / rw))
        b = b_25 * (1 + 0.02 * (temperature - 25.0))
        logger.debug(f"Calculated B = {b:.4f} mho/m per meq/ml at {temperature}°C")
    
    # Convert CEC to Qv (meq/ml pore volume)
    # Qv = CEC * (1 - phi) * rho_grain / (phi * 100)
    # Simplified: assume rho_grain = 2.65 g/cc
    rho_grain = 2.65
    qv = cec * (1 - phi) * rho_grain / (phi * 100.0)
    
    # Waxman-Smits equation
    # Sw = [sqrt((a * Rw) / (phi^m * Rt)) + B * Qv * Rw]^(-2/n)
    with np.errstate(divide='ignore', invalid='ignore'):
        # Archie term
        archie_term = np.sqrt((a * rw) / ((phi ** m) * rt))
        
        # Clay conductivity term
        clay_term = b * qv * rw
        
        # Combined term
        combined = archie_term + clay_term
        
        # Avoid division by zero
        combined = np.where(combined <= 0, np.nan, combined)
        
        # Calculate water saturation
        sw = np.power(combined, -2.0 / n)
    
    # Clip to valid range
    sw = np.clip(sw, 0, 1)
    
    # Count NaN values
    nan_count = np.isnan(sw).sum()
    if nan_count > 0:
        logger.warning(f"Generated {nan_count} NaN values in Waxman-Smits calculation")
    
    return sw
