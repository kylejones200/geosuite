from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd

G = 9.80665  # m/s^2


@dataclass
class GeomechParams:
    """Parameters for geomechanical calculations"""
    phi_initial: float = 0.35  # Initial porosity (fraction)
    beta: float = 0.03  # Compaction coefficient (1/MPa)
    mu: float = 0.69  # Friction coefficient (dimensionless)
    ucs: float = 30.0  # Unconfined Compressive Strength (MPa) 
    tensile_strength: float = 0.0  # Rock tensile strength (MPa)
    delta_p: float = 0.0  # Excess pressure from drilling (MPa)


def sv_from_density(depth_m: pd.Series, rhob_gcc: pd.Series) -> pd.Series:
    """
    Compute vertical stress Sv by integrating density over depth.
    Inputs:
      depth_m: depth array in meters (monotonic increasing)
      rhob_gcc: bulk density in g/cc
    Returns:
      Sv in MPa (approximate, using simple trapezoidal integration)
    """
    z = depth_m.to_numpy()
    rho = rhob_gcc.to_numpy() * 1000.0  # g/cc -> kg/m^3
    if len(z) < 2:
        return pd.Series(np.zeros_like(z, dtype=float))
    dz = np.diff(z)
    # mid-point average density for each interval
    rho_mid = 0.5 * (rho[1:] + rho[:-1])
    # incremental stress Pa
    dsv = rho_mid * G * dz  # Pa
    sv = np.concatenate([[0.0], np.cumsum(dsv)])  # Pa at each depth sample
    sv_mpa = sv / 1e6
    return pd.Series(sv_mpa, index=depth_m.index)


def hydrostatic_pressure(depth_m: pd.Series, grad_mpa_per_km: float = 9.81) -> pd.Series:
    """
    Simple hydrostatic pressure gradient model.
    Default gradient 9.81 MPa/km (freshwater). Returns MPa.
    """
    return (depth_m.to_numpy() / 1000.0) * grad_mpa_per_km


def pore_pressure_eaton(sv: pd.Series, porosity: pd.Series, params: GeomechParams) -> pd.Series:
    """
    Compute pore pressure using Eaton's method based on porosity.
    Pp = Sv - (ln(φ / φ_initial) / -beta)
    
    Inputs:
      sv: vertical stress in MPa
      porosity: porosity as fraction (0-1)
      params: GeomechParams with phi_initial and beta
    Returns:
      Pore pressure in MPa
    """
    porosity = np.clip(porosity, 1e-6, 0.99)  # Avoid log(0) issues
    phi_ratio = porosity / params.phi_initial
    phi_ratio = np.clip(phi_ratio, 1e-6, None)  # Avoid log of negative or zero
    pp = sv - (np.log(phi_ratio) / (-params.beta))
    return pd.Series(pp, index=sv.index)


def effective_stress(sv: pd.Series, pp: pd.Series) -> pd.Series:
    """
    Compute effective stress: σ_eff = Sv - Pp
    
    Inputs:
      sv: vertical stress in MPa
      pp: pore pressure in MPa
    Returns:
      Effective stress in MPa
    """
    return sv - pp


def overpressure(pp: pd.Series, p_hydrostatic: pd.Series) -> pd.Series:
    """
    Compute overpressure: Po = Pp - P_hydrostatic
    
    Inputs:
      pp: pore pressure in MPa
      p_hydrostatic: hydrostatic pressure in MPa
    Returns:
      Overpressure in MPa (positive = overpressured)
    """
    return pp - p_hydrostatic


def friction_coefficient_ratio(mu: float) -> float:
    """
    Calculate the friction coefficient ratio for faulting calculations.
    fμ = ((μ²+1)^0.5 + μ)²
    
    Input:
      mu: friction coefficient
    Returns:
      Friction coefficient ratio
    """
    return ((mu**2 + 1)**0.5 + mu)**2


def shmax_bounds(sv: float, shmin: float, pp: float, params: GeomechParams) -> Dict[str, float]:
    """
    Calculate SHmax bounds for different faulting regimes.
    
    Inputs:
      sv: vertical stress (MPa)
      shmin: minimum horizontal stress (MPa) 
      pp: pore pressure (MPa)
      params: GeomechParams with friction coefficient
    Returns:
      Dictionary with SHmax bounds for each faulting regime
    """
    f_mu = friction_coefficient_ratio(params.mu)
    
    # Strike-slip faulting: SHmax = fμ * (Shmin - Pp) + Pp
    shmax_ss = f_mu * (shmin - pp) + pp
    
    # Reverse faulting: SHmax = fμ * (Sv - Pp) + Pp  
    shmax_rev = f_mu * (sv - pp) + pp
    
    # Normal faulting: Shmin = ((Sv - Pp) / fμ) + Pp (for reference)
    shmin_nf = ((sv - pp) / f_mu) + pp
    
    # Tensile fracture constraint: SHmax_tf = 3*Shmin - 2*Pp - ΔP
    shmax_tf = 3*shmin - 2*pp - params.delta_p
    
    return {
        'shmax_strike_slip': shmax_ss,
        'shmax_reverse': shmax_rev,
        'shmin_normal': shmin_nf,
        'shmax_tensile_fracture': shmax_tf,
        'friction_ratio': f_mu
    }


def stress_polygon_points(sv: float, pp: float, params: GeomechParams) -> List[Tuple[float, float]]:
    """
    Calculate the stress polygon corner points for a given depth.
    
    Inputs:
      sv: vertical stress (MPa)
      pp: pore pressure (MPa)
      params: GeomechParams with friction coefficient
    Returns:
      List of (Shmin, SHmax) tuples for stress polygon corners
    """
    f_mu = friction_coefficient_ratio(params.mu)
    
    # Calculate limiting stress values
    s1_max = f_mu * (sv - pp) + pp  # Maximum S1 when Sv = S3
    s3_min = ((sv - pp) / f_mu) + pp  # Minimum S3 when Sv = S1
    
    # Stress polygon corners (moving from bottom-left to top-right)
    corners = [
        (s3_min, s3_min),    # Corner 1: Normal faulting lower bound
        (s3_min, sv),        # Corner 2: Transition to strike-slip
        (sv, sv),            # Corner 3: Strike-slip center
        (sv, s1_max),        # Corner 4: Transition to reverse
        (s1_max, s1_max)     # Corner 5: Reverse faulting upper bound
    ]
    
    return corners


def porosity_from_density(rhob_gcc: pd.Series, matrix_density: float = 2.65, 
                         fluid_density: float = 1.0) -> pd.Series:
    """
    Calculate porosity from bulk density using the density-porosity relationship.
    φ = (ρ_matrix - ρ_bulk) / (ρ_matrix - ρ_fluid)
    
    Inputs:
      rhob_gcc: bulk density in g/cc
      matrix_density: matrix density in g/cc (default 2.65 for quartz sandstone)
      fluid_density: fluid density in g/cc (default 1.0 for water)
    Returns:
      Porosity as fraction (0-1)
    """
    porosity = (matrix_density - rhob_gcc) / (matrix_density - fluid_density)
    return np.clip(porosity, 0.0, 0.99)  # Ensure physically reasonable values


def mud_weight_equivalent(pressure_mpa: pd.Series, depth_m: pd.Series) -> pd.Series:
    """
    Convert pressure to equivalent mud weight (EMW).
    EMW (g/cc) = Pressure (MPa) / (Depth (m) * 0.00981)
    
    Inputs:
      pressure_mpa: pressure in MPa
      depth_m: depth in meters
    Returns:
      Equivalent mud weight in g/cc
    """
    # Avoid division by zero
    depth_nonzero = np.where(depth_m == 0, 1e-6, depth_m)
    emw = pressure_mpa / (depth_nonzero * 0.00981)
    return pd.Series(emw, index=pressure_mpa.index)


# ============================================================================
# WELLBORE STABILITY ANALYSIS
# ============================================================================

@dataclass
class WellboreStabilityResult:
    """Results from wellbore stability analysis"""
    breakout_pressure: float  # Minimum mud pressure to prevent breakout (MPa)
    fracture_pressure: float  # Maximum mud pressure before tensile failure (MPa)
    breakout_width: float  # Angular width of breakout (degrees)
    breakout_azimuth: float  # Azimuth of breakout (degrees from SHmax)
    safe_mud_weight_min: float  # Minimum safe mud weight (g/cc)
    safe_mud_weight_max: float  # Maximum safe mud weight (g/cc)
    stability_margin: float  # Stability margin (fraction)


def wellbore_stress_concentration(theta: np.ndarray, sv: float, shmax: float, 
                                shmin: float, pp: float, pw: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate stress concentrations around a wellbore using Kirsch equations.
    
    Inputs:
      theta: azimuth angles around wellbore (radians)
      sv: vertical stress (MPa)
      shmax: maximum horizontal stress (MPa) 
      shmin: minimum horizontal stress (MPa)
      pp: pore pressure (MPa)
      pw: wellbore pressure (mud pressure, MPa)
    Returns:
      Tuple of (tangential_stress, radial_stress) in effective stress (MPa)
    """
    # Effective stresses
    sv_eff = sv - pp
    shmax_eff = shmax - pp
    shmin_eff = shmin - pp
    pw_eff = pw - pp
    
    # Average horizontal stress
    sh_avg = (shmax_eff + shmin_eff) / 2
    
    # Differential horizontal stress
    dsh = (shmax_eff - shmin_eff) / 2
    
    # Tangential stress (effective)
    sigma_theta = sh_avg + 2*dsh*np.cos(2*theta) + pw_eff
    
    # Radial stress (effective) 
    sigma_r = pw_eff
    
    return sigma_theta, sigma_r


def breakout_analysis(sv: float, shmax: float, shmin: float, pp: float, 
                     params: GeomechParams, depth: float = 1000.0) -> WellboreStabilityResult:
    """
    Perform comprehensive wellbore stability analysis.
    
    Inputs:
      sv: vertical stress (MPa)
      shmax: maximum horizontal stress (MPa)
      shmin: minimum horizontal stress (MPa) 
      pp: pore pressure (MPa)
      params: GeomechParams with UCS, tensile strength, etc.
      depth: depth for mud weight calculations (m)
    Returns:
      WellboreStabilityResult with all stability parameters
    """
    
    # Breakout analysis - find minimum mud pressure to prevent failure
    # Using Mohr-Coulomb failure criterion at wellbore wall
    
    # Critical angle for maximum tangential stress (90° from SHmax)
    theta_critical = np.pi/2  # 90 degrees
    
    # For breakout prevention, find minimum pw such that failure doesn't occur
    # Mohr-Coulomb: τ = c + σn * tan(φ), where φ = friction angle
    # For UCS test: UCS = 2*c*cos(φ)/(1-sin(φ))
    
    # Convert friction coefficient to friction angle
    phi = np.arctan(params.mu)  # friction angle
    cohesion = params.ucs * (1 - np.sin(phi)) / (2 * np.cos(phi))
    
    # Effective stresses
    sv_eff = sv - pp
    shmax_eff = shmax - pp  
    shmin_eff = shmin - pp
    
    # Average and differential horizontal stress
    sh_avg_eff = (shmax_eff + shmin_eff) / 2
    dsh_eff = (shmax_eff - shmin_eff) / 2
    
    # At breakout location (θ = π/2), tangential stress is maximum
    # σ_θ = sh_avg - 2*dsh + pw_eff (at θ = π/2)
    # For stability: σ_θ_eff ≤ UCS_eff = UCS (since UCS is already effective stress)
    
    # Minimum effective wellbore pressure to prevent breakout
    pw_eff_min = params.ucs - sh_avg_eff + 2*dsh_eff
    breakout_pressure = pw_eff_min + pp
    
    # Tensile fracture analysis - maximum mud pressure
    # Tensile failure occurs when tangential stress becomes tensile
    # At minimum stress location (θ = 0), σ_θ = sh_avg + 2*dsh + pw_eff  
    # For tensile failure: σ_θ ≤ -T0 (tensile strength is negative)
    
    # Maximum effective wellbore pressure before tensile failure
    pw_eff_max = -params.tensile_strength - sh_avg_eff - 2*dsh_eff
    fracture_pressure = pw_eff_max + pp
    
    # Alternative fracture pressure using LOT relationship
    # From notebook: SHmax_tf = 3*Shmin - 2*Pp - ΔP
    fracture_pressure_alt = 3*shmin - 2*pp - params.delta_p
    fracture_pressure = min(fracture_pressure, fracture_pressure_alt)
    
    # Breakout width calculation
    # Simplified model: breakout occurs where tangential stress exceeds UCS
    theta_range = np.linspace(0, 2*np.pi, 360)
    sigma_theta, _ = wellbore_stress_concentration(theta_range, sv, shmax, shmin, pp, breakout_pressure)
    
    # Find angles where failure occurs (effective tangential stress > UCS)
    failed_indices = sigma_theta > params.ucs
    if np.any(failed_indices):
        failed_angles = theta_range[failed_indices]
        # Breakout typically occurs in two symmetric lobes
        if len(failed_angles) > 0:
            breakout_width = np.degrees(np.ptp(failed_angles[failed_angles < np.pi]))  # Width of one lobe
        else:
            breakout_width = 0.0
    else:
        breakout_width = 0.0
    
    # Breakout azimuth (perpendicular to SHmax direction)
    breakout_azimuth = 90.0  # degrees from SHmax
    
    # Convert to mud weights
    safe_mud_weight_min = breakout_pressure / (depth * 0.00981) if depth > 0 else 0
    safe_mud_weight_max = fracture_pressure / (depth * 0.00981) if depth > 0 else 0
    
    # Stability margin
    pressure_window = fracture_pressure - breakout_pressure
    stability_margin = pressure_window / max(breakout_pressure, 1e-6)
    
    return WellboreStabilityResult(
        breakout_pressure=breakout_pressure,
        fracture_pressure=fracture_pressure,
        breakout_width=max(0, breakout_width),
        breakout_azimuth=breakout_azimuth,
        safe_mud_weight_min=max(0, safe_mud_weight_min),
        safe_mud_weight_max=max(0, safe_mud_weight_max),
        stability_margin=stability_margin
    )


def safe_mud_weight_window(df: pd.DataFrame, params: GeomechParams, 
                          shmin_estimate: float = None) -> pd.DataFrame:
    """
    Calculate safe mud weight window for entire well trajectory.
    
    Inputs:
      df: DataFrame with depth, RHOB columns
      params: GeomechParams 
      shmin_estimate: Estimate of Shmin (MPa), if None uses stress polygon
    Returns:
      DataFrame with safe mud weight analysis vs depth
    """
    results = []
    
    # Calculate porosity and stresses
    porosity = porosity_from_density(df["RHOB"], matrix_density=2.65)
    sv = sv_from_density(df["depth_m"], df["RHOB"])
    ph = hydrostatic_pressure(df["depth_m"])
    pp = pore_pressure_eaton(sv, porosity, params)
    
    for i, row in df.iterrows():
        depth = row['depth_m']
        sv_val = sv.iloc[i] if hasattr(sv, 'iloc') else sv[i]
        pp_val = pp.iloc[i] if hasattr(pp, 'iloc') else pp[i]
        
        # Estimate horizontal stresses if not provided
        if shmin_estimate is None:
            # Use simple relationship: Shmin ≈ 0.7 * Sv (typical for normal faulting)
            shmin_val = 0.7 * sv_val
        else:
            shmin_val = shmin_estimate
            
        # Estimate SHmax using stress polygon bounds
        bounds = shmax_bounds(sv_val, shmin_val, pp_val, params)
        shmax_val = min(bounds['shmax_strike_slip'], bounds['shmax_tensile_fracture'])
        
        # Perform wellbore stability analysis
        stability = breakout_analysis(sv_val, shmax_val, shmin_val, pp_val, params, depth)
        
        results.append({
            'depth_m': depth,
            'sv_mpa': sv_val,
            'shmax_mpa': shmax_val, 
            'shmin_mpa': shmin_val,
            'pp_mpa': pp_val,
            'breakout_pressure_mpa': stability.breakout_pressure,
            'fracture_pressure_mpa': stability.fracture_pressure,
            'mud_weight_min_gcc': stability.safe_mud_weight_min,
            'mud_weight_max_gcc': stability.safe_mud_weight_max,
            'breakout_width_deg': stability.breakout_width,
            'stability_margin': stability.stability_margin
        })
    
    return pd.DataFrame(results)


def wellbore_stress_plot_data(sv: float, shmax: float, shmin: float, 
                             pp: float, pw: float, num_points: int = 360) -> pd.DataFrame:
    """
    Generate data for wellbore stress concentration plot.
    
    Inputs:
      sv, shmax, shmin, pp, pw: stress values (MPa)
      num_points: number of points around wellbore circumference
    Returns:
      DataFrame with theta, sigma_theta, sigma_r columns
    """
    theta = np.linspace(0, 2*np.pi, num_points)
    sigma_theta, sigma_r = wellbore_stress_concentration(theta, sv, shmax, shmin, pp, pw)
    
    return pd.DataFrame({
        'theta_deg': np.degrees(theta),
        'theta_rad': theta,
        'sigma_theta_eff': sigma_theta,
        'sigma_r_eff': sigma_r,
        'sigma_theta_total': sigma_theta + pp,
        'sigma_r_total': sigma_r + pp
    })


def drilling_margin_analysis(breakout_pressure: float, fracture_pressure: float,
                           current_mud_weight: float, depth: float) -> Dict[str, float]:
    """
    Analyze drilling safety margins for current mud weight.
    
    Inputs:
      breakout_pressure: minimum pressure to prevent breakout (MPa)
      fracture_pressure: maximum pressure before fracture (MPa)
      current_mud_weight: current mud weight (g/cc)
      depth: depth (m)
    Returns:
      Dictionary with margin analysis
    """
    current_pressure = current_mud_weight * depth * 0.00981
    
    # Calculate margins
    breakout_margin = (current_pressure - breakout_pressure) / breakout_pressure * 100
    fracture_margin = (fracture_pressure - current_pressure) / fracture_pressure * 100
    
    # Safety status
    if current_pressure < breakout_pressure:
        status = "BREAKOUT RISK"
    elif current_pressure > fracture_pressure:
        status = "FRACTURE RISK" 
    else:
        status = "SAFE"
    
    return {
        'current_pressure_mpa': current_pressure,
        'breakout_margin_pct': breakout_margin,
        'fracture_margin_pct': fracture_margin,
        'pressure_window_mpa': fracture_pressure - breakout_pressure,
        'safety_status': status,
        'recommended_mud_weight_gcc': (breakout_pressure + fracture_pressure) / 2 / (depth * 0.00981)
    }


# ============================================================================
# MULTI-WELL FIELD ANALYSIS
# ============================================================================

@dataclass
class FieldOptimizationResult:
    """Results from field-wide drilling optimization"""
    optimal_mud_weights: Dict[str, float]  # Well name -> optimal mud weight
    cost_savings: float  # Total cost savings (currency units)
    time_savings: float  # Total time savings (days)
    risk_reduction: float  # Risk reduction factor
    success_probability: float  # Field success probability
    recommendations: List[str]  # Optimization recommendations


def process_field_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Process multi-well field data into separate well datasets.
    
    Input:
      df: Combined field data with well_name column
    Returns:
      Dictionary of {well_name: well_dataframe}
    """
    wells = {}
    for well_name in df['well_name'].unique():
        well_df = df[df['well_name'] == well_name].copy()
        well_df = well_df.reset_index(drop=True)
        wells[well_name] = well_df
    return wells


def calculate_field_statistics(field_data: pd.DataFrame) -> Dict[str, any]:
    """
    Calculate field-wide drilling statistics and performance metrics.
    
    Input:
      field_data: Multi-well field data
    Returns:
      Dictionary with field statistics
    """
    # Group by well for well-level statistics
    well_stats = field_data.groupby('well_name').agg({
        'mud_weight_used': ['mean', 'std'],
        'cost_per_meter': 'first',
        'days_to_drill': 'first',
        'drilling_status': lambda x: (x == 'Success').mean(),  # Success rate
        'depth_m': ['min', 'max']
    }).round(3)
    
    # Flatten column names
    well_stats.columns = ['_'.join(col).strip() for col in well_stats.columns.values]
    
    # Overall field statistics
    total_wells = field_data['well_name'].nunique()
    overall_success_rate = (field_data.groupby('well_name')['drilling_status']
                           .apply(lambda x: (x == 'Success').all()).mean())
    
    avg_cost_per_meter = field_data.groupby('well_name')['cost_per_meter'].first().mean()
    avg_drilling_days = field_data.groupby('well_name')['days_to_drill'].first().mean()
    
    # Problem analysis
    problems = field_data[field_data['drilling_status'] != 'Success']
    problem_types = problems['drilling_status'].value_counts().to_dict()
    
    # Formation analysis
    formation_performance = field_data.groupby('formation').agg({
        'drilling_status': lambda x: (x == 'Success').mean(),
        'mud_weight_used': 'mean',
        'GR': 'mean',
        'RHOB': 'mean',
        'RT': 'mean'
    }).round(3)
    
    return {
        'total_wells': total_wells,
        'overall_success_rate': overall_success_rate,
        'avg_cost_per_meter': avg_cost_per_meter,
        'avg_drilling_days': avg_drilling_days,
        'well_statistics': well_stats,
        'problem_types': problem_types,
        'formation_performance': formation_performance
    }


def optimize_field_mud_weights(field_data: pd.DataFrame, params: GeomechParams) -> FieldOptimizationResult:
    """
    Optimize mud weights across the field for minimum cost and risk.
    
    Input:
      field_data: Multi-well field data
      params: Geomechanical parameters
    Returns:
      FieldOptimizationResult with optimization recommendations
    """
    wells = process_field_data(field_data)
    optimal_weights = {}
    total_cost_savings = 0
    total_time_savings = 0
    recommendations = []
    
    for well_name, well_df in wells.items():
        # Calculate stability for current well
        stability_df = safe_mud_weight_window(well_df[['depth_m', 'RHOB']], params)
        
        # Get current mud weight usage
        current_mw = well_df['mud_weight_used'].mean()
        current_status = well_df['drilling_status'].iloc[-1]  # Status at TD
        
        # Calculate optimal mud weight (middle of safe window)
        avg_min_mw = stability_df['mud_weight_min_gcc'].mean()
        avg_max_mw = stability_df['mud_weight_max_gcc'].mean()
        optimal_mw = (avg_min_mw + avg_max_mw) / 2
        
        # Ensure optimal MW is within reasonable bounds
        optimal_mw = max(0.9, min(2.0, optimal_mw))
        optimal_weights[well_name] = optimal_mw
        
        # Estimate cost and time savings
        current_cost = well_df['cost_per_meter'].iloc[0]
        current_days = well_df['days_to_drill'].iloc[0]
        
        # Cost reduction from fewer drilling problems
        if current_status != 'Success':
            # Problem well - significant savings possible
            cost_reduction_pct = 0.15  # 15% cost reduction
            time_reduction_days = 3    # 3 days time savings
        elif abs(current_mw - optimal_mw) > 0.1:
            # Sub-optimal but successful well - moderate savings
            cost_reduction_pct = 0.08  # 8% cost reduction
            time_reduction_days = 1    # 1 day time savings
        else:
            # Already optimal well - minimal savings
            cost_reduction_pct = 0.02  # 2% cost reduction
            time_reduction_days = 0.5  # 0.5 day time savings
        
        well_cost_savings = current_cost * (well_df['depth_m'].max() - well_df['depth_m'].min()) * cost_reduction_pct
        total_cost_savings += well_cost_savings
        total_time_savings += time_reduction_days
        
        # Generate recommendations
        if current_status != 'Success':
            recommendations.append(f"{well_name}: Change MW from {current_mw:.2f} to {optimal_mw:.2f} g/cc to prevent {current_status}")
        elif abs(current_mw - optimal_mw) > 0.1:
            recommendations.append(f"{well_name}: Optimize MW from {current_mw:.2f} to {optimal_mw:.2f} g/cc for efficiency")
    
    # Calculate overall risk reduction and success probability
    current_success_rate = (field_data.groupby('well_name')['drilling_status']
                           .apply(lambda x: (x == 'Success').all()).mean())
    estimated_success_rate = min(0.95, current_success_rate + 0.1)  # Conservative improvement
    risk_reduction = (estimated_success_rate - current_success_rate) / max(1 - current_success_rate, 0.01)
    
    return FieldOptimizationResult(
        optimal_mud_weights=optimal_weights,
        cost_savings=total_cost_savings,
        time_savings=total_time_savings,
        risk_reduction=risk_reduction,
        success_probability=estimated_success_rate,
        recommendations=recommendations
    )


def field_correlation_analysis(field_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze correlations between drilling parameters and performance across the field.
    
    Input:
      field_data: Multi-well field data
    Returns:
      Dictionary with correlation analysis results
    """
    # Prepare data for correlation analysis
    well_summary = field_data.groupby('well_name').agg({
        'mud_weight_used': 'mean',
        'cost_per_meter': 'first',
        'days_to_drill': 'first',
        'drilling_status': lambda x: (x == 'Success').all(),
        'GR': 'mean',
        'RHOB': 'mean',
        'NPHI': 'mean', 
        'RT': 'mean',
        'x_coord': 'first',
        'y_coord': 'first'
    })
    
    # Convert success boolean to numeric
    well_summary['success_rate'] = well_summary['drilling_status'].astype(int)
    well_summary = well_summary.drop('drilling_status', axis=1)
    
    # Calculate correlations
    correlations = well_summary.corr()
    
    # Identify key relationships
    mud_weight_correlations = correlations['mud_weight_used'].abs().sort_values(ascending=False)
    cost_correlations = correlations['cost_per_meter'].abs().sort_values(ascending=False)
    success_correlations = correlations['success_rate'].abs().sort_values(ascending=False)
    
    # Spatial analysis
    spatial_data = well_summary[['x_coord', 'y_coord', 'mud_weight_used', 'cost_per_meter', 'success_rate']].copy()
    
    return {
        'correlation_matrix': correlations,
        'mud_weight_factors': mud_weight_correlations,
        'cost_factors': cost_correlations,
        'success_factors': success_correlations,
        'spatial_data': spatial_data,
        'well_summary': well_summary
    }


def identify_drilling_trends(field_data: pd.DataFrame) -> Dict[str, any]:
    """
    Identify drilling trends and patterns across the field.
    
    Input:
      field_data: Multi-well field data
    Returns:
      Dictionary with trend analysis
    """
    trends = {}
    
    # Formation-specific trends
    formation_trends = field_data.groupby('formation').agg({
        'mud_weight_used': ['mean', 'std', 'min', 'max'],
        'drilling_status': lambda x: (x == 'Success').mean(),
        'GR': 'mean',
        'RHOB': 'mean',
        'RT': 'mean'
    })
    trends['formation_trends'] = formation_trends
    
    # Depth trends
    depth_bins = pd.cut(field_data['depth_m'], bins=5, labels=['Shallow', 'Mod-Shallow', 'Mid', 'Mod-Deep', 'Deep'])
    depth_trends = field_data.groupby(depth_bins).agg({
        'mud_weight_used': 'mean',
        'drilling_status': lambda x: (x == 'Success').mean(),
        'cost_per_meter': 'mean'
    })
    trends['depth_trends'] = depth_trends
    
    # Geographic trends (if coordinates available)
    if 'x_coord' in field_data.columns:
        # Simple geographic binning
        x_bins = pd.cut(field_data['x_coord'], bins=3, labels=['West', 'Central', 'East'])
        y_bins = pd.cut(field_data['y_coord'], bins=3, labels=['South', 'Central', 'North'])
        
        geographic_trends = field_data.groupby([x_bins, y_bins]).agg({
            'mud_weight_used': 'mean',
            'drilling_status': lambda x: (x == 'Success').mean(),
            'cost_per_meter': 'mean'
        })
        trends['geographic_trends'] = geographic_trends
    
    # Problem analysis
    problem_data = field_data[field_data['drilling_status'] != 'Success']
    if len(problem_data) > 0:
        problem_trends = {
            'problem_depths': problem_data.groupby('depth_m')['drilling_status'].count(),
            'problem_formations': problem_data.groupby('formation')['drilling_status'].count(),
            'problem_mud_weights': problem_data['mud_weight_used'].describe()
        }
        trends['problem_analysis'] = problem_trends
    
    return trends


def generate_field_recommendations(field_data: pd.DataFrame, optimization: FieldOptimizationResult, 
                                 trends: Dict[str, any]) -> List[str]:
    """
    Generate comprehensive field development recommendations.
    
    Input:
      field_data: Multi-well field data
      optimization: Optimization results
      trends: Trend analysis results
    Returns:
      List of actionable recommendations
    """
    recommendations = []
    
    # Add optimization-specific recommendations
    recommendations.extend(optimization.recommendations)
    
    # Formation-specific recommendations
    if 'formation_trends' in trends:
        formation_trends = trends['formation_trends']
        for formation in formation_trends.index:
            success_rate = formation_trends.loc[formation, ('drilling_status', '<lambda>')]
            avg_mud_weight = formation_trends.loc[formation, ('mud_weight_used', 'mean')]
            
            if success_rate < 0.8:  # Less than 80% success
                recommendations.append(f"Formation {formation}: Low success rate ({success_rate:.1%}). "
                                     f"Consider revised drilling strategy or mud weight adjustment from {avg_mud_weight:.2f} g/cc")
    
    # Cost optimization recommendations
    stats = calculate_field_statistics(field_data)
    high_cost_wells = [well for well, cost in stats['well_statistics']['cost_per_meter_first'].items() 
                      if cost > stats['avg_cost_per_meter'] * 1.2]
    
    if high_cost_wells:
        recommendations.append(f"High-cost wells ({', '.join(high_cost_wells[:3])}{'...' if len(high_cost_wells) > 3 else ''}): "
                              f"Review drilling practices - {len(high_cost_wells)} wells above field average")
    
    # Success rate recommendations
    if stats['overall_success_rate'] < 0.9:
        recommendations.append(f"Field success rate ({stats['overall_success_rate']:.1%}) below target. "
                              f"Focus on problem prevention in formations with issues")
    
    # Spatial recommendations (if geographic data available)
    if 'geographic_trends' in trends:
        geo_trends = trends['geographic_trends']
        if not geo_trends.empty:
            problem_areas = geo_trends[geo_trends[('drilling_status', '<lambda>')] < 0.8]
            if not problem_areas.empty:
                recommendations.append(f"Geographic risk areas identified: {len(problem_areas)} sectors "
                                     f"with elevated drilling challenges")
    
    return recommendations


def economic_analysis(field_data: pd.DataFrame, optimization: FieldOptimizationResult, 
                     economic_params: Dict[str, float] = None) -> Dict[str, float]:
    """
    Perform economic analysis of field development and optimization benefits.
    
    Input:
      field_data: Multi-well field data
      optimization: Optimization results
      economic_params: Economic parameters (oil_price, npv_rate, etc.)
    Returns:
      Dictionary with economic analysis results
    """
    if economic_params is None:
        economic_params = {
            'oil_price_per_barrel': 70.0,  # $/barrel
            'production_rate_per_well': 100.0,  # barrels/day
            'well_life_years': 10.0,
            'npv_discount_rate': 0.10,  # 10%
            'drilling_cost_base': 2000000,  # $2M base cost per well
        }
    
    # Calculate NPV impact of optimization
    wells = field_data['well_name'].nunique()
    
    # Revenue impact from improved success rate
    current_success_rate = (field_data.groupby('well_name')['drilling_status']
                           .apply(lambda x: (x == 'Success').all()).mean())
    
    additional_successful_wells = wells * (optimization.success_probability - current_success_rate)
    
    # Revenue per successful well
    annual_revenue_per_well = (economic_params['production_rate_per_well'] * 365 * 
                              economic_params['oil_price_per_barrel'])
    
    # NPV calculation
    npv_per_well = 0
    for year in range(1, int(economic_params['well_life_years']) + 1):
        npv_per_well += annual_revenue_per_well / ((1 + economic_params['npv_discount_rate']) ** year)
    
    # Total economic impact
    additional_npv = additional_successful_wells * npv_per_well
    
    # Cost savings from optimization
    direct_cost_savings = optimization.cost_savings
    
    # Time value of money for faster drilling
    time_savings_value = (optimization.time_savings * wells * 
                         economic_params['production_rate_per_well'] * 
                         economic_params['oil_price_per_barrel'])
    
    total_economic_benefit = additional_npv + direct_cost_savings + time_savings_value
    
    # Return on investment
    optimization_investment = wells * 50000  # Assume $50k per well optimization cost
    roi = (total_economic_benefit - optimization_investment) / optimization_investment if optimization_investment > 0 else 0
    
    return {
        'additional_npv': additional_npv,
        'direct_cost_savings': direct_cost_savings,
        'time_savings_value': time_savings_value,
        'total_economic_benefit': total_economic_benefit,
        'optimization_investment': optimization_investment,
        'roi': roi,
        'payback_years': optimization_investment / (direct_cost_savings + time_savings_value) if (direct_cost_savings + time_savings_value) > 0 else float('inf')
    }


def risk_assessment_model(field_data: pd.DataFrame, params: GeomechParams) -> Dict[str, any]:
    """
    Comprehensive risk assessment model for field drilling operations.
    
    Input:
      field_data: Multi-well field data
      params: Geomechanical parameters
    Returns:
      Dictionary with risk assessment results
    """
    risk_factors = {}
    
    # Geological risk assessment
    formation_risk = {}
    for formation in field_data['formation'].unique():
        formation_data = field_data[field_data['formation'] == formation]
        success_rate = (formation_data.groupby('well_name')['drilling_status']
                       .apply(lambda x: (x == 'Success').all()).mean())
        
        # Risk score: higher for lower success rates
        risk_score = (1 - success_rate) * 100
        
        # Additional risk factors
        mud_weight_variability = formation_data['mud_weight_used'].std()
        cost_variability = formation_data.groupby('well_name')['cost_per_meter'].first().std()
        
        formation_risk[formation] = {
            'success_rate': success_rate,
            'risk_score': risk_score,
            'mud_weight_variability': mud_weight_variability,
            'cost_variability': cost_variability
        }
    
    risk_factors['formation_risks'] = formation_risk
    
    # Well-specific risk assessment
    well_risks = {}
    wells = process_field_data(field_data)
    
    for well_name, well_df in wells.items():
        # Calculate stability margin for each well
        stability_df = safe_mud_weight_window(well_df[['depth_m', 'RHOB']], params)
        
        # Risk indicators
        avg_window_size = (stability_df['mud_weight_max_gcc'] - 
                          stability_df['mud_weight_min_gcc']).mean()
        
        # Narrower window = higher risk
        window_risk = max(0, (0.5 - avg_window_size) * 200)  # Scale to 0-100
        
        current_status = well_df['drilling_status'].iloc[-1]
        status_risk = 0 if current_status == 'Success' else 50
        
        # Depth risk (deeper = higher risk)
        max_depth = well_df['depth_m'].max()
        depth_risk = min(30, (max_depth - 1000) / 100)  # Cap at 30
        
        total_well_risk = window_risk + status_risk + depth_risk
        
        well_risks[well_name] = {
            'window_risk': window_risk,
            'status_risk': status_risk,
            'depth_risk': depth_risk,
            'total_risk': total_well_risk,
            'risk_category': 'High' if total_well_risk > 60 else 'Medium' if total_well_risk > 30 else 'Low'
        }
    
    risk_factors['well_risks'] = well_risks
    
    # Field-level risk assessment
    overall_success_rate = (field_data.groupby('well_name')['drilling_status']
                           .apply(lambda x: (x == 'Success').all()).mean())
    
    field_risk_score = (1 - overall_success_rate) * 50  # Base field risk
    
    # Add complexity factors
    formation_count = field_data['formation'].nunique()
    complexity_penalty = min(20, (formation_count - 1) * 5)  # More formations = more complex
    
    total_field_risk = field_risk_score + complexity_penalty
    
    risk_factors['field_risk'] = {
        'overall_success_rate': overall_success_rate,
        'field_risk_score': field_risk_score,
        'complexity_penalty': complexity_penalty,
        'total_risk': total_field_risk,
        'risk_category': 'High' if total_field_risk > 40 else 'Medium' if total_field_risk > 20 else 'Low'
    }
    
    return risk_factors


def field_development_sequencing(field_data: pd.DataFrame, optimization: FieldOptimizationResult,
                                risk_assessment: Dict[str, any]) -> Dict[str, any]:
    """
    Optimize the sequence of well drilling for field development.
    
    Input:
      field_data: Multi-well field data
      optimization: Optimization results
      risk_assessment: Risk assessment results
    Returns:
      Dictionary with sequencing recommendations
    """
    well_risks = risk_assessment['well_risks']
    
    # Create well priority scoring
    well_scores = {}
    wells = field_data['well_name'].unique()
    
    for well in wells:
        well_data = field_data[field_data['well_name'] == well]
        
        # Scoring factors (lower score = higher priority)
        risk_score = well_risks[well]['total_risk']  # Lower risk preferred
        cost_score = well_data['cost_per_meter'].iloc[0] / 1000  # Normalize cost
        
        # Economic potential (assume based on formation)
        formation = well_data['formation'].iloc[0]
        economic_score = 50 if formation == 'Sandstone_A' else 20  # Sandstone preferred
        
        # Distance penalty (could be added if coordinates represent true locations)
        distance_score = 0  # Simplified for now
        
        # Combined score (lower is better)
        total_score = risk_score + cost_score - economic_score + distance_score
        
        well_scores[well] = {
            'risk_score': risk_score,
            'cost_score': cost_score,
            'economic_score': economic_score,
            'total_score': total_score
        }
    
    # Sort wells by priority (lowest score first)
    prioritized_wells = sorted(wells, key=lambda w: well_scores[w]['total_score'])
    
    # Group into drilling phases
    phase_size = max(1, len(wells) // 3)  # Divide into 3 phases
    
    phases = {
        'Phase_1_Pilot': prioritized_wells[:phase_size],
        'Phase_2_Development': prioritized_wells[phase_size:2*phase_size],
        'Phase_3_Completion': prioritized_wells[2*phase_size:]
    }
    
    # Generate sequencing recommendations
    recommendations = []
    
    # Phase 1: Low-risk pilot wells
    phase1_avg_risk = np.mean([well_risks[w]['total_risk'] for w in phases['Phase_1_Pilot']])
    recommendations.append(f"Phase 1: Drill pilot wells {', '.join(phases['Phase_1_Pilot'])} (Avg risk: {phase1_avg_risk:.1f})")
    
    # Phase 2: Learning from pilot results
    recommendations.append(f"Phase 2: Apply lessons learned to wells {', '.join(phases['Phase_2_Development'])}")
    
    # Phase 3: Final optimization
    recommendations.append(f"Phase 3: Complete remaining wells {', '.join(phases['Phase_3_Completion'])} with optimized parameters")
    
    return {
        'well_scores': well_scores,
        'prioritized_sequence': prioritized_wells,
        'drilling_phases': phases,
        'sequencing_recommendations': recommendations
    }
