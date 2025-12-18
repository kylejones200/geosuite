"""
Geomechanics routes for stress calculations and analysis.
"""

from flask import render_template, request, jsonify
from . import bp
import sys
import os

# Add the geosuite_lib to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'geosuite_lib'))

try:
    from geomech.basic import calculate_vertical_stress, calculate_pore_pressure, calculate_effective_stress
except ImportError:
    # Fallback functions if geomech module not available
    def calculate_vertical_stress(depth, density=2.5):
        return depth * density * 9.81 / 1000  # Approximate in MPa
    
    def calculate_pore_pressure(depth, gradient=0.44):
        return depth * gradient / 100  # Approximate in MPa
    
    def calculate_effective_stress(total_stress, pore_pressure):
        return total_stress - pore_pressure


@bp.route("/")
def geomech_home():
    """Geomechanics home page."""
    return render_template("geomech/index.html")


@bp.route("/stress-calculator")
def stress_calculator():
    """Stress calculator page."""
    return render_template("geomech/stress_calculator.html")


@bp.route("/api/calculate-stress", methods=["POST"])
def calculate_stress():
    """Calculate stress values for given parameters."""
    try:
        data = request.json
        depth = float(data.get('depth', 0))
        bulk_density = float(data.get('bulk_density', 2.5))
        pp_gradient = float(data.get('pp_gradient', 0.44))
        
        # Perform calculations
        sv = calculate_vertical_stress(depth, bulk_density)
        pp = calculate_pore_pressure(depth, pp_gradient)
        eff_stress = calculate_effective_stress(sv, pp)
        
        return jsonify({
            "status": "success",
            "depth_m": depth,
            "vertical_stress_mpa": round(sv, 2),
            "pore_pressure_mpa": round(pp, 2),
            "effective_stress_mpa": round(eff_stress, 2),
            "parameters": {
                "bulk_density_gcm3": bulk_density,
                "pp_gradient_kpa_m": pp_gradient
            }
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Stress calculation failed",
            "error": str(e)
        }), 400


@bp.route("/wellbore-stability")
def wellbore_stability():
    """Wellbore stability analysis page."""
    return render_template("geomech/wellbore_stability.html")


@bp.route("/api/stability-analysis", methods=["POST"])
def stability_analysis():
    """Perform wellbore stability analysis."""
    try:
        data = request.json
        
        # Basic stability check (placeholder implementation)
        mud_weight = float(data.get('mud_weight', 1.2))
        pp_gradient = float(data.get('pp_gradient', 0.44))
        depth = float(data.get('depth', 1000))
        
        min_mud_weight = pp_gradient * depth / 1000 * 1.05  # 5% safety factor
        max_mud_weight = pp_gradient * depth / 1000 * 1.8   # Fracture gradient estimate
        
        stable = min_mud_weight <= mud_weight <= max_mud_weight
        
        return jsonify({
            "status": "success",
            "stable": stable,
            "mud_weight_sg": mud_weight,
            "min_mud_weight_sg": round(min_mud_weight, 2),
            "max_mud_weight_sg": round(max_mud_weight, 2),
            "safety_status": "SAFE" if stable else "WARNING",
            "depth_m": depth
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Stability analysis failed",
            "error": str(e)
        }), 400
