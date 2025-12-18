"""
Petrophysics routes for reservoir calculations and plots.
"""

from flask import render_template, request, jsonify
from . import bp
import sys
import os
import pandas as pd
import numpy as np

# Add the geosuite_lib to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'geosuite_lib'))

try:
    from petro.archie import archie_saturation
except ImportError:
    # Fallback Archie's equation if module not available
    def archie_saturation(porosity, resistivity, rw=0.1, a=1, m=2, n=2):
        """Archie's equation for water saturation."""
        sw = ((a * rw) / (porosity**m * resistivity))**(1/n)
        return min(1.0, max(0.0, sw))


@bp.route("/")
def petro_home():
    """Petrophysics home page."""
    return render_template("petro/index.html")


@bp.route("/archie-calculator")
def archie_calculator():
    """Archie's equation calculator page."""
    return render_template("petro/archie.html")


@bp.route("/api/archie-saturation", methods=["POST"])
def calculate_archie():
    """Calculate water saturation using Archie's equation."""
    try:
        data = request.json
        porosity = float(data.get('porosity', 0.2))
        resistivity = float(data.get('resistivity', 10))
        rw = float(data.get('rw', 0.1))
        a = float(data.get('a', 1))
        m = float(data.get('m', 2))
        n = float(data.get('n', 2))
        
        sw = archie_saturation(porosity, resistivity, rw, a, m, n)
        sh = 1 - sw
        
        return jsonify({
            "status": "success",
            "water_saturation": round(sw, 3),
            "hydrocarbon_saturation": round(sh, 3),
            "porosity": porosity,
            "resistivity_ohm_m": resistivity,
            "parameters": {
                "rw_ohm_m": rw,
                "tortuosity_factor_a": a,
                "cementation_exponent_m": m,
                "saturation_exponent_n": n
            }
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Archie calculation failed",
            "error": str(e)
        }), 400


@bp.route("/pickett-plot")
def pickett_plot():
    """Pickett plot analysis page."""
    return render_template("petro/pickett.html")


@bp.route("/api/pickett-data", methods=["POST"])
def generate_pickett_data():
    """Generate data for Pickett plot."""
    try:
        data = request.json
        num_points = int(data.get('num_points', 50))
        
        # Generate synthetic log data for demonstration
        np.random.seed(42)
        porosity = np.random.uniform(0.05, 0.35, num_points)
        resistivity = np.random.lognormal(1, 1, num_points) * 10
        
        plot_data = []
        for i in range(num_points):
            plot_data.append({
                "porosity": round(porosity[i], 3),
                "resistivity": round(resistivity[i], 2),
                "log_resistivity": round(np.log10(resistivity[i]), 2),
                "depth": round(1000 + i * 2, 1)
            })
        
        return jsonify({
            "status": "success",
            "data": plot_data,
            "plot_type": "pickett",
            "axes": {
                "x": "porosity",
                "y": "log_resistivity",
                "x_label": "Porosity (fraction)",
                "y_label": "Log₁₀ Resistivity (Ω⋅m)"
            }
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Pickett data generation failed",
            "error": str(e)
        }), 400


@bp.route("/buckles-plot")
def buckles_plot():
    """Buckles plot analysis page."""
    return render_template("petro/buckles.html")


@bp.route("/api/buckles-data", methods=["POST"])
def generate_buckles_data():
    """Generate data for Buckles plot."""
    try:
        data = request.json
        num_points = int(data.get('num_points', 50))
        
        # Generate synthetic data for Buckles plot (Porosity vs Sw*Phi)
        np.random.seed(42)
        porosity = np.random.uniform(0.05, 0.35, num_points)
        sw = np.random.uniform(0.2, 1.0, num_points)
        bulk_volume_water = porosity * sw
        
        plot_data = []
        for i in range(num_points):
            plot_data.append({
                "porosity": round(porosity[i], 3),
                "water_saturation": round(sw[i], 3),
                "bulk_volume_water": round(bulk_volume_water[i], 3),
                "depth": round(1000 + i * 2, 1)
            })
        
        return jsonify({
            "status": "success",
            "data": plot_data,
            "plot_type": "buckles",
            "axes": {
                "x": "porosity",
                "y": "bulk_volume_water",
                "x_label": "Porosity (fraction)",
                "y_label": "Bulk Volume Water (Sw × φ)"
            }
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Buckles data generation failed",
            "error": str(e)
        }), 400
