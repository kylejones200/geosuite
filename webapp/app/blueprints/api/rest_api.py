"""
REST API endpoints for GeoSuite integration.

Provides comprehensive REST API for programmatic access to GeoSuite
functionality including petrophysics, geomechanics, ML, and data operations.
"""
import logging
from flask import Blueprint, request, jsonify, current_app
from functools import wraps
import traceback
import sys
import os

# Add geosuite to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

try:
    import geosuite
    from geosuite.petro import (
        calculate_water_saturation,
        calculate_porosity_from_density,
        pickett_plot
    )
    from geosuite.geomech import (
        calculate_overburden_stress,
        calculate_hydrostatic_pressure,
        calculate_pore_pressure_eaton,
        calculate_effective_stress
    )
    from geosuite.ml import train_and_predict, PermeabilityPredictor, PorosityPredictor
    from geosuite.stratigraphy import detect_pelt, detect_bayesian_online
    from geosuite.data import load_demo_well_logs
    GEOSUITE_AVAILABLE = True
except ImportError as e:
    GEOSUITE_AVAILABLE = False
    logging.warning(f"GeoSuite not available: {e}")

logger = logging.getLogger(__name__)

bp = Blueprint('rest_api', __name__, url_prefix='/api/v1')


def handle_errors(f):
    """Decorator to handle API errors consistently."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            return jsonify({'error': str(e), 'type': 'ValueError'}), 400
        except KeyError as e:
            return jsonify({'error': f'Missing required field: {str(e)}', 'type': 'KeyError'}), 400
        except Exception as e:
            logger.error(f"API error in {f.__name__}: {e}\n{traceback.format_exc()}")
            return jsonify({'error': str(e), 'type': type(e).__name__}), 500
    return decorated_function


@bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'GeoSuite REST API',
        'version': '1.0.0',
        'geosuite_available': GEOSUITE_AVAILABLE
    })


@bp.route('/petrophysics/water-saturation', methods=['POST'])
@handle_errors
def calculate_sw():
    """
    Calculate water saturation using Archie's equation.
    
    Request body:
        {
            "phi": [0.25, 0.30, ...],
            "rt": [10.5, 12.0, ...],
            "rw": 0.05,
            "m": 2.0,
            "n": 2.0,
            "a": 1.0
        }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Request body required'}), 400
    
    phi = data.get('phi')
    rt = data.get('rt')
    rw = data.get('rw', 0.05)
    m = data.get('m', 2.0)
    n = data.get('n', 2.0)
    a = data.get('a', 1.0)
    
    if phi is None or rt is None:
        return jsonify({'error': 'phi and rt are required'}), 400
    
    import numpy as np
    import pandas as pd
    
    sw = calculate_water_saturation(
        phi=np.array(phi),
        rt=np.array(rt),
        rw=rw,
        m=m,
        n=n,
        a=a
    )
    
    return jsonify({
        'sw': sw.tolist(),
        'parameters': {
            'rw': rw,
            'm': m,
            'n': n,
            'a': a
        }
    })


@bp.route('/petrophysics/porosity', methods=['POST'])
@handle_errors
def calculate_porosity():
    """
    Calculate porosity from density.
    
    Request body:
        {
            "rhob": [2.5, 2.6, ...],
            "rho_matrix": 2.65,
            "rho_fluid": 1.0
        }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Request body required'}), 400
    
    rhob = data.get('rhob')
    rho_matrix = data.get('rho_matrix', 2.65)
    rho_fluid = data.get('rho_fluid', 1.0)
    
    if rhob is None:
        return jsonify({'error': 'rhob is required'}), 400
    
    import numpy as np
    
    phi = calculate_porosity_from_density(
        rhob=np.array(rhob),
        rho_matrix=rho_matrix,
        rho_fluid=rho_fluid
    )
    
    return jsonify({
        'porosity': phi.tolist(),
        'parameters': {
            'rho_matrix': rho_matrix,
            'rho_fluid': rho_fluid
        }
    })


@bp.route('/geomechanics/overburden-stress', methods=['POST'])
@handle_errors
def calculate_sv():
    """
    Calculate overburden stress.
    
    Request body:
        {
            "depth": [0, 100, 200, ...],
            "rhob": [2.5, 2.5, 2.6, ...],
            "g": 9.81
        }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Request body required'}), 400
    
    depth = data.get('depth')
    rhob = data.get('rhob')
    g = data.get('g', 9.81)
    
    if depth is None or rhob is None:
        return jsonify({'error': 'depth and rhob are required'}), 400
    
    import numpy as np
    
    sv = calculate_overburden_stress(
        depth=np.array(depth),
        rhob=np.array(rhob),
        g=g
    )
    
    return jsonify({
        'sv': sv.tolist(),
        'units': 'MPa',
        'parameters': {'g': g}
    })


@bp.route('/geomechanics/pore-pressure', methods=['POST'])
@handle_errors
def calculate_pp():
    """
    Calculate pore pressure using Eaton method.
    
    Request body:
        {
            "depth": [0, 100, 200, ...],
            "dt": [200, 195, 190, ...],
            "dt_normal": 200.0,
            "sv": [0, 2.5, 5.0, ...],
            "ph": [0, 1.0, 2.0, ...],
            "exponent": 3.0
        }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Request body required'}), 400
    
    depth = data.get('depth')
    dt = data.get('dt')
    dt_normal = data.get('dt_normal')
    sv = data.get('sv')
    ph = data.get('ph')
    exponent = data.get('exponent', 3.0)
    
    if None in [depth, dt, dt_normal, sv, ph]:
        return jsonify({'error': 'depth, dt, dt_normal, sv, and ph are required'}), 400
    
    import numpy as np
    
    pp = calculate_pore_pressure_eaton(
        depth=np.array(depth),
        dt=np.array(dt),
        dt_normal=dt_normal,
        sv=np.array(sv),
        ph=np.array(ph),
        exponent=exponent
    )
    
    return jsonify({
        'pp': pp.tolist(),
        'units': 'MPa',
        'method': 'Eaton',
        'parameters': {'exponent': exponent}
    })


@bp.route('/ml/train-classifier', methods=['POST'])
@handle_errors
def train_classifier():
    """
    Train a facies classifier.
    
    Request body:
        {
            "features": [[gr1, nphi1, rhob1], [gr2, nphi2, rhob2], ...],
            "targets": ["Sand", "Shale", ...],
            "model_type": "SVM",
            "test_size": 0.2
        }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Request body required'}), 400
    
    features = data.get('features')
    targets = data.get('targets')
    model_type = data.get('model_type', 'SVM')
    test_size = data.get('test_size', 0.2)
    
    if features is None or targets is None:
        return jsonify({'error': 'features and targets are required'}), 400
    
    import numpy as np
    import pandas as pd
    
    df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(len(features[0]))])
    df['Facies'] = targets
    
    feature_cols = [col for col in df.columns if col != 'Facies']
    
    result = train_and_predict(
        df=df,
        feature_cols=feature_cols,
        target_col='Facies',
        model_type=model_type,
        test_size=test_size
    )
    
    return jsonify({
        'model_type': model_type,
        'classes': result.classes_,
        'predictions': result.y_pred.tolist(),
        'probabilities': result.proba.to_dict('records'),
        'report': result.report
    })


@bp.route('/ml/predict-permeability', methods=['POST'])
@handle_errors
def predict_permeability():
    """
    Predict permeability using ML model.
    
    Request body:
        {
            "features": [[phi1, sw1, gr1], [phi2, sw2, gr2], ...],
            "model_type": "random_forest"
        }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Request body required'}), 400
    
    features = data.get('features')
    model_type = data.get('model_type', 'random_forest')
    
    if features is None:
        return jsonify({'error': 'features are required'}), 400
    
    import numpy as np
    import pandas as pd
    
    # Note: This requires a pre-trained model or training data
    # For now, return a placeholder response
    return jsonify({
        'error': 'Model training required. Use /ml/train-regressor endpoint first.',
        'model_type': model_type
    }), 501


@bp.route('/stratigraphy/detect-changepoints', methods=['POST'])
@handle_errors
def detect_changepoints():
    """
    Detect formation boundaries using change-point detection.
    
    Request body:
        {
            "log_values": [45, 48, 50, 52, ...],
            "method": "pelt",
            "penalty": 10.0
        }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Request body required'}), 400
    
    log_values = data.get('log_values')
    method = data.get('method', 'pelt')
    penalty = data.get('penalty', 10.0)
    
    if log_values is None:
        return jsonify({'error': 'log_values is required'}), 400
    
    import numpy as np
    
    log_array = np.array(log_values)
    
    if method == 'pelt':
        changepoints = detect_pelt(log_array, penalty=penalty)
    elif method == 'bayesian':
        changepoints, probs = detect_bayesian_online(log_array)
        return jsonify({
            'changepoints': changepoints.tolist(),
            'probabilities': probs.tolist(),
            'method': 'bayesian'
        })
    else:
        return jsonify({'error': f'Unknown method: {method}'}), 400
    
    return jsonify({
        'changepoints': changepoints.tolist(),
        'method': 'pelt',
        'penalty': penalty
    })


@bp.route('/data/load-demo', methods=['GET'])
@handle_errors
def load_demo_data():
    """
    Load demo well log dataset.
    
    Query parameters:
        - dataset: Type of dataset (default: 'well_logs')
    """
    dataset_type = request.args.get('dataset', 'well_logs')
    
    if dataset_type == 'well_logs':
        df = load_demo_well_logs()
        return jsonify({
            'data': df.to_dict('records'),
            'columns': df.columns.tolist(),
            'shape': df.shape
        })
    else:
        return jsonify({'error': f'Unknown dataset type: {dataset_type}'}), 400

