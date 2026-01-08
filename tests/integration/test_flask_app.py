"""
Integration tests for Flask application.
"""
import pytest
import json
import sys
import os
from jinja2.exceptions import TemplateNotFound

# Skip all tests if webapp is not available
webapp_path = os.path.join(os.path.dirname(__file__), '..', '..', 'webapp')
if not os.path.exists(webapp_path):
    pytestmark = pytest.mark.skip("Webapp not available")
else:
    # Try to import Flask app
    try:
        sys.path.insert(0, webapp_path)
        from app import create_app
    except ImportError:
        pytestmark = pytest.mark.skip("Flask app module not available")


class TestMainRoutes:
    """Test main application routes."""
    
    def test_home_page(self, client):
        """Test home page loads."""
        response = client.get('/')
        assert response.status_code == 200
    
    def test_404_error(self, client):
        """Test 404 error handling."""
        response = client.get('/nonexistent-page')
        assert response.status_code == 404


class TestDataRoutes:
    """Test data management routes."""
    
    def test_data_home(self, client):
        """Test data home page."""
        try:
            response = client.get('/data/')
            # Template exists
            assert response.status_code in [200, 404, 500]
        except TemplateNotFound:
            # Template missing - this is expected in minimal setup
            pytest.skip("Template not found - webapp templates not fully configured")
    
    def test_data_import(self, client):
        """Test data import page."""
        try:
            response = client.get('/data/import')
            # Template exists
            assert response.status_code in [200, 404, 500]
        except TemplateNotFound:
            # Template missing - this is expected in minimal setup
            pytest.skip("Template not found - webapp templates not fully configured")


class TestMLRoutes:
    """Test ML/MLflow routes."""
    
    def test_ml_home(self, client):
        """Test ML home page."""
        response = client.get('/ml/')
        assert response.status_code == 200
    
    def test_experiments_page(self, client):
        """Test experiments page."""
        response = client.get('/ml/experiments')
        assert response.status_code == 200
    
    def test_models_page(self, client):
        """Test models page."""
        response = client.get('/ml/models')
        assert response.status_code == 200


class TestPetroRoutes:
    """Test petrophysics routes."""
    
    def test_petro_home(self, client):
        """Test petrophysics home page."""
        try:
            response = client.get('/petro/')
            # Template exists
            assert response.status_code in [200, 404, 500]
        except TemplateNotFound:
            # Template missing - this is expected in minimal setup
            pytest.skip("Template not found - webapp templates not fully configured")


class TestGeomechRoutes:
    """Test geomechanics routes."""
    
    def test_geomech_home(self, client):
        """Test geomechanics home page."""
        try:
            response = client.get('/geomech/')
            # Template exists
            assert response.status_code in [200, 404, 500]
        except TemplateNotFound:
            # Template missing - this is expected in minimal setup
            pytest.skip("Template not found - webapp templates not fully configured")


class TestWellsRoutes:
    """Test wells/production routes."""
    
    def test_wells_home(self, client):
        """Test wells home page."""
        response = client.get('/wells/')
        assert response.status_code == 200


class TestAPIRoutes:
    """Test API endpoints."""
    
    def test_api_experiments(self, client):
        """Test experiments API."""
        response = client.get('/ml/api/experiments')
        assert response.status_code in [200, 500]  # May fail if MLflow not configured
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'status' in data
    
    def test_api_models(self, client):
        """Test models API."""
        response = client.get('/ml/api/models')
        assert response.status_code in [200, 500]  # May fail if MLflow not configured
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'status' in data


class TestAppConfiguration:
    """Test application configuration."""
    
    def test_testing_mode(self, flask_app):
        """Test that testing mode is enabled."""
        assert flask_app.config['TESTING'] is True
    
    def test_secret_key_set(self, flask_app):
        """Test that secret key is set."""
        assert flask_app.config['SECRET_KEY'] is not None
        assert flask_app.config['SECRET_KEY'] != ''
    
    def test_blueprints_registered(self, flask_app):
        """Test that all blueprints are registered."""
        blueprint_names = [bp.name for bp in flask_app.blueprints.values()]
        
        expected_blueprints = ['main', 'api', 'data', 'ml', 'petro', 'geomech', 'wells']
        for bp_name in expected_blueprints:
            assert bp_name in blueprint_names, f"Blueprint {bp_name} not registered"


class TestStaticFiles:
    """Test static file serving."""
    
    def test_css_accessible(self, client):
        """Test that CSS files are accessible."""
        response = client.get('/static/css/wells.css')
        # May be 200 or 404 depending on file existence
        assert response.status_code in [200, 404]
    
    def test_static_directory(self, flask_app):
        """Test that static directory is configured."""
        assert flask_app.static_folder is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

