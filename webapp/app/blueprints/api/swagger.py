"""
OpenAPI/Swagger documentation for GeoSuite REST API.
"""
from flask import Blueprint, jsonify
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

bp = Blueprint('swagger', __name__, url_prefix='/api/docs')


@bp.route('/openapi.json')
def openapi_spec():
    """Generate OpenAPI 3.0 specification."""
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "GeoSuite REST API",
            "version": "1.0.0",
            "description": "REST API for GeoSuite geoscience workflows",
            "contact": {
                "name": "GeoSuite Support",
                "url": "https://github.com/kylejones200/geosuite"
            }
        },
        "servers": [
            {
                "url": "/api/v1",
                "description": "GeoSuite API v1"
            }
        ],
        "paths": {
            "/health": {
                "get": {
                    "summary": "Health check",
                    "tags": ["System"],
                    "responses": {
                        "200": {
                            "description": "Service is healthy",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string"},
                                            "service": {"type": "string"},
                                            "version": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/petrophysics/water-saturation": {
                "post": {
                    "summary": "Calculate water saturation",
                    "tags": ["Petrophysics"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["phi", "rt"],
                                    "properties": {
                                        "phi": {
                                            "type": "array",
                                            "items": {"type": "number"},
                                            "description": "Porosity values"
                                        },
                                        "rt": {
                                            "type": "array",
                                            "items": {"type": "number"},
                                            "description": "Resistivity values (ohm.m)"
                                        },
                                        "rw": {"type": "number", "default": 0.05},
                                        "m": {"type": "number", "default": 2.0},
                                        "n": {"type": "number", "default": 2.0},
                                        "a": {"type": "number", "default": 1.0}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Water saturation calculated",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "sw": {
                                                "type": "array",
                                                "items": {"type": "number"}
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "400": {"description": "Bad request"}
                    }
                }
            },
            "/geomechanics/overburden-stress": {
                "post": {
                    "summary": "Calculate overburden stress",
                    "tags": ["Geomechanics"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["depth", "rhob"],
                                    "properties": {
                                        "depth": {
                                            "type": "array",
                                            "items": {"type": "number"},
                                            "description": "Depth values (m)"
                                        },
                                        "rhob": {
                                            "type": "array",
                                            "items": {"type": "number"},
                                            "description": "Bulk density (g/cc)"
                                        },
                                        "g": {"type": "number", "default": 9.81}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Overburden stress calculated",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "sv": {
                                                "type": "array",
                                                "items": {"type": "number"}
                                            },
                                            "units": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/ml/train-classifier": {
                "post": {
                    "summary": "Train facies classifier",
                    "tags": ["Machine Learning"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["features", "targets"],
                                    "properties": {
                                        "features": {
                                            "type": "array",
                                            "items": {
                                                "type": "array",
                                                "items": {"type": "number"}
                                            }
                                        },
                                        "targets": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        },
                                        "model_type": {"type": "string", "default": "SVM"},
                                        "test_size": {"type": "number", "default": 0.2}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Model trained successfully"
                        }
                    }
                }
            },
            "/stratigraphy/detect-changepoints": {
                "post": {
                    "summary": "Detect formation boundaries",
                    "tags": ["Stratigraphy"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["log_values"],
                                    "properties": {
                                        "log_values": {
                                            "type": "array",
                                            "items": {"type": "number"}
                                        },
                                        "method": {"type": "string", "default": "pelt"},
                                        "penalty": {"type": "number", "default": 10.0}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Change points detected"
                        }
                    }
                }
            }
        },
        "tags": [
            {"name": "System", "description": "System endpoints"},
            {"name": "Petrophysics", "description": "Petrophysics calculations"},
            {"name": "Geomechanics", "description": "Geomechanics calculations"},
            {"name": "Machine Learning", "description": "ML model training and prediction"},
            {"name": "Stratigraphy", "description": "Stratigraphic analysis"},
            {"name": "Data", "description": "Data loading and management"}
        ]
    }
    
    return jsonify(spec)


@bp.route('/swagger.json')
def swagger_spec():
    """Legacy Swagger 2.0 specification (redirects to OpenAPI)."""
    return openapi_spec()

