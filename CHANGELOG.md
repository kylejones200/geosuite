# Changelog

All notable changes to GeoSuite will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-12-17

### Added
- Initial release of GeoSuite as a pip-installable Python library
- Core modules: `io`, `petro`, `geomech`, `ml`, `geospatial`, `plotting`, `data`
- Data I/O support for LAS, SEG-Y, PPDM, WITSML, and CSV formats
- Petrophysics calculations (Archie equation, Pickett plots, Buckles plots)
- Geomechanics calculations (overburden stress, pore pressure, stress polygons)
- Machine learning with MLflow integration for facies classification
- Apache Sedona integration for geospatial operations
- Visualization utilities (strip charts, crossplots)
- Demo datasets (Kansas University facies data, sample well logs)
- Flask-based web application (optional installation)
- Comprehensive test suite with pytest
- Project structure with proper packaging (setup.py, pyproject.toml)
- Documentation (README, API docs, examples)

### Changed
- Restructured project from monolithic app to modular library
- Separated core library from web application
- Moved notebooks to examples directory
- Consolidated documentation into docs folder
- Organized sample data into data directory
- Archived legacy projects

### Fixed
- Import errors in various modules
- Package dependencies organization
- Module __init__ files for clean API

## [Unreleased]

### Planned
- Enhanced geospatial operations with more Sedona features
- Additional ML models (neural networks, ensemble methods)
- Real-time data streaming capabilities
- Cloud deployment support (AWS, Azure, GCP)
- Database backend integration (PostgreSQL, TimescaleDB)
- API documentation with Swagger/OpenAPI
- Comprehensive documentation website
- Docker and Kubernetes deployment configs
- User authentication for web application
- Databricks integration for big data processing


