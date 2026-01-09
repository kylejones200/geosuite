"""
GeoSuite - A comprehensive Python library for geoscience workflows
"""
from setuptools import setup, find_packages
import os

# Read the README for long description
def read_file(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Core dependencies
INSTALL_REQUIRES = [
    "pandas>=2.0",
    "numpy>=2.0.0",
    "scikit-learn>=1.3",
    "lasio>=0.30",
    "segyio>=1.9",
    "matplotlib>=3.7",
    "seaborn>=0.12",
    "joblib>=1.3.0",
    "cloudpickle>=2.2.0",
    "scipy>=1.13.0",
    "ruptures>=1.1",
    "pyarrow>=14.0.0",
]

# Optional dependencies
EXTRAS_REQUIRE = {
    'ml': [
        'mlflow>=2.8.0',
        'mlflow[extras]>=2.8.0',
    ],
    'geospatial': [
        'apache-sedona==1.5.1',
        'pyspark>=3.4.0,<3.5.0',
        'shapely>=2.0.0',
        'pyproj>=3.6.0',
        'geopandas>=0.14.0',
        'h3>=4.1.0',
        'fiona>=1.9.0',
    ],
    'data': [
        'openpyxl>=3.1.5',
        'duckdb>=1.1.3',
    ],
    'webapp': [
        'Flask>=3.0.0',
        'dash>=2.16',
        'dash-bootstrap-components>=1.4',
        'dash-echarts>=0.0.7',
        'python-dotenv>=1.0.0',
        'gunicorn>=21.0.0',
    ],
    'imaging': [
        'scikit-image>=0.21',
    ],
    'dev': [
        'pytest>=7.4.0',
        'pytest-cov>=4.1.0',
        'pytest-mock>=3.11.1',
        'black>=23.0.0',
        'ruff>=0.1.0',
        'mypy>=1.0.0',
    ],
}

# Add 'all' option to install everything
EXTRAS_REQUIRE['all'] = list(set([
    pkg for extras in EXTRAS_REQUIRE.values() for pkg in extras
]))

setup(
    name="geosuite",
    version="0.1.4",
    author="K. Jones",
    author_email="kyletjones@gmail.com",
    description="A Python library for geoscience workflows: geomechanics, petrophysics, machine learning, and data I/O",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/kylejones200/geosuite",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*", "webapp", "webapp.*", "docs", "archive"]),
    include_package_data=True,
    python_requires=">=3.12",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="geoscience geophysics petrophysics geomechanics petroleum well-log machine-learning geology",
    project_urls={
        "Documentation": "https://github.com/kylejones200/geosuite#readme",
        "Source": "https://github.com/kylejones200/geosuite",
        "Issues": "https://github.com/kylejones200/geosuite/issues",
    },
)

