# GeoSuite Documentation

This directory contains the Sphinx documentation for GeoSuite.

## Building the Documentation

### Prerequisites

Install documentation dependencies:

```bash
pip install -e ".[dev]"
```

Or install just the Sphinx dependencies:

```bash
pip install sphinx sphinx-rtd-theme sphinx-autobuild myst-parser
```

### Build HTML Documentation

From the `docs/` directory:

```bash
make html
```

The generated HTML will be in `build/html/`. Open `build/html/index.html` in your browser.

### Live Preview

For live preview during editing:

```bash
make livehtml
```

This starts a local server at http://127.0.0.1:8000 with auto-reload on file changes.

### Clean Build

To remove all built documentation:

```bash
make clean
```

Then rebuild:

```bash
make html
```

## Documentation Structure

```
docs/
├── source/                    # Documentation source files
│   ├── index.rst             # Main index page
│   ├── installation.rst      # Installation guide
│   ├── quickstart.rst        # Quick start guide
│   ├── performance.rst       # Performance guide
│   ├── examples.rst          # Examples
│   ├── contributing.rst      # Contributing guide
│   ├── changelog.rst         # Changelog
│   ├── api/                  # API reference
│   │   ├── index.rst
│   │   ├── data.rst
│   │   ├── petro.rst
│   │   ├── geomech.rst
│   │   ├── ml.rst
│   │   ├── stratigraphy.rst
│   │   ├── io.rst
│   │   ├── plotting.rst
│   │   ├── imaging.rst
│   │   ├── geospatial.rst
│   │   └── utils.rst
│   ├── guides/               # User guides
│   │   ├── petrophysics.rst
│   │   ├── geomechanics.rst
│   │   ├── machine_learning.rst
│   │   ├── stratigraphy.rst
│   │   └── data_io.rst
│   ├── conf.py               # Sphinx configuration
│   ├── _static/              # Static files (CSS, images)
│   └── _templates/           # Custom templates
├── build/                     # Generated documentation (not in git)
│   └── html/                 # HTML output
├── Makefile                   # Unix build script
├── make.bat                   # Windows build script
└── README.md                  # This file
```

## Documentation Formats

The documentation is written in reStructuredText (.rst) and Markdown (.md) formats.

### reStructuredText (.rst)

Used for structured documentation with cross-references:

- API documentation
- Index pages
- Guides with complex structure

### Markdown (.md)

Supported via myst-parser:

- Simple guides
- Examples
- Contributing guidelines

## Adding Documentation

### Adding a New Guide

1. Create a new `.rst` file in `source/guides/`
2. Add it to the toctree in `source/index.rst`
3. Write your content
4. Build and preview

### Documenting a New Module

1. Create a new `.rst` file in `source/api/`
2. Add automodule directives
3. Add to `source/api/index.rst`
4. Build documentation

Example:

```rst
My Module
=========

.. automodule:: geosuite.mymodule
   :members:
   :undoc-members:
   :show-inheritance:
```

### Adding Code Examples

Use code blocks with language specification:

````rst
.. code-block:: python

   from geosuite import my_function
   result = my_function(42)
````

## Read the Docs

The documentation is automatically built and published on Read the Docs when changes are pushed to the repository.

Configuration: `.readthedocs.yaml` in the project root

## Sphinx Extensions

The documentation uses the following Sphinx extensions:

- `sphinx.ext.autodoc` - Auto-generate API documentation from docstrings
- `sphinx.ext.napoleon` - Support for NumPy and Google style docstrings
- `sphinx.ext.viewcode` - Add links to highlighted source code
- `sphinx.ext.intersphinx` - Link to other project's documentation
- `sphinx.ext.autosummary` - Generate autodoc summaries
- `sphinx.ext.mathjax` - Render math via MathJax
- `sphinx_rtd_theme` - Read the Docs theme
- `myst_parser` - Markdown support

## Docstring Style

Use Google-style docstrings:

```python
def my_function(x: float, y: float) -> float:
    """
    Brief description.
    
    Longer description with more details.
    
    Parameters
    ----------
    x : float
        Description of x
    y : float
        Description of y
        
    Returns
    -------
    float
        Description of return value
        
    Examples
    --------
    >>> my_function(1.0, 2.0)
    3.0
    """
    return x + y
```

## Troubleshooting

### Import Errors

If you see import errors during build, ensure:

1. GeoSuite is installed: `pip install -e .`
2. All dependencies are installed
3. You're in the correct Python environment

### Build Warnings

Some warnings are expected (duplicate object descriptions). As long as the build completes, the documentation should be fine.

### Missing Modules

If autodoc can't find a module:

1. Check that the module is in the Python path
2. Verify the import path in the .rst file
3. Check `conf.py` for `autodoc_mock_imports` if using optional dependencies

## Contributing

When adding or updating documentation:

1. Follow the existing structure and style
2. Build locally to check for errors
3. Preview the HTML output
4. Include code examples where appropriate
5. Update the index/toctree as needed

## Support

For questions or issues with the documentation:

- GitHub Issues: https://github.com/kylejones200/geosuite/issues
- Documentation: https://geosuite.readthedocs.io

