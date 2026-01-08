Versioning Documentation
========================

GeoSuite documentation supports versioning to maintain documentation for multiple
releases. This allows users to access documentation specific to their installed version.

Versioning Setup
----------------

The documentation is configured to support multiple versions using Sphinx's
versioning capabilities. Each release creates a new documentation version.

Accessing Versioned Documentation
----------------------------------

Documentation versions are available at:

- **Latest**: https://geosuite.readthedocs.io/en/latest/
- **Stable**: https://geosuite.readthedocs.io/en/stable/
- **Version-specific**: https://geosuite.readthedocs.io/en/v0.1.3/

Version Selection
-----------------

Users can select their version from the documentation dropdown menu or by
navigating to the version-specific URL.

Versioning Workflow
-------------------

1. **Release Process**: When a new version is released, documentation is
   automatically built and tagged.

2. **Version Tags**: Git tags (e.g., v0.1.3) trigger documentation builds
   for that specific version.

3. **Read the Docs**: Read the Docs automatically builds and hosts
   versioned documentation.

Configuration
-------------

Versioning is configured in:

- ``docs/source/conf.py``: Sphinx configuration with version settings
- ``.readthedocs.yaml``: Read the Docs configuration for automated builds
- ``pyproject.toml``: Package version information

Current Version
---------------

The current documentation version matches the package version defined in
``geosuite/__init__.py`` (``__version__``).

