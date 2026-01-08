# PyPI Publishing Setup

This document explains how to configure automatic publishing to PyPI using GitHub Actions.

## Workflow Overview

The repository includes two GitHub Actions workflows:

1. **CI Workflow** (`.github/workflows/ci.yml`)
   - Runs on every push and pull request
   - Tests on Ubuntu, macOS, and Windows
   - Runs linting and formatting checks
   - Generates coverage reports

2. **Publish Workflow** (`.github/workflows/publish.yml`)
   - Runs when a new release is published or version tag is pushed
   - Runs tests before building
   - Builds wheel and source distributions
   - Publishes to PyPI (on releases) and TestPyPI (on tags)

## Setup Instructions

### Option 1: Trusted Publishing (Recommended)

GitHub now supports publishing to PyPI without API tokens using OIDC trusted publishing.

1. **Configure PyPI Trusted Publisher**
   - Go to https://pypi.org/manage/account/publishing/
   - Add a new publisher:
     - PyPI Project Name: `geosuite`
     - Owner: `kylejones200`
     - Repository name: `geosuite`
     - Workflow name: `publish.yml`
     - Environment name: `pypi`

2. **Configure TestPyPI (Optional)**
   - Go to https://test.pypi.org/manage/account/publishing/
   - Add the same publisher configuration with environment name: `testpypi`

3. **Create GitHub Environments**
   - Go to your repository Settings > Environments
   - Create environment named `pypi`
   - (Optional) Add protection rules to require approval
   - Create environment named `testpypi` for testing

### Option 2: API Token (Alternative)

If you prefer using API tokens instead of trusted publishing:

1. **Generate PyPI API Token**
   - Go to https://pypi.org/manage/account/token/
   - Create a new API token
   - Scope: "Entire account" or specific to "geosuite" project

2. **Add Token to GitHub Secrets**
   - Go to repository Settings > Secrets and variables > Actions
   - Add new repository secret:
     - Name: `PYPI_API_TOKEN`
     - Value: Your PyPI token (starts with `pypi-`)

3. **Update Workflow**
   - Modify `.github/workflows/publish.yml`
   - Replace the "Publish to PyPI" step with:
     ```yaml
     - name: Publish to PyPI
       env:
         TWINE_USERNAME: __token__
         TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
       run: twine upload dist/*
     ```

## Publishing a New Release

### Method 1: GitHub Release (Recommended)

1. Update version in `pyproject.toml` and `setup.py`
2. Update `CHANGELOG.md` with changes
3. Commit changes:
   ```bash
   git add pyproject.toml setup.py CHANGELOG.md
   git commit -m "Bump version to X.Y.Z"
   git push
   ```
4. Create and push a tag:
   ```bash
   git tag -a vX.Y.Z -m "Release version X.Y.Z"
   git push origin vX.Y.Z
   ```
5. Go to GitHub repository > Releases > Create a new release
6. Select the tag you just created
7. Fill in release notes
8. Click "Publish release"
9. GitHub Actions will automatically build and publish to PyPI

### Method 2: Git Tag Only

1. Update version and changelog (same as above)
2. Create and push tag:
   ```bash
   git tag -a vX.Y.Z -m "Release version X.Y.Z"
   git push origin vX.Y.Z
   ```
3. This will trigger publishing to TestPyPI for testing
4. If TestPyPI looks good, create the GitHub release to publish to PyPI

## Testing Before Publishing

To test the build locally before publishing:

```bash
# Install build tools
pip install build twine

# Build distributions
python -m build

# Check the build
twine check dist/*

# Test upload to TestPyPI (optional)
twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ geosuite
```

## Versioning

This project follows [Semantic Versioning](https://semver.org/):

- **MAJOR** version (X.0.0): Incompatible API changes
- **MINOR** version (0.X.0): New functionality, backwards compatible
- **PATCH** version (0.0.X): Bug fixes, backwards compatible

Example versions:
- `0.1.0` - Initial beta release
- `0.2.0` - New features added
- `0.2.1` - Bug fixes
- `1.0.0` - First stable release

## Troubleshooting

### Build Fails

- Check that all dependencies in `pyproject.toml` are correct
- Ensure Python 3.12+ is specified
- Run tests locally: `pytest tests/`

### Tests Fail in CI

- Check test matrix in `.github/workflows/ci.yml`
- Tests must pass on all platforms before publishing
- Review test output in GitHub Actions logs

### PyPI Upload Fails

- Verify PyPI trusted publisher is configured correctly
- Check that version number is unique (not previously published)
- Ensure `pyproject.toml` has correct package metadata
- Review GitHub Actions logs for detailed error messages

### Version Already Exists

PyPI does not allow re-uploading the same version. You must:
1. Increment the version number
2. Create a new tag
3. Publish again

## Package Metadata

Ensure these fields are correct in `pyproject.toml`:

```toml
[project]
name = "geosuite"
version = "0.1.0"  # Update for each release
description = "Professional Python tools for subsurface analysis"
authors = [{name = "K. Jones", email = "kyletjones@gmail.com"}]
readme = "README.md"
requires-python = ">=3.12"
license = {text = "MIT"}
```

## Security Notes

- Never commit API tokens to the repository
- Use GitHub Secrets for sensitive data
- Trusted publishing is more secure than API tokens
- Enable branch protection rules on `main` branch
- Require pull request reviews before merging

## Resources

- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Python Packaging Guide](https://packaging.python.org/)
- [Semantic Versioning](https://semver.org/)






