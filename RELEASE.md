# Releasing GeoSuite to PyPI

This guide explains how to programmatically release GeoSuite to PyPI.

## Method 1: GitHub Actions UI (Recommended)

1. Go to https://github.com/kylejones200/geosuite/actions/workflows/publish.yml
2. Click "Run workflow"
3. Enter the version number (e.g., `0.1.4`)
4. Click "Run workflow"
5. The workflow will:
   - Update version numbers in all files
   - Create a git tag
   - Run tests
   - Build the package
   - Publish to PyPI
   - Create a GitHub release

## Method 2: Git Tag (Automatic)

1. Update version in files manually or use `release.sh`:
   ```bash
   ./release.sh 0.1.4
   ```
2. This will:
   - Update version in `pyproject.toml`, `setup.py`, and `geosuite/__init__.py`
   - Commit the changes
   - Create and push a git tag `v0.1.4`
3. Pushing the tag automatically triggers the publish workflow

## Method 3: Release Script

The `release.sh` script automates the process:

```bash
./release.sh 0.1.4
```

This script:
- Checks you're on the main branch
- Verifies no uncommitted changes
- Updates version in all files
- Commits the version bump
- Creates and pushes a git tag
- Triggers the GitHub Actions workflow

## Prerequisites

1. **PyPI Trusted Publishing** must be configured:
   - Go to https://pypi.org/manage/account/publishing/
   - Add trusted publisher:
     - Project: `geosuite`
     - Owner: `kylejones200`
     - Repository: `geosuite`
     - Workflow: `publish.yml`
     - Environment: `pypi`

2. **GitHub Environment** must exist:
   - Go to repository Settings > Environments
   - Create environment named `pypi`

## Version Format

Follow [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH` (e.g., `0.1.4`)
- Tags must be prefixed with `v` (e.g., `v0.1.4`)

## Verification

After release:
- Check PyPI: https://pypi.org/project/geosuite/
- Check GitHub Releases: https://github.com/kylejones200/geosuite/releases
- Test installation: `pip install geosuite==0.1.4`

