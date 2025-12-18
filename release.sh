#!/bin/bash
# GeoSuite Release Script
# Usage: ./release.sh 0.1.0

set -e

if [ -z "$1" ]; then
    echo "Usage: ./release.sh <version>"
    echo "Example: ./release.sh 0.1.0"
    exit 1
fi

VERSION=$1
TAG="v${VERSION}"

echo "🚀 Releasing GeoSuite ${VERSION}"
echo ""

# Check if we're on main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "⚠️  Warning: You're on branch '${CURRENT_BRANCH}', not 'main'"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if tag already exists
if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "❌ Error: Tag ${TAG} already exists"
    echo "   Delete it first with: git tag -d ${TAG} && git push origin :refs/tags/${TAG}"
    exit 1
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "❌ Error: You have uncommitted changes"
    echo "   Commit or stash them first"
    exit 1
fi

echo "📝 Updating version numbers..."

# Update version in pyproject.toml
sed -i.bak "s/^version = .*/version = \"${VERSION}\"/" pyproject.toml && rm pyproject.toml.bak

# Update version in setup.py
sed -i.bak "s/version=\"[^\"]*\"/version=\"${VERSION}\"/" setup.py && rm setup.py.bak

# Update version in geosuite/__init__.py
sed -i.bak "s/__version__ = \"[^\"]*\"/__version__ = \"${VERSION}\"/" geosuite/__init__.py && rm geosuite/__init__.py.bak

# Update version in docs/source/conf.py
sed -i.bak "s/version = '[^\']*'/version = '${VERSION}'/" docs/source/conf.py && rm docs/source/conf.py.bak
sed -i.bak "s/release = '[^\']*'/release = '${VERSION}'/" docs/source/conf.py && rm docs/source/conf.py.bak

echo "✅ Version updated in pyproject.toml, setup.py, geosuite/__init__.py, and docs/source/conf.py"

# Commit version bump
git add pyproject.toml setup.py geosuite/__init__.py docs/source/conf.py
git commit -m "Bump version to ${VERSION}"

echo "📌 Creating tag ${TAG}..."
git tag -a "$TAG" -m "Release version ${VERSION}"

echo "⬆️  Pushing to GitHub..."
git push origin main
git push origin "$TAG"

echo ""
echo "✅ Done! Release is being published automatically."
echo ""
echo "📊 Watch the progress:"
echo "   https://github.com/kylejones200/geosuite/actions"
echo ""
echo "📦 Package will be available at:"
echo "   https://pypi.org/project/geosuite/ (in ~5 minutes)"
echo ""
echo "🎉 Release completed successfully!"



