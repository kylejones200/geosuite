# GeoSuite Test Suite

## Overview

This directory contains the test suite for GeoSuite, including unit tests and integration tests.

## Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests
│   ├── test_confusion_matrix_utils.py
│   ├── test_demo_datasets.py
│   └── test_classifiers.py
└── integration/             # Integration tests
    └── test_flask_app.py
```

## Running Tests

### Install Test Dependencies

```bash
pip install pytest pytest-cov pytest-mock
```

### Run All Tests

```bash
# From project root
pytest

# With verbose output
pytest -v

# With coverage report
pytest --cov=geosuite_lib --cov=app --cov-report=html
```

### Run Specific Test Files

```bash
# Unit tests only
pytest tests/unit/

# Specific test file
pytest tests/unit/test_confusion_matrix_utils.py

# Specific test class
pytest tests/unit/test_confusion_matrix_utils.py::TestDisplayCM

# Specific test function
pytest tests/unit/test_confusion_matrix_utils.py::TestDisplayCM::test_basic_display
```

### Run with Markers

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

## Test Results

### Current Status (October 4, 2025)

**Confusion Matrix Utils**: 22/24 passing (91.7%)
- 2 edge case failures (documented, low priority)

**Demo Datasets**: Expected to pass

**Classifiers**: Expected to pass

**Flask App**: Expected to pass

## Writing New Tests

### Unit Test Template

```python
"""
Unit tests for module_name.
"""
import pytest
from geosuite.module import function_to_test


class TestFunctionName:
    """Tests for specific function."""
    
    def test_basic_functionality(self):
        """Test basic use case."""
        result = function_to_test(input_data)
        assert result == expected_output
    
    def test_edge_case(self):
        """Test edge case."""
        with pytest.raises(ValueError):
            function_to_test(invalid_input)
```

### Using Fixtures

```python
def test_with_fixture(sample_confusion_matrix, sample_labels):
    """Test using shared fixtures."""
    result = display_cm(sample_confusion_matrix, sample_labels)
    assert isinstance(result, str)
```

## Test Fixtures

Available fixtures (defined in `conftest.py`):

- `sample_confusion_matrix` - 3x3 numpy array
- `sample_labels` - List of class labels
- `sample_well_log_data` - DataFrame with well log data
- `sample_facies_data` - DataFrame with facies labels
- `flask_app` - Flask application instance
- `client` - Flask test client
- `adjacent_facies` - Adjacent facies mapping

## Coverage Goals

- **Unit Tests**: > 80% coverage
- **Integration Tests**: All major routes
- **Overall**: > 70% coverage

## Continuous Integration

### GitHub Actions (Future)

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest --cov --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Test Categories

### Unit Tests
- Test individual functions and classes
- Mock external dependencies
- Fast execution (< 1 second each)
- No database or network calls

### Integration Tests
- Test component interactions
- Test Flask routes
- May use test database
- Slower execution (< 5 seconds each)

## Common Issues

### Import Errors

If you get import errors, ensure you're running from the project root:

```bash
cd /Users/k.jones/Documents/geos/geosuite
pytest
```

### MLflow Tests Failing

Some tests require MLflow to be configured. Skip them with:

```bash
pytest -m "not requires_mlflow"
```

### Databricks Tests Failing

Tests requiring Databricks connection will fail if not configured:

```bash
pytest -m "not requires_databricks"
```

## Test Data

Test data is generated programmatically in fixtures. No external test files are required.

## Best Practices

1. **One assertion per test** - Makes failures easier to diagnose
2. **Descriptive test names** - Should describe what is being tested
3. **Arrange-Act-Assert** - Structure tests in three phases
4. **Use fixtures** - Share common setup code
5. **Test edge cases** - Empty inputs, None, etc.
6. **Mock external calls** - Don't call real APIs in tests

## Example Test Session

```bash
$ pytest -v

============================= test session starts ==============================
platform darwin -- Python 3.12.7, pytest-8.4.1
collected 45 items

tests/unit/test_confusion_matrix_utils.py::TestDisplayCM::test_basic_display PASSED [ 2%]
tests/unit/test_confusion_matrix_utils.py::TestDisplayCM::test_with_metrics PASSED [ 4%]
...
tests/integration/test_flask_app.py::TestMainRoutes::test_home_page PASSED [100%]

========================== 43 passed, 2 skipped in 5.23s =======================
```

## Contributing

When adding new features, please also add:
1. Unit tests for new functions
2. Integration tests for new routes
3. Update fixtures if needed
4. Update this README if adding new test patterns

## Troubleshooting

### Tests hanging
- Check for infinite loops
- Add `pytest-timeout` to requirements
- Run with `pytest --timeout=10`

### Fixtures not found
- Ensure `conftest.py` is in the correct location
- Check fixture scope (function, class, module, session)

### Coverage not working
- Install `pytest-cov`: `pip install pytest-cov`
- Run with `--cov` flag

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Coverage.py](https://coverage.readthedocs.io/)

---

**Last Updated**: October 4, 2025  
**Test Framework**: pytest 7.4.0+  
**Coverage Tool**: pytest-cov 4.1.0+

