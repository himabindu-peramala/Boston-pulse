"""
Boston Pulse - Pytest Configuration and Fixtures

Shared fixtures for all tests:
- Configuration fixtures
- Mock fixtures for external services
"""

import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

# Set test environment
os.environ["BP_ENVIRONMENT"] = "dev"

# =============================================================================
# Path Fixtures
# =============================================================================


@pytest.fixture
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def configs_dir(project_root: Path) -> Path:
    """Get the configs directory."""
    return project_root / "configs"


@pytest.fixture
def schemas_dir(project_root: Path) -> Path:
    """Get the schemas directory."""
    return project_root / "schemas"


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def test_config() -> Any:
    """Get test configuration."""
    from src.shared.config import get_config, reload_config

    # Ensure fresh config for tests
    reload_config("dev")
    return get_config("dev")


@pytest.fixture
def sample_coordinates() -> list[tuple[float, float]]:
    """Sample Boston coordinates for testing."""
    return [
        (42.3601, -71.0589),  # Downtown
        (42.3505, -71.0765),  # Back Bay
        (42.3656, -71.0096),  # East Boston
        (42.2808, -71.0728),  # Mattapan
    ]


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_gcs_client(mocker: Any) -> Any:
    """Mock Google Cloud Storage client."""
    mock_client = mocker.MagicMock()
    mocker.patch("google.cloud.storage.Client", return_value=mock_client)
    return mock_client


@pytest.fixture
def mock_analyze_boston_api(mocker: Any) -> Any:
    """Mock Analyze Boston API responses."""
    mock_response = mocker.MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "success": True,
        "result": {
            "records": [
                {"id": 1, "name": "Test Record 1"},
                {"id": 2, "name": "Test Record 2"},
            ]
        },
    }
    mocker.patch("requests.get", return_value=mock_response)
    return mock_response


# =============================================================================
# Cleanup Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def cleanup_env() -> Generator[None, None, None]:
    """Clean up environment variables after each test."""
    original_env = os.environ.copy()
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
