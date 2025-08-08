import pytest
from unittest.mock import patch, MagicMock
from src.models.model_registry import ModelRegistry

@pytest.fixture
def registry():
    return ModelRegistry()

def make_mock_model(name="model-A", versions=None):
    """Helper to create a fake MLflow model object."""
    mock_model = MagicMock()
    mock_model.name = name
    mock_model.description = "A test model"
    mock_model.tags = {"tag1": "value1"}
    mock_model.creation_timestamp = 1234567890
    mock_model.last_updated_timestamp = 1234567899
    mock_model.latest_versions = versions or []
    return mock_model

def make_mock_version(version="1", stage="Production", status="READY", run_id="run123"):
    """Helper to create a fake MLflow model version object."""
    mv = MagicMock()
    mv.version = version
    mv.current_stage = stage
    mv.creation_timestamp = 1234567890
    mv.status = status
    mv.name = "model-A"
    mv.run_id = run_id
    return mv

# ---------- Tests ----------

@patch("src.models.model_registry.MlflowClient.search_registered_models")
def test_list_mlflow_registered_models_returns_data(mock_search, registry):
    mock_version = make_mock_version()
    mock_model = make_mock_model(versions=[mock_version])
    mock_search.return_value = [mock_model]

    result = registry.list_mlflow_registered_models()
    assert result[0]["name"] == "model-A"
    assert "latest_versions" in result[0]
    mock_search.assert_called_once()

@patch("src.models.model_registry.MlflowClient.search_registered_models", side_effect=Exception("Boom"))
def test_list_mlflow_registered_models_handles_error(mock_search, registry):
    result = registry.list_mlflow_registered_models()
    assert result == []

@patch("src.models.model_registry.MlflowClient.get_model_version")
def test_get_model_version_success(mock_get, registry):
    mock_obj = MagicMock()
    mock_get.return_value = mock_obj
    result = registry.get_model_version("model-A", 1)
    assert result == mock_obj
    mock_get.assert_called_once_with(name="model-A", version=1)

@patch("src.models.model_registry.MlflowClient.get_model_version", side_effect=Exception("Not found"))
def test_get_model_version_error(mock_get, registry):
    result = registry.get_model_version("model-A", 1)
    assert result is None

@patch("src.models.model_registry.MlflowClient.search_model_versions")
def test_get_latest_model_success(mock_search, registry):
    mock_search.return_value = [make_mock_version("1"), make_mock_version("3"), make_mock_version("2")]
    latest = registry.get_lastest_model("model-A")
    assert latest.version == "3"

@patch("src.models.model_registry.MlflowClient.search_model_versions", return_value=[])
def test_get_latest_model_no_versions(mock_search, registry):
    assert registry.get_lastest_model("model-A") is None


@patch("src.models.model_registry.MlflowClient.search_model_versions", side_effect=Exception("Boom"))
def test_load_model_error(mock_search, registry):
    assert registry.load_model("model-A") is None

@patch("src.models.model_registry.MlflowClient.get_run")
@patch("src.models.model_registry.MlflowClient.search_model_versions")
def test_get_model_info_by_model_metrics_success(mock_search, mock_get_run, registry):
    mv1 = make_mock_version("2", run_id="run1")
    mv2 = make_mock_version("1", run_id="run2")
    mock_search.return_value = [mv1, mv2]
    mock_get_run.return_value.data.metrics = {"rmse": 1.23}

    result = registry.get_model_info_by_model_metrics("model-A", no_recent_versions_to_consider=1)
    assert isinstance(result, list)
    assert "metrics" in result[0]

@patch("src.models.model_registry.MlflowClient.search_model_versions", return_value=[])
def test_get_model_info_by_model_metrics_no_models(mock_search, registry):
    result = registry.get_model_info_by_model_metrics("model-A")
    assert result == []
