## Scenario

You’ve been tasked with building a minimal but complete MLOps pipeline for an ML model using a well-known open dataset. Your model should be trained, tracked, versioned, deployed as an API, and monitored for prediction usage.

## Technologies

Git + GitHub
DVC (optional for Iris, useful for housing)
MLflow
Docker
Flask or FastAPI
GitHub Actions
Logging module (basic); Optional: Prometheus/Grafana


mlops-housing-prediction/
├── .github/workflows/         # CI/CD pipelines
├── src/
│   ├── api/                   # FastAPI application
│   ├── models/                # ML model implementations
│   ├── data/                  # Data processing utilities
│   ├── monitoring/            # Prometheus metrics
│   └── utils/                 # Shared utilities
├── tests/                     # Test suites
├── docker/                    # Docker configurations
├── configs/                   # Configuration files
├── data/                      # Dataset storage

