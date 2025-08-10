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

### Prerequisites
- Python 3.12+
- Docker & Docker Compose
- Git

### Clone the repo for local testing
git clone https://github.com/nairanit25/MTechAssignments.git
cd MTechAssignments

### Create python env
a) Install python 3.12.2 (3.12+)
b) python3 -m venv mlops-assign1-venv
c)  .\mlops-assign1-venv\Scripts\activate  #It will activate the env.

### Install dependencies
pip install -r requirements.txt

### Docker container creation
a) install docker and start it as admin
b) docker-compose up --build --force-recreate #It will force build containers in docker
docker-compose up -d  #It will bring up the containers in docker

### train the model
###### Train linear_regression and Register all experiments in MLFlow under linear_regression_housing_price_predictor
a) python.exe -m src.train.train_models --data-path .\data\california_housing.csv --algorithms  'linear_regression'
###### Train decision_tree and Register all experiments in MLFlow under decision_tree_housing_price_predictor
b) python.exe -m src.train.train_models --data-path .\data\california_housing.csv --algorithms  'decision_tree'

### Select the best model and register it as housing_price_predictor
a) python.exe -m src.models.best_model_selector

### Run the application
a) Restart the docker-container api-app and it will load the model for prediction.

### MLFlow Dashboard
a) http://localhost:5000/

