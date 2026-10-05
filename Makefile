.PHONY: setup lint test data train evaluate score serve docker-build docker-run drift mlflow-ui

PYTHON = .venv/Scripts/python
PIP    = .venv/Scripts/pip

setup:
	python -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	$(PYTHON) -m pre_commit install

lint:
	$(PYTHON) -m ruff check src/ tests/ api/ app/
	$(PYTHON) -m ruff format --check src/ tests/ api/ app/

test:
	$(PYTHON) -m pytest --cov=src/churnguard --cov-report=term-missing tests/

data:
	$(PYTHON) -m churnguard.data.load

train:
	$(PYTHON) -m churnguard.models.train

evaluate:
	$(PYTHON) -m churnguard.models.evaluate

score:
	$(PYTHON) -m churnguard.models.predict --file $(FILE)

serve:
	$(PYTHON) -m uvicorn api.main:app --reload --port 8000

docker-build:
	docker build -t churnguard:latest .

docker-run:
	docker run --rm -p 8000:8000 churnguard:latest

drift:
	$(PYTHON) -m churnguard.monitoring.drift

mlflow-ui:
	$(PYTHON) -m mlflow ui --backend-store-uri mlruns
