# AQI Prediction Project — Developer Makefile
# Usage: make <target>
# On Windows use: python -m make or run commands directly

PYTHON  = venv/Scripts/python
PIP     = venv/Scripts/pip
PYTEST  = venv/Scripts/pytest
UVICORN = venv/Scripts/uvicorn

.PHONY: install train tune app api test test-cov quality fetch alert drift \
        predict clean help

# ── Setup ─────────────────────────────────────────────────────
install:        ## Install all dependencies into venv
	$(PIP) install -r requirements.txt

# ── Pipeline ──────────────────────────────────────────────────
train:          ## Train the full pipeline (all models, SHAP, city comparison)
	$(PYTHON) src/realtime_aqi_pipeline.py

tune:           ## Train with Optuna hyperparameter tuning (~5 min)
	$(PYTHON) src/realtime_aqi_pipeline.py --tune

leakage:        ## Run the original GFG data-leakage demonstration
	$(PYTHON) src/aqi_pipeline.py

# ── Serving ───────────────────────────────────────────────────
app:            ## Launch the Streamlit web app
	venv/Scripts/streamlit run src/app.py

api:            ## Start the FastAPI REST endpoint (http://localhost:8000/docs)
	$(UVICORN) src.api:app --host 0.0.0.0 --port 8000 --reload

# ── Testing ───────────────────────────────────────────────────
test:           ## Run all tests
	$(PYTEST) tests/ -v

test-naqi:      ## Run NAQI formula unit tests only (fastest)
	$(PYTEST) tests/test_naqi.py -v

test-cov:       ## Run tests with coverage report
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing --cov-report=html

# ── Data & monitoring ─────────────────────────────────────────
quality:        ## Run data quality validation on the real-time CSV
	$(PYTHON) src/data_quality.py

fetch:          ## Fetch live data for all 20 cities from OpenAQ
	$(PYTHON) src/fetch_historical.py

alert:          ## Dry-run the AQI email alert check
	$(PYTHON) src/aqi_alert.py --dry-run

drift:          ## Log today's predictions and generate drift report
	$(PYTHON) src/drift_tracker.py --log --report

forecast:       ## Print hourly + seasonal forecast for Delhi
	$(PYTHON) src/forecast.py --lat 28.6139 --lng 77.2090 --plot

# ── Quick prediction ──────────────────────────────────────────
predict:        ## Predict AQI for Delhi (lat=28.6, lng=77.2)
	$(PYTHON) src/realtime_aqi_pipeline.py --predict 28.6139 77.2090

# ── Cleanup ───────────────────────────────────────────────────
clean:          ## Remove all generated model artifacts and plots
	$(PYTHON) -c "import glob, os; [os.remove(f) for f in glob.glob('outputs/rt_*') + glob.glob('outputs/figures/rt_*') + glob.glob('outputs/figures/dq_*')]"
	@echo "Cleaned generated artifacts."

# ── Help ──────────────────────────────────────────────────────
help:           ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*##' Makefile | awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
