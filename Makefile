# ---------------------------------------------------------------------------
# ENSO and commodity markets -- reproducible pipeline.
#
#   make setup      install the package and its development tools
#   make data       download a dated real-data snapshot
#   make dataset    build the clean monthly ENSO x commodity panel
#   make raw-events derive ENSO episodes and raw event returns
#   make adjustments build seasonal and common-market adjusted returns
#   make universe    freeze the primary inference family and controls
#   make inference   bootstrap the frozen endpoint and apply candidate FDR
#   make placebo     run calendar-matched neutral-date placebo inference
#   make macro-data  download a dated real-data macro snapshot
#   make macro-dataset build the monthly external-control panel
#   make macro-analysis run macro-adjusted bootstrap and placebo inference
#   make power       estimate the minimum detectable effect at the frozen endpoint
#   make fragility   run leave-one-episode-out bootstrap/FDR checks
#   make robustness  run RONI/ONI and retrospective/observable robustness
#   make specificity run exploratory warm-versus-cold falsification diagnostics
#   make endpoint-diagnostics test convexity, time drift and observed macro regimes
#   make financial-data download real rates, credit spreads and financial conditions
#   make financial-dataset build the monthly financial-control panel
#   make financial-analysis run exploratory extended-control inference
#   make palm-data   download FAOSTAT and NASA POWER palm-oil inputs
#   make palm-dataset build the palm production/weather panel
#   make palm-analysis test the exploratory physical mechanism chain
#   make panel      run the exposure-weighted two-way fixed-effects panel
#   make report     regenerate the current-results note from the run receipts
#   make report-check verify the note still matches the receipts
#   make real-data  run every implemented real-data stage
#   make test       run the test suite
#   make check      lint, type-check and test
# ---------------------------------------------------------------------------

UV ?= uv
PY ?= $(UV) run python

.PHONY: setup data dataset raw-events adjustments universe inference placebo power macro-data macro-dataset macro-analysis fragility robustness specificity endpoint-diagnostics financial-data financial-dataset financial-analysis palm-data palm-dataset palm-analysis panel report report-check real-data test lint format typecheck check help

help:
	@grep -E '^#   ' Makefile | sed 's/^#   //'

setup:
	$(UV) sync --extra dev
	@echo "Optional extras: $(UV) pip install -e '.[spatial,econ]'"

data:
	$(PY) scripts/download_data.py

dataset:
	$(PY) scripts/build_dataset.py

raw-events:
	$(PY) scripts/build_raw_events.py

adjustments:
	$(PY) scripts/build_adjusted_events.py

universe:
	$(PY) scripts/build_universe.py

inference:
	$(PY) scripts/run_inference.py

placebo:
	$(PY) scripts/run_placebo.py

power:
	$(PY) scripts/run_power.py

macro-data:
	$(PY) scripts/download_macro_data.py

macro-dataset:
	$(PY) scripts/build_macro_dataset.py

macro-analysis:
	$(PY) scripts/run_macro_analysis.py

fragility:
	$(PY) scripts/run_fragility.py

robustness:
	$(PY) scripts/run_robustness.py

specificity:
	$(PY) scripts/run_specificity.py

endpoint-diagnostics:
	$(PY) scripts/run_endpoint_diagnostics.py

financial-data:
	$(PY) scripts/download_financial_data.py

financial-dataset:
	$(PY) scripts/build_financial_dataset.py

financial-analysis:
	$(PY) scripts/run_financial_analysis.py

palm-data:
	$(PY) scripts/download_palm_oil_data.py

palm-dataset:
	$(PY) scripts/build_palm_oil_dataset.py

palm-analysis:
	$(PY) scripts/run_palm_oil_mechanism.py

panel:
	$(PY) scripts/run_panel_analysis.py

report:
	$(PY) scripts/build_report.py

report-check:
	$(PY) scripts/build_report.py --check

real-data: data dataset raw-events adjustments universe inference placebo power macro-data macro-dataset macro-analysis fragility robustness specificity endpoint-diagnostics financial-data financial-dataset financial-analysis palm-data palm-dataset palm-analysis panel report

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests scripts

format:
	$(PY) -m ruff format src tests scripts
	$(PY) -m ruff check --fix src tests scripts

typecheck:
	$(PY) -m mypy src

check: lint typecheck test
