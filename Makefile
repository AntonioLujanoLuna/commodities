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
#   make dose-response test response against warm-episode peak RONI amplitude
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
#   make exposure-data download external ASIS and SPAM crop-exposure rasters
#   make exposure-weights build outcome-independent physical exposure weights
#   make panel      run the exposure-weighted two-way fixed-effects panel
#   make climate-data download alternative climate indices for placebo treatments
#   make climate-dataset build the monthly alternative-climate-index panel
#   make surrogate-treatment run the placebo-treatment falsification stage
#   make specification-curve run the whole-year circular-shift joint timing null
#   make forecast     run the locked expanding-window forecast benchmark
#   make program-timing-null run the v2 event-study family timing null
#   make mechanism-v2 WEATHER=... YIELDS=... SUPPLY_REVISIONS=... OUTPUT=... run new-input validation
#   make report     regenerate the current-results note from the run receipts
#   make report-check verify the note still matches the receipts
#   make publication build the compact receipt bundle, scorecard and figures
#   make publication-check verify the committed publication bundle
#   make figures    regenerate the publication bundle and its three figures
#   make real-data  run every implemented real-data stage
#   make test       run the test suite
#   make check      lint, type-check and test
# ---------------------------------------------------------------------------

UV ?= uv
PY ?= $(UV) run python

.PHONY: setup data dataset raw-events adjustments universe inference placebo power dose-response macro-data macro-dataset macro-analysis fragility robustness specificity endpoint-diagnostics financial-data financial-dataset financial-analysis palm-data palm-dataset palm-analysis exposure-data exposure-weights panel climate-data climate-dataset surrogate-treatment specification-curve forecast program-timing-null mechanism-v2 report report-check publication publication-check figures real-data test lint format typecheck check help

help:
	@grep -E '^#   ' Makefile | sed 's/^#   //'

setup:
	$(UV) sync --extra dev --extra spatial
	@echo "Optional econometrics extra: $(UV) pip install -e '.[econ]'"

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

dose-response:
	$(PY) scripts/run_dose_response.py

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

exposure-data:
	$(PY) scripts/download_exposure_data.py

exposure-weights:
	$(PY) scripts/build_external_exposure.py

panel:
	$(PY) scripts/run_panel_analysis.py

climate-data:
	$(PY) scripts/download_climate_data.py

climate-dataset:
	$(PY) scripts/build_climate_dataset.py

surrogate-treatment:
	$(PY) scripts/run_surrogate_treatment.py

specification-curve:
	$(PY) scripts/run_specification_curve.py

forecast:
	$(PY) scripts/run_forecast_analysis.py

program-timing-null:
	$(PY) scripts/run_program_timing_null.py

mechanism-v2:
	$(PY) scripts/run_mechanism_validation.py --weather "$(WEATHER)" --yields "$(YIELDS)" --supply-revisions "$(SUPPLY_REVISIONS)" --output "$(OUTPUT)"

report:
	$(PY) scripts/build_report.py

report-check:
	$(PY) scripts/build_report.py --check

publication:
	$(PY) scripts/build_publication.py

publication-check:
	$(PY) scripts/build_publication.py --check

figures: publication

real-data: data dataset raw-events adjustments universe inference placebo power dose-response macro-data macro-dataset macro-analysis fragility robustness specificity endpoint-diagnostics financial-data financial-dataset financial-analysis palm-data palm-dataset palm-analysis exposure-data exposure-weights panel climate-data climate-dataset surrogate-treatment specification-curve forecast program-timing-null report publication

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
