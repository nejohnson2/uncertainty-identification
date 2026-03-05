.PHONY: setup download dev all eda lexicon classify topics position filtering models visualize tables paper paper-clean clean

PYTHON = .venv/bin/python
SAMPLE_FRAC = 0.05
OLLAMA_MODEL = qwen2.5:7b

# ─── Setup ───────────────────────────────────────────────────────────

setup:
	python3.13 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m spacy download en_core_web_sm

download:
	$(PYTHON) src/data_loading.py

# ─── Analysis Pipeline ───────────────────────────────────────────────

eda:
	$(PYTHON) scripts/01_eda.py

lexicon:
	$(PYTHON) scripts/02_build_lexicon.py

classify:
	$(PYTHON) scripts/03_classify_uncertainty.py --model $(OLLAMA_MODEL)

topics:
	$(PYTHON) scripts/04_topic_modeling.py

position:
	$(PYTHON) scripts/05_position_analysis.py

filtering:
	$(PYTHON) scripts/06_confidence_filtering.py

models:
	$(PYTHON) scripts/07_statistical_models.py

visualize:
	$(PYTHON) scripts/08_visualizations.py

# ─── Paper Generation ──────────────────────────────────────────────

tables:
	$(PYTHON) scripts/09_generate_latex_tables.py

paper: tables
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

paper-clean:
	cd paper && latexmk -C

# ─── Composite Targets ──────────────────────────────────────────────

dev: eda lexicon
	@echo "Dev pipeline complete (EDA + lexicon on full data)"

all: eda lexicon classify topics position filtering models visualize tables paper
	@echo "Full pipeline + paper complete"

clean:
	rm -rf results/tables/*.csv results/figures/*.png results/figures/*.pdf paper/tables/*.tex
	@echo "Cleaned results"
