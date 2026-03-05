# From Hedge to Answer: Uncertainty Dynamics in LLM Reasoning Chains

An empirical study of how large language models express uncertainty during chain-of-thought reasoning, and whether that uncertainty survives into the final response.

## Research Questions

1. **Positional distribution** — How is uncertainty distributed across positions in reasoning chains? Is it concentrated early (hypothesis generation), mid (evaluation), or late (near conclusions)?
2. **Cross-model comparison** — Do different model families (Claude, GPT, Gemini, DeepSeek, GLM, etc.) differ systematically in their reasoning-stage uncertainty patterns?
3. **Confidence filtering** — Does uncertainty present in reasoning traces get suppressed in the final response? Which types of uncertainty survive?

## Dataset

[Solenopsisbot/real-slop](https://huggingface.co/datasets/Solenopsisbot/real-slop) — 155,623 real LLM interactions across 104 model variants (MIT license).

| Statistic | Value |
|-----------|-------|
| Total rows | 155,623 |
| Rows with reasoning traces | 11,366 (7.3%) |
| Rows with both reasoning + response | 10,961 |
| Unique models | 104 |
| Model families with reasoning | Claude Opus, GPT, Gemini, GLM, DeepSeek, MiniMax, Qwen |

## Methods

- **Uncertainty detection**: Multi-category lexical taxonomy (epistemic hedges, evidential markers, explicit uncertainty, probability language, modal hedging, approximators) with POS-aware matching via spaCy, validated by LLM-based classification
- **Topic modeling**: BERTopic on user prompts
- **Statistical analysis**: Mixed-effects logistic regression with position, model family, topic, and NSFW flag as predictors; Wilcoxon signed-rank tests for confidence filtering

## Setup

Requires Python 3.13 (spaCy has compatibility issues with 3.14).

```bash
# Create virtual environment and install dependencies
make setup

# Download dataset (~2.4 GB)
make download
```

## Usage

### Run individual pipeline stages

```bash
make eda          # Exploratory data analysis
make lexicon      # Build and validate uncertainty lexicon
make classify     # LLM-based uncertainty classification (requires ANTHROPIC_API_KEY in .env)
make topics       # BERTopic topic modeling (GPU recommended)
make position     # Core positional uncertainty analysis
make filtering    # Confidence filtering: reasoning vs response
make models       # Mixed-effects regression models
make visualize    # Generate publication-ready figures
```

### Composite targets

```bash
make dev          # Quick pipeline: EDA + lexicon
make all          # Full pipeline: all stages
make clean        # Remove generated results
```

### Cluster deployment (NVWulf)

```bash
sbatch slurm/run_full_analysis.sh
```

## Project Structure

```
├── src/
│   ├── data_loading.py            # Download & load dataset via HuggingFace
│   ├── preprocessing.py           # Filtering, cleaning, model family extraction
│   ├── uncertainty_lexicon.py     # Uncertainty taxonomy & POS-aware detection
│   ├── uncertainty_classifier.py  # LLM-based uncertainty classification
│   ├── topic_modeling.py          # BERTopic topic classification
│   ├── position_analysis.py      # Positional uncertainty analysis
│   ├── confidence_filtering.py   # Reasoning vs response comparison
│   └── statistical_models.py     # Mixed-effects regression
├── scripts/
│   ├── 01_eda.py                  # Exploratory data analysis
│   ├── 02_build_lexicon.py        # Build & validate uncertainty lexicon
│   ├── 03_classify_uncertainty.py # LLM classifier + agreement metrics
│   ├── 04_topic_modeling.py       # BERTopic on user prompts
│   ├── 05_position_analysis.py   # Core positional analysis
│   ├── 06_confidence_filtering.py # Reasoning vs response
│   ├── 07_statistical_models.py  # Regression modeling
│   └── 08_visualizations.py      # All figures (reads saved results only)
├── tests/
│   └── test_uncertainty.py        # Unit tests for uncertainty detection
├── results/
│   ├── tables/                    # Saved statistics & tables
│   └── figures/                   # Publication-ready plots
├── slurm/
│   └── run_full_analysis.sh       # SLURM script for GPU cluster
├── Makefile
└── requirements.txt
```

## Uncertainty Taxonomy

| Category | Examples |
|----------|----------|
| Epistemic hedges | "I think", "perhaps", "maybe", "I believe" |
| Evidential markers | "it seems", "it appears", "apparently" |
| Explicit uncertainty | "I'm not sure", "I may be mistaken" |
| Probability language | "likely", "probably", "possibly" |
| Modal hedging | "might", "could" (POS-verified as auxiliaries) |
| Approximators | "approximately", "roughly", "about N" |

## Running Tests

```bash
.venv/bin/pytest tests/ -v
```
