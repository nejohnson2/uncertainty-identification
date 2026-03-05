#!/bin/bash
#SBATCH --job-name=uncertainty-analysis
#SBATCH --partition=b40x4
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# ── Environment Setup ────────────────────────────────────────────────
module load cuda12.8/toolkit/12.8.0

# Activate conda environment
conda activate uncertainty

# Set HuggingFace cache
export HF_HOME=/lustre/nvwulf/scratch/nijjohnson/hf_cache

# Set working directory
cd /lustre/nvwulf/scratch/nijjohnson/uncertainty-identification

# ── Run Pipeline ─────────────────────────────────────────────────────
echo "=== Starting full analysis pipeline ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"

# Step 1: Download dataset (if not already cached)
echo "--- Step 1: Download dataset ---"
python src/data_loading.py

# Step 2: EDA
echo "--- Step 2: EDA ---"
python scripts/01_eda.py

# Step 3: Lexicon analysis
echo "--- Step 3: Build lexicon ---"
python scripts/02_build_lexicon.py

# Step 4: Topic modeling (uses GPU for embeddings)
echo "--- Step 4: Topic modeling ---"
python scripts/04_topic_modeling.py

# Step 5: Positional analysis (CPU-intensive)
echo "--- Step 5: Positional analysis ---"
python scripts/05_position_analysis.py

# Step 6: Confidence filtering
echo "--- Step 6: Confidence filtering ---"
python scripts/06_confidence_filtering.py

# Step 7: Statistical models
echo "--- Step 7: Statistical models ---"
python scripts/07_statistical_models.py

# Step 8: Visualizations
echo "--- Step 8: Visualizations ---"
python scripts/08_visualizations.py

echo "=== Pipeline complete ==="
echo "Time: $(date)"
