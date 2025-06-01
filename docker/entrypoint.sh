#!/bin/bash
set -e  # Exit on first error

export PYTHONPATH=src

echo ">> Running vectorization pipeline..."
python src/scripts/run_vectorize_pipeline.py

echo ">> Launching Chainlit app..."
exec chainlit run src/application/app.py -w --host 0.0.0.0 --port 8000
