# Raccourcis (Linux / macOS / Git Bash). Sous Windows sans make : utilisez directement `cvmatch ...`.
PY ?= python

.PHONY: help install install-light pipeline demo test lint api ui docker zip clean

help:          ## affiche cette aide
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-14s %s\n", $$1, $$2}'

install:       ## installe tout (SBERT, Gemini, UI, dev)
	$(PY) -m pip install -e ".[all]"

install-light: ## installe sans SBERT/torch (TF-IDF + règles)
	$(PY) -m pip install -e ".[dev,ui]"

pipeline:      ## pipeline complet sur Rekrute (Hugging Face)
	cvmatch pipeline

demo:          ## pipeline 100 % hors-ligne (données fictives, encodeur hashing)
	cvmatch pipeline --offline

test:          ## tests unitaires + intégration
	$(PY) -m pytest

lint:          ## ruff
	ruff check src tests ui

api:           ## API FastAPI  -> http://localhost:8000/docs
	cvmatch serve --reload

ui:            ## interface Streamlit -> http://localhost:8501
	streamlit run ui/app.py

docker:        ## API + UI en conteneurs
	docker compose up --build

zip:           ## archive dist/cv-matching.zip
	cvmatch package

clean:         ## supprime caches et sorties générées (garde les .gitkeep)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache dist
	find data/processed data/cache artifacts results -type f ! -name .gitkeep -delete
