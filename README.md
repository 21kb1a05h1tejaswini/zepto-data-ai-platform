# Zepto Data & AI Platform

## Project Overview
This repository implements an end-to-end AI/ML platform covering:
1. `/data_pipeline`: Raw scraping, currency enrichment (1 GBP = 105.50 INR baseline), normalized SQLite storage, and SQL/pandas parity verification.
2. `/analytics`: Profiling, EDA, outlier and skewness detection, stratified predictive classification, imbalance handling comparison, hyperparameter tuning with OOB score, fare regression, and a serialized end-to-end pipeline. Includes committed fallback `titanic.csv`.
3. `/support_assistant`: Vector ingestion into ChromaDB with `all-MiniLM-L6-v2`, LangGraph deterministic policy routing, Pydantic structured output, and containerized FastAPI `/ask` endpoint.

## Setup & Installation
```bash
pip install -r requirements.txt
