# Zepto Data & AI Platform (Capstone Submission)

A unified platform combining an automated data scraping pipeline, predictive machine learning analytics, and a LangGraph-orchestrated GenAI customer support microservice.
## 1. Environment Setup

All modules share a single consolidated dependencies file at the repository root:

git clone https://github.com/21kb1a05h1tejaswini/zepto-data-ai-platform.git
cd zepto-data-ai-platform
pip install -r requirements.txt
## 2. Module 1: Data Pipeline (/data_pipeline)

### Execution
python data_pipeline/run_pipeline.py

### Parsing & Cleaning Decisions
- Exchange Rate Baseline: Currency conversion from GBP to INR strictly follows the required fixed-rate baseline: 1 GBP = 105.50 INR (constant, project-defined).
- Type Coercion: Handled price_gbp (float), price_inr (float), rating (integer 1-5 mapped from CSS star classes), and in_stock (boolean).
- Relational Schema: Created SQLite database zepto_catalog.db with primary key / foreign key relations across categories and books tables.
- SQL & Parity Verification: 5 SQL queries executed (DISTINCT/ORDER BY, BETWEEN/LIMIT, WHERE IN, GROUP BY, and relational JOIN). A side-by-side parity check confirmed identical records between the SQLite relational JOIN and pandas.merge(books, categories, on='category_id').

## 3. Module 2: Analytics Pipeline (/analytics)

### Execution
python analytics/run_analytics.py

### Key Decisions & Written Interpretations
- Offline Fallback: titanic.csv is committed inside /analytics so the script executes deterministically via pd.read_csv("titanic.csv") without network dependencies.
- Missing Value Strategy:
  - Embarked (<5% missing): Imputed with mode or dropped.
  - Age (5%-30% missing): Imputed with median.
  - Cabin (>75% missing): Imputation would be unreliable; dropped or encoded with a distinct missing indicator.
- Outliers & Skewness:
  - Outliers identified via IQR rule ([Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]) for Age and Fare.
  - Fare distribution is right-skewed, verified by the relationship: Mean > Median > Mode.
- Imbalance Handling: Evaluated Baseline vs class_weight='balanced' vs SMOTE. SMOTE was applied strictly to the training fold to prevent data leakage.
- Hyperparameter Tuning & Regression:
  - RandomForestClassifier(oob_score=True) tuned using GridSearchCV.
  - Multivariate regression on Fare evaluates MAE, RMSE, R2, and Adjusted R2. Residual analysis indicates heteroscedasticity due to higher residual variance at higher fares.
- Artifact: Saved complete fitted preprocessor and estimator pipeline to analytics/model_pipeline.joblib.
## 4. Module 3: Support Assistant (/support_assistant)

### Execution (FastAPI / Uvicorn)
cd support_assistant
uvicorn main:app --host 0.0.0.0 --port 7860

### Docker Execution
cd support_assistant
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant

### RAG Architecture Walkthrough
1. Ingestion: 8 Zepto policy files (docs/doc_01.txt to docs/doc_08.txt) loaded into memory.
2. Embedding: Local dense vector embedding via sentence-transformers (all-MiniLM-L6-v2) at no API cost.
3. Storage: Stored in ChromaDB collection (zepto_policies).
4. LangGraph Flow:
   - classify_intent: Gated by MOCK_LLM. Default mock mode (MOCK_LLM=1) classifies queries using keyword heuristics (delivery, return, refund, tracking, cancel, gift card, support hours, membership).
   - retrieve_and_answer: Queries ChromaDB for top-3 relevant documents via cosine similarity and returns grounded context.
   - direct_answer: Handles general chit-chat queries with a canned refusal.
5. Schema: Pydantic model validates output: {"answer": str, "sources": list[str], "confidence": float}.

### Example Run Transcripts (MOCK_LLM=1)

Example 1: Policy Question (Retrieval Triggered)
Request: POST /ask with {"query": "What is the delivery fee for orders under 149?"}
Response:
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation... Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee.",
  "sources": ["doc_01", "doc_05", "doc_03"],
  "confidence": 1.0
}

Example 2: General Question (Direct Answer, No Retrieval)
Request: POST /ask with {"query": "Hello, how are you today?"}
Response:
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
