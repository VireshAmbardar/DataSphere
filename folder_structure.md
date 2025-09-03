askmydb/
├─ apps/
│  └─ streamlit/                  # UI-only; imports from core
│     ├─ app.py                   # main entry: sidebar > connect > ask > report
│     ├─ pages/
│     │  ├─ 1_🔌_Connect_DB.py
│     │  ├─ 2_❓_Ask_Questions.py
│     │  ├─ 3_📊_Reports.py
│     │  └─ 4_⚙️_Settings.py
│     └─ components/              # small UI widgets
│        ├─ connection_form.py
│        ├─ schema_browser.py
│        └─ chart_viewer.py
├─ core/                          # all business logic (framework-agnostic)
│  ├─ __init__.py
|  |─ global_settings.py
|  |─ upload_processing/
│  │  ├─ upload_files.py          # processing uploaded files
│  ├─ config/
│  │  ├─ settings.py              # pydantic settings (LLM keys, limits)
│  │  └─ prompts/                 # few-shot, system prompts, guardrails msgs
│  ├─ db/
│  │  ├─ connectors.py            # SQLAlchemy engines for Postgres/MySQL/…/MSSQL
│  │  ├─ introspect.py            # tables/columns/foreign-keys + samples
│  │  └─ exec.py                  # safe query execution, LIMITs, pagination
│  ├─ llm/
│  │  ├─ providers.py             # OpenAI-compatible client factory
│  │  ├─ langchain_sql.py         # LangChain SQL chain setup
│  │  ├─ llamaindex_sql.py        # LlamaIndex SQL query engine
│  │  ├─ router.py                # choose LC vs LI; retry/fallback logic
│  │  └─ guardrails.py            # read-only SQL checks, denylist, LLM critique
│  │  └─ Retrivers/            # read-only SQL checks, denylist, LLM critique
│  │    └─ text_embedding_retrivals.py            # read-only SQL checks, denylist, LLM critique
│  ├─ nlp/
│  │  ├─ rephrase.py              # clarifying question, fill-in missing filters
│  │  └─ column_synonyms.py       # mapping “revenue”->table.column using stats
│  ├─ reporting/
│  │  ├─ suggest_charts.py        # auto chart-type heuristics from dataframe
│  │  ├─ build_narrative.py       # LLM narrative summary over results
│  │  ├─ exporters.py             # CSV/XLSX/HTML/PDF export
│  │  └─ fig_factory.py           # plotly figures from dataframe + spec
│  ├─ cache/
│  │  ├─ memory_cache.py          # LRU for schema + query results
│  │  └─ disk_cache.py            # optional disk cache (for dev)
│  └─ utils/
│     ├─ sql_safety.py            # sqlglot parsing, SELECT-only, EXPLAIN checks
│     ├─ sampling.py              # fetch top-N values for columns (prompting aid)
│     └─ logging.py               # structlog/loguru setup
│     └─ chromadb.py               # structlog/loguru setup
├─ tests/
│  ├─ test_sql_safety.py
│  ├─ test_router.py
│  ├─ test_suggest_charts.py
│  └─ fixtures/
│     └─ demo.sqlite              # tiny demo DB for unit tests
├─ docs/
│  ├─ architecture.md
│  ├─ prompts.md
│  └─ threat_model.md
├─ .env.example                   # LLM keys, defaults, safe limits
├─ pyproject.toml                 # or poetry.lock/uv.lock
├─ Dockerfile
└─ docker-compose.yaml            # optional: local Postgres for dev
