#!/usr/bin/env sh
set -e
# If a platform injects PORT, map it to what Streamlit expects:
export STREAMLIT_SERVER_PORT="${PORT:-8501}"
# Keep your other server settings here if you prefer env over config.toml:
export STREAMLIT_SERVER_HEADLESS="true"
export STREAMLIT_SERVER_ENABLE_CORS="false"
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION="true"

exec streamlit run apps/streamlit/DataSphere.py --server.address 0.0.0.0
