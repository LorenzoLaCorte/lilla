#!/bin/sh
# Use the active Python environment. Hosts supply PORT; local defaults stay private.
set -eu
cd "$(dirname "$0")/../.."
exec python -m streamlit run src/visualization/app.py \
    --server.address "${LILLA_BIND_ADDRESS:-127.0.0.1}" \
    --server.port "${PORT:-8501}" \
    --server.headless true \
    --browser.gatherUsageStats false
