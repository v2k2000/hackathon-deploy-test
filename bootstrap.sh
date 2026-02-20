#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

echo
echo "[NEXT] API 키를 설정하세요:"
echo 'export OPENAI_API_KEY="sk-..."'
echo "그 다음 실행: streamlit run app.py"
