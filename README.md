# Structo AI (MVP v0.1)

문제/타겟 입력을 받아 `3×3 전략 프레임워크(진단/설계/실행 × 사용자/해결/비즈니스)`와
요약 블록(MVP 기능 3개, 수익모델 2개, 리스크 3개)을 생성하는 Streamlit 앱.

## 로컬 실행

```bash
cd 해커톤/hackathon-deploy-test
./bootstrap.sh
source .venv/bin/activate
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY_VALUE"
streamlit run app.py
```

## Streamlit Cloud 배포 설정값

- Repo: `hackathon-deploy-test`
- Branch: `main`
- Main file path: `app.py`

## Secrets

```toml
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY_VALUE"
```

참고 템플릿:
- `.env.example`
- `streamlit_secrets.toml.example`
