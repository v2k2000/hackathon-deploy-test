# Hackathon Deploy Test

`배포 URL에서 버튼 클릭 -> OpenAI API 호출 -> 결과 출력` 검증용 최소 Streamlit 앱.

## 로컬 실행

```bash
cd 해커톤/hackathon-deploy-test
./bootstrap.sh
source .venv/bin/activate
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
streamlit run app.py
```

## Streamlit Cloud 배포 설정값

- Repo: `hackathon-deploy-test`
- Branch: `main`
- Main file path: `app.py`

## Secrets

```toml
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

참고 템플릿:
- `.env.example`
- `streamlit_secrets.toml.example`
