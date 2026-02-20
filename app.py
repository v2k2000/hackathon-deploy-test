import os
import streamlit as st
from openai import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

st.set_page_config(page_title="Deploy Test", layout="centered")
st.title("OpenAI API 배포 테스트")

# Streamlit Cloud: st.secrets["OPENAI_API_KEY"]
# Local: environment variable OPENAI_API_KEY
api_key = None
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY가 없습니다. Streamlit secrets 또는 환경변수를 설정하세요.")
    st.stop()

client = OpenAI(api_key=api_key)

text = st.text_input("아무 문장 입력", value="안녕! 배포 테스트 중이야.")
model = st.text_input("모델명", value="gpt-4.1-mini")

if st.button("API 호출"):
    try:
        with st.spinner("호출 중..."):
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": f"다음 문장을 1문장으로 더 짧게 요약: {text}",
                    }
                ],
            )
        st.success("성공")
        st.write(response.choices[0].message.content)
    except AuthenticationError as e:
        st.error("인증 실패: API 키가 잘못됐거나 만료되었습니다.")
        st.code(str(e))
    except RateLimitError as e:
        st.error("요청 제한/크레딧 부족 오류입니다. 결제/한도 상태를 확인하세요.")
        st.code(str(e))
    except BadRequestError as e:
        st.error("요청 형식 오류입니다. 모델명 또는 입력값을 확인하세요.")
        st.code(str(e))
    except APIConnectionError as e:
        st.error("OpenAI 서버 연결 오류입니다. 네트워크 상태를 확인하세요.")
        st.code(str(e))
    except APIError as e:
        st.error("OpenAI API 오류입니다.")
        st.code(str(e))
    except Exception as e:
        st.error("알 수 없는 오류가 발생했습니다.")
        st.code(str(e))
