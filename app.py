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

SYSTEM_PROMPT = """
너는 스타트업 전략/제품 설계 전문가다.
출력은 반드시 3×3 테이블(마크다운) + 요약 블록이어야 한다.
각 셀은 2~4문장으로 작성하고, 실전적이고 검증 가능하게 제시한다.
허세는 금지하되, 구조는 단단하게 유지한다.
""".strip()

USER_PROMPT_TEMPLATE = """
너는 ‘문제 정위(定位) 프레임워크’를 사용해 문제를 9칸으로 구조화하는 전략 설계자다.
입력된 문제와 타겟을 바탕으로 아래 3×3 격자(진단/설계/실행 × 사용자/해결/비즈니스)를 채워라.

규칙:
- 각 셀은 2~4문장. 추상어 대신 구체적 관찰/가설/검증을 제시.
- “대안/비교/트레이드오프”를 최소 1회 이상 언급(차별성 강화).
- 마지막에 요약 블록으로: MVP 기능 3개, 수익모델 2개, 가장 큰 리스크 3개를 bullet로 제시.

출력 형식:
1) 먼저 3×3 마크다운 표
2) 그 다음 "## 요약 블록" 헤더와 불릿

9칸 라벨(고정):
- (진단×사용자) 사용자 증상
- (진단×해결) 문제 메커니즘(원인/구조)
- (진단×비즈니스) 시장 신호(왜 지금?)
- (설계×사용자) 타겟 & JTBD
- (설계×해결) 핵심 가치/차별점
- (설계×비즈니스) 수익모델 가설
- (실행×사용자) 채널/획득 전략
- (실행×해결) MVP 기능 3개
- (실행×비즈니스) 리스크 & 검증 실험

[입력]
문제: {problem}
타겟 사용자: {target}
""".strip()


def split_table_and_summary(content: str) -> tuple[str, str]:
    markers = ["## 요약 블록", "### 요약 블록", "요약 블록"]
    for marker in markers:
        idx = content.find(marker)
        if idx > 0:
            return content[:idx].strip(), content[idx:].strip()
    return content.strip(), ""


st.set_page_config(page_title="Structo AI", layout="centered")
st.title("Structo AI")
st.caption("문제를 9개의 관점으로 정위(定位)해 전략으로 바꿉니다.")

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

problem = st.text_area(
    "문제(Problem)",
    value="새로운 서비스 아이디어는 많은데, 어떤 문제부터 풀어야 할지 팀 내 합의가 자주 흔들린다.",
    help="2~6문장 권장",
    height=140,
)
target = st.text_area(
    "타겟 사용자(Target user)",
    value="초기 제품을 빠르게 검증해야 하는 1~5인 스타트업 팀.",
    help="1~2문장 권장",
    height=100,
)

view_mode = st.radio(
    "출력 보기",
    options=["표+요약 보기", "표만 보기"],
    horizontal=True,
)
model = st.text_input("모델명", value="gpt-4.1-mini")

if st.button("9칸 구조 생성"):
    if not problem.strip() or not target.strip():
        st.warning("문제와 타겟 사용자 입력을 모두 채워주세요.")
        st.stop()

    user_prompt = USER_PROMPT_TEMPLATE.format(
        problem=problem.strip(),
        target=target.strip(),
    )
    try:
        with st.spinner("호출 중..."):
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
            )
        st.success("생성 완료")
        content = response.choices[0].message.content or ""
        table_md, summary_md = split_table_and_summary(content)

        st.markdown("### 3×3 전략 프레임워크")
        st.caption("축: 행(진단/설계/실행) × 열(사용자/해결/비즈니스)")
        st.markdown(table_md)

        if view_mode == "표+요약 보기":
            if summary_md:
                st.markdown("---")
                st.markdown(summary_md)
            else:
                st.info("요약 블록이 감지되지 않았습니다. 원문을 확인하세요.")

        with st.expander("원문 응답 보기"):
            st.markdown(content)
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
