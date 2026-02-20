import html
import json
import os
import re

import streamlit as st
from openai import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

CELL_DEFS = [
    {
        "id": "diag_user",
        "row": "진단",
        "col": "사용자",
        "label": "사용자 증상",
    },
    {
        "id": "diag_solution",
        "row": "진단",
        "col": "해결",
        "label": "문제 메커니즘(원인/구조)",
    },
    {
        "id": "diag_business",
        "row": "진단",
        "col": "비즈니스",
        "label": "시장 신호(왜 지금?)",
    },
    {
        "id": "design_user",
        "row": "설계",
        "col": "사용자",
        "label": "타겟 & JTBD",
    },
    {
        "id": "design_solution",
        "row": "설계",
        "col": "해결",
        "label": "핵심 가치/차별점",
    },
    {
        "id": "design_business",
        "row": "설계",
        "col": "비즈니스",
        "label": "수익모델 가설",
    },
    {
        "id": "exec_user",
        "row": "실행",
        "col": "사용자",
        "label": "채널/획득 전략",
    },
    {
        "id": "exec_solution",
        "row": "실행",
        "col": "해결",
        "label": "MVP 기능 3개",
    },
    {
        "id": "exec_business",
        "row": "실행",
        "col": "비즈니스",
        "label": "리스크 & 검증 실험",
    },
]
CELL_IDS = [item["id"] for item in CELL_DEFS]

SYSTEM_PROMPT = """
너는 스타트업 전략/제품 설계 전문가다.
출력은 실전성, 검증 가능성, 우선순위가 느껴지도록 작성한다.
허세는 금지하고, 명확한 가설과 실행 단위를 제시한다.
문체는 군더더기 없이 함축적 요약 어조로 유지한다.
응답은 JSON만 반환한다.
""".strip()


def inject_style() -> None:
    st.markdown(
        """
<style>
:root {
  --background-color: #f3f4f6;
  --secondary-background-color: #ffffff;
  --text-color: #111827;
}

html, body, [class*="st-"] {
  font-family: "Noto Sans KR", "Apple SD Gothic Neo", "Segoe UI", sans-serif;
  color: #111827;
}

.stApp {
  background: #f3f4f6;
}

.block-container {
  max-width: 1120px;
  padding-top: 1.2rem;
}

h1, h2, h3, h4, p, span, label {
  color: #111827 !important;
}

.stTextInput input,
.stTextArea textarea,
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] > div,
div[data-baseweb="select"] > div {
  background: #ffffff !important;
  color: #111827 !important;
  border: 1px solid #cbd5e1 !important;
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
  color: #6b7280 !important;
}

input,
textarea {
  color: #111827 !important;
  -webkit-text-fill-color: #111827 !important;
}

div[data-baseweb="select"] *,
ul[role="listbox"] *,
li[role="option"] * {
  color: #111827 !important;
}

ul[role="listbox"],
li[role="option"] {
  background: #ffffff !important;
}

div[role="radiogroup"] label {
  background: #ffffff !important;
  border: 1px solid #cbd5e1 !important;
  border-radius: 8px !important;
  padding: 2px 8px !important;
}

div[role="radiogroup"] label p {
  color: #111827 !important;
}

.axis-note {
  background: #ffffff;
  border: 1px solid #cbd5e1;
  border-radius: 10px;
  padding: 10px 12px;
  margin: 4px 0 14px 0;
  font-size: 0.9rem;
}

.control-title {
  margin-top: 12px;
  margin-bottom: 6px;
  font-weight: 700;
  color: #111827;
}

.cell-preview {
  background: #ffffff;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  padding: 10px 12px;
  min-height: 120px;
  color: #111827;
  font-size: 0.9rem;
  line-height: 1.45;
}

.nine-grid {
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  overflow: hidden;
  background: #ffffff;
}

.nine-grid table {
  border-collapse: collapse;
  width: 100%;
  margin: 0;
}

.nine-grid th {
  background: #1f2937;
  color: #f9fafb;
  border: 1px solid #1f2937;
  padding: 10px;
  text-align: left;
}

.nine-grid td {
  background: #ffffff;
  border: 1px solid #cbd5e1;
  padding: 10px;
  vertical-align: top;
}

.nine-grid td.changed {
  background: #fff7d6;
  border: 2px solid #d97706;
}

.changed-chip {
  display: inline-block;
  background: #fff7d6;
  border: 1px solid #d97706;
  border-radius: 999px;
  color: #92400e;
  padding: 3px 10px;
  margin: 2px 6px 2px 0;
  font-size: 0.82rem;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def read_api_key() -> str | None:
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return os.getenv("OPENAI_API_KEY")


def build_initial_prompt(problem: str, target: str, global_feedback: str) -> str:
    return f"""
아래 입력을 기준으로 3×3 프레임워크를 작성하라.

반드시 JSON만 반환하고 코드블록/설명문을 붙이지 마라.

JSON 스키마:
{{
  "cells": {{
    "diag_user": "2~3문장",
    "diag_solution": "2~3문장",
    "diag_business": "2~3문장",
    "design_user": "2~3문장",
    "design_solution": "2~3문장",
    "design_business": "2~3문장",
    "exec_user": "2~3문장",
    "exec_solution": "2~3문장",
    "exec_business": "2~3문장"
  }},
  "summary": {{
    "mvp_features": ["항목1", "항목2", "항목3"],
    "revenue_models": ["항목1", "항목2"],
    "top_risks": ["항목1", "항목2", "항목3"]
  }}
}}

규칙:
- 각 셀은 구체적 관찰/가설/검증을 포함해 2~3문장으로 작성.
- 문장은 짧게 끊고 함축적 요약 어조를 유지.
- 전체 9셀 중 최소 1회는 대안/비교/트레이드오프를 명시.
- 과장된 표현 대신 실행 가능한 문장으로 작성.
- 요약 블록 각 항목은 1문장으로 간결히 작성.

입력:
문제: {problem}
타겟 사용자: {target}
전체 피드백: {global_feedback if global_feedback else "(없음)"}
""".strip()


def build_refine_prompt(
    problem: str,
    target: str,
    current_cells: dict[str, str],
    controls: dict[str, dict[str, str]],
    global_feedback: str,
) -> str:
    return f"""
아래 3×3 프레임워크를 셀별 제어 조건에 따라 업데이트하라.

반드시 JSON만 반환하고 코드블록/설명문을 붙이지 마라.

JSON 스키마:
{{
  "cells": {{
    "diag_user": "문자열",
    "diag_solution": "문자열",
    "diag_business": "문자열",
    "design_user": "문자열",
    "design_solution": "문자열",
    "design_business": "문자열",
    "exec_user": "문자열",
    "exec_solution": "문자열",
    "exec_business": "문자열"
  }},
  "summary": {{
    "mvp_features": ["항목1", "항목2", "항목3"],
    "revenue_models": ["항목1", "항목2"],
    "top_risks": ["항목1", "항목2", "항목3"]
  }}
}}

모드 규칙:
- mode="고정": 기존 셀 내용을 유지.
- mode="입력": manual_input이 있으면 그 내용을 우선 반영해 2~3문장으로 정리.
- mode="비우기": 빈 문자열("")로 반환.
- cell_feedback, 전체 피드백을 반영해 품질을 높여라.
- 문체는 함축적 요약 톤으로 유지.

입력:
문제: {problem}
타겟 사용자: {target}
전체 피드백: {global_feedback if global_feedback else "(없음)"}
현재 셀 JSON: {json.dumps(current_cells, ensure_ascii=False)}
셀 제어 JSON: {json.dumps(controls, ensure_ascii=False)}
""".strip()


def parse_json_response(raw_text: str) -> dict:
    text = raw_text.strip()
    code_fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if code_fence_match:
        text = code_fence_match.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 파싱 실패: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("응답 JSON 루트는 객체여야 합니다.")
    return data


def normalize_list(value: object, expected_size: int) -> list[str]:
    if not isinstance(value, list):
        return [""] * expected_size
    cleaned = [str(item).strip() for item in value if str(item).strip()]
    while len(cleaned) < expected_size:
        cleaned.append("")
    return cleaned[:expected_size]


def normalize_result(data: dict, fallback_cells: dict[str, str] | None = None) -> dict:
    fallback_cells = fallback_cells or {}
    raw_cells = data.get("cells", {})
    cells: dict[str, str] = {}
    for cell_id in CELL_IDS:
        value = ""
        if isinstance(raw_cells, dict) and cell_id in raw_cells:
            value = str(raw_cells[cell_id]).strip()
        else:
            value = str(fallback_cells.get(cell_id, "")).strip()
        cells[cell_id] = value

    raw_summary = data.get("summary", {})
    summary = {
        "mvp_features": normalize_list(
            raw_summary.get("mvp_features", []) if isinstance(raw_summary, dict) else [],
            3,
        ),
        "revenue_models": normalize_list(
            raw_summary.get("revenue_models", []) if isinstance(raw_summary, dict) else [],
            2,
        ),
        "top_risks": normalize_list(
            raw_summary.get("top_risks", []) if isinstance(raw_summary, dict) else [],
            3,
        ),
    }
    return {"cells": cells, "summary": summary}


def fallback_summary_from_cells(cells: dict[str, str]) -> dict[str, list[str]]:
    def to_lines(text: str, n: int) -> list[str]:
        fragments = [
            item.strip(" -•\t")
            for item in re.split(r"[\n;]", text)
            if item.strip(" -•\t")
        ]
        if not fragments and text.strip():
            fragments = [text.strip()]
        while len(fragments) < n:
            fragments.append("")
        return fragments[:n]

    return {
        "mvp_features": to_lines(cells.get("exec_solution", ""), 3),
        "revenue_models": to_lines(cells.get("design_business", ""), 2),
        "top_risks": to_lines(cells.get("exec_business", ""), 3),
    }


def render_html_table(cells: dict[str, str], highlight_ids: set[str] | None = None) -> str:
    cell_map = {item["id"]: item for item in CELL_DEFS}
    row_ids = [
        ["diag_user", "diag_solution", "diag_business"],
        ["design_user", "design_solution", "design_business"],
        ["exec_user", "exec_solution", "exec_business"],
    ]
    highlight_ids = highlight_ids or set()
    rows = [
        "<div class='nine-grid'>",
        "<table>",
        "<thead>",
        "<tr><th>단계</th><th>사용자</th><th>해결</th><th>비즈니스</th></tr>",
        "</thead>",
        "<tbody>",
    ]
    for ids in row_ids:
        row_name = cell_map[ids[0]]["row"]
        col_values: list[str] = []
        for cell_id in ids:
            escaped = html.escape(cells.get(cell_id, "")).replace("|", "\\|").replace(
                "\n", "<br>"
            )
            label = cell_map[cell_id]["label"]
            cell_class = "changed" if cell_id in highlight_ids else ""
            col_values.append(
                f"<td class='{cell_class}'><b>{label}</b><br>{escaped if escaped else '(비어 있음)'}</td>"
            )
        rows.append(
            f"<tr><td><b>{row_name}</b></td>{col_values[0]}{col_values[1]}{col_values[2]}</tr>"
        )
    rows.extend(["</tbody>", "</table>", "</div>"])
    return "".join(rows)


def init_session_state() -> None:
    if "generated" not in st.session_state:
        st.session_state.generated = False
    if "cells" not in st.session_state:
        st.session_state.cells = {cell_id: "" for cell_id in CELL_IDS}
    if "summary" not in st.session_state:
        st.session_state.summary = {
            "mvp_features": ["", "", ""],
            "revenue_models": ["", ""],
            "top_risks": ["", "", ""],
        }
    if "problem_input" not in st.session_state:
        st.session_state.problem_input = ""
    if "target_input" not in st.session_state:
        st.session_state.target_input = ""
    if "global_feedback" not in st.session_state:
        st.session_state.global_feedback = ""
    if "latest_refine" not in st.session_state:
        st.session_state.latest_refine = None

    for cell_id in CELL_IDS:
        mode_key = f"mode_{cell_id}"
        manual_key = f"manual_{cell_id}"
        feedback_key = f"feedback_{cell_id}"
        if mode_key not in st.session_state:
            st.session_state[mode_key] = "고정"
        if manual_key not in st.session_state:
            st.session_state[manual_key] = ""
        if feedback_key not in st.session_state:
            st.session_state[feedback_key] = ""


def call_generation(client: OpenAI, model: str, user_prompt: str) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content or ""
    parsed = parse_json_response(content)
    return normalize_result(parsed)


st.set_page_config(page_title="Structo AI", layout="wide")
inject_style()
init_session_state()

st.title("Structo AI")
st.caption("문제를 9개의 관점으로 정위(定位)해 전략으로 바꿉니다.")
st.markdown(
    '<div class="axis-note"><b>축 안내</b> · 행: 진단 → 설계 → 실행 / 열: 사용자 → 해결 → 비즈니스 / 모드: 입력 · 고정 · 비우기</div>',
    unsafe_allow_html=True,
)

api_key = read_api_key()
if not api_key:
    st.error("OPENAI_API_KEY가 없습니다. Streamlit secrets 또는 환경변수를 설정하세요.")
    st.stop()

client = OpenAI(api_key=api_key)

left, right = st.columns([2.2, 1.0], gap="large")
with left:
    st.text_area(
        "문제(Problem)",
        key="problem_input",
        height=160,
        placeholder="해결하고 싶은 문제를 2~6문장으로 입력하세요.",
    )
    st.text_area(
        "타겟 사용자(Target user)",
        key="target_input",
        height=110,
        placeholder="누가 이 문제를 겪는지 1~2문장으로 입력하세요.",
    )
with right:
    model = st.text_input("모델명", value="gpt-5.2")
    st.text_area(
        "전체 피드백(선택)",
        key="global_feedback",
        height=160,
        placeholder="전체 방향, 톤, 강조하고 싶은 기준을 입력하세요.",
    )
    initial_clicked = st.button(
        "초기 9셀 생성",
        type="primary",
        use_container_width=True,
    )

if initial_clicked:
    if not st.session_state.problem_input.strip() or not st.session_state.target_input.strip():
        st.warning("문제와 타겟 사용자 입력을 모두 채워주세요.")
    else:
        try:
            prompt = build_initial_prompt(
                problem=st.session_state.problem_input.strip(),
                target=st.session_state.target_input.strip(),
                global_feedback=st.session_state.global_feedback.strip(),
            )
            with st.spinner("초기 9셀 생성 중..."):
                result = call_generation(client=client, model=model, user_prompt=prompt)
            st.session_state.cells = result["cells"]
            st.session_state.summary = result["summary"]
            st.session_state.latest_refine = None
            st.session_state.generated = True
            for cell_id in CELL_IDS:
                st.session_state[f"mode_{cell_id}"] = "고정"
                st.session_state[f"manual_{cell_id}"] = st.session_state.cells[cell_id]
                st.session_state[f"feedback_{cell_id}"] = ""
            st.success("초기 9셀 생성이 완료되었습니다.")
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
        except ValueError as e:
            st.error("모델 응답을 JSON으로 해석하지 못했습니다. 다시 시도하세요.")
            st.code(str(e))
        except Exception as e:
            st.error("알 수 없는 오류가 발생했습니다.")
            st.code(str(e))

if st.session_state.generated:
    st.markdown("### 3×3 전략 프레임워크")
    st.markdown(render_html_table(st.session_state.cells), unsafe_allow_html=True)

    st.markdown("### 요약 블록")
    summary_cols = st.columns(3, gap="medium")
    summary_keys = [
        ("MVP 기능 3개", "mvp_features"),
        ("수익모델 2개", "revenue_models"),
        ("가장 큰 리스크 3개", "top_risks"),
    ]
    for col, (title, key) in zip(summary_cols, summary_keys):
        with col:
            st.markdown(f"**{title}**")
            for item in st.session_state.summary.get(key, []):
                st.markdown(f"- {item if item else '(비어 있음)'}")

    st.markdown("---")
    st.markdown(
        '<div class="control-title">셀별 재수정 컨트롤</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "입력: 직접 내용 반영 | 고정: 현재 셀 유지 | 비우기: 셀을 공란으로 유지. "
        "필요하면 셀별 피드백을 적고 재생성하세요."
    )

    row_to_ids = {
        "진단": ["diag_user", "diag_solution", "diag_business"],
        "설계": ["design_user", "design_solution", "design_business"],
        "실행": ["exec_user", "exec_solution", "exec_business"],
    }
    cell_meta = {item["id"]: item for item in CELL_DEFS}

    for row_name, ids in row_to_ids.items():
        st.markdown(f"#### {row_name}")
        cols = st.columns(3, gap="large")
        for idx, cell_id in enumerate(ids):
            meta = cell_meta[cell_id]
            with cols[idx]:
                st.markdown(f"**{meta['col']} · {meta['label']}**")
                st.selectbox(
                    "모드",
                    options=["고정", "입력", "비우기"],
                    key=f"mode_{cell_id}",
                    label_visibility="collapsed",
                )
                mode = st.session_state[f"mode_{cell_id}"]
                if mode == "입력":
                    st.text_area(
                        "입력 내용",
                        key=f"manual_{cell_id}",
                        height=130,
                        placeholder="이 셀에 반영할 내용을 입력하세요.",
                        label_visibility="collapsed",
                    )
                else:
                    preview = html.escape(st.session_state.cells.get(cell_id, "")).replace(
                        "\n",
                        "<br>",
                    )
                    st.markdown(
                        f"<div class='cell-preview'>{preview if preview else '(비어 있음)'}</div>",
                        unsafe_allow_html=True,
                    )
                st.text_input(
                    "셀별 피드백",
                    key=f"feedback_{cell_id}",
                    placeholder="이 셀만 보완할 포인트(선택)",
                    label_visibility="collapsed",
                )

    action_col_1, action_col_2 = st.columns([1.0, 1.0], gap="medium")
    with action_col_1:
        refine_clicked = st.button("선택 반영 후 재생성", type="primary", use_container_width=True)
    with action_col_2:
        reset_modes = st.button("모드 초기화", use_container_width=True)

    if reset_modes:
        for cell_id in CELL_IDS:
            st.session_state[f"mode_{cell_id}"] = "고정"
            st.session_state[f"manual_{cell_id}"] = st.session_state.cells[cell_id]
            st.session_state[f"feedback_{cell_id}"] = ""
        st.success("모드를 고정 기준으로 초기화했습니다.")

    if refine_clicked:
        controls: dict[str, dict[str, str]] = {}
        for cell_id in CELL_IDS:
            controls[cell_id] = {
                "mode": st.session_state[f"mode_{cell_id}"],
                "manual_input": st.session_state[f"manual_{cell_id}"].strip(),
                "cell_feedback": st.session_state[f"feedback_{cell_id}"].strip(),
            }

        try:
            prev_cells = dict(st.session_state.cells)
            prompt = build_refine_prompt(
                problem=st.session_state.problem_input.strip(),
                target=st.session_state.target_input.strip(),
                current_cells=st.session_state.cells,
                controls=controls,
                global_feedback=st.session_state.global_feedback.strip(),
            )
            with st.spinner("셀별 제어 조건을 반영해 재생성 중..."):
                result = call_generation(client=client, model=model, user_prompt=prompt)
                new_cells = result["cells"]

            for cell_id in CELL_IDS:
                mode = controls[cell_id]["mode"]
                if mode == "고정":
                    new_cells[cell_id] = st.session_state.cells[cell_id]
                elif mode == "비우기":
                    new_cells[cell_id] = ""
                elif mode == "입력" and controls[cell_id]["manual_input"]:
                    new_cells[cell_id] = controls[cell_id]["manual_input"]

            new_summary = result["summary"]
            if not any(new_summary["mvp_features"]) and not any(new_summary["revenue_models"]) and not any(new_summary["top_risks"]):
                new_summary = fallback_summary_from_cells(new_cells)

            changed_ids = [
                cell_id
                for cell_id in CELL_IDS
                if prev_cells.get(cell_id, "").strip() != new_cells.get(cell_id, "").strip()
            ]

            st.session_state.cells = new_cells
            st.session_state.summary = new_summary
            st.session_state.latest_refine = {
                "new_cells": new_cells,
                "changed_ids": changed_ids,
            }

            for cell_id in CELL_IDS:
                if st.session_state[f"mode_{cell_id}"] != "입력":
                    st.session_state[f"manual_{cell_id}"] = st.session_state.cells[cell_id]
            st.success("재수정 결과가 반영되었습니다.")
            st.rerun()
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
        except ValueError as e:
            st.error("모델 응답을 JSON으로 해석하지 못했습니다. 재시도하세요.")
            st.code(str(e))
        except Exception as e:
            st.error("알 수 없는 오류가 발생했습니다.")
            st.code(str(e))

    latest_refine = st.session_state.get("latest_refine")
    if isinstance(latest_refine, dict) and latest_refine.get("new_cells"):
        changed_ids = latest_refine.get("changed_ids", [])
        cell_map = {item["id"]: item for item in CELL_DEFS}
        st.markdown("---")
        st.markdown("### 최근 재수정 결과 (하단 표시)")
        if changed_ids:
            chips = []
            for cell_id in changed_ids:
                meta = cell_map.get(cell_id, {})
                chips.append(
                    f"<span class='changed-chip'>{meta.get('row', '')}/{meta.get('col', '')}: {meta.get('label', cell_id)}</span>"
                )
            st.markdown("".join(chips), unsafe_allow_html=True)
        else:
            st.info("변경된 셀이 없습니다. 피드백이나 모드를 조정해 다시 재생성하세요.")

        st.markdown(
            render_html_table(
                latest_refine["new_cells"],
                highlight_ids=set(changed_ids),
            ),
            unsafe_allow_html=True,
        )
else:
    st.info("문제와 타겟을 입력하고 `초기 9셀 생성`을 눌러 시작하세요.")
