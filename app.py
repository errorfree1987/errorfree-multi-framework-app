import os
import json
import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from io import BytesIO

import streamlit as st
import pdfplumber
from docx import Document
from openai import OpenAI
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# =========================
# OpenAI client
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# =========================
# Account config
# =========================
ACCOUNTS = {
    "admin@errorfree.com": {"password": "1111", "role": "admin"},
    "dr.chiu@errorfree.com": {"password": "2222", "role": "pro"},
    "guest@errorfree.com": {"password": "3333", "role": "pro"},
}


def get_model_and_limit(role: str):
    """Return (model_name, daily_limit). daily_limit=None means unlimited."""
    if role == "free":
        return "gpt-4.1-mini", 5
    if role == "advanced":
        return "gpt-4.1", 10
    if role in ["pro", "admin"]:
        return "gpt-5.1", None
    return "gpt-4.1-mini", 2


# =========================
# Framework definitions
# =========================
FRAMEWORKS: Dict[str, Dict] = {
    "omission": {
        "name_zh": "Error-Free® 遺漏錯誤檢查框架",
        "name_en": "Error-Free® Omission Error Check Framework",
        "wrapper_zh": (
            "你是一位 Error-Free® 遺漏錯誤檢查專家。"
            "請分析文件中可能遺漏的重要內容、條件、假設、角色、步驟、風險或例外，"
            "並說明遺漏的影響與具體補強建議，最後整理成條列與一個簡單的 Markdown 表格。"
        ),
        "wrapper_en": (
            "You are an Error-Free® omission error expert. "
            "Review the document, find important missing information or conditions, "
            "explain the impact, and give concrete suggestions, plus a simple Markdown table."
        ),
    },
    "technical": {
        "name_zh": "Error-Free® 技術風險檢查框架",
        "name_en": "Error-Free® Technical Risk Check Framework",
        "wrapper_zh": (
            "你是一位 Error-Free® 技術風險檢查專家。"
            "請從技術假設、邊界條件、相容性、安全性、可靠度與單點失敗等面向分析文件，"
            "列出技術風險、風險等級與實務改善建議，並以 Markdown 表格整理重點。"
        ),
        "wrapper_en": (
            "You are an Error-Free® technical risk review expert. "
            "Analyze the document for technical assumptions, edge cases, compatibility, "
            "safety and single points of failure. List risks, risk level and mitigation, "
            "and provide a summary Markdown table."
        ),
    },
}

# =========================
# State persistence
# =========================
STATE_FILE = Path("user_state.json")


def save_state_to_disk():
    data = {
        "user_email": st.session_state.get("user_email"),
        "user_role": st.session_state.get("user_role"),
        "is_authenticated": st.session_state.get("is_authenticated", False),
        "lang": st.session_state.get("lang", "zh"),
        "usage_date": st.session_state.get("usage_date"),
        "usage_count": st.session_state.get("usage_count", 0),
        "last_doc_text": st.session_state.get("last_doc_text", ""),
        "framework_states": st.session_state.get("framework_states", {}),
        "selected_framework_key": st.session_state.get("selected_framework_key"),
    }
    try:
        STATE_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def restore_state_from_disk():
    if not STATE_FILE.exists():
        return
    try:
        raw = STATE_FILE.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception:
        return
    for key, value in data.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =========================
# File reading
# =========================
def read_file_to_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".pdf"):
            text_pages: List[str] = []
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    t = page.extract_text() or ""
                    text_pages.append(t)
            return "\n".join(text_pages)
        elif name.endswith(".docx"):
            doc = Document(uploaded_file)
            return "\n".join(p.text for p in doc.paragraphs)
        elif name.endswith(".txt"):
            return uploaded_file.read().decode("utf-8", errors="ignore")
        else:
            return ""
    except Exception as e:
        return f"[讀取檔案時發生錯誤: {e}]"


# =========================
# LLM helpers
# =========================
def run_llm_analysis(
    framework_key: str, language: str, document_text: str, model_name: str
) -> str:
    fw = FRAMEWORKS[framework_key]
    system_prompt = fw["wrapper_zh"] if language == "zh" else fw["wrapper_en"]
    if language == "zh":
        user_prompt = "以下是要分析的文件內容：\n\n" + document_text
    else:
        user_prompt = "Here is the document to analyze:\n\n" + document_text

    if client is None:
        return "[Error] OPENAI_API_KEY 尚未設定，無法連線至 OpenAI。"

    try:
        response = client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_output_tokens=2500,
        )
        return response.output_text
    except Exception as e:
        return f"[呼叫 OpenAI API 時發生錯誤: {e}]"


def run_followup_qa(
    framework_key: str,
    language: str,
    document_text: str,
    analysis_output: str,
    user_question: str,
    model_name: str,
) -> str:
    fw = FRAMEWORKS[framework_key]

    if language == "zh":
        system_prompt = (
            "你是一位 Error-Free® 文件與風險顧問，熟悉框架："
            + fw["name_zh"]
            + "。你已經做過一次分析，現在只需根據原始文件與先前分析結果，回答追問，"
            "請避免重複完整報告，專注於補充說明與具體建議。"
        )
        doc_label = "以下是原始文件內容（節錄）："
        analysis_label = "以下是你過去產出的分析重點（節錄）："
        question_label = "使用者的新提問如下："
    else:
        system_prompt = (
            "You are an Error-Free® consultant for framework: "
            + fw["name_en"]
            + ". You have already produced an initial analysis. "
              "Now answer follow-up questions based on the original document "
              "and your previous analysis, without re-creating the full report."
        )
        doc_label = "Here is an excerpt of the original document:"
        analysis_label = "Here is an excerpt of your previous analysis:"
        question_label = "The user's new question is:"

    max_doc_chars = 8000
    max_analysis_chars = 8000
    doc_excerpt = document_text[:max_doc_chars]
    analysis_excerpt = analysis_output[:max_analysis_chars]

    user_content = (
        f"{doc_label}\n{doc_excerpt}\n\n"
        f"{analysis_label}\n{analysis_excerpt}\n\n"
        f"{question_label}\n{user_question}"
    )

    if client is None:
        return "[Error] OPENAI_API_KEY 尚未設定，無法連線至 OpenAI。"

    try:
        response = client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_output_tokens=2000,
        )
        return response.output_text
    except Exception as e:
        return f"[呼叫 OpenAI API 時發生錯誤: {e}]"


# =========================
# Report building
# =========================
def build_full_report(lang: str, framework_key: str, state: Dict) -> str:
    doc_text = st.session_state.get("last_doc_text", "")
    analysis_output = state.get("analysis_output", "")
    followup_history: List[Tuple[str, str]] = state.get("followup_history", [])
    fw = FRAMEWORKS[framework_key]
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_email = st.session_state.get("user_email", "anonymous")

    if lang == "zh":
        header = [
            "Error-Free® 多框架 AI 文件分析完整報告",
            f"產生時間：{now}",
            f"使用者帳號：{user_email}",
            f"使用框架：{fw['name_zh']}",
            "",
            "==============================",
            "一、原始文件內容（節錄）",
            "==============================",
            doc_text,
            "",
            "==============================",
            "二、第一次 AI 框架分析結果",
            "==============================",
            analysis_output,
        ]
        if followup_history:
            header.append("")
            header.append("==============================")
            header.append("三、後續提問與回覆（Q&A）")
            header.append("==============================")
            for i, (q, a) in enumerate(followup_history, start=1):
                header.append(f"[Q{i}] {q}")
                header.append(f"[A{i}] {a}")
                header.append("")
    else:
        header = [
            "Error-Free® Multi-framework AI Document Analysis - Full Report",
            f"Generated at: {now}",
            f"User account: {user_email}",
            f"Framework: {fw['name_en']}",
            "",
            "==============================",
            "1. Original document (excerpt)",
            "==============================",
            doc_text,
            "",
            "==============================",
            "2. Initial AI framework analysis",
            "==============================",
            analysis_output,
        ]
        if followup_history:
            header.append("")
            header.append("==============================")
            header.append("3. Follow-up Q&A")
            header.append("==============================")
            for i, (q, a) in enumerate(followup_history, start=1):
                header.append(f"[Q{i}] {q}")
                header.append(f"[A{i}] {a}")
                header.append("")

    return "\n".join(header)


def build_docx_bytes(text: str) -> bytes:
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()


def build_pdf_bytes(text: str) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    y = height - 40
    for line in text.split("\n"):
        c.drawString(40, y, line[:1000])
        y -= 14
        if y < 40:
            c.showPage()
            y = height - 40
    c.save()
    buf.seek(0)
    return buf.getvalue()


# =========================
# Main app
# =========================
def main():
    st.set_page_config(
        page_title="Error-Free Multi-framework AI Document Analyzer",
        layout="wide",
    )

    restore_state_from_disk()

    # init session
    if "user_email" not in st.session_state:
        st.session_state.user_email = None
    if "user_role" not in st.session_state:
        st.session_state.user_role = None
    if "usage_date" not in st.session_state:
        st.session_state.usage_date = None
    if "usage_count" not in st.session_state:
        st.session_state.usage_count = 0
    if "is_authenticated" not in st.session_state:
        st.session_state.is_authenticated = False
    if "login_success" not in st.session_state:
        st.session_state.login_success = False
    if "lang" not in st.session_state:
        st.session_state.lang = "zh"
    if "last_doc_text" not in st.session_state:
        st.session_state.last_doc_text = ""
    if "framework_states" not in st.session_state:
        st.session_state.framework_states = {}
    if "selected_framework_key" not in st.session_state:
        st.session_state.selected_framework_key = list(FRAMEWORKS.keys())[0]

    # sidebar
    with st.sidebar:
        any_analysis = any(
            state.get("analysis_output")
            for state in st.session_state.framework_states.values()
        )
        current_lang = st.session_state.lang

        if any_analysis:
            if current_lang == "zh":
                st.markdown("### 語言")
                st.caption("目前語言：繁體中文（本次分析期間無法切換語言）")
            else:
                st.markdown("### Language")
                st.caption(
                    "Current language: English (language is locked while analysis data exists)."
                )
            lang = current_lang
        else:
            if current_lang == "en":
                st.markdown("### Language")
                new_lang = st.radio(
                    "Select language",
                    ["zh", "en"],
                    index=1,
                    format_func=lambda x: "Chinese" if x == "zh" else "English",
                )
            else:
                st.markdown("### 語言")
                new_lang = st.radio(
                    "請選擇介面語言",
                    ["zh", "en"],
                    index=0,
                    format_func=lambda x: "繁體中文" if x == "zh" else "English",
                )
            st.session_state.lang = new_lang
            lang = new_lang
            save_state_to_disk()

        st.markdown("---")
        if st.session_state.is_authenticated:
            if lang == "zh":
                st.subheader("帳號資訊")
                st.write(f"使用者：{st.session_state.user_email}")
                st.write(f"角色：{st.session_state.user_role}")
                if st.button("登出"):
                    st.session_state.user_email = None
                    st.session_state.user_role = None
                    st.session_state.is_authenticated = False
                    st.session_state.last_doc_text = ""
                    st.session_state.framework_states = {}
                    if STATE_FILE.exists():
                        try:
                            STATE_FILE.unlink()
                        except Exception:
                            pass
                    st.rerun()
            else:
                st.subheader("Account")
                st.write(f"User: {st.session_state.user_email}")
                st.write(f"Role: {st.session_state.user_role}")
                if st.button("Logout"):
                    st.session_state.user_email = None
                    st.session_state.user_role = None
                    st.session_state.is_authenticated = False
                    st.session_state.last_doc_text = ""
                    st.session_state.framework_states = {}
                    if STATE_FILE.exists():
                        try:
                            STATE_FILE.unlink()
                        except Exception:
                            pass
                    st.rerun()
        else:
            if lang == "zh":
                st.subheader("尚未登入")
                st.caption("請在主畫面輸入帳號密碼登入")
            else:
                st.subheader("Not logged in")
                st.caption("Please log in on the main page.")

    # login page
    if not st.session_state.is_authenticated:
        if lang == "zh":
            st.title("Error-Free® 多框架 AI 文件分析")
            st.subheader("請先登入")
        else:
            st.title("Error-Free® Multi-framework AI Document Analyzer")
            st.subheader("Please log in first")

        email = st.text_input("Email")
        if lang == "zh":
            password = st.text_input("密碼", type="password")
            login_btn = st.button("登入")
        else:
            password = st.text_input("Password", type="password")
            login_btn = st.button("Login")

        if login_btn:
            account = ACCOUNTS.get(email)
            if account and account["password"] == password:
                st.session_state.user_email = email
                st.session_state.user_role = account["role"]
                st.session_state.is_authenticated = True
                st.session_state.login_success = True
                st.session_state.usage_date = datetime.date.today().isoformat()
                st.session_stat_
