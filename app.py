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
    extra_text: str = "",
) -> str:
    """Follow-up QA, 可附加一份額外上傳文件內容（extra_text）。"""
    fw = FRAMEWORKS[framework_key]

    # 全部用 ASCII，避免任何未結束字串問題
    if language == "zh":
        system_prompt = (
            "You are an Error-Free consultant familiar with framework: "
            + fw["name_zh"]
            + ". You have already produced one full analysis for this document. "
              "Now you only need to answer follow-up questions based on the original "
              "document and your previous analysis. Avoid repeating the full report; "
              "focus on additional explanations and concrete recommendations."
        )
    else:
        system_prompt = (
            "You are an Error-Free consultant for framework: "
            + fw["name_en"]
            + ". You have already produced an initial analysis. "
              "Now answer follow-up questions based on the original document "
              "and your previous analysis, without recreating the full report. "
              "Focus on clarifications and practical suggestions."
        )

    doc_label = "Original document excerpt:"
    analysis_label = "Previous analysis excerpt:"
    question_label = "User's new question:"
    extra_label = "Extra reference content (from uploaded file):"

    max_doc_chars = 8000
    max_analysis_chars = 8000
    max_extra_chars = 4000

    doc_excerpt = document_text[:max_doc_chars]
    analysis_excerpt = analysis_output[:max_analysis_chars]
    extra_excerpt = extra_text[:max_extra_chars] if extra_text else ""

    user_parts = [
        f"{doc_label}\n{doc_excerpt}",
        f"{analysis_label}\n{analysis_excerpt}",
        f"{question_label}\n{user_question}",
    ]
    if extra_excerpt:
        user_parts.append(f"{extra_label}\n{extra_excerpt}")

    user_content = "\n\n".join(user_parts)

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
# Report text cleaning
# =========================
def clean_report_text(text: str) -> str:
    """整理報告文字，避免黑色方塊與怪符號，讓 TXT / DOCX / PDF 看起來專業。"""
    replacements = {
        "■": "-",      # 你檔案裡看到的黑色方塊
        "\u2022": "-",  # bullet
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


# =========================
# Report building
# 下載報告：只包含「分析結果 + Q&A」，不含原始文件
# =========================
def build_full_report(lang: str, framework_key: str, state: Dict) -> str:
    analysis_output = state.get("analysis_output", "")
    followup_history: List[Tuple[str, str]] = state.get("followup_history", [])
    fw = FRAMEWORKS[framework_key]
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_email = st.session_state.get("user_email", "anonymous")

    if lang == "zh":
        header = [
            "Error-Free® 多框架 AI 文件分析報告（僅分析 + Q&A）",
            f"產生時間：{now}",
            f"使用者帳號：{user_email}",
            f"使用框架：{fw['name_zh']}",
            "",
            "==============================",
            "一、AI 框架分析結果",
            "==============================",
            analysis_output,
        ]
        if followup_history:
            header.append("")
            header.append("==============================")
            header.append("二、後續提問與回覆（Q&A）")
            header.append("==============================")
            for i, (q, a) in enumerate(followup_history, start=1):
                header.append(f"[Q{i}] {q}")
                header.append(f"[A{i}] {a}")
                header.append("")
    else:
        header = [
            "Error-Free® Multi-framework AI Analysis Report (Analysis + Q&A only)",
            f"Generated at: {now}",
            f"User account: {user_email}",
            f"Framework: {fw['name_en']}",
            "",
            "==============================",
            "1. Framework analysis result",
            "==============================",
            analysis_output,
        ]
        if followup_history:
            header.append("")
            header.append("==============================")
            header.append("2. Follow-up Q&A")
            header.append("==============================")
            for i, (q, a) in enumerate(followup_history, start=1):
                header.append(f"[Q{i}] {q}")
                header.append(f"[A{i}] {a}")
                header.append("")

    raw_text = "\n".join(header)
    return clean_report_text(raw_text)


def build_docx_bytes(text: str) -> bytes:
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()


def build_pdf_bytes(text: str) -> bytes:
    # 這裡假設文字已經先經過 clean_report_text
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
                st.session_state.usage_count = 0
                save_state_to_disk()
                st.rerun()
            else:
                if lang == "zh":
                    st.error("帳號或密碼錯誤，請再試一次。")
                else:
                    st.error("Invalid email or password. Please try again.")
        return

    # logged-in main UI
    if lang == "zh":
        st.title("Error-Free® 多框架 AI 文件分析")
        if st.session_state.login_success:
            st.success("Login successful！已成功登入。")
            st.session_state.login_success = False
            save_state_to_disk()
    else:
        st.title("Error-Free® Multi-framework AI Document Analyzer")
        if st.session_state.login_success:
            st.success("Login successful!")
            st.session_state.login_success = False
            save_state_to_disk()

    role = st.session_state.user_role or "free"
    model_name, daily_limit = get_model_and_limit(role)
    today = datetime.date.today().isoformat()
    if st.session_state.usage_date != today:
        st.session_state.usage_date = today
        st.session_state.usage_count = 0
        save_state_to_disk()

    if lang == "zh":
        with st.expander("目前使用方案與模型", expanded=True):
            st.write(f"角色：{role}")
            st.write(f"使用模型：{model_name}")
            if daily_limit is None:
                st.write("每日使用次數：無上限")
            else:
                st.write(
                    f"每日使用次數上限：{daily_limit}，今日已使用：{st.session_state.usage_count}"
                )
    else:
        with st.expander("Plan & model info", expanded=True):
            st.write(f"Role: {role}")
            st.write(f"Model: {model_name}")
            if daily_limit is None:
                st.write("Daily limit: unlimited")
            else:
                st.write(
                    f"Daily limit: {daily_limit}, used today: {st.session_state.usage_count}"
                )

    st.markdown("---")

    # Step 1: upload
    if lang == "zh":
        st.subheader("步驟一：上傳文件")
        uploaded_file = st.file_uploader(
            "請上傳 PDF / DOCX / TXT 檔案", type=["pdf", "docx", "txt"]
        )
    else:
        st.subheader("Step 1: Upload a document")
        uploaded_file = st.file_uploader(
            "Upload a PDF / DOCX / TXT file", type=["pdf", "docx", "txt"]
        )

    if uploaded_file is not None:
        doc_text = read_file_to_text(uploaded_file)
        if doc_text:
            st.session_state.last_doc_text = doc_text
            save_state_to_disk()

    # Step 2: framework selection
    if lang == "zh":
        st.subheader("步驟二：選擇分析框架（可來回切換）")
    else:
        st.subheader("Step 2: Choose an analysis framework (you can switch freely)")

    framework_keys = list(FRAMEWORKS.keys())
    if lang == "zh":
        framework_labels = [FRAMEWORKS[k]["name_zh"] for k in framework_keys]
    else:
        framework_labels = [FRAMEWORKS[k]["name_en"] for k in framework_keys]

    key_to_label = dict(zip(framework_keys, framework_labels))
    label_to_key = dict(zip(framework_labels, framework_keys))

    current_selected_label = key_to_label.get(
        st.session_state.selected_framework_key, framework_labels[0]
    )
    selected_label = st.selectbox(
        "Framework" if lang == "en" else "請選擇框架",
        framework_labels,
        index=framework_labels.index(current_selected_label),
    )
    selected_framework_key = label_to_key[selected_label]
    st.session_state.selected_framework_key = selected_framework_key

    framework_states: Dict[str, Dict] = st.session_state.framework_states
    if selected_framework_key not in framework_states:
        framework_states[selected_framework_key] = {
            "analysis_done": False,
            "analysis_output": "",
            "followup_history": [],
        }
    current_state = framework_states[selected_framework_key]

    st.markdown("---")

    # Run analysis
    can_run_analysis = not current_state["analysis_done"]

    if can_run_analysis:
        run_label = "開始分析（Run Analysis）" if lang == "zh" else "Run Analysis"
        run_button = st.button(run_label)
    else:
        run_button = False
        if lang == "zh":
            st.info(
                "本框架已完成一次分析。可切換至其他框架分析，或按 Reset 重新上傳新文件。"
            )
        else:
            st.info(
                "Analysis for this framework has already been run once. "
                "You can switch to another framework, or click Reset for a new document."
            )

    reset_label = "Reset / 重新開始（新文件）" if lang == "zh" else "Reset / New document"
    if st.button(reset_label):
        st.session_state.last_doc_text = ""
        st.session_state.framework_states = {}
        save_state_to_disk()
        st.rerun()

    if run_button and can_run_analysis:
        if not st.session_state.last_doc_text:
            if lang == "zh":
                st.error("請先上傳一份要分析的文件。")
            else:
                st.error("Please upload a document to analyze.")
        else:
            if daily_limit is not None and st.session_state.usage_count >= daily_limit:
                if lang == "zh":
                    st.error("已達今日使用次數上限。")
                else:
                    st.error("You have reached your daily usage limit.")
            else:
                with st.spinner("Running analysis..."):
                    analysis_text = run_llm_analysis(
                        framework_key=selected_framework_key,
                        language=lang,
                        document_text=st.session_state.last_doc_text,
                        model_name=model_name,
                    )
                current_state["analysis_done"] = True
                current_state["analysis_output"] = analysis_text
                current_state["followup_history"] = []
                st.session_state.usage_count += 1
                save_state_to_disk()
                if lang == "zh":
                    st.success("分析完成！請往下捲動查看各模組的分析結果與問答。")
                else:
                    st.success(
                        "Analysis completed! Scroll down to see each framework's results and Q&A."
                    )

    # ===== 顯示所有已分析過的模組 =====
    any_has_analysis = False
    for fw_key in FRAMEWORKS.keys():
        state = framework_states.get(
            fw_key,
            {"analysis_done": False, "analysis_output": "", "followup_history": []},
        )
        if not state.get("analysis_output"):
            continue

        any_has_analysis = True
        fw = FRAMEWORKS[fw_key]

        st.markdown("---")
        # 模組抬頭
        if lang == "zh":
            title_text = f"{fw['name_zh']}：分析結果與問答"
        else:
            title_text = f"{fw['name_en']}: Analysis & Q&A"

        if fw_key == selected_framework_key:
            st.subheader("⭐ " + title_text)
            if lang == "zh":
                st.caption("（目前選取的模組，底部追問會針對這個模組。）")
            else:
                st.caption(
                    "(Currently selected framework; follow-up chat at bottom is for this framework.)"
                )
        else:
            st.subheader(title_text)

        # 分析結果
        if lang == "zh":
            st.markdown("#### 分析結果")
        else:
            st.markdown("#### Analysis result")
        st.markdown(state["analysis_output"])

        # Q&A 歷史
        if lang == "zh":
            st.markdown("#### 後續提問歷史（Q&A）")
        else:
            st.markdown("#### Follow-up Q&A history")

        if state["followup_history"]:
            for idx, (q, a) in enumerate(state["followup_history"], start=1):
                st.markdown(f"**Q{idx}:** {q}")
                st.markdown(f"**A{idx}:** {a}")
                st.markdown("---")
        else:
            if lang == "zh":
                st.info("這個模組目前尚未有追問。")
            else:
                st.info("No follow-up questions yet for this framework.")

        # Download block（每個模組各自一份）
        if lang == "zh":
            st.markdown("##### Download full report（完整報告下載）")
            st.caption("此報告只包含：此模組的分析結果，以及此模組的所有 Q&A（不含原始文件）。")
        else:
            st.markdown("##### Download full report")
            st.caption(
                "Report includes only this framework's analysis and all Q&A (no original document text)."
            )

        report_text = build_full_report(lang, fw_key, state)
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_bytes = report_text.encode("utf-8")
        docx_bytes = build_docx_bytes(report_text)
        pdf_bytes = build_pdf_bytes(report_text)

        if lang == "zh":
            fmt = st.selectbox(
                "選擇下載格式",
                ["TXT", "Word (DOCX)", "PDF"],
                key=f"download_format_{fw_key}",
            )
            download_label = "Download"
        else:
            fmt = st.selectbox(
                "Select format",
                ["TXT", "Word (DOCX)", "PDF"],
                key=f"download_format_{fw_key}",
            )
            download_label = "Download"

        if fmt == "TXT":
            data = txt_bytes
            mime = "text/plain"
            ext = "txt"
        elif fmt == "Word (DOCX)":
            data = docx_bytes
            mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ext = "docx"
        else:
            data = pdf_bytes
            mime = "application/pdf"
            ext = "pdf"

        st.download_button(
            label=download_label,
            data=data,
            file_name=f"errorfree_report_{fw_key}_{now_str}.{ext}",
            mime=mime,
            key=f"download_button_{fw_key}",
        )

    # ===== 底部：Follow-up chat（只針對目前選取的模組）=====
    if any_has_analysis and framework_states[selected_framework_key].get(
        "analysis_output"
    ):
        # 附檔上傳（只對本次追問有效）
        if lang == "zh":
            extra_label = "若要針對此模組追問，您可以在這裡上傳相關文件（選擇性，僅本次問題使用）："
        else:
            extra_label = "Attach a related document for this framework (optional, only for this question):"

        extra_file = st.file_uploader(
            extra_label,
            type=["pdf", "docx", "txt"],
            key=f"followup_extra_{selected_framework_key}",
        )

        extra_text = ""
        if extra_file is not None:
            extra_text = read_file_to_text(extra_file)

        if lang == "zh":
            fw_name = FRAMEWORKS[selected_framework_key]["name_zh"]
            prompt = f"請輸入你針對「{fw_name}」的追問（按 Enter 送出）"
        else:
            fw_name = FRAMEWORKS[selected_framework_key]["name_en"]
            prompt = f"Enter your follow-up question for [{fw_name}]"

        followup_q = st.chat_input(prompt)

        if followup_q:
            state = framework_states[selected_framework_key]
            with st.spinner("Thinking..."):
                answer = run_followup_qa(
                    framework_key=selected_framework_key,
                    language=lang,
                    document_text=st.session_state.last_doc_text,
                    analysis_output=state["analysis_output"],
                    user_question=followup_q.strip(),
                    model_name=model_name,
                    extra_text=extra_text,
                )
            state["followup_history"].append((followup_q.strip(), answer))
            save_state_to_disk()
            st.rerun()

    save_state_to_disk()


if __name__ == "__main__":
    main()
