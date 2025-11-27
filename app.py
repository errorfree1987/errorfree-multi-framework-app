import os
import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
import pdfplumber
from docx import Document
from openai import OpenAI

# =========================================
# OpenAI client
# =========================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# =========================================
# 使用者帳號設定
# =========================================
ACCOUNTS = {
    "admin@errorfree.com": {"password": "1111", "role": "admin"},
    "dr.chiu@errorfree.com": {"password": "2222", "role": "pro"},
    "guest@errorfree.com": {"password": "3333", "role": "pro"},
}


# =========================================
# 依角色選模型
# =========================================
def get_model_and_limit(role: str):
    if role == "free":
        return "gpt-4.1-mini", 5
    if role == "advanced":
        return "gpt-4.1", 10
    if role in ["pro", "admin"]:
        return "gpt-5.1", None
    return "gpt-4.1-mini", 2


# =========================================
# 多框架設定
# =========================================
FRAMEWORKS: Dict[str, Dict] = {
    "omission": {
        "name_zh": "Error-Free® 遺漏錯誤檢查框架",
        "name_en": "Error-Free® Omission Error Check Framework",
        "description_zh": (
            "針對文件中「該出現卻沒出現」的內容進行系統性盤點，"
            "運用 Error-Free® 的遺漏檢查觀點，找出可能被忽略的關鍵資訊。"
        ),
        "description_en": (
            "Systematically checks for content that SHOULD be present but is missing."
        ),
        "wrapper_zh": """
你是一位 Error-Free® 遺漏錯誤檢查專家。

請分析文件並輸出：
1. 重點摘要
2. 可能的遺漏錯誤（逐點說明「哪裡遺漏」與「為何重要」）
3. 具體補強建議（可直接補寫的句子或檢查點）
4. Markdown 表格：遺漏類別、發現內容、影響、建議補強方向
        """,
        "wrapper_en": """
You are an Error-Free® omission error expert.

Output:
1. Summary
2. Omission issues (what is missing + why it matters)
3. Concrete recommendations
4. Markdown table
        """,
    }
}


# =========================================
# 工具：讀取檔案文字
# =========================================
def read_file_to_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    name = uploaded_file.name.lower()

    try:
        if name.endswith(".pdf"):
            pages = []
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    pages.append(page.extract_text() or "")
            return "\n".join(pages)

        elif name.endswith(".docx"):
            doc = Document(uploaded_file)
            return "\n".join(p.text for p in doc.paragraphs)

        elif name.endswith(".txt"):
            return uploaded_file.read().decode("utf-8", errors="ignore")

        return ""

    except Exception as e:
        return f"[讀取檔案錯誤: {e}]"


# =========================================
# 第一次分析
# =========================================
def run_llm_analysis(framework_key, language, document_text, model_name):
    fw = FRAMEWORKS[framework_key]
    system_prompt = fw["wrapper_zh"] if language == "zh" else fw["wrapper_en"]

    user_prompt = (
        "以下是文件內容：\n\n" + document_text
        if language == "zh"
        else "Here is the document:\n\n" + document_text
    )

    if client is None:
        return "[Error] OPENAI_API_KEY 未設定，無法連線 OpenAI"

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
        return f"[OpenAI API 錯誤: {e}]"


# =========================================
# 後續 Q&A
# =========================================
def run_followup_qa(framework_key, language, document_text, analysis_output, question, model_name):
    fw = FRAMEWORKS[framework_key]

    system_prompt = (
        f"你已做過完整分析。請根據文件與先前分析回答新問題。"
        if language == "zh"
        else "You already produced the initial analysis. Answer follow-up questions based on document & previous analysis."
    )

    combined_prompt = (
        f"文件（節錄）：\n{document_text[:7000]}\n\n"
        f"前次分析（節錄）：\n{analysis_output[:7000]}\n\n"
        f"提問：{question}"
    )

    try:
        resp = client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": combined_prompt},
            ],
            max_output_tokens=2000,
        )
        return resp.output_text

    except Exception as e:
        return f"[OpenAI 錯誤: {e}]"


# =========================================
# 生成下載報告
# =========================================
def build_full_report(lang: str) -> str:
    doc_text = st.session_state.get("last_doc_text", "")
    analysis = st.session_state.get("last_analysis_output", "")
    followups: List[Tuple[str, str]] = st.session_state.get("followup_history", [])
    fw_key = st.session_state.get("last_framework_key")

    if not fw_key:
        return ""

    fw = FRAMEWORKS[fw_key]
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user = st.session_state.get("user_email", "")

    report = []

    if lang == "zh":
        report.append("Error-Free® 多框架 AI 文件分析完整報告")
        report.append(f"產生時間：{now}")
        report.append(f"使用者：{user}")
        report.append(f"使用框架：{fw['name_zh']}")
        report.append("\n==== 原始文件（節錄） ====\n")
    else:
        report.append("Error-Free® Multi-framework Full Report")
        report.append(f"Generated at: {now}")
        report.append(f"User: {user}")
        report.append(f"Framework: {fw['name_en']}")
        report.append("\n==== Original Document (excerpt) ====\n")

    report.append(doc_text)

    report.append("\n==== Initial AI Analysis ====\n")
    report.append(analysis)

    if followups:
        report.append("\n==== Follow-up Q&A ====\n")
        for i, (q, a) in enumerate(followups, 1):
            report.append(f"[Q{i}] {q}")
            report.append(f"[A{i}] {a}\n")

    return "\n".join(report)


# =========================================
# Streamlit App
# =========================================
def main():
    st.set_page_config(page_title="Error-Free Multi-framework Analyzer", layout="wide")

    # ---- Session 狀態 ----
    for key, default in [
        ("user_email", None),
        ("user_role", None),
        ("usage_date", None),
        ("usage_count", 0),
        ("lang", "zh"),
        ("last_doc_text", ""),
        ("last_framework_key", None),
        ("last_analysis_output", ""),
        ("followup_history", []),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    lang = st.session_state.lang

    # =========================================
    # Sidebar：語言鎖定
    # =========================================
    with st.sidebar:
        st.markdown("### 語言" if lang == "zh" else "### Language")

        if st.session_state.last_analysis_output:
            st.caption(
                "語言已鎖定（本次分析期間無法切換）"
                if lang == "zh"
                else "Language locked for this analysis"
            )
        else:
            new_lang = st.radio(
                "Language" if lang == "en" else "選擇語言",
                ["zh", "en"],
                index=0 if lang == "zh" else 1,
                format_func=lambda v: "繁體中文" if v == "zh" else "English",
            )
            st.session_state.lang = new_lang
            lang = new_lang

    # =========================================
    # Title
    # =========================================
    logo = Path(__file__).parent / "logo.png"
    col1, col2 = st.columns([1, 4])

    with col2:
        if lang == "zh":
            st.title("零錯誤多框架 AI 文件分析器")
            st.caption("邱強博士零錯誤研發團隊1987年至今")
        else:
            st.title("Error-Free Multi-framework AI Document Analyzer")
            st.caption("Dr. Chong Chiu’s Error-Free Team — Advancing Practices")

    st.markdown("---")

    # =========================================
    # Sidebar：登入 + 框架選擇
    # =========================================
    with st.sidebar:
        st.markdown("### 帳號" if lang == "zh" else "### Account")

        if st.session_state.user_email:
            st.success(
                f"已登入：{st.session_state.user_email}"
                if lang == "zh"
                else f"Signed in as: {st.session_state.user_email}"
            )
            if st.button("登出" if lang == "zh" else "Log out"):
                for key in [
                    "user_email",
                    "user_role",
                    "usage_date",
                    "usage_count",
                    "last_doc_text",
                    "last_framework_key",
                    "last_analysis_output",
                    "followup_history",
                ]:
                    st.session_state[key] = (
                        None if "email" in key or "role" in key else ""
                    )
                st.rerun()

        else:
            email = st.text_input("Email")
            pw = st.text_input("密碼" if lang == "zh" else "Password", type="password")

            if st.button("登入" if lang == "zh" else "Log in"):
                acc = ACCOUNTS.get(email)
                if acc and acc["password"] == pw:
                    st.session_state.user_email = email
                    st.session_state.user_role = acc["role"]
                    st.success("登入成功！" if lang == "zh" else "Logged in.")
                    st.rerun()
                else:
                    st.error("Email 或密碼錯誤" if lang == "zh" else "Invalid login")

        st.markdown("---")

        st.markdown("### 框架" if lang == "zh" else "### Framework")
        framework_key = st.selectbox(
            "框架" if lang == "zh" else "Framework",
            list(FRAMEWORKS.keys()),
            format_func=lambda k: (
                FRAMEWORKS[k]["name_zh"] if lang == "zh" else FRAMEWORKS[k]["name_en"]
            ),
        )

        # 清除分析按鈕
        if st.session_state.last_analysis_output:
            if st.button("清除本次分析" if lang == "zh" else "Clear analysis"):
                for key in [
                    "last_doc_text",
                    "last_framework_key",
                    "last_analysis_output",
                    "followup_history",
                ]:
                    st.session_state[key] = "" if "history" not in key else []
                st.rerun()

    # =========================================
    # 主畫面：框架說明
    # =========================================
    fw = FRAMEWORKS[framework_key]

    st.caption(fw["description_zh"] if lang == "zh" else fw["description_en"])

    st.markdown("---")

    # =========================================
    # 上傳文件
    # =========================================
    uploaded = st.file_uploader(
        "上傳文件（PDF, Word, TXT）" if lang == "zh" else "Upload document",
        type=["pdf", "docx", "txt"],
    )

    if uploaded:
        text = read_file_to_text(uploaded)
        st.text_area("文件預覽" if lang == "zh" else "Preview", text[:1000], height=200)
    else:
        text = ""

    # =========================================
    # 第一次分析按鈕（分析後鎖定）
    # =========================================
    analysis_done = bool(st.session_state.last_analysis_output)

    run_btn = st.button(
        "開始進行 AI 分析" if lang == "zh" else "Run analysis",
        disabled=analysis_done,
    )

    if run_btn and not analysis_done:
        if not st.session_state.user_email:
            st.error("請先登入" if lang == "zh" else "Please log in")
            return

        if not text:
            st.error("請先上傳文件" if lang == "zh" else "Upload a document first")
            return

        today = datetime.date.today()
        if st.session_state.usage_date != today:
            st.session_state.usage_date = today
            st.session_state.usage_count = 0

        model_name, limit = get_model_and_limit(st.session_state.user_role)

        if limit and st.session_state.usage_count >= limit:
            st.error("今日已達上限" if lang == "zh" else "Daily limit reached")
            return

        max_chars = 120000
        text_to_use = text[:max_chars]

        with st.spinner("分析中…" if lang == "zh" else "Analyzing..."):
            ai_output = run_llm_analysis(
                framework_key, lang, text_to_use, model_name
            )

        st.session_state.usage_count += 1
        st.session_state.last_doc_text = text_to_use
        st.session_state.last_framework_key = framework_key
        st.session_state.last_analysis_output = ai_output
        st.session_state.followup_history = []

        st.rerun()

    # =========================================
    # 顯示分析結果
    # =========================================
    if st.session_state.last_analysis_output:
        st.markdown("---")
        st.subheader("AI 分析結果" if lang == "zh" else "AI Analysis")
        st.write(st.session_state.last_analysis_output)

    # =========================================
    # Q&A 區
    # =========================================
    if st.session_state.last_analysis_output:
        st.markdown("---")
        st.subheader("後續問答" if lang == "zh" else "Follow-up Q&A")

        # 顯示歷史問答
        for i, (q, a) in enumerate(st.session_state.followup_history, 1):
            st.markdown(f"**Q{i}: {q}**")
            st.write(a)
            st.markdown("---")

        # 新增提問
        question = st.text_area(
            "輸入你的問題…" if lang == "zh" else "Enter your question…"
        )

        if st.button("送出提問" if lang == "zh" else "Ask"):
            if question.strip():
                model_name, _ = get_model_and_limit(st.session_state.user_role)
                answer = run_followup_qa(
                    st.session_state.last_framework_key,
                    lang,
                    st.session_state.last_doc_text,
                    st.session_state.last_analysis_output,
                    question,
                    model_name,
                )
                st.session_state.followup_history.append((question, answer))
                st.rerun()
            else:
                st.error("請輸入問題" if lang == "zh" else "Enter a question")

        # =========================================
        # 下載報告按鈕（永遠在最底部）
        # =========================================
        full_report = build_full_report(lang)
        st.download_button(
            "下載完整報告（文字檔）" if lang == "zh" else "Download full report",
            full_report,
            file_name="errorfree_report.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
