import os
import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
import pdfplumber
from docx import Document
from openai import OpenAI

# =============================
# OpenAI client（使用 Railway 環境變數 OPENAI_API_KEY）
# =============================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =============================
# 帳號系統設定
# =============================
# 先用簡單密碼測試，確認登入無誤，之後你再自行修改成正式密碼即可。
ACCOUNTS = {
    "admin@errorfree.com": {
        "password": "1111",      # TODO: 換成你自己的密碼
        "role": "admin",
        "max_runs": 9999,
        "used_runs": 0,
    },
    "dr.chiu@errorfree.com": {
        "password": "2222",      # TODO: 換成邱博士密碼
        "role": "pro",
        "max_runs": 999,
        "used_runs": 0,
    },
    "guest@errorfree.com": {
        "password": "3333",      # TODO: 內部測試帳號
        "role": "pro",
        "max_runs": 100,
        "used_runs": 0,
    },
}


def get_model_and_limit(role: str):
    """依照帳號角色決定模型與每日次數上限"""
    if role == "free":
        return "gpt-4.1-mini", 5
    if role == "advanced":
        return "gpt-4.1", 10
    if role == "pro":
        return "gpt-5.1", None  # None = 不限
    if role == "admin":
        return "gpt-5.1", None
    return "gpt-4.1-mini", 2


# =============================
# 多框架設定
# =============================
FRAMEWORKS: Dict[str, Dict] = {
    "omission": {
        "name_zh": "Error-Free® 遺漏錯誤檢查框架",
        "name_en": "Error-Free® Omission Error Check Framework",
        "description_zh": (
            "針對文件中「該出現卻沒出現」的內容進行系統性盤點，"
            "運用 Error-Free® 的遺漏檢查觀點，找出可能被忽略的關鍵資訊。"
        ),
        "description_en": (
            "Systematically checks for content that SHOULD be present but is missing, "
            "using the Error-Free® omission perspective to reveal overlooked elements."
        ),
        "wrapper_zh": """
你是一位 Error-Free® 遺漏錯誤檢查專家，精通遺漏檢查方法與遺漏錯誤型態。

請你扮演「文件顧問」，用這個框架來分析輸入的文件，找出：
1. 文件可能遺漏的重要內容、條件、假設、角色、步驟、風險或例外情況。
2. 這些遺漏，可能在實務上造成什麼後果（例如專案失敗、誤解、作業風險、客訴…）。
3. 文件撰寫人應該如何具體補強、修改或新增內容。

請特別注意：
- 不只是改錯字，而是針對「沒有寫出來」但應該要寫的地方。
- 你可以引用文件中的關鍵句子來說明，但不要原封不動複製太長的段落。
- 回答時請用條列方式，讓使用者可以直接依序修正文件。

輸出格式請用繁體中文，依照下列結構：

一、文件重點摘要（3–5 行）

二、可能的遺漏錯誤（條列敘述，每點說明「哪裡遺漏」與「為何重要」）

三、具體修正與補強建議（逐點對應上面的遺漏，提出可直接採用的補寫句子、段落或檢查項目）

四、請額外產出一個 Markdown 表格，欄位為：
| 遺漏類別/主題 | 發現內容（簡述） | 可能影響 | 建議補強方向 |
表格中每一列對應一個重要的遺漏點，讓使用者可以快速掃描與追蹤。
        """,
        "wrapper_en": """
You are an Error-Free® omission error expert.

Your job is to review the document and identify:
1. Important information, conditions, assumptions, roles, steps, risks, or exceptions that SHOULD
   be present but are missing.
2. The potential practical impact of those omissions (e.g. project failure, misunderstandings,
   operational risk, customer complaints).
3. Concrete and actionable suggestions on how to revise or extend the document.

Please:
- Focus on omissions (missing content), not minor wording issues.
- You may quote short phrases from the document as examples, but do not copy long sections.
- Use a clear bullet-list style so the user can directly revise the document.

Answer structure in English:

1. Concise summary of the document (3–5 sentences)
2. Potential omission errors (bullet list, each with “what is missing” and “why it matters”)
3. Concrete revision suggestions (bullet list with sample wording or specific guidance)
4. Additionally, output a Markdown table with columns:
   | Category/Theme | Finding (short) | Potential impact | Recommendation |
Each row should correspond to one important omission so that the user can quickly scan and track it.
        """,
    },
    "technical": {
        "name_zh": "Error-Free® 技術風險檢查框架",
        "name_en": "Error-Free® Technical Risk Check Framework",
        "description_zh": (
            "從技術假設、邊界條件、相容性、安全性與可維護性等關鍵面向出發，"
            "協助識別方案或文件中可能被忽略的技術風險與隱患。"
        ),
        "description_en": (
            "Identifies hidden technical risks from assumptions, edge cases, compatibility, "
            "safety and maintainability perspectives in your solution or documentation."
        ),
        "wrapper_zh": """
你是一位 Error-Free® 技術風險檢查專家，擅長在需求文件、設計文件、方案提案中找出
常被忽略的技術風險與隱患。

請針對輸入的文件，依下列面向進行分析（如適用）：
- 技術假設是否合理？有沒有沒說清楚的前提或限制？
- 邊界條件與例外情況是否有被考慮？
- 與其他系統／模組／流程的相容性與整合風險是什麼？
- 安全性、可靠度、可維護性、可監控性，有沒有潛在風險？
- 有沒有容易被忽略的「單點失敗」、「人為操作風險」或「外部依賴」？

輸出格式（繁體中文）請用：

一、技術內容與設計重點的簡要摘要

二、可能的技術風險（條列說明，每點包含：風險內容、成因、涉及範圍）

三、風險等級判斷與影響說明（高／中／低，並說明理由）

四、具體改善建議（可以是設計調整、補充檢查點、額外文件、測試建議等）

五、請額外產出一個 Markdown 表格，欄位為：
| 風險項目 | 風險等級（高/中/低） | 影響說明 | 建議對策 |
表格列出你認為最值得優先處理的風險項目，方便後續追蹤與管理。
        """,
        "wrapper_en": """
You are an Error-Free® technical risk review expert. Your job is to find hidden
technical risks in requirements, design documents or solution proposals.

Please review the document and analyze (when applicable):
- Are the technical assumptions reasonable and clearly stated?
- Are edge cases and exceptional conditions considered?
- What are the compatibility / integration risks with other systems or processes?
- Are there potential risks regarding safety, reliability, maintainability, observability?
- Are there single points of failure, human-operation risks, or external dependencies?

Answer structure in English:

1. Brief summary of the technical content and main design ideas
2. Potential technical risks (bullet list, each with cause / context / affected scope)
3. Risk level and impact (High / Medium / Low, with short justification)
4. Concrete mitigation suggestions (design changes, additional checks, tests, documentation, etc.)
5. Additionally, output a Markdown table with columns:
   | Risk item | Risk level (High/Medium/Low) | Impact description | Mitigation suggestion |
Each row should correspond to one important risk that should be prioritized for follow-up.
        """,
    },
}


# =============================
# 工具：讀取檔案文字
# =============================
def read_file_to_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    name = uploaded_file.name.lower()
    try:
        if name.endswith(".pdf"):
            text = []
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    t = page.extract_text() or ""
                    text.append(t)
            return "\n".join(text)
        elif name.endswith(".docx"):
            doc = Document(uploaded_file)
            return "\n".join(p.text for p in doc.paragraphs)
        elif name.endswith(".txt"):
            return uploaded_file.read().decode("utf-8", errors="ignore")
        else:
            return ""
    except Exception as e:
        return f"[讀取檔案時發生錯誤: {e}]"


# =============================
# 工具：第一次框架分析
# =============================
def run_llm_analysis(
    framework_key: str,
    language: str,
    document_text: str,
    model_name: str,
) -> str:
    fw = FRAMEWORKS[framework_key]
    if language == "zh":
        system_prompt = fw["wrapper_zh"]
    else:
        system_prompt = fw["wrapper_en"]

    user_prompt = (
        ("以下是要分析的文件內容：\n\n" if language == "zh" else "Here is the document to analyze:\n\n")
        + document_text
    )

    response = client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_output_tokens=2500,
    )

    return response.output_text


# =============================
# 工具：後續追問 / Q&A
# =============================
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
        system_prompt = f"""
你是一位 Error-Free® 文件與風險顧問，熟悉以下框架：
- {fw['name_zh']}

你已經對這份文件做過一次完整的框架分析，現在使用者想要在此基礎上進一步追問，
請你扮演「後續諮詢顧問」，根據原始文件與先前的分析結果，回答使用者的新問題。

請特別注意：
- 不要重新產生一整份完整分析報告，也不要重覆貼出長篇的舊內容。
- 以「補充說明、優先排序、具體寫法、範例句子、實務建議」為主。
- 回答要清楚、條列、有行動建議，可以引用少量原文作為說明。
        """
        doc_label = "以下是原始文件內容（節錄）："
        analysis_label = "以下是你過去產出的分析重點（節錄）："
        question_label = "使用者的新提問如下："
    else:
        system_prompt = f"""
You are an Error-Free® consultant, familiar with the following framework:
- {fw['name_en']}

You have already produced an initial structured analysis for this document.
Now the user is asking follow-up questions. Act as a consultant who knows
both the original document and your previous analysis.

Important:
- Do NOT re-generate a full analysis report.
- Focus on deeper explanation, prioritization, concrete wording, examples,
  and practical recommendations.
- Keep the answer structured and actionable. You may quote short phrases
  from the document or your previous analysis, but avoid long repetition.
        """
        doc_label = "Here is an excerpt of the original document:"
        analysis_label = "Here is an excerpt of your previous analysis:"
        question_label = "The user's new question is:"

    # 控制長度，避免爆 token
    max_doc_chars = 8000
    max_analysis_chars = 8000
    doc_excerpt = document_text[:max_doc_chars]
    analysis_excerpt = analysis_output[:max_analysis_chars]

    user_content = (
        f"{doc_label}\n{doc_excerpt}\n\n"
        f"{analysis_label}\n{analysis_excerpt}\n\n"
        f"{question_label}\n{user_question}"
    )

    response = client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_output_tokens=2000,
    )

    return response.output_text


# =============================
# 工具：組合「完整報告」文字（分析 + 後續 Q&A）
# =============================
def build_full_report_text(
    lang: str,
    framework_key: str,
    ai_output: str,
    followup_history: List[Tuple[str, str]],
) -> str:
    fw = FRAMEWORKS[framework_key]
    lines = []

    if lang == "zh":
        lines.append("=== Error-Free® 多框架 AI 文件分析報告 ===")
        lines.append(f"使用框架：{fw['name_zh']}")
        lines.append("")
        lines.append("【一、第一次 AI 分析結果】")
        lines.append(ai_output)
        if followup_history:
            lines.append("")
            lines.append("【二、後續問答紀錄】")
            for i, (q, a) in enumerate(followup_history, start=1):
                lines.append(f"\nQ{i}：{q}\n")
                lines.append(f"A{i}：\n{a}\n")
    else:
        lines.append("=== Error-Free® Multi-framework AI Document Analysis Report ===")
        lines.append(f"Framework: {fw['name_en']}")
        lines.append("")
        lines.append("【1. Initial AI analysis】")
        lines.append(ai_output)
        if followup_history:
            lines.append("")
            lines.append("【2. Follow-up Q&A】")
            for i, (q, a) in enumerate(followup_history, start=1):
                lines.append(f"\nQ{i}: {q}\n")
                lines.append(f"A{i}:\n{a}\n")

    return "\n".join(lines)


# =============================
# Streamlit 主程式
# =============================
def main():
    st.set_page_config(
        page_title="Error-Free Multi-framework AI Document Analyzer",
        layout="wide",
    )

    # Session 初始狀態
    if "user_email" not in st.session_state:
        st.session_state.user_email = None
        st.session_state.user_role = None
        st.session_state.usage_date = None
        st.session_state.usage_count = 0

    if "lang" not in st.session_state:
        st.session_state.lang = "zh"

    # 用來支援後續追問與下載
    if "last_doc_text" not in st.session_state:
        st.session_state.last_doc_text = ""
    if "last_framework_key" not in st.session_state:
        st.session_state.last_framework_key = None
    if "last_analysis_output" not in st.session_state:
        st.session_state.last_analysis_output = ""
    if "followup_history" not in st.session_state:
        st.session_state.followup_history = []

    # ------------ 側邊欄：語言切換 ------------
    with st.sidebar:
        current_lang = st.session_state.lang

        st.markdown("### Language" if current_lang == "en" else "### 語言")

        new_lang = st.radio(
            "Select language" if current_lang == "en" else "選擇語言",
            ["zh", "en"],
            index=0 if current_lang == "zh" else 1,
            format_func=lambda x: "繁體中文" if x == "zh" else "English",
        )

        st.session_state.lang = new_lang
        lang = new_lang

    # ------------ 標題區（主畫面上方） ------------
    logo_path = Path(__file__).parent / "logo.png"
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        if logo_path.exists():
            st.image(str(logo_path), use_column_width=True)

    with col_title:
        if lang == "zh":
            st.title("零錯誤多框架 AI 文件分析器")
            st.caption("邱強博士零錯誤研發團隊1987年至今")
        else:
            st.title("Error-Free Multi-framework AI Document Analyzer")
            st.caption("Dr. Chong Chiu’s Error-Free Team — Advancing Error-Free Practices")

    st.markdown("---")

    # ------------ 再回到側邊欄：帳號 / 框架資訊 ------------
    with st.sidebar:
        # 帳號區
        st.markdown("### Account" if lang == "en" else "### 帳號")

        # 已登入
        if st.session_state.user_email:
            email = st.session_state.user_email
            role = st.session_state.user_role
            model_name, daily_limit = get_model_and_limit(role)

            if lang == "zh":
                if role == "free":
                    role_label = "Free（一般測試）"
                elif role == "advanced":
                    role_label = "Advanced（進階）"
                elif role == "pro":
                    role_label = "Pro（內部專業帳號）"
                else:
                    role_label = "Admin（管理者）"
                st.success(f"已登入：{email}（{role_label}）")
                if st.button("登出"):
                    st.session_state.user_email = None
                    st.session_state.user_role = None
                    st.session_state.usage_date = None
                    st.session_state.usage_count = 0
                    st.session_state.last_doc_text = ""
                    st.session_state.last_framework_key = None
                    st.session_state.last_analysis_output = ""
                    st.session_state.followup_history = []
                    st.experimental_rerun()
            else:
                if role == "free":
                    role_label = "Free (basic)"
                elif role == "advanced":
                    role_label = "Advanced"
                elif role == "pro":
                    role_label = "Pro (internal)"
                else:
                    role_label = "Admin"
                st.success(f"Signed in as: {email} ({role_label})")
                if st.button("Log out"):
                    st.session_state.user_email = None
                    st.session_state.user_role = None
                    st.session_state.usage_date = None
                    st.session_state.usage_count = 0
                    st.session_state.last_doc_text = ""
                    st.session_state.last_framework_key = None
                    st.session_state.last_analysis_output = ""
                    st.session_state.followup_history = []
                    st.experimental_rerun()
        else:
            # 尚未登入
            login_email = st.text_input("Email")

            if lang == "zh":
                login_password = st.text_input("密碼", type="password")
                if st.button("登入"):
                    account = ACCOUNTS.get(login_email)
                    if account and account["password"] == login_password:
                        st.session_state.user_email = login_email
                        st.session_state.user_role = account["role"]
                        st.session_state.usage_date = None
                        st.session_state.usage_count = 0
                        st.session_state.last_doc_text = ""
                        st.session_state.last_framework_key = None
                        st.session_state.last_analysis_output = ""
                        st.session_state.followup_history = []
                        st.success("登入成功！")
                        st.experimental_rerun()
                    else:
                        st.error("Email 或密碼錯誤。")
                st.info("目前僅開放授權帳號使用，請向管理者申請帳號。")
            else:
                login_password = st.text_input("Password", type="password")
                if st.button("Log in"):
                    account = ACCOUNTS.get(login_email)
                    if account and account["password"] == login_password:
                        st.session_state.user_email = login_email
                        st.session_state.user_role = account["role"]
                        st.session_state.usage_date = None
                        st.session_state.usage_count = 0
                        st.session_state.last_doc_text = ""
                        st.session_state.last_framework_key = None
                        st.session_state.last_analysis_output = ""
                        st.session_state.followup_history = []
                        st.success("Login successful.")
                        st.experimental_rerun()
                    else:
                        st.error("Invalid email or password.")
                st.info("Access is limited to authorized accounts. Please contact the administrator.")

        st.markdown("---")

        # 框架選擇
        if lang == "zh":
            st.markdown("### 選擇框架")
            framework_label = "框架"
        else:
            st.markdown("### Analysis framework")
            framework_label = "Framework"

        framework_key = st.selectbox(
            framework_label,
            options=list(FRAMEWORKS.keys()),
            format_func=lambda k: FRAMEWORKS[k]["name_zh"]
            if lang == "zh"
            else FRAMEWORKS[k]["name_en"],
        )

        # 使用次數資訊（登入後才顯示）
        if st.session_state.user_role is not None:
            model_name, daily_limit = get_model_and_limit(st.session_state.user_role)
            today = datetime.date.today()
            if st.session_state.usage_date != today:
                today_usage = 0
            else:
                today_usage = st.session_state.usage_count
            remaining = None if daily_limit is None else max(daily_limit - today_usage, 0)

            if lang == "zh":
                st.caption(f"目前使用模型：**{model_name}**")
                if daily_limit is None:
                    st.caption(f"今日已用次數：{today_usage}（無上限）")
                else:
                    st.caption(
                        f"今日已用次數：{today_usage} / {daily_limit}，剩餘：{remaining}"
                    )
            else:
                st.caption(f"Current model: **{model_name}**")
                if daily_limit is None:
                    st.caption(f"Today used: {today_usage} (no limit)")
                else:
                    st.caption(
                        f"Today used: {today_usage} / {daily_limit}; remaining: {remaining}"
                    )

        # Admin 額外資訊
        if st.session_state.user_role == "admin":
            st.markdown("---")
            if lang == "zh":
                st.markdown("### 管理者資訊（本 Session）")
                st.write(
                    f"Session 使用日期 = {st.session_state.usage_date}, "
                    f"今日次數 = {st.session_state.usage_count}"
                )
                if st.button("重置本 Session 今日次數"):
                    today = datetime.date.today()
                    st.session_state.usage_date = today
                    st.session_state.usage_count = 0
                    st.success("已重置本 Session 的今日使用次數。")
            else:
                st.markdown("### Admin panel (this session)")
                st.write(
                    f"Session usage_date = {st.session_state.usage_date}, "
                    f"today usage_count = {st.session_state.usage_count}"
                )
                if st.button("Reset today's usage (this session)"):
                    today = datetime.date.today()
                    st.session_state.usage_date = today
                    st.session_state.usage_count = 0
                    st.success("Today's usage has been reset for this session.")

    # ------------ 主畫面：框架說明 + 上傳 ------------
    fw = FRAMEWORKS[framework_key]

    if lang == "zh":
        st.markdown(
            "此平台專為專案文件、技術方案與關鍵溝通內容設計，"
            "結合 Error-Free® 多種專業框架與 OpenAI 模型，協助你在事前發現遺漏與風險，降低錯誤成本。"
        )
        st.caption(f"目前選用框架：{fw['name_zh']}")
        st.markdown(f"**框架說明：** {fw['description_zh']}")
        upload_label = "上傳要分析的文件（支援 PDF、Word .docx、純文字 .txt）"
        start_button_label = "開始進行 AI 分析"
        warn_no_file = "請先上傳一個文件。"
        warn_no_login = "請先登入授權帳號，才可執行分析。"
        result_title = "AI 分析結果"
        preview_label = "文件預覽"
    else:
        st.markdown(
            "This platform is designed for project documents, technical proposals and critical communication. "
            "Powered by Error-Free® frameworks and OpenAI models, it helps you detect omissions and risks "
            "before they turn into costly errors."
        )
        st.caption(f"Current framework: {fw['name_en']}")
        st.markdown(f"**Framework description:** {fw['description_en']}")
        upload_label = "Upload a document (PDF, Word .docx, or plain .txt)"
        start_button_label = "Run AI analysis"
        warn_no_file = "Please upload a document first."
        warn_no_login = "Please log in with an authorized account before running analysis."
        result_title = "AI analysis result"
        preview_label = "Document preview"

    st.markdown("---")

    uploaded_file = st.file_uploader(
        upload_label,
        type=["pdf", "docx", "txt"],
    )

    if uploaded_file is not None:
        text = read_file_to_text(uploaded_file)
        if lang == "zh":
            st.info("✅ 文件已上傳並讀取完成。下方為前 1,000 字預覽，實際分析會使用更長內容。")
        else:
            st.info("✅ File uploaded and parsed. Below is a preview of the first 1,000 characters.")
        st.text_area(preview_label, value=text[:1000], height=200)
    else:
        text = ""

    # ------------ 第一次分析按鈕 ------------
    if st.button(start_button_label):
        if st.session_state.user_role is None:
            st.error(warn_no_login)
            return

        if not text:
            st.warning(warn_no_file)
            return

        today = datetime.date.today()
        if st.session_state.usage_date != today:
            st.session_state.usage_date = today
            st.session_state.usage_count = 0

        model_name, daily_limit = get_model_and_limit(st.session_state.user_role)

        if daily_limit is not None and st.session_state.usage_count >= daily_limit:
            if lang == "zh":
                st.error("今天的分析次數已達上限，請明天再試，或使用更高等級帳號。")
            else:
                st.error("You have reached today's analysis limit. Please try again tomorrow or use a higher tier.")
            return

        # 增加可分析字數上限（例如 120,000 字）
        max_chars = 120000
        if len(text) > max_chars:
            text_to_use = text[:max_chars]
            truncated = True
        else:
            text_to_use = text
            truncated = False

        with st.spinner("正在進行 AI 分析..." if lang == "zh" else "Running AI analysis..."):
            ai_output = run_llm_analysis(
                framework_key, lang, text_to_use, model_name
            )

        st.session_state.usage_count += 1

        # 儲存，供後續追問與下載使用
        st.session_state.last_doc_text = text_to_use
        st.session_state.last_framework_key = framework_key
        st.session_state.last_analysis_output = ai_output
        st.session_state.followup_history = []

        st.markdown("---")
        st.subheader(result_title)
        st.write(ai_output)

        # 下載按鈕（第一次分析就可以下載）
        full_report_text = build_full_report_text(
            lang,
            framework_key,
            ai_output,
            st.session_state.followup_history,
        )
        filename = (
            "ErrorFree_report.txt" if lang == "en" else "ErrorFree_零錯誤分析報告.txt"
        )
        st.download_button(
            label="Download full report" if lang == "en" else "下載完整報告（文字檔）",
            data=full_report_text.encode("utf-8"),
            file_name=filename,
            mime="text/plain",
        )

        if truncated:
            if lang == "zh":
                st.caption(
                    f"（提示：文件篇幅較長，為了確保穩定度與成本可控，本次僅分析前 {max_chars:,} 個字元。"
                    "若有更長篇的專案文件，可考慮分段上傳分析。）"
                )
            else:
                st.caption(
                    f"(Note: The document is long. To ensure stability and cost control, "
                    f"only the first {max_chars:,} characters were analyzed. "
                    "For very long documents, consider splitting into multiple uploads.)"
                )

    # ------------ 後續追問 / Q&A 區塊 ------------
    if st.session_state.last_analysis_output:
        st.markdown("---")
        if lang == "zh":
            st.subheader("後續互動 / 深入提問")
            st.markdown(
                "根據 **上方的 AI 分析結果** 與 **原始文件內容**，你可以在這裡向「Error-Free® 文件顧問」"
                "提出進一步問題，例如：優先順序、補寫句子、模擬客戶問題、實務落地建議等。"
            )
        else:
            st.subheader("Follow-up Q&A")
            st.markdown(
                "Based on the **analysis above** and the **original document**, you can ask "
                "follow-up questions to the Error-Free® consultant here. For example: "
                "prioritization, suggested wording, example text, or practical actions."
            )

        # 顯示歷史問答
        if st.session_state.followup_history:
            for i, (q, a) in enumerate(st.session_state.followup_history, start=1):
                if lang == "zh":
                    st.markdown(f"**Q{i}：{q}**")
                    st.markdown(f"**A{i}：**")
                else:
                    st.markdown(f"**Q{i}: {q}**")
                    st.markdown(f"**A{i}:**")
                st.write(a)
                st.markdown("---")

        # 問題輸入
        if lang == "zh":
            followup_question = st.text_area(
                "請輸入你想追問的問題（與本次文件與分析相關）",
                key="followup_input",
            )
            ask_label = "送出提問"
            warn_need_q = "請先輸入一個問題。"
            warn_need_login = "請先登入授權帳號，才可使用後續提問功能。"
        else:
            followup_question = st.text_area(
                "Enter your follow-up question (related to this document and analysis)",
                key="followup_input",
            )
            ask_label = "Send question"
            warn_need_q = "Please enter a question first."
            warn_need_login = "Please log in before using follow-up Q&A."

        if st.button(ask_label):
            if st.session_state.user_role is None:
                st.error(warn_need_login)
            elif not followup_question.strip():
                st.warning(warn_need_q)
            else:
                model_name, _ = get_model_and_limit(st.session_state.user_role)
                with st.spinner("正在產生回覆..." if lang == "zh" else "Generating answer..."):
                    answer = run_followup_qa(
                        framework_key=st.session_state.last_framework_key,
                        language=lang,
                        document_text=st.session_state.last_doc_text,
                        analysis_output=st.session_state.last_analysis_output,
                        user_question=followup_question.strip(),
                        model_name=model_name,
                    )
                st.session_state.followup_history.append(
                    (followup_question.strip(), answer)
                )

                # 重新組合「完整報告」供下載（含新的 Q&A）
                full_report_text = build_full_report_text(
                    lang,
                    st.session_state.last_framework_key,
                    st.session_state.last_analysis_output,
                    st.session_state.followup_history,
                )
                filename = (
                    "ErrorFree_report.txt" if lang == "en" else "ErrorFree_零錯誤分析報告.txt"
                )
                st.download_button(
                    label="Download full report" if lang == "en" else "下載完整報告（文字檔）",
                    data=full_report_text.encode("utf-8"),
                    file_name=filename,
                    mime="text/plain",
                )

                # 立刻顯示最新一輪 Q&A
                if lang == "zh":
                    st.markdown(f"**Q{len(st.session_state.followup_history)}：{followup_question.strip()}**")
                    st.markdown(f"**A{len(st.session_state.followup_history)}：**")
                else:
                    st.markdown(f"**Q{len(st.session_state.followup_history)}: {followup_question.strip()}**")
                    st.markdown(f"**A{len(st.session_state.followup_history)}:**")
                st.write(answer)
                st.markdown("---")

        # 即使沒有按新問題，也提供最新的下載按鈕
        full_report_text = build_full_report_text(
            lang,
            st.session_state.last_framework_key,
            st.session_state.last_analysis_output,
            st.session_state.followup_history,
        )
        filename = (
            "ErrorFree_report.txt" if lang == "en" else "ErrorFree_零錯誤分析報告.txt"
        )
        st.download_button(
            label="Download full report" if lang == "en" else "下載完整報告（文字檔）",
            data=full_report_text.encode("utf-8"),
            file_name=filename,
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
