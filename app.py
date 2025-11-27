import os
import datetime
from typing import Dict

import streamlit as st
import pdfplumber
from docx import Document
from openai import OpenAI

# ------------------------------------
# OpenAI 設定（讀取 Railway 的環境變數）
# ------------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------------------------
# 簡易帳號系統（你可以在這裡新增/修改帳號）
# ------------------------------------
# role 說明：
#   - "guest": 訪客（無帳號），使用 gpt-4.1-mini，每天 2 次
#   - "free":  一般註冊 / 測試帳號，gpt-4.1-mini，每天 5 次
#   - "advanced": 進階帳號，gpt-4.1，每天 10 次
#   - "pro":  企業 / 內部專業帳號，gpt-5.1，次數不限（或你自己限制）
#   - "admin": 管理者帳號，與 pro 同等，但多顯示管理資訊
ACCOUNTS = {
    "free@example.com": {
        "password": "free123",
        "role": "free",
    },
    "advanced@example.com": {
        "password": "adv123",
        "role": "advanced",
    },
    "pro@example.com": {
        "password": "pro123",
        "role": "pro",
    },
    "admin@example.com": {
        "password": "admin123",
        "role": "admin",
    },
}

# ------------------------------------
# 多框架設定
# ------------------------------------
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

# ------------------------------------
# 工具：從上傳檔案讀取文字
# ------------------------------------
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


# ------------------------------------
# 工具：依照角色決定模型與次數上限
# ------------------------------------
def get_model_and_limit(role: str):
    """
    回傳 (model_name, daily_limit)
    role 可以是 "guest", "free", "advanced", "pro", "admin"
    """
    if role == "guest":
        return "gpt-4.1-mini", 2
    if role == "free":
        return "gpt-4.1-mini", 5
    if role == "advanced":
        return "gpt-4.1", 10
    if role == "pro":
        return "gpt-5.1", None  # None 代表不限制
    if role == "admin":
        return "gpt-5.1", None
    # fallback
    return "gpt-4.1-mini", 2


# ------------------------------------
# 工具：呼叫 OpenAI 進行分析
# ------------------------------------
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


# ------------------------------------
# Streamlit 介面
# ------------------------------------
def main():
    st.set_page_config(
        page_title="Error-Free® AI 文件風險稽核平台｜Multi-framework Analyzer",
        layout="wide",
    )

    # 初始化 session 狀態
    if "user_email" not in st.session_state:
        st.session_state.user_email = None
        st.session_state.user_role = "guest"
        st.session_state.usage_date = None
        st.session_state.usage_count = 0

    # 側邊欄：登入區 + 語言 + 框架 + 使用資訊
    with st.sidebar:
        st.markdown("### 帳號 / Account")

        if st.session_state.user_email:
            # 已登入
            email = st.session_state.user_email
            role = st.session_state.user_role
            if role == "guest":
                role_label = "訪客 Guest"
            elif role == "free":
                role_label = "Free"
            elif role == "advanced":
                role_label = "Advanced"
            elif role == "pro":
                role_label = "Pro"
            else:
                role_label = "Admin（管理者）"

            st.success(f"已登入：{email}（{role_label}）")
            if st.button("登出 / Log out"):
                st.session_state.user_email = None
                st.session_state.user_role = "guest"
                st.session_state.usage_date = None
                st.session_state.usage_count = 0
        else:
            # 尚未登入：顯示登入表單 + 訪客模式
            login_email = st.text_input("Email")
            login_password = st.text_input("密碼 / Password", type="password")

            if st.button("登入 / Log in"):
                account = ACCOUNTS.get(login_email)
                if account and account["password"] == login_password:
                    st.session_state.user_email = login_email
                    st.session_state.user_role = account["role"]
                    st.session_state.usage_date = None
                    st.session_state.usage_count = 0
                    st.success("登入成功！")
                else:
                    st.error("Email 或密碼錯誤。")

            if st.button("以訪客身份繼續 / Continue as guest"):
                st.session_state.user_email = "Guest"
                st.session_state.user_role = "guest"
                st.session_state.usage_date = None
                st.session_state.usage_count = 0
                st.info("目前以訪客模式使用（每天分析次數有限）。")

        st.markdown("---")

        st.markdown("### Language / 語言")
        lang = st.radio(
            "Language",
            ["zh", "en"],
            index=0,
            format_func=lambda x: "繁體中文" if x == "zh" else "English",
        )

        st.markdown("---")
        st.markdown("### 分析框架 / Analysis framework")

        framework_key = st.selectbox(
            "Framework",
            options=list(FRAMEWORKS.keys()),
            format_func=lambda k: FRAMEWORKS[k]["name_zh"]
            if lang == "zh"
            else FRAMEWORKS[k]["name_en"],
        )

        # 顯示目前使用的模型與次數限制
        model_name, daily_limit = get_model_and_limit(st.session_state.user_role)

        # 計算今日使用次數 / 剩餘次數
        today = datetime.date.today()
        if st.session_state.usage_date != today:
            # 這裡只更新顯示，不重設，重設會在按分析時真正處理
            today_usage = 0
        else:
            today_usage = st.session_state.usage_count

        if daily_limit is None:
            remaining = None
        else:
            remaining = max(daily_limit - today_usage, 0)

        if lang == "zh":
            st.caption(f"目前使用模型：**{model_name}**")
            if daily_limit is None:
                st.caption(f"今日已用次數：{today_usage}（無上限，請留意 API 成本）")
            else:
                st.caption(
                    f"今日已用次數：{today_usage} 次／上限 {daily_limit} 次；剩餘：{remaining} 次"
                )
        else:
            st.caption(f"Current model: **{model_name}**")
            if daily_limit is None:
                st.caption(f"Today used: {today_usage} (no daily limit; be mindful of API cost)")
            else:
                st.caption(
                    f"Today used: {today_usage} / {daily_limit}; remaining: {remaining}"
                )

        # 若為管理者，顯示一些額外資訊
        if st.session_state.user_role == "admin":
            st.markdown("---")
            st.markdown("### 管理者資訊 / Admin panel")
            st.write(
                f"Session 狀態：使用日期 = {st.session_state.usage_date}, "
                f"今日已用次數 = {st.session_state.usage_count}"
            )
            if st.button("重置本 Session 今日使用次數 / Reset today's usage for this session"):
                st.session_state.usage_date = today
                st.session_state.usage_count = 0
                st.success("已重置本 Session 的今日使用次數。")

    fw = FRAMEWORKS[framework_key]

    # 主畫面：商業版文案
    if lang == "zh":
        st.title("Error-Free® AI 文件風險稽核平台（多框架）")
        st.markdown(
            "這是一個專為專案文件、技術方案與關鍵溝通內容設計的 AI 輔助審查工具，"
            "結合 Error-Free® 的多種專業框架，協助你在事前發現遺漏與風險，降低錯誤成本。"
        )
        st.caption(f"目前選用框架：**{fw['name_zh']}**")
        st.markdown(f"**框架說明：** {fw['description_zh']}")
        upload_label = "上傳要分析的文件（支援 PDF、Word .docx、純文字 .txt）"
        start_button_label = "開始進行 AI 分析"
        warn_no_file = "請先上傳一個文件。"
        result_title = "AI 分析結果"
    else:
        st.title("Error-Free® AI Document Risk Audit Platform (Multi-framework)")
        st.markdown(
            "This platform is designed for project documents, technical proposals and critical communication. "
            "Powered by Error-Free® frameworks and OpenAI models, it helps you detect omissions and risks "
            "before they turn into costly errors."
        )
        st.caption(f"Current framework: **{fw['name_en']}**")
        st.markdown(f"**Framework description:** {fw['description_en']}")
        upload_label = "Upload a document (PDF, Word .docx, or plain .txt)"
        start_button_label = "Run AI analysis"
        warn_no_file = "Please upload a document first."
        result_title = "AI analysis result"

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
        st.text_area(
            "Document preview / 文件預覽",
            value=text[:1000],
            height=200,
        )
    else:
        text = ""

    # 按鈕：開始分析
    if st.button(start_button_label):
        if not text:
            st.warning(warn_no_file)
        else:
            # 檢查每日使用次數（在按下分析時才正式重置）
            today = datetime.date.today()
            if st.session_state.usage_date != today:
                st.session_state.usage_date = today
                st.session_state.usage_count = 0

            model_name, daily_limit = get_model_and_limit(st.session_state.user_role)

            if daily_limit is not None and st.session_state.usage_count >= daily_limit:
                # 超過今日上限
                if lang == "zh":
                    st.error("今天的分析次數已達上限，請明天再試，或使用更高等級帳號。")
                else:
                    st.error("You have reached today's analysis limit. Please try again tomorrow or use a higher plan.")
                return

            # 避免一次丟太多內容，設定較大的上限（字元）
            max_chars = 120000  # 提升上限：12 萬字元，足夠多數實務文件使用
            if len(text) > max_chars:
                text_to_use = text[:max_chars]
                truncated = True
            else:
                text_to_use = text
                truncated = False

            with st.spinner("Running AI analysis..."):
                ai_output = run_llm_analysis(
                    framework_key, lang, text_to_use, model_name
                )

            # 使用次數 +1
            st.session_state.usage_count += 1

            st.markdown("---")
            st.subheader(result_title)
            st.write(ai_output)

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


if __name__ == "__main__":
    main()
