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

# =========================================
# OpenAI client（從環境變數讀取 API Key）
# =========================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# =========================================
# 簡易帳號系統（你可以在這裡改帳號密碼）
# =========================================
ACCOUNTS = {
    # 管理者帳號
    "admin@errorfree.com": {
        "password": "1111",
        "role": "admin",
    },
    # 內部 Pro 使用者
    "dr.chiu@errorfree.com": {
        "password": "2222",
        "role": "pro",
    },
    # 內部測試用帳號（不是匿名 guest）
    "guest@errorfree.com": {
        "password": "3333",
        "role": "pro",
    },
}


# =========================================
# 依角色選模型 + 每日次數上限
# =========================================
def get_model_and_limit(role: str):
    """
    回傳 (model_name, daily_limit)
    daily_limit = None 表示無上限
    """
    if role == "free":
        return "gpt-4.1-mini", 5
    if role == "advanced":
        return "gpt-4.1", 10
    if role == "pro":
        return "gpt-5.1", None
    if role == "admin":
        return "gpt-5.1", None
    # 預設
    return "gpt-4.1-mini", 2


# =========================================
# Multi-framework 設定（可再增加）
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
3. Risk level and impact (High/Medium/Low, with short justification)
4. Concrete mitigation suggestions (design changes, additional checks, tests, documentation, etc.)
5. Additionally, output a Markdown table with columns:
   | Risk item | Risk level (High/Medium/Low) | Impact description | Mitigation suggestion |
Each row should correspond to one important risk that should be prioritized for follow-up.
        """,
    },
}

# =========================================
# 狀態持久化：離開網頁也保留
# =========================================
STATE_FILE = Path("user_state.json")


def save_state_to_disk():
    """把重要狀態存到本機檔案，下次開頁可以自動回復。"""
    data = {
        "user_email": st.session_state.get("user_email"),
        "user_role": st.session_state.get("user_role"),
        "is_authenticated": st.session_state.get("is_authenticated", False),
        "lang": st.session_state.get("lang", "zh"),
        "usage_date": st.session_state.get("usage_date"),
        "usage_count": st.session_state.get("usage_count", 0),
        "last_doc_text": st.session_state.get("last_doc_text", ""),
        "last_framework_key": st.session_state.get("last_framework_key"),
        "last_analysis_output": st.session_state.get("last_analysis_output", ""),
        "followup_history": st.session_state.get("followup_history", []),
        "analysis_done": st.session_state.get("analysis_done", False),
    }
    try:
        STATE_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def restore_state_from_disk():
    """如果有存過狀態，開啟網頁時自動回復，不用再登入。"""
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


# =========================================
# 工具：讀取上傳檔案文字
# =========================================
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


# =========================================
# 工具：第一次框架分析
# =========================================
def run_llm_analysis(
    framework_key: str,
    language: str,
    document_text: str,
    model_name: str,
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


# =========================================
# 工具：後續追問 / Q&A
# =========================================
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


# =========================================
# 工具：組合「完整報告」文字，供下載
# =========================================
def build_full_report(lang: str) -> str:
    doc_text = st.session_state.get("last_doc_text", "")
    framework_key = st.session_state.get("last_framework_key")
    analysis_output = st.session_state.get("last_analysis_output", "")
    followup_history: List[Tuple[str, str]] = st.session_state.get(
        "followup_history", []
    )

    if not framework_key:
        return ""

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
    """將報告文字轉為 DOCX bytes"""
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()


def build_pdf_bytes(text: str) -> bytes:
    """將報告文字轉為簡單 PDF bytes（需要 reportlab）"""
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    y = height - 40
    for line in text.split("\n"):
        # 避免太長超出右邊，簡單裁切
        c.drawString(40, y, line[:1000])
        y -= 14
        if y < 40:
            c.showPage()
            y = height - 40
    c.save()
    buf.seek(0)
    return buf.getvalue()


# =========================================
# Streamlit 主程式
# =========================================
def main():
    st.set_page_config(
        page_title="Error-Free Multi-framework AI Document Analyzer",
        layout="wide",
    )

    # ------- 先嘗試從硬碟還原上次狀態（避免重新登入與資料遺失） -------
    restore_state_from_disk()

    # -------- Session 初始狀態 --------
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
    if "last_framework_key" not in st.session_state:
        st.session_state.last_framework_key = None
    if "last_analysis_output" not in st.session_state:
        st.session_state.last_analysis_output = ""
    if "followup_history" not in st.session_state:
        st.session_state.followup_history = []
    if "analysis_truncated" not in st.session_state:
        st.session_state.analysis_truncated = False
    if "analysis_max_chars" not in st.session_state:
        st.session_state.analysis_max_chars = 0
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False

    # =====================================
    # 側邊欄：語言切換（有分析時鎖定語言）+ 登入狀態
    # =====================================
    with st.sidebar:
        current_lang = st.session_state.lang
        analysis_active = bool(st.session_state.last_analysis_output)

        if analysis_active:
            # 分析進行中：語言鎖定，不顯示 radio
            if current_lang == "zh":
                st.markdown("### 語言")
                st.caption("目前語言：繁體中文（本次分析期間無法切換語言）")
            else:
                st.markdown("### Language")
                st.caption(
                    "Current language: English (language is locked while this analysis is active)."
                )
            lang = current_lang
        else:
            # 尚未分析，可以自由切換語言
            if current_lang == "en":
                st.markdown("### Language")
                new_lang = st.radio(
                    "Select language",
                    ["zh", "en"],
                    index=1 if current_lang == "en" else 0,
                    format_func=lambda x: "Chinese" if x == "zh" else "English",
                )
            else:
                st.markdown("### 語言")
                new_lang = st.radio(
                    "請選擇介面語言",
                    ["zh", "en"],
                    index=0 if current_lang == "zh" else 1,
                    format_func=lambda x: "繁體中文" if x == "zh" else "English",
                )
            st.session_state.lang = new_lang
            lang = new_lang
            save_state_to_disk()

        # 登入資訊區
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
                    st.session_state.last_framework_key = None
                    st.session_state.last_analysis_output = ""
                    st.session_state.followup_history = []
                    st.session_state.analysis_done = False
                    if STATE_FILE.exists():
                        try:
                            STATE_FILE.unlink()
                        except Exception:
                            pass
                    st.experimental_rerun()
            else:
                st.subheader("Account")
                st.write(f"User: {st.session_state.user_email}")
                st.write(f"Role: {st.session_state.user_role}")
                if st.button("Logout"):
                    st.session_state.user_email = None
                    st.session_state.user_role = None
                    st.session_state.is_authenticated = False
                    st.session_state.last_doc_text = ""
                    st.session_state.last_framework_key = None
                    st.session_state.last_analysis_output = ""
                    st.session_state.followup_history = []
                    st.session_state.analysis_done = False
                    if STATE_FILE.exists():
                        try:
                            STATE_FILE.unlink()
                        except Exception:
                            pass
                    st.experimental_rerun()
        else:
            if lang == "zh":
                st.subheader("尚未登入")
                st.caption("請在主畫面輸入帳號密碼登入")
            else:
                st.subheader("Not logged in")
                st.caption("Please log in on the main page.")

    # =====================================
    # 主畫面：登入 + 分析介面
    # =====================================
    col_main = st.container()

    with col_main:
        # ======= 尚未登入：顯示登入（按一次就好） =======
        if not st.session_state.is_authenticated:
            if lang == "zh":
                st.title("Error-Free® 多框架 AI 文件分析")
                st.subheader("請先登入")
            else:
                st.title("Error-Free® Multi-framework AI Document Analyzer")
                st.subheader("Please log in first")

            # 不用 form，避免「要按兩次」的行為
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
                    # 重設當日使用次數
                    st.session_state.usage_date = datetime.date.today().isoformat()
                    st.session_state.usage_count = 0
                    # 清空舊分析
                    st.session_state.last_doc_text = ""
                    st.session_state.last_framework_key = None
                    st.session_state.last_analysis_output = ""
                    st.session_state.followup_history = []
                    st.session_state.analysis_done = False
                    save_state_to_disk()
                    # 立刻跳轉主畫面（不留在登入頁）
                    st.experimental_rerun()
                else:
                    if lang == "zh":
                        st.error("帳號或密碼錯誤，請再試一次。")
                    else:
                        st.error("Invalid email or password. Please try again.")

            return  # 未登入時不顯示後續內容

        # ======= 已登入：顯示主功能畫面 =======
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

        # 顯示當日使用次數＆模型資訊
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

        # ======= 文件上傳 & 框架選擇 =======
        if lang == "zh":
            st.subheader("步驟一：上傳文件或貼上文字")
            uploaded_file = st.file_uploader(
                "請上傳 PDF / DOCX / TXT 檔案（擇一），或在下方文字框貼上內容",
                type=["pdf", "docx", "txt"],
            )
            manual_text = st.text_area(
                "或直接在此貼上要分析的文字內容（如果同時上傳檔案與貼文字，系統以上傳檔案為主）",
                height=180,
            )
            st.subheader("步驟二：選擇分析框架")
        else:
            st.subheader("Step 1: Upload a document or paste text")
            uploaded_file = st.file_uploader(
                "Upload a PDF / DOCX / TXT file, or paste your content below",
                type=["pdf", "docx", "txt"],
            )
            manual_text = st.text_area(
                "Or paste the text to analyze (if both file and text are provided, the file will take precedence).",
                height=180,
            )
            st.subheader("Step 2: Choose an analysis framework")

        framework_keys = list(FRAMEWORKS.keys())
        if lang == "zh":
            framework_labels = [FRAMEWORKS[k]["name_zh"] for k in framework_keys]
            framework_label_to_key = dict(zip(framework_labels, framework_keys))
            selected_label = st.selectbox("請選擇框架", framework_labels)
            selected_framework_key = framework_label_to_key[selected_label]
        else:
            framework_labels = [FRAMEWORKS[k]["name_en"] for k in framework_keys]
            framework_label_to_key = dict(zip(framework_labels, framework_keys))
            selected_label = st.selectbox("Select framework", framework_labels)
            selected_framework_key = framework_label_to_key[selected_label]

        st.markdown("---")

        # ======= Run Analysis（只允許跑一次） =======
        # 一旦按下去就鎖死，避免重複分析
        can_run_analysis = not st.session_state.analysis_done

        if can_run_analysis:
            if lang == "zh":
                run_button = st.button("開始分析（Run Analysis）")
            else:
                run_button = st.button("Run Analysis")
        else:
            run_button = False
            if lang == "zh":
                st.info("已對本份文件完成一次分析。如要重新分析，請上傳新文件並按下 Reset。")
            else:
                st.info(
                    "Analysis for this document has already been run once. To analyze a new document, upload a new file and click Reset."
                )

        # Reset 功能：用來重新上傳新文件與重新分析
        if lang == "zh":
            reset_clicked = st.button("Reset / 重新開始（新文件）")
        else:
            reset_clicked = st.button("Reset / Start with new document")

        if reset_clicked:
            st.session_state.last_doc_text = ""
            st.session_state.last_framework_key = None
            st.session_state.last_analysis_output = ""
            st.session_state.followup_history = []
            st.session_state.analysis_done = False
            save_state_to_disk()
            st.experimental_rerun()

        # 執行分析
        if run_button and can_run_analysis:
            # 按下的瞬間就鎖死，避免被按第二次
            st.session_state.analysis_done = True
            save_state_to_disk()

            # 檢查每日使用次數限制
            if daily_limit is not None and st.session_state.usage_count >= daily_limit:
                if lang == "zh":
                    st.error("已達今日使用次數上限。請明天再試，或聯絡管理者提升方案。")
                else:
                    st.error(
                        "You have reached your daily usage limit. Please try again tomorrow or contact the admin."
                    )
            else:
                # 取得文件文字
                if uploaded_file is not None:
                    doc_text = read_file_to_text(uploaded_file)
                else:
                    doc_text = manual_text.strip()

                if not doc_text:
                    if lang == "zh":
                        st.error("請至少上傳一份文件或貼上一些文字內容。")
                    else:
                        st.error("Please upload a document or paste some text to analyze.")
                    # 如果失敗，解鎖分析權限
                    st.session_state.analysis_done = False
                    save_state_to_disk()
                else:
                    with st.spinner("Running analysis..."):
                        analysis_text = run_llm_analysis(
                            framework_key=selected_framework_key,
                            language=lang,
                            document_text=doc_text,
                            model_name=model_name,
                        )

                    st.session_state.last_doc_text = doc_text
                    st.session_state.last_framework_key = selected_framework_key
                    st.session_state.last_analysis_output = analysis_text
                    st.session_state.followup_history = []
                    st.session_state.usage_count += 1
                    save_state_to_disk()

                    if lang == "zh":
                        st.success("分析完成！下方顯示第一次框架分析結果。")
                    else:
                        st.success("Analysis completed! See the initial result below.")

        # ======= 顯示第一次分析結果（最上方） =======
        if st.session_state.last_analysis_output:
            if lang == "zh":
                st.subheader("第一次框架分析結果")
            else:
                st.subheader("Initial framework analysis result")

            st.markdown(st.session_state.last_analysis_output)

        # 後續區塊 (Q&A 歷史 + Download + chat_input) 只在分析完成後出現
        followup_q = None

        if st.session_state.analysis_done and st.session_state.last_analysis_output:
            st.markdown("---")
            # ======= Q&A 歷史（中間區塊） =======
            if lang == "zh":
                st.subheader("後續提問歷史（Q&A）")
                st.caption("以下是你與系統之間的所有追問與回覆。")
            else:
                st.subheader("Follow-up Q&A history")
                st.caption("All your follow-up questions and answers are listed below.")

            if st.session_state.followup_history:
                for idx, (q, a) in enumerate(st.session_state.followup_history, start=1):
                    st.markdown(f"**Q{idx}:** {q}")
                    st.markdown(f"**A{idx}:** {a}")
                    st.markdown("---")
            else:
                if lang == "zh":
                    st.info("目前尚未有任何追問，歡迎在下方聊天框提出你的問題。")
                else:
                    st.info("No follow-up questions yet. Use the chat box below to ask.")

            # ======= Download 區塊（TXT / Word / PDF） =======
            st.markdown("---")
            if lang == "zh":
                st.subheader("下載完整報告（TXT / Word / PDF）")
            else:
                st.subheader("Download full report (TXT / Word / PDF)")

            report_text = build_full_report(lang)
            if report_text:
                now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                txt_bytes = report_text.encode("utf-8")
                docx_bytes = build_docx_bytes(report_text)
                pdf_bytes = build_pdf_bytes(report_text)

                col_txt, col_docx, col_pdf = st.columns(3)
                with col_txt:
                    st.download_button(
                        label="TXT",
                        data=txt_bytes,
                        file_name=f"errorfree_report_{now_str}.txt",
                        mime="text/plain",
                    )
                with col_docx:
                    st.download_button(
                        label="Word (DOCX)",
                        data=docx_bytes,
                        file_name=f"errorfree_report_{now_str}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                with col_pdf:
                    st.download_button(
                        label="PDF",
                        data=pdf_bytes,
                        file_name=f"errorfree_report_{now_str}.pdf",
                        mime="application/pdf",
                    )

        # ======= 最底部：聊天輸入框（st.chat_input，一直固定在底端） =======
        if st.session_state.analysis_done and st.session_state.last_analysis_output:
            if lang == "zh":
                prompt = "請輸入你的追問（像 ChatGPT 一樣可以一直問）"
            else:
                prompt = "Enter your follow-up question (you can keep asking)"

            followup_q = st.chat_input(prompt)

            if followup_q:
                with st.spinner("Thinking..."):
                    answer = run_followup_qa(
                        framework_key=st.session_state.last_framework_key,
                        language=lang,
                        document_text=st.session_state.last_doc_text,
                        analysis_output=st.session_state.last_analysis_output,
                        user_question=followup_q.strip(),
                        model_name=model_name,
                    )
                st.session_state.followup_history.append(
                    (followup_q.strip(), answer)
                )
                save_state_to_disk()
                # 立即在畫面上加一條（這一輪 rerun 的結尾）
                st.markdown("---")
                idx = len(st.session_state.followup_history)
                st.markdown(f"**Q{idx}:** {followup_q.strip()}")
                st.markdown(f"**A{idx}:** {answer}")

    # 最後再保險存一次狀態
    save_state_to_disk()


if __name__ == "__main__":
    main()
