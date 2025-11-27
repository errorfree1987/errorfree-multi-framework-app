import os
import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from io import BytesIO

import streamlit as st
import pdfplumber
from docx import Document
from openai import OpenAI

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


# =========================================
# 工具：轉成 Word（.docx）與 PDF（.pdf）
# =========================================
def build_full_report_docx(lang: str) -> bytes:
    """
    用 python-docx 把完整報告轉成 .docx
    """
    text = build_full_report(lang)
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()


def build_full_report_pdf(lang: str):
    """
    嘗試用 reportlab 產生簡單的 PDF。
    若未安裝 reportlab，回傳 (None, error_message)
    若成功，回傳 (pdf_bytes, None)
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError:
        return None, "目前伺服器尚未安裝 reportlab，因此無法產生 PDF 檔案。請管理者於 requirements.txt 加入 reportlab 再重新部署。"

    text = build_full_report(lang)
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    x = 40
    y = height - 40
    line_height = 14

    for line in text.split("\n"):
        if y < 40:
            c.showPage()
            y = height - 40
        # 避免行太長出界，簡單截斷
        c.drawString(x, y, line[:2000])
        y -= line_height

    c.save()
    buf.seek(0)
    return buf.getvalue(), None


# =========================================
# Streamlit 主程式
# =========================================
def main():
    st.set_page_config(
        page_title="Error-Free Multi-framework AI Document Analyzer",
        layout="wide",
    """


::contentReference[oaicite:1]{index=1}
