import os
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
# 多框架設定
# ------------------------------------
FRAMEWORKS: Dict[str, Dict] = {
    "omission": {
        "name_zh": "Error-Free® 遺漏錯誤檢查框架",
        "name_en": "Error-Free® Omission Error Check Framework",
        "description_zh": (
            "使用 12 種遺漏檢查方法與 12 類遺漏錯誤型態（O1–O12），"
            "系統性檢查文件有沒有「該出現卻沒出現」的內容。"
        ),
        "description_en": (
            "Uses 12 omission-check methods and 12 omission error types (O1–O12) "
            "to systematically find content that SHOULD be present but is missing."
        ),
        "wrapper_zh": """
你是一位 Error-Free® 遺漏錯誤檢查專家，精通 12 種遺漏檢查方法與 12 類遺漏錯誤型態（O1–O12）。

請你扮演「文件顧問」，用這個框架來分析輸入的文件，找出：
1. 文件可能遺漏的重要內容、條件、假設、角色、步驟、風險或例外情況。
2. 這些遺漏，可能在實務上造成什麼後果（例如專案失敗、誤解、作業風險、客訴…）。
3. 文件撰寫人應該如何具體補強、修改或新增內容。

請特別注意：
- 不只是改錯字，而是針對「沒有寫出來」但應該要寫的地方。
- 你可以引用文件中的關鍵句子來說明，但不要原封不動複製太長的段落。
- 回答時請用條列方式，讓使用者可以直接依序修正文件。

輸出格式請用繁體中文，依照下列結構：

一、文件簡要摘要（3–5 行）
二、可能的遺漏錯誤（逐點列出，每點說明「哪裡遺漏」與「為何重要」）
三、具體修正建議（逐點對應上面的遺漏，提出可直接採用的補寫句子或段落）
        """,
        "wrapper_en": """
You are an Error-Free® omission error expert, using 12 omission-check methods
and 12 omission error types (O1–O12).

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

1. Brief summary of the document (3–5 sentences)
2. Potential omission errors (bullet list, each with “what is missing” and “why it matters”)
3. Concrete revision suggestions (bullet list with sample wording or specific guidance)
        """,
    },
    "technical": {
        "name_zh": "Error-Free® 技術風險檢查框架",
        "name_en": "Error-Free® Technical Risk Check Framework",
        "description_zh": (
            "從技術正確性、假設條件、邊界情況、相容性、安全性與可維護性等面向，"
            "檢查方案或文件中可能被忽略的技術風險。"
        ),
        "description_en": (
            "Checks technical risks from perspectives such as correctness, assumptions, "
            "edge cases, compatibility, safety and maintainability."
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

一、技術內容簡要摘要
二、可能的技術風險（逐點列出，每點說明風險內容與成因）
三、風險等級判斷與影響說明（高／中／低，並說明理由）
四、具體改善建議（可以是設計調整、補充檢查點、額外文件、測試建議等）
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

1. Brief summary of the technical content
2. Potential technical risks (bullet list, each with cause / context)
3. Risk level and impact (High / Medium / Low, with short justification)
4. Concrete mitigation suggestions (design changes, additional checks, tests, documentation, etc.)
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
# 工具：呼叫 OpenAI 進行分析
# ------------------------------------
def run_llm_analysis(
    framework_key: str, language: str, document_text: str
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
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_output_tokens=2000,
    )

    # responses API 的便利屬性，會把所有文字組合好
    return response.output_text


# ------------------------------------
# Streamlit 介面
# ------------------------------------
def main():
    st.set_page_config(
        page_title="Error-Free® Multi-framework AI Document Analyzer",
        layout="wide",
    )

    # 側邊欄：語言與框架選擇
    with st.sidebar:
        st.markdown("### Language / 語言")
        lang = st.radio("Language", ["zh", "en"], format_func=lambda x: "繁體中文" if x == "zh" else "English")

        st.markdown("---")
        st.markdown("### 選擇框架 / Choose framework")

        framework_key = st.selectbox(
            "Framework",
            options=list(FRAMEWORKS.keys()),
            format_func=lambda k: FRAMEWORKS[k]["name_zh"] if lang == "zh" else FRAMEWORKS[k]["name_en"],
        )

    fw = FRAMEWORKS[framework_key]

    # 主畫面
    if lang == "zh":
        st.title("Error-Free® 多框架 AI 文件分析器")
        st.caption(f"目前框架：{fw['name_zh']}")
        st.markdown(f"**說明：** {fw['description_zh']}")
        upload_label = "上傳要分析的文件（PDF, Word .docx, 或純文字 .txt）"
        start_button_label = "開始分析"
        warn_no_file = "請先上傳一個文件。"
        result_title = "分析結果"
    else:
        st.title("Error-Free® Multi-framework AI Document Analyzer")
        st.caption(f"Current framework: {fw['name_en']}")
        st.markdown(f"**Description:** {fw['description_en']}")
        upload_label = "Upload a document to analyze (PDF, Word .docx, or plain .txt)"
        start_button_label = "Start analysis"
        warn_no_file = "Please upload a document first."
        result_title = "Analysis result"

    st.markdown("---")

    uploaded_file = st.file_uploader(
        upload_label,
        type=["pdf", "docx", "txt"],
    )

    if uploaded_file is not None:
        text = read_file_to_text(uploaded_file)
        if lang == "zh":
            st.info("✅ 文件已上傳並讀取完成。下方可以看到前 1,000 字預覽。")
        else:
            st.info("✅ File uploaded and parsed. Preview of the first 1,000 characters below.")
        st.text_area(
            "Document preview",
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
            # 避免 token 過大，適度截斷
            max_chars = 8000
            if len(text) > max_chars:
                text_to_use = text[:max_chars]
                truncated = True
            else:
                text_to_use = text
                truncated = False

            with st.spinner("Running AI analysis..."):
                ai_output = run_llm_analysis(framework_key, lang, text_to_use)

            st.markdown("---")
            st.subheader(result_title)
            st.write(ai_output)

            if truncated:
                if lang == "zh":
                    st.caption("（提示：文件很長，為了避免超過模型限制，本次僅分析前 8,000 字。）")
                else:
                    st.caption("(Note: The document is long. For safety, only the first 8,000 characters were analyzed.)")


if __name__ == "__main__":
    main()
