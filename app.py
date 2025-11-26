
import streamlit as st
from io import StringIO
import pdfplumber
from docx import Document

# 注意：上傳大小限制現在改由 .streamlit/config.toml 控制
# 這裡不再呼叫 st.set_option("server.maxUploadSize", ...)

# ===============================
# Config: languages & frameworks
# ===============================

LANGS = {
    "zh": "繁體中文",
    "en": "English",
}

FRAMEWORKS = {
    "omission": {
        "name": {
            "zh": "Error-Free® 遺漏錯誤檢查框架（Omission）",
            "en": "Error-Free® Omission Error Check Framework",
        },
        "description": {
            "zh": "運用 12 種查漏方法與 12 類遺漏錯誤（O1–O12），系統性檢查文件中可能的遺漏與缺口。",
            "en": "Use 12 omission-check methods and 12 omission error types (O1–O12) to systematically find missing content in the document.",
        },
        "template": {
            "zh": """你是一名受過 Error-Free® 訓練的「遺漏錯誤檢查顧問」。

【分析對象】
以下是使用者提供的文件內容，請僅根據明示資訊進行分析，不得自行杜撰：

------------------------------------------------------------
{document_text}
------------------------------------------------------------

請依照四個階段輸出嚴謹、可追溯的分析結果：

一、文件摘要（禁止推測）
- 條列式整理：目的、主體內容、重要流程、角色、決策、假設與前提。
- 僅可引用文件文字，不得補充未出現的內容。

二、依 12 種查漏方法檢查是否有遺漏
1 Exception（例外）  
2 Balance（平衡）  
3 Continuity（連續）  
4 Difference（差異）  
5 Expectation（期望）  
6 Framework（框架比對）  
7 Independent Verification（獨立驗證）  
8 Cause–Effect（三元素）  
9 Causal Probability（因果機率）  
10 Association（連想）  
11 Experience（經驗）  
12 Unmet Requirements（未滿足需求）  

對於每一種方法，請回答：
- 是否發現疑似遺漏？（是／否）
- 引用的文件原文或段落位置為何？
- 你推定可能的遺漏情況為何？說明理由（不可超出文件內容）。
若未發現遺漏，請簡要說明理由。

三、將每一項問題對應到 12 類遺漏錯誤（O1–O12）
請為每一個疑似問題標註：
- 遺漏類型（O1–O12）
- 使用到的查漏方法（上述 12 種之一）
- 涉及的原文片段
- 為何判定為該類遺漏？
- 可能造成的後果？
- 建議補充或修正的具體內容（可直接寫進文件的句子或段落）。

四、輸出《Error-Free® 遺漏錯誤修正清單》
請按照優先級（高／中／低）列出所有問題，格式包含：
- 問題編號與遺漏類型
- 優先級
- 問題摘要
- 建議補充內容
- 風險說明

重要原則：
- 不得虛構文件未出現的資訊。
- 每一項判定都需有原文依據。
- 用專案稽核與品質工程顧問語氣撰寫。
""",
            "en": """You are an Error-Free® consultant specializing in omission errors.

[Document to review]
The user has provided the following document. Base all analysis only on what is explicitly written. Do not invent new facts.

------------------------------------------------------------
{document_text}
------------------------------------------------------------

Follow these four stages and provide a structured, auditable report:

I. Document summary (no guessing)
- Bullet-point the purpose, main content, key processes, roles, decisions, assumptions.
- Only quote or paraphrase what is present in the document.

II. Check for omissions using 12 methods
Use the following methods one by one:
1 Exception  
2 Balance  
3 Continuity  
4 Difference  
5 Expectation  
6 Framework comparison  
7 Independent verification  
8 Cause–effect (cause, mechanism, result)  
9 Causal probability  
10 Association  
11 Experience  
12 Unmet requirements  

For each method, state:
- Whether you find a possible omission.
- Which part of the document it relates to (quote or describe).
- Why this may indicate something important is missing.
If nothing is found, briefly explain why.

III. Map issues to the 12 omission error types (O1–O12)
For every suspected issue, label:
- Error type (O1–O12)
- Which omission method found it
- The related text
- Why it fits this type
- Possible consequences
- Concrete content the document should add or clarify.

IV. Output an "Error-Free® Omission Correction Action List"
List all issues with:
- ID and error type
- Priority (High / Medium / Low)
- Short description
- Recommended added / revised content
- Risk explanation

Principles:
- Do not hallucinate or create information not in the document.
- Every judgment must be traceable to the text.
- Use a professional tone suitable for an audit / quality report.
""",
        },
    },

    "technical": {
        "name": {
            "zh": "Error-Free® 技術性錯誤檢查框架（Technical）",
            "en": "Error-Free® Technical Error Check Framework",
        },
        "description": {
            "zh": "依據 39 種技術錯誤檢查方法與 12 類技術錯誤（T1–T4），審查模型、計算與技術假設。",
            "en": "Use 39 technical error-finding methods and 12 technical error types (T1–T4) to review models, calculations and technical assumptions.",
        },
        "template": {
            "zh": """你是一名 Error-Free® 技術性錯誤檢查顧問。

【分析對象】
以下是使用者提供的技術文件，可能包含模型、公式、假設、分析與結論：

------------------------------------------------------------
{document_text}
------------------------------------------------------------

請依四個階段輸出嚴謹、可稽核的技術審查報告：

一、技術內容摘要
- 條列說明：使用了哪些模型、公式、方法、資料、統計或實驗？有哪些前提與限制條件？
- 僅可根據文件原文，不得自行補充假設。

二、依 39 種技術錯誤檢查方法審查
依類別檢查是否存在：
- 規範與法規違反
- 需求與顧客要求違反
- 邏輯錯誤（演繹、歸納、因果、數學邏輯）
- 數學公式使用錯誤
- 科學原理與工程原理違反
- 統計原則錯誤
- 社會倫理原則違反
- 從現象推回技術錯誤（異常、平衡、連續性、差異、期望、獨立驗證、因果三要素、因果機率、經驗等）

對於每一類別，說明：
- 是否發現疑似技術錯誤？
- 相關的文件原文或段落位置？
- 推定錯誤的原因與機制（不可超出文件內容）。

三、將問題對應到 12 類技術錯誤（T1–T4）
請為每一項疑似錯誤標註：
- 類型：T1（方法執行）、T2（適用性）、T3（準確性）、T4（技術保證）
- 更細分類（例如 T1.2 計算錯誤、T3.1 樣本量不足等）
- 涉及原文片段
- 可能後果
- 建議補充或修正的具體技術內容。

四、輸出《Error-Free® 技術性錯誤修正清單》
以優先級（高／中／低）列出所有項目：
- 問題編號與錯誤類型
- 優先級
- 問題摘要
- 建議修正或補充的內容
- 風險說明

原則：
- 嚴禁虛構技術細節。
- 所有判定需可追溯到文件文字。
- 使用工程安全與技術審查的專業語氣。
""",
            "en": """You are an Error-Free® consultant specializing in technical commission errors.

[Document to review]
The user provides a technical document that may include models, formulas, assumptions, analyses and conclusions:

------------------------------------------------------------
{document_text}
------------------------------------------------------------

Produce a rigorous, auditable technical review in four stages:

I. Technical summary
- Bullet-point the models, formulas, methods, data, statistics or experiments used.
- List key assumptions, boundary conditions and limitations.
- Do not add assumptions that are not in the document.

II. Review using the 39 technical error-finding methods
Check for issues such as:
- Violations of regulations, codes, standards or internal rules
- Violations of requirements or customer needs
- Logical errors (deductive, inductive, causal, mathematical logic)
- Incorrect use of mathematical formulas
- Violations of scientific / engineering principles
- Misuse of statistics
- Violations of social or professional ethics
- Errors inferred from phenomena (anomalies, imbalance, broken continuity, unexplained differences, unmet expectations, lack of independent verification, incomplete cause–effect chain, wrong causal probability, conflicts with experience)

For each category, state:
- Whether you find a possible technical error.
- The related text in the document.
- The likely mechanism of error, staying within the evidence of the document.

III. Map issues to the 12 technical error types (T1–T4)
For each suspected error, label:
- Category: T1 Method execution, T2 Applicability, T3 Accuracy, or T4 Assurance
- More specific subtype if applicable (e.g. T1.2 calculation error, T3.1 small sample size).
- Related text, possible consequences, and concrete additions or corrections needed.

IV. Output an "Error-Free® Technical Correction Action List"
List all items with:
- ID and error type
- Priority (High / Medium / Low)
- Short description
- Recommended correction or additional content
- Risk explanation

Principles:
- Do not hallucinate or invent technical details.
- Every conclusion must be traceable to the text.
- Use a professional, engineering-audit tone.
""",
        },
    },
}

# ===============================
# Helper functions
# ===============================

def read_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()

    if name.endswith(".txt"):
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
        return text.strip()

    if name.endswith(".docx"):
        doc = Document(uploaded_file)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

    return ""

def build_prompt(template, doc_text):
    return template.format(document_text=doc_text or "")

def call_llm(prompt):
    # Demo：目前只回傳部分 Prompt，之後可改成實際 LLM 呼叫
    return "【Demo output / 模擬輸出】\n\nBelow is the first part of the prompt sent to the model:\n\n" + prompt[:1500]

# ===============================
# Streamlit App
# ===============================

def main():
    st.set_page_config(page_title="Error-Free® 多框架 / Multi-framework")

    # 語言選擇
    lang = st.sidebar.radio("Language / 語言", options=list(LANGS.keys()),
                            format_func=lambda k: LANGS[k])

    # 框架選擇
    framework_key = st.sidebar.selectbox(
        "選擇框架 / Choose framework",
        options=list(FRAMEWORKS.keys()),
        format_func=lambda k: FRAMEWORKS[k]["name"][lang],
    )
    fw = FRAMEWORKS[framework_key]

    # 主畫面
    if lang == "zh":
        st.title("Error-Free® 多框架 AI 文檔分析")
        st.caption(f"目前使用框架：{fw['name']['zh']}")
        st.markdown(f"**說明：** {fw['description']['zh']}")
        uploader_label = "上傳要分析的文件（支援：PDF、Word（.docx）、純文字 .txt）"
        button_label = "開始分析"
        warn_no_file = "請先上傳一份文件。"
        result_title = "分析結果（示意）"
        expander_label = "查看送入模型的完整 Prompt"
    else:
        st.title("Error-Free® Multi-framework AI Document Analyzer")
        st.caption(f"Current framework: {fw['name']['en']}")
        st.markdown(f"**Description:** {fw['description']['en']}")
        uploader_label = "Upload a document to analyze (PDF, Word .docx, or plain .txt)"
        button_label = "Start analysis"
        warn_no_file = "Please upload a file first."
        result_title = "Analysis result (demo)"
        expander_label = "View full prompt sent to the model"

    uploaded = st.file_uploader(uploader_label, type=["pdf", "docx", "txt"])

    if st.button(button_label):
        if not uploaded:
            st.warning(warn_no_file)
        else:
            with st.spinner("Processing... / 分析中…"):
                text = read_uploaded_file(uploaded)
                template = fw["template"][lang]
                prompt = build_prompt(template, text)
                result = call_llm(prompt)

            st.subheader(result_title)
            st.write(result)

            with st.expander(expander_label):
                st.code(prompt)

if __name__ == "__main__":
    main()
