import os, json, datetime, secrets
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import streamlit as st
import pdfplumber
from docx import Document
from PIL import Image

# NOTE (更正2 + Railway):
# - Do NOT import PyMuPDF / fitz here. Railway environment may not have it installed and will crash on import.
# - PDF text extraction is handled via pdfplumber.

# =========================
# Constants / Config
# =========================
BRAND_TITLE_EN = "Error-Free® Intelligence Engine"
BRAND_TITLE_ZH = "Error-Free® 智能引擎"
MAX_CHARS = 180_000  # keep prompt bounded
STATE_FILE = Path(".app_state.json")

# =========================
# Helpers (language)
# =========================
def zh(tw: str, cn: str) -> str:
    return cn if st.session_state.get("zh_variant", "tw") == "cn" else tw


def language_selector():
    """Language selector shown before/after login.

    Requirement (更正2):
    - If user chooses English, after login the UI should be fully English and
      Chinese labels (中文简体/中文繁體 etc.) should be hidden.
    """
    current_lang = st.session_state.get("lang", "zh")
    current_variant = st.session_state.get("zh_variant", "tw")

    # If English is selected, hide Chinese options entirely.
    if current_lang == "en":
        _ = st.radio("Language", ("English",), index=0, key="lang_choice_en")
        st.session_state.lang = "en"
        if "zh_variant" not in st.session_state:
            st.session_state.zh_variant = "tw"
        return

    # Chinese mode: allow switching among English / 简体 / 繁體.
    index = 1 if current_variant == "cn" else 2
    choice = st.radio("Language / 語言", ("English", "中文简体", "中文繁體"), index=index, key="lang_choice_zh")

    if choice == "English":
        st.session_state.lang = "en"
        if "zh_variant" not in st.session_state:
            st.session_state.zh_variant = "tw"
    else:
        st.session_state.lang = "zh"
        st.session_state.zh_variant = "cn" if choice == "中文简体" else "tw"


# =========================
# Storage helpers
# =========================
def save_state_to_disk() -> None:
    try:
        data = {
            "lang": st.session_state.get("lang", "zh"),
            "zh_variant": st.session_state.get("zh_variant", "tw"),
            "logged_in": st.session_state.get("logged_in", False),
            "is_admin": st.session_state.get("is_admin", False),
            "admin_email": st.session_state.get("admin_email", ""),
            "current_doc_id": st.session_state.get("current_doc_id"),
            "last_doc_name": st.session_state.get("last_doc_name", ""),
            "last_doc_text": st.session_state.get("last_doc_text", ""),
            "selected_framework_key": st.session_state.get("selected_framework_key"),
            "framework_states": st.session_state.get("framework_states", {}),
            "document_type": st.session_state.get("document_type"),
            # Backward compatible fields (older deployments)
            "reference_history": st.session_state.get("reference_history", []),
            "ref_pending": st.session_state.get("ref_pending", False),
            # 更正2: split references into upstream (single) and quote (multi)
            "upstream_reference": st.session_state.get("upstream_reference"),
            "quote_reference_history": st.session_state.get("quote_reference_history", []),
            "quote_ref_pending": st.session_state.get("quote_ref_pending", False),
            "quote_uploader_key": st.session_state.get("quote_uploader_key", 0),
            "upstream_uploader_key": st.session_state.get("upstream_uploader_key", 0),
        }
        STATE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        # Never crash the app due to persistence
        pass


def restore_state_from_disk() -> None:
    if not STATE_FILE.exists():
        return
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        # Only restore non-sensitive / non-secret state
        for k, v in data.items():
            if k not in st.session_state:
                st.session_state[k] = v
    except Exception:
        pass


# =========================
# File text extraction
# =========================
def read_file_to_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    name = (uploaded_file.name or "").lower()
    suffix = Path(name).suffix.lower()

    try:
        if suffix == ".pdf":
            text_parts: List[str] = []
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    t = page.extract_text() or ""
                    if t.strip():
                        text_parts.append(t)
            return "\n\n".join(text_parts).strip()

        if suffix == ".docx":
            doc = Document(uploaded_file)
            return "\n".join(p.text for p in doc.paragraphs).strip()

        if suffix == ".txt":
            return uploaded_file.getvalue().decode("utf-8", errors="ignore").strip()

        if suffix in [".png", ".jpg", ".jpeg"]:
            # Keep as minimal to avoid heavy OCR; if OCR is needed, user should provide text-based doc.
            # We return a placeholder so LLM can still respond without failing.
            return f"[Image uploaded: {uploaded_file.name}. No OCR performed.]"

    except Exception:
        return ""

    return ""


# =========================
# OpenAI call wrapper
# =========================
def call_openai(system_prompt: str, user_prompt: str, model_name: str = "gpt-4o-mini") -> str:
    """
    NOTE: This app expects OPENAI_API_KEY in env.
    Keep as in original app.py (framework/logic unchanged).
    """
    # Import inside to avoid import-time failure in some environments
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


# =========================
# Prompt builders (original structure)
# =========================
def clean_report_text(text: str) -> str:
    # Keep original behavior: light cleanup only
    return (text or "").strip()


def build_step5_prompt(lang: str, doc_type: str, framework_key: str, doc_text: str) -> Tuple[str, str]:
    """
    Keep original logic: step5 performs main framework analysis.
    """
    # (Truncated here only in prompt content; actual analysis uses bounded doc_text)
    doc_text = (doc_text or "")[:MAX_CHARS]

    fw_name_en = FRAMEWORKS[framework_key]["name_en"]
    fw_name_zh = FRAMEWORKS[framework_key]["name_zh"]

    if lang == "zh":
        sys = "你是 Dr. Chiu 的 Error-Free® 零錯誤框架專家。請嚴格依框架輸出，避免臆測。"
        user = "\n".join(
            [
                f"【文件類型】{doc_type}",
                f"【使用框架】{fw_name_zh}",
                "",
                "【主文件內容】",
                doc_text,
                "",
                "【任務】請依照所選零錯誤框架，完成對主文件的分析並輸出可直接採用的結果。",
            ]
        )
    else:
        sys = "You are an Error-Free® framework analyst. Follow the framework strictly; do not hallucinate."
        user = "\n".join(
            [
                f"[Document type] {doc_type}",
                f"[Framework] {fw_name_en}",
                "",
                "[Main document]",
                doc_text,
                "",
                "[Task] Apply the selected Error-Free® framework to analyze the main document and produce directly usable outputs.",
            ]
        )

    return sys, user


def summarize_reference_doc(lang: str, ref_text: str, ref_name: str, model_name: str) -> str:
    ref_text = (ref_text or "")[:MAX_CHARS]
    if lang == "zh":
        sys = "你是文件摘要助理。請用條列式摘要，保留可驗證的章節線索與關鍵句。"
        user = "\n".join(
            [
                f"請摘要參考文件：{ref_name}",
                "",
                ref_text,
                "",
                "要求：",
                "- 條列重點",
                "- 如果有章節/條款號碼請保留",
                "- 不要臆測未出現的內容",
            ]
        )
    else:
        sys = "You are a document summarizer. Provide bullet points with verifiable anchors (sections/clauses) when possible."
        user = "\n".join(
            [
                f"Summarize the reference document: {ref_name}",
                "",
                ref_text,
                "",
                "Requirements:",
                "- Bullet points",
                "- Keep section/clause identifiers if present",
                "- Do not fabricate content",
            ]
        )

    return call_openai(sys, user, model_name=model_name)


def build_relevance_file(framework_key: str, fw_name: str, main_analysis: str, ref_summaries: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    lines += [
        "==============================",
        "REFERENCE RELEVANCE INPUT FILE",
        "==============================",
        "",
        f"Framework: {fw_name}",
        "",
        "------------------------------",
        "MAIN ANALYSIS (STEP 5 OUTPUT)",
        "------------------------------",
        main_analysis or "",
        "",
        "------------------------------",
        "REFERENCE SUMMARIES",
        "------------------------------",
    ]
    for r in ref_summaries:
        lines += [
            f"[Reference] {r.get('name','')}",
            r.get("summary", ""),
            "",
        ]
    return "\n".join(lines).strip()


def derive_relevance_points(lang: str, relevance_file_text: str, model_name: str) -> str:
    relevance_file_text = (relevance_file_text or "")[:MAX_CHARS]
    if lang == "zh":
        sys = "你是零錯誤（Error-Free®）的參考文件相關性分析助理。請基於提供的檔案內容做比對，不得臆測。"
        user = "\n".join(
            [
                "請根據以下『MAIN ANALYSIS』與『REFERENCE SUMMARIES』，產出參考文件的相關性重點：",
                "- 哪些內容相符/支持？",
                "- 哪些內容矛盾或缺漏？",
                "- 建議主文件應如何修正或補充（可操作）？",
                "",
                relevance_file_text,
            ]
        )
    else:
        sys = "You are an Error-Free® reference relevance analyst. Compare provided texts only; do not hallucinate."
        user = "\n".join(
            [
                "Based on the following MAIN ANALYSIS and REFERENCE SUMMARIES, produce reference relevance key points:",
                "- What aligns/supports?",
                "- What conflicts or is missing?",
                "- Actionable suggestions for improving the main document.",
                "",
                relevance_file_text,
            ]
        )
    return call_openai(sys, user, model_name=model_name)


def run_quote_reference_relevance(lang: str, main_doc_text: str, quote_ref_name: str, quote_ref_text: str, model_name: str) -> str:
    """Step 6-2: Quote Reference Relevance Analysis (更正2).

    Uses the provided guideline (uploaded as 'Step 6 Quote Reference Relevance Analysis.docx'):

Step 6 Quote Reference Relevance Analysis

Purpose:
This step checks whether the “quoted / referenced statements” in the MAIN document are consistent with the QUOTE reference document.

Inputs:
- MAIN document text
- QUOTE reference document text (one file per run)

Output:
- A structured comparison report (table + summary) that:
  1) extracts likely quoted/cited statements from MAIN
  2) finds matching evidence (or explicitly says Not found)
  3) verifies faithfulness (no distortion / omission / value change)
  4) provides actionable fixes

Rules:
- Use ONLY the two provided texts (no hallucination).
- If evidence is missing: write Not found / 找不到對應.
- Be concise but verifiable (include excerpts and/or section pointers when possible).

    The output is a concise, verifiable comparison report in Markdown.
    """
    # Keep prompts bounded
    main_doc_text = (main_doc_text or "")[:MAX_CHARS]
    quote_ref_text = (quote_ref_text or "")[:MAX_CHARS]

    if lang == "zh":
        sys = "你是零錯誤（Error-Free®）的文件引用一致性審查助理。你必須基於提供的兩份文本，不得臆測或編造。若找不到對應內容，請明確寫『找不到』。"
        user = (
            "請依據『Step 6 Quote Reference Relevance Analysis』的規則，檢查主文件中引用/引述的內容是否與引述參考文件一致。\n\n"
            "任務：\n"
            "1) 從主文件中找出『看起來是在引用外部文件』的敘述（例如：引用條款、數值、結論、或使用引號的句子）。\n"
            "2) 以此引述參考文件作為唯一對照來源，逐條核對：\n"
            "   - 是否能在參考文件中找到對應（或高度相近）的段落/句子？\n"
            "   - 若找到，主文件的引述是否忠實（不扭曲、不省略關鍵條件、不改變數值/語意）？\n"
            "   - 若找不到，標註為『找不到對應』。\n"
            "3) 產出一個表格（Markdown）至少包含：\n"
            "   - 主文件引述/引用摘錄（可簡短，必要時截斷）\n"
            "   - 參考文件對應證據（可用摘錄或指出章節/關鍵句；若找不到則寫找不到）\n"
            "   - 判定：一致 / 不一致 / 找不到\n"
            "   - 說明與建議修正（要可操作）\n"
            "4) 最後給一段總結：主要不一致點、風險、以及建議下一步。\n\n"
            f"【主文件】\n{main_doc_text}\n\n"
            f"【引述參考文件：{quote_ref_name}】\n{quote_ref_text}"
        )
    else:
        sys = "You are an Error-Free® reference consistency reviewer. Use only the provided texts; do not hallucinate. If evidence is missing, say 'Not found'."
        user = (
            "Using the 'Step 6 Quote Reference Relevance Analysis' guideline, verify whether the MAIN document's quoted/cited statements are consistent with the QUOTE reference document.\n\n"
            "Tasks:\n"
            "1) Identify statements in the MAIN document that appear to cite/quote external sources (e.g., quoted requirements, numeric values, conclusions, or explicit citations).\n"
            "2) Treat the provided QUOTE reference as the ONLY verification source, and check each extracted item:\n"
            "   - Can you find matching/highly similar evidence in the reference?\n"
            "   - If found, is the MAIN statement faithful (no distortion, no critical condition omitted, no numeric/meaning change)?\n"
            "   - If not found, mark as 'Not found'.\n"
            "3) Produce a Markdown table with at least:\n"
            "   - Quoted/Cited statement from MAIN (short excerpt)\n"
            "   - Evidence in QUOTE reference (excerpt or section pointer; 'Not found' if missing)\n"
            "   - Verdict: Consistent / Inconsistent / Not found\n"
            "   - Notes + actionable fix\n"
            "4) End with a short summary of key inconsistencies, risk, and recommended next steps.\n\n"
            f"[MAIN DOCUMENT]\n{main_doc_text}\n\n"
            f"[QUOTE REFERENCE: {quote_ref_name}]\n{quote_ref_text}"
        )

    return call_openai(sys, user, model_name=model_name)


def build_final_integration_input(lang: str, doc_type: str, framework_key: str, step5_output: str, step6_output: str) -> str:
    fw_name_en = FRAMEWORKS[framework_key]["name_en"]
    fw_name_zh = FRAMEWORKS[framework_key]["name_zh"]

    if lang == "zh":
        return "\n".join(
            [
                "【最終整合分析輸入（步驟七）】",
                f"- 文件類型：{doc_type}",
                f"- 使用框架：{fw_name_zh}",
                "",
                "==============================",
                "一、步驟五：主文件零錯誤框架分析結果",
                "==============================",
                step5_output or "",
                "",
                "==============================",
                "二、步驟六：參考文件相關性/一致性結果",
                "==============================",
                step6_output or "",
                "",
                "【任務】",
                "請你用同一個零錯誤框架，將步驟五與步驟六整合成『最終成品分析報告』：去重、補強、並提供可執行的修正/澄清問題清單。",
            ]
        )
    else:
        return "\n".join(
            [
                "[Final Integration Input (Step 7)]",
                f"- Document type: {doc_type}",
                f"- Framework: {fw_name_en}",
                "",
                "==============================",
                "1) Step 5: Main document framework analysis",
                "==============================",
                step5_output or "",
                "",
                "==============================",
                "2) Step 6: Reference relevance / consistency outputs",
                "==============================",
                step6_output or "",
                "",
                "[Task]",
                "Using the same framework, integrate Step 5 + Step 6 into a single final deliverable: de-duplicate, strengthen, and provide actionable fixes / clarification questions.",
            ]
        )


def run_llm_analysis(lang: str, final_input: str, model_name: str) -> str:
    final_input = (final_input or "")[:MAX_CHARS]
    if lang == "zh":
        sys = "你是 Dr. Chiu 的 Error-Free® 零錯誤框架專家。請提供可直接採用的最終報告。"
        user = final_input
    else:
        sys = "You are an Error-Free® framework expert. Provide a final deliverable that can be used directly."
        user = final_input
    return call_openai(sys, user, model_name=model_name)


def build_full_report(lang: str, framework_key: str, state: Dict, include_followups: bool = True) -> str:
    fw_name_en = FRAMEWORKS[framework_key]["name_en"]
    fw_name_zh = FRAMEWORKS[framework_key]["name_zh"]

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if lang == "zh":
        header = [
            ((f"{BRAND_TITLE_ZH} 報告（分析 + Q&A）") if include_followups else (f"{BRAND_TITLE_ZH} 報告（分析）")),
            f"生成時間：{now_str}",
            f"使用框架：{fw_name_zh}",
            "",
        ]
    else:
        header = [
            ((f"{BRAND_TITLE_EN} Report (Analysis + Q&A)") if include_followups else (f"{BRAND_TITLE_EN} Report (Analysis)")),
            f"Generated at: {now_str}",
            f"Framework: {fw_name_en}",
            "",
        ]

    main = state.get("analysis_output", "")
    followups = state.get("followup_history", [])

    sections = header + [main]

    if include_followups and followups:
        if lang == "zh":
            sections += [
                "",
                "==============================",
                "Follow-up Q&A 紀錄",
                "==============================",
            ]
            for i, item in enumerate(followups, start=1):
                q = item.get("q", "")
                a = item.get("a", "")
                sections += [f"Q{i}: {q}", "", f"A{i}: {a}", "", "---", ""]
        else:
            sections += [
                "",
                "==============================",
                "Follow-up Q&A History",
                "==============================",
            ]
            for i, item in enumerate(followups, start=1):
                q = item.get("q", "")
                a = item.get("a", "")
                sections += [f"Q{i}: {q}", "", f"A{i}: {a}", "", "---", ""]

    return "\n".join(sections).strip()


# =========================
# Framework definitions (as original)
# =========================
FRAMEWORKS = {
    "Error-Free® Omission Error Check Framework": {
        "name_en": "Error-Free® Omission Error Check Framework",
        "name_zh": "Error-Free® 遺漏錯誤檢核框架",
    },
    "Error-Free® Document Review Framework": {
        "name_en": "Error-Free® Document Review Framework",
        "name_zh": "Error-Free® 文件審查框架",
    },
    "Error-Free® Risk Analysis Framework": {
        "name_en": "Error-Free® Risk Analysis Framework",
        "name_zh": "Error-Free® 風險分析框架",
    },
}

# =========================
# UI / Session init
# =========================
st.set_page_config(page_title=BRAND_TITLE_EN, layout="wide")

restore_state_from_disk()

if "lang" not in st.session_state:
    st.session_state.lang = "zh"
if "zh_variant" not in st.session_state:
    st.session_state.zh_variant = "tw"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False
if "admin_email" not in st.session_state:
    st.session_state.admin_email = ""

if "framework_states" not in st.session_state:
    st.session_state.framework_states = {}
if "selected_framework_key" not in st.session_state:
    st.session_state.selected_framework_key = "Error-Free® Omission Error Check Framework"

if "document_type" not in st.session_state:
    st.session_state.document_type = None

# backward-compat
if "reference_history" not in st.session_state:
    st.session_state.reference_history = []
if "ref_pending" not in st.session_state:
    st.session_state.ref_pending = False

# 更正2 new states
if "upstream_reference" not in st.session_state:
    st.session_state.upstream_reference = None
if "quote_reference_history" not in st.session_state:
    st.session_state.quote_reference_history = []
if "quote_ref_pending" not in st.session_state:
    st.session_state.quote_ref_pending = False
if "quote_uploader_key" not in st.session_state:
    st.session_state.quote_uploader_key = 0
if "upstream_uploader_key" not in st.session_state:
    st.session_state.upstream_uploader_key = 0

if "current_doc_id" not in st.session_state:
    st.session_state.current_doc_id = None
if "last_doc_name" not in st.session_state:
    st.session_state.last_doc_name = ""
if "last_doc_text" not in st.session_state:
    st.session_state.last_doc_text = ""

# =========================
# Sidebar
# =========================
with st.sidebar:
    language_selector()
    lang = st.session_state.get("lang", "zh")

    st.markdown("---")
    st.markdown("## " + ("Account" if lang == "en" else "Account"))

    if st.session_state.logged_in:
        st.write(("Email : " if lang == "en" else "Email : ") + (st.session_state.admin_email or ""))
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.is_admin = False
            st.session_state.admin_email = ""
            save_state_to_disk()
            st.rerun()
    else:
        st.info("Not logged in" if lang == "en" else zh("尚未登入", "尚未登录"))

    st.markdown("---")
    # Keep original button placement; only change label per 更正2 when English
    if st.session_state.is_admin:
        st.button("Admin Dashboard", key="admin_dashboard_btn")


# =========================
# Main page
# =========================
st.title(BRAND_TITLE_EN if lang == "en" else BRAND_TITLE_ZH)
st.markdown(
    "An AI-enhanced intelligence engine that helps organizations analyze risks, prevent errors, and make better decisions."
    if lang == "en"
    else zh(
        "一個 AI 強化的智能引擎，協助組織分析風險、預防錯誤、做出更好的決策。",
        "一个 AI 强化的智能引擎，协助组织分析风险、预防错误、做出更好的决策。",
    )
)

# Model selection (keep original structure)
model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# =========================
# Main workflow state per framework
# =========================
selected_key = st.session_state.get("selected_framework_key", "Error-Free® Omission Error Check Framework")
if selected_key not in st.session_state.framework_states:
    st.session_state.framework_states[selected_key] = {
        "step5_done": False,
        "step5_output": "",
        # old step6 fields retained for backward compat, but not used after 更正2
        "step6_done": False,
        "step6_output": "",
        # 更正2 step6 split
        "step6_upstream_done": False,
        "step6_upstream_output": "",
        "step7_done": False,
        "step7_output": "",
        "analysis_output": "",
        "followup_history": [],
    }
current_state = st.session_state.framework_states[selected_key]

# =========================
# Step 1: Upload review document
# =========================
st.subheader("Step 1: Upload Review Document" if lang == "en" else zh("步驟一：上傳審查文件", "步骤一：上传审查文件"))
st.caption(
    "Note: Only 1 document can be uploaded for a complete content analysis."
    if lang == "en"
    else zh("注意：完整內容分析僅能上傳 1 份文件。", "注意：完整内容分析仅能上传 1 份文件。")
)

doc_file = st.file_uploader(
    "Upload review document (PDF / DOCX / TXT / Image)" if lang == "en" else zh("上傳審查文件（PDF / DOCX / TXT / 圖片）", "上传审查文件（PDF / DOCX / TXT / 图片）"),
    type=["pdf", "docx", "txt", "jpg", "jpeg", "png"],
    key="main_doc_uploader",
)

if doc_file is not None:
    # Store main doc
    st.session_state.last_doc_name = doc_file.name
    st.session_state.last_doc_text = read_file_to_text(doc_file)
    if not st.session_state.current_doc_id:
        st.session_state.current_doc_id = secrets.token_hex(8)
    save_state_to_disk()
    st.success(
        (f"Review document uploaded: {doc_file.name}")
        if lang == "en"
        else zh(f"審查文件已上傳：{doc_file.name}", f"审查文件已上传：{doc_file.name}")
    )
else:
    if st.session_state.last_doc_name:
        st.info(
            (f"Review document uploaded: {st.session_state.last_doc_name}. To change it, please use Reset Whole Document in Step 7.")
            if lang == "en"
            else zh(
                f"審查文件已上傳：{st.session_state.last_doc_name}。如需更換，請在步驟七使用 Reset Whole Document。",
                f"审查文件已上传：{st.session_state.last_doc_name}。如需更换，请在步骤七使用 Reset Whole Document。",
            )
        )

st.markdown("---")

# =========================
# Step 2: Select document type (locks after Step 5)
# =========================
st.subheader("Step 2: Select Document Type" if lang == "en" else zh("步驟二：選擇文件類型", "步骤二：选择文件类型"))
st.caption(zh("單選", "单选") if lang == "zh" else "Single selection")

    doc_type_locked = bool(current_state.get("step5_done", False))

DOC_TYPES = [
    "Conceptual Design",
    "Preliminary Design",
    "Final Design",
    "Equivalency Engineering Evaluation",
    "Root Cause Analysis",
    "Safety Analysis",
    "Specifications and Requirements",
    "Calculations and Analysis",
]

DOC_TYPE_LABELS_ZH_TW = {
    "Conceptual Design": "概念設計",
    "Preliminary Design": "初步設計",
    "Final Design": "最終設計",
    "Equivalency Engineering Evaluation": "等效工程評估",
    "Root Cause Analysis": "根本原因分析",
    "Safety Analysis": "安全分析",
    "Specifications and Requirements": "規格與需求",
    "Calculations and Analysis": "計算與分析",
}
DOC_TYPE_LABELS_ZH_CN = {
    "Conceptual Design": "概念设计",
    "Preliminary Design": "初步设计",
    "Final Design": "最终设计",
    "Equivalency Engineering Evaluation": "等效工程评估",
    "Root Cause Analysis": "根本原因分析",
    "Safety Analysis": "安全分析",
    "Specifications and Requirements": "规格与需求",
    "Calculations and Analysis": "计算与分析",
}

if st.session_state.get("document_type") not in DOC_TYPES:
    st.session_state.document_type = DOC_TYPES[0]

if lang == "zh":
    mapping = DOC_TYPE_LABELS_ZH_CN if st.session_state.get("zh_variant", "tw") == "cn" else DOC_TYPE_LABELS_ZH_TW
    labels = [mapping.get(x, x) for x in DOC_TYPES]
    label_to_value = {mapping.get(x, x): x for x in DOC_TYPES}
    value_to_label = {x: mapping.get(x, x) for x in DOC_TYPES}
    current_label = value_to_label.get(st.session_state.document_type, labels[0])

    picked_label = st.selectbox(
        zh("選擇文件類型", "选择文件类型"),
        labels,
        index=labels.index(current_label) if current_label in labels else 0,
        key="document_type_select_zh",
        disabled=doc_type_locked,
    )
    st.session_state.document_type = label_to_value.get(picked_label, DOC_TYPES[0])
else:
    st.session_state.document_type = st.selectbox(
        "Select document type",
        DOC_TYPES,
        index=DOC_TYPES.index(st.session_state.document_type),
        key="document_type_select",
        disabled=doc_type_locked,
    )

# 更正2：提醒並鎖定邏輯（步驟五開始後，步驟二不可再切換）
if st.session_state.document_type == "Specifications and Requirements" and not doc_type_locked:
    st.info(
        "Once you click Step 5 to start analysis, Step 2 (Document Type) will be locked and cannot be changed. To run a new review with a different main document or different selections, use Step 7: Reset Whole Document."
        if lang == "en"
        else zh(
            "提醒：一旦在步驟五按下開始分析後，步驟二（文件類型）就會鎖定，無法再切換；如要以不同主文件或不同選擇重新審查，請到步驟七按 Reset Whole Document，開啟新一輪審查。",
            "提醒：一旦在步骤五按下开始分析后，步骤二（文件类型）就会锁定，无法再切换；如要以不同主文件或不同选择重新审查，请到步驟七按 Reset Whole Document，开启新一轮審查。",
        )
    )
if doc_type_locked:
    st.warning(
        "Step 2 is locked because Step 5 has already started. Use Step 7: Reset Whole Document to start a new review."
        if lang == "en"
        else zh(
            "步驟二已鎖定（因已開始步驟五分析）。如需重新開始請到步驟七按 Reset Whole Document。",
            "步骤二已锁定（因已开始步骤五分析）。如需重新开始请到步驟七按 Reset Whole Document。",
        )
    )

save_state_to_disk()

# =========================
# Step 3: Reference documents (optional)
# =========================
st.subheader(
    zh("步驟三：上傳參考文件（選填）", "步骤三：上传参考文件（选填）") if lang == "zh" else "Step 3: Upload Reference Documents (optional)"
)

# 更正2：拆分為 3-1 主參考（Upstream, 只能一次）與 3-2 次參考（Quote, 可多輪）
if "upstream_reference" not in st.session_state:
    st.session_state.upstream_reference = None  # {"name","ext","text","uploaded_at"}
if "quote_reference_history" not in st.session_state:
    st.session_state.quote_reference_history = []  # list of {"name","ext","text","uploaded_at","analyzed","output"}
if "quote_ref_pending" not in st.session_state:
    st.session_state.quote_ref_pending = False
if "quote_uploader_key" not in st.session_state:
    st.session_state.quote_uploader_key = 0
if "upstream_uploader_key" not in st.session_state:
    st.session_state.upstream_uploader_key = 0

# 3-1 Upstream (single)
st.markdown("#### 3-1 " + ("Upload Upstream Reference Documents (optional)" if lang == "en" else zh("上傳上游參考文件（選填）", "上传上游参考文件（选填）")))
st.caption(
    (
        "Upstream reference can be uploaded only once per whole-document review. After upload, this area will be locked until you Reset Whole Document in Step 7."
        if lang == "en"
        else zh(
            "上游參考文件在「同一輪主文件審查」中只能上傳一次；上傳後此區域會鎖定，直到步驟七 Reset Whole Document 才會清空重新開始。",
            "上游参考文件在「同一轮主文件审查」中只能上传一次；上传后此区域会锁定，直到步骤七 Reset Whole Document 才会清空重新开始。",
        )
    )
)

if st.session_state.upstream_reference:
    up = st.session_state.upstream_reference
    st.markdown(
        (zh("已上傳：", "已上传：") if lang == "zh" else "Uploaded: ")
        + f"{up.get('name','')}"
    )

upstream_disabled = bool(st.session_state.upstream_reference)
upstream_file = st.file_uploader(
    zh("上傳上游參考文件（PDF / DOCX / TXT / 圖片）", "上传上游参考文件（PDF / DOCX / TXT / 图片）") if lang == "zh" else "Upload upstream reference document (PDF / DOCX / TXT / Image)",
    type=["pdf", "docx", "txt", "jpg", "jpeg", "png"],
    key=f"upstream_uploader_{st.session_state.upstream_uploader_key}",
    disabled=upstream_disabled,
)

if upstream_file and not upstream_disabled:
    up_text = read_file_to_text(upstream_file)
    st.session_state.upstream_reference = {
        "name": upstream_file.name,
        "ext": Path(upstream_file.name).suffix.lstrip("."),
        "text": up_text,
        "uploaded_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    # upstream relevance needs to be re-run (once) after upload
    current_state["step6_upstream_done"] = False
    current_state["step6_upstream_output"] = ""
    save_state_to_disk()
    st.rerun()

st.divider()

# 3-2 Quote reference (multi-round, one-at-a-time)
st.markdown("#### 3-2 " + ("Upload Quote Reference Document (optional)" if lang == "en" else zh("上傳引述參考文件（選填）", "上传引述参考文件（选填）")))
st.caption(
    (
        "Quote reference supports multiple rounds. Each round: upload ONE quote reference, run quote relevance in Step 6, then use Reset Document (Quote) here to upload the next one. This does NOT affect the upstream reference."
        if lang == "en"
        else zh(
            "引述參考文件可多輪上傳分析。每一輪：先上傳 1 份引述參考文件 → 到步驟六執行 Run Analysis (quote relevance) → 回到此處按 Reset Document（Quote）後才可上傳下一份。此流程不影響 3-1 的上游參考文件。",
            "引述参考文件可多轮上传分析。每一轮：先上传 1 份引述参考文件 → 到步骤六执行 Run Analysis (quote relevance) → 回到此处按 Reset Document（Quote）后才可上传下一份。此流程不影响 3-1 的上游参考文件。",
        )
    )
)

quote_file = st.file_uploader(
    zh("上傳引述參考文件（PDF / DOCX / TXT / 圖片）", "上传引述参考文件（PDF / DOCX / TXT / 图片）") if lang == "zh" else "Upload quote reference document (PDF / DOCX / TXT / Image)",
    type=["pdf", "docx", "txt", "jpg", "jpeg", "png"],
    key=f"quote_uploader_{st.session_state.quote_uploader_key}",
    disabled=st.session_state.quote_ref_pending,
)

# Reset (quote) to allow next upload
if st.session_state.quote_ref_pending:
    if st.button(
        zh("Reset Document（僅引述參考文件）", "Reset Document（仅引述参考文件）") if lang == "zh" else "Reset Document (quote only)",
        key="reset_quote_only",
    ):
        st.session_state.quote_ref_pending = False
        st.session_state.quote_uploader_key += 1
        save_state_to_disk()
        st.rerun()

if quote_file and not st.session_state.quote_ref_pending:
    q_text = read_file_to_text(quote_file)
    st.session_state.quote_reference_history.append(
        {
            "name": quote_file.name,
            "ext": Path(quote_file.name).suffix.lstrip("."),
            "text": q_text,
            "uploaded_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "analyzed": False,
            "output": "",
        }
    )
    st.session_state.quote_ref_pending = True
    # quote relevance can be run for this new quote doc
    save_state_to_disk()
    st.rerun()

# Display quote upload / analysis history (sorted by upload time)
if st.session_state.quote_reference_history:
    with st.expander(zh("引述參考文件歷史紀錄", "引述参考文件历史纪录") if lang == "zh" else "Quote reference history", expanded=False):
        for i, r in enumerate(st.session_state.quote_reference_history, start=1):
            status = "DONE" if r.get("analyzed") else "PENDING"
            st.markdown(f"- {i}. {status} {r.get('name','')}")

# =========================
# Step 4: Select framework (locks after Step 5)
# =========================
st.subheader("Step 4: Select Framework" if lang == "en" else zh("步驟四：選擇框架", "步骤四：选择框架"))
st.caption(
    zh("單選。建議使用步驟七 Reset Whole Document（一次一輪主文件審查僅分析一個 Framework）。", "单选。建议使用步骤七 Reset Whole Document（一次一轮主文件审查仅分析一个 Framework）。")
    if lang == "zh"
    else "Single selection. To analyze a different framework, use Step 7: Reset Whole Document (one framework per whole-document review) to avoid confused outputs."
)

framework_locked = bool(current_state.get("step5_done", False))
if st.session_state.get("selected_framework_key") == "Error-Free® Omission Error Check Framework" and not framework_locked:
    st.info(
        "Once you click Step 5 to start analysis, Step 4 (Framework) will be locked and cannot be changed. To analyze a different framework, use Step 7: Reset Whole Document to start a new review."
        if lang == "en"
        else zh(
            "提醒：一旦在步驟五按下開始分析後，步驟四（框架）就會鎖定，無法再切換；如要分析不同框架，請到步驟七按 Reset Whole Document，開啟新一輪審查。",
            "提醒：一旦在步骤五按下开始分析后，步骤四（框架）就会锁定，无法再切换；如要分析不同框架，请到步驟七按 Reset Whole Document，开启新一轮審查。",
        )
    )
if framework_locked:
    st.warning(
        "Step 4 is locked because Step 5 has already started. Use Step 7: Reset Whole Document to start a new review."
        if lang == "en"
        else zh(
            "步驟四已鎖定（因已開始步驟五分析）。如需重新開始請到步驟七按 Reset Whole Document。",
            "步骤四已锁定（因已开始步骤五分析）。如需重新开始请到步驟七按 Reset Whole Document。",
        )
    )

fw_keys = list(FRAMEWORKS.keys())
fw_labels = [FRAMEWORKS[k]["name_en"] if lang == "en" else FRAMEWORKS[k]["name_zh"] for k in fw_keys]
label_to_key = {fw_labels[i]: fw_keys[i] for i in range(len(fw_keys))}
current_label = FRAMEWORKS[selected_key]["name_en"] if lang == "en" else FRAMEWORKS[selected_key]["name_zh"]
picked_fw_label = st.selectbox(
    "Select framework" if lang == "en" else zh("選擇框架", "选择框架"),
    fw_labels,
    index=fw_labels.index(current_label) if current_label in fw_labels else 0,
    key="framework_selectbox",
    disabled=framework_locked,
)
st.session_state.selected_framework_key = label_to_key.get(picked_fw_label, fw_keys[0])
selected_key = st.session_state.selected_framework_key
if selected_key not in st.session_state.framework_states:
    st.session_state.framework_states[selected_key] = {
        "step5_done": False,
        "step5_output": "",
        "step6_done": False,
        "step6_output": "",
        "step6_upstream_done": False,
        "step6_upstream_output": "",
        "step7_done": False,
        "step7_output": "",
        "analysis_output": "",
        "followup_history": [],
    }
current_state = st.session_state.framework_states[selected_key]

st.markdown("---")

# =========================
# Step 5: Analyze main document
# =========================
st.subheader("Step 5: Analyze the main document" if lang == "en" else zh("步驟五：分析主文件", "步骤五：分析主文件"))

doc_ready = bool(st.session_state.last_doc_text and st.session_state.document_type)
run_step5 = st.button(
    "Run analysis" if lang == "en" else zh("Run analysis", "Run analysis"),
    key="run_step5_btn",
    disabled=not doc_ready or current_state.get("step5_done", False),
)

if run_step5:
    with st.spinner(zh("正在分析主文件...", "正在分析主文件...") if lang == "zh" else "Analyzing main document..."):
        sys, user = build_step5_prompt(lang, st.session_state.document_type, selected_key, st.session_state.last_doc_text)
        out = call_openai(sys, user, model_name=model_name)

    current_state["step5_done"] = True
    current_state["step5_output"] = clean_report_text(out)
    save_state_to_disk()
    st.success(zh("步驟五完成。", "步骤五完成。") if lang == "zh" else "Step 5 completed.")

step5_done = bool(current_state.get("step5_done", False))
if step5_done:
    st.markdown(current_state.get("step5_output", ""))

st.markdown("---")

# =========================
# Step 6: Reference relevance analysis (更正2)
# =========================
has_refs = bool(st.session_state.get("upstream_reference") or st.session_state.get("quote_reference_history"))

if has_refs:
    st.subheader(
        zh("步驟六：參考文件相關性分析（有上傳參考文件才會啟用）", "步骤六：参考文件相关性分析（有上传參考文件才会启用）")
        if lang == "zh"
        else "Step 6: Reference relevance analysis (enabled only if references uploaded)"
    )
    st.caption(
        zh(
            "更正2：本步驟分為 6-1（上游參考一致性）與 6-2（引述參考一致性）。若同時上傳 3-1 與 3-2，請先完成 6-1 才能做 6-2。",
            "更正2：本步骤分为 6-1（上游参考一致性）与 6-2（引述参考一致性）。若同时上传 3-1 与 3-2，请先完成 6-1 才能做 6-2。",
        )
        if lang == "zh"
        else "Correction #2: Step 6 is split into 6-1 (upstream reference relevance) and 6-2 (quote reference relevance). If both 3-1 and 3-2 are uploaded, complete 6-1 before running 6-2."
    )

    has_upstream = bool(st.session_state.get("upstream_reference"))
    quote_history = st.session_state.get("quote_reference_history", []) or []
    pending_quote_idx = None
    for idx in range(len(quote_history) - 1, -1, -1):
        if not quote_history[idx].get("analyzed", False):
            pending_quote_idx = idx
            break
    has_pending_quote = pending_quote_idx is not None

    step6_upstream_done = bool(current_state.get("step6_upstream_done", False))
    step5_done = bool(current_state.get("step5_done", False))

    # 6-1 Upstream reference relevance (one-time per whole-document review)
    st.markdown("##### 6-1 " + ("Upstream Reference Relevance Analysis" if lang == "en" else zh("上游參考相關性分析", "上游参考相关性分析")))
    step6_1_can_run = bool(step5_done and has_upstream and (not step6_upstream_done))
    run_step6_1 = st.button(
        zh("Run analysis（上游相關性）", "Run analysis（上游相关性）") if lang == "zh" else "Run analysis (upstream relevance)",
        key="run_step6_upstream_btn",
        disabled=not step6_1_can_run,
    )

    if run_step6_1:
        # summarize upstream reference and compare to Step-5 output
        up = st.session_state.upstream_reference
        with st.spinner(zh("正在產生上游參考相關性重點...", "正在产生上游参考相关性重点...") if lang == "zh" else "Generating upstream relevance key points..."):
            ref_summary = summarize_reference_doc(lang, up.get("text", ""), up.get("name", ""), model_name)
            relevance_file_text = build_relevance_file(
                framework_key=selected_key,
                fw_name=FRAMEWORKS[selected_key]["name_en"] if lang == "en" else FRAMEWORKS[selected_key]["name_zh"],
                main_analysis=current_state.get("step5_output", ""),
                ref_summaries=[{"name": up.get("name", ""), "summary": ref_summary}],
            )
            relevance_points = derive_relevance_points(lang, relevance_file_text, model_name)

        current_state["step6_upstream_done"] = True
        current_state["step6_upstream_output"] = clean_report_text(relevance_points)
        save_state_to_disk()
        st.success(zh("6-1 完成：已產出上游參考相關性重點。", "6-1 完成：已产出上游参考相关性重点。") if lang == "zh" else "6-1 completed. Upstream relevance key points generated.")

    if step6_upstream_done:
        with st.expander(zh("查看 6-1 結果", "查看 6-1 结果") if lang == "zh" else "View 6-1 output", expanded=False):
            st.markdown(current_state.get("step6_upstream_output", ""))

    st.divider()

    # 6-2 Quote reference relevance (multi-round)
    st.markdown("##### 6-2 " + ("Quote Reference Relevance Analysis" if lang == "en" else zh("引述參考一致性分析", "引述参考一致性分析")))
    if has_pending_quote:
        q = quote_history[pending_quote_idx]
        st.caption((zh("待分析引述參考文件：", "待分析引述参考文件：") if lang == "zh" else "Pending quote reference: ") + q.get("name", ""))
    else:
        st.caption(zh("尚無待分析的引述參考文件。請先在步驟三（3-2）上傳。", "尚无待分析的引述参考文件。请先在步骤三（3-2）上传。") if lang == "zh" else "No pending quote reference. Please upload one in Step 3 (3-2).")

    # Gate: if both upstream + quote exist, upstream must be done first
    step6_2_blocked = bool(has_upstream and has_pending_quote and (not step6_upstream_done))
    if step6_2_blocked:
        st.warning(
            zh("已同時上傳 3-1 與 3-2。請先完成 6-1（上游相關性）後才能執行 6-2。", "已同时上传 3-1 与 3-2。请先完成 6-1（上游相关性）后才能执行 6-2。")
            if lang == "zh"
            else "Both upstream and quote references are present. Please complete 6-1 first, then run 6-2."
        )

    step6_2_can_run = bool(step5_done and has_pending_quote and (not step6_2_blocked))
    run_step6_2 = st.button(
        zh("Run analysis（引述一致性）", "Run analysis（引述一致性）") if lang == "zh" else "Run analysis (quote relevance)",
        key="run_step6_quote_btn",
        disabled=not step6_2_can_run,
    )

    if run_step6_2:
        q = quote_history[pending_quote_idx]
        with st.spinner(zh("正在執行引述一致性分析...", "正在执行引述一致性分析...") if lang == "zh" else "Running quote reference relevance analysis..."):
            out = run_quote_reference_relevance(
                lang=lang,
                main_doc_text=st.session_state.get("last_doc_text", ""),
                quote_ref_name=q.get("name", ""),
                quote_ref_text=q.get("text", ""),
                model_name=model_name,
            )
        quote_history[pending_quote_idx]["analyzed"] = True
        quote_history[pending_quote_idx]["output"] = clean_report_text(out)
        st.session_state.quote_reference_history = quote_history
        save_state_to_disk()
        st.success(zh("6-2 完成：已產出引述一致性分析結果。", "6-2 完成：已产出引述一致性分析结果。") if lang == "zh" else "6-2 completed. Quote relevance output generated.")

    # Show latest analyzed quote outputs
    analyzed_quotes = [r for r in quote_history if r.get("analyzed")]
    if analyzed_quotes:
        with st.expander(zh("查看 6-2 結果（歷史）", "查看 6-2 结果（历史）") if lang == "zh" else "View 6-2 outputs (history)", expanded=False):
            for i, r in enumerate(analyzed_quotes, start=1):
                st.markdown(f"**{i}. {r.get('name','')}**")
                st.markdown(r.get("output", ""))
                st.markdown("---")

    st.markdown("---")

# =========================
# Step 7: Final integration
# =========================
st.subheader("Step 7: Final integration" if lang == "en" else zh("步驟七：最終整合成品", "步骤七：最终整合成品"))
st.caption(
    zh("需先完成步驟五。若本輪有參考文件，需先完成對應的步驟六。", "需先完成步骤五。若本轮有参考文件，需先完成对应的步骤六。")
    if lang == "zh"
    else "Requires Step 5. If references exist in this review, Step 6 must be completed accordingly."
)

has_upstream = bool(st.session_state.get("upstream_reference"))
quote_history = st.session_state.get("quote_reference_history", []) or []
pending_quote = any(not r.get("analyzed", False) for r in quote_history)
step6_upstream_done = (bool(current_state.get("step6_upstream_done", False)) if has_upstream else True)

# 更正2：如果同輪有參考文件，步驟七需先完成相應的步驟六（且不可有未分析的引述參考文件）
step7_need_step6 = bool(has_upstream or quote_history)
step7_blocked = bool((has_upstream and (not step6_upstream_done)) or pending_quote)

if step7_blocked:
    if has_upstream and (not step6_upstream_done):
        st.warning(
            zh("步驟七尚不可執行：已上傳上游參考文件，但尚未完成 6-1。", "步骤七尚不可执行：已上传上游参考文件，但尚未完成 6-1。")
            if lang == "zh"
            else "Step 7 is blocked: upstream reference is uploaded but 6-1 has not been completed."
        )
    if pending_quote:
        st.warning(
            zh("步驟七尚不可執行：存在尚未分析的引述參考文件。請先完成 6-2，或回到步驟三按 Reset Document（Quote）取消本輪引述分析。", "步骤七尚不可执行：存在尚未分析的引述参考文件。请先完成 6-2，或回到步骤三按 Reset Document（Quote）取消本轮引述分析。")
            if lang == "zh"
            else "Step 7 is blocked: there is a pending quote reference that has not been analyzed. Complete 6-2 or Reset Document (quote only) in Step 3."
        )

step7_can_run = (
    step5_done
    and (not current_state.get("step7_done", False))
    and (not step7_blocked)
)

run_step7 = st.button(
    "Run analysis" if lang == "en" else zh("Run analysis", "Run analysis"),
    key="run_step7_btn",
    disabled=not step7_can_run,
)

if run_step7:
    with st.spinner(zh("正在整合最終成品...", "正在整合最终成品...") if lang == "zh" else "Building final deliverable..."):
        # Combine Step 6 outputs (upstream + quote) if available, then integrate in Step 7
        relevance_parts: List[str] = []
        if has_upstream and current_state.get("step6_upstream_output"):
            relevance_parts.append("### 6-1 Upstream reference relevance\n" + (current_state.get("step6_upstream_output") or ""))
        analyzed_quotes = [r for r in quote_history if r.get("analyzed") and r.get("output")]
        if analyzed_quotes:
            for i, r in enumerate(analyzed_quotes, start=1):
                relevance_parts.append(f"### 6-2 Quote reference #{i}: {r.get('name','')}\n{r.get('output','')}")

        combined_relevance = "\n\n".join(relevance_parts).strip()

        if combined_relevance:
            final_input = build_final_integration_input(
                lang,
                st.session_state.document_type,
                selected_key,
                current_state.get("step5_output", ""),
                combined_relevance,
            )
        else:
            # No references: finalize based on step5 only, but keep final form.
            if lang == "zh":
                final_input = "\n".join(
                    [
                        "【最終整合分析輸入（步驟七）】",
                        f"- 文件類型：{st.session_state.document_type or '（未選擇）'}",
                        "",
                        "==============================",
                        "一、步驟五：主文件零錯誤框架分析結果",
                        "==============================",
                        current_state.get("step5_output", ""),
                        "",
                        "【任務】",
                        "請你用同一個零錯誤框架，將上述內容整理成『最終成品分析報告』：去重、補強、並提供可執行的修正/澄清問題清單。",
                    ]
                )
            else:
                final_input = "\n".join(
                    [
                        "[Final Integration Input (Step 7)]",
                        f"- Document type: {st.session_state.document_type or '(not selected)'}",
                        "",
                        "==============================",
                        "1) Step 5: Main document framework analysis",
                        "==============================",
                        current_state.get("step5_output", ""),
                        "",
                        "[Task]",
                        "Using the same framework, rewrite the above into a single final deliverable: de-duplicate, strengthen, and provide actionable fixes / clarification questions.",
                    ]
                )
        final_output = run_llm_analysis(lang, final_input, model_name)

    current_state["step7_done"] = True
    current_state["step7_output"] = clean_report_text(final_output)

    # Build analysis_output (download + followups) without UI duplication
    if lang == "zh":
        prefix_lines = [
            "【審查資訊】",
            f"- 主文件：{st.session_state.last_doc_name or '（未上傳）'}",
            f"- 文件類型：{st.session_state.document_type or '（未選擇）'}",
            f"- 使用框架：{FRAMEWORKS[selected_key]['name_zh']}",
        ]

        up_ref = st.session_state.get("upstream_reference")
        q_hist = st.session_state.get("quote_reference_history", []) or []

        if up_ref or q_hist:
            prefix_lines.append("- 參考文件（Reference Documents）上傳紀錄：")
            if up_ref:
                ext = (up_ref.get("ext", "") or "").upper()
                prefix_lines.append(f"  Upstream: {up_ref.get('name','')}" + (f" ({ext})" if ext else ""))
            if q_hist:
                prefix_lines.append("  Quote references:")
                for i, r in enumerate(q_hist, start=1):
                    ext = (r.get("ext", "") or "").upper()
                    prefix_lines.append(f"    {i}. {r.get('name','')}" + (f" ({ext})" if ext else ""))
        else:
            prefix_lines.append("- 參考文件（Reference Documents）：（未上傳）")

        prefix = "\n".join(prefix_lines) + "\n\n"

        combined_sections = [
            prefix,
            "==============================",
            "（步驟五）主文件分析結果",
            "==============================",
            current_state.get("step5_output", ""),
        ]

        combined_sections += [
            "",
            "==============================",
            "（步驟六）參考文件相關性/一致性結果",
            "==============================",
        ]

        # 6-1 upstream
        if current_state.get("step6_upstream_done"):
            combined_sections += [
                "【6-1 上游參考相關性】",
                current_state.get("step6_upstream_output", ""),
                "",
            ]
        elif st.session_state.get("upstream_reference"):
            combined_sections += ["【6-1 上游參考相關性】（尚未執行）", "", ""]

        # 6-2 quote (history)
        q_hist = st.session_state.get("quote_reference_history", []) or []
        analyzed_quotes = [r for r in q_hist if r.get("analyzed") and r.get("output")]
        if analyzed_quotes:
            combined_sections.append("【6-2 引述參考一致性（歷史）】")
            for i, r in enumerate(analyzed_quotes, start=1):
                combined_sections += [
                    f"--- Quote #{i}: {r.get('name','')} ---",
                    r.get("output", ""),
                    "",
                ]
        elif q_hist:
            combined_sections += ["【6-2 引述參考一致性】（尚未執行）", "", ""]

        combined_sections += [
            "",
            "==============================",
            "（步驟七）最終整合成品",
            "==============================",
            current_state.get("step7_output", ""),
        ]
        current_state["analysis_output"] = "\n".join(combined_sections).strip()

    else:
        prefix_lines = [
            "[Review metadata]",
            f"- Main document: {st.session_state.last_doc_name or '(not uploaded)'}",
            f"- Document type: {st.session_state.document_type or '(not selected)'}",
            f"- Framework: {FRAMEWORKS[selected_key]['name_en']}",
        ]

        up_ref = st.session_state.get("upstream_reference")
        q_hist = st.session_state.get("quote_reference_history", []) or []

        if up_ref or q_hist:
            prefix_lines.append("- Reference documents upload log:")
            if up_ref:
                ext = (up_ref.get("ext", "") or "").upper()
                prefix_lines.append(f"  Upstream: {up_ref.get('name','')}" + (f" ({ext})" if ext else ""))
            if q_hist:
                prefix_lines.append("  Quote references:")
                for i, r in enumerate(q_hist, start=1):
                    ext = (r.get("ext", "") or "").upper()
                    prefix_lines.append(f"    {i}. {r.get('name','')}" + (f" ({ext})" if ext else ""))
        else:
            prefix_lines.append("- Reference documents: (none)")

        prefix = "\n".join(prefix_lines) + "\n\n"

        combined_sections = [
            prefix,
            "==============================",
            "(Step 5) Main document analysis",
            "==============================",
            current_state.get("step5_output", ""),
        ]

        combined_sections += [
            "",
            "==============================",
            "(Step 6) Reference relevance / consistency outputs",
            "==============================",
        ]

        # 6-1 upstream
        if current_state.get("step6_upstream_done"):
            combined_sections += [
                "[6-1 Upstream relevance]",
                current_state.get("step6_upstream_output", ""),
                "",
            ]
        elif st.session_state.get("upstream_reference"):
            combined_sections += ["[6-1 Upstream relevance] (not run yet)", "", ""]

        # 6-2 quote (history)
        q_hist = st.session_state.get("quote_reference_history", []) or []
        analyzed_quotes = [r for r in q_hist if r.get("analyzed") and r.get("output")]
        if analyzed_quotes:
            combined_sections.append("[6-2 Quote relevance (history)]")
            for i, r in enumerate(analyzed_quotes, start=1):
                combined_sections += [
                    f"--- Quote #{i}: {r.get('name','')} ---",
                    r.get("output", ""),
                    "",
                ]
        elif q_hist:
            combined_sections += ["[6-2 Quote relevance] (not run yet)", "", ""]

        combined_sections += [
            "",
            "==============================",
            "(Step 7) Final deliverable",
            "==============================",
            current_state.get("step7_output", ""),
        ]
        current_state["analysis_output"] = "\n".join(combined_sections).strip()

    save_state_to_disk()
    st.success(zh("步驟七完成。", "步骤七完成。") if lang == "zh" else "Step 7 completed.")

if current_state.get("step7_done"):
    st.markdown(current_state.get("step7_output", ""))

# =========================
# Results ordered by steps (kept, but avoid duplication in long report preview)
# =========================
st.markdown("## " + (zh("Results (ordered by steps)", "Results (ordered by steps)") if lang == "zh" else "Results (ordered by steps)"))

if current_state.get("step5_done"):
    st.markdown("### " + (zh("步驟五：主文件分析結果", "步骤五：主文件分析结果") if lang == "zh" else "Step 5: Main document analysis output"))
    st.markdown(current_state.get("step5_output", ""))

if has_refs:
    st.markdown("### " + (zh("步驟六：參考文件結果", "步骤六：参考文件结果") if lang == "zh" else "Step 6: Reference outputs"))

    # 6-1 upstream
    if st.session_state.get("upstream_reference"):
        st.markdown("#### 6-1 " + ("Upstream reference relevance" if lang == "en" else zh("上游參考相關性", "上游参考相关性")))
        if current_state.get("step6_upstream_done"):
            st.markdown(current_state.get("step6_upstream_output", ""))
        else:
            st.info(zh("尚未執行 6-1。", "尚未执行 6-1。") if lang == "zh" else "6-1 has not been run yet.")

    # 6-2 quote
    q_hist = st.session_state.get("quote_reference_history", []) or []
    if q_hist:
        st.markdown("#### 6-2 " + ("Quote reference relevance (history)" if lang == "en" else zh("引述參考一致性（歷史）", "引述参考一致性（历史）")))
        analyzed = [r for r in q_hist if r.get("analyzed") and r.get("output")]
        if analyzed:
            for i, r in enumerate(analyzed, start=1):
                st.markdown(f"**{i}. {r.get('name','')}**")
                st.markdown(r.get("output", ""))
                st.markdown("---")
        else:
            st.info(zh("尚未執行 6-2。", "尚未执行 6-2。") if lang == "zh" else "6-2 has not been run yet.")

st.markdown("### " + (zh("步驟七：最終整合成品", "步骤七：最终整合成品") if lang == "zh" else "Step 7: Final deliverable"))
if current_state.get("step7_done"):
    st.markdown(current_state.get("step7_output", ""))
else:
    st.info(zh("尚未完成步驟七。", "尚未完成步骤七。") if lang == "zh" else "Step 7 has not been completed yet.")

# =========================
# Download + Preview (avoid duplicate long output in page)
# =========================
st.markdown("---")

if current_state.get("step7_done"):
    with st.expander(zh("預覽整份報告內容", "预览整份报告内容") if lang == "zh" else "Preview full report content", expanded=False):
        st.markdown(current_state.get("analysis_output", ""))

    st.markdown("#### " + ("Download" if lang == "en" else zh("下載", "下载")))
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Error-Free_Report_{now_str}.txt"

    include_qa = st.checkbox(
        zh("下載時包含 Q&A 紀錄（Follow-up）", "下载时包含 Q&A 纪录（Follow-up）") if lang == "zh" else "Include Q&A history (follow-ups) in download",
        value=True,
        key=f"include_qa_{selected_key}",
    )
    report = build_full_report(lang, selected_key, current_state, include_followups=include_qa)

    st.download_button(
        label=("Download report (.txt)" if lang == "en" else zh("下載報告（.txt）", "下载报告（.txt）")),
        data=report.encode("utf-8"),
        file_name=filename,
        mime="text/plain",
    )

    # Step 7: Reset Whole Document (confirmation required) - 更正2
    if "confirm_reset_whole" not in st.session_state:
        st.session_state.confirm_reset_whole = False

    if st.button(
        zh("Reset Whole Document（清空整輪審查，重新開始）", "Reset Whole Document（清空整轮审查，重新开始）") if lang == "zh" else "Reset Whole Document (start a new review)",
        key="reset_whole_btn",
    ):
        st.session_state.confirm_reset_whole = True

    if st.session_state.confirm_reset_whole:
        st.warning(
            zh(
                "警告：Reset Whole Document 會清空本輪主文件、所有參考文件、步驟五/六/七結果與 Q&A 紀錄，並重新開始。此動作不可復原。",
                "警告：Reset Whole Document 会清空本轮主文件、所有参考文件、步骤五/六/七结果与 Q&A 纪录，并重新开始。此动作不可复原。",
            )
            if lang == "zh"
            else "Warning: Reset Whole Document will clear the main document, all reference documents, Step 5/6/7 outputs, and Q&A history, and then restart the app state. This cannot be undone."
        )
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button(zh("確認清空並重新開始", "确认清空并重新开始") if lang == "zh" else "Confirm reset", key="confirm_reset_whole_btn"):
                # Clear core workflow state (do not change authentication / company)
                st.session_state.framework_states = {}
                st.session_state.last_doc_text = ""
                st.session_state.last_doc_name = ""
                st.session_state.document_type = None
                st.session_state.current_doc_id = None

                # References (更正2)
                st.session_state.upstream_reference = None
                st.session_state.quote_reference_history = []
                st.session_state.quote_ref_pending = False
                st.session_state.quote_uploader_key = 0
                st.session_state.upstream_uploader_key = 0

                # Backward compatible fields
                st.session_state.reference_history = []
                st.session_state.ref_pending = False

                st.session_state.confirm_reset_whole = False
                save_state_to_disk()
                st.rerun()
        with col_b:
            if st.button(zh("取消", "取消") if lang == "zh" else "Cancel", key="cancel_reset_whole_btn"):
                st.session_state.confirm_reset_whole = False
                st.rerun()

else:
    st.info(
        zh("尚未完成最終整合（步驟七）。完成後才能下載完整報告。", "尚未完成最终整合（步骤七）。完成后才能下载完整报告。")
        if lang == "zh"
        else "Step 7 is not completed yet. Complete Step 7 to enable download."
    )
