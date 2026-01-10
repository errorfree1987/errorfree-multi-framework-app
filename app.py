# -*- coding: utf-8 -*-
"""
Error-Free® Intelligence Engine - Streamlit App
Updated per: 更正2.docx + Step 6 Quote Reference Relevance Analysis.docx
"""

import os
import re
import io
import json
import time
import uuid
import math
import base64
import hashlib
import random
import string
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import streamlit as st

# Doc parsing
import docx
import fitz  # PyMuPDF
from PIL import Image

# Report exports
from docx import Document as DocxDocument
from docx.shared import Pt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as pdf_canvas

# PPTX
from pptx import Presentation
from pptx.util import Inches, Pt as PPTPt

# LLM
import openai

# -----------------------
# Paths / Config
# -----------------------
BASE_DIR = Path(__file__).resolve().parent
STATE_PATH = BASE_DIR / "state.json"
USAGE_PATH = BASE_DIR / "usage.json"
DOC_TRACKING_PATH = BASE_DIR / "doc_tracking.json"
FRAMEWORKS_PATH = BASE_DIR / "frameworks.json"

LOGO_PATH = str(BASE_DIR / "static" / "logo.png")

BRAND_TITLE_ZH = "Error-Free®\nIntelligence Engine"
BRAND_TITLE_EN = "Error-Free®\nIntelligence Engine"
BRAND_TAGLINE_ZH = "AI-enhanced intelligence engine that helps organizations analyze risks, prevent errors, and make better decisions."
BRAND_TAGLINE_EN = "AI-enhanced intelligence engine that helps organizations analyze risks, prevent errors, and make better decisions."
BRAND_SUBTITLE_ZH = "Pioneered and refined by Dr. Chiu’s Error-Free® team since 1987."
BRAND_SUBTITLE_EN = "Pioneered and refined by Dr. Chiu’s Error-Free® team since 1987."

DEFAULT_MODEL = "gpt-4o-mini"

# -----------------------
# Utilities
# -----------------------


def zh(tw: str, cn: str) -> str:
    """Return zh text by variant."""
    return cn if st.session_state.get("zh_variant", "tw") == "cn" else tw


def load_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def save_json(path: Path, data):
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def load_state_from_disk() -> Dict:
    return load_json(STATE_PATH, {})


def save_state_to_disk():
    state = {
        "user_email": st.session_state.get("user_email"),
        "user_role": st.session_state.get("user_role"),
        "is_authenticated": st.session_state.get("is_authenticated"),
        "show_admin": st.session_state.get("show_admin"),
        "lang": st.session_state.get("lang"),
        "zh_variant": st.session_state.get("zh_variant"),
        "framework_states": st.session_state.get("framework_states", {}),
        "selected_framework_key": st.session_state.get("selected_framework_key"),
        "last_doc_text": st.session_state.get("last_doc_text", ""),
        "last_doc_name": st.session_state.get("last_doc_name", ""),
        "document_type": st.session_state.get("document_type"),
        "current_doc_id": st.session_state.get("current_doc_id"),
        "download_used_map": st.session_state.get("download_used_map", {}),
        "doc_cycle_locked": st.session_state.get("doc_cycle_locked", False),
        "confirm_reset_whole": st.session_state.get("confirm_reset_whole", False),
        "doc_ref_state": st.session_state.get("doc_ref_state", {}),
    }
    save_json(STATE_PATH, state)


def load_usage() -> Dict:
    return load_json(USAGE_PATH, {})


def save_usage(data: Dict):
    save_json(USAGE_PATH, data)


def record_usage(user_email: str, framework_key: str, action: str):
    usage = load_usage()
    if user_email not in usage:
        usage[user_email] = {}
    if framework_key not in usage[user_email]:
        usage[user_email][framework_key] = {}
    usage[user_email][framework_key][action] = usage[user_email][framework_key].get(action, 0) + 1
    save_usage(usage)


def load_doc_tracking() -> Dict:
    return load_json(DOC_TRACKING_PATH, {})


def save_doc_tracking(data: Dict):
    save_json(DOC_TRACKING_PATH, data)


def clean_report_text(s: str) -> str:
    if not s:
        return ""
    # Minimal cleanup only
    s = s.strip()
    s = s.replace("\r\n", "\n")
    return s


def hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


# -----------------------
# File reading
# -----------------------


def read_pdf_to_text(file_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        texts = []
        for page in doc:
            texts.append(page.get_text("text"))
        return "\n".join(texts).strip()
    except Exception:
        return ""


def read_docx_to_text(file_bytes: bytes) -> str:
    try:
        f = io.BytesIO(file_bytes)
        d = docx.Document(f)
        paras = [p.text for p in d.paragraphs if p.text]
        return "\n".join(paras).strip()
    except Exception:
        return ""


def read_txt_to_text(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="ignore").strip()
    except Exception:
        try:
            return file_bytes.decode("latin-1", errors="ignore").strip()
        except Exception:
            return ""


def read_image_to_text(file_bytes: bytes) -> str:
    # NOTE: No OCR in this app (by design). Keep original behavior: return placeholder.
    # If the app previously supported OCR, it would be implemented here.
    return ""


def read_file_to_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".pdf"):
        return read_pdf_to_text(data)
    if name.endswith(".docx"):
        return read_docx_to_text(data)
    if name.endswith(".txt"):
        return read_txt_to_text(data)
    if name.endswith(".png") or name.endswith(".jpg") or name.endswith(".jpeg"):
        return read_image_to_text(data)
    return ""


# -----------------------
# LLM helpers
# -----------------------


def run_simple_llm(system_prompt: str, user_prompt: str, model_name: str) -> str:
    try:
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=model_name or DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"LLM error: {e}"


def load_frameworks() -> Dict[str, Dict]:
    if FRAMEWORKS_PATH.exists():
        try:
            return json.loads(FRAMEWORKS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


FRAMEWORKS = load_frameworks()


def run_llm_analysis(framework_key: str, lang: str, input_text: str, model_name: str) -> str:
    fw = FRAMEWORKS.get(framework_key, {})
    sys_prompt = fw.get("system_prompt_zh") if lang == "zh" else fw.get("system_prompt_en")
    usr_tpl = fw.get("user_prompt_zh") if lang == "zh" else fw.get("user_prompt_en")

    if not sys_prompt:
        sys_prompt = "You are a helpful assistant."
    if not usr_tpl:
        usr_tpl = "{input}"

    user_prompt = usr_tpl.replace("{input}", input_text)

    return run_simple_llm(sys_prompt, user_prompt, model_name)


# -----------------------
# Existing Step5 input builder (kept)
# -----------------------


def build_main_only_input(lang: str, document_type: str, main_text: str) -> str:
    if lang == "zh":
        return f"""【文件類型】{document_type}

【主文件內容】
{main_text}

【任務】
請根據所選的 Error-Free 框架，對主文件進行分析（不含任何參考文件）。"""
    return f"""[Document type] {document_type}

[Main document]
{main_text}

[Task]
Analyze the main document using the selected Error-Free framework (no reference documents)."""


# -----------------------
# Reporting / export
# -----------------------


def build_full_report(lang: str, framework_key: str, state: Dict, include_qa: bool = True) -> str:
    fw = FRAMEWORKS.get(framework_key, {})
    fw_name = fw.get("name_zh") if lang == "zh" else fw.get("name_en") or framework_key

    title = zh("Error-Free® Intelligence Engine 報告", "Error-Free® Intelligence Engine 报告") if lang == "zh" else "Error-Free® Intelligence Engine Report"
    parts = [f"# {title}", ""]
    parts.append(f"## {'框架' if lang == 'zh' else 'Framework'}: {fw_name}")
    parts.append("")
    parts.append(f"## {'主文件' if lang == 'zh' else 'Review document'}: {st.session_state.get('last_doc_name','')}")
    parts.append("")

    analysis_output = state.get("analysis_output") or ""
    if analysis_output:
        parts.append("## " + (zh("正式報告", "正式报告") if lang == "zh" else "Formal report"))
        parts.append(analysis_output)
        parts.append("")

    followups = state.get("chat_history", [])
    if include_qa and followups:
        parts.append("## " + (zh("對話框歷史紀錄（Q&A）", "对话框历史纪录（Q&A）") if lang == "zh" else "Q&A history"))
        for msg in followups:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role and content:
                parts.append(f"**{role}**: {content}")
        parts.append("")

    return "\n".join(parts).strip()


def build_docx_bytes(report_markdown: str) -> bytes:
    doc = DocxDocument()
    for line in report_markdown.splitlines():
        doc.add_paragraph(line)
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()


def build_pdf_bytes(report_markdown: str) -> bytes:
    bio = io.BytesIO()
    c = pdf_canvas.Canvas(bio, pagesize=letter)
    width, height = letter
    x = 50
    y = height - 50
    for line in report_markdown.splitlines():
        if y < 50:
            c.showPage()
            y = height - 50
        c.drawString(x, y, line[:120])
        y -= 14
    c.save()
    return bio.getvalue()


def build_pptx_bytes(report_markdown: str, lang: str) -> bytes:
    prs = Presentation()
    slide_layout = prs.slide_layouts[1]
    lines = report_markdown.splitlines()
    chunk = []
    max_lines = 14

    def add_slide(lines_chunk: List[str]):
        slide = prs.slides.add_slide(slide_layout)
        slide.shapes.title.text = "Error-Free® Report"
        body = slide.shapes.placeholders[1].text_frame
        body.clear()
        for i, ln in enumerate(lines_chunk):
            p = body.add_paragraph() if i > 0 else body.paragraphs[0]
            p.text = ln[:200]
            p.font.size = PPTPt(14)

    for ln in lines:
        chunk.append(ln)
        if len(chunk) >= max_lines:
            add_slide(chunk)
            chunk = []
    if chunk:
        add_slide(chunk)

    bio = io.BytesIO()
    prs.save(bio)
    return bio.getvalue()


# -----------------------
# Step 6 new analyses (per spec)
# -----------------------


def derive_upstream_relevance(language: str, main_doc_analysis: str, upstream_name: str, upstream_text: str, model_name: str) -> str:
    """
    Step 6 (upstream relevance): check consistency between upstream reference and the main document.
    The output focuses on upstream document errors (critical info missing/mismatched).
    """
    if language == "zh":
        sys = "你是一位嚴謹的審查與一致性核對專家。不得杜撰。輸出要可追溯、可執行。"
        user = (
            "任務：執行『主要參考文件（Upstream）相關性分析』。\n\n"
            "[判準（請逐條檢查）]\n"
            "1) Upstream 參考文件中的關鍵資訊（目的/背景/限制/要求/結論/假設）在主文件中是否被忽略或缺失？（缺失= upstream document error）\n"
            "2) 主文件目的是否與 Upstream 目的一致或可推導（衍生）？\n"
            "3) Upstream 內引用/要求/規範與主文件對應段落是否一致（包含數值、條件、限制、責任分工）？\n"
            "4) Upstream 的結論是否與主文件的目的/分析/結論矛盾？\n\n"
            "[輸出格式]\n"
            "- 請用 Markdown。\n"
            "- 先給一段 5~10 行的 Executive Summary。\n"
            "- 接著用表格列出每一個發現（至少包含：Issue 類型、主文件對應、Upstream 對應、風險/影響、建議修正）。\n"
            "- 最後給『需要澄清的問題清單』。\n\n"
            f"[主文件（Step 5）框架分析輸出]\n{main_doc_analysis}\n\n"
            f"[Upstream 參考文件：{upstream_name}]\n{upstream_text}\n"
        )
    else:
        sys = "You are a strict review and consistency checker. Do not hallucinate. The output must be traceable and actionable."
        user = (
            "Task: Perform an 'Upstream Reference Relevance Analysis' (consistency check).\n\n"
            "[Checks]\n"
            "1) Are critical items in the upstream reference (purpose/context/constraints/requirements/conclusions/assumptions) missing or ignored in the main document? (Missing = upstream document error)\n"
            "2) Is the main document purpose consistent with, or a derivative of, the upstream purpose?\n"
            "3) Are quoted requirements/specs in the upstream consistent with corresponding parts in the main document (values/conditions/limits/responsibilities)?\n"
            "4) Do upstream conclusions contradict the purpose/analysis/conclusions of the main document?\n\n"
            "[Output format]\n"
            "- Markdown.\n"
            "- Executive Summary (5-10 lines).\n"
            "- A table of findings (at least: Issue type, Main doc reference, Upstream reference, Risk/impact, Recommended fix).\n"
            "- A final list of clarification questions.\n\n"
            f"[Main document Step-5 framework output]\n{main_doc_analysis}\n\n"
            f"[Upstream reference: {upstream_name}]\n{upstream_text}\n"
        )

    return run_simple_llm(sys, user, model_name)


def derive_quote_relevance(language: str, main_doc_text: str, quote_name: str, quote_text: str, model_name: str) -> str:
    """
    Step 6 (quote relevance): check whether quotes in the main document that reference the quote reference
    are consistent with the quote reference document. Focus on reference inconsistency errors.
    """
    if language == "zh":
        sys = "你是一位嚴謹的引用核對專家。目標是找出『引用不一致』：主文件中引用/摘錄/轉述的內容，與參考文件原文不一致或被錯誤解讀。不得杜撰。"
        user = (
            "任務：執行『Quote Reference Relevance Analysis』。\n\n"
            "[要求]\n"
            "1) 先在主文件中辨識『看起來像引用/轉述/摘錄』的內容（例如：引號、引用句、條列式引用、出處/文件名/章節號、或明顯在描述某份參考文件的要求/結論）。\n"
            "2) 針對每一則疑似引用，去 Quote Reference 文件中找出最可能對應的原文/段落，並判斷是否一致：\n"
            "   - 文字/數值/條件是否被改寫成不同意義\n"
            "   - 是否遺漏關鍵限制/前提\n"
            "   - 是否把建議寫成必須、或把必須寫成建議\n"
            "   - 是否錯用章節/條款/編號\n"
            "3) 將每一項不一致標記為『reference inconsistency error』，並給出修正建議。\n\n"
            "[輸出格式]\n"
            "- Markdown。\n"
            "- 先給 Executive Summary。\n"
            "- 接著用表格列出：主文件引用內容、Quote Reference 對應原文/依據、判定（一致/不一致/無法確認）、理由、建議修正。\n"
            "- 最後列出『需要人工確認的項目』與『建議補充的引用資訊（例如頁碼/章節）』。\n\n"
            f"[主文件全文]\n{main_doc_text}\n\n"
            f"[Quote Reference：{quote_name}]\n{quote_text}\n"
        )
    else:
        sys = "You are a strict quotation/attribution checker. Identify 'reference inconsistency errors': quotes, paraphrases, or cited requirements in the main document that do not match the quote reference. Do not hallucinate."
        user = (
            "Task: Perform a 'Quote Reference Relevance Analysis'.\n\n"
            "[Requirements]\n"
            "1) First, identify in the main document any content that appears to be a quote, paraphrase, excerpt, or a specific reference to an external document (quotes, numbered clauses, 'per X', etc.).\n"
            "2) For each candidate quote/paraphrase, locate the best-matching source passage in the Quote Reference and judge consistency:\n"
            "   - text/values/conditions changed meaningfully\n"
            "   - missing critical constraints/assumptions\n"
            "   - turning 'should' into 'shall' or vice versa\n"
            "   - misused clause/section numbers\n"
            "3) Flag each inconsistency as a 'reference inconsistency error' and propose a concrete fix.\n\n"
            "[Output format]\n"
            "- Markdown.\n"
            "- Executive Summary.\n"
            "- A table with: Main-doc quoted/paraphrased content, Quote-reference evidence, Verdict (consistent/inconsistent/uncertain), Rationale, Recommended fix.\n"
            "- A final list of items requiring manual confirmation and suggested citation details (page/section).\n\n"
            f"[Main document full text]\n{main_doc_text}\n\n"
            f"[Quote Reference: {quote_name}]\n{quote_text}\n"
        )

    return run_simple_llm(sys, user, model_name)


# -----------------------
# Language selector (updated per spec)
# -----------------------


def language_selector():
    """Language selection (before login); hide Chinese UI labels after login if English selected."""
    # Per UX requirement: once logged in and English is selected, hide the language selector
    # and all Chinese UI labels in the sidebar.
    if st.session_state.get("is_authenticated") and st.session_state.get("lang") == "en":
        return

    current_lang = st.session_state.get("lang", "zh")
    current_variant = st.session_state.get("zh_variant", "tw")

    if current_lang == "en":
        label = "Language"
        options = ("English", "Simplified Chinese", "Traditional Chinese")
        index = 0
    else:
        label = "Language / 語言"
        options = ("English", "中文简体", "中文繁體")
        index = 1 if current_variant == "cn" else 2

    choice = st.radio(label, options, index=index)

    if choice in ("English",):
        st.session_state.lang = "en"
        st.session_state.zh_variant = st.session_state.get("zh_variant", "tw")
    elif choice in ("Simplified Chinese", "中文简体"):
        st.session_state.lang = "zh"
        st.session_state.zh_variant = "cn"
    else:
        st.session_state.lang = "zh"
        st.session_state.zh_variant = "tw"


# -----------------------
# Main App
# -----------------------


def main():
    st.set_page_config(page_title="Error-Free® Intelligence Engine", layout="wide")

    # init defaults
    defaults = [
        ("user_email", None),
        ("user_role", None),
        ("is_authenticated", False),
        ("show_admin", False),
        ("lang", "zh"),
        ("zh_variant", "tw"),
        ("framework_states", {}),
        ("selected_framework_key", None),
        ("last_doc_text", ""),
        ("last_doc_name", ""),
        ("document_type", None),
        ("current_doc_id", None),
        ("download_used_map", {}),
        ("doc_cycle_locked", False),
        ("confirm_reset_whole", False),
        ("doc_ref_state", {}),
    ]
    for k, v in defaults:
        if k not in st.session_state:
            st.session_state[k] = v

    # restore from disk on first load
    disk = load_state_from_disk()
    if disk and not st.session_state.get("_restored", False):
        for k, _ in defaults:
            if k in disk and disk[k] is not None:
                st.session_state[k] = disk[k]
        st.session_state["_restored"] = True

    lang = st.session_state.lang
    is_guest = (st.session_state.get("user_role") == "guest")
    user_email = st.session_state.get("user_email") or ""

    # Sidebar
    doc_tracking = load_doc_tracking()

    with st.sidebar:
        lang = st.session_state.lang

        language_selector()
        lang = st.session_state.lang

        if st.session_state.is_authenticated and st.session_state.user_role in ["admin", "pro", "company_admin"]:
            if st.button((zh("管理後台", "管理后台") + " Admin Dashboard") if lang == "zh" else "Admin Dashboard"):
                st.session_state.show_admin = True
                save_state_to_disk()
                st.rerun()

        st.markdown("---")

        if st.session_state.is_authenticated:
            st.subheader(zh("帳號資訊", "账号信息") if lang == "zh" else "Account")
            st.write(f"Email：{st.session_state.user_email}")

            if st.button(zh("登出", "退出登录") if lang == "zh" else "Logout"):
                st.session_state.user_email = None
                st.session_state.user_role = None
                st.session_state.is_authenticated = False
                st.session_state.framework_states = {}
                st.session_state.last_doc_text = ""
                st.session_state.last_doc_name = ""
                st.session_state.document_type = None
                st.session_state.current_doc_id = None
                st.session_state.doc_cycle_locked = False
                st.session_state.confirm_reset_whole = False
                st.session_state.doc_ref_state = {}
                save_state_to_disk()
                st.rerun()
        else:
            st.subheader(zh("尚未登入", "尚未登录") if lang == "zh" else "Not Logged In")
            if lang == "zh":
                st.markdown(
                    "- " + zh("上方：內部員工 / 會員登入。", "上方：内部员工 / 会员登录。") + "\n"
                    "- " + zh("中間：公司管理者（企業端窗口）登入 / 註冊。", "中间：公司管理者（企业端窗口）登录 / 注册。") + "\n"
                    "- " + zh("下方：學生 / 客戶的 Guest 試用登入 / 註冊。", "下方：学员 / 客户的 Guest 试用登录 / 注册。")
                )
            else:
                st.markdown(
                    "- Top: internal Error-Free employees / members.\n"
                    "- Middle: **Company Admins** for each client company.\n"
                    "- Bottom: students / end-users using **Guest trial accounts**."
                )

    # ======= Login screen =======
    if not st.session_state.is_authenticated:
        lang = st.session_state.lang

        if Path(LOGO_PATH).exists():
            st.image(LOGO_PATH, width=260)

        title = BRAND_TITLE_ZH if lang == "zh" else BRAND_TITLE_EN
        tagline = BRAND_TAGLINE_ZH if lang == "zh" else BRAND_TAGLINE_EN
        subtitle = BRAND_SUBTITLE_ZH if lang == "zh" else BRAND_SUBTITLE_EN

        st.title(title)
        st.write(tagline)
        st.caption(subtitle)
        st.markdown("---")

        # Minimal login UI (kept)
        if lang == "zh":
            st.markdown(zh("請登入後使用系統。", "请登录后使用系统。"))
        else:
            st.markdown("Please log in to use the system.")

        # Existing login logic placeholder (kept)
        email = st.text_input("Email" if lang == "en" else zh("Email", "Email"))
        role = st.selectbox("Role" if lang == "en" else zh("角色", "角色"), ["guest", "pro", "admin", "company_admin"])
        if st.button("Login"):
            st.session_state.user_email = email.strip()
            st.session_state.user_role = role
            st.session_state.is_authenticated = True
            save_state_to_disk()
            st.rerun()

        return

    # ======= Main UI =======
    st.title(BRAND_TITLE_ZH if lang == "zh" else BRAND_TITLE_EN)
    st.write(BRAND_TAGLINE_ZH if lang == "zh" else BRAND_TAGLINE_EN)
    st.caption(BRAND_SUBTITLE_ZH if lang == "zh" else BRAND_SUBTITLE_EN)
    st.markdown("---")

    model_name = st.session_state.get("model_name") or DEFAULT_MODEL

    # Step 1: Upload review document
    st.subheader(zh("步驟一：上傳審閱文件", "步骤一：上传审阅文件") if lang == "zh" else "Step 1: Upload Review Document")
    st.caption(
        zh("注意：一次只能上傳 1 份文件以完成內容分析。", "注意：一次只能上传 1 份文件以完成内容分析。") if lang == "zh"
        else "Note: Only 1 document can be uploaded for a complete content analysis."
    )

    doc_locked = bool(st.session_state.get("last_doc_text"))

    if not doc_locked:
        uploaded = st.file_uploader(
            zh("請上傳 PDF / DOCX / TXT / 圖片", "请上传 PDF / DOCX / TXT / 图片") if lang == "zh" else "Upload PDF / DOCX / TXT / Image",
            type=["pdf", "docx", "txt", "jpg", "jpeg", "png"],
            key="review_doc_uploader",
        )

        if uploaded is not None:
            doc_text = read_file_to_text(uploaded)
            if doc_text:
                if is_guest:
                    docs = doc_tracking.get(user_email, [])
                    if len(docs) >= 3 and st.session_state.current_doc_id not in docs:
                        st.error(zh("試用帳號最多上傳 3 份文件", "试用账号最多上传 3 份文件") if lang == "zh" else "Trial accounts may upload up to 3 documents only")
                    else:
                        if st.session_state.current_doc_id not in docs:
                            new_id = f"doc_{datetime.datetime.now().timestamp()}"
                            docs.append(new_id)
                            doc_tracking[user_email] = docs
                            st.session_state.current_doc_id = new_id
                            save_doc_tracking(doc_tracking)
                        st.session_state.last_doc_text = doc_text
                        st.session_state.last_doc_name = uploaded.name
                        save_state_to_disk()
                        st.rerun()
                else:
                    st.session_state.current_doc_id = f"doc_{datetime.datetime.now().timestamp()}"
                    st.session_state.last_doc_text = doc_text
                    st.session_state.last_doc_name = uploaded.name
                    save_state_to_disk()
                    st.rerun()
    else:
        shown_name = st.session_state.get("last_doc_name") or zh("（已上傳）", "（已上传）")
        st.info(
            zh(f"已上傳審閱文件：{shown_name}。如需更換文件，請使用 Reset Whole Document。", f"已上传审阅文件：{shown_name}。如需更换文件，请使用 Reset Whole Document。")
            if lang == "zh"
            else f"Review document uploaded: {shown_name}. To change it, use Reset Whole Document."
        )

    st.markdown("---")

    # Step 2: Document Type Selection (Fix zh labels, keep value = English)
    st.subheader(zh("步驟二：文件類型選擇（單選）", "步骤二：文件类型选择（单选）") if lang == "zh" else "Step 2: Document Type Selection")
    st.caption(zh("單選", "单选") if lang == "zh" else "Single selection")

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
            disabled=st.session_state.get("doc_cycle_locked", False),
            key="document_type_select_zh",
        )
        st.session_state.document_type = label_to_value.get(picked_label, DOC_TYPES[0])
    else:
        st.session_state.document_type = st.selectbox(
            "Select document type",
            DOC_TYPES,
            index=DOC_TYPES.index(st.session_state.document_type),
            disabled=st.session_state.get("doc_cycle_locked", False),
            key="document_type_select",
        )

    save_state_to_disk()

    if (not st.session_state.get("doc_cycle_locked", False)) and (st.session_state.get("document_type") == "Specifications and Requirements"):
        st.warning(
            (zh("提醒：一旦在步驟五開始分析後，將無法再更換文件類型。若需更換，請到步驟七使用 Reset Whole Document 重新開始新一輪審查。",
                "提醒：一旦在步骤五开始分析后，将无法再更换文件类型。若需更换，请到步骤七使用 Reset Whole Document 重新开始新一轮审查。")
             if lang == "zh"
             else "Note: After you run Step 5, you cannot change the document type. To change it, use Step 7 → Reset Whole Document to start a new review cycle.")
        )

    # Step 3: Reference documents (optional)
    st.subheader(zh("步驟三：上傳參考文件（選填）", "步骤三：上传参考文件（选填）") if lang == "zh" else "Step 3: Upload Reference Documents (optional)")
    st.caption(
        zh(
            "本步驟分為兩個分支：3-1 主要參考文件（Upstream）只能上傳一次；3-2 次要參考文件（Quote Reference）可多次上傳並重複分析。"
            "若 3-1 有上傳，需先完成 upstream relevance 後才會開放 quote relevance（避免等待過久）。",
            "本步骤分为两个分支：3-1 主要参考文件（Upstream）只能上传一次；3-2 次要参考文件（Quote Reference）可多次上传并重复分析。"
            "若 3-1 有上传，需先完成 upstream relevance 后才会开放 quote relevance（避免等待过久）。",
        )
        if lang == "zh"
        else "This step has two branches: 3-1 Upstream reference can be uploaded only once; 3-2 Quote reference can be uploaded/analyzed multiple times. If 3-1 is uploaded, quote relevance is enabled only after upstream relevance completes."
    )

    # Per-document-cycle reference state (not framework-specific)
    doc_ref_state = st.session_state.get("doc_ref_state") or {}
    doc_ref_state.setdefault("upstream_ref_text", "")
    doc_ref_state.setdefault("upstream_ref_name", "")
    doc_ref_state.setdefault("quote_ref_text", "")
    doc_ref_state.setdefault("quote_ref_name", "")
    doc_ref_state.setdefault("quote_history", [])  # list of {"name":..., "output":..., "ts":...}
    st.session_state.doc_ref_state = doc_ref_state

    # ---------- 3-1 Upstream ----------
    st.markdown("### 3-1 " + (zh("上傳主要參考文件（Upstream）（選填）", "上传主要参考文件（Upstream）（选填）") if lang == "zh" else "Upload Upstream Reference Document (optional)"))
    st.caption(
        zh(
            "主要參考文件只能上傳一次；上傳後即鎖定。完成步驟六（upstream relevance）後按鍵反灰，資料會保留到步驟七 Reset Whole Document。",
            "主要参考文件只能上传一次；上传后即锁定。完成步骤六（upstream relevance）后按键反灰，资料会保留到步骤七 Reset Whole Document。",
        )
        if lang == "zh"
        else "Upstream reference can be uploaded once and is locked immediately. After Step 6 (upstream relevance) runs, the button is disabled; data persists until Step 7 → Reset Whole Document."
    )

    upstream_locked = bool(doc_ref_state.get("upstream_ref_text"))
    if not upstream_locked:
        up_file = st.file_uploader(
            zh("上傳主要參考文件（PDF / DOCX / TXT / 圖片）", "上传主要参考文件（PDF / DOCX / TXT / 图片）") if lang == "zh" else "Upload upstream reference (PDF / DOCX / TXT / Image)",
            type=["pdf", "docx", "txt", "jpg", "jpeg", "png"],
            key="upstream_uploader",
        )
        if up_file is not None:
            up_text = read_file_to_text(up_file)
            if up_text:
                doc_ref_state["upstream_ref_text"] = up_text
                doc_ref_state["upstream_ref_name"] = up_file.name
                st.session_state.doc_ref_state = doc_ref_state
                save_state_to_disk()
                st.success(zh("已上傳主要參考文件並鎖定。", "已上传主要参考文件并锁定。") if lang == "zh" else "Upstream reference uploaded and locked.")
            else:
                st.error(zh("無法讀取該文件內容。", "无法读取该文件内容。") if lang == "zh" else "Unable to read the file content.")
    else:
        st.info(
            (zh(f"主要參考文件已上傳：{doc_ref_state.get('upstream_ref_name') or '（已上傳）'}（已鎖定）",
                f"主要参考文件已上传：{doc_ref_state.get('upstream_ref_name') or '（已上传）'}（已锁定）")
             if lang == "zh"
             else f"Upstream reference uploaded: {doc_ref_state.get('upstream_ref_name') or '(uploaded)'} (locked)")
        )

    st.markdown("---")

    # ---------- 3-2 Quote reference ----------
    st.markdown("### 3-2 " + (zh("上傳次要參考文件（Quote Reference）（選填）", "上传次要参考文件（Quote Reference）（选填）") if lang == "zh" else "Upload Quote Reference Document (optional)"))
    st.caption(
        zh(
            "次要參考文件可多次上傳做 quote relevance 分析；每次上傳後會鎖定。"
            "若需更換新的次要參考文件，請使用本區的 Reset Document 重新上傳。歷史分析結果會保留並依序排列。",
            "次要参考文件可多次上传做 quote relevance 分析；每次上传后会锁定。"
            "若需更换新的次要参考文件，请使用本区的 Reset Document 重新上传。历史分析结果会保留并依序排列。",
        )
        if lang == "zh"
        else "Quote reference can be uploaded and analyzed multiple times. Each upload locks immediately. Use Reset Document here to upload a new quote reference; prior results remain in history."
    )

    quote_locked = bool(doc_ref_state.get("quote_ref_text"))
    if not quote_locked:
        q_file = st.file_uploader(
            zh("上傳次要參考文件（PDF / DOCX / TXT / 圖片）", "上传次要参考文件（PDF / DOCX / TXT / 图片）") if lang == "zh" else "Upload quote reference (PDF / DOCX / TXT / Image)",
            type=["pdf", "docx", "txt", "jpg", "jpeg", "png"],
            key="quote_uploader",
        )
        if q_file is not None:
            q_text = read_file_to_text(q_file)
            if q_text:
                doc_ref_state["quote_ref_text"] = q_text
                doc_ref_state["quote_ref_name"] = q_file.name
                st.session_state.doc_ref_state = doc_ref_state
                save_state_to_disk()
                st.success(zh("已上傳次要參考文件並鎖定。", "已上传次要参考文件并锁定。") if lang == "zh" else "Quote reference uploaded and locked.")
            else:
                st.error(zh("無法讀取該文件內容。", "无法读取该文件内容。") if lang == "zh" else "Unable to read the file content.")
    else:
        st.info(
            (zh(f"次要參考文件已上傳：{doc_ref_state.get('quote_ref_name') or '（已上傳）'}（已鎖定）",
                f"次要参考文件已上传：{doc_ref_state.get('quote_ref_name') or '（已上传）'}（已锁定）")
             if lang == "zh"
             else f"Quote reference uploaded: {doc_ref_state.get('quote_ref_name') or '(uploaded)'} (locked)")
        )
        if st.button(zh("Reset Document（更換次要參考文件）", "Reset Document（更换次要参考文件）") if lang == "zh" else "Reset Document (replace quote reference)", key="reset_quote_doc_btn"):
            doc_ref_state["quote_ref_text"] = ""
            doc_ref_state["quote_ref_name"] = ""
            st.session_state.doc_ref_state = doc_ref_state
            save_state_to_disk()
            st.rerun()

    save_state_to_disk()

    # Step 4: select framework
    st.subheader(zh("步驟四：選擇分析框架（僅單選）", "步骤四：选择分析框架（仅单选）") if lang == "zh" else "Step 4: Select Framework")
    st.caption(
        zh(
            "僅單選。如需展開新一輪審查，請到步驟七使用 Reset Whole Document。",
            "仅单选。如需展开新一轮审查，请到步骤七使用 Reset Whole Document。",
        )
        if lang == "zh"
        else "Single selection only. To start a new review cycle, use Step 7 → Reset Whole Document."
    )

    if not FRAMEWORKS:
        st.error(zh("尚未在 frameworks.json 中定義任何框架。", "尚未在 frameworks.json 中定义任何框架。") if lang == "zh" else "No frameworks defined in frameworks.json.")
        return

    fw_keys = list(FRAMEWORKS.keys())
    fw_labels = []
    label_to_key = {}
    for k in fw_keys:
        fw = FRAMEWORKS[k]
        label = (fw.get("name_zh") if lang == "zh" else fw.get("name_en")) or k
        fw_labels.append(label)
        label_to_key[label] = k

    # default selected framework
    selected_key = st.session_state.get("selected_framework_key") or fw_keys[0]
    selected_label_default = None
    for lb, ky in label_to_key.items():
        if ky == selected_key:
            selected_label_default = lb
            break
    if selected_label_default is None:
        selected_label_default = fw_labels[0]

    selected_label = st.selectbox(
        (zh("選擇框架", "选择框架") if lang == "zh" else "Select framework"),
        fw_labels,
        index=fw_labels.index(selected_label_default) if selected_label_default in fw_labels else 0,
        disabled=st.session_state.get("doc_cycle_locked", False),
        key="framework_selectbox",
    )
    selected_key = label_to_key.get(selected_label, fw_keys[0])
    st.session_state.selected_framework_key = selected_key

    if (not st.session_state.get("doc_cycle_locked", False)) and (("Omission" in (selected_label or "")) or ("遺漏" in (selected_label or ""))):
        st.warning(
            (zh("提醒：一旦在步驟五開始分析後，將無法再更換框架。若需更換，請到步驟七使用 Reset Whole Document 重新開始新一輪審查。",
                "提醒：一旦在步骤五开始分析后，将无法再更换框架。若需更换，请到步骤七使用 Reset Whole Document 重新开始新一轮审查。")
             if lang == "zh"
             else "Note: After you run Step 5, you cannot change the framework. To change it, use Step 7 → Reset Whole Document to start a new review cycle.")
        )

    framework_states = st.session_state.framework_states
    if selected_key not in framework_states:
        framework_states[selected_key] = {
            "step5_done": False,
            "step5_output": "",
            "upstream_done": False,
            "upstream_output": "",
            "analysis_output": "",
            "step7_done": False,
            "download_used": False,
            "chat_history": [],
        }
        st.session_state.framework_states = framework_states

    current_state = framework_states[selected_key]

    st.markdown("---")

    # =========================
    # Step 5 / 6 / 7 (always visible)
    # =========================

    st.subheader(zh("步驟五：先分析主要文件（快速）", "步骤五：先分析主要文件（快速）") if lang == "zh" else "Step 5: Analyze MAIN document first (fast)")
    st.caption(
        zh(
            "此步驟只分析主要文件，不處理參考文件，先快速產生第一份分析結果。完成後將鎖定文件類型與框架選擇（除非步驟七 Reset Whole Document）。",
            "此步骤只分析主要文件，不处理参考文件，先快速产生第一份分析结果。完成后将锁定文件类型与框架选择（除非步骤七 Reset Whole Document）。",
        )
        if lang == "zh"
        else "This step analyzes ONLY the main document (no references) to produce a fast first result. After completion, document type and framework are locked (unless Step 7 → Reset Whole Document)."
    )

    step5_can_run = (not current_state.get("step5_done", False))
    run_step5 = st.button(
        zh("Run analysis（main only）", "Run analysis（main only）") if lang == "zh" else "Run analysis (main only)",
        key="run_step5_btn",
        disabled=(not step5_can_run),
    )

    if run_step5:
        if not st.session_state.last_doc_text:
            st.error(zh("請先上傳審閱文件（Step 1）", "请先上传审阅文件（Step 1）") if lang == "zh" else "Please upload the review document first (Step 1).")
        elif not st.session_state.document_type:
            st.error(zh("請先在步驟二選擇文件類型", "请先在步骤二选择文件类型") if lang == "zh" else "Please select a document type in Step 2.")
        else:
            with st.spinner(zh("分析中...（主文件）", "分析中...（主文件）") if lang == "zh" else "Analyzing... (main document)"):
                main_input = build_main_only_input(lang, st.session_state.document_type, st.session_state.last_doc_text)
                out = run_llm_analysis(selected_key, lang, main_input, model_name)
                current_state["step5_done"] = True
                current_state["step5_output"] = clean_report_text(out)
                # Lock selections for the current review cycle
                st.session_state.doc_cycle_locked = True
                save_state_to_disk()
            st.success(zh("步驟五完成！已產出主文件分析結果。", "步骤五完成！已产出主文件分析结果。") if lang == "zh" else "Step 5 completed. Main document analysis generated.")

    if current_state.get("step5_output"):
        with st.expander(zh("步驟五輸出（主文件分析）", "步骤五输出（主文件分析）") if lang == "zh" else "Step 5 output (main analysis)", expanded=False):
            st.markdown(current_state.get("step5_output", ""))

    st.markdown("---")

    # =========================
    # Step 6: Reference relevance analysis (two independent buttons)
    # =========================

    st.subheader(zh("步驟六：參考文件相關性分析", "步骤六：参考文件相关性分析") if lang == "zh" else "Step 6: Reference Relevance Analysis")
    st.caption(
        zh(
            "本步驟包含兩個獨立按鍵：upstream relevance（只能跑一次）與 quote relevance（可跑多次）。",
            "本步骤包含两个独立按键：upstream relevance（只能跑一次）与 quote relevance（可跑多次）。",
        )
        if lang == "zh"
        else "This step has two independent analyses: upstream relevance (run once) and quote relevance (run multiple times)."
    )

    step5_done = bool(current_state.get("step5_done", False))
    doc_ref_state = st.session_state.get("doc_ref_state") or {}

    # ---- Upstream relevance ----
    upstream_available = bool(doc_ref_state.get("upstream_ref_text"))
    upstream_done = bool(current_state.get("upstream_done", False))
    step6_upstream_can_run = step5_done and upstream_available and (not upstream_done)

    col_u, col_q = st.columns(2)

    with col_u:
        st.markdown("**" + ((zh("Run analysis（upstream relevance）", "Run analysis（upstream relevance）") if lang == "zh" else "Run analysis (upstream relevance)")) + "**")
        run_upstream = st.button(
            zh("開始分析", "开始分析") if lang == "zh" else "Run",
            key="run_step6_upstream_btn",
            disabled=not step6_upstream_can_run,
        )
        if not upstream_available:
            st.caption(zh("未上傳 3-1 主要參考文件時，此按鍵會反灰。", "未上传 3-1 主要参考文件时，此按键会反灰。") if lang == "zh" else "Disabled if no upstream reference was uploaded in 3-1.")
        elif upstream_done:
            st.caption(zh("已完成 upstream relevance（只能跑一次）。", "已完成 upstream relevance（只能跑一次）。") if lang == "zh" else "Upstream relevance already completed (run-once).")

    # ---- Quote relevance ----
    quote_available = bool(doc_ref_state.get("quote_ref_text"))
    # gating: if upstream was uploaded, require upstream_done first
    quote_gate_ok = (not upstream_available) or upstream_done
    step6_quote_can_run = step5_done and quote_available and quote_gate_ok

    with col_q:
        st.markdown("**" + ((zh("Run Analysis（quote relevance）", "Run Analysis（quote relevance）") if lang == "zh" else "Run analysis (quote relevance)")) + "**")
        run_quote = st.button(
            zh("開始分析", "开始分析") if lang == "zh" else "Run",
            key="run_step6_quote_btn",
            disabled=not step6_quote_can_run,
        )
        if not quote_available:
            st.caption(zh("請先在 3-2 上傳次要參考文件。", "请先在 3-2 上传次要参考文件。") if lang == "zh" else "Upload a quote reference in 3-2 first.")
        elif upstream_available and (not upstream_done):
            st.caption(zh("若已上傳 3-1，需先完成 upstream relevance 才會開放 quote relevance。", "若已上传 3-1，需先完成 upstream relevance 才会开放 quote relevance。") if lang == "zh" else "If 3-1 is uploaded, complete upstream relevance first to unlock quote relevance.")

    if run_upstream:
        with st.spinner(zh("分析中...（upstream relevance）", "分析中...（upstream relevance）") if lang == "zh" else "Analyzing... (upstream relevance)"):
            upstream_output = derive_upstream_relevance(
                lang,
                current_state.get("step5_output", ""),
                doc_ref_state.get("upstream_ref_name", "upstream_reference"),
                doc_ref_state.get("upstream_ref_text", ""),
                model_name,
            )
            current_state["upstream_done"] = True
            current_state["upstream_output"] = clean_report_text(upstream_output)
            save_state_to_disk()
        st.success(zh("upstream relevance 完成。", "upstream relevance 完成。") if lang == "zh" else "Upstream relevance completed.")

    if run_quote:
        with st.spinner(zh("分析中...（quote relevance）", "分析中...（quote relevance）") if lang == "zh" else "Analyzing... (quote relevance)"):
            quote_output = derive_quote_relevance(
                lang,
                st.session_state.last_doc_text or "",
                doc_ref_state.get("quote_ref_name", "quote_reference"),
                doc_ref_state.get("quote_ref_text", ""),
                model_name,
            )
            entry = {
                "name": doc_ref_state.get("quote_ref_name", "quote_reference"),
                "output": clean_report_text(quote_output),
                "ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            doc_ref_state.setdefault("quote_history", [])
            doc_ref_state["quote_history"].append(entry)
            st.session_state.doc_ref_state = doc_ref_state
            save_state_to_disk()
        st.success(zh("quote relevance 完成，已加入歷史紀錄。", "quote relevance 完成，已加入历史纪录。") if lang == "zh" else "Quote relevance completed and appended to history.")

    # Show Step 6 outputs
    if current_state.get("upstream_output"):
        with st.expander(zh("步驟六輸出：upstream relevance（僅一次）", "步骤六输出：upstream relevance（仅一次）") if lang == "zh" else "Step 6 output: upstream relevance (run-once)", expanded=False):
            st.markdown(current_state.get("upstream_output", ""))

    if doc_ref_state.get("quote_history"):
        with st.expander(zh("步驟六輸出：quote relevance（歷史）", "步骤六输出：quote relevance（历史）") if lang == "zh" else "Step 6 output: quote relevance (history)", expanded=False):
            for i, it in enumerate(doc_ref_state.get("quote_history", []), start=1):
                st.markdown(f"#### {i}. {it.get('name','quote_reference')}  ({it.get('ts','')})")
                st.markdown(it.get("output", ""))
                st.markdown("---")

    st.markdown("---")

    # =========================
    # Step 7: Final integration + Download + Reset Whole Document
    # =========================

    st.subheader(zh("步驟七：最終整合（正式報告）", "步骤七：最终整合（正式报告）") if lang == "zh" else "Step 7: Final integration (formal report)")
    st.caption(
        zh(
            "此步驟會用零錯誤框架整合步驟五與步驟六的結果，輸出正式報告（建議包含表格）。",
            "此步骤会用零错误框架整合步骤五与步骤六的结果，输出正式报告（建议包含表格）。",
        )
        if lang == "zh"
        else "This step re-applies the Error-Free framework and integrates Step 5 + Step 6 outputs into a formal report (preferably with tables)."
    )

    step7_can_run = bool(current_state.get("step5_done", False))
    run_step7 = st.button(
        zh("Run final analysis（final integration）", "Run final analysis（final integration）") if lang == "zh" else "Run final analysis (final integration)",
        key="run_step7_btn",
        disabled=not step7_can_run,
    )

    if run_step7:
        with st.spinner(zh("分析中...（最終整合）", "分析中...（最终整合）") if lang == "zh" else "Analyzing... (final integration)"):
            quote_hist = doc_ref_state.get("quote_history", [])
            quote_block = ""
            if quote_hist:
                chunks = []
                for i, it in enumerate(quote_hist, start=1):
                    chunks.append(f"[Quote relevance #{i}: {it.get('name','')} | {it.get('ts','')}]\n{it.get('output','')}")
                quote_block = "\n\n".join(chunks)

            final_input = "\n".join(
                [
                    "[Final Integration Context]",
                    f"- Document type: {st.session_state.document_type or '(not selected)'}",
                    "",
                    "==============================",
                    "1) Step 5: Main document framework analysis",
                    "==============================",
                    current_state.get("step5_output", ""),
                    "",
                    "==============================",
                    "2) Step 6: Upstream relevance (optional, run-once)",
                    "==============================",
                    current_state.get("upstream_output", "") if current_state.get("upstream_output") else "(not provided)",
                    "",
                    "==============================",
                    "3) Step 6: Quote relevance history (optional, multi-run)",
                    "==============================",
                    quote_block if quote_block else "(not provided)",
                    "",
                    "[Task]",
                    "Using the same Error-Free framework, produce ONE complete, formal report that integrates and de-duplicates all above findings.",
                    "Requirements:",
                    "- Include an Executive Summary and a clear issue taxonomy aligned to the framework.",
                    "- Include at least one well-structured table (e.g., Issue / Evidence / Impact / Recommendation / Priority).",
                    "- Explicitly indicate: supported conclusions, contradictions, and omissions.",
                    "- Provide an actionable remediation plan and a list of clarification questions.",
                    "Do NOT repeat the same content in multiple sections; consolidate and reference back to the table where appropriate.",
                ]
            )

            final_output = run_llm_analysis(selected_key, lang, final_input, model_name)
            current_state["analysis_output"] = clean_report_text(final_output)
            current_state["step7_done"] = True
            save_state_to_disk()
        st.success(zh("步驟七完成！已產出正式報告。", "步骤七完成！已产出正式报告。") if lang == "zh" else "Step 7 completed. Formal report generated.")

    if current_state.get("analysis_output"):
        st.markdown("### " + (zh("正式報告", "正式报告") if lang == "zh" else "Formal report"))
        st.markdown(current_state.get("analysis_output", ""))

    st.markdown("---")

    # Download (with optional Q&A history)
    if current_state.get("step7_done"):
        st.subheader((zh("下載整份報告", "下载整份报告") if lang == "zh" else "Download full report"))
        st.caption(zh("報告只包含分析與（可選）Q&A，不含原始文件。", "报告只包含分析与（可选）Q&A，不含原始文件。") if lang == "zh" else "Report includes analysis and optional Q&A history only (no original document).")

        include_qa = st.checkbox(
            zh("包含對話框歷史紀錄（Q&A）", "包含对话框历史纪录（Q&A）") if lang == "zh" else "Include Q&A history",
            value=False,
            key=f"include_qa_{selected_key}",
        )

        if is_guest and current_state.get("download_used"):
            st.error(zh("已達下載次數上限（1 次）", "已达下载次数上限（1 次）") if lang == "zh" else "Download limit reached (1 time).")
        else:
            report = build_full_report(lang, selected_key, current_state, include_qa=include_qa)
            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            with st.expander("Download"):
                fmt = st.radio(
                    zh("選擇格式", "选择格式") if lang == "zh" else "Select format",
                    ["Word (DOCX)", "PDF", "PowerPoint (PPTX)"],
                    key=f"fmt_{selected_key}",
                )

                data: bytes
                mime: str
                ext: str

                if fmt.startswith("Word"):
                    data = build_docx_bytes(report)
                    mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    ext = "docx"
                elif fmt.startswith("PDF"):
                    data = build_pdf_bytes(report)
                    mime = "application/pdf"
                    ext = "pdf"
                else:
                    data = build_pptx_bytes(report, lang)
                    mime = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                    ext = "pptx"

                if data:
                    clicked = st.download_button(
                        zh("開始下載", "开始下载") if lang == "zh" else "Download",
                        data=data,
                        file_name=f"errorfree_{selected_key}_{now_str}.{ext}",
                        mime=mime,
                        key=f"dl_{selected_key}_{ext}_{'qa' if include_qa else 'noqa'}",
                    )
                    if clicked:
                        current_state["download_used"] = True
                        save_state_to_disk()
                        record_usage(user_email, selected_key, "download")
    else:
        st.info(zh("尚未完成最終整合（步驟七）。完成後才能下載整份報告。", "尚未完成最终整合（步骤七）。完成后才能下载整份报告。") if lang == "zh" else "Final integration (Step 7) not completed yet. Complete it to enable full report download.")

    st.markdown("---")

    # Reset Whole Document (confirmation)
    st.subheader(zh("Reset Whole Document（新一輪主文件審查）", "Reset Whole Document（新一轮主文件审查）") if lang == "zh" else "Reset Whole Document (start a new review cycle)")
    if st.session_state.get("confirm_reset_whole", False):
        st.warning(
            zh(
                "溫馨提醒：您已經下載資料了嗎？我們公司不留存您上傳的資料。確認後，將清除本輪所有文件、分析與紀錄並重新開始。",
                "温馨提醒：您已经下载资料了吗？我们公司不留存您上传的资料。确认后，将清除本轮所有文件、分析与纪录并重新开始。",
            )
            if lang == "zh"
            else "Reminder: Have you downloaded your report? We do not retain your uploaded data. Confirming will clear ALL files, analyses, and history for this review cycle and start over."
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button(zh("確認清除並重新開始", "确认清除并重新开始") if lang == "zh" else "Confirm reset", key="confirm_reset_whole_btn"):
                # Clear whole cycle
                st.session_state.framework_states = {}
                st.session_state.last_doc_text = ""
                st.session_state.last_doc_name = ""
                st.session_state.document_type = None
                st.session_state.current_doc_id = None
                st.session_state.doc_cycle_locked = False
                st.session_state.doc_ref_state = {}
                st.session_state.confirm_reset_whole = False
                save_state_to_disk()
                st.rerun()
        with c2:
            if st.button(zh("取消", "取消") if lang == "zh" else "Cancel", key="cancel_reset_whole_btn"):
                st.session_state.confirm_reset_whole = False
                save_state_to_disk()
                st.rerun()
    else:
        if st.button(zh("Reset Whole Document", "Reset Whole Document") if lang == "zh" else "Reset Whole Document", key="reset_whole_btn"):
            st.session_state.confirm_reset_whole = True
            save_state_to_disk()
            st.rerun()


if __name__ == "__main__":
    main()
