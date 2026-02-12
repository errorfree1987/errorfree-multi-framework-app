import streamlit as st
import os
# ===== Error-Free® Portal-only SSO (Portal is the ONLY entry) =====
# Analyzer MUST NOT show any internal login UI when Portal SSO is enforced.
# Portal should open Analyzer with query params:
#   email=<user email>&lang=<en|zh-tw|zh-cn>&ts=<unix seconds>&token=<hmac sha256 hex>
# token = HMAC_SHA256(PORTAL_SSO_SECRET, f"{email}|{normalized_lang}|{ts}")
#
# Transitional support (staging only):
#   You may also pass: portal_token=demo-from-portal&email=...
#   Controlled by env: ALLOW_DEMO_PORTAL_TOKEN (default: false for safety)
#
# Railway Variables (minimum):
#   PORTAL_BASE_URL
#   PORTAL_SSO_SECRET
#
import hmac
import hashlib
import time

PORTAL_BASE_URL = (os.getenv("PORTAL_BASE_URL", "") or "").strip()
PORTAL_SSO_SECRET = (os.getenv("PORTAL_SSO_SECRET", "") or "").strip()
SSO_MAX_AGE_SECONDS = int(os.getenv("SSO_MAX_AGE_SECONDS", "300") or "300")  # 5 minutes default

ALLOW_DEMO_PORTAL_TOKEN = (os.getenv("ALLOW_DEMO_PORTAL_TOKEN", "") or "").lower() in ("1","true","yes","y","on")
DEMO_EXPECTED_TOKEN = "demo-from-portal"  # only used when ALLOW_DEMO_PORTAL_TOKEN=true


def _qp_get(key: str) -> str:
    # Streamlit newer API
    try:
        v = st.query_params.get(key)
        if v is None:
            return ""
        if isinstance(v, list):
            return v[0] if v else ""
        return str(v)
    except Exception:
        pass
    # Streamlit older API
    try:
        qp = st.experimental_get_query_params()
        arr = qp.get(key, [""])
        return arr[0] if arr else ""
    except Exception:
        return ""


def _qp_clear_all():
    try:
        st.query_params.clear()
        return
    except Exception:
        pass
    try:
        st.experimental_set_query_params()
    except Exception:
        pass


def _normalize_lang(raw: str) -> str:
    s = (raw or "").strip().lower()
    if s in ("zh", "zh-tw", "zh_tw", "zh-hant", "zh_hant", "tw", "traditional"):
        return "zh-tw"
    if s in ("zh-cn", "zh_cn", "zh-hans", "zh_hans", "cn", "simplified"):
        return "zh-cn"
    if s in ("en", "en-us", "en-gb", "english"):
        return "en"
    return "en"


def _apply_portal_lang(lang_raw: str):
    lang_norm = _normalize_lang(lang_raw)
    if lang_norm == "en":
        st.session_state["lang"] = "en"
    elif lang_norm == "zh-cn":
        st.session_state["lang"] = "zh"
        st.session_state["zh_variant"] = "cn"
    else:
        st.session_state["lang"] = "zh"
        st.session_state["zh_variant"] = "tw"
    st.session_state["_lang_locked"] = True


def _compute_sig(email: str, lang_norm: str, ts: str, secret: str) -> str:
    msg = f"{email}|{lang_norm}|{ts}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def _verify_portal_sso(email: str, lang_raw: str, ts: str, token: str) -> (bool, str):
    if not email:
        return False, "Missing email"
    if not ts:
        return False, "Missing ts"
    if not token:
        return False, "Missing token"

    try:
        ts_int = int(ts)
    except Exception:
        return False, "Invalid ts"

    now = int(time.time())
    age = abs(now - ts_int)
    if age > SSO_MAX_AGE_SECONDS:
        return False, f"Expired token (age {age}s)"

    if not PORTAL_SSO_SECRET:
        return False, "Missing PORTAL_SSO_SECRET"

    lang_norm = _normalize_lang(lang_raw)
    expected = _compute_sig(email=email, lang_norm=lang_norm, ts=str(ts_int), secret=PORTAL_SSO_SECRET)
    if not hmac.compare_digest(expected, token):
        return False, "Invalid token signature"

    return True, "OK"


def _render_portal_only_block(reason: str = ""):
    st.error("請從 Error-Free® Portal 進入此分析框架（Portal-only）。")
    if reason:
        st.caption(f"Reason: {reason}")
    if PORTAL_BASE_URL:
        st.link_button("回到 Portal / Back to Portal", PORTAL_BASE_URL)
    else:
        st.info("（管理員）請在 Railway Variables 設定 PORTAL_BASE_URL。")
    st.stop()


def try_portal_sso_login():
    # Only check once per session
    if st.session_state.get("_portal_sso_checked", False):
        return

    # 1) Preferred: HMAC-based SSO
    email = _qp_get("email")
    lang = _qp_get("lang")
    ts = _qp_get("ts")
    token = _qp_get("token")

    if email and token and ts:
        ok, why = _verify_portal_sso(email=email, lang_raw=lang, ts=ts, token=token)
        if not ok:
            st.session_state["_portal_sso_checked"] = True
            st.session_state["is_authenticated"] = False
            _render_portal_only_block(why)

        # success
        st.session_state["_portal_sso_checked"] = True
        st.session_state["is_authenticated"] = True
        st.session_state["user_email"] = email
        # Keep your existing role logic; Portal can optionally pass role
        role = _qp_get("role") or st.session_state.get("user_role") or "pro"
        st.session_state["user_role"] = role

        _apply_portal_lang(lang)
        # Hygiene: clear query params so token isn't left in the URL
        _qp_clear_all()
        st.rerun()

    # 2) Transitional DEMO token (staging only)
    portal_token = _qp_get("portal_token")
    if portal_token and ALLOW_DEMO_PORTAL_TOKEN and portal_token == DEMO_EXPECTED_TOKEN:
        st.session_state["_portal_sso_checked"] = True
        st.session_state["is_authenticated"] = True
        st.session_state["user_email"] = email or st.session_state.get("user_email") or "unknown"
        st.session_state["user_role"] = st.session_state.get("user_role") or "pro"
        _apply_portal_lang(lang)
        _qp_clear_all()
        st.rerun()

    # 3) No valid SSO -> block (Portal-only)
    st.session_state["_portal_sso_checked"] = True
    st.session_state["is_authenticated"] = False
    _render_portal_only_block("No valid Portal SSO parameters")

# ===== End Portal-only SSO =====
import os, json, datetime, secrets

from pathlib import Path

from typing import Dict, List, Optional

from io import BytesIO

import base64



import streamlit as st

import streamlit.components.v1 as components

import pdfplumber

from docx import Document

from openai import OpenAI

from reportlab.pdfgen import canvas

from reportlab.lib.pagesizes import letter

from reportlab.pdfbase import pdfmetrics

from reportlab.pdfbase.ttfonts import TTFont

from reportlab.pdfbase.cidfonts import UnicodeCIDFont





PDF_FONT_NAME = "Helvetica"

PDF_FONT_REGISTERED = False

PDF_TTF_PATH = os.getenv("PDF_TTF_PATH")  # Optional: path to a Unicode TTF font for PDF export





def ensure_pdf_font():
    """Register a Unicode-capable font for PDF export to avoid black boxes / garbled text."""
    global PDF_FONT_NAME, PDF_FONT_REGISTERED
    if PDF_FONT_REGISTERED:
        return

    try:
        try:
            pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
            PDF_FONT_NAME = "STSong-Light"
        except Exception:
            if PDF_TTF_PATH and Path(PDF_TTF_PATH).exists():
                pdfmetrics.registerFont(TTFont("ErrorFreeUnicode", PDF_TTF_PATH))
                PDF_FONT_NAME = "ErrorFreeUnicode"
            else:
                PDF_FONT_NAME = "Helvetica"
    except Exception:
        PDF_FONT_NAME = "Helvetica"
    finally:
        PDF_FONT_REGISTERED = True





# =========================
# Company multi-tenant support
# =========================

COMPANY_FILE = Path("companies.json")


def load_companies() -> dict:
    if not COMPANY_FILE.exists():
        return {}
    try:
        return json.loads(COMPANY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_companies(data: dict):
    try:
        COMPANY_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass





# =========================
# Accounts
# =========================

ACCOUNTS = {
    "admin@errorfree.com": {"password": "1111", "role": "admin"},
    "dr.chiu@errorfree.com": {"password": "2222", "role": "pro"},
    "test@errorfree.com": {"password": "3333", "role": "pro"},
}

GUEST_FILE = Path("guest_accounts.json")


def load_guest_accounts() -> Dict[str, Dict]:
    if not GUEST_FILE.exists():
        return {}
    try:
        return json.loads(GUEST_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_guest_accounts(data: Dict[str, Dict]):
    try:
        GUEST_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass





# =========================
# Framework definitions (external JSON)
# =========================

FRAMEWORK_FILE = Path("frameworks.json")


def load_frameworks() -> Dict[str, Dict]:
    """Load framework definitions from an external JSON file."""
    if not FRAMEWORK_FILE.exists():
        return {}
    try:
        return json.loads(FRAMEWORK_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


FRAMEWORKS: Dict[str, Dict] = load_frameworks()



# =========================
# State persistence & usage tracking (4A)
# =========================

STATE_FILE = Path("user_state.json")
DOC_TRACK_FILE = Path("user_docs.json")
USAGE_FILE = Path("usage_stats.json")  # 使用量統計


def load_doc_tracking() -> Dict[str, List[str]]:
    if not DOC_TRACK_FILE.exists():
        return {}
    try:
        return json.loads(DOC_TRACK_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_doc_tracking(data: Dict[str, List[str]]):
    try:
        DOC_TRACK_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def load_usage_stats() -> Dict[str, Dict]:
    if not USAGE_FILE.exists():
        return {}
    try:
        return json.loads(USAGE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_usage_stats(data: Dict[str, Dict]):
    try:
        USAGE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def record_usage(user_email: str, framework_key: str, kind: str):
    """
    kind: 'analysis', 'followup', 'download'
    """
    if not user_email:
        return
    data = load_usage_stats()
    user_entry = data.get(user_email, {})
    fw_map = user_entry.get("frameworks", {})
    fw_entry = fw_map.get(
        framework_key,
        {
            "analysis_runs": 0,
            "followups": 0,
            "downloads": 0,
        },
    )
    if kind == "analysis":
        fw_entry["analysis_runs"] = fw_entry.get("analysis_runs", 0) + 1
    elif kind == "followup":
        fw_entry["followups"] = fw_entry.get("followups", 0) + 1
    elif kind == "download":
        fw_entry["downloads"] = fw_entry.get("downloads", 0) + 1

    fw_map[framework_key] = fw_entry
    user_entry["frameworks"] = fw_map
    user_entry["last_used"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data[user_email] = user_entry
    save_usage_stats(data)


def save_state_to_disk():
    data = {
        "user_email": st.session_state.get("user_email"),
        "user_role": st.session_state.get("user_role"),
        "is_authenticated": st.session_state.get("is_authenticated", False),
        "lang": st.session_state.get("lang", "zh"),
        "zh_variant": st.session_state.get("zh_variant", "tw"),
        "usage_date": st.session_state.get("usage_date"),
        "usage_count": st.session_state.get("usage_count", 0),
        "last_doc_text": st.session_state.get("last_doc_text", ""),
        "last_doc_name": st.session_state.get("last_doc_name", ""),
        "document_type": st.session_state.get("document_type"),
        "framework_states": st.session_state.get("framework_states", {}),
        "selected_framework_key": st.session_state.get("selected_framework_key"),
        "current_doc_id": st.session_state.get("current_doc_id"),
        "company_code": st.session_state.get("company_code"),
        "show_admin": st.session_state.get("show_admin", False),

        # Step 3 split references (更正2)
        "upstream_reference": st.session_state.get("upstream_reference"),
        "quote_current": st.session_state.get("quote_current"),
        "quote_history": st.session_state.get("quote_history", []),
        "quote_upload_nonce": st.session_state.get("quote_upload_nonce", 0),
            "review_upload_nonce": st.session_state.get("review_upload_nonce", 0),
            "upstream_upload_nonce": st.session_state.get("upstream_upload_nonce", 0),
        "quote_upload_finalized": st.session_state.get("quote_upload_finalized", False),
        "upstream_step6_done": st.session_state.get("upstream_step6_done", False),
        "upstream_step6_output": st.session_state.get("upstream_step6_output", ""),
        "quote_step6_done_current": st.session_state.get("quote_step6_done_current", False),

        # Step 7 history (re-run when new quote refs are analyzed)
        "step7_history": st.session_state.get("step7_history", []),
        "integration_history": st.session_state.get("integration_history", []),

        # Follow-up clear flag (fix StreamlitAPIException)
        "_pending_clear_followup_key": st.session_state.get("_pending_clear_followup_key"),
    }
    try:
        STATE_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def restore_state_from_disk():
    if not STATE_FILE.exists():
        return
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return
    for k, v in data.items():
        if k not in st.session_state:
            st.session_state[k] = v





# =========================
# OpenAI client & model selection
# =========================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def resolve_model_for_user(role: str) -> str:
    if role in ["admin", "pro", "company_admin"]:
        return "gpt-5.1"
    if role == "free":
        return "gpt-4.1-mini"
    return "gpt-5.1"





# =========================
# Language helpers
# =========================

def zh(tw: str, cn: str = None) -> str:
    """Return zh text by variant when lang == 'zh'. Default variant is 'tw'."""
    if st.session_state.get("lang") != "zh":
        return tw
    if st.session_state.get("zh_variant", "tw") == "cn" and cn is not None:
        return cn
    return tw





# =========================
# File reading
# =========================

def ocr_image_to_text(file_bytes: bytes, filename: str) -> str:
    """Use OpenAI vision model to perform OCR on an image and return plain text."""
    if client is None:
        return "[Error] OPENAI_API_KEY 尚未設定，無法進行圖片 OCR。"

    fname = filename.lower()
    img_format = "png" if fname.endswith(".png") else "jpeg"

    role = st.session_state.get("user_role", "free")
    model_name = resolve_model_for_user(role)

    b64_data = base64.b64encode(file_bytes).decode("utf-8")

    lang = st.session_state.get("lang", "zh")
    if lang == "zh":
        prompt = (
            "請將這張圖片中的所有可見文字完整轉成純文字，"
            "保持原本的段落與換行。不要加上任何說明或總結，只輸出文字內容。"
        )
    else:
        prompt = (
            "Transcribe all visible text in this image into plain text. "
            "Preserve paragraphs and line breaks. Do not add any commentary or summary."
        )

    try:
        response = client.responses.create(
            model=model_name,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image": {"data": b64_data, "format": img_format}},
                    ],
                }
            ],
            max_output_tokens=2000,
        )
        text_out = response.output_text or ""
        return text_out.strip()
    except Exception as e:
        return f"[圖片 OCR 時發生錯誤: {e}]"


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
        elif name.endswith((".jpg", ".jpeg", ".png")):
            file_bytes = uploaded_file.read()
            if not file_bytes:
                return "[讀取圖片檔案時發生錯誤：空檔案]"
            return ocr_image_to_text(file_bytes, uploaded_file.name)
        else:
            return ""
    except Exception as e:
        return f"[讀取檔案時發生錯誤: {e}]"





# =========================
# Core LLM logic (keep wrapper as-is)
# =========================

def run_llm_analysis(framework_key: str, language: str, document_text: str, model_name: str) -> str:
    # Diagnostics (do not leak key)
    st.session_state["_ef_last_openai_error"] = ""
    st.session_state["_ef_last_openai_error_type"] = ""

    if framework_key not in FRAMEWORKS:
        return f"[Error] Framework '{framework_key}' not found in frameworks.json."

    fw = FRAMEWORKS[framework_key]
    system_prompt = fw["wrapper_zh"] if language == "zh" else fw["wrapper_en"]
    prefix = "以下是要分析的文件內容：\n\n" if language == "zh" else "Here is the document to analyze:\n\n"
    user_prompt = prefix + (document_text or "")

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
        msg = str(e)
        low = msg.lower()
        err_type = "unknown"
        if ("insufficient_quota" in msg) or ("error code: 429" in low) or ("quota" in low):
            err_type = "quota_or_429"
        elif ("rate_limit" in low) or ("too many requests" in low):
            err_type = "rate_limit"
        elif ("invalid_api_key" in low) or ("api key" in low and "invalid" in low):
            err_type = "invalid_key"
        elif ("timed out" in low) or ("timeout" in low) or ("connection" in low):
            err_type = "network"
        st.session_state["_ef_last_openai_error_type"] = err_type
        st.session_state["_ef_last_openai_error"] = (msg[:240] + ("..." if len(msg) > 240 else ""))
        # Do NOT leak or guess billing/quota in the main output. Store details in diagnostics instead.
        return f"[OpenAI API ERROR: {err_type or 'unknown'}]"


def _openai_simple(system_prompt: str, user_prompt: str, model_name: str, max_output_tokens: int) -> str:
    # Diagnostics (do not leak key)
    st.session_state["_ef_last_openai_error"] = ""
    st.session_state["_ef_last_openai_error_type"] = ""

    if client is None:
        return "[Error] OPENAI_API_KEY 尚未設定，無法連線至 OpenAI。"
    try:
        response = client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_output_tokens=max_output_tokens,
        )
        return (response.output_text or "").strip()
    except Exception as e:
        msg = str(e)
        low = msg.lower()
        err_type = "unknown"
        if ("insufficient_quota" in msg) or ("error code: 429" in low) or ("quota" in low):
            err_type = "quota_or_429"
        elif ("rate_limit" in low) or ("too many requests" in low):
            err_type = "rate_limit"
        elif ("invalid_api_key" in low) or ("api key" in low and "invalid" in low):
            err_type = "invalid_key"
        elif ("timed out" in low) or ("timeout" in low) or ("connection" in low):
            err_type = "network"
        st.session_state["_ef_last_openai_error_type"] = err_type
        st.session_state["_ef_last_openai_error"] = (msg[:240] + ("..." if len(msg) > 240 else ""))
        # Do NOT leak or guess billing/quota in the main output. Store details in diagnostics instead.
        return f"[OpenAI API ERROR: {err_type or 'unknown'}]"


def _chunk_text(text: str, chunk_size: int = 12000, overlap: int = 600) -> List[str]:
    """Used ONLY for reference summarization to control token size."""
    if not text:
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    n = len(text)
    chunks = []
    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


def summarize_reference_text(language: str, ref_name: str, ref_text: str, model_name: str) -> str:
    """Compress reference doc into a faithful structured summary (not framework analysis)."""
    chunks = _chunk_text(ref_text, chunk_size=12000, overlap=600)
    if not chunks:
        return ""

    if language == "zh":
        sys = "你是一個嚴謹的文件摘要助手。你的任務是忠實壓縮內容，不要發明不存在的資訊。"

        def one_chunk_prompt(i: int, total: int, c: str) -> str:
            return (
                f"請將以下參考文件內容做摘要（第 {i}/{total} 段），保留：\n"
                "1) 重要定義/範圍\n2) 關鍵要求/限制/數值\n3) 任何例外/前提\n4) 可能影響判斷的條款\n\n"
                f"【參考文件】{ref_name}\n【內容】\n{c}"
            )

        reduce_sys = "你是一個嚴謹的摘要整合助手。請合併多段摘要，去重但不漏掉關鍵要求與限制。"

        def reduce_prompt(t: str) -> str:
            return (
                "請把以下多段摘要整合為一份『參考文件總摘要』，結構化輸出：\n"
                "A. 定義/範圍\nB. 主要要求/限制\nC. 例外/前提\nD. 可能影響判斷的條款\n\n"
                f"【參考文件】{ref_name}\n【多段摘要】\n{t}"
            )

    else:
        sys = "You are a careful document summarization assistant. Summarize faithfully and do not hallucinate."

        def one_chunk_prompt(i: int, total: int, c: str) -> str:
            return (
                f"Summarize the following reference document chunk ({i}/{total}). Preserve:\n"
                "1) definitions/scope\n2) key requirements/constraints/values\n3) exceptions/prereqs\n4) clauses that affect decisions\n\n"
                f"[Reference] {ref_name}\n[Content]\n{c}"
            )

        reduce_sys = "You consolidate summaries. Merge, dedupe, keep key constraints."

        def reduce_prompt(t: str) -> str:
            return (
                "Consolidate chunk summaries into ONE reference master summary with sections:\n"
                "A. Definitions/Scope\nB. Requirements/Constraints\nC. Exceptions/Prereqs\nD. Decision-impacting clauses\n\n"
                f"[Reference] {ref_name}\n[Chunk summaries]\n{t}"
            )

    partials = []
    total = len(chunks)
    for i, c in enumerate(chunks, start=1):
        partials.append(_openai_simple(sys, one_chunk_prompt(i, total, c), model_name, max_output_tokens=900))

    current = partials[:]
    while len(current) > 1:
        nxt = []
        batch_size = 8
        for i in range(0, len(current), batch_size):
            joined = "\n\n---\n\n".join(current[i : i + batch_size])
            nxt.append(_openai_simple(reduce_sys, reduce_prompt(joined), model_name, max_output_tokens=1100))
        current = nxt

    return current[0].strip()


def clean_report_text(text: str) -> str:
    replacements = {"■": "-", "•": "-", "–": "-", "—": "-"}
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

# =========================
# Step 6: Relevance analysis (更正2)
# =========================

def is_openai_error_output(text: str) -> bool:
    """Detect our placeholder error strings from OpenAI call failures."""
    if not text:
        return False
    t = str(text).strip()
    return t.startswith("[OpenAI API") or t.startswith("[OpenAI")

def render_openai_error(language: str) -> None:
    """Render a helpful OpenAI error without polluting step outputs."""
    err_type = st.session_state.get("_ef_last_openai_error_type", "")
    err_msg = st.session_state.get("_ef_last_openai_error", "")
    if language == "en":
        st.error("OpenAI API call failed. This is not an app logic error. Please check your OpenAI Project billing/usage and model access.")
        st.caption("Tip: In OpenAI dashboard, verify the key belongs to a Project with billing enabled and usage limits > 0. Then redeploy/restart.")
    else:
        st.error("OpenAI API 呼叫失敗。這不是流程邏輯錯誤，請檢查 OpenAI 專案的 Billing/用量與模型權限。")
        st.caption("建議：到 OpenAI Dashboard 確認此 API key 所屬 Project 已開啟計費、用量上限 > 0，且有模型使用權限；之後重啟/重新部署。")
    with st.expander("Diagnostics / 詳細錯誤" if language == "en" else "Diagnostics / 詳細錯誤", expanded=False):
        st.write(f"error_type: {err_type or 'unknown'}")
        if err_msg:
            st.code(err_msg)
        else:
            st.write("(no additional error details captured)")

# ...(以下內容維持你原本 app.py 不變)
# 由於訊息長度限制，這裡我無法安全地把你整份 2000+ 行完整貼出來而不出錯。
# 我已經精準定位要改的區塊，請你用「區塊替換」方式改，最安全、也最符合你「不要亂改」的要求。
