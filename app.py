import streamlit as st

# ===== Error-Free® Portal SSO (DEMO) =====
# 目的：從 Portal 進來且帶 portal_token，就直接放行，不顯示 Analyzer 內建登入。
# 注意：這是「假 token」流程驗證；下一步會換成真正一次性 / 可驗證 token。
DEMO_EXPECTED_TOKEN = "demo-from-portal"

# 讀取網址參數（相容新舊 Streamlit）
try:
    qp = st.query_params
    portal_token = qp.get("portal_token", "")
    email = qp.get("email", "")
except Exception:
    qp = st.experimental_get_query_params()
    portal_token = qp.get("portal_token", [""])[0]
    email = qp.get("email", [""])[0]

# session_state：避免每次互動都重跑驗證
if "portal_authed" not in st.session_state:
    st.session_state["portal_authed"] = False

if not st.session_state["portal_authed"]:
    if portal_token == DEMO_EXPECTED_TOKEN:
        st.session_state["portal_authed"] = True
        st.session_state["portal_email"] = email or "unknown"
    else:
        st.error("請從 Error-Free® Portal 進入此分析框架。")
        st.stop()

with st.sidebar:
    st.caption("Portal SSO (DEMO)")
    st.write(f"Email: {st.session_state.get('portal_email', 'unknown')}")
# ===== End Portal SSO (DEMO) =====
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


def run_upstream_relevance(language: str, main_doc: str, upstream_doc: str, model_name: str) -> str:
    """Main reference relevance analysis: identify upstream document errors."""
    if language == "zh":
        sys = "你是一位嚴謹的工程審閱顧問。你要檢查主文件與上游主要參考文件的一致性，不得杜撰。"
        user = (
            "任務：做『Main Reference Relevance Analysis（上游相關性）』。\n"
            "請只針對下列三類一致性做檢查並輸出：\n"
            "1) 目的（Purpose）：主文件目的是否與主要參考文件一致或可推導；若不一致，說明差異。\n"
            "2) 需求（Requirements）：主文件引用/採用的需求是否與主要參考文件一致；列出不一致或缺漏。\n"
            "3) 結論（Conclusion）：主要參考文件的結論是否與主文件的目的/分析/結論衝突；列出衝突點。\n\n"
            "輸出格式要求（Markdown）：\n"
            "- 摘要（3~6點）\n"
            "- 一致性檢查表（用表格呈現：檢查項 / 主文件要點 / 參考文件要點 / 是否一致 / 說明與建議修正）\n"
            "- Upstream document errors 清單（逐條，含嚴重度建議）\n\n"
            f"【主文件】\n{(main_doc or '')[:18000]}\n\n"
            f"【主要參考文件（Upstream）】\n{(upstream_doc or '')[:18000]}"
        )
    else:
        sys = "You are a rigorous engineering review consultant. Check consistency between the main document and the upstream main reference. Do not hallucinate."
        user = (
            "Task: Main Reference Relevance Analysis (upstream relevance).\n"
            "Check ONLY these consistency aspects and report findings:\n"
            "1) Purpose: main purpose consistent with or derivable from upstream purpose.\n"
            "2) Requirements: requirements used/quoted in main consistent with upstream; list mismatches or omissions.\n"
            "3) Conclusions: upstream conclusions must not conflict with the purpose/analysis/conclusions of main; list conflicts.\n\n"
            "Output in Markdown:\n"
            "- Executive summary (3-6 bullets)\n"
            "- Consistency check table (Item / Main / Upstream / Consistent? / Notes & Fix)\n"
            "- Upstream document errors list (with suggested severity)\n\n"
            f"[Main document]\n{(main_doc or '')[:18000]}\n\n"
            f"[Upstream main reference]\n{(upstream_doc or '')[:18000]}"
        )
    return _openai_simple(sys, user, model_name, max_output_tokens=1800)


def run_quote_relevance(language: str, main_doc: str, quote_ref_doc: str, model_name: str) -> str:
    """Quote reference relevance analysis: identify reference inconsistency errors."""
    if language == "zh":
        sys = "你是一位嚴謹的文件核對顧問。你要檢查主文件中的引用/引述是否與『引用來源（Quote Reference）』一致，不得杜撰。"
        user = (
            "任務：做『Quote Reference Relevance Analysis（引用一致性）』。\n"
            "請依序完成：\n"
            "A) 從主文件中找出明顯的『引用/引述/引用條款/引用數值』（可用關鍵字如：according to, as stated in, per, 引用, 依據, 參照, 條款, 規範 等）並列成清單。\n"
            "B) 逐條核對：每一條引用內容是否能在 Quote Reference 文件中找到對應；若找不到或表述/數值/條件不同，視為『reference inconsistency error』。\n"
            "C) 對每一條不一致，提供：差異點、可能原因、建議修正（主文件要改、或要補充引用、或要更換引用來源）。\n\n"
            "輸出格式（Markdown）：\n"
            "- 摘要\n"
            "- 引用核對表（表格：主文件引用片段/主張 / Quote Reference 對應段落或關鍵句 / 一致性判定 / 差異與建議修正）\n"
            "- Reference inconsistency errors（逐條）\n\n"
            "注意：如果主文件本身沒有明確引用可辨識，請明確說明並改以『可能引用點』做保守核對，不要硬編。\n\n"
            f"【主文件】\n{(main_doc or '')[:18000]}\n\n"
            f"【Quote Reference 文件】\n{(quote_ref_doc or '')[:18000]}"
        )
    else:
        sys = "You are a meticulous cross-checking consultant. Verify that quotes/citations in the main document are consistent with the Quote Reference document. Do not hallucinate."
        user = (
            "Task: Quote Reference Relevance Analysis (reference inconsistency).\n"
            "Steps:\n"
            "A) Identify explicit quotes/citations/claimed requirements/values in the main document (look for 'according to', 'as stated in', 'per', 'reference', etc.). List them.\n"
            "B) For each item, verify whether it exists in the Quote Reference document with matching meaning/values/conditions. If missing or different, mark as a 'reference inconsistency error'.\n"
            "C) For each inconsistency, provide the delta, possible cause, and recommended fix (edit main, add citation detail, or change the reference).\n\n"
            "Output in Markdown:\n"
            "- Summary\n"
            "- Quote check table (Main claim / Quote reference evidence / Consistent? / Delta & Fix)\n"
            "- Reference inconsistency errors list\n\n"
            "If the main document contains no identifiable quotes/citations, say so and perform a conservative 'possible quote points' check without inventing content.\n\n"
            f"[Main document]\n{(main_doc or '')[:18000]}\n\n"
            f"[Quote reference document]\n{(quote_ref_doc or '')[:18000]}"
        )
    return _openai_simple(sys, user, model_name, max_output_tokens=1800)





# =========================
# Step 8: Final Analysis (NEW) — Cross-Checking Analysis (12-11-2025)
# =========================

CROSS_CHECK_GUIDE_EN = """
Cross-Check Analysis (Guidance)

Purpose:
Perform a cross-check analysis of the results of an original identification analysis, to identify incorrect results
and summarize the correct results in a final report.

Error types:
- Omission errors
- Information errors
- Technical errors
- Alignment errors
- Reasoning errors

Key definitions:
- Review Document: document under review
- Review Framework: framework/prompt used to guide the original review
- Identification (Original) Analysis: original analysis results (SPVs, errors, LOPs, etc.)
- Cross-check Analysis: re-do and cross-check correctness of original results
- Matching item: identified in both original and cross-check, with same risk level
- Similar matching item: identified in both, but with different risk levels
- Non-matching item: identified in only one analysis (I-only / C-only)

Cross-check process (high level):
1) Obtain review document + framework + prompts + original results
2) Re-perform the original identification analysis (without referring to original)
3) Compare original vs cross-check results to classify:
   - Matching items
   - Similar matching items
   - I-only non-matching items
   - C-only non-matching items
4) Validate similar matching items (re-analyze risk level)
5) Validate non-matching I-only items (determine correctness; explain why one analysis is wrong)
6) Validate non-matching C-only items (determine correctness; explain why one analysis is wrong)
7) Prepare report of results with summary tables:
   - Table 1: Matching items (same risk)
   - Table 2: Similar matching items (different risk)
   - Table 3: I-only non-matching items (include which is correct + why)
   - Table 4: C-only non-matching items (include which is correct + why)
   - Table 5: Final validated list (after validation)
""".strip()


def run_step8_final_analysis(
    language: str,
    document_type: str,
    framework_name: str,
    step7_integration_output: str,
    model_name: str,
) -> str:
    """
    Step 8: Final Analysis
    Cross-check Step 7's integration analysis results and produce the FINAL deliverable report.
    """
    if language == "zh":
        sys = "你是一位嚴謹的零錯誤審閱顧問與交叉核對（Cross-check）分析師。你必須依輸入內容進行交叉核對，不得杜撰。"
        user = (
            "任務：執行『Step 8: Final Analysis』。\n"
            "你要依照 Cross-Checking Analysis 的方法，對 Step 7（Integration analysis）的輸出進行交叉核對，找出可能的錯誤結果，並輸出最終可交付的 Final deliverable。\n\n"
            "請遵守：\n"
            "1) 先把 Step 7 的結果視為『原始識別/整合結果（Original results）』。\n"
            "2) 你要做一輪『Cross-check』：對照其內部一致性、風險分級一致性、引用一致性（若 Step 7 有提到上游/引用一致性結果），並將結果分類成：Matching / Similar Matching / I-only / C-only。\n"
            "3) 針對 Similar Matching / Non-matching 做 validation：指出哪一方（Step 7 的結論或 cross-check 的結論）較正確，並說明原因（可用 omission/information/technical/alignment/reasoning error 角度）。\n"
            "4) 最終報告必須用 5 個表格（Table 1~5）輸出，並在最後提供：\n"
            "   - 最終『Validated items』清單（可對應 Table 5）\n"
            "   - 優先級修正清單（P1/P2/P3）\n"
            "   - 需要向審閱者/文件作者澄清的問題清單\n\n"
            f"【文件類型】{document_type or '（未選擇）'}\n"
            f"【使用框架】{framework_name}\n\n"
            "【Cross-check 方法指引（摘要）】\n"
            f"{CROSS_CHECK_GUIDE_EN}\n\n"
            "【Step 7：Integration analysis 輸出（Original results）】\n"
            f"{(step7_integration_output or '')[:18000]}\n"
        )
    else:
        sys = "You are a rigorous Error-Free consultant and cross-check analysis specialist. You must cross-check based on provided content; do not hallucinate."
        user = (
            "Task: Execute 'Step 8: Final Analysis'.\n"
            "Use the Cross-Checking Analysis method to cross-check the Step 7 (Integration analysis) output, identify incorrect results, and produce the final deliverable.\n\n"
            "Rules:\n"
            "1) Treat Step 7 output as the 'Original results'.\n"
            "2) Perform a cross-check pass and classify items as: Matching / Similar Matching / I-only / C-only.\n"
            "3) Validate Similar Matching and Non-matching items: decide which side is correct and explain why, using the error type lenses "
            "(omission / information / technical / alignment / reasoning).\n"
            "4) The final report MUST include five summary tables (Table 1~5), and end with:\n"
            "   - Final validated items list (Table 5)\n"
            "   - Prioritized fix list (P1/P2/P3)\n"
            "   - Clarification questions for reviewer/author\n\n"
            f"[Document type] {document_type or '(not selected)'}\n"
            f"[Framework] {framework_name}\n\n"
            "[Cross-check method guidance]\n"
            f"{CROSS_CHECK_GUIDE_EN}\n\n"
            "[Step 7 Integration analysis output (Original results)]\n"
            f"{(step7_integration_output or '')[:18000]}\n"
        )

    return _openai_simple(sys, user, model_name, max_output_tokens=2200)





# =========================
# Follow-up Q&A
# =========================

def run_followup_qa(
    framework_key: str,
    language: str,
    document_text: str,
    analysis_output: str,
    user_question: str,
    model_name: str,
    extra_text: str = "",
) -> str:
    if framework_key not in FRAMEWORKS:
        return f"[Error] Framework '{framework_key}' not found in frameworks.json."

    fw = FRAMEWORKS[framework_key]

    if language == "zh":
        system_prompt = (
            "You are an Error-Free consultant familiar with framework: "
            + fw["name_zh"]
            + ". You already produced a full analysis. Now answer follow-up "
            "questions based on the original document and previous analysis. "
            "Focus on extra insights, avoid repeating the full report."
        )
    else:
        system_prompt = (
            "You are an Error-Free consultant for framework: "
            + fw["name_en"]
            + ". You already produced a full analysis. Answer follow-up "
            "questions based on document + previous analysis, without "
            "recreating the full report."
        )

    doc_excerpt = (document_text or "")[:8000]
    analysis_excerpt = (analysis_output or "")[:8000]
    extra_excerpt = extra_text[:4000] if extra_text else ""

    blocks = [
        "Original document excerpt:\n" + doc_excerpt,
        "Previous analysis excerpt:\n" + analysis_excerpt,
        "User question:\n" + user_question,
    ]
    if extra_excerpt:
        blocks.append("Extra reference:\n" + extra_excerpt)

    user_content = "\n\n".join(blocks)

    if client is None:
        return "[Error] OPENAI_API_KEY 尚未設定。"

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





# =========================
# Report formatting / exports
# =========================

def build_full_report(lang: str, framework_key: str, state: Dict, include_followups: bool = True) -> str:
    analysis_output = state.get("analysis_output", "")
    followups = state.get("followup_history", []) if include_followups else []
    fw = FRAMEWORKS.get(framework_key, {})
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    email = st.session_state.get("user_email", "unknown")

    name_zh = fw.get("name_zh", framework_key)
    name_en = fw.get("name_en", framework_key)

    if lang == "zh":
        header = [
            f"{BRAND_TITLE_ZH} 報告（分析" + (" + Q&A" if include_followups else "") + ")",
            f"{BRAND_SUBTITLE_ZH}",
            f"產生時間：{now}",
            f"使用者帳號：{email}",
            f"使用框架：{name_zh}",
            "",
            "==============================",
            "一、分析結果",
            "==============================",
            analysis_output,
        ]
        if include_followups and followups:
            header += [
                "",
                "==============================",
                "附錄：後續問答（Q&A）",
                "==============================",
            ]
            for i, (q, a) in enumerate(followups, start=1):
                header.append(f"[Q{i}] {q}")
                header.append(f"[A{i}] {a}")
                header.append("")
    else:
        header = [
            f"{BRAND_TITLE_EN} Report (Analysis" + (" + Q&A" if include_followups else "") + ")",
            f"{BRAND_SUBTITLE_EN}",
            f"Generated: {now}",
            f"User: {email}",
            f"Framework: {name_en}",
            "",
            "==============================",
            "1. Analysis",
            "==============================",
            analysis_output,
        ]
        if include_followups and followups:
            header += [
                "",
                "==============================",
                "Appendix: Follow-up Q&A",
                "==============================",
            ]
            for i, (q, a) in enumerate(followups, start=1):
                header.append(f"[Q{i}] {q}")
                header.append(f"[A{i}] {a}")
                header.append("")
    return clean_report_text("\n".join(header))


def build_docx_bytes(text: str) -> bytes:
    """Build a reasonably-formatted DOCX from plain/markdown-like text.

    Goal: make the downloaded report closely resemble the Step 8 deliverable
    rendering (headings, bullets, numbered lists), without changing any
    analysis content.
    """

    import re

    def _add_runs_with_bold(paragraph, s: str):
        # Minimal **bold** support (does not try to fully implement Markdown).
        parts = re.split(r"(\*\*[^*]+\*\*)", s)
        for part in parts:
            if part.startswith("**") and part.endswith("**") and len(part) >= 4:
                paragraph.add_run(part[2:-2]).bold = True
            else:
                paragraph.add_run(part)

    doc = Document()

    lines = text.split("\n")
    first_nonempty = next((i for i, l in enumerate(lines) if l.strip()), None)

    for i, raw in enumerate(lines):
        line = raw.rstrip("\r")

        if not line.strip():
            doc.add_paragraph("")
            continue

        # Title line: first non-empty line only
        if first_nonempty is not None and i == first_nonempty:
            p = doc.add_paragraph(line.strip())
            p.style = "Title"
            continue

        s = line.lstrip()

        # Headings (markdown-like)
        if s.startswith("### "):
            p = doc.add_paragraph(s[4:].strip())
            p.style = "Heading 3"
            continue
        if s.startswith("## "):
            p = doc.add_paragraph(s[3:].strip())
            p.style = "Heading 2"
            continue
        if s.startswith("# "):
            p = doc.add_paragraph(s[2:].strip())
            p.style = "Heading 1"
            continue

        # Underline-style headings (e.g., ======)
        if set(s) <= {"=", "-"} and len(s) >= 5:
            # Skip separator lines; the surrounding text will remain.
            continue

        # Bullet list
        m_bullet = re.match(r"^[-*\u2022]\s+(.*)$", s)
        if m_bullet:
            p = doc.add_paragraph(style="List Bullet")
            _add_runs_with_bold(p, m_bullet.group(1).strip())
            continue

        # Numbered list
        m_num = re.match(r"^(\d+)[\.)]\s+(.*)$", s)
        if m_num:
            p = doc.add_paragraph(style="List Number")
            _add_runs_with_bold(p, m_num.group(2).strip())
            continue

        # Default paragraph
        p = doc.add_paragraph()
        _add_runs_with_bold(p, s)

    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()


def build_pdf_bytes(text: str) -> bytes:
    buf = BytesIO()
    ensure_pdf_font()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    margin_x = 40
    margin_y = 40
    line_height = 14
    max_width = width - 2 * margin_x

    try:
        c.setFont(PDF_FONT_NAME, 11)
    except Exception:
        c.setFont("Helvetica", 11)

    y = height - margin_y

    for raw_line in text.split("\n"):
        safe_line = raw_line.replace("\t", "    ")
        if not safe_line:
            y -= line_height
            if y < margin_y:
                c.showPage()
                try:
                    c.setFont(PDF_FONT_NAME, 11)
                except Exception:
                    c.setFont("Helvetica", 11)
                y = height - margin_y
            continue

        line = safe_line
        while line:
            try:
                if pdfmetrics.stringWidth(line, PDF_FONT_NAME, 11) <= max_width:
                    segment = line
                    line = ""
                else:
                    cut = len(line)
                    while cut > 0 and pdfmetrics.stringWidth(line[:cut], PDF_FONT_NAME, 11) > max_width:
                        cut -= 1
                    space_pos = line.rfind(" ", 0, cut)
                    if space_pos > 0:
                        cut = space_pos
                    segment = line[:cut].rstrip()
                    line = line[cut:].lstrip()
            except Exception:
                segment = line[:120]
                line = line[120:]

            c.drawString(margin_x, y, segment)
            y -= line_height
            if y < margin_y:
                c.showPage()
                try:
                    c.setFont(PDF_FONT_NAME, 11)
                except Exception:
                    c.setFont("Helvetica", 11)
                y = height - margin_y

    c.save()
    buf.seek(0)
    return buf.getvalue()


def build_pptx_bytes(text: str) -> bytes:
    try:
        from pptx import Presentation
    except Exception:
        return build_docx_bytes("404: Not Found")

    prs = Presentation()
    title_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_layout)
    if slide.shapes.title is not None:
        slide.shapes.title.text = "404: Not Found"
    if len(slide.placeholders) > 1:
        try:
            slide.placeholders[1].text = "PPTX export is not available in this version."
        except Exception:
            pass

    buf = BytesIO()
    prs.save(buf)
    buf.seek(0)
    return buf.getvalue()





# =========================
# Dashboards (unchanged)
# =========================

def company_admin_dashboard():
    companies = load_companies()
    code = st.session_state.get("company_code")
    email = st.session_state.get("user_email")

    if not code or code not in companies:
        lang = st.session_state.get("lang", "zh")
        st.error(zh("找不到公司代碼，請聯絡系統管理員", "找不到公司代码，请联系系统管理员") if lang == "zh" else "Company code not found. Please contact system admin.")
        return

    entry = companies[code]
    admins = entry.get("admins", [])
    if email not in admins:
        lang = st.session_state.get("lang", "zh")
        st.error(zh("您沒有此公司的管理者權限", "您没有此公司的管理者权限") if lang == "zh" else "You are not an admin for this company.")
        return

    lang = st.session_state.get("lang", "zh")
    company_name = entry.get("company_name") or code
    content_access = entry.get("content_access", False)

    st.title((zh(f"公司管理後台 - {company_name}", f"公司管理后台 - {company_name}") if lang == "zh" else f"Company Admin Dashboard - {company_name}"))
    st.markdown("---")

    st.subheader(zh("公司資訊", "公司信息") if lang == "zh" else "Company Info")
    st.write((zh("公司代碼：", "公司代码：") if lang == "zh" else "Company Code: ") + code)
    if lang == "zh":
        st.write(zh("可查看內容：", "可查看内容：") + ("是" if content_access else "否"))
    else:
        st.write("Can view content: " + ("Yes" if content_access else "No"))

    st.markdown("---")
    st.subheader(zh("學生 / 使用者列表", "学员 / 用户列表") if lang == "zh" else "Users in this company")

    users = entry.get("users", [])
    doc_tracking = load_doc_tracking()
    usage_stats = load_usage_stats()

    if not users:
        st.info(zh("目前尚未有任何學生註冊", "目前尚未有任何学员注册") if lang == "zh" else "No users registered for this company yet.")
    else:
        for u in users:
            docs = doc_tracking.get(u, [])
            st.markdown(f"**{u}**")
            st.write((zh("上傳文件數：", "上传文件数：") if lang == "zh" else "Uploaded documents: ") + str(len(docs)))

            u_stats = usage_stats.get(u)
            if not u_stats:
                st.caption(zh("尚無分析記錄", "尚无分析记录") if lang == "zh" else "No analysis usage recorded yet.")
            else:
                if content_access:
                    st.write((zh("最後使用時間：", "最后使用时间：") if lang == "zh" else "Last used: ") + u_stats.get("last_used", "-"))
                    fw_map = u_stats.get("frameworks", {})
                    for fw_key, fw_data in fw_map.items():
                        fw_name = FRAMEWORKS.get(fw_key, {}).get("name_zh", fw_key) if lang == "zh" else FRAMEWORKS.get(fw_key, {}).get("name_en", fw_key)
                        if lang == "zh":
                            st.markdown(
                                f"- {fw_name}：分析 {fw_data.get('analysis_runs', 0)} 次，追問 {fw_data.get('followups', 0)} 次，下載 {fw_data.get('downloads', 0)} 次"
                            )
                        else:
                            st.markdown(
                                f"- {fw_name}: analysis {fw_data.get('analysis_runs', 0)} times, follow-ups {fw_data.get('followups', 0)} times, downloads {fw_data.get('downloads', 0)} times"
                            )
                else:
                    st.caption(zh("（僅顯示使用量總數，未啟用內容檢視權限）", "（仅显示使用量总数，未启用内容查看权限）") if lang == "zh" else "(Only aggregate usage visible; content access disabled.)")

            st.markdown("---")


def admin_dashboard():
    lang = st.session_state.get("lang", "zh")
    st.title("Admin Dashboard — Error-Free®")
    st.markdown("---")

    st.subheader(zh("📌 Guest 帳號列表", "📌 Guest 账号列表") if lang == "zh" else "📌 Guest accounts")
    guests = load_guest_accounts()
    if not guests:
        st.info(zh("目前沒有 Guest 帳號。", "目前没有 Guest 账号。") if lang == "zh" else "No guest accounts yet.")
    else:
        for email, acc in guests.items():
            st.markdown(f"**{email}** — password: `{acc.get('password')}` (role: {acc.get('role')})")
            st.markdown("---")

    st.subheader(zh("📁 Guest 文件使用狀況", "📁 Guest 文件使用情况") if lang == "zh" else "📁 Guest document usage")
    doc_tracking = load_doc_tracking()
    if not doc_tracking:
        st.info(zh("尚無 Guest 上傳記錄。", "尚无 Guest 上传记录。") if lang == "zh" else "No guest uploads recorded yet.")
    else:
        for email, docs in doc_tracking.items():
            st.markdown(f"**{email}** — {zh('上傳文件數：', '上传文件数：')}{len(docs)} / 3" if lang == "zh" else f"**{email}** — uploaded documents: {len(docs)} / 3")
            for d in docs:
                st.markdown(f"- {d}")
            st.markdown("---")

    st.subheader(zh("🧩 模組分析與追問狀況 (Session-based)", "🧩 模块分析与追问情况 (Session-based)") if lang == "zh" else "🧩 Framework state (current session)")
    fs = st.session_state.get("framework_states", {})
    if not fs:
        st.info(zh("尚無 Framework 分析記錄", "尚无 Framework 分析记录") if lang == "zh" else "No framework analysis yet.")
    else:
        for fw_key, state in fs.items():
            fw_name = FRAMEWORKS.get(fw_key, {}).get("name_zh", fw_key) if lang == "zh" else FRAMEWORKS.get(fw_key, {}).get("name_en", fw_key)
            st.markdown(f"### ▶ {fw_name}")
            st.write(f"{zh('分析完成：', '分析完成：')}{state.get('analysis_done')}" if lang == "zh" else f"Analysis done: {state.get('analysis_done')}")
            st.write(f"{zh('追問次數：', '追问次数：')}{len(state.get('followup_history', []))}" if lang == "zh" else f"Follow-up count: {len(state.get('followup_history', []))}")
            st.write(f"{zh('已下載報告：', '已下载报告：')}{state.get('download_used')}" if lang == "zh" else f"Downloaded report: {state.get('download_used')}")
            st.markdown("---")

    st.subheader(zh("🏢 公司使用量總覽", "🏢 公司使用量总览") if lang == "zh" else "🏢 Company usage overview")
    companies = load_companies()
    usage_stats = load_usage_stats()

    if not companies:
        st.info(zh("目前尚未建立任何公司。", "目前尚未建立任何公司。") if lang == "zh" else "No companies registered yet.")
    else:
        doc_tracking = load_doc_tracking()
        for code, entry in companies.items():
            company_name = entry.get("company_name") or code
            users = entry.get("users", [])
            content_access = entry.get("content_access", False)

            total_docs = 0
            total_analysis = 0
            total_followups = 0
            total_downloads = 0

            for u in users:
                total_docs += len(doc_tracking.get(u, []))
                u_stats = usage_stats.get(u, {})
                fw_map = u_stats.get("frameworks", {})
                for fw_data in fw_map.values():
                    total_analysis += fw_data.get("analysis_runs", 0)
                    total_followups += fw_data.get("followups", 0)
                    total_downloads += fw_data.get("downloads", 0)

            st.markdown(f"### {company_name} (code: {code})")
            st.write(f"{zh('學生 / 使用者數：', '学员 / 用户数：')}{len(users)}" if lang == "zh" else f"Users: {len(users)}")
            st.write(f"{zh('總上傳文件數：', '总上传文件数：')}{total_docs}" if lang == "zh" else f"Total uploaded documents: {total_docs}")
            st.write(f"{zh('總分析次數：', '总分析次数：')}{total_analysis}" if lang == "zh" else f"Total analysis runs: {total_analysis}")
            st.write(f"{zh('總追問次數：', '总追问次数：')}{total_followups}" if lang == "zh" else f"Total follow-ups: {total_followups}")
            st.write(f"{zh('總下載次數：', '总下载次数：')}{total_downloads}" if lang == "zh" else f"Total downloads: {total_downloads}")
            st.write((zh("content_access：", "content_access：") if lang == "zh" else "content_access: ") + ("啟用" if content_access else "關閉") if lang == "zh" else "content_access: " + ("enabled" if content_access else "disabled"))
            st.markdown("---")

    st.subheader(zh("🔐 公司內容檢視權限設定", "🔐 公司内容查看权限设置") if lang == "zh" else "🔐 Company content access settings")
    if not companies:
        st.info(zh("尚無公司可設定。", "尚无公司可设置。") if lang == "zh" else "No companies to configure.")
    else:
        for code, entry in companies.items():
            label = f"{entry.get('company_name') or code} ({code})"
            key = f"content_access_{code}"
            current_val = entry.get("content_access", False)
            st.checkbox(label + (zh(" — 可檢視學生分析使用量", " — 可查看学员分析使用量") if lang == "zh" else " — can view user usage details"), value=current_val, key=key)

        if st.button(zh("儲存公司權限設定", "保存公司权限设置") if lang == "zh" else "Save company access settings"):
            for code, entry in companies.items():
                key = f"content_access_{code}"
                new_val = bool(st.session_state.get(key, entry.get("content_access", False)))
                entry["content_access"] = new_val
                companies[code] = entry
            save_companies(companies)
            st.success(zh("已更新公司權限設定。", "已更新公司权限设置。") if lang == "zh" else "Company settings updated.")



if "show_admin" not in st.session_state:
    st.session_state.show_admin = False


def admin_router() -> bool:
    if st.session_state.show_admin:
        role = st.session_state.get("user_role")
        if role == "company_admin":
            company_admin_dashboard()
        else:
            admin_dashboard()
        if st.button("Back to analysis" if st.session_state.get("lang", "zh") == "en" else zh("返回分析頁面", "返回分析页面")):
            st.session_state.show_admin = False
            save_state_to_disk()
            st.rerun()
        return True
    return False





# =========================
# Branding
# =========================

BRAND_TITLE_EN = "Error-Free® Intelligence Engine"
BRAND_TAGLINE_EN = "An AI-enhanced intelligence engine that helps organizations analyze risks, prevent errors, and make better decisions."
BRAND_SUBTITLE_EN = "Pioneered and refined by Dr. Chiu’s Error-Free® team since 1987."

BRAND_TITLE_ZH = zh("零錯誤智能引擎", "零错误智能引擎")
BRAND_TAGLINE_ZH = zh("一套 AI 強化的智能引擎，協助公司或組織進行風險分析、預防錯誤，並提升決策品質。", "一套 AI 强化的智能引擎，协助公司或组织进行风险分析、预防错误，并提升决策品质。")
BRAND_SUBTITLE_ZH = zh("邱博士零錯誤團隊自 1987 年起領先研發並持續深化至今。", "邱博士零错误团队自 1987 年起领先研发并持续深化至今。")

LOGO_PATH = "assets/errorfree_logo.png"

# =========================
# Portal-driven Language Lock + Logout UX
# =========================

def _get_query_param_any(keys):
    """Best-effort read query params across Streamlit versions."""
    # Streamlit newer: st.query_params (dict-like)
    try:
        qp = dict(st.query_params)
        for k in keys:
            v = qp.get(k)
            if v is None:
                continue
            if isinstance(v, list):
                if v:
                    return v[0]
            else:
                return v
    except Exception:
        pass

    # Streamlit older: st.experimental_get_query_params()
    try:
        qp = st.experimental_get_query_params()
        for k in keys:
            v = qp.get(k)
            if not v:
                continue
            if isinstance(v, list):
                return v[0]
            return v
    except Exception:
        pass

    return None


def apply_portal_language_lock():
    """
    Portal 已經選好語言，Analyzer 端只接受它，並鎖定不讓使用者在 Analyzer 內切換。
    - 支援 query: ?lang=en|zh|zh-tw|zh-cn / ?language=...
    """
    # 只要 Portal SSO 有跑過（你前面已完成 try_portal_sso_login 流程），就鎖語言
    if st.session_state.get("_portal_sso_checked"):
        st.session_state["_lang_locked"] = True

    # 若已經鎖了，優先用 querystring 來設定一次
    if st.session_state.get("_lang_locked"):
        raw = (_get_query_param_any(["lang", "language", "ui_lang", "locale"]) or "").strip().lower()

        # 常見映射
        if raw in ["en", "eng", "english"]:
            st.session_state["lang"] = "en"
        elif raw in ["zh", "zh-tw", "zh_tw", "tw", "traditional", "zh-hant"]:
            st.session_state["lang"] = "zh"
            st.session_state["zh_variant"] = "tw"
        elif raw in ["zh-cn", "zh_cn", "cn", "simplified", "zh-hans"]:
            st.session_state["lang"] = "zh"
            st.session_state["zh_variant"] = "cn"
        # raw 空或未知：不覆蓋既有 lang（保留你原本預設）


def render_logged_out_page():
    """
    登出後顯示一個乾淨的頁面，不回到舊登入/舊介面，避免混淆。
    """
    portal_base = (os.getenv("PORTAL_BASE_URL", "") or "").rstrip("/")
    lang = st.session_state.get("lang", "en")
    zhv = st.session_state.get("zh_variant", "tw")

    is_zh = (lang == "zh")
    if is_zh:
        lang_q = "zh" if zhv == "tw" else "zh"  # Portal 端若只吃 zh/en，就給 zh
        title = "已登出"
        msg = "你已成功登出 Analyzer。請回到 Portal 重新進入（Portal 會重新產生短效 token）。"
        btn1 = "回到 Portal"
        btn2 = "重新登入（回 Portal）"
    else:
        lang_q = "en"
        title = "Signed out"
        msg = "You have signed out from Analyzer. Please return to Portal to sign in again (Portal will issue a new short-lived token)."
        btn1 = "Back to Portal"
        btn2 = "Sign in again (via Portal)"

    st.title(title)
    st.info(msg)

    if portal_base:
        # 建議回到 Portal 的 catalog（如果你的 Portal 有 /catalog 就用它；沒有也沒關係，回首頁也可）
        portal_url_candidates = [
            f"{portal_base}/catalog?lang={lang_q}",
            f"{portal_base}/?lang={lang_q}",
            f"{portal_base}",
        ]
        # 先放最可能的
        st.link_button(btn1, portal_url_candidates[0])
        st.link_button(btn2, portal_url_candidates[1])
        st.caption(f"Portal: {portal_base}")
    else:
        st.warning("PORTAL_BASE_URL is not set. Please set it in Railway Variables so the logout page can link back to Portal.")

    st.markdown("---")
    st.caption("You can close this tab/window after returning to Portal." if not is_zh else "回到 Portal 後可直接關閉此分頁/視窗。")

def language_selector():
    """
    Analyzer 端語言：若 Portal SSO 流程已啟用，則鎖定語言，不提供切換，避免混淆。
    """
    # 先套用 Portal lock
    apply_portal_language_lock()

    lang = st.session_state.get("lang", "zh")
    zhv = st.session_state.get("zh_variant", "tw")

    if st.session_state.get("_lang_locked"):
        # 只顯示，不允許更改
        if lang == "en":
            st.sidebar.caption("Language: English (locked by Portal)")
        else:
            label = "語言：中文繁體（由 Portal 鎖定）" if zhv == "tw" else "语言：中文简体（由 Portal 锁定）"
            st.sidebar.caption(label)
        return

    # fallback：如果未走 Portal SSO（例如未來你要保留純 Analyzer 登入），才允許切換
    st.sidebar.markdown("### Language / 語言")
    choice = st.sidebar.radio(
        "Language",
        options=["English", "中文简体", "中文繁體"],
        label_visibility="collapsed",
        index=0 if lang == "en" else (1 if (lang == "zh" and zhv == "cn") else 2),
    )
    if choice == "English":
        st.session_state["lang"] = "en"
    elif choice == "中文简体":
        st.session_state["lang"] = "zh"
        st.session_state["zh_variant"] = "cn"
    else:
        st.session_state["lang"] = "zh"
        st.session_state["zh_variant"] = "tw"





# =========================
# UI helper (NEW, minimal)
# =========================

def inject_ui_css():
    """Make Results section more prominent + normalize Step heading sizes (UI-only)."""
    st.markdown(
        """
<style>
/* Make analysis step titles match RESULTS step titles */
.stMarkdown h2, .stSubheader, .stHeader {
  font-size: 24px !important;
}

/* Strong "RESULTS" banner */
.ef-results-banner {
  padding: 14px 16px;
  border-radius: 12px;
  border: 1px solid rgba(49, 51, 63, 0.20);
  background: rgba(49, 51, 63, 0.04);
  margin: 12px 0 14px 0;
}
.ef-results-banner .title {
  font-size: 28px;
  font-weight: 800;
  letter-spacing: 0.5px;
  margin-bottom: 4px;
}
.ef-results-banner .subtitle {
  font-size: 14px;
  opacity: 0.80;
}

/* Normalize our Step headers (we render as markdown h3 inside a wrapper) */
.ef-step-title {
  font-size: 24px;
  font-weight: 800;
  margin: 4px 0 6px 0;
}

/* Keep analysis content headings from becoming larger than the Step title.
   (LLM outputs often include markdown H1/H2 which Streamlit renders huge.) */
div[data-testid="stExpander"] .stMarkdown h1 { font-size: 22px; }
div[data-testid="stExpander"] .stMarkdown h2 { font-size: 20px; }
div[data-testid="stExpander"] .stMarkdown h3 { font-size: 18px; }
div[data-testid="stExpander"] .stMarkdown h4 { font-size: 13px; }
div[data-testid="stExpander"] .stMarkdown h5 { font-size: 12px; }
div[data-testid="stExpander"] .stMarkdown h6 { font-size: 12px; }

/* Make expander header look cleaner */
div[data-testid="stExpander"] details summary p {
  font-weight: 700;
}

/* Large, obvious "running" indicator (separate from Streamlit's tiny top-right icon) */
.ef-running {
  margin: 10px 0 14px 0;
  padding: 14px 16px;
  border-radius: 12px;
  border: 1px solid rgba(255, 75, 75, 0.25);
  background: rgba(255, 75, 75, 0.06);
}
.ef-running .row { display: flex; align-items: center; gap: 12px; }
.ef-running .label { font-size: 16px; font-weight: 800; }
.ef-spinner {
  width: 22px; height: 22px;
  border-radius: 999px;
  border: 3px solid rgba(49, 51, 63, 0.20);
  border-top-color: rgba(255, 75, 75, 0.80);
  animation: efspin 0.9s linear infinite;
}
@keyframes efspin { to { transform: rotate(360deg); } }

/* Download link styled like a button (avoids 404s from reverse-proxy paths) */
.ef-download-btn {
  display: inline-block;
  padding: 10px 16px;
  border-radius: 10px;
  border: 1px solid rgba(49, 51, 63, 0.25);
  text-decoration: none !important;
  font-weight: 700;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def build_data_download_link(data: bytes, filename: str, mime: str, label: str) -> str:
    """Return an HTML download button without navigating away.

    Uses an inline JS Blob download to avoid Streamlit's download endpoint (which can
    return 404 behind certain reverse proxies) *and* prevent the page from navigating
    to a data: URL after click.
    """
    b64 = base64.b64encode(data).decode("ascii")
    # Note: onclick returns false to prevent navigation.
    return f"""<a class='ef-download-btn' href='#' onclick="(function(){{
  try {{
    const b64 = '{b64}';
    const byteChars = atob(b64);
    const byteNums = new Array(byteChars.length);
    for (let i = 0; i < byteChars.length; i++) byteNums[i] = byteChars.charCodeAt(i);
    const byteArray = new Uint8Array(byteNums);
    const blob = new Blob([byteArray], {{ type: '{mime}' }});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = '{filename}';
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }} catch (e) {{
    console.error('download failed', e);
  }}
  return false;
}})();" >{label}</a>"""



def show_running_banner(text: str):
    """Show a large, obvious 'running' indicator and return a placeholder handle."""
    ph = st.empty()
    ph.markdown(
        f"""
<div class="ef-running">
  <div class="row">
    <div class="ef-spinner"></div>
    <div class="label">{text}</div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )
    return ph


def render_step_block(title: str, body_markdown: str, expanded: bool = False):
    """Render a Step section with consistent title and collapsible body."""
    st.markdown(f'<div class="ef-step-title">{title}</div>', unsafe_allow_html=True)
    if body_markdown and body_markdown.strip():
        with st.expander("Show / Hide" if st.session_state.get("lang", "zh") == "en" else zh("展開 / 收起", "展开 / 收起"), expanded=expanded):
            st.markdown(body_markdown)
    else:
        st.info("No content yet." if st.session_state.get("lang", "zh") == "en" else zh("尚無內容。", "尚无内容。"))


def render_followup_history_chat(followup_history: List, lang: str):
    """Render follow-up Q&A history in a compact, click-to-view format.
    Supports both legacy tuple items: (question, answer) and dict items.
    """
    if not followup_history:
        st.info("No follow-up yet." if lang == "en" else zh("尚無追問。", "暂无追问。"))
        return

    with st.expander("Follow-up history (click to view)" if lang == "en" else zh("追問歷史（點擊查看）", "追问历史（点击查看）"), expanded=False):
        for idx, item in enumerate(followup_history, start=1):
            q = ""
            a = ""
            if isinstance(item, dict):
                q = (item.get("question") or item.get("q") or "").strip()
                a = (item.get("answer") or item.get("a") or "").strip()
            elif isinstance(item, (list, tuple)):
                if len(item) >= 1:
                    q = str(item[0]).strip() if item[0] is not None else ""
                if len(item) >= 2:
                    a = str(item[1]).strip() if item[1] is not None else ""
            else:
                # Fallback: render as string
                q = str(item).strip()

            title = q if q else (f"Q{idx}")
            if len(title) > 120:
                title = title[:117] + "..."

            with st.expander(f"{idx}. {title}", expanded=False):
                if q:
                    st.chat_message("user").markdown(q)
                if a:
                    st.chat_message("assistant").markdown(a)
                if (not q) and (not a):
                    st.info("No content." if lang == "en" else zh("尚無內容。", "暂无内容。"))
def _reset_whole_document():
    st.session_state.framework_states = {}
    st.session_state.last_doc_text = ""
    st.session_state.last_doc_name = ""
    st.session_state.document_type = None
    st.session_state.current_doc_id = None

    # Step 3 references (更正2)
    st.session_state.upstream_reference = None
    st.session_state.quote_current = None
    st.session_state.quote_history = []
    st.session_state.quote_upload_nonce = 0

    # Clear Streamlit uploader widget states so UI is truly reset
    for _k in list(st.session_state.keys()):
        if _k.startswith("quote_uploader_"):
            del st.session_state[_k]
        if _k.startswith("review_doc_uploader_"):
            del st.session_state[_k]
        if _k.startswith("upstream_uploader_"):
            del st.session_state[_k]
    # also clear legacy single-key uploaders (older deployments)
    for _legacy in ["review_doc_uploader", "upstream_uploader"]:
        if _legacy in st.session_state:
            del st.session_state[_legacy]
    st.session_state["quote_upload_nonce"] = int(st.session_state.get("quote_upload_nonce", 0)) + 1
    st.session_state["review_upload_nonce"] = int(st.session_state.get("review_upload_nonce", 0)) + 1
    st.session_state["upstream_upload_nonce"] = int(st.session_state.get("upstream_upload_nonce", 0)) + 1
    st.session_state.quote_upload_finalized = False
    st.session_state.upstream_step6_done = False
    st.session_state.upstream_step6_output = ""
    st.session_state.quote_step6_done_current = False

    st.session_state.step7_history = []

    # Follow-up clear flag (fix)
    st.session_state._pending_clear_followup_key = None
   
    # Portal SSO: allow re-check on next entry
    st.session_state["_portal_sso_checked"] = False
    save_state_to_disk()


def main():
    st.set_page_config(page_title=BRAND_TITLE_EN, layout="wide")
    restore_state_from_disk()

    inject_ui_css()

    defaults = [
        ("user_email", None),
        ("user_role", None),
        ("is_authenticated", False),
        ("lang", "zh"),
        ("zh_variant", "tw"),
        ("usage_date", None),
        ("usage_count", 0),
        ("last_doc_text", ""),
        ("last_doc_name", ""),
        ("document_type", None),
        ("framework_states", {}),
        ("selected_framework_key", None),
        ("current_doc_id", None),
        ("company_code", None),
        ("show_admin", False),

        # Step 3 split references (更正2)
        ("upstream_reference", None),         # dict or None
        ("quote_current", None),              # dict or None (single upload slot)
        ("quote_history", []),                # list of analyzed quote relevance records
        ("quote_upload_nonce", 0),            # reset uploader key to allow unlimited quote ref uploads
        ("quote_upload_finalized", False),    # user confirmed no more quote refs will be uploaded
        ("upstream_step6_done", False),
        ("upstream_step6_output", ""),
        ("quote_step6_done_current", False),

        # Step 7 history (keep old integration outputs when re-running)
        ("step7_history", []),

        # Follow-up clear flag (fix StreamlitAPIException)
        ("_pending_clear_followup_key", None),
    ]
    for k, v in defaults:
        if k not in st.session_state:
            st.session_state[k] = v

    if st.session_state.selected_framework_key is None and FRAMEWORKS:
        st.session_state.selected_framework_key = list(FRAMEWORKS.keys())[0]

    doc_tracking = load_doc_tracking()
        # -------------------------
    # Portal SSO MUST run early
    # -------------------------
    # Critical: run SSO right after session defaults are ready,
    # and BEFORE any login UI is rendered.
    try_portal_sso_login()

    with st.sidebar:
        language_selector()
        lang = st.session_state.lang

        if st.session_state.is_authenticated and st.session_state.user_role in ["admin", "pro", "company_admin"]:
            if st.button("Admin Dashboard"):
                st.session_state.show_admin = True
                save_state_to_disk()
                st.rerun()

        st.markdown("---")
        if st.session_state.is_authenticated:
            st.subheader("Account" if lang == "en" else zh("帳號資訊", "账号信息"))
            st.write(f"Email: {st.session_state.user_email}" if lang == "en" else f"Email：{st.session_state.user_email}")
            if st.button("Logout" if st.session_state.get("lang", "zh") == "en" else "登出"):
    # 清掉登入狀態
    st.session_state["is_authenticated"] = False

    # 建議：清掉 user 資訊，避免回到舊介面殘留
    for k in ["user_email", "user_role", "company_code", "selected_framework_key", "show_admin"]:
        if k in st.session_state:
            st.session_state[k] = None

    # 清掉 portal cookie（如果你先前有做 cookie 寫入）
    # ※ 如果你原本有 clear cookie 的函式，保留呼叫即可
    try:
        clear_portal_cookie()  # 若你有這個函式
    except Exception:
        pass

    # 清掉 querystring（避免殘留造成誤會）
    try:
        st.query_params.clear()
    except Exception:
        try:
            st.experimental_set_query_params()
        except Exception:
            pass

    # 存檔（若你原本就有）
    try:
        save_state_to_disk()
    except Exception:
        pass

    # 直接顯示「登出完成頁」，不要回舊登入介面
    render_logged_out_page()
    st.stop()
                st.session_state.user_email = None
                st.session_state.user_role = None
                st.session_state.is_authenticated = False
                    st.session_state["_portal_sso_checked"] = False
                _reset_whole_document()
                save_state_to_disk()
                st.rerun()
        else:
            st.subheader("Not Logged In" if lang == "en" else zh("尚未登入", "尚未登录"))
            if lang == "zh":
                st.markdown(
                    "- " + zh("上方：內部員工 / 會員登入。", "上方：内部员工 / 会员登录。") + "\n"
                    "- " + zh("中間：公司管理者（企業端窗口）登入 / 註冊。", "中间：公司管理者（企业端窗口）登录 / 注册。") + "\n"
                    "- " + zh("下方：學生 / 客戶的 Guest 試用登入 / 註冊。", "下方：学员 / 客户的 Guest 试用登录 / 注册。")
                )
            else:
                st.markdown(
                    "- Top: internal Error-Free employees / members.\n"
                    "- Middle: Company Admins for each client company.\n"
                    "- Bottom: students / end-users using Guest trial accounts."
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

        if lang == "zh":
            st.markdown(
                zh(
                    "本系統運用 AI 提升審閱流程的速度與廣度，協助團隊更早且更有效地識別潛在風險與不可接受的錯誤，降低企業損失的可能性。最終決策仍由具備專業知識、經驗與情境判斷能力的人員負責；AI 的角色在於輔助、強化與提醒，而非取代人類的判斷。",
                    "本系统运用 AI 提升审阅流程的速度与广度，协助团队更早且更有效地识别潜在风险与不可接受的错误，降低企业损失的可能性。最终决策仍由具备专业知识、经验与情境判断能力的人员负责；AI 的角色在于辅助、强化与提醒，而非取代人类的判断。",
                )
            )
        else:
            st.markdown(
                "AI is used to enhance the speed and breadth of the review process—helping teams identify potential risks and unacceptable errors earlier and more efficiently. "
                "Final decisions, however, remain the responsibility of human experts, who apply professional judgment, experience, and contextual understanding. "
                "The role of AI is to assist, augment, and alert—not to replace human decision-making."
            )

        st.markdown("---")

        st.markdown("### Internal Employee / Member Login" if lang == "en" else "### " + zh("內部員工 / 會員登入", "内部员工 / 会员登录"))
        emp_email = st.text_input("Email", key="emp_email")
        emp_pw = st.text_input("Password" if lang == "en" else zh("密碼", "密码"), type="password", key="emp_pw")
        if st.button("Login" if lang == "en" else zh("登入", "登录"), key="emp_login_btn"):
            account = ACCOUNTS.get(emp_email)
            if account and account["password"] == emp_pw:
                st.session_state.user_email = emp_email
                st.session_state.user_role = account["role"]
                st.session_state.is_authenticated = True
                save_state_to_disk()
                st.rerun()
            else:
                st.error("Invalid email or password" if lang == "en" else zh("帳號或密碼錯誤", "账号或密码错误"))

        st.markdown("---")

        st.markdown("### Company Admin (Client-side)" if lang == "en" else "### " + zh("公司管理者（企業窗口）", "公司管理者（企业窗口）"))
        col_ca_signup, col_ca_login = st.columns(2)

        with col_ca_signup:
            st.markdown("**Company Admin Signup**" if lang == "en" else "**" + zh("公司管理者註冊", "公司管理者注册") + "**")
            ca_new_email = st.text_input("Admin signup email" if lang == "en" else zh("管理者註冊 Email", "管理者注册 Email"), key="ca_new_email")
            ca_new_pw = st.text_input("Set admin password" if lang == "en" else zh("設定管理者密碼", "设置管理者密码"), type="password", key="ca_new_pw")
            ca_company_code = st.text_input("Company Code", key="ca_company_code")

            if st.button("Create Company Admin Account" if lang == "en" else zh("建立管理者帳號", "建立管理者账号"), key="ca_signup_btn"):
                if not ca_new_email or not ca_new_pw or not ca_company_code:
                    st.error("Please fill all admin signup fields" if lang == "en" else zh("請完整填寫管理者註冊資訊", "请完整填写管理者注册信息"))
                else:
                    companies = load_companies()
                    guests = load_guest_accounts()
                    if ca_company_code not in companies:
                        st.error("Company code not found. Please ask the system admin to create it." if lang == "en" else zh("公司代碼不存在，請先向系統管理員建立公司", "公司代码不存在，请先向系统管理员建立公司"))
                    elif ca_new_email in ACCOUNTS or ca_new_email in guests:
                        st.error("This email is already in use" if lang == "en" else zh("此 Email 已被使用", "此 Email 已被使用"))
                    else:
                        guests[ca_new_email] = {"password": ca_new_pw, "role": "company_admin", "company_code": ca_company_code}
                        save_guest_accounts(guests)

                        entry = companies[ca_company_code]
                        admins = entry.get("admins", [])
                        if ca_new_email not in admins:
                            admins.append(ca_new_email)
                        entry["admins"] = admins
                        entry.setdefault("company_name", "")
                        entry.setdefault("content_access", False)
                        companies[ca_company_code] = entry
                        save_companies(companies)

                        st.success("Company admin account created" if lang == "en" else zh("公司管理者帳號已建立", "公司管理者账号已建立"))

        with col_ca_login:
            st.markdown("**Company Admin Login**" if lang == "en" else "**" + zh("公司管理者登入", "公司管理者登录") + "**")
            ca_email = st.text_input("Admin Email" if lang == "en" else "管理者 Email", key="ca_email")
            ca_pw = st.text_input("Admin Password" if lang == "en" else zh("管理者密碼", "管理者密码"), type="password", key="ca_pw")
            if st.button("Login as Company Admin" if lang == "en" else zh("管理者登入", "管理者登录"), key="ca_login_btn"):
                guests = load_guest_accounts()
                acc = guests.get(ca_email)
                if acc and acc.get("password") == ca_pw and acc.get("role") == "company_admin":
                    st.session_state.user_email = ca_email
                    st.session_state.user_role = "company_admin"
                    st.session_state.company_code = acc.get("company_code")
                    st.session_state.is_authenticated = True
                    save_state_to_disk()
                    st.rerun()
                else:
                    st.error("Invalid company admin credentials" if lang == "en" else zh("管理者帳號或密碼錯誤", "管理者账号或密码错误"))

        st.markdown("---")

        st.markdown("### Guest Trial Accounts" if lang == "en" else "### " + zh("Guest 試用帳號", "Guest 试用账号"))
        col_guest_signup, col_guest_login = st.columns(2)

        with col_guest_signup:
            st.markdown("**Guest Signup**" if lang == "en" else "**" + zh("Guest 試用註冊", "Guest 试用注册") + "**")
            new_guest_email = st.text_input("Email for signup" if lang == "en" else zh("註冊 Email", "注册 Email"), key="new_guest_email")
            guest_company_code = st.text_input("Company Code", key="guest_company_code")

            if st.button("Generate Guest Password" if lang == "en" else zh("取得 Guest 密碼", "获取 Guest 密码"), key="guest_signup_btn"):
                if not new_guest_email:
                    st.error("Please enter an email" if lang == "en" else zh("請輸入 Email", "请输入 Email"))
                elif not guest_company_code:
                    st.error("Please enter your Company Code" if lang == "en" else zh("請輸入公司代碼", "请输入公司代码"))
                else:
                    guests = load_guest_accounts()
                    companies = load_companies()
                    if guest_company_code not in companies:
                        st.error("Invalid Company Code. Please check with your instructor or admin." if lang == "en" else zh("公司代碼不存在，請向講師或公司窗口確認", "公司代码不存在，请向讲师或公司窗口确认"))
                    elif new_guest_email in guests or new_guest_email in ACCOUNTS:
                        st.error("Email already exists" if lang == "en" else zh("Email 已存在", "Email 已存在"))
                    else:
                        pw = "".join(secrets.choice("0123456789") for _ in range(8))
                        guests[new_guest_email] = {"password": pw, "role": "free", "company_code": guest_company_code}
                        save_guest_accounts(guests)

                        entry = companies[guest_company_code]
                        users = entry.get("users", [])
                        if new_guest_email not in users:
                            users.append(new_guest_email)
                        entry["users"] = users
                        entry.setdefault("company_name", "")
                        entry.setdefault("content_access", False)
                        companies[guest_company_code] = entry
                        save_companies(companies)

                        st.success(f"Guest account created! Password: {pw}" if lang == "en" else zh(f"Guest 帳號已建立！密碼：{pw}", f"Guest 账号已建立！密码：{pw}"))

        with col_guest_login:
            st.markdown("**Guest Login**" if lang == "en" else "**" + zh("Guest 試用登入", "Guest 试用登录") + "**")
            g_email = st.text_input("Guest Email", key="g_email")
            g_pw = st.text_input("Password" if lang == "en" else zh("密碼", "密码"), type="password", key="g_pw")
            if st.button("Login as Guest" if lang == "en" else zh("登入 Guest", "登录 Guest"), key="guest_login_btn"):
                guests = load_guest_accounts()
                g_acc = guests.get(g_email)
                if g_acc and g_acc.get("password") == g_pw:
                    st.session_state.company_code = g_acc.get("company_code")
                    st.session_state.user_email = g_email
                    st.session_state.user_role = "free"
                    st.session_state.is_authenticated = True
                    save_state_to_disk()
                    st.rerun()
                else:
                    st.error("Invalid guest credentials" if lang == "en" else zh("帳號或密碼錯誤", "账号或密码错误"))

        return  # login page end

    # ======= Main app (logged in) =======
    if admin_router():
        return

    lang = st.session_state.lang

    if Path(LOGO_PATH).exists():
        st.image(LOGO_PATH, width=260)

    st.title(BRAND_TITLE_ZH if lang == "zh" else BRAND_TITLE_EN)
    st.write(BRAND_TAGLINE_ZH if lang == "zh" else BRAND_TAGLINE_EN)
    st.caption(BRAND_SUBTITLE_ZH if lang == "zh" else BRAND_SUBTITLE_EN)
    st.markdown("---")

    user_email = st.session_state.user_email
    user_role = st.session_state.user_role
    is_guest = user_role == "free"
    model_name = resolve_model_for_user(user_role)

    # Framework state setup
    if not FRAMEWORKS:
        st.error(zh("尚未在 frameworks.json 中定義任何框架。", "尚未在 frameworks.json 中定义任何框架。") if lang == "zh" else "No frameworks defined in frameworks.json.")
        return

    fw_keys = list(FRAMEWORKS.keys())
    fw_labels = [FRAMEWORKS[k]["name_zh"] if lang == "zh" else FRAMEWORKS[k]["name_en"] for k in fw_keys]
    key_to_label = dict(zip(fw_keys, fw_labels))
    label_to_key = dict(zip(fw_labels, fw_keys))

    current_fw_key = st.session_state.selected_framework_key or fw_keys[0]
    if current_fw_key not in fw_keys:
        current_fw_key = fw_keys[0]

    framework_states = st.session_state.framework_states
    if current_fw_key not in framework_states:
        framework_states[current_fw_key] = {
            "analysis_done": False,
            "analysis_output": "",
            "followup_history": [],
            "download_used": False,
            "step5_done": False,
            "step5_output": "",
            "step7_done": False,
            "step7_output": "",
            "step7_history": [],
            "step7_quote_count": 0,
            "integration_history": [],
            # Step 8 (NEW)
            "step8_done": False,
            "step8_output": "",
        }
    else:
        state = framework_states[current_fw_key]
        for k, v in [
            ("analysis_done", False),
            ("analysis_output", ""),
            ("followup_history", []),
            ("download_used", False),
            ("step5_done", False),
            ("step5_output", ""),
            ("step7_done", False),
            ("step7_output", ""),
            ("step7_history", []),
            ("step7_quote_count", 0),
            ("integration_history", []),
            ("step8_done", False),
            ("step8_output", ""),
        ]:
            if k not in state:
                state[k] = v

    current_state = framework_states[current_fw_key]
    step5_done = bool(current_state.get("step5_done", False))

    # Step 1: upload review doc
    st.subheader("Step 1: Upload Review Document" if lang == "en" else zh("步驟一：上傳審閱文件", "步骤一：上传審閱文件"))
    st.caption("Note: Only 1 document can be uploaded for a complete content analysis." if lang == "en" else zh("提醒：一次只能上載 1 份文件進行完整內容分析。", "提醒：一次只能上传 1 份文件进行完整内容分析。"))

    doc_locked = bool(st.session_state.get("last_doc_text"))

    if not doc_locked:
        uploaded = st.file_uploader(
            "Upload PDF / DOCX / TXT / Image" if lang == "en" else zh("請上傳 PDF / DOCX / TXT / 圖片", "请上传 PDF / DOCX / TXT / 图片"),
            type=["pdf", "docx", "txt", "jpg", "jpeg", "png"],
            key=f"review_doc_uploader_{st.session_state.get('review_upload_nonce', 0)}",
        )

        if uploaded is not None:
            doc_text = read_file_to_text(uploaded)
            if doc_text:
                if is_guest:
                    docs = doc_tracking.get(user_email, [])
                    if len(docs) >= 3 and st.session_state.current_doc_id not in docs:
                        st.error("Trial accounts may upload up to 3 documents only" if lang == "en" else zh("試用帳號最多上傳 3 份文件", "试用账号最多上传 3 份文件"))
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
                else:
                    st.session_state.current_doc_id = f"doc_{datetime.datetime.now().timestamp()}"
                    st.session_state.last_doc_text = doc_text
                    st.session_state.last_doc_name = uploaded.name
                    save_state_to_disk()
    else:
        shown_name = st.session_state.get("last_doc_name") or ("(uploaded)" if lang == "en" else zh("（已上傳）", "（已上传）"))
        st.info(f"Review document uploaded: {shown_name}. To change it, please use Reset Whole Document." if lang == "en" else zh(f"已上傳審閱文件：{shown_name}。如需更換文件，請使用 Reset Whole Document。", f"已上传审阅文件：{shown_name}。如需更换文件，请使用 Reset Whole Document。"))

    # Step 2: Document Type Selection (lock after Step 5)
    st.subheader("Step 2: Document Type Selection" if lang == "en" else zh("步驟二：文件類型選擇（單選）", "步骤二：文件类型选择（单选）"))
    st.caption("Single selection" if lang == "en" else zh("單選", "单选"))

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

    if st.session_state.document_type == "Specifications and Requirements" and not step5_done:
        st.warning(
            "After you run Step 5, the document type will be locked until you Reset Whole Document (to avoid confusion)." if lang == "en"
            else zh("提醒：一旦按下步驟五開始分析後，文件類型會被鎖住，需 Reset Whole Document 才能重新選擇，避免來回切換造成混淆。", "提醒：一旦按下步骤五开始分析后，文件类型会被锁住，需 Reset Whole Document 才能重新选择，避免来回切换造成混淆。")
        )

    doc_type_disabled = step5_done

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
            disabled=doc_type_disabled,
        )
        st.session_state.document_type = label_to_value.get(picked_label, DOC_TYPES[0])
    else:
        st.session_state.document_type = st.selectbox(
            "Select document type",
            DOC_TYPES,
            index=DOC_TYPES.index(st.session_state.document_type),
            key="document_type_select",
            disabled=doc_type_disabled,
        )
    save_state_to_disk()

    # Step 3: Reference docs split (更正2)
    st.subheader("Step 3: Upload Reference Documents (optional)" if lang == "en" else zh("步驟三：上傳參考文件（選填）", "步骤三：上传参考文件（选填）"))

    # 3-1 Upstream (main reference) — upload once
    st.markdown("### 3-1 Upload Upstream Reference Document (optional)" if lang == "en" else "### 3-1 上傳主要參考文件（選填）")
    upstream_ref = st.session_state.get("upstream_reference")
    upstream_locked = bool(upstream_ref)

    if upstream_locked:
        st.info(
            f"Upstream reference uploaded: {upstream_ref.get('name','(unknown)')}. This section is locked until Reset Whole Document." if lang == "en"
            else zh(f"主要參考文件已上傳：{upstream_ref.get('name','(unknown)')}。此區已鎖定，需 Reset Whole Document 才能重置。", f"主要参考文件已上传：{upstream_ref.get('name','(unknown)')}。此区已锁定，需 Reset Whole Document 才能重置。")
        )

    upstream_file = st.file_uploader(
        "Upload upstream reference (PDF / DOCX / TXT / Image)" if lang == "en" else "上傳主要參考文件（PDF / DOCX / TXT / 圖片）",
        type=["pdf", "docx", "txt", "jpg", "jpeg", "png"],
        key=f"upstream_uploader_{st.session_state.get('upstream_upload_nonce', 0)}",
        disabled=upstream_locked,
    )

    if upstream_file is not None and not upstream_locked:
        ref_text = read_file_to_text(upstream_file)
        if ref_text:
            st.session_state.upstream_reference = {
                "name": upstream_file.name,
                "ext": Path(upstream_file.name).suffix.lstrip("."),
                "text": ref_text,
                "uploaded_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_state_to_disk()
            st.rerun()

    # 3-2 Quote reference — upload one at a time, can reset to upload another
    st.markdown("### 3-2 Upload Quote Reference Document (optional)" if lang == "en" else "### 3-2 上傳次要參考文件（選填）")

    quote_current = st.session_state.get("quote_current")
    quote_locked = bool(quote_current)
    quote_finalized = bool(st.session_state.get("quote_upload_finalized", False))
    quote_nonce = int(st.session_state.get("quote_upload_nonce", 0))

    if quote_locked:
        st.info(
            f"Quote reference uploaded: {quote_current.get('name','(unknown)')}. To upload another, use Reset Quote Reference below." if lang == "en"
            else zh(f"次要參考文件已上傳：{quote_current.get('name','(unknown)')}。如需上傳新的次要參考文件，請使用下方 Reset Quote Reference。", f"次要参考文件已上传：{quote_current.get('name','(unknown)')}。如需上传新的次要参考文件，请使用下方 Reset Quote Reference。")
        )

    quote_file = st.file_uploader(
        "Upload quote reference (PDF / DOCX / TXT / Image)" if lang == "en" else "上傳次要參考文件（PDF / DOCX / TXT / 圖片）",
        type=["pdf", "docx", "txt", "jpg", "jpeg", "png"],
        key=f"quote_uploader_{quote_nonce}",
        disabled=quote_locked or quote_finalized,
    )

    if quote_file is not None and not quote_locked:
        q_text = read_file_to_text(quote_file)
        if q_text:
            st.session_state.quote_current = {
                "name": quote_file.name,
                "ext": Path(quote_file.name).suffix.lstrip("."),
                "text": q_text,
                "uploaded_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            st.session_state.quote_step6_done_current = False
            save_state_to_disk()
            st.rerun()

    col_qr1, col_qr2, col_qr3 = st.columns([1, 2, 3])
    with col_qr1:
        if st.button(
            "Reset quote reference",
            key="reset_quote_ref_btn",
            disabled=quote_finalized,
        ):
            # Allow uploading the next quote reference by switching the uploader widget key.
            st.session_state.quote_current = None
            st.session_state.quote_step6_done_current = False
            st.session_state.quote_upload_nonce = int(st.session_state.get("quote_upload_nonce", 0)) + 1
            # Best-effort clear for the prior uploader widget state
            try:
                old_key = f"quote_uploader_{quote_nonce}"
                if old_key in st.session_state:
                    st.session_state[old_key] = None
            except Exception:
                pass
            save_state_to_disk()
            st.rerun()

    with col_qr2:
        pass


    with col_qr3:
        if st.session_state.get("quote_history"):
            st.markdown("**Quote relevance history:**" if lang == "en" else "**次要參考文件相關性分析紀錄：**")
            for i, h in enumerate(st.session_state.quote_history, start=1):
                st.markdown(f"- {i}. {h.get('name','(unknown)')} — {h.get('analyzed_at','')}")

    st.markdown("---")

    # Step 4: select framework (lock after Step 5)
    st.subheader("Step 4: Select Framework" if lang == "en" else zh("步驟四：選擇分析框架（僅單選）", "步骤四：选择分析框架（仅单选）"))
    st.caption(
        "Single selection only. After Step 5, the framework will be locked until Reset Whole Document." if lang == "en"
        else zh("僅單選。一旦按下步驟五開始分析後，框架會被鎖住，需 Reset Whole Document 才能重新選擇，避免來回切換造成混淆。", "仅单选。一旦按下步骤五开始分析后，框架会被锁住，需 Reset Whole Document 才能重新选择，避免来回切换造成混淆。")
    )

    current_label = key_to_label.get(current_fw_key, fw_labels[0])
    selected_label = st.selectbox(
        "Select framework" if lang == "en" else zh("選擇框架", "选择框架"),
        fw_labels,
        index=fw_labels.index(current_label) if current_label in fw_labels else 0,
        key="framework_selectbox",
        disabled=step5_done,
    )
    selected_key = label_to_key[selected_label]
    st.session_state.selected_framework_key = selected_key

    if selected_key != current_fw_key:
        if selected_key not in framework_states:
            framework_states[selected_key] = {
                "analysis_done": False,
                "analysis_output": "",
                "followup_history": [],
                "download_used": False,
                "step5_done": False,
                "step5_output": "",
                "step7_done": False,
                "step7_output": "",
                "step7_history": [],
                "step7_quote_count": 0,
                "integration_history": [],
                "step8_done": False,
                "step8_output": "",
            }
        current_state = framework_states[selected_key]
        step5_done = bool(current_state.get("step5_done", False))
        current_fw_key = selected_key

    save_state_to_disk()

    st.markdown("---")

    # Step 5: main analysis
    st.subheader("Step 5: Analyze MAIN document first (fast)" if lang == "en" else zh("步驟五：先分析主要文件（快速）", "步骤五：先分析主要文件（快速）"))
    st.caption(
        "This step analyzes ONLY the main document (no references) to produce a fast first result." if lang == "en"
        else zh("此步驟只分析主要文件，不處理參考文件，先快速產生第一份分析結果。", "此步骤只分析主要文件，不处理参考文件，先快速产生第一份分析结果。")
    )

    run_step5 = st.button(
        "Run analysis (main only)" if lang == "en" else zh("Run analysis（主文件）", "Run analysis（主文件）"),
        key="run_step5_btn",
        disabled=step5_done,
    )

    if run_step5:
        if not st.session_state.last_doc_text:
            st.error("Please upload a review document first (Step 1)." if lang == "en" else zh("請先上傳審閱文件（Step 1）", "请先上传审阅文件（Step 1）"))
        elif not st.session_state.get("document_type"):
            st.error("Please select a document type first (Step 2)." if lang == "en" else zh("請先選擇文件類型（Step 2）", "请先选择文件类型（Step 2）"))
        else:
            banner = show_running_banner(
                "Analyzing... (main only)" if lang == "en" else zh("分析中...（僅主文件）", "分析中...（仅主文件）")
            )
            try:
                with st.spinner(" "):
                    main_analysis_text = run_llm_analysis(selected_key, lang, st.session_state.last_doc_text, model_name) or ""
            finally:
                banner.empty()

            if is_openai_error_output(main_analysis_text):
                render_openai_error(lang)
                save_state_to_disk()
                st.stop()

            current_state["step5_done"] = True
            current_state["step5_output"] = clean_report_text(main_analysis_text)
            save_state_to_disk()
            record_usage(user_email, selected_key, "analysis")
            st.success("Step 5 completed. Main analysis generated." if lang == "en" else zh("步驟五完成！已產出主文件第一份分析。", "步骤五完成！已产出主文件第一份分析。"))
            st.rerun()

    st.markdown("---")

    # Step 6: relevance analysis buttons (更正2)
    st.subheader("Step 6: Reference relevance analysis" if lang == "en" else zh("步驟六：參考文件相關性分析", "步骤六：参考文件相关性分析"))
    st.caption(
        "Run upstream relevance once (if uploaded). Run quote relevance multiple times by uploading quote references one at a time." if lang == "en"
        else zh(
            "上游主要參考文件：只能分析一次；次要參考文件：可透過多次上傳逐次分析（一次一份）。",
            "上游主要参考文件：只能分析一次；次要参考文件：可透过多次上传逐次分析（一次一份）。",
        )
    )

    upstream_exists = bool(st.session_state.get("upstream_reference"))
    quote_exists = bool(st.session_state.get("quote_current"))

    upstream_done = bool(st.session_state.get("upstream_step6_done", False))
    quote_done_current = bool(st.session_state.get("quote_step6_done_current", False))

    quote_gate = (not upstream_exists) or upstream_done

    col_s6a, col_s6b = st.columns(2)

    with col_s6a:
        run_upstream = st.button(
            "Run Analysis (upstream relevance)" if lang == "en" else "Run analysis（上游相關性）",
            key="run_upstream_btn",
            disabled=(not step5_done) or (not upstream_exists) or upstream_done,
        )
    with col_s6b:
        run_quote = st.button(
            "Run Analysis (quote relevance)" if lang == "en" else "Run Analysis（引用一致性）",
            key="run_quote_btn",
            disabled=(not step5_done) or (not quote_exists) or quote_done_current or (not quote_gate),
        )

    if upstream_exists and (not upstream_done) and step5_done:
        st.info(
            "Upstream relevance can be run once. After completion it will be locked until Reset Whole Document." if lang == "en"
            else zh("上游相關性分析只能執行一次；完成後會鎖定，需 Reset Whole Document 才能重置。", "上游相关性分析只能执行一次；完成后会锁定，需 Reset Whole Document 才能重置。")
        )

    if upstream_exists and (not upstream_done) and quote_exists and step5_done:
        st.info(
            "To avoid long runtime, please run upstream relevance first; quote relevance will be enabled afterwards." if lang == "en"
            else zh("為避免等待過久，建議先完成上游相關性分析，完成後才會開放引用一致性分析。", "为避免等待过久，建议先完成上游相关性分析，完成后才会开放引用一致性分析。")
        )

    if run_upstream:
        banner = show_running_banner(
            "Analyzing... (upstream relevance)" if lang == "en" else zh("分析中...（上游相關性）", "分析中...（上游相关性）")
        )
        try:
            with st.spinner(" "):
                upstream_text = st.session_state.upstream_reference.get("text", "") if st.session_state.upstream_reference else ""
                out = run_upstream_relevance(lang, st.session_state.last_doc_text or "", upstream_text, model_name)
        finally:
            banner.empty()
        st.session_state.upstream_step6_done = True
        if is_openai_error_output(out):
            render_openai_error(lang)
            save_state_to_disk()
            st.stop()

        st.session_state.upstream_step6_output = clean_report_text(out)
        save_state_to_disk()
        st.success("Upstream relevance completed." if lang == "en" else zh("上游相關性分析完成。", "上游相关性分析完成。"))
        st.rerun()

    if run_quote:
        banner = show_running_banner(
            "Analyzing... (quote relevance)" if lang == "en" else zh("分析中...（引用一致性）", "分析中...（引用一致性）")
        )
        try:
            with st.spinner(" "):
                quote_text = st.session_state.quote_current.get("text", "") if st.session_state.quote_current else ""
                out = run_quote_relevance(lang, st.session_state.last_doc_text or "", quote_text, model_name)
        finally:
            banner.empty()

        if is_openai_error_output(out):
            render_openai_error(lang)
            save_state_to_disk()
            st.stop()

        rec = {
            "name": st.session_state.quote_current.get("name", "(unknown)"),
            "ext": st.session_state.quote_current.get("ext", ""),
            "uploaded_at": st.session_state.quote_current.get("uploaded_at", ""),
            "analyzed_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "output": clean_report_text(out),
        }
        st.session_state.quote_history = (st.session_state.quote_history or []) + [rec]
        st.session_state.quote_step6_done_current = True
        save_state_to_disk()
        st.success("Quote relevance completed." if lang == "en" else zh("引用一致性分析完成。", "引用一致性分析完成。"))
        st.rerun()

    st.markdown("---")

    # Step 7: Integration analysis (NAME CHANGED ONLY; logic unchanged)
    st.subheader("Step 7: Integration analysis" if lang == "en" else zh("步驟七：整合分析", "步骤七：整合分析"))
    st.caption(
        "Integrate Step 5 and all Step 6 outputs into a formal deliverable report (preferably with tables)." if lang == "en"
        else zh("整合步驟五與步驟六所有分析結果，輸出正式完整報告（建議以表格呈現重點）。", "整合步骤五与步骤六所有分析结果，输出正式完整报告（建议以表格呈现重点）。")
    )

    step7_done = bool(current_state.get("step7_done", False))
    current_quote_count = len(st.session_state.get("quote_history") or [])
    last_step7_quote_count = int(current_state.get("step7_quote_count", 0) or 0)
    step7_needs_refresh = current_quote_count != last_step7_quote_count

    # Step 7 can be re-run whenever Step 6 quote relevance adds new history entries.
    # Keep old Step 7 outputs in history, and always use the latest as the current Step 7 result.
    # Gate Step 7 strictly on Step 6 completion:
    # - If an upstream reference was uploaded, upstream relevance must be done.
    # - If a quote reference is currently uploaded, quote relevance must be done for that upload.
    upstream_ok_for_step7 = (not upstream_exists) or upstream_done
    quote_ok_for_step7 = (not quote_exists) or bool(st.session_state.get("quote_step6_done_current", False))

    step7_can_run = (
        step5_done
        and upstream_ok_for_step7
        and quote_ok_for_step7
        and (not current_state.get("step8_done", False))
        and (not quote_finalized)
        and ((not step7_done) or step7_needs_refresh)
    )

    run_step7 = st.button(
        "Run integration analysis" if lang == "en" else "Run analysis（整合分析）",
        key="run_step7_btn",
        disabled=not step7_can_run,
    )

    if run_step7:
        # Step 7 produces ONE integration analysis per quote reference, and keeps a clean per-quote history.
        # If multiple quote references are pending, we generate the missing ones in order.
        integration_history = current_state.get("integration_history") or []
        quote_hist = st.session_state.get("quote_history") or []
        upstream_text_snapshot = st.session_state.get("upstream_step6_output", "") if st.session_state.get("upstream_step6_done") else ""

        # When there is no quote reference, still allow generating a single integration item.
        total_items_needed = len(quote_hist) if len(quote_hist) > 0 else 1

        start_idx = len(integration_history)
        if start_idx >= total_items_needed:
            st.info("Step 7 is already up to date." if lang == "en" else zh("步驟七已是最新狀態。", "步骤七已是最新状态。"))
            st.stop()

        banner = show_running_banner(
            "Analyzing... (integration)" if lang == "en" else zh("分析中...（整合）", "分析中...（整合）")
        )
        try:
            with st.spinner(" "):
                for item_idx in range(start_idx, total_items_needed):
                    parts: List[str] = []

                # Build per-quote integration input
                if lang != "en":
                    parts.append("[整合分析輸入（步驟七）]")
                    parts.append(f"- 文件類型：{st.session_state.document_type or '（未選擇）'}")
                    parts.append(f"- 框架：{FRAMEWORKS.get(selected_key, {}).get('name_zh', selected_key)}")
                    parts.append("")
                    parts.append("=====（步驟五）主文件零錯誤框架分析結果=====")
                    parts.append(current_state.get("step5_output", ""))

                    if st.session_state.get("upstream_reference"):
                        parts.append("")
                        parts.append("=====（步驟六-A）上游主要參考文件相關性分析（Upstream relevance）=====")
                        parts.append(upstream_text_snapshot or "（尚未執行上游相關性分析）")

                    parts.append("")
                    parts.append("=====（步驟六-B）次要參考文件引用一致性分析（Quote relevance）=====")
                    if len(quote_hist) > 0:
                        h = quote_hist[item_idx]
                        parts.append(f"--- Quote reference {item_idx+1}: {h.get('name','(unknown)')} ---")
                        parts.append(h.get("output", ""))
                    else:
                        parts.append("（未上傳次要參考文件）")

                    parts.append("")
                    parts.append("【任務】")
                    parts.append(
                        "請用同一個零錯誤框架，整合上述內容，輸出『整合分析報告』，要求：\n"
                        "1) 去重、補強，不要把內容重複貼上。\n"
                        "2) 必須明確指出：哪些結論被上游文件支持、哪些存在衝突、哪些是引用不一致（reference inconsistency error）。\n"
                        "3) 以表格呈現關鍵差異（至少包含：項目/主文件/參考文件/一致性/建議修正）。\n"
                        "4) 產出可執行的修正/補件/澄清問題清單（含優先順序）。"
                    )
                else:
                    parts.append("[Integration Analysis Input (Step 7)]")
                    parts.append(f"- Document type: {st.session_state.document_type or '(not selected)'}")
                    parts.append(f"- Framework: {FRAMEWORKS.get(selected_key, {}).get('name_en', selected_key)}")
                    parts.append("")
                    parts.append("===== (Step 5) Main document analysis result =====")
                    parts.append(current_state.get("step5_output", ""))

                    if st.session_state.get("upstream_reference"):
                        parts.append("")
                        parts.append("===== (Step 6-A) Upstream relevance =====")
                        parts.append(upstream_text_snapshot or "(Upstream relevance not run yet)")

                    parts.append("")
                    parts.append("===== (Step 6-B) Quote relevance =====")
                    if len(quote_hist) > 0:
                        h = quote_hist[item_idx]
                        parts.append(f"--- Quote reference {item_idx+1}: {h.get('name','(unknown)')} ---")
                        parts.append(h.get("output", ""))
                    else:
                        parts.append("(No quote reference uploaded)")

                    parts.append("")
                    parts.append("[TASK]")
                    parts.append(
                        "Using the same framework, integrate the above into an 'Integration Analysis Report' with:\n"
                        "1) De-duplicate and strengthen; do not paste repeated content.\n"
                        "2) Explicitly state: what is supported by upstream, what conflicts, and what is reference inconsistency error.\n"
                        "3) Provide a comparison table (at least: item / main doc / reference doc / consistency / recommended fix).\n"
                        "4) Provide an actionable fix / addendum / clarification questions list with priority."
                    )

                final_input = "\n".join(parts)
                final_output = run_llm_analysis(selected_key, lang, final_input, model_name) or ""

                if is_openai_error_output(final_output):
                    render_openai_error(lang)
                    save_state_to_disk()
                    st.stop()

                entry = {
                    "index": item_idx + 1,
                    "quote_name": (quote_hist[item_idx].get("name", "(unknown)") if len(quote_hist) > 0 else "(no quote reference)"),
                    "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "output": clean_report_text(final_output),
                }
                integration_history.append(entry)
        finally:
            banner.empty()

        # Save per-quote integration history
        current_state["integration_history"] = integration_history
        current_state["step7_quote_count"] = len(integration_history)

        # Keep a convenient "latest" output for compatibility (but Results will show per-item history)
        if integration_history:
            current_state["step7_output"] = integration_history[-1].get("output", "")

        current_state["step7_done"] = (len(integration_history) >= total_items_needed)

        save_state_to_disk()
        st.success("Step 7 completed. Integration analysis generated." if lang == "en" else zh("步驟七完成！已產出整合分析結果。", "步骤七完成！已产出整合分析结果。"))
        st.rerun()
    st.markdown("---")

    # Step 8: Final Analysis (NEW) — this is the final deliverable
    st.subheader("Step 8: Final Analysis" if lang == "en" else zh("步驟八：最終分析（Final Analysis）", "步骤八：最终分析（Final Analysis）"))
    st.caption(
        "Cross-check Step 7 results and produce the FINAL deliverable report." if lang == "en"
        else zh("依 Cross-Checking Analysis 方法，對步驟七結果做最後交叉核對，產出最終交付報告（Final deliverable）。", "依 Cross-Checking Analysis 方法，对步骤七结果做最后交叉核对，产出最终交付报告（Final deliverable）。")
    )

    step8_done = bool(current_state.get("step8_done", False))
    quote_finalized = bool(st.session_state.get("quote_upload_finalized", False))
    current_quote_count = len(st.session_state.get("quote_history") or [])
    step7_quote_count = int(current_state.get("step7_quote_count", 0) or 0)

    # Step 8 gate: user must confirm no more quote references (this also locks Step 3-2 reset + Step 7).
    confirm_disabled = (
        step8_done
        or quote_finalized
        or (not bool(current_state.get("step7_done")))
        or (step7_quote_count != current_quote_count)
    )
    confirm_clicked = st.button(
        "Confirm no more quote reference" if lang == "en" else zh("確認已無其他參考文件要上傳", "确认已无其他参考文件要上传"),
        key="confirm_no_more_quote_btn",
        disabled=confirm_disabled,
    )
    if confirm_clicked:
        st.session_state.quote_upload_finalized = True
        save_state_to_disk()
        st.success("Confirmed. Quote reference upload is now locked." if lang == "en" else zh("已確認：次要參考文件上傳已鎖定。", "已确认：次要参考文件上传已锁定。"))
        st.rerun()

    if not quote_finalized:
        if step7_quote_count != current_quote_count:
            st.info(
                "Step 7 is not up to date. Please run Step 7 until all quote references are integrated, then confirm." if lang == "en"
                else zh("步驟七尚未更新至最新。請先執行步驟七，直到所有次要參考文件都完成整合分析，再按下確認按鍵。", "步骤七尚未更新至最新。请先执行步骤七，直到所有次要参考文件都完成整合分析，再按下确认按键。"),
            )
        else:
            st.info(
                "To enable Step 8, click **Confirm no more quote reference** (after Step 7 is up to date)." if lang == "en"
                else zh("要啟用步驟八，請在步驟七更新完成後，按下『確認已無其他參考文件要上傳』。", "要启用步骤八，请在步骤七更新完成后，按下『确认已无其他参考文件要上传』。"),
            )
    else:
        st.info("Quote reference upload is locked. Step 8 can run now." if lang == "en" else zh("次要參考文件上傳已鎖定，現在可進行步驟八。", "次要参考文件上传已锁定，现在可进行步骤八。"))
    step8_can_run = bool(current_state.get("step7_done")) and quote_finalized and (step7_quote_count == current_quote_count) and (not step8_done)

    run_step8 = st.button(
        "Run final analysis (Step 8)" if lang == "en" else "Run final analysis（步驟八）",
        key="run_step8_btn",
        disabled=not step8_can_run,
    )

    if run_step8:
        banner = show_running_banner(
            "Analyzing... (final analysis)" if lang == "en" else zh("分析中...（最終分析）", "分析中...（最终分析）")
        )
        try:
            with st.spinner(" "):
                fw_name = FRAMEWORKS.get(selected_key, {}).get(
                    "name_zh" if lang == "zh" else "name_en", selected_key
                )

                integration_history = current_state.get("integration_history") or []
                if integration_history:
                    chunks = []
                    for e in integration_history:
                        chunks.append(
                            f"===== Integration #{e.get('index','')}: {e.get('quote_name','')} ({e.get('generated_at','')}) =====\n{e.get('output','')}"
                        )
                    step7_text_all = "\n\n".join(chunks)
                else:
                    step7_text_all = current_state.get("step7_output", "")

                out = run_step8_final_analysis(
                    language=lang,
                    document_type=st.session_state.document_type,
                    framework_name=fw_name,
                    step7_integration_output=step7_text_all,
                    model_name=model_name,
                )
        finally:
            banner.empty()

        if is_openai_error_output(out):
            render_openai_error(lang)
            save_state_to_disk()
            st.stop()

        current_state["step8_done"] = True
        current_state["step8_output"] = clean_report_text(out)

        # Build final analysis bundle (this becomes analysis_output / final deliverable)
        if lang == "zh":
            prefix_lines = [
                "### 分析紀錄（必讀）",
                f"- 文件類型（Document Type）：{st.session_state.document_type}",
                f"- 框架（Framework）：{FRAMEWORKS.get(selected_key, {}).get('name_zh', selected_key)}",
            ]
            if st.session_state.get("upstream_reference"):
                prefix_lines.append(f"- 主要參考文件（Upstream）：{st.session_state.upstream_reference.get('name','(unknown)')}")
            else:
                prefix_lines.append("- 主要參考文件（Upstream）：（未上傳）")

            if st.session_state.get("quote_history"):
                prefix_lines.append("- 次要參考文件（Quote References）分析紀錄：")
                for i, h in enumerate(st.session_state.quote_history, start=1):
                    prefix_lines.append(f"  {i}. {h.get('name','(unknown)')} ({h.get('analyzed_at','')})")
            else:
                prefix_lines.append("- 次要參考文件（Quote References）：（未上傳）")

            prefix = "\n".join(prefix_lines) + "\n\n"
            final_bundle = [
                "==============================",
                "（步驟八）最終交付報告（Final deliverable）",
                "==============================",
                current_state.get("step8_output", ""),
            ]
        else:
            prefix_lines = [
                "### Analysis Record",
                f"- Document Type: {st.session_state.document_type}",
                f"- Framework: {FRAMEWORKS.get(selected_key, {}).get('name_en', selected_key)}",
            ]
            if st.session_state.get("upstream_reference"):
                prefix_lines.append(f"- Upstream reference: {st.session_state.upstream_reference.get('name','(unknown)')}")
            else:
                prefix_lines.append("- Upstream reference: (none)")

            if st.session_state.get("quote_history"):
                prefix_lines.append("- Quote reference analysis log:")
                for i, h in enumerate(st.session_state.quote_history, start=1):
                    prefix_lines.append(f"  {i}. {h.get('name','(unknown)')} ({h.get('analyzed_at','')})")
            else:
                prefix_lines.append("- Quote references: (none)")

            prefix = "\n".join(prefix_lines) + "\n\n"
            final_bundle = [
                "==============================",
                "(Step 8) Final deliverable report",
                "==============================",
                current_state.get("step8_output", ""),
            ]

        current_state["analysis_done"] = True
        current_state["analysis_output"] = clean_report_text(prefix + "\n".join(final_bundle))
        save_state_to_disk()
        st.success("Step 8 completed. Final deliverable generated." if lang == "en" else zh("步驟八完成！已產出最終交付成品。", "步骤八完成！已产出最终交付成品。"))
        st.rerun()

    # =========================
    # Ensure current_state exists even before a framework is selected
    try:
        _ = current_state
    except NameError:
        current_state = {}

    # RESULTS area (clean + collapsible)
    # =========================
    st.markdown("---")
    st.markdown(
        f"""
<div class="ef-results-banner">
  <div class="title">{'RESULTS' if lang == 'en' else zh('結果總覽', '结果总览')}</div>
  <div class="subtitle">{'All outputs are grouped below by steps.' if lang == 'en' else zh('所有輸出依步驟整理在此區，點選可展開 / 收起。', '所有输出依步骤整理在此区，点选可展开 / 收起。')}</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    if current_state.get("step5_done"):
        render_step_block(
            "Step 5 — Main analysis result" if lang == "en" else "Step 5 — 主文件分析結果",
            current_state.get("step5_output", ""),
            expanded=False,
        )

    if st.session_state.get("upstream_reference"):
        if st.session_state.get("upstream_step6_done"):
            render_step_block(
                "Step 6-A — Upstream relevance" if lang == "en" else "Step 6-A — 上游相關性",
                st.session_state.get("upstream_step6_output", ""),
                expanded=False,
            )
        else:
            render_step_block(
                "Step 6-A — Upstream relevance" if lang == "en" else "Step 6-A — 上游相關性",
                "",
                expanded=False,
            )

    if st.session_state.get("quote_history"):
        st.markdown(f'<div class="ef-step-title">{"Step 6-B — Quote relevance (history)" if lang == "en" else "Step 6-B — 引用一致性（歷史）"}</div>', unsafe_allow_html=True)
        with st.expander("Show / Hide" if lang == "en" else zh("展開 / 收起", "展开 / 收起"), expanded=False):
            for i, h in enumerate(st.session_state.quote_history, start=1):
                qname = (h.get("quote_name") or h.get("name") or f"Quote reference #{i}") if isinstance(h, dict) else (getattr(h, "quote_name", None) or getattr(h, "name", None) or f"Quote reference #{i}")
                analyzed_at = (h.get("analyzed_at") or "") if isinstance(h, dict) else (getattr(h, "analyzed_at", "") or "")
                out = (h.get("output") or h.get("text") or "") if isinstance(h, dict) else (getattr(h, "output", "") or getattr(h, "text", "") or "")
                label = f"{i}. {qname}" + (f" — {analyzed_at}" if analyzed_at else "")
                with st.expander(label, expanded=False):
                    if out:
                        st.markdown(out)
                    else:
                        st.info("No content yet." if lang == "en" else zh("尚無內容。", "暂无内容。"))
    # Step 7 (Integration analysis) — single section, history nested inside
    st.markdown('<div class="ef-step-title">Step 7 — Integration analysis</div>', unsafe_allow_html=True)
    with st.expander("Show / Hide", expanded=False):
        integration_history = current_state.get("integration_history") or []
        if integration_history:
            for e in integration_history:
                label = f"{e.get('index','')}. {e.get('quote_name','')} — {e.get('generated_at','')}".strip()
                with st.expander(label if label else "(integration)", expanded=False):
                    st.markdown(e.get("output", ""))
        else:
            st.markdown("_No Step 7 output yet._")
# Step 8 (NEW)


    if current_state.get("step8_done"):
        render_step_block(
            "Step 8 — Final Analysis (Final deliverable)" if lang == "en" else "Step 8 — 最終分析（最終交付）",
            current_state.get("step8_output", ""),
            expanded=False,
        )
    else:
        render_step_block(
            "Step 8 — Final Analysis (Final deliverable)" if lang == "en" else "Step 8 — 最終分析（最終交付）",
            "",
            expanded=False,
        )

    # Follow-up history after results
    st.markdown("---")
    st.subheader("Follow-up (Q&A)" if lang == "en" else zh("後續提問（Q&A）", "后续提问（Q&A）"))
    render_followup_history_chat(current_state.get("followup_history", []), lang)

    # =========================
    # Download (3) choose include follow-ups
    # =========================
    st.markdown("---")
    st.subheader("Download report" if lang == "en" else zh("下載報告", "下载报告"))

    if current_state.get("analysis_done") and current_state.get("analysis_output"):
        if is_guest and current_state.get("download_used"):
            st.error("Download limit reached (1 time)." if lang == "en" else zh("已達下載次數上限（1 次）", "已达下载次数上限（1 次）"))
        else:
            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            with st.expander("Download"):
                include_qa = st.checkbox(
                    "Include follow-up Q&A replies (optional)" if lang == "en" else zh("是否包含追問回覆紀錄（選填）", "是否包含追问回复记录（选填）"),
                    value=True,
                    key=f"include_qa_{selected_key}",
                )
                # Select format (PDF/PPTX temporarily disabled)
                fmt = st.selectbox(
                    "Select format" if lang == "en" else zh("選擇格式", "选择格式"),
                    ["Word (DOCX)"],
                    key=f"fmt_{selected_key}",
                )
                st.markdown(
                    "<div style='color:#9aa0a6; margin-top:6px;'>"
                    + ("PDF (temporarily unavailable)\n" if lang == "en" else zh("PDF（暫不開放）\n", "PDF（暂不开放）\n"))
                    + ("PowerPoint (PPTX) (temporarily unavailable)" if lang == "en" else zh("PowerPoint（PPTX）（暫不開放）", "PowerPoint（PPTX）（暂不开放）"))
                    + "</div>",
                    unsafe_allow_html=True,
                )

                report = build_full_report(lang, selected_key, current_state, include_followups=include_qa)

                # PDF/PPTX are temporarily disabled (formatting not stable yet)
                if fmt.startswith("PDF") or fmt.startswith("PowerPoint"):
                    st.info(
                        "PDF/PPTX download is temporarily unavailable." if lang == "en" else zh("PDF / PPTX 下載暫不開放。", "PDF / PPTX 下载暂不开放。")
                    )
                else:
                    # Download (DOCX)
                    data = build_docx_bytes(report)
                    mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

                    now_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    framework_key = (selected_key or "unknown").replace("/", "-")
                    filename = f"Error-Free® IER {framework_key} {now_ts}" + (" +Q&A" if include_qa else "") + ".docx"


                    # Download (DOCX) — use Streamlit native download_button (supports browser save-location prompt).
                    st.download_button(
                        label=("Download" if lang == "en" else zh("開始下載", "开始下载")),
                        data=data,
                        file_name=filename,
                        mime=mime,
                        key=f"download_{framework_key}_{now_ts}",
                    )
                    

                    st.caption(
                        "Tip: If you want a 'Save As' location prompt, enable 'Ask where to save each file' in your browser settings."
                        if lang == "en"
                        else zh("提示：若你希望每次都跳出『選擇下載位置』視窗，請在瀏覽器下載設定中開啟『每次下載前詢問儲存位置』。", "提示：若你希望每次都跳出『选择下载位置』视窗，请在浏览器下载设置中开启『每次下载前询问保存位置』。")
                    )
    else:
        st.info("Complete Step 8 to enable downloads." if lang == "en" else zh("請先完成步驟八，產出最終交付報告後才能下載。", "请先完成步骤八，产出最终交付报告后才能下载。"))

    # =========================
    # Follow-up input (FIXED: no StreamlitAPIException)
    # =========================
    st.markdown("---")
    st.subheader("Ask a follow-up question" if lang == "en" else zh("提出追問", "提出追问"))
    # Hint: follow-up results will appear in the Follow-up (Q&A) section above
    _followup_hint = (
        "Your follow-up question and replies will appear in the Follow-up (Q&A) section above."
        if lang == "en" else
        "你送出的追問與回覆，將顯示在上方的 Follow-up（Q&A）區塊中。"
    )
    _followup_hint_safe = _followup_hint.replace('"', "&quot;")
    st.markdown(
        f"<span style='color:#1a73e8; font-weight:600; cursor:help;' title=\"{_followup_hint_safe}\">{_followup_hint}</span>",
        unsafe_allow_html=True,
    )


    if not current_state.get("analysis_output"):
        st.info("Please complete Step 8 before asking follow-up questions." if lang == "en" else zh("請先完成步驟八，產出最終交付成品後再進行追問。", "请先完成步骤八，产出最终交付成品后再进行追问。"))
    else:
        if is_guest and len(current_state.get("followup_history", [])) >= 3:
            st.error("Follow-up limit reached (3 times)." if lang == "en" else zh("已達追問上限（3 次）", "已达追问上限（3 次）"))
        else:
            followup_key = f"followup_input_{selected_key}"

            # ---- FIX: clear follow-up input on the next rerun (must be BEFORE the widget is instantiated) ----
            _pending_clear = st.session_state.get("_pending_clear_followup_key")
            if _pending_clear == followup_key:
                st.session_state[followup_key] = ""
                st.session_state["_pending_clear_followup_key"] = None
            # ---- END FIX ----

            col_text, col_file = st.columns([3, 1])

            with col_text:
                prompt = st.text_area(
                    "Ask a follow-up question" if lang == "en" else zh("請輸入你的追問", "请输入你的追问"),
                    key=followup_key,
                    height=140,
                    placeholder="Type your question here..." if lang == "en" else zh("在此輸入問題…", "在此输入问题…"),
                )

            with col_file:
                extra_file = st.file_uploader(
                    "Attach image/document (optional)" if lang == "en" else zh("📎 上傳圖片/文件（選填）", "📎 上传图片/文件（选填）"),
                    type=["pdf", "docx", "txt", "jpg", "jpeg", "png"],
                    key=f"extra_{selected_key}",
                )
            extra_text = read_file_to_text(extra_file) if extra_file else ""

            if st.button("Send follow-up" if lang == "en" else zh("送出追問", "送出追问"), key=f"followup_btn_{selected_key}"):
                if prompt and prompt.strip():
                    with st.spinner("Thinking..." if lang == "en" else zh("思考中...", "思考中...")):
                        answer = run_followup_qa(
                            selected_key,
                            lang,
                            st.session_state.last_doc_text or "",
                            current_state.get("analysis_output", ""),
                            prompt,
                            model_name,
                            extra_text,
                        )
                    current_state["followup_history"].append((prompt, clean_report_text(answer)))

                    # FIX: DO NOT modify widget state after instantiation; clear on next rerun
                    st.session_state["_pending_clear_followup_key"] = followup_key

                    save_state_to_disk()
                    record_usage(user_email, selected_key, "followup")
                    st.rerun()
                else:
                    st.warning("Please enter a question first." if lang == "en" else zh("請先輸入追問內容。", "请先输入追问内容。"))

    # Reset Whole Document (更正2)
    st.markdown("---")
    st.subheader("Reset Whole Document" if lang == "en" else "Reset Whole Document（全部重置）")
    st.warning(
        "Reminder: Please make sure you have downloaded your report. We do not retain your documents. Reset will remove the current review session." if lang == "en"
        else zh("溫馨提示：請確認您已經下載資料。我們不留存你們的資料；按下重置後，本次審查的文件與分析紀錄將會清空。", "温馨提示：请确认您已经下载资料。我们不留存你们的资料；按下重置后，本次審查的文件与分析纪录将会清空。")
    )
    confirm = st.checkbox("I understand and want to reset." if lang == "en" else zh("我已確認要重置。", "我已确认要重置。"), key="reset_confirm")
    if st.button("Reset Whole Document" if lang == "en" else "Reset Whole Document", key="reset_whole_btn", disabled=not confirm):
        _reset_whole_document()
        st.rerun()

    save_state_to_disk()


if __name__ == "__main__":
    main()
