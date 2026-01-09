import os, json, datetime, secrets
from pathlib import Path
from typing import Dict, List
from io import BytesIO
import base64

import streamlit as st
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
    """Register a Unicode-capable font for PDF export to avoid black boxes / garbled text.

    Priority:
    1) Try built-in CID font STSong-Light (suitable for CJK).
    2) If environment variable PDF_TTF_PATH is provided and valid, register that TTF.
    3) Fallback to Helvetica (may not cover all CJK characters).
    """
    global PDF_FONT_NAME, PDF_FONT_REGISTERED
    if PDF_FONT_REGISTERED:
        return

    try:
        # 1) Try CID font for better CJK support
        try:
            pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
            PDF_FONT_NAME = "STSong-Light"
        except Exception:
            # 2) Try external TTF if provided
            if PDF_TTF_PATH and Path(PDF_TTF_PATH).exists():
                pdfmetrics.registerFont(TTFont("ErrorFreeUnicode", PDF_TTF_PATH))
                PDF_FONT_NAME = "ErrorFreeUnicode"
            else:
                # 3) Fallback basic Latin font
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
        COMPANY_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
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
    """Load framework definitions from an external JSON file.

    Expected JSON structure:
    {
      "omission": {
        "name_zh": "...",
        "name_en": "...",
        "wrapper_zh": "...",
        "wrapper_en": "..."
      },
      ...
    }
    """
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
        DOC_TRACK_FILE.write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )
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
        USAGE_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
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
        "zh_variant": st.session_state.get("zh_variant", "tw"),  # 'tw' or 'cn'
        "usage_date": st.session_state.get("usage_date"),
        "usage_count": st.session_state.get("usage_count", 0),
        "last_doc_text": st.session_state.get("last_doc_text", ""),
        "last_doc_name": st.session_state.get("last_doc_name", ""),
        "document_type": st.session_state.get("document_type"),
        "reference_history": st.session_state.get("reference_history", []),
        "ref_pending": st.session_state.get("ref_pending", False),
        "framework_states": st.session_state.get("framework_states", {}),
        "selected_framework_key": st.session_state.get("selected_framework_key"),
        "current_doc_id": st.session_state.get("current_doc_id"),
        "company_code": st.session_state.get("company_code"),
        "show_admin": st.session_state.get("show_admin", False),
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
# File reading
# =========================


def ocr_image_to_text(file_bytes: bytes, filename: str) -> str:
    """Use OpenAI vision model to perform OCR on an image and return plain text."""
    if client is None:
        return "[Error] OPENAI_API_KEY 尚未設定，無法進行圖片 OCR。"

    # Determine image format from filename
    fname = filename.lower()
    if fname.endswith(".png"):
        img_format = "png"
    else:
        # default to jpeg for jpg / jpeg / others
        img_format = "jpeg"

    # Select model based on current user role
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
                        {
                            "type": "input_image",
                            "image": {
                                "data": b64_data,
                                "format": img_format,
                            },
                        },
                    ],
                }
            ],
            max_output_tokens=2000,
        )
        text = response.output_text or ""
        return text.strip()
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
            # 使用 OpenAI 進行圖片 OCR，將辨識結果當作文件內容
            file_bytes = uploaded_file.read()
            if not file_bytes:
                return "[讀取圖片檔案時發生錯誤：空檔案]"
            return ocr_image_to_text(file_bytes, uploaded_file.name)
        else:
            return ""
    except Exception as e:
        return f"[讀取檔案時發生錯誤: {e}]"


# =========================
# OpenAI client & model selection
# =========================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def resolve_model_for_user(role: str) -> str:
    # 高階帳號 → GPT-5.1
    if role in ["admin", "pro"]:
        return "gpt-5.1"
    # Guest 走 mini
    if role == "free":
        return "gpt-4.1-mini"
    # 公司管理者預設給高階
    return "gpt-5.1"


# =========================
# Language helpers (簡體 / 繁體)
# =========================


def zh(tw: str, cn: str = None) -> str:
    """Return zh text by variant when lang == 'zh'. Default variant is 'tw'."""
    if st.session_state.get("lang") != "zh":
        return tw
    if st.session_state.get("zh_variant", "tw") == "cn" and cn is not None:
        return cn
    return tw


# =========================
# LLM logic
# =========================


def build_analysis_input(
    language: str,
    document_text: str,
    document_type: str,
    framework_key: str,
    reference_history: List[Dict],
) -> str:
    """
    Compose analysis input so Step 5 analysis combines:
    - Step 1: Review document
    - Step 2: Document Type Selection
    - Step 3: Reference docs (uploaded so far)
    - Step 4: Framework selection
    And ensure analysis record shows reference docs list.
    """
    fw = FRAMEWORKS.get(framework_key, {})
    fw_name = fw.get("name_zh", framework_key) if language == "zh" else fw.get("name_en", framework_key)

    if language == "zh":
        lines = [
            "【分析設定】",
            f"- 文件類型（Document Type）：{document_type or '（未選擇）'}",
            f"- 分析框架（Framework）：{fw_name}",
        ]
        if reference_history:
            lines.append("- 參考文件（Reference Documents）上傳紀錄：")
            for i, r in enumerate(reference_history, start=1):
                fname = r.get("name", f"ref_{i}")
                ext = r.get("ext", "").upper()
                lines.append(f"  {i}. {fname}" + (f" ({ext})" if ext else ""))
        else:
            lines.append("- 參考文件（Reference Documents）：（未上傳）")

        lines += [
            "",
            "【Step 1：審查文件內容】",
            document_text or "",
        ]

        if reference_history:
            lines.append("")
            lines.append("【Step 3：參考文件內容】")
            for i, r in enumerate(reference_history, start=1):
                fname = r.get("name", f"ref_{i}")
                lines.append("")
                lines.append(f"--- 參考文件 {i}: {fname} ---")
                lines.append(r.get("text", "") or "")
    else:
        lines = [
            "[Analysis Settings]",
            f"- Document Type: {document_type or '(not selected)'}",
            f"- Framework: {fw_name}",
        ]
        if reference_history:
            lines.append("- Reference Documents upload log:")
            for i, r in enumerate(reference_history, start=1):
                fname = r.get("name", f"ref_{i}")
                ext = r.get("ext", "").upper()
                lines.append(f"  {i}. {fname}" + (f" ({ext})" if ext else ""))
        else:
            lines.append("- Reference Documents: (none)")

        lines += [
            "",
            "[Step 1: Review Document]",
            document_text or "",
        ]

        if reference_history:
            lines.append("")
            lines.append("[Step 3: Reference Documents]")
            for i, r in enumerate(reference_history, start=1):
                fname = r.get("name", f"ref_{i}")
                lines.append("")
                lines.append(f"--- Reference {i}: {fname} ---")
                lines.append(r.get("text", "") or "")

    return "\n".join(lines)


def run_llm_analysis(
    framework_key: str, language: str, document_text: str, model_name: str
) -> str:
    if framework_key not in FRAMEWORKS:
        return f"[Error] Framework '{framework_key}' not found in frameworks.json."

    fw = FRAMEWORKS[framework_key]
    system_prompt = fw["wrapper_zh"] if language == "zh" else fw["wrapper_en"]
    prefix = (
        "以下是要分析的文件內容：\n\n"
        if language == "zh"
        else "Here is the document to analyze:\n\n"
    )
    user_prompt = prefix + document_text

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

    doc_excerpt = document_text[:8000]
    analysis_excerpt = analysis_output[:8000]
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
        return f"[呼叫 OpenAI API 時發生錯誤: {e}]"


# =========================
# Report formatting
# =========================


def clean_report_text(text: str) -> str:
    replacements = {
        "■": "-",
        "•": "-",
        "–": "-",
        "—": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def build_full_report(lang: str, framework_key: str, state: Dict) -> str:
    analysis_output = state.get("analysis_output", "")
    followups = state.get("followup_history", [])
    fw = FRAMEWORKS.get(framework_key, {})
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    email = st.session_state.get("user_email", "unknown")

    name_zh = fw.get("name_zh", framework_key)
    name_en = fw.get("name_en", framework_key)

    if lang == "zh":
        header = [
            f"{BRAND_TITLE_ZH} 報告（分析 + Q&A）",
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
        if followups:
            header += [
                "",
                "==============================",
                "二、後續問答（Q&A）",
                "==============================",
            ]
            for i, (q, a) in enumerate(followups, start=1):
                header.append(f"[Q{i}] {q}")
                header.append(f"[A{i}] {a}")
                header.append("")
    else:
        header = [
            f"{BRAND_TITLE_EN} Report (Analysis + Q&A)",
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
        if followups:
            header += [
                "",
                "==============================",
                "2. Follow-up Q&A",
                "==============================",
            ]
            for i, (q, a) in enumerate(followups, start=1):
                header.append(f"[Q{i}] {q}")
                header.append(f"[A{i}] {a}")
                header.append("")

    return clean_report_text("\n".join(header))


def build_whole_report(lang: str, framework_states: Dict[str, Dict]) -> str:
    """Build a combined report for all frameworks (analysis + Q&A).

    The order of sections follows FRAMEWORKS definition, and only frameworks
    with completed analysis are included.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    email = st.session_state.get("user_email", "unknown")

    lines: List[str] = []
    if lang == "zh":
        lines.extend(
            [
                f"{BRAND_TITLE_ZH} 總報告（全部框架）",
                f"{BRAND_SUBTITLE_ZH}",
                f"產生時間：{now}",
                f"使用者帳號：{email}",
                "",
                "==============================",
            ]
        )
    else:
        lines.extend(
            [
                f"{BRAND_TITLE_EN} Consolidated Report (All frameworks)",
                f"{BRAND_SUBTITLE_EN}",
                f"Generated: {now}",
                f"User: {email}",
                "",
                "==============================",
            ]
        )

    for fw_key in FRAMEWORKS.keys():
        state = framework_states.get(fw_key)
        if not state or not state.get("analysis_output"):
            continue

        fw = FRAMEWORKS.get(fw_key, {})
        name_zh = fw.get("name_zh", fw_key)
        name_en = fw.get("name_en", fw_key)

        if lang == "zh":
            lines.append(f"◎ 框架：{name_zh}")
            lines.append("------------------------------")
            lines.append("一、分析結果")
        else:
            lines.append(f"◎ Framework: {name_en}")
            lines.append("------------------------------")
            lines.append("1. Analysis")

        lines.append(state.get("analysis_output", ""))

        followups = state.get("followup_history", [])
        if followups:
            if lang == "zh":
                lines.append("")
                lines.append("二、後續問答（Q&A）")
            else:
                lines.append("")
                lines.append("2. Follow-up Q&A")

            for i, (q, a) in enumerate(followups, start=1):
                lines.append(f"[Q{i}] {q}")
                lines.append(f"[A{i}] {a}")
                lines.append("")

        lines.append("")
        lines.append("================================")
        lines.append("")

    if not lines:
        return ""

    return clean_report_text("\n".join(lines))


def build_docx_bytes(text: str) -> bytes:
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()


def build_pdf_bytes(text: str) -> bytes:
    """Build a PDF using a Unicode-capable font and basic word-wrapping
    to reduce black squares / garbled characters and layout issues."""
    buf = BytesIO()
    ensure_pdf_font()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    margin_x = 40
    margin_y = 40
    line_height = 14
    max_width = width - 2 * margin_x

    # Set font; if anything fails, fallback silently to Helvetica
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
                    # Find a cut position that fits within the line width
                    cut = len(line)
                    while (
                        cut > 0
                        and pdfmetrics.stringWidth(line[:cut], PDF_FONT_NAME, 11)
                        > max_width
                    ):
                        cut -= 1
                    # Prefer breaking at a space for nicer wrapping
                    space_pos = line.rfind(" ", 0, cut)
                    if space_pos > 0:
                        cut = space_pos
                    segment = line[:cut].rstrip()
                    line = line[cut:].lstrip()
            except Exception:
                # If measurement fails, fall back to a hard cut
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
    """Build a minimal PowerPoint file that intentionally shows a 404 message.

    Per UI requirement, when users download a PPTX, the slide should display
    "404: Not Found" instead of a full slide deck.
    """
    try:
        from pptx import Presentation
    except Exception:
        # Fallback: still return a valid binary file, even if not a real PPTX.
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
# Dashboards
# =========================


def company_admin_dashboard():
    """Dashboard for company_admin role, scoped to a single company_code."""
    companies = load_companies()
    code = st.session_state.get("company_code")
    email = st.session_state.get("user_email")

    if not code or code not in companies:
        lang = st.session_state.get("lang", "zh")
        st.error(
            zh("找不到公司代碼，請聯絡系統管理員", "找不到公司代码，请联系系统管理员")
            if lang == "zh"
            else "Company code not found. Please contact system admin."
        )
        return

    entry = companies[code]
    admins = entry.get("admins", [])
    if email not in admins:
        lang = st.session_state.get("lang", "zh")
        st.error(
            zh("您沒有此公司的管理者權限", "您没有此公司的管理者权限")
            if lang == "zh"
            else "You are not an admin for this company."
        )
        return

    lang = st.session_state.get("lang", "zh")
    company_name = entry.get("company_name") or code
    content_access = entry.get("content_access", False)

    st.title(
        (zh(f"公司管理後台 - {company_name}", f"公司管理后台 - {company_name}") if lang == "zh" else f"Company Admin Dashboard - {company_name}")
    )
    st.markdown("---")

    st.subheader(zh("公司資訊", "公司信息") if lang == "zh" else "Company Info")
    st.write((zh("公司代碼：", "公司代码：") if lang == "zh" else "Company Code: ") + code)
    if lang == "zh":
        st.write(zh("可查看內容：", "可查看内容：") + (zh("是", "是") if content_access else zh("否", "否")))
    else:
        st.write("Can view content: " + ("Yes" if content_access else "No"))

    st.markdown("---")
    st.subheader(zh("學生 / 使用者列表", "学员 / 用户列表") if lang == "zh" else "Users in this company")

    users = entry.get("users", [])
    doc_tracking = load_doc_tracking()
    usage_stats = load_usage_stats()

    if not users:
        st.info(
            zh("目前尚未有任何學生註冊", "目前尚未有任何学员注册")
            if lang == "zh"
            else "No users registered for this company yet."
        )
    else:
        for u in users:
            docs = doc_tracking.get(u, [])
            st.markdown(f"**{u}**")
            st.write(
                (zh("上傳文件數：", "上传文件数：") if lang == "zh" else "Uploaded documents: ")
                + str(len(docs))
            )

            u_stats = usage_stats.get(u)
            if not u_stats:
                st.caption(
                    zh("尚無分析記錄", "尚无分析记录")
                    if lang == "zh"
                    else "No analysis usage recorded yet."
                )
            else:
                if content_access:
                    st.write(
                        (zh("最後使用時間：", "最后使用时间：") if lang == "zh" else "Last used: ")
                        + u_stats.get("last_used", "-")
                    )
                    fw_map = u_stats.get("frameworks", {})
                    for fw_key, fw_data in fw_map.items():
                        fw_name = (
                            FRAMEWORKS.get(fw_key, {}).get("name_zh", fw_key)
                            if lang == "zh"
                            else FRAMEWORKS.get(fw_key, {}).get("name_en", fw_key)
                        )
                        st.markdown(
                            f"- {fw_name}：{zh('分析', '分析')} {fw_data.get('analysis_runs', 0)} {zh('次', '次')}，"
                            f"{zh('追問', '追问')} {fw_data.get('followups', 0)} {zh('次', '次')}，"
                            f"{zh('下載', '下载')} {fw_data.get('downloads', 0)} {zh('次', '次')}"
                            if lang == "zh"
                            else f"- {fw_name}: "
                            f"analysis {fw_data.get('analysis_runs', 0)} times, "
                            f"follow-ups {fw_data.get('followups', 0)} times, "
                            f"downloads {fw_data.get('downloads', 0)} times"
                        )
                else:
                    st.caption(
                        zh("（僅顯示使用量總數，未啟用內容檢視權限）", "（仅显示使用量总数，未启用内容查看权限）")
                        if lang == "zh"
                        else "(Only aggregate usage visible; content access disabled.)"
                    )

            st.markdown("---")


def admin_dashboard():
    lang = st.session_state.get("lang", "zh")
    st.title("Admin Dashboard — Error-Free®")
    st.markdown("---")

    # 1) Guest accounts
    st.subheader(zh("📌 Guest 帳號列表", "📌 Guest 账号列表") if lang == "zh" else "📌 Guest accounts")
    guests = load_guest_accounts()
    if not guests:
        st.info(zh("目前沒有 Guest 帳號。", "目前没有 Guest 账号。") if lang == "zh" else "No guest accounts yet.")
    else:
        for email, acc in guests.items():
            st.markdown(
                f"**{email}** — password: `{acc.get('password')}` (role: {acc.get('role')})"
            )
            st.markdown("---")

    # 2) Guest document usage
    st.subheader(zh("📁 Guest 文件使用狀況", "📁 Guest 文件使用情况") if lang == "zh" else "📁 Guest document usage")
    doc_tracking = load_doc_tracking()
    if not doc_tracking:
        st.info(
            zh("尚無 Guest 上傳記錄。", "尚无 Guest 上传记录。") if lang == "zh" else "No guest uploads recorded yet."
        )
    else:
        for email, docs in doc_tracking.items():
            st.markdown(
                f"**{email}** — {zh('上傳文件數：', '上传文件数：')}{len(docs)} / 3"
                if lang == "zh"
                else f"**{email}** — uploaded documents: {len(docs)} / 3"
            )
            for d in docs:
                st.markdown(f"- {d}")
            st.markdown("---")

    # 3) Framework state in current session
    st.subheader(
        zh("🧩 模組分析與追問狀況 (Session-based)", "🧩 模块分析与追问情况 (Session-based)")
        if lang == "zh"
        else "🧩 Framework state (current session)"
    )
    fs = st.session_state.get("framework_states", {})
    if not fs:
        st.info(zh("尚無 Framework 分析記錄", "尚无 Framework 分析记录") if lang == "zh" else "No framework analysis yet.")
    else:
        for fw_key, state in fs.items():
            fw_name = (
                FRAMEWORKS.get(fw_key, {}).get("name_zh", fw_key)
                if lang == "zh"
                else FRAMEWORKS.get(fw_key, {}).get("name_en", fw_key)
            )
            st.markdown(f"### ▶ {fw_name}")
            st.write(
                f"{zh('分析完成：', '分析完成：')}{state.get('analysis_done')}"
                if lang == "zh"
                else f"Analysis done: {state.get('analysis_done')}"
            )
            st.write(
                f"{zh('追問次數：', '追问次数：')}{len(state.get('followup_history', []))}"
                if lang == "zh"
                else f"Follow-up count: {len(state.get('followup_history', []))}"
            )
            st.write(
                f"{zh('已下載報告：', '已下载报告：')}{state.get('download_used')}"
                if lang == "zh"
                else f"Downloaded report: {state.get('download_used')}"
            )
            st.markdown("---")

    # 4) 公司使用量總覽（4A）
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
            st.write(
                f"{zh('學生 / 使用者數：', '学员 / 用户数：')}{len(users)}"
                if lang == "zh"
                else f"Users: {len(users)}"
            )
            st.write(
                f"{zh('總上傳文件數：', '总上传文件数：')}{total_docs}"
                if lang == "zh"
                else f"Total uploaded documents: {total_docs}"
            )
            st.write(
                f"{zh('總分析次數：', '总分析次数：')}{total_analysis}"
                if lang == "zh"
                else f"Total analysis runs: {total_analysis}"
            )
            st.write(
                f"{zh('總追問次數：', '总追问次数：')}{total_followups}"
                if lang == "zh"
                else f"Total follow-ups: {total_followups}"
            )
            st.write(
                f"{zh('總下載次數：', '总下载次数：')}{total_downloads}"
                if lang == "zh"
                else f"Total downloads: {total_downloads}"
            )
            st.write(
                (zh("content_access：", "content_access：") if lang == "zh" else "content_access: ")
                + (zh("啟用", "启用") if content_access else zh("關閉", "关闭"))
                if lang == "zh"
                else "content_access: " + ("enabled" if content_access else "disabled")
            )
            st.markdown("---")

    # 5) 公司權限設定（4C 控制開關）
    st.subheader(zh("🔐 公司內容檢視權限設定", "🔐 公司内容查看权限设置") if lang == "zh" else "🔐 Company content access settings")
    if not companies:
        st.info(zh("尚無公司可設定。", "尚无公司可设置。") if lang == "zh" else "No companies to configure.")
    else:
        for code, entry in companies.items():
            label = f"{entry.get('company_name') or code} ({code})"
            key = f"content_access_{code}"
            current_val = entry.get("content_access", False)
            st.checkbox(
                label + (zh(" — 可檢視學生分析使用量", " — 可查看学员分析使用量") if lang == "zh" else " — can view user usage details"),
                value=current_val,
                key=key,
            )

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
        if st.button(
            zh("返回分析頁面", "返回分析页面") if st.session_state.get("lang", "zh") == "zh" else "Back to analysis"
        ):
            st.session_state.show_admin = False
            save_state_to_disk()
            st.rerun()
        return True
    return False


# =========================
# Main app
# =========================

# =========================
# Branding (Title / Subtitle / Logo)
# =========================

BRAND_TITLE_EN = "Error-Free® Intelligence Engine"
BRAND_TAGLINE_EN = "An AI-enhanced intelligence engine that helps organizations analyze risks, prevent errors, and make better decisions."
BRAND_SUBTITLE_EN = "Pioneered and refined by Dr. Chiu’s Error-Free® team since 1987."

BRAND_TITLE_ZH = zh("零錯誤智能引擎", "零错误智能引擎")
BRAND_TAGLINE_ZH = zh("一套 AI 強化的智能引擎，協助公司或組織進行風險分析、預防錯誤，並提升決策品質。", "一套 AI 强化的智能引擎，协助公司或组织进行风险分析、预防错误，并提升决策品质。")
BRAND_SUBTITLE_ZH = zh("邱博士零錯誤團隊自 1987 年起領先研發並持續深化至今。", "邱博士零错误团队自 1987 年起领先研发并持续深化至今。")

# Put your logo file in repo, e.g. assets/errorfree_logo.png
LOGO_PATH = "assets/errorfree_logo.png"


def language_selector():
    """Top-level language toggle: English / 中文简体 / 中文繁體."""
    current_lang = st.session_state.get("lang", "zh")
    current_variant = st.session_state.get("zh_variant", "tw")

    # Determine default index
    if current_lang == "en":
        index = 0
    else:
        index = 1 if current_variant == "cn" else 2

    choice = st.radio("Language / 語言", ("English", "中文简体", "中文繁體"), index=index)

    if choice == "English":
        st.session_state.lang = "en"
        # Keep last variant for later switching
        if "zh_variant" not in st.session_state:
            st.session_state.zh_variant = "tw"
    else:
        st.session_state.lang = "zh"
        st.session_state.zh_variant = "cn" if choice == "中文简体" else "tw"


def main():
    st.set_page_config(page_title=BRAND_TITLE_EN, layout="wide")
    restore_state_from_disk()

    # 初始化 session
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
        ("reference_history", []),
        ("ref_pending", False),
        ("framework_states", {}),
        ("selected_framework_key", None),
        ("current_doc_id", None),
        ("company_code", None),
        ("show_admin", False),
    ]
    for k, v in defaults:
        if k not in st.session_state:
            st.session_state[k] = v

    # 如果還沒選擇框架，就用 frameworks.json 的第一個 key
    if st.session_state.selected_framework_key is None and FRAMEWORKS:
        st.session_state.selected_framework_key = list(FRAMEWORKS.keys())[0]

    doc_tracking = load_doc_tracking()

    # Sidebar
    with st.sidebar:
        lang = st.session_state.lang

        # 語言切換放在 sidebar 頂部
        language_selector()
        lang = st.session_state.lang

        if (
            st.session_state.is_authenticated
            and st.session_state.user_role in ["admin", "pro", "company_admin"]
        ):
            if st.button("管理後台 Admin Dashboard"):
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
                st.session_state.reference_history = []
                st.session_state.ref_pending = False
                st.session_state.current_doc_id = None
                save_state_to_disk()
                st.rerun()
        else:
            st.subheader(zh("尚未登入", "尚未登录") if lang == "zh" else "Not Logged In")
            # Move the original bullet list under "Not Logged In"
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

        # Replace the previous bullet list area with the AI disclaimer text
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

        # 1. 內部員工 / 會員登入
        st.markdown(
            ("### " + zh("內部員工 / 會員登入", "内部员工 / 会员登录")) if lang == "zh" else "### Internal Employee / Member Login"
        )
        emp_email = st.text_input("Email", key="emp_email")
        emp_pw = st.text_input(
            zh("密碼", "密码") if lang == "zh" else "Password",
            type="password",
            key="emp_pw",
        )
        if st.button(zh("登入", "登录") if lang == "zh" else "Login", key="emp_login_btn"):
            account = ACCOUNTS.get(emp_email)
            if account and account["password"] == emp_pw:
                st.session_state.user_email = emp_email
                st.session_state.user_role = account["role"]
                st.session_state.is_authenticated = True
                save_state_to_disk()
                st.rerun()
            else:
                st.error(
                    zh("帳號或密碼錯誤", "账号或密码错误")
                    if lang == "zh"
                    else "Invalid email or password"
                )

        st.markdown("---")

        # 2. 公司管理者註冊 － 公司管理者登入（同一橫排）
        st.markdown(
            ("### " + zh("公司管理者（企業窗口）", "公司管理者（企业窗口）"))
            if lang == "zh"
            else "### Company Admin (Client-side)"
        )
        col_ca_signup, col_ca_login = st.columns(2)

        # 公司管理者註冊
        with col_ca_signup:
            st.markdown("**" + (zh("公司管理者註冊", "公司管理者注册") if lang == "zh" else "Company Admin Signup") + "**")
            ca_new_email = st.text_input(
                zh("管理者註冊 Email", "管理者注册 Email") if lang == "zh" else "Admin signup email",
                key="ca_new_email",
            )
            ca_new_pw = st.text_input(
                zh("設定管理者密碼", "设置管理者密码") if lang == "zh" else "Set admin password",
                type="password",
                key="ca_new_pw",
            )
            ca_company_code = st.text_input("公司代碼 Company Code", key="ca_company_code")

            if st.button(
                zh("建立管理者帳號", "建立管理者账号") if lang == "zh" else "Create Company Admin Account",
                key="ca_signup_btn",
            ):
                if not ca_new_email or not ca_new_pw or not ca_company_code:
                    st.error(
                        zh("請完整填寫管理者註冊資訊", "请完整填写管理者注册信息")
                        if lang == "zh"
                        else "Please fill all admin signup fields"
                    )
                else:
                    companies = load_companies()
                    guests = load_guest_accounts()
                    if ca_company_code not in companies:
                        st.error(
                            zh("公司代碼不存在，請先向系統管理員建立公司", "公司代码不存在，请先向系统管理员建立公司")
                            if lang == "zh"
                            else "Company code not found. Please ask the system admin to create it."
                        )
                    elif ca_new_email in ACCOUNTS or ca_new_email in guests:
                        st.error(
                            zh("此 Email 已被使用", "此 Email 已被使用")
                            if lang == "zh"
                            else "This email is already in use"
                        )
                    else:
                        guests[ca_new_email] = {
                            "password": ca_new_pw,
                            "role": "company_admin",
                            "company_code": ca_company_code,
                        }
                        save_guest_accounts(guests)

                        entry = companies[ca_company_code]
                        admins = entry.get("admins", [])
                        if ca_new_email not in admins:
                            admins.append(ca_new_email)
                        entry["admins"] = admins
                        if "company_name" not in entry:
                            entry["company_name"] = ""
                        if "content_access" not in entry:
                            entry["content_access"] = False
                        companies[ca_company_code] = entry
                        save_companies(companies)

                        st.success(
                            zh("公司管理者帳號已建立", "公司管理者账号已建立")
                            if lang == "zh"
                            else "Company admin account created"
                        )

        # 公司管理者登入
        with col_ca_login:
            st.markdown("**" + (zh("公司管理者登入", "公司管理者登录") if lang == "zh" else "Company Admin Login") + "**")
            ca_email = st.text_input(
                zh("管理者 Email", "管理者 Email") if lang == "zh" else "Admin Email",
                key="ca_email",
            )
            ca_pw = st.text_input(
                zh("管理者密碼", "管理者密码") if lang == "zh" else "Admin Password",
                type="password",
                key="ca_pw",
            )
            if st.button(
                zh("管理者登入", "管理者登录") if lang == "zh" else "Login as Company Admin",
                key="ca_login_btn",
            ):
                guests = load_guest_accounts()
                acc = guests.get(ca_email)
                if (
                    acc
                    and acc.get("password") == ca_pw
                    and acc.get("role") == "company_admin"
                ):
                    st.session_state.user_email = ca_email
                    st.session_state.user_role = "company_admin"
                    st.session_state.company_code = acc.get("company_code")
                    st.session_state.is_authenticated = True
                    save_state_to_disk()
                    st.rerun()
                else:
                    st.error(
                        zh("管理者帳號或密碼錯誤", "管理者账号或密码错误")
                        if lang == "zh"
                        else "Invalid company admin credentials"
                    )

        st.markdown("---")

        # 3. Guest 註冊 － Guest 登入（同一橫排）
        st.markdown("### " + (zh("Guest 試用帳號", "Guest 试用账号") if lang == "zh" else "Guest Trial Accounts"))
        col_guest_signup, col_guest_login = st.columns(2)

        # Guest 註冊
        with col_guest_signup:
            st.markdown("**" + (zh("Guest 試用註冊", "Guest 试用注册") if lang == "zh" else "Guest Signup") + "**")
            new_guest_email = st.text_input(
                zh("註冊 Email", "注册 Email") if lang == "zh" else "Email for signup",
                key="new_guest_email",
            )
            guest_company_code = st.text_input(
                zh("公司代碼 Company Code", "公司代码 Company Code") if lang == "zh" else "Company Code",
                key="guest_company_code",
            )

            if st.button(
                zh("取得 Guest 密碼", "获取 Guest 密码") if lang == "zh" else "Generate Guest Password",
                key="guest_signup_btn",
            ):
                if not new_guest_email:
                    st.error(
                        zh("請輸入 Email", "请输入 Email")
                        if lang == "zh"
                        else "Please enter an email"
                    )
                elif not guest_company_code:
                    st.error(
                        zh("請輸入公司代碼", "请输入公司代码")
                        if lang == "zh"
                        else "Please enter your Company Code"
                    )
                else:
                    guests = load_guest_accounts()
                    companies = load_companies()
                    if guest_company_code not in companies:
                        st.error(
                            zh("公司代碼不存在，請向講師或公司窗口確認", "公司代码不存在，请向讲师或公司窗口确认")
                            if lang == "zh"
                            else "Invalid Company Code. Please check with your instructor or admin."
                        )
                    elif new_guest_email in guests or new_guest_email in ACCOUNTS:
                        st.error(
                            zh("Email 已存在", "Email 已存在")
                            if lang == "zh"
                            else "Email already exists"
                        )
                    else:
                        pw = "".join(secrets.choice("0123456789") for _ in range(8))
                        guests[new_guest_email] = {
                            "password": pw,
                            "role": "free",
                            "company_code": guest_company_code,
                        }
                        save_guest_accounts(guests)

                        entry = companies[guest_company_code]
                        users = entry.get("users", [])
                        if new_guest_email not in users:
                            users.append(new_guest_email)
                        entry["users"] = users
                        if "company_name" not in entry:
                            entry["company_name"] = entry.get("company_name", "")
                        if "content_access" not in entry:
                            entry["content_access"] = False
                        companies[guest_company_code] = entry
                        save_companies(companies)

                        st.success(
                            (zh(f"Guest 帳號已建立！密碼：{pw}", f"Guest 账号已建立！密码：{pw}") if lang == "zh" else f"Guest account created! Password: {pw}")
                        )

        # Guest 登入
        with col_guest_login:
            st.markdown("**" + (zh("Guest 試用登入", "Guest 试用登录") if lang == "zh" else "Guest Login") + "**")
            g_email = st.text_input("Guest Email", key="g_email")
            g_pw = st.text_input(
                zh("密碼", "密码") if lang == "zh" else "Password",
                type="password",
                key="g_pw",
            )
            if st.button(
                zh("登入 Guest", "登录 Guest") if lang == "zh" else "Login as Guest",
                key="guest_login_btn",
            ):
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
                    st.error(
                        zh("帳號或密碼錯誤", "账号或密码错误")
                        if lang == "zh"
                        else "Invalid guest credentials"
                    )

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

    # Step 1: upload review doc (single file only)
    st.subheader(zh("步驟一：上傳審閱文件", "步骤一：上传审阅文件") if lang == "zh" else "Step 1: Upload Review Document")
    st.caption(
        zh("提醒：一次只能上載 1 份文件進行完整內容分析。", "提醒：一次只能上传 1 份文件进行完整内容分析。")
        if lang == "zh"
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
                        st.error(
                            zh("試用帳號最多上傳 3 份文件", "试用账号最多上传 3 份文件")
                            if lang == "zh"
                            else "Trial accounts may upload up to 3 documents only"
                        )
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
        shown_name = st.session_state.get("last_doc_name") or zh("（已上傳）", "（已上传）")
        st.info(
            (zh(f"已上傳審閱文件：{shown_name}。如需更換文件，請使用 Reset document。", f"已上传审阅文件：{shown_name}。如需更换文件，请使用 Reset document。") if lang == "zh"
             else f"Review document uploaded: {shown_name}. To change it, please use Reset document.")
        )

    # Step 2: Document Type Selection (single select)
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

    # Keep existing state logic: default to first option if not set
    if st.session_state.get("document_type") not in DOC_TYPES:
        st.session_state.document_type = DOC_TYPES[0]

    st.session_state.document_type = st.selectbox(
        zh("選擇文件類型", "选择文件类型") if lang == "zh" else "Select document type",
        DOC_TYPES,
        index=DOC_TYPES.index(st.session_state.document_type),
        key="document_type_select",
    )
    save_state_to_disk()

    # Step 3: Reference docs (optional, one file per analysis cycle)
    st.subheader(
        zh("步驟三：上傳參考文件（選填）", "步骤三：上传参考文件（选填）")
        if lang == "zh"
        else "Step 3: Upload Reference Documents (optional)"
    )

    st.caption(
        zh(
            "一次只能上傳 1 份參考文件。第一次分析可上傳 1 份；分析完成後，可再上傳第 2 份（依此類推），避免分析時間過長或輸出錯亂。",
            "一次只能上传 1 份参考文件。第一次分析可上传 1 份；分析完成后，可再上传第 2 份（依此类推），避免分析时间过长或输出错乱。",
        )
        if lang == "zh"
        else "You can upload only 1 reference document at a time. Upload 1 for the first analysis; after analysis completes, you may upload the 2nd (and so on) to avoid long runtimes or confused outputs."
    )

    if "reference_history" not in st.session_state:
        st.session_state.reference_history = []
    if "ref_pending" not in st.session_state:
        st.session_state.ref_pending = False

    # Show uploaded reference log
    if st.session_state.reference_history:
        st.markdown("**" + (zh("已上傳參考文件紀錄：", "已上传参考文件记录：") if lang == "zh" else "Reference documents uploaded:") + "**")
        for i, r in enumerate(st.session_state.reference_history, start=1):
            fname = r.get("name", f"ref_{i}")
            ext = r.get("ext", "").upper()
            st.markdown(f"- {i}. {fname}" + (f" ({ext})" if ext else ""))

    ref_disabled = bool(st.session_state.ref_pending)

    ref_uploader_key = f"ref_uploader_{len(st.session_state.reference_history)}"
    reference_file = st.file_uploader(
        zh("上傳參考文件（PDF / DOCX / TXT / 圖片）", "上传参考文件（PDF / DOCX / TXT / 图片）")
        if lang == "zh"
        else "Upload reference document (PDF / DOCX / TXT / Image)",
        type=["pdf", "docx", "txt", "jpg", "jpeg", "png"],
        key=ref_uploader_key,
        disabled=ref_disabled,
    )

    if ref_disabled:
        st.info(
            zh(
                "已上傳 1 份參考文件，請先完成一次分析後再上傳下一份。",
                "已上传 1 份参考文件，请先完成一次分析后再上传下一份。",
            )
            if lang == "zh"
            else "A reference document has been uploaded. Please run analysis once before uploading the next reference."
        )

    if reference_file is not None and not ref_disabled:
        ref_text = read_file_to_text(reference_file)
        if ref_text:
            name = reference_file.name
            ext = Path(name).suffix.lstrip(".")
            st.session_state.reference_history.append(
                {
                    "name": name,
                    "ext": ext,
                    "text": ref_text,
                    "uploaded_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            st.session_state.ref_pending = True
            save_state_to_disk()
            st.rerun()

    # Step 4: select framework (single select note)
    st.subheader(zh("步驟四：選擇分析框架（僅單選）", "步骤四：选择分析框架（仅单选）") if lang == "zh" else "Step 4: Select Framework")
    st.caption(
        zh(
            "僅單選。如需分析下一個 Framework，建議先 Reset document（一次分析一個 Framework），避免分析時間過長或輸出錯亂。",
            "仅单选。如需分析下一个 Framework，建议先 Reset document（一次分析一个 Framework），避免分析时间过长或输出错乱。",
        )
        if lang == "zh"
        else "Single selection only. To analyze the next framework, it is recommended to Reset document (one framework per run) to avoid long runtimes or confused outputs."
    )

    if not FRAMEWORKS:
        st.error(
            zh("尚未在 frameworks.json 中定義任何框架。", "尚未在 frameworks.json 中定义任何框架。")
            if lang == "zh"
            else "No frameworks defined in frameworks.json."
        )
        return

    fw_keys = list(FRAMEWORKS.keys())
    fw_labels = [
        FRAMEWORKS[k]["name_zh"] if lang == "zh" else FRAMEWORKS[k]["name_en"]
        for k in fw_keys
    ]
    key_to_label = dict(zip(fw_keys, fw_labels))
    label_to_key = dict(zip(fw_labels, fw_keys))

    current_fw_key = st.session_state.selected_framework_key or fw_keys[0]
    current_label = key_to_label.get(current_fw_key, fw_labels[0])

    selected_label = st.selectbox(
        zh("選擇框架", "选择框架") if lang == "zh" else "Select framework",
        fw_labels,
        index=fw_labels.index(current_label) if current_label in fw_labels else 0,
        key="framework_selectbox",
    )
    selected_key = label_to_key[selected_label]
    st.session_state.selected_framework_key = selected_key

    framework_states = st.session_state.framework_states
    if selected_key not in framework_states:
        framework_states[selected_key] = {
            "analysis_done": False,
            "analysis_output": "",
            "followup_history": [],
            "download_used": False,
        }
    save_state_to_disk()
    current_state = framework_states[selected_key]

    st.markdown("---")

    # Step 5: run analysis (must combine Step 1/2/3/4)
    st.subheader(zh("步驟五：執行分析", "步骤五：执行分析") if lang == "zh" else "Step 5: Run Analysis")
    st.caption(
        zh(
            "分析結果會綜合：Step 1 審查文件、Step 2 文件類型、Step 3 參考文件（如有）、Step 4 框架選擇後產出。",
            "分析结果会综合：Step 1 审查文件、Step 2 文件类型、Step 3 参考文件（如有）、Step 4 框架选择后产出。",
        )
        if lang == "zh"
        else "The analysis output will be generated by combining: Step 1 review document, Step 2 document type, Step 3 reference docs (if any), and Step 4 framework selection."
    )

    can_run = not current_state["analysis_done"]

    if can_run:
        run_btn = st.button(zh("開始分析", "开始分析") if lang == "zh" else "Run analysis", key="run_analysis_btn")
    else:
        run_btn = False
        st.info(
            zh("此框架已完成一次分析", "此框架已完成一次分析")
            if lang == "zh"
            else "Analysis already completed for this framework."
        )

    # 只有非 Guest 才能 Reset
    if not is_guest:
        if st.button(zh("重置（新文件）", "重置（新文件）") if lang == "zh" else "Reset document"):
            st.session_state.framework_states = {}
            st.session_state.last_doc_text = ""
            st.session_state.last_doc_name = ""
            st.session_state.document_type = None
            st.session_state.reference_history = []
            st.session_state.ref_pending = False
            st.session_state.current_doc_id = None
            save_state_to_disk()
            st.rerun()

    if run_btn and can_run:
        if not st.session_state.last_doc_text:
            st.error(zh("請先上傳審閱文件（Step 1）", "请先上传审阅文件（Step 1）") if lang == "zh" else "Please upload a review document first (Step 1).")
        elif not st.session_state.get("document_type"):
            st.error(zh("請先選擇文件類型（Step 2）", "请先选择文件类型（Step 2）") if lang == "zh" else "Please select a document type first (Step 2).")
        else:
            # Compose combined input (Step 1 + Step 2 + Step 3 + Step 4)
            combined_input = build_analysis_input(
                lang,
                st.session_state.last_doc_text,
                st.session_state.document_type,
                selected_key,
                st.session_state.reference_history,
            )
            with st.spinner(zh("分析中...", "分析中...") if lang == "zh" else "Running analysis..."):
                analysis_text = run_llm_analysis(
                    selected_key,
                    lang,
                    combined_input,
                    model_name,
                )

            current_state["analysis_done"] = True

            # Prefix to guarantee record shows doc type + reference upload log
            if lang == "zh":
                prefix_lines = [
                    "### 分析紀錄（必讀）",
                    f"- 文件類型（Document Type）：{st.session_state.document_type}",
                    f"- 框架（Framework）：{FRAMEWORKS.get(selected_key, {}).get('name_zh', selected_key)}",
                ]
                if st.session_state.reference_history:
                    prefix_lines.append("- 參考文件（Reference Documents）上傳紀錄：")
                    for i, r in enumerate(st.session_state.reference_history, start=1):
                        fname = r.get("name", f"ref_{i}")
                        ext = r.get("ext", "").upper()
                        prefix_lines.append(f"  {i}. {fname}" + (f" ({ext})" if ext else ""))
                else:
                    prefix_lines.append("- 參考文件（Reference Documents）：（未上傳）")
                prefix = "\n".join(prefix_lines) + "\n\n"
            else:
                prefix_lines = [
                    "### Analysis Record",
                    f"- Document Type: {st.session_state.document_type}",
                    f"- Framework: {FRAMEWORKS.get(selected_key, {}).get('name_en', selected_key)}",
                ]
                if st.session_state.reference_history:
                    prefix_lines.append("- Reference documents upload log:")
                    for i, r in enumerate(st.session_state.reference_history, start=1):
                        fname = r.get("name", f"ref_{i}")
                        ext = r.get("ext", "").upper()
                        prefix_lines.append(f"  {i}. {fname}" + (f" ({ext})" if ext else ""))
                else:
                    prefix_lines.append("- Reference documents: (none)")
                prefix = "\n".join(prefix_lines) + "\n\n"

            current_state["analysis_output"] = clean_report_text(prefix + (analysis_text or ""))
            current_state["followup_history"] = []
            save_state_to_disk()
            record_usage(user_email, selected_key, "analysis")

            # After analysis completes, allow uploading next reference doc
            st.session_state.ref_pending = False
            save_state_to_disk()

            st.success(zh("分析完成！", "分析完成！") if lang == "zh" else "Analysis completed!")

    # Show all framework results (unchanged)
    any_analysis = False
    for fw_key in FRAMEWORKS.keys():
        state = framework_states.get(fw_key)
        if not state or not state.get("analysis_output"):
            continue

        any_analysis = True
        st.markdown("---")
        fw = FRAMEWORKS[fw_key]
        fw_name = fw["name_zh"] if lang == "zh" else fw["name_en"]
        if lang == "zh":
            title_text = f"⭐ {fw_name}：分析結果 + Download"
        else:
            title_text = f"⭐ {fw_name}: Analysis result + Download"
        st.subheader(title_text)

        # 分析結果
        st.markdown("#### " + (zh("分析結果", "分析结果") if lang == "zh" else "Analysis result"))
        st.markdown(state["analysis_output"])

        # Download 區塊
        st.markdown("##### " + (zh("下載報告", "下载报告") if lang == "zh" else "Download report"))
        st.caption(
            zh("報告只包含分析與 Q&A，不含原始文件。", "报告只包含分析与 Q&A，不含原始文件。")
            if lang == "zh"
            else "Report includes analysis + Q&A only (no original document)."
        )

        if is_guest and state.get("download_used"):
            st.error(
                zh("已達下載次數上限（1 次）", "已达下载次数上限（1 次）")
                if lang == "zh"
                else "Download limit reached (1 time)."
            )
        else:
            report = build_full_report(lang, fw_key, state)
            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            with st.expander("Download"):
                fmt = st.radio(
                    zh("選擇格式", "选择格式") if lang == "zh" else "Select format",
                    ["Word (DOCX)", "PDF", "PowerPoint (PPTX)"],
                    key=f"fmt_{fw_key}",
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
                    try:
                        data = build_pptx_bytes(report)
                        mime = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                        ext = "pptx"
                    except Exception as e:
                        st.error(
                            (zh(f"PPTX 匯出失敗：{e}", f"PPTX 导出失败：{e}") if lang == "zh" else f"PPTX export failed: {e}")
                        )
                        data = b""
                        mime = "application/octet-stream"
                        ext = "pptx"

                if data:
                    clicked = st.download_button(
                        zh("開始下載", "开始下载") if lang == "zh" else "Download",
                        data=data,
                        file_name=f"errorfree_{fw_key}_{now_str}.{ext}",
                        mime=mime,
                        key=f"dl_{fw_key}_{ext}",
                    )
                    if clicked:
                        state["download_used"] = True
                        save_state_to_disk()
                        record_usage(user_email, fw_key, "download")

    # Follow-up/Q&A area (unchanged)
    if any_analysis:
        # 5-1) Follow-up Q&A history (all frameworks)
        st.markdown("---")
        st.subheader(zh("後續提問紀錄（Q&A history）", "后续提问记录（Q&A history）") if lang == "zh" else "Follow-up Q&A history")

        has_any_followups = False
        for fw_key in FRAMEWORKS.keys():
            state = framework_states.get(fw_key)
            if not state or not state.get("followup_history"):
                continue
            has_any_followups = True
            for i, (q, a) in enumerate(state["followup_history"], start=1):
                st.markdown(f"**Q{i}:** {q}")
                st.markdown(f"**A{i}:** {a}")
                st.markdown("---")
        if not has_any_followups:
            st.info(zh("尚無追問", "尚无追问") if lang == "zh" else "No follow-up questions yet.")

        # 5-2) Download whole report (all frameworks)
        st.subheader(zh("下載全部報告（All frameworks）", "下载全部报告（All frameworks）") if lang == "zh" else "Download whole report")
        st.caption(
            zh("一次匯出目前所有框架的分析與 Q&A，不含原始文件。", "一次导出目前所有框架的分析与 Q&A，不含原始文件。")
            if lang == "zh"
            else "Export a single report that includes analysis + Q&A from all frameworks (no original document)."
        )

        whole_report = build_whole_report(lang, framework_states)
        now_str_all = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        with st.expander(zh("Download whole report（全部框架）", "Download whole report（全部框架）") if lang == "zh" else "Download whole report"):
            fmt_all = st.radio(
                zh("選擇格式", "选择格式") if lang == "zh" else "Select format",
                ["Word (DOCX)", "PDF", "PowerPoint (PPTX)"],
                key="fmt_all",
            )

            data_all: bytes
            mime_all: str
            ext_all: str

            if fmt_all.startswith("Word"):
                data_all = build_docx_bytes(whole_report)
                mime_all = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                ext_all = "docx"
            elif fmt_all.startswith("PDF"):
                data_all = build_pdf_bytes(whole_report)
                mime_all = "application/pdf"
                ext_all = "pdf"
            else:
                try:
                    data_all = build_pptx_bytes(whole_report)
                    mime_all = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                    ext_all = "pptx"
                except Exception as e:
                    st.error(
                        (zh(f"PPTX 匯出失敗：{e}", f"PPTX 导出失败：{e}") if lang == "zh" else f"PPTX export failed: {e}")
                    )
                    data_all = b""
                    mime_all = "application/octet-stream"
                    ext_all = "pptx"

            if data_all:
                clicked_all = st.download_button(
                    zh("開始下載", "开始下载") if lang == "zh" else "Download",
                    data=data_all,
                    file_name=f"errorfree_all_frameworks_{now_str_all}.{ext_all}",
                    mime=mime_all,
                    key=f"dl_all_{ext_all}",
                )
                if clicked_all:
                    # Record as a special "whole_report" framework in usage stats
                    record_usage(user_email, "whole_report", "download")

        # 5-3) Global follow-up area（針對目前選中的框架）
        st.markdown("---")
        st.subheader(zh("後續提問", "后续提问") if lang == "zh" else "Follow-up questions")

        curr_state = framework_states[selected_key]
        if is_guest and len(curr_state["followup_history"]) >= 3:
            st.error(
                zh("已達追問上限（3 次）", "已达追问上限（3 次）")
                if lang == "zh"
                else "Follow-up limit reached (3 times)."
            )
        else:
            col_text, col_file = st.columns([3, 1])

            followup_key = f"followup_input_{selected_key}"
            with col_text:
                prompt_label = (
                    f"{zh('針對', '针对')} {FRAMEWORKS[selected_key]['name_zh']} {zh('的追問', '的追问')}"
                    if lang == "zh"
                    else "Ask Error-Free® Intelligence Engine a follow-up?"
                )
                prompt = st.text_area(
                    prompt_label,
                    key=followup_key,
                    height=150,
                )

            with col_file:
                extra_file = st.file_uploader(
                    zh("📎 上傳圖片/文件（選填）", "📎 上传图片/文件（选填）")
                    if lang == "zh"
                    else "📎 Attach image/document (optional)",
                    type=["pdf", "docx", "txt", "jpg", "jpeg", "png"],
                    key=f"extra_{selected_key}",
                )

            extra_text = read_file_to_text(extra_file) if extra_file else ""

            if st.button(
                zh("送出追問", "送出追问") if lang == "zh" else "Send follow-up",
                key=f"followup_btn_{selected_key}",
            ):
                if prompt and prompt.strip():
                    with st.spinner(zh("思考中...", "思考中...") if lang == "zh" else "Thinking..."):
                        answer = run_followup_qa(
                            selected_key,
                            lang,
                            st.session_state.last_doc_text or "",
                            curr_state["analysis_output"],
                            prompt,
                            model_name,
                            extra_text,
                        )
                    curr_state["followup_history"].append(
                        (prompt, clean_report_text(answer))
                    )
                    save_state_to_disk()
                    record_usage(user_email, selected_key, "followup")
                    st.rerun()

    save_state_to_disk()


if __name__ == "__main__":
    main()
