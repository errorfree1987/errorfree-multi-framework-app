import os, json, datetime, secrets
from pathlib import Path
from typing import Dict, List
from io import BytesIO

import streamlit as st
import pdfplumber
from docx import Document
from openai import OpenAI
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

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
        "usage_date": st.session_state.get("usage_date"),
        "usage_count": st.session_state.get("usage_count", 0),
        "last_doc_text": st.session_state.get("last_doc_text", ""),
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
            # 目前僅標記有附加圖片，無法直接讀取圖片文字
            return (
                f"[附加圖片檔案：{uploaded_file.name}，目前系統無法自動擷取圖片內文字，"
                "請盡量提供文字版本的文件。]"
            )
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
# LLM logic
# =========================


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
            "Error-Free® 多框架 AI 文件分析報告（分析 + Q&A）",
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
            "Error-Free® Multi-framework AI Report (Analysis + Q&A)",
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


def build_docx_bytes(text: str) -> bytes:
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()


def build_pdf_bytes(text: str) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    y = height - 40

    for line in text.split("\n"):
        safe_line = line.replace("\t", "    ")
        c.drawString(40, y, safe_line[:1000])
        y -= 14
        if y < 40:
            c.showPage()
            y = height - 40
    c.save()
    buf.seek(0)
    return buf.getvalue()


def build_pptx_bytes(text: str) -> bytes:
    """Very simple PPTX: one slide with bullet points from the report text.
    If python-pptx is not installed, fall back to a DOCX file content in PPTX container.
    """
    try:
        from pptx import Presentation
        from pptx.util import Pt
    except Exception:
        # Fallback: still return a valid binary file, even if not a real PPTX.
        return build_docx_bytes("PowerPoint export requires python-pptx.\n\n" + text)

    prs = Presentation()
    layout = prs.slide_layouts[1]  # title + content
    slide = prs.slides.add_slide(layout)
    slide.shapes.title.text = "Error-Free Analysis Report"

    body = slide.placeholders[1].text_frame
    body.clear()
    first = True
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if first:
            body.text = line
            p = body.paragraphs[0]
            p.font.size = Pt(18)
            first = False
        else:
            p = body.add_paragraph()
            p.text = line
            p.level = 0
            p.font.size = Pt(14)

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
            "找不到公司代碼，請聯絡系統管理員"
            if lang == "zh"
            else "Company code not found. Please contact system admin."
        )
        return

    entry = companies[code]
    admins = entry.get("admins", [])
    if email not in admins:
        lang = st.session_state.get("lang", "zh")
        st.error(
            "您沒有此公司的管理者權限"
            if lang == "zh"
            else "You are not an admin for this company."
        )
        return

    lang = st.session_state.get("lang", "zh")
    company_name = entry.get("company_name") or code
    content_access = entry.get("content_access", False)

    st.title(
        f"公司管理後台 - {company_name}"
        if lang == "zh"
        else f"Company Admin Dashboard - {company_name}"
    )
    st.markdown("---")

    st.subheader("公司資訊" if lang == "zh" else "Company Info")
    st.write(("公司代碼：" if lang == "zh" else "Company Code: ") + code)
    if lang == "zh":
        st.write("可查看內容：" + ("是" if content_access else "否"))
    else:
        st.write("Can view content: " + ("Yes" if content_access else "No"))

    st.markdown("---")
    st.subheader("學生 / 使用者列表" if lang == "zh" else "Users in this company")

    users = entry.get("users", [])
    doc_tracking = load_doc_tracking()
    usage_stats = load_usage_stats()

    if not users:
        st.info(
            "目前尚未有任何學生註冊"
            if lang == "zh"
            else "No users registered for this company yet."
        )
    else:
        for u in users:
            docs = doc_tracking.get(u, [])
            st.markdown(f"**{u}**")
            st.write(
                ("上傳文件數：" if lang == "zh" else "Uploaded documents: ")
                + str(len(docs))
            )

            u_stats = usage_stats.get(u)
            if not u_stats:
                st.caption(
                    "尚無分析記錄"
                    if lang == "zh"
                    else "No analysis usage recorded yet."
                )
            else:
                if content_access:
                    st.write(
                        "最後使用時間："
                        + u_stats.get("last_used", "-")
                        if lang == "zh"
                        else "Last used: " + u_stats.get("last_used", "-")
                    )
                    fw_map = u_stats.get("frameworks", {})
                    for fw_key, fw_data in fw_map.items():
                        fw_name = (
                            FRAMEWORKS.get(fw_key, {}).get("name_zh", fw_key)
                            if lang == "zh"
                            else FRAMEWORKS.get(fw_key, {}).get("name_en", fw_key)
                        )
                        st.markdown(
                            f"- {fw_name}：分析 {fw_data.get('analysis_runs', 0)} 次，"
                            f"追問 {fw_data.get('followups', 0)} 次，"
                            f"下載 {fw_data.get('downloads', 0)} 次"
                            if lang == "zh"
                            else f"- {fw_name}: "
                            f"analysis {fw_data.get('analysis_runs', 0)} times, "
                            f"follow-ups {fw_data.get('followups', 0)} times, "
                            f"downloads {fw_data.get('downloads', 0)} times"
                        )
                else:
                    st.caption(
                        "（僅顯示使用量總數，未啟用內容檢視權限）"
                        if lang == "zh"
                        else "(Only aggregate usage visible; content access disabled.)"
                    )

            st.markdown("---")


def admin_dashboard():
    lang = st.session_state.get("lang", "zh")
    st.title("Admin Dashboard — Error-Free®")
    st.markdown("---")

    # 1) Guest accounts
    st.subheader("📌 Guest 帳號列表" if lang == "zh" else "📌 Guest accounts")
    guests = load_guest_accounts()
    if not guests:
        st.info("目前沒有 Guest 帳號。" if lang == "zh" else "No guest accounts yet.")
    else:
        for email, acc in guests.items():
            st.markdown(
                f"**{email}** — password: `{acc.get('password')}` (role: {acc.get('role')})"
            )
            st.markdown("---")

    # 2) Guest document usage
    st.subheader("📁 Guest 文件使用狀況" if lang == "zh" else "📁 Guest document usage")
    doc_tracking = load_doc_tracking()
    if not doc_tracking:
        st.info(
            "尚無 Guest 上傳記錄。" if lang == "zh" else "No guest uploads recorded yet."
        )
    else:
        for email, docs in doc_tracking.items():
            st.markdown(
                f"**{email}** — 上傳文件數：{len(docs)} / 3"
                if lang == "zh"
                else f"**{email}** — uploaded documents: {len(docs)} / 3"
            )
            for d in docs:
                st.markdown(f"- {d}")
            st.markdown("---")

    # 3) Framework state in current session
    st.subheader(
        "🧩 模組分析與追問狀況 (Session-based)"
        if lang == "zh"
        else "🧩 Framework state (current session)"
    )
    fs = st.session_state.get("framework_states", {})
    if not fs:
        st.info("尚無 Framework 分析記錄" if lang == "zh" else "No framework analysis yet.")
    else:
        for fw_key, state in fs.items():
            fw_name = (
                FRAMEWORKS.get(fw_key, {}).get("name_zh", fw_key)
                if lang == "zh"
                else FRAMEWORKS.get(fw_key, {}).get("name_en", fw_key)
            )
            st.markdown(f"### ▶ {fw_name}")
            st.write(
                f"分析完成：{state.get('analysis_done')}"
                if lang == "zh"
                else f"Analysis done: {state.get('analysis_done')}"
            )
            st.write(
                f"追問次數：{len(state.get('followup_history', []))}"
                if lang == "zh"
                else f"Follow-up count: {len(state.get('followup_history', []))}"
            )
            st.write(
                f"已下載報告：{state.get('download_used')}"
                if lang == "zh"
                else f"Downloaded report: {state.get('download_used')}"
            )
            st.markdown("---")

    # 4) 公司使用量總覽（4A）
    st.subheader("🏢 公司使用量總覽" if lang == "zh" else "🏢 Company usage overview")
    companies = load_companies()
    usage_stats = load_usage_stats()

    if not companies:
        st.info("目前尚未建立任何公司。" if lang == "zh" else "No companies registered yet.")
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
                f"學生 / 使用者數：{len(users)}"
                if lang == "zh"
                else f"Users: {len(users)}"
            )
            st.write(
                f"總上傳文件數：{total_docs}"
                if lang == "zh"
                else f"Total uploaded documents: {total_docs}"
            )
            st.write(
                f"總分析次數：{total_analysis}"
                if lang == "zh"
                else f"Total analysis runs: {total_analysis}"
            )
            st.write(
                f"總追問次數：{total_followups}"
                if lang == "zh"
                else f"Total follow-ups: {total_followups}"
            )
            st.write(
                f"總下載次數：{total_downloads}"
                if lang == "zh"
                else f"Total downloads: {total_downloads}"
            )
            st.write(
                "content_access：" + ("啟用" if content_access else "關閉")
                if lang == "zh"
                else "content_access: " + ("enabled" if content_access else "disabled")
            )
            st.markdown("---")

    # 5) 公司權限設定（4C 控制開關）
    st.subheader("🔐 公司內容檢視權限設定" if lang == "zh" else "🔐 Company content access settings")
    if not companies:
        st.info("尚無公司可設定。" if lang == "zh" else "No companies to configure.")
    else:
        for code, entry in companies.items():
            label = f"{entry.get('company_name') or code} ({code})"
            key = f"content_access_{code}"
            current_val = entry.get("content_access", False)
            st.checkbox(
                label + (" — 可檢視學生分析使用量" if lang == "zh" else " — can view user usage details"),
                value=current_val,
                key=key,
            )

        if st.button(
            "儲存公司權限設定" if lang == "zh" else "Save company access settings"
        ):
            for code, entry in companies.items():
                key = f"content_access_{code}"
                new_val = bool(st.session_state.get(key, entry.get("content_access", False)))
                entry["content_access"] = new_val
                companies[code] = entry
            save_companies(companies)
            st.success("已更新公司權限設定。" if lang == "zh" else "Company settings updated.")


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
            "返回分析頁面"
            if st.session_state.get("lang", "zh") == "zh"
            else "Back to analysis"
        ):
            st.session_state.show_admin = False
            save_state_to_disk()
            st.rerun()
        return True
    return False


# =========================
# Main app
# =========================


def language_selector():
    """Top-level language toggle: English (on top) / 中文 (below)."""
    current = st.session_state.get("lang", "zh")
    index = 0 if current == "en" else 1
    choice = st.radio("Language / 語言", ("English", "中文"), index=index)
    st.session_state.lang = "en" if choice == "English" else "zh"


def main():
    st.set_page_config(
        page_title="Error-Free® Multi-framework Analyzer", layout="wide"
    )
    restore_state_from_disk()

    # 初始化 session
    defaults = [
        ("user_email", None),
        ("user_role", None),
        ("is_authenticated", False),
        ("lang", "zh"),
        ("usage_date", None),
        ("usage_count", 0),
        ("last_doc_text", ""),
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
            st.subheader("帳號資訊" if lang == "zh" else "Account")
            st.write(f"Email：{st.session_state.user_email}")
            if st.button("登出" if lang == "zh" else "Logout"):
                st.session_state.user_email = None
                st.session_state.user_role = None
                st.session_state.is_authenticated = False
                st.session_state.framework_states = {}
                st.session_state.last_doc_text = ""
                st.session_state.current_doc_id = None
                save_state_to_disk()
                st.rerun()
        else:
            st.subheader("尚未登入" if lang == "zh" else "Not Logged In")

    # ======= Login screen =======
    if not st.session_state.is_authenticated:
        lang = st.session_state.lang

        title = (
            "Error-Free® 多框架文件分析"
            if lang == "zh"
            else "Error-Free® Multi-framework Document Analyzer"
        )
        st.title(title)
        st.markdown("---")

        # 登入說明
        if lang == "zh":
            st.markdown(
                "- 上方為內部員工 / 會員登入。\n"
                "- 中間為「公司管理者」（企業端窗口）登入 / 註冊。\n"
                "- 下方為學生 / 客戶的 Guest 試用登入 / 註冊。"
            )
        else:
            st.markdown(
                "- Top: internal Error-Free employees / members.\n"
                "- Middle: **Company Admins** for each client company.\n"
                "- Bottom: students / end-users using **Guest trial accounts**."
            )

        st.markdown("---")

        # 1. 內部員工 / 會員登入
        st.markdown(
            "### 內部員工 / 會員登入"
            if lang == "zh"
            else "### Internal Employee / Member Login"
        )
        emp_email = st.text_input("Email", key="emp_email")
        emp_pw = st.text_input(
            "密碼" if lang == "zh" else "Password",
            type="password",
            key="emp_pw",
        )
        if st.button("登入" if lang == "zh" else "Login", key="emp_login_btn"):
            account = ACCOUNTS.get(emp_email)
            if account and account["password"] == emp_pw:
                st.session_state.user_email = emp_email
                st.session_state.user_role = account["role"]
                st.session_state.is_authenticated = True
                save_state_to_disk()
                st.rerun()
            else:
                st.error(
                    "帳號或密碼錯誤"
                    if lang == "zh"
                    else "Invalid email or password"
                )

        st.markdown("---")

        # 2. 公司管理者註冊 － 公司管理者登入（同一橫排）
        st.markdown(
            "### 公司管理者（企業窗口）"
            if lang == "zh"
            else "### Company Admin (Client-side)"
        )
        col_ca_signup, col_ca_login = st.columns(2)

        # 公司管理者註冊
        with col_ca_signup:
            st.markdown("**公司管理者註冊**" if lang == "zh" else "**Company Admin Signup**")
            ca_new_email = st.text_input(
                "管理者註冊 Email" if lang == "zh" else "Admin signup email",
                key="ca_new_email",
            )
            ca_new_pw = st.text_input(
                "設定管理者密碼" if lang == "zh" else "Set admin password",
                type="password",
                key="ca_new_pw",
            )
            ca_company_code = st.text_input(
                "公司代碼 Company Code", key="ca_company_code"
            )

            if st.button(
                "建立管理者帳號"
                if lang == "zh"
                else "Create Company Admin Account",
                key="ca_signup_btn",
            ):
                if not ca_new_email or not ca_new_pw or not ca_company_code:
                    st.error(
                        "請完整填寫管理者註冊資訊"
                        if lang == "zh"
                        else "Please fill all admin signup fields"
                    )
                else:
                    companies = load_companies()
                    guests = load_guest_accounts()
                    if ca_company_code not in companies:
                        st.error(
                            "公司代碼不存在，請先向系統管理員建立公司"
                            if lang == "zh"
                            else "Company code not found. Please ask the system admin to create it."
                        )
                    elif ca_new_email in ACCOUNTS or ca_new_email in guests:
                        st.error(
                            "此 Email 已被使用"
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

                        entry = companies.get(ca_company_code, {})
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
                            "公司管理者帳號已建立"
                            if lang == "zh"
                            else "Company admin account created"
                        )

        # 公司管理者登入
        with col_ca_login:
            st.markdown("**公司管理者登入**" if lang == "zh" else "**Company Admin Login**")
            ca_email = st.text_input(
                "管理者 Email" if lang == "zh" else "Admin Email",
                key="ca_email",
            )
            ca_pw = st.text_input(
                "管理者密碼" if lang == "zh" else "Admin Password",
                type="password",
                key="ca_pw",
            )
            if st.button(
                "管理者登入" if lang == "zh" else "Login as Company Admin",
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
                        "管理者帳號或密碼錯誤"
                        if lang == "zh"
                        else "Invalid company admin credentials"
                    )

        st.markdown("---")

        # 3. Guest 註冊 － Guest 登入（同一橫排）
        st.markdown("### Guest 試用帳號" if lang == "zh" else "### Guest Trial Accounts")
        col_guest_signup, col_guest_login = st.columns(2)

        # Guest 註冊
        with col_guest_signup:
            st.markdown("**Guest 試用註冊**" if lang == "zh" else "**Guest Signup**")
            new_guest_email = st.text_input(
                "註冊 Email" if lang == "zh" else "Email for signup",
                key="new_guest_email",
            )
            guest_company_code = st.text_input(
                "公司代碼 Company Code" if lang == "zh" else "Company Code",
                key="guest_company_code",
            )

            if st.button(
                "取得 Guest 密碼"
                if lang == "zh"
                else "Generate Guest Password",
                key="guest_signup_btn",
            ):
                if not new_guest_email:
                    st.error(
                        "請輸入 Email"
                        if lang == "zh"
                        else "Please enter an email"
                    )
                elif not guest_company_code:
                    st.error(
                        "請輸入公司代碼"
                        if lang == "zh"
                        else "Please enter your Company Code"
                    )
                else:
                    guests = load_guest_accounts()
                    companies = load_companies()
                    if guest_company_code not in companies:
                        st.error(
                            "公司代碼不存在，請向講師或公司窗口確認"
                            if lang == "zh"
                            else "Invalid Company Code. Please check with your instructor or admin."
                        )
                    elif new_guest_email in guests or new_guest_email in ACCOUNTS:
                        st.error(
                            "Email 已存在"
                            if lang == "zh"
                            else "Email already exists"
                        )
                    else:
                        pw = "".join(
                            secrets.choice("0123456789") for _ in range(8)
                        )
                        guests[new_guest_email] = {
                            "password": pw,
                            "role": "free",
                            "company_code": guest_company_code,
                        }
                        save_guest_accounts(guests)

                        entry = companies.get(guest_company_code, {})
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
                            f"Guest 帳號已建立！密碼：{pw}"
                            if lang == "zh"
                            else f"Guest account created! Password: {pw}"
                        )

        # Guest 登入
        with col_guest_login:
            st.markdown("**Guest 試用登入**" if lang == "zh" else "**Guest Login**")
            g_email = st.text_input("Guest Email", key="g_email")
            g_pw = st.text_input(
                "密碼" if lang == "zh" else "Password",
                type="password",
                key="g_pw",
            )
            if st.button(
                "登入 Guest" if lang == "zh" else "Login as Guest",
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
                        "帳號或密碼錯誤"
                        if lang == "zh"
                        else "Invalid guest credentials"
                    )

        return  # login page end

    # ======= Main app (logged in) =======
    if admin_router():
        return

    lang = st.session_state.lang
    st.title(
        "Error-Free® 多框架 AI 文件分析"
        if lang == "zh"
        else "Error-Free® Multi-framework Analyzer"
    )
    st.markdown("---")

    user_email = st.session_state.user_email
    user_role = st.session_state.user_role
    is_guest = user_role == "free"
    model_name = resolve_model_for_user(user_role)

    # Step 1: upload
    st.subheader("步驟一：上傳文件" if lang == "zh" else "Step 1: Upload Document")
    uploaded = st.file_uploader(
        "請上傳 PDF / DOCX / TXT / 圖片"
        if lang == "zh"
        else "Upload PDF / DOCX / TXT / Image",
        type=["pdf", "docx", "txt", "jpg", "jpeg", "png"],
    )

    if uploaded is not None:
        doc_text = read_file_to_text(uploaded)
        if doc_text:
            if is_guest:
                docs = doc_tracking.get(user_email, [])
                if len(docs) >= 3 and st.session_state.current_doc_id not in docs:
                    st.error(
                        "試用帳號最多上傳 3 份文件"
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
                    save_state_to_disk()
            else:
                st.session_state.current_doc_id = (
                    f"doc_{datetime.datetime.now().timestamp()}"
                )
                st.session_state.last_doc_text = doc_text
                save_state_to_disk()

    # Step 2: select framework
    st.subheader("步驟二：選擇分析框架" if lang == "zh" else "Step 2: Select Framework")
    if not FRAMEWORKS:
        st.error(
            "尚未在 frameworks.json 中定義任何框架。"
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
        "選擇框架" if lang == "zh" else "Select framework",
        fw_labels,
        index=fw_labels.index(current_label) if current_label in fw_labels else 0,
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

    # Step 3: run analysis
    st.subheader("步驟三：執行分析" if lang == "zh" else "Step 3: Run Analysis")
    can_run = not current_state["analysis_done"]

    if can_run:
        run_btn = st.button(
            "開始分析" if lang == "zh" else "Run analysis", key="run_analysis_btn"
        )
    else:
        run_btn = False
        st.info(
            "此框架已完成一次分析"
            if lang == "zh"
            else "Analysis already completed for this framework."
        )

    # 只有非 Guest 才能 Reset
    if not is_guest:
        if st.button("重置（新文件）" if lang == "zh" else "Reset document"):
            st.session_state.framework_states = {}
            st.session_state.last_doc_text = ""
            st.session_state.current_doc_id = None
            save_state_to_disk()
            st.rerun()

    if run_btn and can_run:
        if not st.session_state.last_doc_text:
            st.error(
                "請先上傳文件" if lang == "zh" else "Please upload a document first."
            )
        else:
            with st.spinner("分析中..." if lang == "zh" else "Running analysis..."):
                analysis_text = run_llm_analysis(
                    selected_key,
                    lang,
                    st.session_state.last_doc_text,
                    model_name,
                )
            current_state["analysis_done"] = True
            current_state["analysis_output"] = clean_report_text(analysis_text)
            current_state["followup_history"] = []
            save_state_to_disk()
            record_usage(user_email, selected_key, "analysis")
            st.success("分析完成！" if lang == "zh" else "Analysis completed!")

    # Step 4: show all framework results
    any_analysis = False
    for fw_key in FRAMEWORKS.keys():
        state = framework_states.get(fw_key)
        if not state or not state.get("analysis_output"):
            continue

        any_analysis = True
        st.markdown("---")
        fw = FRAMEWORKS[fw_key]
        fw_name = fw["name_zh"] if lang == "zh" else fw["name_en"]
        st.subheader(
            ("⭐ " if fw_key == selected_key else "") + fw_name + "：分析與問答"
            if lang == "zh"
            else ("⭐ " if fw_key == selected_key else "") + fw_name + ": Analysis & Q&A"
        )

        # 分析結果
        st.markdown("#### 分析結果" if lang == "zh" else "#### Analysis result")
        st.markdown(state["analysis_output"])

        # Q&A 歷史
        st.markdown(
            "#### 後續提問（Q&A）" if lang == "zh" else "#### Follow-up Q&A history"
        )
        if state["followup_history"]:
            for i, (q, a) in enumerate(state["followup_history"], start=1):
                st.markdown(f"**Q{i}:** {q}")
                st.markdown(f"**A{i}:** {a}")
                st.markdown("---")
        else:
            st.info("尚無追問" if lang == "zh" else "No follow-up questions yet.")

        # Download 區塊
        st.markdown("##### 下載報告" if lang == "zh" else "##### Download report")
        st.caption(
            "報告只包含分析與 Q&A，不含原始文件。"
            if lang == "zh"
            else "Report includes analysis + Q&A only (no original document)."
        )

        if is_guest and state.get("download_used"):
            st.error(
                "已達下載次數上限（1 次）"
                if lang == "zh"
                else "Download limit reached (1 time)."
            )
        else:
            report = build_full_report(lang, fw_key, state)
            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            with st.expander("Download"):
                fmt = st.radio(
                    "選擇格式" if lang == "zh" else "Select format",
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
                            f"PPTX 匯出失敗：{e}"
                            if lang == "zh"
                            else f"PPTX export failed: {e}"
                        )
                        data = b""
                        mime = "application/octet-stream"
                        ext = "pptx"

                if data:
                    clicked = st.download_button(
                        "開始下載" if lang == "zh" else "Download",
                        data=data,
                        file_name=f"errorfree_{fw_key}_{now_str}.{ext}",
                        mime=mime,
                        key=f"dl_{fw_key}_{ext}",
                    )
                    if clicked:
                        state["download_used"] = True
                        save_state_to_disk()
                        record_usage(user_email, fw_key, "download")

    # Step 5: global follow-up area（針對目前選中的框架）
    if any_analysis:
        st.markdown("---")
        st.subheader("後續提問" if lang == "zh" else "Follow-up questions")

        curr_state = framework_states[selected_key]
        if is_guest and len(curr_state["followup_history"]) >= 3:
            st.error(
                "已達追問上限（3 次）"
                if lang == "zh"
                else "Follow-up limit reached (3 times)."
            )
        else:
            extra_file = st.file_uploader(
                "上傳附加文件（可選）"
                if lang == "zh"
                else "Upload supplementary file (optional)",
                type=["pdf", "docx", "txt", "jpg", "jpeg", "png"],
                key=f"extra_{selected_key}",
            )
            extra_text = read_file_to_text(extra_file) if extra_file else ""

            prompt = st.chat_input(
                f"針對 {FRAMEWORKS[selected_key]['name_zh']} 的追問"
                if lang == "zh"
                else f"Ask a follow-up about {FRAMEWORKS[selected_key]['name_en']}"
            )
            if prompt:
                with st.spinner("思考中..." if lang == "zh" else "Thinking..."):
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
