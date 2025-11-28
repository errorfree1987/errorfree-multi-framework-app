# =========================
# PART 0 — Company Multi-Tenant Support (New)
# =========================

import json
from pathlib import Path

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
# app_final.py (Part 1/?)
# Core imports, constants, and basic account setup
# =========================

import os
import json
import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from io import BytesIO
import secrets

import streamlit as st
import pdfplumber
from docx import Document
from openai import OpenAI
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# OpenAI client (global)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# =========================
# ACCOUNT CONFIGURATION
# =========================
# High-level internal accounts (no limits, GPT-5.1)
ACCOUNTS = {
    "admin@errorfree.com": {"password": "1111", "role": "admin"},
    "dr.chiu@errorfree.com": {"password": "2222", "role": "pro"},
    "test@errorfree.com": {"password": "3333", "role": "pro"},
}

# Guest accounts stored dynamically
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
# PART 2 — Framework definitions + State persistence
# =========================

# Error-Free® analysis frameworks
FRAMEWORKS: Dict[str, Dict] = {
    "omission": {
        "name_zh": "Error-Free® 遺漏錯誤檢查框架",
        "name_en": "Error-Free® Omission Error Check Framework",
        "wrapper_zh": (
            "你是一位 Error-Free® 遺漏錯誤檢查專家。"
            "請分析文件中可能遺漏的重要內容、條件、假設、角色、步驟、風險或例外，"
            "並說明遺漏的影響與具體補強建議，最後整理成條列與一個簡單的 Markdown 表格。"
        ),
        "wrapper_en": (
            "You are an Error-Free® omission error expert. "
            "Review the document, find important missing information or conditions, "
            "explain the impact, and give concrete suggestions, plus a simple Markdown table."
        ),
    },
    "technical": {
        "name_zh": "Error-Free® 技術風險檢查框架",
        "name_en": "Error-Free® Technical Risk Check Framework",
        "wrapper_zh": (
            "你是一位 Error-Free® 技術風險檢查專家。"
            "請從技術假設、邊界條件、相容性、安全性、可靠度與單點失敗等面向分析文件，"
            "列出技術風險、風險等級與實務改善建議，並以 Markdown 表格整理重點。"
        ),
        "wrapper_en": (
            "You are an Error-Free® technical risk review expert. "
            "Analyze the document for technical assumptions, edge cases, compatibility, "
            "safety and single points of failure. List risks, risk level and mitigation, "
            "and provide a summary Markdown table."
        ),
    },
}

# =========================
# STATE PERSISTENCE
# =========================
STATE_FILE = Path("user_state.json")

# Tracks uploaded documents per user (for guest limits)
DOC_TRACK_FILE = Path("user_docs.json")


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

    for key, value in data.items():
        if key not in st.session_state:
            st.session_state[key] = value

# =========================
# PART 3 — File reading + LLM helpers + Report building
# =========================

def read_file_to_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".pdf"):
            text_pages = []
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    t = page.extract_text() or ""
                    text_pages.append(t)
            return "
".join(text_pages)
        elif name.endswith(".docx"):
            doc = Document(uploaded_file)
            return "
".join(p.text for p in doc.paragraphs)
        elif name.endswith(".txt"):
            return uploaded_file.read().decode("utf-8", errors="ignore")
        else:
            return ""
    except Exception as e:
        return f"[讀取檔案時發生錯誤: {e}]"


# =========================
# Force models:
#   - internal accounts → GPT‑5.1
#   - guest accounts → GPT‑4.1‑mini
# =========================

def resolve_model_for_user(role: str) -> str:
    # role = admin / pro → full 5.1
    if role in ["admin", "pro"]:
        return "gpt-5.1"
    # guest accounts
    return "gpt-4.1-mini"


# =========================
# LLM logic
# =========================

def run_llm_analysis(framework_key: str, language: str, document_text: str, model_name: str) -> str:
    fw = FRAMEWORKS[framework_key]
    system_prompt = fw["wrapper_zh"] if language == "zh" else fw["wrapper_en"]
    user_prompt = ("以下是要分析的文件內容：

" if language == "zh" else "Here is the document to analyze:

") + document_text

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
    fw = FRAMEWORKS[framework_key]

    # Fully ASCII → avoid syntax issues
    if language == "zh":
        system_prompt = (
            "You are an Error-Free consultant familiar with framework: "
            + fw["name_zh"]
            + ". You already produced a full analysis. Now answer follow-up questions based on the original document and previous analysis. Focus on extra insights, avoid repeating the full report."
        )
    else:
        system_prompt = (
            "You are an Error-Free consultant for framework: "
            + fw["name_en"]
            + ". You already produced a full analysis. Answer follow-up questions based on document + previous analysis, without recreating the full report."
        )

    doc_excerpt = document_text[:8000]
    analysis_excerpt = analysis_output[:8000]
    extra_excerpt = extra_text[:4000] if extra_text else ""

    blocks = [
        "Original document excerpt:
" + doc_excerpt,
        "Previous analysis excerpt:
" + analysis_excerpt,
        "User question:
" + user_question,
    ]
    if extra_excerpt:
        blocks.append("Extra reference:
" + extra_excerpt)

    user_content = "

".join(blocks)

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
# REPORT FORMATTING
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
    fw = FRAMEWORKS[framework_key]
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    email = st.session_state.get("user_email", "unknown")

    if lang == "zh":
        header = [
            "Error-Free® 多框架 AI 文件分析報告（分析 + Q&A）",
            f"產生時間：{now}",
            f"使用者帳號：{email}",
            f"使用框架：{fw['name_zh']}",
            "",
            "==============================",
            "一、分析結果",
            "==============================",
            analysis_output,
        ]
        if followups:
            header.append("")
            header.append("==============================")
            header.append("二、後續問答（Q&A）")
            header.append("==============================")
            for i, (q, a) in enumerate(followups, start=1):
                header.append(f"[Q{i}] {q}")
                header.append(f"[A{i}] {a}")
                header.append("")
    else:
        header = [
            "Error-Free® Multi-framework AI Report (Analysis + Q&A)",
            f"Generated: {now}",
            f"User: {email}",
            f"Framework: {fw['name_en']}",
            "",
            "==============================",
            "1. Analysis",
            "==============================",
            analysis_output,
        ]
        if followups:
            header.append("")
            header.append("==============================")
            header.append("2. Follow-up Q&A")
            header.append("==============================")
            for i, (q, a) in enumerate(followups, start=1):
                header.append(f"[Q{i}] {q}")
                header.append(f"[A{i}] {a}")
                header.append("")

    return clean_report_text("
".join(header))


def build_docx_bytes(text: str) -> bytes:
    doc = Document()
    for line in text.split("
"):
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

    for line in text.split("
"):
        c.drawString(40, y, line[:1000])
        y -= 14
        if y < 40:
            c.showPage()
            y = height - 40
    c.save()
    buf.seek(0)
    return buf.getvalue()

# =========================
# PART 4 — Main Application (Login, Upload, Framework, Limits)
# =========================

def main():
    st.set_page_config(page_title="Error-Free® Multi-framework Analyzer", layout="wide")

    # Restore session from disk
    restore_state_from_disk()

    # Initialize missing session fields
    for key, default in [
        ("user_email", None),
        ("user_role", None),
        ("is_authenticated", False),
        ("lang", "zh"),
        ("usage_date", None),
        ("usage_count", 0),
        ("last_doc_text", ""),
        ("framework_states", {}),
        ("selected_framework_key", list(FRAMEWORKS.keys())[0]),
        ("current_doc_id", None),
        ("company_code", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # Load doc tracking
    doc_tracking = load_doc_tracking()

    # =========================
    # SIDEBAR (only basic info — no model details)
    # =========================
    with st.sidebar:
        # ============= Admin Dashboard Entry Button =============
        if st.session_state.is_authenticated and st.session_state.user_role in ["admin", "pro", "company_admin"]:
            if st.button("管理後台 Admin Dashboard"):
                st.session_state.show_admin = True
                st.rerun()
        # ========================================================

        if st.session_state.is_authenticated:
            st.subheader("帳號資訊" if st.session_state.lang == "zh" else "Account")
            st.write(f"Email：{st.session_state.user_email}")

            if st.button("登出" if st.session_state.lang == "zh" else "Logout"):
                st.session_state.user_email = None
                st.session_state.user_role = None
                st.session_state.is_authenticated = False
                st.session_state.framework_states = {}
                st.session_state.last_doc_text = ""
                save_state_to_disk()
                st.rerun()
        else:
            st.subheader("尚未登入" if st.session_state.lang == "zh" else "Not Logged In")

    # =========================
    # LOGIN SCREEN
    # =========================
    if not st.session_state.is_authenticated:
        lang = st.session_state.lang

        st.title("Error-Free® 多框架文件分析" if lang == "zh" else "Error-Free® Multi-framework Analyzer")
        st.markdown("---")

        # Two sections: (1) 員工/會員 (2) Guest
        col_emp, col_guest = st.columns(2)

        # =============== EMPLOYEE / MEMBER LOGIN ===============
        with col_emp:
            st.markdown("### 內部員工 / 會員登入" if lang == "zh" else "### Employee / Member Login")

            emp_email = st.text_input("Email", key="emp_email")
            emp_pw = st.text_input("密碼" if lang == "zh" else "Password", type="password", key="emp_pw")
            if st.button("登入" if lang == "zh" else "Login", key="emp_login_btn"):
                account = ACCOUNTS.get(emp_email)
                if account and account["password"] == emp_pw:
                    st.session_state.user_email = emp_email
                    st.session_state.user_role = account["role"]  # admin / pro → GPT-5.1
                    st.session_state.is_authenticated = True
                    st.session_state.login_success = True
                    save_state_to_disk()
                    st.rerun()
                else:
                    st.error("帳號或密碼錯誤" if lang == "zh" else "Invalid email or password")

        # =============== GUEST LOGIN / SIGNUP ===============
                # 公司管理者登入區
        st.markdown("---")
        st.markdown("### 公司管理者登入" if lang == "zh" else "### Company Admin Login")
        ca_email = st.text_input("管理者 Email" if lang == "zh" else "Admin Email", key="ca_email")
        ca_pw = st.text_input("管理者密碼" if lang == "zh" else "Admin Password", type="password", key="ca_pw")
        if st.button("管理者登入" if lang == "zh" else "Login as Company Admin", key="ca_login_btn"):
            guests = load_guest_accounts()
            acc = guests.get(ca_email)
            if acc and acc.get("password") == ca_pw and acc.get("role") == "company_admin":
                st.session_state.user_email = ca_email
                st.session_state.user_role = "company_admin"
                st.session_state.company_code = acc.get("company_code")
                st.session_state.is_authenticated = True
                st.session_state.login_success = True
                save_state_to_disk()
                st.rerun()
            else:
                st.error("管理者帳號或密碼錯誤" if lang == "zh" else "Invalid company admin credentials")

        st.markdown("---")
        st.markdown("### 公司管理者註冊" if lang == "zh" else "### Company Admin Signup")
        ca_new_email = st.text_input("管理者註冊 Email" if lang == "zh" else "Admin signup email", key="ca_new_email")
        ca_new_pw = st.text_input("設定管理者密碼" if lang == "zh" else "Set admin password", type="password", key="ca_new_pw")
        ca_company_code = st.text_input("公司代碼 Company Code", key="ca_company_code")

        if st.button("建立管理者帳號" if lang == "zh" else "Create Company Admin Account", key="ca_signup_btn"):
            if not ca_new_email or not ca_new_pw or not ca_company_code:
                st.error("請完整填寫管理者註冊資訊" if lang == "zh" else "Please fill all admin signup fields")
            else:
                companies = load_companies()
                guests = load_guest_accounts()
                if ca_company_code not in companies:
                    st.error("公司代碼不存在，請先向系統管理員建立公司" if lang == "zh" else "Company code not found. Please ask the system admin to create it.")
                elif ca_new_email in ACCOUNTS or ca_new_email in guests:
                    st.error("此 Email 已被使用" if lang == "zh" else "This email is already in use")
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
                    st.success("公司管理者帳號已建立" if lang == "zh" else "Company admin account created")

        with col_guest:
            st.markdown("### Guest 試用登入" if lang == "zh" else "### Guest Login")

            g_email = st.text_input("Guest Email", key="g_email")
            g_pw = st.text_input("密碼" if lang == "zh" else "Password", type="password", key="g_pw")
            if st.button("登入 Guest" if lang == "zh" else "Login as Guest", key="guest_login_btn"):
                guests = load_guest_accounts()
                g_acc = guests.get(g_email)
                if g_acc and g_acc.get("password") == g_pw:
                    # 綁定 guest 所屬公司代碼
                    st.session_state.company_code = g_acc.get("company_code")
                    st.session_state.user_email = g_email
                    st.session_state.user_role = "free"  # guest always free
                    st.session_state.is_authenticated = True
                    st.session_state.login_success = True
                    save_state_to_disk()
                    st.rerun()
                else:
                    st.error("帳號或密碼錯誤" if lang == "zh" else "Invalid guest credentials")

            st.markdown("---")
            st.markdown("### Guest 試用註冊" if lang == "zh" else "### Guest Signup")

            new_guest_email = st.text_input("註冊 Email" if lang == "zh" else "Email for signup", key="new_guest_email")
            guest_company_code = st.text_input("公司代碼 Company Code" if lang == "zh" else "Company Code", key="guest_company_code")

            if st.button("取得 Guest 密碼" if lang == "zh" else "Generate Guest Password", key="guest_signup_btn"):
                if not new_guest_email:
                    st.error("請輸入 Email" if lang == "zh" else "Please enter an email")
                elif not guest_company_code:
                    st.error("請輸入公司代碼" if lang == "zh" else "Please enter your Company Code")
                else:
                    guests = load_guest_accounts()
                    companies = load_companies()
                    if guest_company_code not in companies:
                        st.error("公司代碼不存在，請向講師或公司窗口確認" if lang == "zh" else "Invalid Company Code. Please check with your instructor or admin.")
                    elif new_guest_email in guests or new_guest_email in ACCOUNTS:
                        st.error("Email 已存在" if lang == "zh" else "Email already exists")
                    else:
                        pw = "".join(secrets.choice("0123456789") for _ in range(8))
                        guests[new_guest_email] = {"password": pw, "role": "free", "company_code": guest_company_code}
                        save_guest_accounts(guests)

                        # 把使用者綁定到公司 users 清單
                        company_entry = companies.get(guest_company_code, {})
                        users = company_entry.get("users", [])
                        if new_guest_email not in users:
                            users.append(new_guest_email)
                        company_entry["users"] = users
                        if "company_name" not in company_entry:
                            company_entry["company_name"] = ""
                        if "content_access" not in company_entry:
                            company_entry["content_access"] = False
                        companies[guest_company_code] = company_entry
                        save_companies(companies)

                        st.success(
                            f"Guest 帳號已建立！密碼：{pw}" if lang == "zh" else f"Guest account created! Password: {pw}"
                        )
        return

    # 若目前在管理後台模式，優先顯示後台
    if admin_router():
        return

    # =========================
    # MAIN APP (LOGGED IN)
    # =========================
    lang = st.session_state.lang
    st.title("Error-Free® 多框架 AI 文件分析" if lang == "zh" else "Error-Free® Multi-framework Analyzer")
    st.markdown("---")

    user_email = st.session_state.user_email
    user_role = st.session_state.user_role
    is_guest = (user_role == "free")

    # Determine model
    model_name = resolve_model_for_user(user_role)

    # =========================
    # STEP 1: UPLOAD DOCUMENT (with guest limit)
    # =========================
    st.subheader("步驟一：上傳文件" if lang == "zh" else "Step 1: Upload Document")

    uploaded = st.file_uploader(
        "請上傳 PDF / DOCX / TXT" if lang == "zh" else "Upload PDF / DOCX / TXT",
        type=["pdf", "docx", "txt"],
    )

    if uploaded is not None:
        # Assign doc_id
        doc_text = read_file_to_text(uploaded)
        if doc_text:
            if is_guest:
                # Guest limit: max 3 documents
                docs = doc_tracking.get(user_email, [])
                if len(docs) >= 3 and st.session_state.current_doc_id not in docs:
                    st.error("試用帳號最多上傳 3 份文件" if lang == "zh" else "Trial accounts may upload up to 3 documents only")
                else:
                    # Register new doc if not existing
                    if st.session_state.current_doc_id not in docs:
                        new_id = f"doc_{datetime.datetime.now().timestamp()}"
                        docs.append(new_id)
                        doc_tracking[user_email] = docs
                        st.session_state.current_doc_id = new_id
                        save_doc_tracking(doc_tracking)
                    st.session_state.last_doc_text = doc_text
                    save_state_to_disk()
            else:
                # Unlimited for internal users
                st.session_state.current_doc_id = f"doc_{datetime.datetime.now().timestamp()}"
                st.session_state.last_doc_text = doc_text
                save_state_to_disk()

    # =========================
    # STEP 2: CHOOSE FRAMEWORK
    # =========================
    st.subheader("步驟二：選擇分析框架" if lang == "zh" else "Step 2: Select Framework")

    fw_keys = list(FRAMEWORKS.keys())
    fw_labels = [FRAMEWORKS[k]["name_zh"] if lang == "zh" else FRAMEWORKS[k]["name_en"] for k in fw_keys]
    k2l = dict(zip(fw_keys, fw_labels))
    l2k = dict(zip(fw_labels, fw_keys))

    current_fw = st.session_state.selected_framework_key
    selected_label = k2l[current_fw]

    new_label = st.selectbox("選擇框架" if lang == "zh" else "Select framework", fw_labels, index=fw_labels.index(selected_label))
    new_key = l2
    new_key = l2k[new_label]
    st.session_state.selected_framework_key = new_key

    # Ensure framework state exists
    framework_states = st.session_state.framework_states
    if new_key not in framework_states:
        framework_states[new_key] = {
            "analysis_done": False,
            "analysis_output": "",
            "followup_history": [],
            "download_used": False,
        }

    save_state_to_disk()
    current_state = framework_states[new_key]

    st.markdown("---")

    # =========================
    # STEP 3: RUN ANALYSIS (guest: limit 1 per framework)
    # =========================
    st.subheader("步驟三：執行分析" if lang == "zh" else "Step 3: Run Analysis")

    # Button
    can_run = not current_state["analysis_done"]

    if can_run:
        run_btn = st.button("開始分析" if lang == "zh" else "Run Analysis", key="run_analysis_btn")
    else:
        run_btn = False
        st.info("此框架已完成一次分析" if lang == "zh" else "Analysis already completed for this framework")

    # Reset button for internal users only
    if not is_guest:
        if st.button("重置（新文件）" if lang == "zh" else "Reset Document", key="reset_btn"):
            st.session_state.framework_states = {}
            st.session_state.last_doc_text = ""
            st.session_state.current_doc_id = None
            save_state_to_disk()
            st.rerun()

    # Execute analysis
    if run_btn and can_run:
        if not st.session_state.last_doc_text:
            st.error("請先上傳文件" if lang == "zh" else "Please upload a document first")
        else:
            with st.spinner("分析中..." if lang == "zh" else "Running analysis..."):
                analysis_text = run_llm_analysis(
                    new_key,
                    lang,
                    st.session_state.last_doc_text,
                    model_name,
                )
            current_state["analysis_done"] = True
            current_state["analysis_output"] = analysis_text
            current_state["followup_history"] = []
            save_state_to_disk()
            st.success("分析完成！" if lang == "zh" else "Analysis completed!")

    # =========================
    # STEP 4 — SHOW ALL FRAMEWORK RESULTS
    # =========================

    any_analysis = any(s.get("analysis_output") for s in framework_states.values())

    for fw_key in FRAMEWORKS.keys():
        state = framework_states.get(fw_key)
        if not state or not state.get("analysis_output"):
            continue

        st.markdown("---")

        fw = FRAMEWORKS[fw_key]
        if lang == "zh":
            title = f"{fw['name_zh']}：分析與問答"
        else:
            title = f"{fw['name_en']}: Analysis & Q&A"

        # Highlight selected framework
        if fw_key == new_key:
            st.subheader("⭐ " + title)
        else:
            st.subheader(title)

        # Analysis
        st.markdown("#### 分析結果" if lang == "zh" else "#### Analysis Result")
        st.markdown(state["analysis_output"])

        # Q&A history
        st.markdown("#### 後續提問（Q&A）" if lang == "zh" else "#### Follow-up Q&A")
        if state["followup_history"]:
            for i, (q, a) in enumerate(state["followup_history"], start=1):
                st.markdown(f"**Q{i}:** {q}")
                st.markdown(f"**A{i}:** {a}")
                st.markdown("---")
        else:
            st.info("尚無追問" if lang == "zh" else "No follow-up questions yet")

        # =========================
        # DOWNLOAD REPORT (guest: 1 time only)
        # =========================
        st.markdown("##### 下載報告" if lang == "zh" else "##### Download Report")
        st.caption("報告僅包含分析與Q&A，不含原始文件" if lang == "zh" else "Report includes analysis + Q&A only")

        if is_guest and state.get("download_used"):
            st.error("已達下載次數上限（1 次）" if lang == "zh" else "Download limit reached (1 time)")
        else:
            report = build_full_report(lang, fw_key, state)
            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            fmt = st.selectbox(
                "格式" if lang == "zh" else "Format",
                ["TXT", "Word", "PDF"],
                key=f"fmt_{fw_key}",
            )

            if fmt == "TXT":
                data = report.encode("utf-8")
                mime = "text/plain"
                ext = "txt"
            elif fmt == "Word":
                data = build_docx_bytes(report)
                mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                ext = "docx"
            else:
                data = build_pdf_bytes(report)
                mime = "application/pdf"
                ext = "pdf"

            if st.download_button(
                "下載" if lang == "zh" else "Download",
                data=data,
                file_name=f"errorfree_{fw_key}_{now_str}.{ext}",
                mime=mime,
                key=f"dl_{fw_key}",
            ):
                state["download_used"] = True
                save_state_to_disk()

    # =========================
    # FOLLOW-UP CHAT (limit guest: 3 times)
    # =========================
    if any_analysis:
        st.markdown("---")
        st.subheader("後續提問" if lang == "zh" else "Follow-up Question")

        curr_state = framework_states[new_key]

        # Guest limit: max 3 follow-ups
        if is_guest and len(curr_state["followup_history"]) >= 3:
            st.error("已達追問上限（3 次）" if lang == "zh" else "Follow-up limit reached (3 times)")
        else:
            extra_file = st.file_uploader(
                "上傳附加文件（可選）" if lang == "zh" else "Upload supplementary file (optional)",
                type=["pdf", "docx", "txt"],
                key=f"extra_{new_key}",
            )

            extra_text = read_file_to_text(extra_file) if extra_file else ""

            prompt = st.chat_input(
                f"針對 {FRAMEWORKS[new_key]['name_zh']} 的追問" if lang == "zh" else f"Ask a follow-up for {FRAMEWORKS[new_key]['name_en']}"
            )

            if prompt:
                with st.spinner("思考中..." if lang == "zh" else "Thinking..."):
                    answer = run_followup_qa(
                        new_key,
                        lang,
                        st.session_state.last_doc_text,
                        curr_state["analysis_output"],
                        prompt,
                        model_name,
                        extra_text,
                    )
                curr_state["followup_history"].append((prompt, answer))
                save_state_to_disk()
                st.rerun()

    save_state_to_disk()


# =========================
# ADMIN DASHBOARD (shown only to admin/pro internal accounts)
# =========================

def company_admin_dashboard():
    """Dashboard for company_admin role, scoped to a single company_code."""
    companies = load_companies()
    code = st.session_state.get("company_code")
    email = st.session_state.get("user_email")

    if not code or code not in companies:
        st.error("找不到公司代碼，請聯絡系統管理員" if st.session_state.get("lang", "zh") == "zh" else "Company code not found. Please contact system admin.")
        return

    entry = companies[code]
    admins = entry.get("admins", [])
    if email not in admins:
        st.error("您沒有此公司的管理者權限" if st.session_state.get("lang", "zh") == "zh" else "You are not an admin for this company.")
        return

    lang = st.session_state.get("lang", "zh")
    company_name = entry.get("company_name") or code
    content_access = entry.get("content_access", False)

    st.title("公司管理後台 - " + company_name if lang == "zh" else f"Company Admin Dashboard - {company_name}")
    st.markdown("---")

    st.subheader("公司資訊" if lang == "zh" else "Company Info")
    st.write(("公司代碼：" if lang == "zh" else "Company Code: ") + code)
    st.write(("可查看內容：" if lang == "zh" else "Can view content: ") + ("是" if content_access else "否" if lang == "zh" else ("Yes" if content_access else "No")))

    st.markdown("---")
    st.subheader("學生 / 使用者列表" if lang == "zh" else "Users in this company")

    users = entry.get("users", [])
    doc_tracking = load_doc_tracking()

    if not users:
        st.info("目前尚未有任何學生註冊" if lang == "zh" else "No users registered for this company yet.")
    else:
        for u in users:
            docs = doc_tracking.get(u, [])
            st.markdown(f"**{u}**")
            st.write(("上傳文件數：" if lang == "zh" else "Uploaded documents: ") + str(len(docs)))
            if content_access:
                st.caption("（未來可擴充檢視分析內容與 Q&A）" if lang == "zh" else "(Future: view analysis & Q&A content)")
            st.markdown("---")

# =========================
# ADMIN DASHBOARD (shown only to admin/pro internal accounts)
# =========================
# =========================

def admin_dashboard():
    st.title("Admin Dashboard — Error-Free®")
    st.markdown("---")

    st.subheader("📌 Guest 帳號列表")
    guests = load_guest_accounts()
    if not guests:
        st.info("目前沒有 Guest 帳號。")
    else:
        for email, acc in guests.items():
            st.markdown(f"**{email}** — password: `{acc.get('password')}`")
            st.markdown("---")

    st.subheader("📁 Guest 文件使用狀況")
    doc_tracking = load_doc_tracking()
    if not doc_tracking:
        st.info("尚無 Guest 上傳記錄。")
    else:
        for email, docs in doc_tracking.items():
            st.markdown(f"**{email}** — 上傳文件數：{len(docs)} / 3")
            for d in docs:
                st.markdown(f"- {d}")
            st.markdown("---")

    st.subheader("🧩 模組分析與追問狀況 (Session-based)")
    fs = st.session_state.get("framework_states", {})
    if not fs:
        st.info("尚無 Framework 分析記錄")
    else:
        for fw_key, state in fs.items():
            fw = FRAMEWORKS[fw_key]["name_zh"]
            st.markdown(f"### ▶ {fw}")
            st.write(f"分析完成：{state.get('analysis_done')}")
            st.write(f"追問次數：{len(state.get('followup_history', []))}")
            st.write(f"下載次數：1 / {1 if state.get('download_used') else 0}")
            st.markdown("---")

# Add admin dashboard routing
# Detect if current user wants dashboard
if "show_admin" not in st.session_state:
    st.session_state.show_admin = False


def admin_router():
    if st.session_state.show_admin:
        role = st.session_state.get("user_role")
        if role == "company_admin":
            company_admin_dashboard()
        else:
            admin_dashboard()
        if st.button("返回分析頁面"):
            st.session_state.show_admin = False
            st.rerun()
        return True
    return False

# =========================
# MAIN ENTRY
# =========================
if __name__ == "__main__":
    main()

