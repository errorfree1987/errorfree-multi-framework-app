"""
Error-Free® Multi-Tenant Trial Operations - Admin UI (MVP)

Phase B1.1: 登入/權限（Streamlit MVP 版本）

功能：
1. 簡單密碼保護（環境變數 ADMIN_PASSWORD）
2. Session 狀態管理
3. 登出功能
4. 記錄所有操作到 audit_events

技術棧：
- Streamlit
- Supabase (PostgreSQL)
- Railway 部署
"""

import streamlit as st
import os
import hashlib
import hmac
import secrets
import requests
from datetime import datetime
import json

# ==========================================
# 設定頁面配置（必須在最前面）
# ==========================================
st.set_page_config(
    page_title="Error-Free® Admin - Operations Dashboard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Supabase 連線設定
# ==========================================
def get_supabase_client():
    """取得 Supabase 連線資訊"""
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    service_key = os.getenv("SUPABASE_SERVICE_KEY", "").strip()
    
    if not supabase_url or not service_key:
        return None, None
    
    return supabase_url, service_key

def _log_audit_event(action: str, result: str, actor_email: str = "admin", 
                     tenant: str = "", deny_reason: str = "", context: dict = None):
    """
    記錄審計日誌到 audit_events 表
    
    Args:
        action: 操作類型（例如 'admin_login', 'admin_logout', 'tenant_created'）
        result: 結果（'success' / 'denied' / 'error'）
        actor_email: 操作者 email（預設 'admin'）
        tenant: 租戶 slug（選填）
        deny_reason: 拒絕原因（選填）
        context: JSON 上下文（選填）
    """
    supabase_url, service_key = get_supabase_client()
    if not supabase_url or not service_key:
        return
    
    endpoint = f"{supabase_url}/rest/v1/audit_events"
    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }
    
    payload = {
        "action": action,
        "tenant_slug": tenant or None,
        "email": actor_email,
        "result": result,
        "deny_reason": deny_reason or None,
        "context": context or {}
    }
    
    try:
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=5)
        # Best-effort：如果失敗也不中斷主流程
    except Exception:
        pass

# ==========================================
# Query Parameter 工具函數
# ==========================================
def get_query_param(key: str, default: str = "") -> str:
    """從 URL query parameters 獲取值"""
    try:
        params = st.query_params
        value = params.get(key, default)
        if isinstance(value, list):
            return value[0] if value else default
        return str(value) if value else default
    except:
        return default

def set_query_param(key: str, value: str):
    """設定 URL query parameter"""
    try:
        st.query_params[key] = value
    except:
        pass

def clear_query_params():
    """清除所有 query parameters"""
    try:
        st.query_params.clear()
    except:
        pass

# ==========================================
# 登入驗證函數
# ==========================================
def check_password():
    """
    檢查密碼是否正確
    使用環境變數 ADMIN_PASSWORD
    
    使用 URL query parameter 保持登入狀態（即使重新整理）
    """
    
    def generate_session_token(password: str) -> str:
        """生成安全的 session token"""
        # Token 格式：{timestamp}.{hash}
        timestamp = str(int(datetime.utcnow().timestamp()))
        combined = f"{password}{timestamp}"
        token_hash = hashlib.sha256(combined.encode()).hexdigest()[:32]
        return f"{timestamp}.{token_hash}"
    
    def verify_session_token(token: str, password: str) -> bool:
        """驗證 session token"""
        if not token or "." not in token:
            return False
        
        try:
            parts = token.split(".")
            if len(parts) != 2:
                return False
            
            timestamp_str, token_hash = parts
            
            # 驗證 token 格式
            if not timestamp_str.isdigit() or len(token_hash) != 32:
                return False
            
            # 重新計算 hash 驗證
            combined = f"{password}{timestamp_str}"
            expected_hash = hashlib.sha256(combined.encode()).hexdigest()[:32]
            
            return hmac.compare_digest(token_hash, expected_hash)
        except:
            return False
    
    def password_entered():
        """驗證輸入的密碼"""
        admin_password = os.getenv("ADMIN_PASSWORD", "").strip()
        
        if not admin_password:
            st.session_state["login_error"] = "⚠️ Admin password not configured. Please set ADMIN_PASSWORD environment variable."
            return
        
        entered_password = st.session_state.get("password_input", "")
        
        if hmac.compare_digest(entered_password, admin_password):
            # 生成 session token
            session_token = generate_session_token(admin_password)
            
            # 設定 session state
            st.session_state["authenticated"] = True
            st.session_state["admin_email"] = "admin@errorfree.com"
            st.session_state["login_time"] = datetime.utcnow().isoformat()
            
            # 設定 URL query parameter（這會保留在重新整理後）
            set_query_param("session", session_token)
            
            # 清除密碼輸入和錯誤訊息
            if "password_input" in st.session_state:
                del st.session_state["password_input"]
            if "login_error" in st.session_state:
                del st.session_state["login_error"]
            
            # 記錄登入成功
            _log_audit_event(
                action="admin_login",
                result="success",
                actor_email=st.session_state.get("admin_email", "admin"),
                context={"source": "admin_ui", "timestamp": st.session_state["login_time"]}
            )
            
            # 移除 st.rerun()，讓 Streamlit 自動重新運行
            # Streamlit 會在 callback 結束後自動重新運行
        else:
            st.session_state["authenticated"] = False
            st.session_state["login_error"] = "❌ Incorrect password. Please try again."
            
            # 記錄登入失敗
            _log_audit_event(
                action="admin_login",
                result="denied",
                deny_reason="incorrect_password",
                actor_email="unknown",
                context={"source": "admin_ui", "timestamp": datetime.utcnow().isoformat()}
            )

    # 先檢查 session state
    if st.session_state.get("authenticated", False):
        return True
    
    # 檢查 URL query parameter 中的 session token
    admin_password = os.getenv("ADMIN_PASSWORD", "").strip()
    session_token = get_query_param("session", "")
    
    if session_token and admin_password and verify_session_token(session_token, admin_password):
        # Token 有效，恢復登入狀態
        st.session_state["authenticated"] = True
        st.session_state["admin_email"] = "admin@errorfree.com"
        if "login_time" not in st.session_state:
            st.session_state["login_time"] = datetime.utcnow().isoformat()
        return True

    # 顯示登入頁面
    st.markdown("### 🔐 Error-Free® Admin Login")
    st.markdown("---")
    
    # 顯示錯誤訊息（如果有）
    if "login_error" in st.session_state:
        st.error(st.session_state["login_error"])
    
    # 密碼輸入框
    st.text_input(
        "Password",
        type="password",
        key="password_input",
        on_change=password_entered,
        help="Enter the admin password to access the operations dashboard"
    )
    
    st.info("💡 **Note**: This password is configured via the `ADMIN_PASSWORD` environment variable in Railway.")
    
    return False

# ==========================================
# 登出函數
# ==========================================
def logout():
    """登出並清除 session"""
    actor_email = st.session_state.get("admin_email", "admin")
    
    # 記錄登出
    _log_audit_event(
        action="admin_logout",
        result="success",
        actor_email=actor_email,
        context={"source": "admin_ui", "timestamp": datetime.utcnow().isoformat()}
    )
    
    # 清除 URL query parameters
    clear_query_params()
    
    # 清除所有 session state
    keys_to_clear = list(st.session_state.keys())
    for key in keys_to_clear:
        del st.session_state[key]
    
    st.rerun()

# ==========================================
# 主要管理界面（登入後）
# ==========================================
def show_admin_dashboard():
    """顯示管理儀表板（登入後）"""
    
    # 頂部導覽
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.title("🔐 Error-Free® Operations Dashboard")
    
    with col2:
        st.write("")
        st.write("")
        admin_email = st.session_state.get("admin_email", "admin")
        st.caption(f"👤 {admin_email}")
    
    with col3:
        st.write("")
        st.write("")
        if st.button("🚪 Logout", type="secondary", use_container_width=True):
            logout()
    
    st.markdown("---")
    
    # 側邊欄導覽
    with st.sidebar:
        st.header("📋 Navigation")
        
        page = st.radio(
            "Select a section:",
            [
                "🏠 Dashboard",
                "🏢 Tenants",
                "👥 Members",
                "🚫 Revoke Access",
                "📊 Usage & Caps",
                "📜 Audit Logs"
            ],
            key="nav_page"
        )
        
        st.markdown("---")
        st.caption(f"**Logged in**: {st.session_state.get('login_time', 'N/A')}")
        st.caption(f"**Version**: Phase B1.1 MVP")
    
    # 主內容區域
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🏢 Tenants":
        show_tenants()
    elif page == "👥 Members":
        show_members()
    elif page == "🚫 Revoke Access":
        show_revoke()
    elif page == "📊 Usage & Caps":
        show_usage()
    elif page == "📜 Audit Logs":
        show_audit_logs()

# ==========================================
# 各個頁面的實作（目前為佔位符）
# ==========================================
def show_dashboard():
    """儀表板首頁"""
    st.header("🏠 Dashboard Overview")
    
    # 檢查 Supabase 連線
    supabase_url, service_key = get_supabase_client()
    
    if not supabase_url or not service_key:
        st.error("⚠️ **Supabase not configured**")
        st.warning("""
        Please set the following environment variables in Railway:
        - `SUPABASE_URL`
        - `SUPABASE_SERVICE_KEY`
        """)
        return
    
    st.success("✅ Supabase connected")
    
    # 顯示基本統計（Phase B2 會實作）
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Tenants", "-", help="Coming in Phase B2")
    with col2:
        st.metric("Total Members", "-", help="Coming in Phase B2")
    with col3:
        st.metric("Today's Usage", "-", help="Coming in Phase B2")
    with col4:
        st.metric("Expiring Soon", "-", help="Coming in Phase B2")
    
    st.info("""
    ### 🚀 Phase B1.1 Complete!
    
    ✅ **Login/Logout** - Simple password protection
    ✅ **Session Management** - Persistent across page refreshes
    ✅ **Audit Logging** - All login/logout events recorded
    
    ### 📋 Next Steps (Phase B2)
    
    The following sections will be implemented in subsequent phases:
    - 🏢 **Tenants** - Create, view, manage tenants
    - 👥 **Members** - Batch add/remove members
    - 🚫 **Revoke Access** - Emergency session revocation
    - 📊 **Usage & Caps** - Set limits and monitor usage
    - 📜 **Audit Logs** - View all operations history
    """)

def show_tenants():
    """租戶管理（Phase B2.1 實作）"""
    st.header("🏢 Tenant Management")
    
    # 檢查 Supabase 連線
    supabase_url, service_key = get_supabase_client()
    if not supabase_url or not service_key:
        st.error("⚠️ Supabase not configured")
        return
    
    # 建立 tabs
    tab1, tab2 = st.tabs(["📋 Tenant List", "➕ Create New Tenant"])
    
    # ==========================================
    # Tab 1: 租戶列表
    # ==========================================
    with tab1:
        st.subheader("All Tenants")
        
        # 獲取所有租戶
        try:
            endpoint = f"{supabase_url}/rest/v1/tenants"
            headers = {
                "apikey": service_key,
                "Authorization": f"Bearer {service_key}",
                "Accept": "application/json"
            }
            params = {
                "select": "id,slug,name,display_name,status,trial_start,trial_end,is_active,created_at",
                "order": "created_at.desc"
            }
            
            resp = requests.get(endpoint, headers=headers, params=params, timeout=10)
            
            if resp.status_code == 200:
                tenants = resp.json()
                
                if not tenants:
                    st.info("No tenants found. Create your first tenant in the 'Create New Tenant' tab.")
                else:
                    st.success(f"✅ Found {len(tenants)} tenant(s)")
                    
                    # Search tenants
                    tenant_search = st.text_input("🔍 Search tenants", placeholder="Type slug or name to filter...", key="tenant_list_search")
                    if tenant_search:
                        q = tenant_search.lower()
                        tenants = [t for t in tenants if q in (t.get('slug') or '').lower() or q in (t.get('name') or '').lower()]
                        if not tenants:
                            st.info("No tenants match your search.")
                    
                    # Display tenant list
                    for tenant in tenants:
                        with st.expander(f"**{tenant['slug']}** - {tenant['name']}", expanded=False):
                            show_tenant_details(tenant, supabase_url, service_key)
            else:
                st.error(f"Failed to fetch tenants: HTTP {resp.status_code}")
                
        except Exception as e:
            st.error(f"Error fetching tenants: {str(e)}")
    
    # ==========================================
    # Tab 2: 建立新租戶
    # ==========================================
    with tab2:
        st.subheader("Create New Tenant")
        
        # 檢查是否剛建立完租戶（用來清空表單）
        if st.session_state.get("tenant_just_created", False):
            # 清空標記
            st.session_state["tenant_just_created"] = False
            # 顯示提示訊息
            st.info("✨ Form cleared. You can now create another tenant or switch to the Tenant List tab.")
        
        with st.form("create_tenant_form", clear_on_submit=True):
            st.markdown("### Basic Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                slug = st.text_input(
                    "Tenant Slug *",
                    value="",  # 明確設定為空
                    help="Unique identifier (lowercase, no spaces, e.g., 'acme-corp')"
                )
                name = st.text_input(
                    "Name *",
                    value="",  # 明確設定為空
                    help="Company name (e.g., 'Acme Corporation')"
                )
            
            with col2:
                display_name = st.text_input(
                    "Display Name",
                    value="",  # 明確設定為空
                    help="Optional display name (defaults to Name if empty)"
                )
                trial_days = st.number_input(
                    "Trial Days",
                    min_value=1,
                    max_value=365,
                    value=30,
                    help="Number of days for trial period"
                )
            
            st.markdown("### Default Settings")
            
            col3, col4 = st.columns(2)
            
            with col3:
                daily_review_cap = st.number_input(
                    "Daily Review Cap",
                    min_value=0,
                    value=50,
                    help="Daily review limit (0 = disabled, leave empty for unlimited)"
                )
            
            with col4:
                daily_download_cap = st.number_input(
                    "Daily Download Cap",
                    min_value=0,
                    value=20,
                    help="Daily download limit (0 = disabled, leave empty for unlimited)"
                )
            
            submitted = st.form_submit_button("🚀 Create Tenant", type="primary")
            
            if submitted:
                # 驗證輸入
                if not slug or not name:
                    st.error("❌ Slug and Name are required")
                elif not slug.replace("-", "").replace("_", "").isalnum():
                    st.error("❌ Slug must contain only letters, numbers, hyphens, and underscores")
                elif slug != slug.lower():
                    st.error("❌ Slug must be lowercase")
                else:
                    # 建立租戶
                    create_tenant(
                        slug=slug,
                        name=name,
                        display_name=display_name or name,
                        trial_days=trial_days,
                        daily_review_cap=daily_review_cap if daily_review_cap > 0 else None,
                        daily_download_cap=daily_download_cap if daily_download_cap > 0 else None,
                        supabase_url=supabase_url,
                        service_key=service_key
                    )


def show_tenant_details(tenant: dict, supabase_url: str, service_key: str):
    """顯示租戶詳情和操作按鈕"""
    
    # 基本資訊
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Status**")
        status_color = "🟢" if tenant['is_active'] else "🔴"
        st.write(f"{status_color} {tenant['status']}")
    
    with col2:
        st.markdown("**Trial Period**")
        trial_start = tenant.get('trial_start', '')[:10] if tenant.get('trial_start') else 'N/A'
        trial_end = tenant.get('trial_end', '')[:10] if tenant.get('trial_end') else 'N/A'
        st.write(f"{trial_start} → {trial_end}")
    
    with col3:
        st.markdown("**Created**")
        created = tenant.get('created_at', '')[:10] if tenant.get('created_at') else 'N/A'
        st.write(created)
    
    st.markdown("---")
    
    # 統計資訊
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    
    with stats_col1:
        # 獲取成員數
        member_count = get_tenant_member_count(tenant['id'], supabase_url, service_key)
        st.metric("Members", member_count)
    
    with stats_col2:
        # 獲取今日用量
        today_usage = get_tenant_today_usage(tenant['id'], supabase_url, service_key)
        st.metric("Today's Usage", today_usage)
    
    with stats_col3:
        # 獲取 epoch
        epoch = get_tenant_epoch(tenant['slug'], supabase_url, service_key)
        st.metric("Epoch", epoch)
    
    st.markdown("---")
    
    # 操作按鈕
    st.markdown("### 🔧 Tenant Operations")
    
    col_op1, col_op2 = st.columns(2)
    
    with col_op1:
        # Trial 管理（延長或修改）
        st.markdown("**📅 Trial Period Management**")
        
        trial_action = st.radio(
            "Select Action",
            ["Extend (Add Days)", "Set End Date"],
            key=f"trial_action_{tenant['id']}",
            horizontal=True
        )
        
        if trial_action == "Extend (Add Days)":
            with st.form(f"extend_trial_{tenant['id']}"):
                extend_days = st.number_input(
                    "Days to Add", 
                    min_value=1, 
                    max_value=365, 
                    value=30, 
                    key=f"extend_{tenant['id']}"
                )
                if st.form_submit_button("➕ Extend Trial", type="secondary"):
                    extend_tenant_trial(tenant, extend_days, supabase_url, service_key)
        else:  # Set End Date
            with st.form(f"set_trial_date_{tenant['id']}"):
                from datetime import datetime, date
                
                # 解析當前結束日期
                try:
                    current_end = datetime.fromisoformat(tenant['trial_end'].replace('Z', '+00:00')).date()
                except:
                    current_end = date.today()
                
                st.caption(f"Current end date: {current_end}")
                
                new_end_date = st.date_input(
                    "New End Date",
                    value=current_end,
                    min_value=date.today(),
                    key=f"new_date_{tenant['id']}"
                )
                
                if st.form_submit_button("📅 Update End Date", type="secondary"):
                    update_tenant_trial_date(tenant, new_end_date, supabase_url, service_key)
    
    with col_op2:
        # 狀態管理和刪除
        st.markdown("**⚙️ Status & Management**")
        
        # 停用/啟用
        if tenant['is_active']:
            if st.button(
                "🔴 Disable Tenant", 
                key=f"disable_{tenant['id']}", 
                type="secondary",
                use_container_width=True
            ):
                toggle_tenant_status(tenant, False, supabase_url, service_key)
        else:
            if st.button(
                "🟢 Enable Tenant", 
                key=f"enable_{tenant['id']}", 
                type="primary",
                use_container_width=True
            ):
                toggle_tenant_status(tenant, True, supabase_url, service_key)
        
        # 刪除租戶
        with st.form(f"delete_tenant_{tenant['id']}"):
            st.caption("⚠️ Danger Zone")
            confirm_slug = st.text_input(
                f"Type '{tenant['slug']}' to delete", 
                key=f"confirm_delete_{tenant['id']}"
            )
            if st.form_submit_button("🗑️ Delete Tenant", type="secondary"):
                if confirm_slug == tenant['slug']:
                    delete_tenant(tenant, supabase_url, service_key)
                else:
                    st.error(f"❌ Confirmation failed. Type '{tenant['slug']}' exactly.")
        
        # Quick Info
        if st.button(
            "ℹ️ View Details", 
            key=f"details_{tenant['id']}",
            use_container_width=True
        ):
            st.info(f"**Tenant ID**: `{tenant['id']}`\n\n**Slug**: `{tenant['slug']}`")


def get_tenant_member_count(tenant_id: str, supabase_url: str, service_key: str) -> int:
    """獲取租戶成員數量"""
    try:
        endpoint = f"{supabase_url}/rest/v1/tenant_members"
        headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Accept": "application/json"
        }
        params = {
            "tenant_id": f"eq.{tenant_id}",
            "select": "id"
        }
        resp = requests.get(endpoint, headers=headers, params=params, timeout=5)
        if resp.status_code == 200:
            return len(resp.json())
    except:
        pass
    return 0


def get_tenant_today_usage(tenant_id: str, supabase_url: str, service_key: str) -> int:
    """獲取租戶今日 review 用量"""
    r, _ = get_tenant_today_usage_full(tenant_id, supabase_url, service_key)
    return r


def get_tenant_members_usage_today(tenant_id: str, supabase_url: str, service_key: str) -> list:
    """
    取得租戶內各 member 今日用量（按 email 分組）。
    回傳 [{"email": "...", "review": n, "download": n}, ...]
    """
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).date().isoformat()
    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Accept": "application/json"
    }
    result = {}
    try:
        for utype in ("review", "download"):
            endpoint = f"{supabase_url}/rest/v1/tenant_usage_events"
            params = {
                "tenant_id": f"eq.{tenant_id}",
                "usage_type": f"eq.{utype}",
                "created_at": f"gte.{today}T00:00:00Z",
                "select": "email,quantity"
            }
            resp = requests.get(endpoint, headers=headers, params=params, timeout=5)
            if resp.status_code == 200:
                for r in resp.json() or []:
                    em = (r.get("email") or "").lower()
                    qty = r.get("quantity", 1)
                    if em not in result:
                        result[em] = {"email": em, "review": 0, "download": 0}
                    result[em][utype] = result[em].get(utype, 0) + qty
    except Exception:
        pass
    return list(result.values())


def get_tenant_today_usage_full(tenant_id: str, supabase_url: str, service_key: str) -> tuple:
    """獲取租戶今日 review 和 download 用量，回傳 (review_count, download_count)"""
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).date().isoformat()
    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Accept": "application/json"
    }
    review_count = download_count = 0
    try:
        for utype in ("review", "download"):
            endpoint = f"{supabase_url}/rest/v1/tenant_usage_events"
            params = {
                "tenant_id": f"eq.{tenant_id}",
                "usage_type": f"eq.{utype}",
                "created_at": f"gte.{today}T00:00:00Z",
                "select": "quantity"
            }
            resp = requests.get(endpoint, headers=headers, params=params, timeout=5)
            if resp.status_code == 200:
                rows = resp.json() or []
                total = sum(r.get("quantity", 1) for r in rows)
                if utype == "review":
                    review_count = total
                else:
                    download_count = total
    except Exception:
        pass
    return review_count, download_count


def get_tenant_usage_caps(tenant_id: str, supabase_url: str, service_key: str) -> dict:
    """獲取租戶 usage caps（daily_review_cap, daily_download_cap），若無則回傳空"""
    try:
        endpoint = f"{supabase_url}/rest/v1/tenant_usage_caps"
        headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Accept": "application/json"
        }
        params = {
            "tenant_id": f"eq.{tenant_id}",
            "select": "id,daily_review_cap,daily_download_cap",
            "limit": "1"
        }
        resp = requests.get(endpoint, headers=headers, params=params, timeout=5)
        if resp.status_code == 200 and resp.json():
            return resp.json()[0]
    except Exception:
        pass
    return {}


def init_tenant_usage_caps(tenant_id: str, supabase_url: str, service_key: str,
                          daily_review_cap: int = 50, daily_download_cap: int = 20) -> bool:
    """
    為沒有 caps 記錄的租戶建立 tenant_usage_caps 記錄。
    用於透過 ensure_individual_tenant 等途徑建立的租戶（未走 Create Tenant 流程）。
    """
    from datetime import datetime, timezone
    try:
        headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
        payload = {
            "tenant_id": tenant_id,
            "daily_review_cap": daily_review_cap,
            "daily_download_cap": daily_download_cap,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "last_modified_by": st.session_state.get("admin_email", "admin")
        }
        endpoint = f"{supabase_url}/rest/v1/tenant_usage_caps"
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=10)
        return resp.status_code in [200, 201]
    except Exception:
        return False


def update_tenant_usage_caps(cap_id: str, daily_review_cap, daily_download_cap,
                             supabase_url: str, service_key: str) -> bool:
    """
    更新租戶 usage caps。
    None=unlimited, 0=disabled, >0=limit
    """
    from datetime import datetime, timezone
    try:
        headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "daily_review_cap": daily_review_cap,
            "daily_download_cap": daily_download_cap,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "last_modified_by": st.session_state.get("admin_email", "admin")
        }
        endpoint = f"{supabase_url}/rest/v1/tenant_usage_caps?id=eq.{cap_id}"
        resp = requests.patch(endpoint, json=payload, headers=headers, timeout=10)
        return resp.status_code in [200, 204]
    except Exception:
        return False


def get_tenant_epoch(tenant_slug: str, supabase_url: str, service_key: str) -> int:
    """獲取租戶 epoch"""
    try:
        endpoint = f"{supabase_url}/rest/v1/tenant_session_epoch"
        headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Accept": "application/json"
        }
        params = {
            "tenant": f"eq.{tenant_slug}",
            "select": "epoch"
        }
        resp = requests.get(endpoint, headers=headers, params=params, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                return data[0].get('epoch', 0)
    except:
        pass
    return 0


def create_tenant(slug: str, name: str, display_name: str, trial_days: int,
                  daily_review_cap: int, daily_download_cap: int,
                  supabase_url: str, service_key: str):
    """建立新租戶"""
    from datetime import datetime, timedelta, timezone
    
    try:
        headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
        
        # 1. 建立租戶
        tenant_payload = {
            "slug": slug,
            "name": name,
            "display_name": display_name,
            "status": "trial",
            "trial_start": datetime.now(timezone.utc).isoformat(),
            "trial_end": (datetime.now(timezone.utc) + timedelta(days=trial_days)).isoformat(),
            "is_active": True
        }
        
        endpoint = f"{supabase_url}/rest/v1/tenants"
        resp = requests.post(endpoint, json=tenant_payload, headers=headers, timeout=10)
        
        if resp.status_code != 201:
            st.error(f"❌ Failed to create tenant: HTTP {resp.status_code}")
            st.code(resp.text)
            return
        
        tenant_data = resp.json()[0]
        tenant_id = tenant_data['id']
        
        st.success(f"✅ Tenant created: {slug}")
        
        # 2. 初始化 epoch
        try:
            epoch_payload = {"tenant": slug, "epoch": 0}
            endpoint = f"{supabase_url}/rest/v1/tenant_session_epoch"
            resp = requests.post(endpoint, json=epoch_payload, headers=headers, timeout=5)
            
            if resp.status_code == 201:
                st.success("✅ Epoch initialized")
            else:
                st.warning(f"⚠️ Epoch initialization: HTTP {resp.status_code}")
        except Exception as e:
            st.warning(f"⚠️ Epoch initialization error: {str(e)}")
        
        # 3. 設定 usage caps
        try:
            caps_payload = {
                "tenant_id": tenant_id,
                "daily_review_cap": daily_review_cap,
                "daily_download_cap": daily_download_cap
            }
            endpoint = f"{supabase_url}/rest/v1/tenant_usage_caps"
            resp = requests.post(endpoint, json=caps_payload, headers=headers, timeout=5)
            
            if resp.status_code == 201:
                st.success("✅ Usage caps configured")
            else:
                st.warning(f"⚠️ Usage caps setup: HTTP {resp.status_code} - {resp.text}")
        except Exception as e:
            st.warning(f"⚠️ Usage caps error: {str(e)}")
        
        # 4. 記錄 audit event
        _log_audit_event(
            action="tenant_created",
            tenant=slug,
            result="success",
            actor_email=st.session_state.get("admin_email", "admin"),
            context={
                "tenant_id": tenant_id,
                "trial_days": trial_days,
                "daily_review_cap": daily_review_cap,
                "daily_download_cap": daily_download_cap
            }
        )
        
        # 5. 顯示完成訊息
        st.success("🎉 Tenant setup complete!")
        st.balloons()
        
        # 6. 設定標記表示租戶已建立（用來清空表單）
        st.session_state["tenant_just_created"] = True
        
        # 7. 延遲後重新載入
        import time
        time.sleep(3)  # 增加到 3 秒，確保所有訊息都能看到
        
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error creating tenant: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        _log_audit_event(
            action="tenant_created",
            tenant=slug,
            result="error",
            actor_email=st.session_state.get("admin_email", "admin"),
            context={"error": str(e)}
        )


def extend_tenant_trial(tenant: dict, extend_days: int, supabase_url: str, service_key: str):
    """延長租戶試用期"""
    from datetime import datetime, timedelta
    
    try:
        # 計算新的結束日期
        current_end = datetime.fromisoformat(tenant['trial_end'].replace('Z', '+00:00'))
        new_end = current_end + timedelta(days=extend_days)
        
        # 更新租戶
        headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json"
        }
        
        payload = {"trial_end": new_end.isoformat()}
        endpoint = f"{supabase_url}/rest/v1/tenants?id=eq.{tenant['id']}"
        resp = requests.patch(endpoint, json=payload, headers=headers, timeout=10)
        
        if resp.status_code in [200, 204]:
            st.success(f"✅ Trial extended by {extend_days} days")
            st.info(f"New end date: {new_end.date()}")
            
            # 記錄 audit event
            _log_audit_event(
                action="tenant_trial_extended",
                tenant=tenant['slug'],
                result="success",
                actor_email=st.session_state.get("admin_email", "admin"),
                context={
                    "extend_days": extend_days,
                    "new_trial_end": new_end.isoformat()
                }
            )
            
            # 延遲重新載入，讓用戶看到成功訊息
            import time
            time.sleep(2)
            st.rerun()
        else:
            st.error(f"❌ Failed to extend trial: HTTP {resp.status_code}")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


def update_tenant_trial_date(tenant: dict, new_end_date, supabase_url: str, service_key: str):
    """直接修改租戶試用期結束日期（可以縮短或延長）"""
    from datetime import datetime, timezone
    
    try:
        # 解析當前和新的日期
        current_end = datetime.fromisoformat(tenant['trial_end'].replace('Z', '+00:00'))
        
        # 將 date 轉換為 datetime（使用 UTC 午夜）
        new_end_datetime = datetime.combine(new_end_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        
        # 計算差異
        days_diff = (new_end_datetime - current_end).days
        
        # 更新租戶
        headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json"
        }
        
        payload = {"trial_end": new_end_datetime.isoformat()}
        endpoint = f"{supabase_url}/rest/v1/tenants?id=eq.{tenant['id']}"
        resp = requests.patch(endpoint, json=payload, headers=headers, timeout=10)
        
        if resp.status_code in [200, 204]:
            if days_diff > 0:
                st.success(f"✅ Trial period extended by {days_diff} days")
            elif days_diff < 0:
                st.success(f"✅ Trial period shortened by {abs(days_diff)} days")
            else:
                st.info("ℹ️ Trial end date unchanged")
            
            st.info(f"New end date: {new_end_date}")
            
            # 記錄 audit event
            _log_audit_event(
                action="tenant_trial_date_updated",
                tenant=tenant['slug'],
                result="success",
                actor_email=st.session_state.get("admin_email", "admin"),
                context={
                    "old_trial_end": current_end.isoformat(),
                    "new_trial_end": new_end_datetime.isoformat(),
                    "days_diff": days_diff
                }
            )
            
            # 延遲重新載入
            import time
            time.sleep(2)
            st.rerun()
        else:
            st.error(f"❌ Failed to update trial date: HTTP {resp.status_code}")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def toggle_tenant_status(tenant: dict, new_status: bool, supabase_url: str, service_key: str):
    """切換租戶啟用/停用狀態"""
    try:
        headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "is_active": new_status,
            "status": "trial" if new_status else "suspended"
        }
        
        endpoint = f"{supabase_url}/rest/v1/tenants?id=eq.{tenant['id']}"
        resp = requests.patch(endpoint, json=payload, headers=headers, timeout=10)
        
        if resp.status_code in [200, 204]:
            action_text = "enabled" if new_status else "disabled"
            st.success(f"✅ Tenant {action_text}")
            
            # 記錄 audit event
            _log_audit_event(
                action=f"tenant_{'enabled' if new_status else 'disabled'}",
                tenant=tenant['slug'],
                result="success",
                actor_email=st.session_state.get("admin_email", "admin"),
                context={"new_status": new_status}
            )
            
            # 延遲重新載入，讓用戶看到成功訊息
            import time
            time.sleep(1.5)
            st.rerun()
        else:
            st.error(f"❌ Failed to update status: HTTP {resp.status_code}")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


def delete_tenant(tenant: dict, supabase_url: str, service_key: str):
    """刪除租戶（危險操作）"""
    try:
        headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json"
        }
        
        tenant_id = tenant['id']
        tenant_slug = tenant['slug']
        
        # 1. 刪除 usage events（如果有）
        try:
            endpoint = f"{supabase_url}/rest/v1/tenant_usage_events?tenant_id=eq.{tenant_id}"
            requests.delete(endpoint, headers=headers, timeout=5)
        except:
            pass
        
        # 2. 刪除 usage caps
        try:
            endpoint = f"{supabase_url}/rest/v1/tenant_usage_caps?tenant_id=eq.{tenant_id}"
            requests.delete(endpoint, headers=headers, timeout=5)
        except:
            pass
        
        # 3. 刪除 members
        try:
            endpoint = f"{supabase_url}/rest/v1/tenant_members?tenant_id=eq.{tenant_id}"
            requests.delete(endpoint, headers=headers, timeout=5)
        except:
            pass
        
        # 4. 刪除 epoch
        try:
            endpoint = f"{supabase_url}/rest/v1/tenant_session_epoch?tenant=eq.{tenant_slug}"
            requests.delete(endpoint, headers=headers, timeout=5)
        except:
            pass
        
        # 5. 刪除 tenant（主表）
        endpoint = f"{supabase_url}/rest/v1/tenants?id=eq.{tenant_id}"
        resp = requests.delete(endpoint, headers=headers, timeout=10)
        
        if resp.status_code in [200, 204]:
            st.success(f"✅ Tenant '{tenant_slug}' deleted successfully")
            
            # 記錄 audit event
            _log_audit_event(
                action="tenant_deleted",
                tenant=tenant_slug,
                result="success",
                actor_email=st.session_state.get("admin_email", "admin"),
                context={
                    "tenant_id": tenant_id,
                    "tenant_name": tenant['name']
                }
            )
            
            # 延遲重新載入
            import time
            time.sleep(2)
            st.rerun()
        else:
            st.error(f"❌ Failed to delete tenant: HTTP {resp.status_code}")
            st.code(resp.text)
            
    except Exception as e:
        st.error(f"❌ Error deleting tenant: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def show_members():
    """成員管理（Phase B3.1 實作）"""
    st.header("👥 Member Management")
    
    supabase_url, service_key = get_supabase_client()
    if not supabase_url or not service_key:
        st.error("⚠️ Supabase not configured")
        return
    
    # 分頁
    tab1, tab2, tab3 = st.tabs(["📋 Member List", "➕ Batch Add Members", "⚙️ Batch Operations"])
    
    with tab1:
        show_member_list(supabase_url, service_key)
    
    with tab2:
        show_batch_add_members(supabase_url, service_key)
    
    with tab3:
        show_batch_operations(supabase_url, service_key)


def show_member_list(supabase_url: str, service_key: str):
    """顯示成員列表"""
    st.subheader("All Members")
    
    # 租戶篩選器
    tenants = get_all_tenants(supabase_url, service_key)
    if not tenants:
        st.warning("⚠️ No tenants found. Please create a tenant first.")
        return
    
    tenant_options = ["All Tenants"] + [f"{t['slug']} - {t['name']}" for t in tenants]
    selected_tenant = st.selectbox("Filter by Tenant", tenant_options, key="member_filter_tenant")
    
    # 解析選擇的租戶 slug
    if selected_tenant == "All Tenants":
        filter_slug = None
    else:
        filter_slug = selected_tenant.split(" - ")[0]
    
    # 獲取成員列表
    members = get_members(supabase_url, service_key, filter_slug)
    
    if not members:
        st.info("ℹ️ No members found.")
        return
    
    # 搜尋框
    search_query = st.text_input("🔍 Search by email", placeholder="Type to search...", key="member_search")
    if search_query:
        members = [m for m in members if search_query.lower() in (m.get('email') or '').lower()]
    
    # 統計與狀態篩選（點擊可篩選顯示）
    active_count = sum(1 for m in members if m['is_active'])
    inactive_count = len(members) - active_count
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Members", len(members))
    with col2:
        st.metric("Active Members", active_count)
    with col3:
        st.metric("Inactive Members", inactive_count)
    
    status_filter = st.radio(
        "Filter by status",
        ["All", "Active only", "Inactive only"],
        horizontal=True,
        key="member_status_filter"
    )
    if status_filter == "Active only":
        members = [m for m in members if m['is_active']]
    elif status_filter == "Inactive only":
        members = [m for m in members if not m['is_active']]
    
    st.markdown("---")
    
    # 批量刪除區域（使用 counter 讓成功後清空選取）
    if members:
        st.markdown("**Batch Delete**")
        delete_counter = st.session_state.get("member_delete_counter", 0)
        member_options = [f"{m['email']} @ {m['tenant_slug']} {'✅' if m['is_active'] else '❌'}" for m in members]
        opt_to_member = {opt: m for opt, m in zip(member_options, members)}
        
        selected_for_delete = st.multiselect(
            "Select members to delete",
            member_options,
            key=f"member_batch_delete_select_{delete_counter}"
        )
        
        if selected_for_delete:
            to_delete = [opt_to_member[opt] for opt in selected_for_delete if opt in opt_to_member]
            if st.button("🗑️ Delete Selected", type="secondary"):
                batch_delete_members(to_delete, supabase_url, service_key)
        st.markdown("---")
    
    # 成員列表
    for member in members:
        with st.expander(
            f"{'✅' if member['is_active'] else '❌'} **{member['email']}** ({member['tenant_slug']})",
            expanded=False
        ):
            show_member_details(member, supabase_url, service_key)


def _role_display(role: str) -> str:
    """將 DB 角色轉為顯示名稱（DB: user/tenant_admin/guest）"""
    if role == "tenant_admin": return "Admin"
    if role == "guest": return "Guest"
    return "User"


def show_member_details(member: dict, supabase_url: str, service_key: str):
    """顯示單個成員的詳細資訊"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Basic Info**")
        st.write(f"**Email**: {member['email']}")
        st.write(f"**Tenant**: {member['tenant_slug']}")
        st.write(f"**Role**: {_role_display(member.get('role', 'user'))}")
        status = "🟢 Active" if member['is_active'] else "🔴 Inactive"
        st.write(f"**Status**: {status}")
        st.write(f"**Created**: {member.get('created_at', 'N/A')[:10]}")
    
    with col2:
        st.markdown("**Actions**")
        
        # 切換狀態
        if member['is_active']:
            if st.button(f"🔴 Disable", key=f"disable_member_{member['id']}", type="secondary"):
                toggle_member_status(member, False, supabase_url, service_key)
        else:
            if st.button(f"🟢 Enable", key=f"enable_member_{member['id']}", type="primary"):
                toggle_member_status(member, True, supabase_url, service_key)
        
        # 刪除成員
        if st.button(f"🗑️ Delete", key=f"delete_member_{member['id']}", type="secondary"):
            delete_member(member, supabase_url, service_key)
        
        # 更改角色（DB 使用 user / tenant_admin / guest）
        with st.form(f"role_form_{member['id']}"):
            role_opts = ["user", "tenant_admin", "guest"]
            role_idx = role_opts.index(member.get('role', 'user')) if member.get('role', 'user') in role_opts else 0
            new_role = st.selectbox(
                "Change Role",
                options=role_opts,
                format_func=_role_display,
                index=role_idx,
                key=f"role_{member['id']}"
            )
            if st.form_submit_button("Update Role"):
                update_member_role(member, new_role, supabase_url, service_key)


def show_batch_add_members(supabase_url: str, service_key: str):
    """批量新增成員"""
    st.subheader("Batch Add Members")
    
    add_type = st.radio(
        "Add to",
        ["Tenant", "Individual (Guest)"],
        horizontal=True,
        key="batch_add_type"
    )
    
    if add_type == "Tenant":
        tenants = get_all_tenants(supabase_url, service_key)
        if not tenants:
            st.warning("⚠️ No tenants found. Please create a tenant first.")
            return
        
        tenant_search = st.text_input("🔍 Search tenants", placeholder="Type slug or name to filter...", key="batch_add_tenant_search")
        if tenant_search:
            q = tenant_search.lower()
            tenants = [t for t in tenants if q in (t.get('slug') or '').lower() or q in (t.get('name') or '').lower()]
        
        tenant_options = [f"{t['slug']} - {t['name']}" for t in tenants]
        if not tenant_options:
            st.info("No tenants match your search.")
            return
        selected_tenant = st.selectbox("Select Tenant", tenant_options, key="batch_add_tenant")
        tenant_slug = selected_tenant.split(" - ")[0]
        role_default = "user"
    else:
        # Individual (Guest) - 使用 individual 租戶
        if not ensure_individual_tenant(supabase_url, service_key):
            st.error("❌ Failed to ensure 'individual' tenant exists. Please create it manually.")
            return
        tenant_slug = "individual"
        role_default = "guest"
        st.info("ℹ️ Adding as **Individual (Guest)** — user will not belong to any company tenant.")
    
    st.markdown("---")
    
    # 輸入方式選擇
    input_method = st.radio(
        "Input Method",
        ["Paste Emails", "Manual Entry"],
        horizontal=True
    )
    
    # 使用 counter 讓表單在成功後清空（key 改變會重置 widget）
    batch_counter = st.session_state.get("batch_add_counter", 0)
    
    if input_method == "Paste Emails":
        st.markdown("**Paste email addresses** (one per line):")
        email_text = st.text_area(
            "Email List",
            height=200,
            placeholder="user1@example.com\nuser2@example.com\nuser3@example.com",
            key=f"batch_email_text_{batch_counter}"
        )
        
        if st.button("➕ Add All Members", type="primary"):
            if not email_text.strip():
                st.error("❌ Please enter at least one email address.")
            else:
                emails = [line.strip() for line in email_text.strip().split('\n') if line.strip()]
                emails = [e for e in emails if '@' in e]
                if not emails:
                    st.error("❌ No valid email addresses found.")
                else:
                    batch_add_members(tenant_slug, emails, supabase_url, service_key, role_default)
    
    else:  # Manual Entry
        st.markdown("**Add a single member:**")
        manual_counter = st.session_state.get("manual_add_counter", 0)
        email = st.text_input("Email", placeholder="user@example.com", key=f"manual_add_email_{manual_counter}")
        role = st.selectbox(
            "Role",
            options=["user", "tenant_admin", "guest"],
            format_func=_role_display,
            index=["user", "tenant_admin", "guest"].index(role_default),
            key=f"manual_add_role_{manual_counter}"
        )
        
        if st.button("➕ Add Member", type="primary"):
            if not email or '@' not in email:
                st.error("❌ Please enter a valid email address.")
            else:
                batch_add_members(tenant_slug, [email], supabase_url, service_key, role, source="manual")


def show_batch_operations(supabase_url: str, service_key: str):
    """批量操作成員（顯示列表 + 個別 checkbox，非下拉選單）"""
    st.subheader("Batch Operations")
    
    tenants = get_all_tenants(supabase_url, service_key)
    if not tenants:
        st.warning("⚠️ No tenants found.")
        return
    
    tenant_search = st.text_input("🔍 Search tenants", placeholder="Type slug or name to filter...", key="batch_ops_tenant_search")
    if tenant_search:
        q = tenant_search.lower()
        tenants = [t for t in tenants if q in (t.get('slug') or '').lower() or q in (t.get('name') or '').lower()]
    
    if not tenants:
        st.info("No tenants match your search.")
        return
    
    tenant_options = [f"{t['slug']} - {t['name']}" for t in tenants]
    selected_tenant = st.selectbox("Select Tenant", tenant_options, key="batch_ops_tenant")
    tenant_slug = selected_tenant.split(" - ")[0]
    
    members = get_members(supabase_url, service_key, tenant_slug)
    if not members:
        st.info("ℹ️ No members found for this tenant.")
        return
    
    ops_counter = st.session_state.get("batch_ops_counter", 0)
    
    # 搜尋
    ops_search = st.text_input("🔍 Search members", placeholder="Type to filter...", key="batch_ops_search")
    if ops_search:
        members = [m for m in members if ops_search.lower() in (m.get('email') or '').lower()]
    
    if not members:
        st.info("ℹ️ No members match your search.")
        return
    
    # 篩選：All / Active only / Inactive only
    view_filter = st.session_state.get("batch_ops_view", "all")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Select All", key="batch_select_all"):
            st.session_state["batch_ops_view"] = "all"
            for m in members:
                st.session_state[f"batch_cb_{m['id']}_{ops_counter}"] = True
            st.rerun()
    with col_b:
        if st.button("Select Active", key="batch_select_active"):
            st.session_state["batch_ops_view"] = "active"
            for m in members:
                st.session_state[f"batch_cb_{m['id']}_{ops_counter}"] = m['is_active']
            st.rerun()
    with col_c:
        if st.button("Select Inactive", key="batch_select_inactive"):
            st.session_state["batch_ops_view"] = "inactive"
            for m in members:
                st.session_state[f"batch_cb_{m['id']}_{ops_counter}"] = not m['is_active']
            st.rerun()
    
    if view_filter == "active":
        members = [m for m in members if m['is_active']]
    elif view_filter == "inactive":
        members = [m for m in members if not m['is_active']]
    
    if not members:
        st.info("ℹ️ No members in this filter.")
        return
    
    st.markdown("---")
    st.markdown("**Select members** (check each account):")
    
    selected_ids = set()
    for m in members:
        cb_key = f"batch_cb_{m['id']}_{ops_counter}"
        if st.checkbox(
            f"{m['email']}  {'✅ Active' if m['is_active'] else '❌ Inactive'}",
            key=cb_key
        ):
            selected_ids.add(m['id'])
    
    selected_members = [m for m in members if m['id'] in selected_ids]
    selected_emails = [m['email'] for m in selected_members]
    
    if not selected_emails:
        st.info("👆 Check the accounts above, or use Select All / Select Active / Select Inactive for quick selection.")
        return
    
    st.write(f"**Selected**: {len(selected_emails)} member(s)")
    
    # 根據選取的狀態顯示對應按鈕
    sel_active = sum(1 for m in selected_members if m['is_active'])
    sel_inactive = len(selected_members) - sel_active
    
    col1, col2 = st.columns(2)
    with col1:
        if sel_inactive > 0:
            if st.button("🟢 Enable Selected", type="primary", use_container_width=True):
                batch_toggle_members(tenant_slug, selected_emails, True, supabase_url, service_key)
    with col2:
        if sel_active > 0:
            if st.button("🔴 Disable Selected", type="secondary", use_container_width=True):
                batch_toggle_members(tenant_slug, selected_emails, False, supabase_url, service_key)
    
    if sel_active == 0 and sel_inactive == 0:
        st.caption("(Enable / Disable buttons appear after selecting members)")


# ==========================================
# 成員管理 Helper Functions
# ==========================================

def ensure_individual_tenant(supabase_url: str, service_key: str) -> bool:
    """確保 individual 租戶存在（用於 Guest 個人使用者）"""
    try:
        from datetime import datetime, timedelta, timezone
        headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json"
        }
        endpoint = f"{supabase_url}/rest/v1/tenants?slug=eq.individual&select=id"
        resp = requests.get(endpoint, headers=headers, timeout=10)
        if resp.status_code == 200 and resp.json():
            return True
        # 不存在則建立
        now = datetime.now(timezone.utc)
        trial_end = now + timedelta(days=3650)  # 10 年
        payload = {
            "slug": "individual",
            "name": "Individual Users",
            "display_name": "Individual (Guest)",
            "status": "active",
            "is_active": True,
            "trial_start": now.isoformat(),
            "trial_end": trial_end.isoformat()
        }
        resp = requests.post(f"{supabase_url}/rest/v1/tenants", json=payload, headers={**headers, "Prefer": "return=minimal"}, timeout=10)
        return resp.status_code in [200, 201]
    except Exception:
        return False


def get_tenant_ids_by_member_email_search(search: str, supabase_url: str, service_key: str) -> set:
    """
    依 member email 搜尋，回傳包含該成員的 tenant_id 集合。
    支援部分比對（例如 coco@gmail.com 或 coco）。
    """
    if not search or not search.strip():
        return set()
    q = search.strip().lower()
    try:
        headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Accept": "application/json"
        }
        # PostgREST: ilike 用 * 表示任意字元，*coco* = 包含 coco
        endpoint = f"{supabase_url}/rest/v1/tenant_members"
        params = {
            "select": "tenant_id",
            "email": f"ilike.*{q}*",
            "limit": "500"
        }
        resp = requests.get(endpoint, headers=headers, params=params, timeout=10)
        if resp.status_code == 200:
            rows = resp.json() or []
            return {r["tenant_id"] for r in rows if r.get("tenant_id")}
    except Exception:
        pass
    return set()


def get_all_tenants(supabase_url: str, service_key: str) -> list:
    """獲取所有租戶"""
    try:
        headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json"
        }
        
        endpoint = f"{supabase_url}/rest/v1/tenants?select=id,slug,name&order=created_at.desc"
        resp = requests.get(endpoint, headers=headers, timeout=10)
        
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"❌ Failed to fetch tenants: HTTP {resp.status_code}")
            return []
    except Exception as e:
        st.error(f"❌ Error fetching tenants: {str(e)}")
        return []


def get_audit_events(supabase_url: str, service_key: str,
                    time_range: str = "Last 7 days",
                    tenant_slug: str = None,
                    action: str = None,
                    result: str = None,
                    limit: int = 200) -> list:
    """
    取得 audit_events 列表（支援篩選）
    
    Args:
        time_range: "Today" | "Last 7 days" | "Last 30 days"
        tenant_slug: 租戶 slug 篩選（None = 全部）
        action: action 篩選（None = 全部）
        result: result 篩選（None = 全部）
        limit: 最多筆數
    """
    from datetime import datetime, timedelta, timezone
    try:
        headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Accept": "application/json"
        }
        now = datetime.now(timezone.utc)
        if time_range == "Today":
            since = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif time_range == "Last 30 days":
            since = now - timedelta(days=30)
        else:
            since = now - timedelta(days=7)
        since_iso = since.isoformat().replace("+00:00", "Z")
        
        endpoint = f"{supabase_url}/rest/v1/audit_events"
        params = {
            "select": "id,created_at,action,tenant_slug,email,result,deny_reason,context,actor_email,notes",
            "created_at": f"gte.{since_iso}",
            "order": "created_at.desc",
            "limit": str(limit)
        }
        if tenant_slug:
            params["tenant_slug"] = f"eq.{tenant_slug}"
        if action:
            params["action"] = f"eq.{action}"
        if result:
            params["result"] = f"eq.{result}"
        
        resp = requests.get(endpoint, headers=headers, params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        st.error(f"❌ Failed to fetch audit events: HTTP {resp.status_code}")
        return []
    except Exception as e:
        st.error(f"❌ Error fetching audit events: {str(e)}")
        return []


def get_members(supabase_url: str, service_key: str, tenant_slug: str = None) -> list:
    """獲取成員列表（JOIN tenants 表以取得 slug）"""
    try:
        headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json"
        }
        
        # 使用 select 來 JOIN tenants 表
        if tenant_slug:
            # 先獲取 tenant_id
            tenant_endpoint = f"{supabase_url}/rest/v1/tenants?slug=eq.{tenant_slug}&select=id"
            tenant_resp = requests.get(tenant_endpoint, headers=headers, timeout=10)
            
            if tenant_resp.status_code != 200 or not tenant_resp.json():
                st.error(f"❌ Tenant '{tenant_slug}' not found")
                return []
            
            tenant_id = tenant_resp.json()[0]['id']
            endpoint = f"{supabase_url}/rest/v1/tenant_members?tenant_id=eq.{tenant_id}&select=*&order=created_at.desc"
        else:
            endpoint = f"{supabase_url}/rest/v1/tenant_members?select=*&order=created_at.desc"
        
        resp = requests.get(endpoint, headers=headers, timeout=10)
        
        if resp.status_code == 200:
            members = resp.json()
            
            # 為每個成員添加 tenant_slug（通過 tenant_id 查詢）
            tenant_cache = {}  # 快取 tenant_id -> slug 的對應
            
            for member in members:
                tenant_id = member.get('tenant_id')
                if tenant_id:
                    # 檢查快取
                    if tenant_id not in tenant_cache:
                        tenant_endpoint = f"{supabase_url}/rest/v1/tenants?id=eq.{tenant_id}&select=slug"
                        tenant_resp = requests.get(tenant_endpoint, headers=headers, timeout=10)
                        if tenant_resp.status_code == 200 and tenant_resp.json():
                            tenant_cache[tenant_id] = tenant_resp.json()[0]['slug']
                        else:
                            tenant_cache[tenant_id] = 'unknown'
                    
                    member['tenant_slug'] = tenant_cache[tenant_id]
                else:
                    member['tenant_slug'] = 'unknown'
            
            return members
        else:
            st.error(f"❌ Failed to fetch members: HTTP {resp.status_code}")
            return []
    except Exception as e:
        st.error(f"❌ Error fetching members: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return []


def get_existing_member_emails(supabase_url: str, service_key: str, tenant_id: str) -> set:
    """取得指定租戶已存在的成員 email（含 Active 和 Inactive）"""
    try:
        headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json"
        }
        endpoint = f"{supabase_url}/rest/v1/tenant_members?tenant_id=eq.{tenant_id}&select=email,is_active"
        resp = requests.get(endpoint, headers=headers, timeout=10)
        if resp.status_code == 200:
            return {(m['email'].lower(), m['is_active']) for m in resp.json()}
        return set()
    except Exception:
        return set()


def get_all_existing_emails_global(supabase_url: str, service_key: str) -> set:
    """取得全系統已存在的成員 email（跨所有租戶，含 individual）"""
    try:
        headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json"
        }
        endpoint = f"{supabase_url}/rest/v1/tenant_members?select=email"
        resp = requests.get(endpoint, headers=headers, timeout=10)
        if resp.status_code == 200:
            return {m['email'].lower() for m in resp.json()}
        return set()
    except Exception:
        return set()


def batch_add_members(tenant_slug: str, emails: list, supabase_url: str, service_key: str, role: str = "user", source: str = "paste"):
    """批量新增成員（使用 tenant_id，會檢查重複）"""
    try:
        headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
        
        # 先獲取 tenant_id
        tenant_endpoint = f"{supabase_url}/rest/v1/tenants?slug=eq.{tenant_slug}&select=id"
        tenant_resp = requests.get(tenant_endpoint, headers=headers, timeout=10)
        
        if tenant_resp.status_code != 200 or not tenant_resp.json():
            st.error(f"❌ Tenant '{tenant_slug}' not found")
            return
        
        tenant_id = tenant_resp.json()[0]['id']
        
        # 檢查全系統已存在的 email（跨所有租戶，含 individual），同一 email 不可重複
        existing_emails = get_all_existing_emails_global(supabase_url, service_key)
        
        emails_lower = [e.lower().strip() for e in emails]
        duplicates = [e for e in emails_lower if e in existing_emails]
        to_add = [e for e in emails_lower if e not in existing_emails]
        
        if duplicates:
            st.warning(f"⚠️ **The following account(s) already exist in the system (any tenant or individual) and cannot be added**: {', '.join(duplicates)}")
            if not to_add:
                st.info("ℹ️ No accounts can be added. Please check and try again.")
                return
        
        if not to_add:
            return
        
        # 準備批量插入的數據（只插入不重複的）
        members_data = []
        for email in to_add:
            members_data.append({
                "tenant_id": tenant_id,
                "email": email,
                "role": role,
                "is_active": True
            })
        
        # 批量插入
        endpoint = f"{supabase_url}/rest/v1/tenant_members"
        resp = requests.post(endpoint, json=members_data, headers=headers, timeout=30)
        
        if resp.status_code in [200, 201]:
            added = resp.json()
            st.success(f"✅ Successfully added {len(added)} member(s)!")
            st.balloons()
            
            # 記錄 audit event
            _log_audit_event(
                action="members_batch_added",
                tenant=tenant_slug,
                result="success",
                actor_email=st.session_state.get("admin_email", "admin"),
                context={
                    "count": len(added),
                    "emails": emails,
                    "role": role
                }
            )
            
            # Success: increment counter(s) to clear form
            st.session_state["batch_add_counter"] = st.session_state.get("batch_add_counter", 0) + 1
            if source == "manual":
                st.session_state["manual_add_counter"] = st.session_state.get("manual_add_counter", 0) + 1
            import time
            time.sleep(2)
            st.rerun()
        elif resp.status_code == 409:
            st.warning("⚠️ Some members already exist. Skipping duplicates...")
            st.info("💡 Tip: Existing members were not modified.")
        else:
            st.error(f"❌ Failed to add members: HTTP {resp.status_code}")
            st.code(resp.text)
            
    except Exception as e:
        st.error(f"❌ Error adding members: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def toggle_member_status(member: dict, new_status: bool, supabase_url: str, service_key: str):
    """切換成員狀態"""
    try:
        headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json"
        }
        
        payload = {"is_active": new_status}
        endpoint = f"{supabase_url}/rest/v1/tenant_members?id=eq.{member['id']}"
        resp = requests.patch(endpoint, json=payload, headers=headers, timeout=10)
        
        if resp.status_code in [200, 204]:
            action_text = "enabled" if new_status else "disabled"
            st.success(f"✅ Member {action_text}!")
            
            # 記錄 audit event
            _log_audit_event(
                action=f"member_{'enabled' if new_status else 'disabled'}",
                tenant=member['tenant_slug'],
                result="success",
                actor_email=st.session_state.get("admin_email", "admin"),
                context={
                    "member_email": member['email'],
                    "new_status": new_status
                }
            )
            
            import time
            time.sleep(1.5)
            st.rerun()
        else:
            st.error(f"❌ Failed to update member: HTTP {resp.status_code}")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


def update_member_role(member: dict, new_role: str, supabase_url: str, service_key: str):
    """更新成員角色"""
    try:
        headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json"
        }
        
        payload = {"role": new_role}
        endpoint = f"{supabase_url}/rest/v1/tenant_members?id=eq.{member['id']}"
        resp = requests.patch(endpoint, json=payload, headers=headers, timeout=10)
        
        if resp.status_code in [200, 204]:
            st.success(f"✅ Role updated to '{_role_display(new_role)}'!")
            
            # 記錄 audit event
            _log_audit_event(
                action="member_role_updated",
                tenant=member['tenant_slug'],
                result="success",
                actor_email=st.session_state.get("admin_email", "admin"),
                context={
                    "member_email": member['email'],
                    "old_role": member.get('role', 'user'),
                    "new_role": new_role
                }
            )
            
            import time
            time.sleep(1.5)
            st.rerun()
        else:
            st.error(f"❌ Failed to update role: HTTP {resp.status_code}")
            try:
                err_detail = resp.json() or resp.text
                if err_detail:
                    st.code(str(err_detail), language="json")
            except Exception:
                pass
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


def delete_member(member: dict, supabase_url: str, service_key: str):
    """刪除單個成員"""
    try:
        headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json"
        }
        endpoint = f"{supabase_url}/rest/v1/tenant_members?id=eq.{member['id']}"
        resp = requests.delete(endpoint, headers=headers, timeout=10)
        if resp.status_code in [200, 204]:
            st.success(f"✅ Member {member['email']} deleted.")
            _log_audit_event(
                action="member_deleted",
                tenant=member.get('tenant_slug', ''),
                result="success",
                actor_email=st.session_state.get("admin_email", "admin"),
                context={"member_email": member['email']}
            )
            import time
            time.sleep(1.5)
            st.rerun()
        else:
            st.error(f"❌ Failed to delete: HTTP {resp.status_code}")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


def batch_delete_members(members: list, supabase_url: str, service_key: str):
    """批量刪除成員"""
    try:
        headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json"
        }
        success = 0
        for member in members:
            endpoint = f"{supabase_url}/rest/v1/tenant_members?id=eq.{member['id']}"
            resp = requests.delete(endpoint, headers=headers, timeout=10)
            if resp.status_code in [200, 204]:
                success += 1
        if success > 0:
            st.success(f"✅ {success} member(s) deleted.")
            tenant = members[0].get('tenant_slug', '') if members else ''
            _log_audit_event(
                action="members_batch_deleted",
                tenant=tenant,
                result="success",
                actor_email=st.session_state.get("admin_email", "admin"),
                context={
                    "count": success,
                    "emails": [m['email'] for m in members[:success]]
                }
            )
            import time
            st.session_state["member_delete_counter"] = st.session_state.get("member_delete_counter", 0) + 1
            time.sleep(2)
            st.rerun()
        else:
            st.error("❌ No members were deleted.")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


def batch_toggle_members(tenant_slug: str, emails: list, new_status: bool, supabase_url: str, service_key: str):
    """批量切換成員狀態（使用 tenant_id）"""
    try:
        headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json"
        }
        
        # 先獲取 tenant_id
        tenant_endpoint = f"{supabase_url}/rest/v1/tenants?slug=eq.{tenant_slug}&select=id"
        tenant_resp = requests.get(tenant_endpoint, headers=headers, timeout=10)
        
        if tenant_resp.status_code != 200 or not tenant_resp.json():
            st.error(f"❌ Tenant '{tenant_slug}' not found")
            return
        
        tenant_id = tenant_resp.json()[0]['id']
        
        # 對每個 email 進行更新
        success_count = 0
        for email in emails:
            payload = {"is_active": new_status}
            endpoint = f"{supabase_url}/rest/v1/tenant_members?tenant_id=eq.{tenant_id}&email=eq.{email}"
            resp = requests.patch(endpoint, json=payload, headers=headers, timeout=10)
            
            if resp.status_code in [200, 204]:
                success_count += 1
        
        if success_count > 0:
            action_text = "enabled" if new_status else "disabled"
            st.success(f"✅ {success_count} member(s) {action_text}!")
            
            # 記錄 audit event
            _log_audit_event(
                action=f"members_batch_{'enabled' if new_status else 'disabled'}",
                tenant=tenant_slug,
                result="success",
                actor_email=st.session_state.get("admin_email", "admin"),
                context={
                    "count": success_count,
                    "emails": emails,
                    "new_status": new_status
                }
            )
            
            # 成功後增加 counter，使 Select Members 清空
            st.session_state["batch_ops_counter"] = st.session_state.get("batch_ops_counter", 0) + 1
            import time
            time.sleep(2)
            st.rerun()
        else:
            st.error("❌ No members were updated.")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def show_revoke():
    """撤權管理（Phase B4.1 實作）- 一鍵撤權 per tenant"""
    st.header("🚫 Emergency Access Revocation")
    
    supabase_url, service_key = get_supabase_client()
    if not supabase_url or not service_key:
        st.error("⚠️ Supabase not configured")
        return
    
    st.markdown("""
    **Revoke all sessions** for a tenant by bumping its epoch. All existing sessions will immediately require re-login from the Portal.
    """)
    
    # 獲取租戶列表
    tenants = get_all_tenants(supabase_url, service_key)
    if not tenants:
        st.warning("⚠️ No tenants found.")
        return
    
    tenant_options = [f"{t['slug']} - {t['name']}" for t in tenants]
    selected = st.selectbox("Select Tenant to Revoke", tenant_options, key="revoke_tenant_select")
    tenant_slug = selected.split(" - ")[0]
    
    # 顯示當前 epoch
    current_epoch = get_tenant_epoch(tenant_slug, supabase_url, service_key)
    st.metric("Current Epoch", current_epoch)
    st.caption("Epoch is bumped on revoke; all sessions with older epoch will be invalidated.")
    
    st.markdown("---")
    st.markdown("**⚠️ Confirm Revocation**")
    st.caption(f"Type the tenant slug `{tenant_slug}` to confirm. This will immediately invalidate all active sessions.")
    
    confirm_input = st.text_input(
        "Type tenant slug to confirm",
        placeholder=tenant_slug,
        key="revoke_confirm_input"
    )
    
    if st.button("🚨 Revoke All Sessions", type="primary"):
        if confirm_input != tenant_slug:
            st.error(f"❌ Confirmation failed. Type `{tenant_slug}` exactly to revoke.")
        else:
            revoke_tenant_sessions(tenant_slug, supabase_url, service_key)


def revoke_tenant_sessions(tenant_slug: str, supabase_url: str, service_key: str):
    """
    撤銷指定租戶的所有 session（bump epoch）
    
    流程：
    1. 取得當前 epoch
    2. 更新 tenant_session_epoch 將 epoch + 1
    3. 記錄 audit_events
    4. 顯示新 epoch
    """
    import time
    from datetime import datetime, timezone
    
    try:
        headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        # 1. 取得當前 epoch
        endpoint = f"{supabase_url}/rest/v1/tenant_session_epoch"
        params = {"tenant": f"eq.{tenant_slug}", "select": "epoch", "limit": "1"}
        resp = requests.get(endpoint, headers=headers, params=params, timeout=10)
        
        if resp.status_code != 200:
            st.error(f"❌ Failed to fetch epoch: HTTP {resp.status_code}")
            return
        
        rows = resp.json()
        if not rows:
            st.error(f"❌ Tenant '{tenant_slug}' has no epoch record. Create tenant first or run epoch init SQL.")
            return
        
        old_epoch = int(rows[0].get("epoch", 0))
        new_epoch = old_epoch + 1
        
        # 2. 更新 epoch（PATCH by tenant slug）
        patch_endpoint = f"{supabase_url}/rest/v1/tenant_session_epoch?tenant=eq.{tenant_slug}"
        patch_payload = {
            "epoch": new_epoch,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        patch_resp = requests.patch(patch_endpoint, json=patch_payload, headers=headers, timeout=10)
        
        if patch_resp.status_code not in [200, 204]:
            st.error(f"❌ Failed to bump epoch: HTTP {patch_resp.status_code}")
            return
        
        # 3. 記錄 audit event
        _log_audit_event(
            action="epoch_revoke",
            tenant=tenant_slug,
            result="success",
            actor_email=st.session_state.get("admin_email", "admin"),
            context={
                "old_epoch": old_epoch,
                "new_epoch": new_epoch,
                "source": "admin_ui_revoke"
            }
        )
        
        # 4. 顯示成功訊息與新 epoch
        st.success(f"✅ **Revoked!** All sessions for `{tenant_slug}` are now invalid.")
        st.info(f"**New epoch:** {new_epoch} (was {old_epoch})")
        st.caption("Users must re-enter from the Portal to get a new session.")
        
        time.sleep(2)
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error revoking sessions: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        _log_audit_event(
            action="epoch_revoke",
            tenant=tenant_slug,
            result="error",
            actor_email=st.session_state.get("admin_email", "admin"),
            context={"error": str(e)}
        )

def show_usage():
    """用量管理（Phase B6.1 實作）- 今日用量、caps 設定"""
    st.header("📊 Usage & Caps Management")
    
    supabase_url, service_key = get_supabase_client()
    if not supabase_url or not service_key:
        st.error("⚠️ Supabase not configured")
        return
    
    st.markdown("View today's usage and set daily caps per tenant. **0** = disabled, **Unlimited** = no cap.")
    
    tenants = get_all_tenants(supabase_url, service_key)
    if not tenants:
        st.warning("No tenants found.")
        return
    
    # 快速搜尋（支援 slug/name 或 member email，例如 individual、coco@gmail.com）
    usage_search = st.text_input("🔍 Search tenants", placeholder="Type slug, name, or member email (e.g. individual, coco@gmail.com)...", key="usage_tenant_search")
    if usage_search:
        q = usage_search.strip().lower()
        by_slug_name = [t for t in tenants if q in (t.get('slug') or '').lower() or q in (t.get('name') or '').lower()]
        by_member = get_tenant_ids_by_member_email_search(usage_search, supabase_url, service_key)
        tenants_filtered = [t for t in tenants if t['id'] in by_member or t in by_slug_name]
        tenants = list({t['id']: t for t in tenants_filtered}.values())  # 去重
        if not tenants:
            st.info("No tenants match your search (by slug, name, or member email).")
            return
    
    for tenant in tenants:
        tenant_id = tenant["id"]
        slug = tenant["slug"]
        name = tenant.get("name", slug)
        caps = get_tenant_usage_caps(tenant_id, supabase_url, service_key)
        review_count, download_count = get_tenant_today_usage_full(tenant_id, supabase_url, service_key)
        
        review_cap = caps.get("daily_review_cap")
        download_cap = caps.get("daily_download_cap")
        cap_id = caps.get("id")
        
        # 顯示標題與用量
        review_status = "Unlimited" if review_cap is None else f"{review_count} / {review_cap}"
        download_status = "Unlimited" if download_cap is None else f"{download_count} / {download_cap}"
        
        with st.expander(f"**{slug}** - {name} | Review: {review_status} | Download: {download_status}", expanded=False):
            # 各 member 今日用量（individual 等租戶可清楚看到個別狀況）
            members_usage = get_tenant_members_usage_today(tenant_id, supabase_url, service_key)
            members_list = get_members(supabase_url, service_key, slug)
            if members_list or members_usage:
                st.markdown("**👥 Members & Today's Usage**")
                usage_by_email = {m["email"]: m for m in members_usage}
                all_emails = {m.get("email") for m in members_list if m.get("email")} | {m.get("email") for m in members_usage if m.get("email")}
                for em in sorted(all_emails, key=lambda x: (x or "").lower()):
                    u = usage_by_email.get((em or "").lower(), {"review": 0, "download": 0})
                    st.caption(f"• **{em}**: Review {u.get('review', 0)}, Download {u.get('download', 0)}")
                st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📈 Today's Usage (Total)**")
                st.metric("Review", review_count)
                st.metric("Download", download_count)
                # 進度條（有 cap 時）
                if review_cap is not None and review_cap > 0:
                    pct = min(1.0, review_count / review_cap)
                    st.progress(pct)
                    st.caption(f"Review: {review_count}/{review_cap} ({int(pct*100)}%)")
                if download_cap is not None and download_cap > 0:
                    pct = min(1.0, download_count / download_cap)
                    st.progress(pct)
                    st.caption(f"Download: {download_count}/{download_cap} ({int(pct*100)}%)")
            
            with col2:
                st.markdown("**⚙️ Set Caps**")
                if not cap_id:
                    st.caption("No caps record. This tenant was created without caps (e.g. Individual/Guest). Click below to initialize.")
                    if st.button("➕ Initialize Caps", key=f"init_caps_{tenant_id}"):
                        if init_tenant_usage_caps(tenant_id, supabase_url, service_key, 50, 20):
                            st.success("✅ Caps initialized (default: Review 50, Download 20).")
                            _log_audit_event(
                                action="usage_caps_initialized",
                                tenant=slug,
                                result="success",
                                actor_email=st.session_state.get("admin_email", "admin"),
                                context={"daily_review_cap": 50, "daily_download_cap": 20}
                            )
                            import time
                            time.sleep(1.5)
                            st.rerun()
                        else:
                            st.error("❌ Failed to initialize caps.")
                    continue
                with st.form(f"caps_form_{tenant_id}"):
                    rev_val = review_cap if review_cap is not None else 50
                    dwn_val = download_cap if download_cap is not None else 20
                    rev_unlimited = st.checkbox("Review: Unlimited", value=(review_cap is None), key=f"rev_unl_{tenant_id}")
                    if not rev_unlimited:
                        new_review_cap = st.number_input("Daily Review Cap", min_value=0, value=rev_val if rev_val else 50, key=f"rev_cap_{tenant_id}")
                    else:
                        new_review_cap = None
                    dwn_unlimited = st.checkbox("Download: Unlimited", value=(download_cap is None), key=f"dwn_unl_{tenant_id}")
                    if not dwn_unlimited:
                        new_download_cap = st.number_input("Daily Download Cap", min_value=0, value=dwn_val if dwn_val else 20, key=f"dwn_cap_{tenant_id}")
                    else:
                        new_download_cap = None
                    if st.form_submit_button("💾 Save Caps"):
                        ok = update_tenant_usage_caps(
                            cap_id, new_review_cap, new_download_cap,
                            supabase_url, service_key
                        )
                        if ok:
                            st.success("✅ Caps updated!")
                            _log_audit_event(
                                action="usage_caps_updated",
                                tenant=slug,
                                result="success",
                                actor_email=st.session_state.get("admin_email", "admin"),
                                context={
                                    "daily_review_cap": new_review_cap,
                                    "daily_download_cap": new_download_cap
                                }
                            )
                            import time
                            time.sleep(1.5)
                            st.rerun()
                        else:
                            st.error("❌ Failed to update caps.")

def show_audit_logs():
    """審計日誌（Phase B5.1 實作）- Audit Log 列表、篩選、詳情"""
    st.header("📜 Audit Logs")
    
    supabase_url, service_key = get_supabase_client()
    if not supabase_url or not service_key:
        st.error("⚠️ Supabase not configured")
        return
    
    st.markdown("View audit events with filters. Expand a row to see context JSON.")
    
    # 篩選器
    col1, col2, col3 = st.columns(3)
    
    with col1:
        time_range = st.selectbox(
            "Time Range",
            ["Today", "Last 7 days", "Last 30 days"],
            key="audit_time_range"
        )
    
    with col2:
        # 取得所有 distinct actions 供篩選
        tenants = get_all_tenants(supabase_url, service_key)
        tenant_options = ["All Tenants"] + [f"{t['slug']} - {t['name']}" for t in tenants]
        tenant_filter = st.selectbox("Tenant", tenant_options, key="audit_tenant_filter")
    
    with col3:
        action_filter = st.selectbox(
            "Action",
            [
                "All",
                "admin_login",
                "admin_logout",
                "tenant_created",
                "tenant_deleted",
                "tenant_enabled",
                "tenant_disabled",
                "tenant_trial_extended",
                "tenant_trial_date_updated",
                "epoch_revoke",
                "members_batch_added",
                "member_enabled",
                "member_disabled",
                "member_role_updated",
                "members_batch_enabled",
                "members_batch_disabled",
                "member_deleted",
                "members_batch_deleted",
                "sso_verify",
                "analyzer_launch",
                "access_denied"
            ],
            key="audit_action_filter"
        )
    
    result_filter = st.selectbox(
        "Result",
        ["All", "success", "denied", "error"],
        key="audit_result_filter"
    )
    
    # 取得 audit events
    tenant_slug_filter = None
    if tenant_filter and tenant_filter != "All Tenants":
        tenant_slug_filter = tenant_filter.split(" - ")[0]
    events = get_audit_events(
        supabase_url, service_key,
        time_range=time_range,
        tenant_slug=tenant_slug_filter,
        action=action_filter if action_filter != "All" else None,
        result=result_filter if result_filter != "All" else None
    )
    
    if not events:
        st.info("No audit events match the filters.")
        return
    
    st.success(f"Found {len(events)} event(s)")
    
    # 轉成 DataFrame 顯示（選取常用欄位）
    import pandas as pd
    def _ctx_preview(ctx):
        s = json.dumps(ctx or {})
        return (s[:80] + "...") if len(s) > 80 else s
    df = pd.DataFrame([
        {
            "Time": (e.get("created_at") or "")[:19].replace("T", " "),
            "Action": e.get("action", ""),
            "Tenant": e.get("tenant_slug") or "-",
            "Actor": e.get("actor_email") or e.get("email") or "-",
            "Result": e.get("result", ""),
            "Context": _ctx_preview(e.get("context"))
        }
        for e in events
    ])
    
    # 使用 dataframe 顯示，設定 height
    st.dataframe(df, use_container_width=True, height=400)
    
    # 詳情展開：選擇一筆查看 context
    st.markdown("---")
    st.markdown("**View Details**")
    event_options = [
        f"{e.get('created_at','')[:19]} | {e.get('action','')} | {e.get('tenant_slug') or '-'} | {e.get('result','')}"
        for e in events
    ]
    selected_idx = st.selectbox(
        "Select an event to view context",
        range(len(events)),
        format_func=lambda i: event_options[i],
        key="audit_detail_select"
    )
    if selected_idx is not None and 0 <= selected_idx < len(events):
        ev = events[selected_idx]
        with st.expander("Context (JSON)", expanded=True):
            ctx = ev.get("context") or {}
            if ctx:
                st.json(ctx)
            else:
                st.caption("No context")
        if ev.get("deny_reason"):
            st.caption(f"Deny reason: {ev['deny_reason']}")
        if ev.get("notes"):
            st.caption(f"Notes: {ev['notes']}")

# ==========================================
# 主程式入口
# ==========================================
def main():
    """主程式"""
    
    # 檢查登入狀態
    if not check_password():
        st.stop()
    
    # 顯示管理儀表板
    show_admin_dashboard()

if __name__ == "__main__":
    main()
