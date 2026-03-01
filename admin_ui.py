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
            
            # 清除密碼輸入
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
            
            # 重新運行以更新 UI
            st.rerun()
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
    """租戶管理（Phase B2 實作）"""
    st.header("🏢 Tenant Management")
    st.info("🚧 Coming in Phase B2 - Tenant Management")
    st.markdown("""
    **Planned features:**
    - View all tenants (list with search/filter)
    - Create new tenant
    - Extend trial period
    - Enable/disable tenant
    - View tenant details and statistics
    """)

def show_members():
    """成員管理（Phase B3 實作）"""
    st.header("👥 Member Management")
    st.info("🚧 Coming in Phase B3 - Member Management")
    st.markdown("""
    **Planned features:**
    - View members (by tenant)
    - Batch add members (paste emails)
    - Batch enable/disable members
    - Set member roles
    """)

def show_revoke():
    """撤權管理（Phase B4 實作）"""
    st.header("🚫 Emergency Access Revocation")
    st.info("🚧 Coming in Phase B4 - One-Click Revocation")
    st.markdown("""
    **Planned features:**
    - Per-tenant session revocation (epoch bump)
    - Confirmation flow (type tenant slug to confirm)
    - Show current epoch and affected sessions
    - Revocation history
    """)

def show_usage():
    """用量管理（Phase B6 實作）"""
    st.header("📊 Usage & Caps Management")
    st.info("🚧 Coming in Phase B6 - Usage Controls")
    st.markdown("""
    **Planned features:**
    - View today's usage (all tenants)
    - Set/adjust caps per tenant
    - Usage trends (7-day chart)
    - Alerts for tenants approaching limits
    """)

def show_audit_logs():
    """審計日誌（Phase B5 實作）"""
    st.header("📜 Audit Logs")
    st.info("🚧 Coming in Phase B5 - Audit Log Viewer")
    st.markdown("""
    **Planned features:**
    - View all audit events
    - Filter by tenant/action/result/time
    - View detailed context (JSON)
    - Export to CSV
    """)

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
