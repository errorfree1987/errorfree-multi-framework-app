#!/bin/bash

# ============================================
# 多租戶緊急撤權（Epoch Revoke）自動化測試腳本
# ============================================
# 用途：自動執行 epoch revoke 測試流程
# 使用方法：
#   chmod +x test_epoch_revoke.sh
#   ./test_epoch_revoke.sh abc
# ============================================

set -e  # Exit on error

# ===== 配置區 =====
TENANT="${1:-abc}"  # 第一個參數為租戶名，預設 abc
ANALYZER_URL="${ANALYZER_URL:-https://your-analyzer.railway.app}"
SUPABASE_URL="${SUPABASE_URL}"
SUPABASE_SERVICE_KEY="${SUPABASE_SERVICE_KEY}"

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ===== 檢查必要環境變數 =====
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Epoch Revoke 測試腳本${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if [ -z "$SUPABASE_URL" ] || [ -z "$SUPABASE_SERVICE_KEY" ]; then
    echo -e "${RED}❌ 錯誤：缺少必要的環境變數${NC}"
    echo "請設定："
    echo "  export SUPABASE_URL=https://xxxxx.supabase.co"
    echo "  export SUPABASE_SERVICE_KEY=eyJhbGc..."
    exit 1
fi

echo -e "${GREEN}✅ 環境變數檢查通過${NC}"
echo -e "   Tenant: ${YELLOW}${TENANT}${NC}"
echo -e "   Analyzer: ${ANALYZER_URL}"
echo ""

# ===== 函數：查詢 Supabase epoch =====
get_epoch() {
    local tenant=$1
    local endpoint="${SUPABASE_URL}/rest/v1/tenant_session_epoch"
    
    local response=$(curl -s \
        -H "apikey: ${SUPABASE_SERVICE_KEY}" \
        -H "Authorization: Bearer ${SUPABASE_SERVICE_KEY}" \
        -G \
        --data-urlencode "select=epoch" \
        --data-urlencode "tenant=eq.${tenant}" \
        --data-urlencode "limit=1" \
        "${endpoint}")
    
    # 解析 JSON（使用 jq 如果有，否則用 grep/sed）
    if command -v jq &> /dev/null; then
        echo "$response" | jq -r '.[0].epoch // 0'
    else
        echo "$response" | grep -o '"epoch":[0-9]*' | grep -o '[0-9]*' || echo "0"
    fi
}

# ===== 函數：增加 epoch =====
bump_epoch() {
    local tenant=$1
    local endpoint="${SUPABASE_URL}/rest/v1/tenant_session_epoch"
    
    # 使用 PATCH 更新（RPC 方式）
    curl -s -X PATCH \
        -H "apikey: ${SUPABASE_SERVICE_KEY}" \
        -H "Authorization: Bearer ${SUPABASE_SERVICE_KEY}" \
        -H "Content-Type: application/json" \
        -H "Prefer: return=representation" \
        --data "{\"epoch\": \"epoch + 1\", \"updated_at\": \"now()\"}" \
        "${endpoint}?tenant=eq.${tenant}" > /dev/null
}

# ===== Step 1: 查詢初始 epoch =====
echo -e "${BLUE}📊 Step 1: 查詢租戶當前 epoch${NC}"
INITIAL_EPOCH=$(get_epoch "$TENANT")
echo -e "   當前 epoch: ${YELLOW}${INITIAL_EPOCH}${NC}"
echo ""

if [ "$INITIAL_EPOCH" = "0" ] || [ -z "$INITIAL_EPOCH" ]; then
    echo -e "${YELLOW}⚠️  警告：租戶 epoch 為 0 或不存在${NC}"
    echo -e "   建議先在 Supabase 執行："
    echo -e "   ${YELLOW}INSERT INTO tenant_session_epoch (tenant, epoch, updated_at) VALUES ('${TENANT}', 0, NOW()) ON CONFLICT (tenant) DO NOTHING;${NC}"
    echo ""
fi

# ===== Step 2: 提示用戶取得舊 token URL =====
echo -e "${BLUE}📝 Step 2: 準備舊 token URL${NC}"
echo -e "   ${YELLOW}請執行以下操作：${NC}"
echo -e "   1. 開啟瀏覽器，從 Portal 以 ${YELLOW}${TENANT}${NC} 租戶登入"
echo -e "   2. 進入 Analyzer"
echo -e "   3. 在 URL 後加上 ${YELLOW}&debug_epoch=1${NC}"
echo -e "   4. 確認看到 ${GREEN}token_epoch=${INITIAL_EPOCH} current_epoch=${INITIAL_EPOCH}${NC}"
echo -e "   5. 複製完整 URL"
echo ""
echo -e "${YELLOW}請將完整 URL 貼到下面（含 analyzer_session 參數）：${NC}"
read -r OLD_TOKEN_URL

if [ -z "$OLD_TOKEN_URL" ]; then
    echo -e "${RED}❌ 錯誤：未輸入 URL${NC}"
    exit 1
fi

# 檢查 URL 是否包含 analyzer_session
if [[ ! "$OLD_TOKEN_URL" =~ analyzer_session= ]]; then
    echo -e "${RED}❌ 錯誤：URL 缺少 analyzer_session 參數${NC}"
    exit 1
fi

# 確保 URL 包含 debug_epoch=1
if [[ ! "$OLD_TOKEN_URL" =~ debug_epoch=1 ]]; then
    if [[ "$OLD_TOKEN_URL" =~ \? ]]; then
        OLD_TOKEN_URL="${OLD_TOKEN_URL}&debug_epoch=1"
    else
        OLD_TOKEN_URL="${OLD_TOKEN_URL}?debug_epoch=1"
    fi
    echo -e "${YELLOW}已自動加上 debug_epoch=1${NC}"
fi

echo -e "${GREEN}✅ 舊 token URL 已記錄${NC}"
echo ""

# ===== Step 3: Bump epoch =====
echo -e "${BLUE}🚨 Step 3: 執行緊急撤權（Bump epoch）${NC}"
echo -e "   ${YELLOW}即將將 ${TENANT} 的 epoch 從 ${INITIAL_EPOCH} 增加到 $((INITIAL_EPOCH + 1))${NC}"
echo -e "   ${RED}這會讓所有該租戶的現有 session 失效！${NC}"
echo -e "   ${YELLOW}按 Enter 繼續，或 Ctrl+C 取消...${NC}"
read -r

echo -e "   正在更新 epoch..."
bump_epoch "$TENANT"
sleep 2  # 等待 DB 更新

NEW_EPOCH=$(get_epoch "$TENANT")
echo -e "   ${GREEN}✅ Epoch 已更新：${INITIAL_EPOCH} → ${NEW_EPOCH}${NC}"
echo ""

if [ "$NEW_EPOCH" -le "$INITIAL_EPOCH" ]; then
    echo -e "${RED}❌ 錯誤：Epoch 未成功增加${NC}"
    echo -e "   請手動在 Supabase SQL Editor 執行："
    echo -e "   ${YELLOW}UPDATE tenant_session_epoch SET epoch = epoch + 1, updated_at = NOW() WHERE tenant = '${TENANT}';${NC}"
    exit 1
fi

# ===== Step 4: 提示用戶測試撤權 =====
echo -e "${BLUE}🧪 Step 4: 測試撤權效果${NC}"
echo -e "   ${YELLOW}請執行以下操作：${NC}"
echo -e "   1. 開啟${YELLOW}新的無痕視窗/私密瀏覽${NC}"
echo -e "   2. 直接貼上以下 URL：${NC}"
echo ""
echo -e "   ${GREEN}${OLD_TOKEN_URL}${NC}"
echo ""
echo -e "   ${YELLOW}預期結果：${NC}"
echo -e "   ${GREEN}✅ 應該看到：Session revoked (tenant epoch mismatch)${NC}"
echo -e "   ${GREEN}✅ Debug 顯示：token_epoch=${INITIAL_EPOCH} current_epoch=${NEW_EPOCH}${NC}"
echo -e "   ${RED}❌ 不應該看到：No valid Portal SSO parameters${NC}"
echo ""
echo -e "${YELLOW}測試完成後按 Enter 繼續...${NC}"
read -r

# ===== Step 5: 詢問是否恢復 epoch =====
echo -e "${BLUE}🔄 Step 5: 測試後清理${NC}"
echo -e "   是否要將 epoch 恢復到原始值 ${INITIAL_EPOCH}？"
echo -e "   ${YELLOW}(y/n，預設 n)：${NC}"
read -r RESTORE_CHOICE

if [[ "$RESTORE_CHOICE" =~ ^[Yy]$ ]]; then
    echo -e "   正在恢復 epoch..."
    
    # 使用 SQL 直接設定 epoch（需要 Supabase RPC 或直接 SQL）
    echo -e "${YELLOW}請在 Supabase SQL Editor 手動執行：${NC}"
    echo -e "${GREEN}UPDATE tenant_session_epoch SET epoch = ${INITIAL_EPOCH}, updated_at = NOW() WHERE tenant = '${TENANT}';${NC}"
else
    echo -e "   ${GREEN}保持新 epoch: ${NEW_EPOCH}${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✅ 測試完成！${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}測試摘要：${NC}"
echo -e "   租戶: ${TENANT}"
echo -e "   初始 epoch: ${INITIAL_EPOCH}"
echo -e "   新 epoch: ${NEW_EPOCH}"
echo -e "   舊 token URL: ${OLD_TOKEN_URL:0:80}..."
echo ""
echo -e "${YELLOW}後續步驟：${NC}"
echo -e "   1. 確認測試結果符合預期"
echo -e "   2. 記錄測試結果到 test_epoch_revoke.md"
echo -e "   3. 如果測試通過，可以 commit 並 push 程式碼"
echo ""
