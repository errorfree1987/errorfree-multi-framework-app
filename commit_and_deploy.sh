#!/bin/bash

# ============================================
# Git Commit & Deploy 準備腳本
# ============================================
# 用途：在驗收通過後，快速提交並部署到 Railway
# 使用方法：./commit_and_deploy.sh
# ============================================

set -e

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Git Commit & Deploy 準備${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# ===== 檢查是否在正確的目錄 =====
if [ ! -f "app.py" ]; then
    echo -e "${RED}❌ 錯誤：請在專案根目錄執行此腳本${NC}"
    exit 1
fi

# ===== 顯示當前變更 =====
echo -e "${BLUE}📋 當前變更檔案：${NC}"
git status --short
echo ""

# ===== 顯示 app.py 的 diff =====
echo -e "${BLUE}📝 app.py 變更內容：${NC}"
git diff app.py | head -40
echo ""
echo -e "${YELLOW}... (完整 diff 請執行 'git diff app.py')${NC}"
echo ""

# ===== 確認是否已完成測試 =====
echo -e "${YELLOW}⚠️  請確認以下測試項目已完成：${NC}"
echo ""
echo "  ✅ 舊 token (epoch < current) 被拒絕"
echo "  ✅ 錯誤訊息顯示 'Session revoked (tenant epoch mismatch)'"
echo "  ✅ Debug 顯示 token_epoch ≠ current_epoch"
echo "  ✅ 新 token (epoch = current) 可正常訪問"
echo "  ✅ 過期/簽章錯誤的 token 仍顯示 generic error"
echo ""
echo -e "${YELLOW}是否已完成所有測試？ (y/n)：${NC}"
read -r TEST_COMPLETE

if [[ ! "$TEST_COMPLETE" =~ ^[Yy]$ ]]; then
    echo -e "${RED}❌ 請先完成測試後再提交${NC}"
    exit 1
fi

# ===== 詢問是否要 commit 測試腳本 =====
echo ""
echo -e "${YELLOW}是否要一起提交測試文檔（test_epoch_revoke.md, sql_epoch_management.sql 等）？ (y/n)：${NC}"
read -r INCLUDE_TEST_FILES

if [[ "$INCLUDE_TEST_FILES" =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}✅ 將包含測試文檔${NC}"
    TEST_FILES="test_epoch_revoke.md sql_epoch_management.sql test_epoch_revoke.sh"
else
    echo -e "${YELLOW}⚠️  只提交 app.py${NC}"
    TEST_FILES=""
fi

# ===== 執行 git add =====
echo ""
echo -e "${BLUE}📦 準備提交...${NC}"
git add app.py $TEST_FILES

# ===== 顯示即將提交的內容 =====
echo -e "${GREEN}即將提交的檔案：${NC}"
git status --short
echo ""

# ===== Commit =====
COMMIT_MSG="Fix epoch revoke UX: show specific 'Session revoked' message

- Move epoch mismatch check from verify_analyzer_session to _enforce_epoch_or_block
- Old tokens with mismatched epoch now show 'Session revoked (tenant epoch mismatch)' instead of generic 'No valid Portal SSO parameters'
- Maintains all security checks: signature, expiration, required fields
- Verified: tenant epoch revoke immediately blocks old sessions"

echo -e "${BLUE}📝 Commit 訊息：${NC}"
echo -e "${YELLOW}${COMMIT_MSG}${NC}"
echo ""

echo -e "${YELLOW}是否要執行 commit？ (y/n)：${NC}"
read -r DO_COMMIT

if [[ "$DO_COMMIT" =~ ^[Yy]$ ]]; then
    git commit -m "$COMMIT_MSG"
    echo -e "${GREEN}✅ Commit 完成${NC}"
else
    echo -e "${RED}❌ 已取消 commit${NC}"
    exit 1
fi

# ===== 詢問是否要 push =====
echo ""
echo -e "${YELLOW}是否要 push 到遠端 (main branch)？ (y/n)：${NC}"
read -r DO_PUSH

if [[ "$DO_PUSH" =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}🚀 Pushing to remote...${NC}"
    git push origin main
    echo -e "${GREEN}✅ Push 完成${NC}"
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}🎉 部署完成！${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo -e "${YELLOW}後續步驟：${NC}"
    echo "  1. Railway：專案須為 errorfree-techincal review-app-staging（見 RAILWAY_STAGING_CANONICAL.md）"
    echo "  2. 前往 Dashboard 查看部署狀態"
    echo "  3. 等待部署完成（約 2-3 分鐘）"
    echo "  4. 在環境重新執行驗收測試"
    echo "  5. 記錄測試結果"
else
    echo -e "${YELLOW}⚠️  已 commit 但未 push${NC}"
    echo -e "   稍後可手動執行：${GREEN}git push origin main${NC}"
fi

echo ""
