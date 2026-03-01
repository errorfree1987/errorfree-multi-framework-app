#!/bin/bash

# Phase B1.1 本地測試腳本
# 用途：快速設定環境變數並啟動 Admin UI 進行本地測試

echo "🚀 Phase B1.1 本地測試腳本"
echo "================================"
echo ""

# 檢查是否在正確的目錄
if [ ! -f "admin_ui.py" ]; then
    echo "❌ 錯誤：找不到 admin_ui.py"
    echo "請確認你在正確的專案目錄中"
    exit 1
fi

echo "✅ 找到 admin_ui.py"
echo ""

# 設定環境變數
echo "📝 設定環境變數..."
echo ""

# Admin 密碼（測試用）
export ADMIN_PASSWORD="test123"
echo "✅ ADMIN_PASSWORD=test123 (測試用)"

# Supabase URL（需要你提供）
if [ -z "$SUPABASE_URL" ]; then
    echo "⚠️  SUPABASE_URL 未設定"
    echo "請手動設定："
    echo "  export SUPABASE_URL='https://xxxxx.supabase.co'"
    echo ""
    read -p "輸入你的 SUPABASE_URL（或按 Enter 跳過）: " input_url
    if [ ! -z "$input_url" ]; then
        export SUPABASE_URL="$input_url"
        echo "✅ SUPABASE_URL 已設定"
    fi
else
    echo "✅ SUPABASE_URL 已設定"
fi

# Supabase Service Key（需要你提供）
if [ -z "$SUPABASE_SERVICE_KEY" ]; then
    echo "⚠️  SUPABASE_SERVICE_KEY 未設定"
    echo "請手動設定："
    echo "  export SUPABASE_SERVICE_KEY='eyJhbGc...'"
    echo ""
    read -p "輸入你的 SUPABASE_SERVICE_KEY（或按 Enter 跳過）: " input_key
    if [ ! -z "$input_key" ]; then
        export SUPABASE_SERVICE_KEY="$input_key"
        echo "✅ SUPABASE_SERVICE_KEY 已設定"
    fi
else
    echo "✅ SUPABASE_SERVICE_KEY 已設定"
fi

echo ""
echo "================================"
echo "🎯 測試重點："
echo "================================"
echo "1. 登入測試"
echo "   - 密碼：test123（應該成功）"
echo "   - 密碼：wrong（應該失敗）"
echo ""
echo "2. Session 測試"
echo "   - 登入後重新整理（應該保持登入狀態）"
echo ""
echo "3. 登出測試"
echo "   - 點擊右上角 Logout（應該回到登入頁）"
echo ""
echo "4. Supabase 連線"
echo "   - Dashboard 應該顯示 '✅ Supabase connected'"
echo ""
echo "================================"
echo ""

# 啟動 Streamlit
echo "🚀 啟動 Admin UI..."
echo "瀏覽器會自動開啟 http://localhost:8501"
echo ""
echo "按 Ctrl+C 停止"
echo ""

streamlit run admin_ui.py
