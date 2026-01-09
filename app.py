# app.py
# ============================================================
# Error-Free® Multi-Framework App (Stable Sequential Analysis)
# Step 5 Logic:
# 1. Analyze MAIN document with Error-Free framework
# 2. Compare MAIN analysis with REFERENCE documents (relevance)
# 3. Analyze the RELEVANCE result again with Error-Free framework
# ============================================================

import os
import json
import streamlit as st
from openai import OpenAI
from pathlib import Path
from docx import Document
import pdfplumber

# ------------------------
# Basic Config
# ------------------------
st.set_page_config(page_title="Error-Free® Analysis Engine", layout="wide")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

FRAMEWORK_FILE = Path("frameworks.json")

# ------------------------
# Utilities
# ------------------------
def zh(tw, cn):
    return cn if st.session_state.get("zh_variant", "tw") == "cn" else tw

def read_file(uploaded):
    if not uploaded:
        return ""
    name = uploaded.name.lower()
    try:
        if name.endswith(".pdf"):
            text = []
            with pdfplumber.open(uploaded) as pdf:
                for p in pdf.pages:
                    text.append(p.extract_text() or "")
            return "\n".join(text)
        if name.endswith(".docx"):
            doc = Document(uploaded)
            return "\n".join(p.text for p in doc.paragraphs)
        if name.endswith(".txt"):
            return uploaded.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return f"[FILE READ ERROR] {e}"
    return ""

def load_frameworks():
    if not FRAMEWORK_FILE.exists():
        return {}
    return json.loads(FRAMEWORK_FILE.read_text(encoding="utf-8"))

FRAMEWORKS = load_frameworks()

def call_openai(system_prompt, user_prompt, max_tokens=2000):
    if not client:
        return "[ERROR] OPENAI_API_KEY not set."
    try:
        res = client.responses.create(
            model="gpt-5.1",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_output_tokens=max_tokens,
        )
        return res.output_text.strip()
    except Exception as e:
        return f"[OpenAI API ERROR] {e}"

# ------------------------
# Language
# ------------------------
with st.sidebar:
    lang = st.radio("Language / 語言", ["中文繁體", "中文简体", "English"])
    if lang == "English":
        st.session_state.lang = "en"
    else:
        st.session_state.lang = "zh"
        st.session_state.zh_variant = "cn" if lang == "中文简体" else "tw"

# ------------------------
# UI
# ------------------------
st.title(zh("零錯誤分析引擎", "零错误分析引擎") if st.session_state.lang == "zh" else "Error-Free Analysis Engine")

if not FRAMEWORKS:
    st.error("frameworks.json not found")
    st.stop()

# Step 1
st.header(zh("步驟一：上傳主要文件", "步骤一：上传主要文件"))
main_file = st.file_uploader("Main Document", type=["pdf", "docx", "txt"])
main_text = read_file(main_file)
main_name = main_file.name if main_file else ""

# Step 2
st.header(zh("步驟二：文件類型", "步骤二：文件类型"))
doc_type = st.selectbox(
    zh("選擇文件類型", "选择文件类型"),
    ["規格與需求", "安全分析", "設計文件", "根因分析"]
)

# Step 3
st.header(zh("步驟三：參考文件（可多份）", "步骤三：参考文件（可多份）"))
ref_files = st.file_uploader(
    zh("上傳參考文件", "上传参考文件"),
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

ref_texts = []
ref_names = []
if ref_files:
    for f in ref_files:
        ref_texts.append(read_file(f))
        ref_names.append(f.name)

# Step 4
st.header(zh("步驟四：選擇零錯誤分析框架", "步骤四：选择零错误分析框架"))
fw_key = st.selectbox(
    zh("選擇框架", "选择框架"),
    list(FRAMEWORKS.keys()),
    format_func=lambda k: FRAMEWORKS[k]["name_zh"] if st.session_state.lang == "zh" else FRAMEWORKS[k]["name_en"]
)

framework = FRAMEWORKS[fw_key]
wrapper = framework["wrapper_zh"] if st.session_state.lang == "zh" else framework["wrapper_en"]

# ------------------------
# Step 5 Core Logic (YOUR DESIGN)
# ------------------------
st.header(zh("步驟五：零錯誤同步分析", "步骤五：零错误同步分析"))

if st.button(zh("開始分析", "开始分析")):
    if not main_text:
        st.error(zh("請先上傳主要文件", "请先上传主要文件"))
        st.stop()

    with st.spinner(zh("第一階段：分析主要文件...", "第一阶段：分析主要文件...")):
        main_analysis = call_openai(
            wrapper,
            f"""
你正在使用零錯誤框架，請對以下【主要文件】進行「整體、完整、不分段」的分析。
文件類型：{doc_type}

【主要文件內容】
{main_text}
""",
            max_tokens=1800
        )

    with st.spinner(zh("第二階段：比對參考文件相關性...", "第二阶段：比对参考文件相关性...")):
        relevance_result = call_openai(
            wrapper,
            f"""
以下是「主要文件的分析結果」，請你將它與參考文件進行相關性比對，
找出支持、衝突、補強、不一致之處。

【主要文件分析結果】
{main_analysis}

【參考文件內容】
{chr(10).join(ref_texts)}
""",
            max_tokens=1800
        )

    with st.spinner(zh("第三階段：零錯誤最終分析...", "第三阶段：零错误最终分析...")):
        final_result = call_openai(
            wrapper,
            f"""
請使用零錯誤框架，對以下「相關性比對結果」進行最終分析，
並產出可直接交付的正式分析報告。

【相關性比對結果】
{relevance_result}
""",
            max_tokens=2200
        )

    st.success(zh("分析完成", "分析完成"))
    st.markdown("---")
    st.subheader(zh("分析結果", "分析结果"))
    st.markdown(final_result)
