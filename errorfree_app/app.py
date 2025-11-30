extra_file = None
col_upload, col_input = st.columns([0.15, 0.85])

with col_upload:
    extra_file = st.file_uploader(
        "➕",
        type=["pdf", "docx", "txt", "jpg", "jpeg", "png"],
        key=f"extra_{selected_key}",
        label_visibility="collapsed",
    )
    if lang == "zh":
        st.caption("➕ 上傳圖片或文件（選填）")
    else:
        st.caption("➕ Upload image / file (optional)")

extra_text = read_file_to_text(extra_file) if extra_file else ""

with col_input:
    # 這裡是 chat_input（見上方第 4 點）
    prompt = st.chat_input(followup_prompt_label)
