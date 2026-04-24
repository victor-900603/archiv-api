import streamlit as st
from api.rag import query_rag

st.set_page_config(page_title="Archive API", page_icon=":material/robot:")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.chat_message("AI"):
    st.markdown("歡迎使用 Archive API！請輸入您的問題，我會盡力回答。")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("說點什麼吧..."):
    # 顯示使用者訊息
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        response = query_rag(prompt, st.session_state.messages[:-1])

        if response["status"] == "success":
            answer = response["answer"]
            with st.chat_message("assistant"):
                st.markdown(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            with st.chat_message("assistant"):
                st.markdown("抱歉，發生錯誤！")

            st.session_state.messages.append({"role": "assistant", "content": "抱歉，發生錯誤！"})
    except Exception as exc:
        error_message = f"後端服務連線失敗：{exc}"
        with st.chat_message("assistant"):
            st.markdown(error_message)

        st.session_state.messages.append({"role": "assistant", "content": error_message})