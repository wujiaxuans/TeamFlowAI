"""
TeamFlowAI RAG 问答系统 - Streamlit Web 界面
提供文件上传、交互式问答、来源引用展示功能。
"""

import os
import streamlit as st
from dotenv import load_dotenv
from rag import create_qa_chain, ask, get_embeddings, get_or_create_vectorstore, PDF_DIR, DOCS_DIR

load_dotenv()

# --- 页面配置 ---
st.set_page_config(page_title="TeamFlowAI RAG", page_icon="📖", layout="wide")
st.title("📖 TeamFlowAI RAG 问答系统")


# --- 初始化 session state ---
def init_state():
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore_ready" not in st.session_state:
        st.session_state.vectorstore_ready = False


init_state()


def _get_api_key() -> str | None:
    """获取 API Key，兼容本地 .env 和 Streamlit Cloud Secrets"""
    key = os.getenv("GLM_API_KEY")
    if not key:
        try:
            key = st.secrets["GLM_API_KEY"]
        except (AttributeError, KeyError):
            pass
    return key


def load_qa_chain():
    """初始化或重建 QA 链"""
    api_key = _get_api_key()
    if not api_key:
        st.error("未检测到 GLM_API_KEY，请在 .env 或 Streamlit Secrets 中配置")
        return False
    with st.spinner("正在加载知识库..."):
        chain = create_qa_chain(api_key)
    if chain:
        st.session_state.qa_chain = chain
        st.session_state.vectorstore_ready = True
        return True
    st.warning("知识库为空，请先上传文档")
    return False


def reload_vectorstore():
    """增量更新向量库（处理新上传的文件）"""
    api_key = _get_api_key()
    if not api_key:
        return
    with st.spinner("正在更新知识库..."):
        embeddings = get_embeddings(api_key)
        vs = get_or_create_vectorstore(embeddings)
        if vs:
            load_qa_chain()


# --- 侧边栏：文件上传 ---
with st.sidebar:
    st.header("📁 文档管理")

    uploaded_files = st.file_uploader(
        "上传文档",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="支持 PDF、TXT、Markdown 文件",
    )

    if uploaded_files:
        for f in uploaded_files:
            if f.name.lower().endswith(".pdf"):
                save_dir = PDF_DIR
            else:
                save_dir = DOCS_DIR
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f.name)
            if not os.path.exists(save_path):
                with open(save_path, "wb") as out:
                    out.write(f.getbuffer())
                st.success(f"已上传: {f.name}")
            else:
                st.info(f"已存在: {f.name}")

    # 文档列表
    st.divider()
    st.subheader("已加载文档")

    doc_list = []
    if os.path.exists(DOCS_DIR):
        for f in sorted(os.listdir(DOCS_DIR)):
            if f.lower().endswith((".txt", ".md")):
                doc_list.append(f"📄 {f}")
    if os.path.exists(PDF_DIR):
        for f in sorted(os.listdir(PDF_DIR)):
            if f.lower().endswith(".pdf"):
                doc_list.append(f"📕 {f}")

    if doc_list:
        for item in doc_list:
            st.text(item)
    else:
        st.caption("暂无文档")

    # 重新加载按钮
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    if st.button("🔄 重新加载知识库", use_container_width=True):
        reload_vectorstore()
        st.rerun()

    # 首次自动加载
    if not st.session_state.vectorstore_ready:
        if st.button("🚀 启动知识库", use_container_width=True, type="primary"):
            load_qa_chain()


# --- 主区域：对话 ---
# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📎 查看引用来源"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['file']}**")
                    st.caption(src["content"][:300])
                    st.divider()

# 问答输入
if prompt := st.chat_input("输入你的问题..."):
    # 检查 QA 链是否就绪
    if not st.session_state.qa_chain:
        if not load_qa_chain():
            st.stop()

    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用 RAG 获取回答
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            result = ask(st.session_state.qa_chain, prompt, st.session_state.chat_history)

        st.markdown(result["answer"])

        # 整理来源（去重）
        sources = []
        seen = set()
        for doc in result["sources"]:
            file_name = os.path.basename(doc.metadata.get("source", "未知"))
            content = doc.page_content
            key = (file_name, content[:100])
            if key not in seen:
                seen.add(key)
                sources.append({"file": file_name, "content": content})

        if sources:
            with st.expander("📎 查看引用来源"):
                for src in sources:
                    st.markdown(f"**{src['file']}**")
                    st.caption(src["content"][:300])
                    st.divider()

    # 保存助手消息
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": sources,
    })
    # 追加对话历史（用于 ConversationalRetrievalChain 问题改写）
    st.session_state.chat_history.append((prompt, result["answer"]))
