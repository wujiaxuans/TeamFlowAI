"""
TeamFlowAI RAG 问答系统 - Streamlit Web 界面
提供文件上传、交互式问答、来源引用展示、文档生成功能。
"""

import os
import streamlit as st
from dotenv import load_dotenv
from rag import create_qa_chain, ask, get_embeddings, get_or_create_vectorstore, PDF_DIR, DOCS_DIR
from agent import create_content_agent, run_agent

load_dotenv()

# --- 页面配置 ---
st.set_page_config(page_title="TeamFlowAI", page_icon="📖", layout="wide")


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
    if "content_agent" not in st.session_state:
        st.session_state.content_agent = None
    if "generated_doc" not in st.session_state:
        st.session_state.generated_doc = ""


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
        st.error("未检测到 GLM_API_KEY")
        return False
    with st.spinner("正在加载知识库..."):
        chain = create_qa_chain(api_key)
    if chain:
        st.session_state.qa_chain = chain
        st.session_state.vectorstore_ready = True
        return True
    st.warning("知识库为空，请先上传文档")
    return False


def load_content_agent():
    """初始化文档生成 Agent"""
    api_key = _get_api_key()
    if not api_key:
        st.error("未检测到 GLM_API_KEY")
        return False
    try:
        with st.spinner("正在初始化 Agent..."):
            st.session_state.content_agent = create_content_agent(api_key)
        return True
    except ValueError as e:
        st.error(str(e))
        return False


def reload_vectorstore():
    """增量更新向量库"""
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

    # 操作按钮
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    if st.button("🔄 重新加载知识库", use_container_width=True):
        reload_vectorstore()
        st.rerun()

    if not st.session_state.vectorstore_ready:
        if st.button("🚀 启动知识库", use_container_width=True, type="primary"):
            load_qa_chain()


# --- 主区域：标签页 ---
tab1, tab2 = st.tabs(["💬 RAG 问答", "📝 文档生成"])

# ========== Tab 1: RAG 问答 ==========
with tab1:
    st.header("💬 RAG 问答")

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
        if not st.session_state.qa_chain:
            if not load_qa_chain():
                st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                result = ask(st.session_state.qa_chain, prompt, st.session_state.chat_history)

            st.markdown(result["answer"])

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

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": sources,
        })
        st.session_state.chat_history.append((prompt, result["answer"]))


# ========== Tab 2: 文档生成 ==========
with tab2:
    st.header("📝 AI 文档生成")
    st.caption("基于知识库自动生成 PRD、会议纪要、周报等结构化文档")

    # 文档类型选择
    doc_type = st.selectbox(
        "文档类型",
        options=["PRD", "会议纪要", "周报"],
        index=0,
    )

    # 需求描述输入
    requirement = st.text_area(
        "需求描述",
        placeholder="例如：帮我写一份关于 MCP 协议的 PRD",
        height=100,
    )

    # 生成按钮
    if st.button("🚀 生成文档", type="primary"):
        if not requirement.strip():
            st.warning("请输入需求描述")
        else:
            if not st.session_state.content_agent:
                if not load_content_agent():
                    st.stop()

            full_input = f"帮我写一份{requirement.strip()}，文档类型是{doc_type}"

            with st.spinner("Agent 正在生成文档..."):
                output = run_agent(st.session_state.content_agent, full_input)

            st.session_state.generated_doc = output

    # 展示生成结果
    if st.session_state.generated_doc:
        st.divider()
        st.subheader("📄 生成结果")

        st.markdown(st.session_state.generated_doc)

        # 下载按钮
        st.download_button(
            label="⬇️ 下载 Markdown",
            data=st.session_state.generated_doc,
            file_name=f"{doc_type}.md",
            mime="text/markdown",
        )