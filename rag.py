"""
RAG 核心模块
  1. 从 docs/ 加载 .txt/.md 文档（PDF 通过 MinerU 自动转换）
  2. 文本分割 → 智谱 Embedding → ChromaDB 向量存储
  3. 混合检索（向量 + BM25） + ConversationalRetrievalChain 多轮问答
"""

import os
import sys
import shutil
import tempfile
import subprocess
import time
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import PromptTemplate

from prompts import QA_PROMPT, CONDENSE_PROMPT

# 智谱 GLM API 地址（OpenAI 兼容接口）
ZHIPU_API_BASE = "https://open.bigmodel.cn/api/paas/v4"
# ChromaDB 向量库持久化目录
CHROMA_PERSIST_DIR = "chroma_db"
# 文档存放目录
DOCS_DIR = "docs"
# PDF 源文件目录
PDF_DIR = os.path.join(DOCS_DIR, "pdf")


class ZhipuEmbeddings(Embeddings):
    """基于 requests 库调用智谱 Embedding API，绕过 OpenAI 库在当前环境的挂死问题"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = f"{ZHIPU_API_BASE}/embeddings"

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """单批次调用 Embedding API，网络异常自动重试"""
        for attempt in range(3):
            try:
                resp = requests.post(
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": "embedding-3", "input": texts},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()["data"]
                data.sort(key=lambda x: x["index"])
                return [item["embedding"] for item in data]
            except (requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ConnectionError):
                if attempt == 2:
                    raise
                print(f"    网络重试 ({attempt+1}/3)...")
                time.sleep(2)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 智谱 API 单次最多处理 64 条，分批发送
        batch_size = 64
        results = []
        for i in range(0, len(texts), batch_size):
            results.extend(self._embed_batch(texts[i : i + batch_size]))
        return results

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]


def get_embeddings(api_key: str) -> ZhipuEmbeddings:
    """创建智谱 Embedding 模型实例，用于将文本转为向量"""
    return ZhipuEmbeddings(api_key)


def get_llm(api_key: str) -> ChatOpenAI:
    """创建智谱 GLM 大语言模型实例，用于生成问答回答"""
    return ChatOpenAI(
        model="glm-4-flash",
        openai_api_key=api_key,
        openai_api_base=ZHIPU_API_BASE,
        temperature=0.1,
    )


def _get_doc_files() -> list[str]:
    """获取 docs/ 目录下所有 .txt/.md 文件的绝对路径"""
    if not os.path.exists(DOCS_DIR):
        return []
    files = []
    for root, _, filenames in os.walk(DOCS_DIR):
        for f in filenames:
            if f.endswith((".txt", ".md")):
                files.append(os.path.abspath(os.path.join(root, f)))
    return files


def _load_file(filepath: str) -> list:
    """加载单个文件并返回文档列表"""
    loader = TextLoader(filepath, encoding="utf-8")
    return loader.load()


def _get_indexed_files(vectorstore: Chroma) -> dict[str, float]:
    """从向量库元数据中提取已索引的文件路径及其修改时间"""
    existing = vectorstore.get(include=["metadatas"])
    indexed = {}
    for meta in existing.get("metadatas", []):
        if meta and "source" in meta:
            # source 存的是相对路径，统一转绝对路径比较
            src = os.path.abspath(meta["source"])
            mtime = meta.get("mtime", 0.0)
            # 同一文件可能有多条记录，保留最大的 mtime
            indexed[src] = max(indexed.get(src, 0.0), mtime)
    return indexed


def _split_docs(docs: list) -> list:
    """将文档列表分割为小块"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""],
    )
    return splitter.split_documents(docs)


def _add_mtime(chunks: list) -> None:
    """为 chunk 列表添加 mtime 元数据，用于增量判断"""
    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        abs_source = os.path.abspath(source)
        if os.path.exists(abs_source):
            chunk.metadata["mtime"] = os.path.getmtime(abs_source)


def build_vector_store(docs: list, embeddings: ZhipuEmbeddings) -> Chroma:
    """将文档分割为小块，生成向量并存入 ChromaDB"""
    chunks = _split_docs(docs)
    _add_mtime(chunks)
    print(f"文档已分割为 {len(chunks)} 个片段")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    return vectorstore


def _convert_new_pdfs():
    """扫描 docs/pdf/ 下的 PDF 文件，将未转换的 PDF 通过 MinerU 转为 Markdown 存入 docs/"""
    if not os.path.exists(PDF_DIR):
        return

    pdf_files = [
        os.path.join(root, f)
        for root, _, filenames in os.walk(PDF_DIR)
        for f in filenames
        if f.lower().endswith(".pdf")
    ]
    if not pdf_files:
        return

    converted = 0
    for pdf_path in pdf_files:
        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        md_path = os.path.join(DOCS_DIR, f"{stem}.md")
        if os.path.exists(md_path):
            continue

        print(f"正在转换 PDF: {os.path.basename(pdf_path)}")
        with tempfile.TemporaryDirectory() as tmp_dir:
            # MinerU 对特殊字符文件名不兼容，复制为纯英文临时文件名
            tmp_pdf = os.path.join(tmp_dir, "input.pdf")
            shutil.copy2(pdf_path, tmp_pdf)

            # 查找 MinerU 可执行文件（Windows: Scripts/mineru.exe, Linux: bin/mineru）
            mineru_exe = os.path.join(
                os.path.dirname(sys.executable),
                "Scripts" if os.name == "nt" else "bin",
                "mineru.exe" if os.name == "nt" else "mineru",
            )
            if not os.path.isfile(mineru_exe):
                print("MinerU 未安装，跳过 PDF 自动转换")
                return
            result = subprocess.run(
                [mineru_exe, "-p", tmp_pdf, "-o", tmp_dir, "-b", "pipeline"],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                print(f"  转换失败: {result.stderr[:200]}")
                continue

            # 从 MinerU 输出目录结构中找到 .md 文件
            md_found = None
            for root, _, files in os.walk(tmp_dir):
                for f in files:
                    if f.lower().endswith(".md") and f != "input.md":
                        md_found = os.path.join(root, f)
                        break
                if md_found:
                    break
            # 回退：也检查 input.md
            if not md_found:
                candidate = os.path.join(tmp_dir, "input", "auto", "input.md")
                if os.path.exists(candidate):
                    md_found = candidate

            if not md_found:
                print(f"  未找到转换后的 Markdown 文件")
                continue

            shutil.copy2(md_found, md_path)
            converted += 1
            print(f"  已生成: {md_path}")

    if converted:
        print(f"共转换 {converted} 个 PDF 文件")


def get_or_create_vectorstore(embeddings: ZhipuEmbeddings) -> Chroma | None:
    """加载已有向量库并增量添加新/更新的文档，若无向量库则新建"""
    _convert_new_pdfs()
    docs_files = _get_doc_files()
    if not docs_files:
        if not os.path.exists(DOCS_DIR):
            os.makedirs(DOCS_DIR)
            print(f"已创建文档目录: {DOCS_DIR}/，请将文档放入其中后重新运行。")
        return None

    # 无已有向量库，全量构建
    if not os.path.exists(CHROMA_PERSIST_DIR) or not os.listdir(CHROMA_PERSIST_DIR):
        print("正在构建向量库...")
        docs = []
        for f in docs_files:
            docs.extend(_load_file(f))
        print(f"已加载 {len(docs)} 个文档")
        return build_vector_store(docs, embeddings)

    # 已有向量库，增量更新
    print("正在加载已有向量库...")
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
    )
    indexed = _get_indexed_files(vectorstore)

    # 找出新文件和被修改过的文件
    changed_files = []
    for f in docs_files:
        mtime = os.path.getmtime(f)
        if f not in indexed or mtime > indexed[f]:
            changed_files.append(f)

    if not changed_files:
        print("没有新文档需要更新")
        return vectorstore

    # 加载变更的文档
    new_docs = []
    for f in changed_files:
        new_docs.extend(_load_file(f))

    # 如果文件被修改过，先删除旧数据再添加新的
    for f in changed_files:
        # 用相对路径和绝对路径两种方式匹配删除
        for src_match in [f, os.path.relpath(f)]:
            ids_to_delete = vectorstore.get(
                where={"source": src_match},
                include=[],
            )["ids"]
            if ids_to_delete:
                vectorstore.delete(ids_to_delete)

    # 添加新/更新的文档
    chunks = _split_docs(new_docs)
    _add_mtime(chunks)

    vectorstore.add_documents(chunks)
    print(f"已增量更新 {len(changed_files)} 个文档（{len(chunks)} 个片段）")

    return vectorstore


def _chinese_tokenizer(text: str) -> list[str]:
    """中文分词器，用于 BM25 关键词检索"""
    import jieba
    return [w for w in jieba.cut(text) if w.strip()]


def _build_ensemble_retriever(vectorstore: Chroma) -> EnsembleRetriever:
    """构建混合检索器：向量检索 + BM25 关键词检索，各 50% 权重"""
    stored = vectorstore.get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(stored["documents"], stored["metadatas"])
    ]
    print(f"BM25 索引文档数: {len(docs)}")

    bm25 = BM25Retriever.from_documents(docs, k=3, preprocess_func=_chinese_tokenizer)
    vector = vectorstore.as_retriever(search_kwargs={"k": 3})

    return EnsembleRetriever(retrievers=[vector, bm25], weights=[0.5, 0.5])


def create_qa_chain(api_key: str) -> ConversationalRetrievalChain | None:
    """
    创建多轮对话 RAG 问答链
    流程：结合对话历史改写问题 -> 混合检索（向量+BM25） -> LLM 基于文档生成回答
    """
    embeddings = get_embeddings(api_key)
    vectorstore = get_or_create_vectorstore(embeddings)
    if vectorstore is None:
        return None

    retriever = _build_ensemble_retriever(vectorstore)

    llm = get_llm(api_key)
    prompt = PromptTemplate(
        template=QA_PROMPT,
        input_variables=["context", "question"],
    )
    condense_prompt = PromptTemplate(
        template=CONDENSE_PROMPT,
        input_variables=["chat_history", "question"],
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
        condense_question_prompt=condense_prompt,
    )
    return qa_chain


def ask(qa_chain: ConversationalRetrievalChain, question: str, chat_history: list | None = None) -> dict:
    """执行多轮问答，返回包含 answer（回答）和 sources（来源文档）的字典"""
    if chat_history is None:
        chat_history = []
    # 只保留最近 5 轮对话历史，避免 token 消耗过大
    chat_history = chat_history[-5:]
    result = qa_chain.invoke({"question": question, "chat_history": chat_history})
    return {
        "answer": result["answer"],
        "sources": result["source_documents"],
    }
