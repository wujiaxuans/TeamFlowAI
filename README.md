# TeamFlowAI

基于 LangChain + ChromaDB + 智谱 GLM 的 **RAG 知识库问答** 与 **AI 文档生成** 系统。

---

## 核心功能

### 模块一：RAG 知识库问答

- 多轮对话问答，自动将追问改写为独立问题再检索
- 混合检索（向量 + BM25 关键词），各占 50% 权重
- 支持 `.txt` / `.md` / PDF 文档，PDF 自动通过 MinerU 转 Markdown
- 向量库增量更新，仅处理新增或修改的文档
- Streamlit Web 界面：文件上传、对话式问答、引用来源展示

### 模块二：AI 文档生成 Agent

- ReAct 模式 Agent，自主调用 RAG 检索 + 模板填充
- 支持生成 PRD、会议纪要、周报三种结构化文档
- 自动搜索背景资料 → 填充模板 → 润色优化
- 输出 Markdown 格式，可下载 `.md` 文件

---

## 技术栈

| 模块 | 技术 |
|------|------|
| LLM | 智谱 GLM-4-flash（问答）+ GLM-4-flash（Agent） |
| Embedding | 智谱 embedding-3 |
| 向量库 | ChromaDB |
| 检索 | EnsembleRetriever（向量 + BM25） |
| 分词 | jieba（中文 BM25） |
| Agent | LangChain ReAct + Tool Calling |
| PDF 解析 | MinerU |
| Web 界面 | Streamlit |

---

## 架构流程

### RAG 问答流程

```
用户提问
    ↓
问题改写（结合对话历史生成独立问题）
    ↓
混合检索（向量检索 + BM25 关键词检索）
    ↓
LLM 基于检索结果生成回答
    ↓
返回答案 + 引用来源
```

### 文档生成 Agent 流程

```
用户输入需求描述
    ↓
Agent 决策 → 调用 rag_search 搜索背景资料
    ↓
Agent 决策 → 调用 generate_content 填充模板
    ↓
Agent 决策 → 调用 refine_content 润色优化
    ↓
输出结构化 Markdown 文档
```

---

## 本地运行

### 1. 创建环境

```bash
conda create -n teamflowai python=3.12 -y
conda activate teamflowai
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入智谱 API Key
```

### 3. 准备文档

将 `.txt` / `.md` 文件放入 `docs/`，PDF 文件放入 `docs/pdf/`。

### 4. 启动 Web 界面

```bash
streamlit run web.py
```

### 5. 命令行模式

```bash
# RAG 问答
python app.py

# 文档生成 Agent
python agent.py
```

---

## 在线 Demo

🔗 [TeamFlowAI Streamlit Cloud](https://teamflowai-ijuhzrwdt6dbvyanqcuxpy.streamlit.app/)

---

## 项目结构

```
├── web.py              # Streamlit Web 界面（双模块）
├── app.py              # RAG 问答 CLI 入口
├── agent.py            # 文档生成 Agent
├── rag.py              # RAG 核心逻辑
├── prompts.py          # 提示词与文档模板
├── requirements.txt    # 依赖
├── .env.example        # 环境变量示例
├── docs/               # 文档目录
│   └── pdf/            # PDF 源文件
└── chroma_db/          # ChromaDB 向量库
```

---

## Chunk Size 评测

使用 10 个测试问题对比三种 chunk_size 配置（overlap=20%）：

| chunk_size | Top3 召回率 |
|------------|-------------|
| 300 | 80% |
| 500 | 80% |
| 1000 | 80% |

**结论**：召回率一致，未命中的问题是语义匹配问题，与分块粒度无关。默认使用 `chunk_size=500` + `overlap=100`。