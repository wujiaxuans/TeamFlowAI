# TeamFlowAI RAG 问答系统

基于 LangChain + ChromaDB + 智谱 GLM API 的检索增强生成（RAG）问答系统，支持多轮对话和 PDF 文档自动解析。

## 功能

- 多轮对话 RAG 问答，自动将追问改写为独立问题再检索
- 支持 .txt / .md / PDF 文档，PDF 自动通过 MinerU 转为 Markdown
- 向量库增量更新，仅处理新增或修改的文档
- Streamlit Web 界面：文件上传、对话式问答、引用来源展示
- 命令行交互模式

## 快速开始

### 1. 创建环境并安装依赖

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

将 .txt / .md 文件放入 `docs/`，PDF 文件放入 `docs/pdf/`。

### 4. 启动

**Web 界面：**
```bash
streamlit run web.py
```

**命令行：**
```bash
python app.py
```

## 项目结构

```
├── app.py              # 命令行交互入口
├── web.py              # Streamlit Web 界面
├── rag.py              # RAG 核心逻辑（文档加载、向量化、检索问答）
├── requirements.txt    # 依赖
├── .env.example        # 环境变量示例
├── docs/               # 文档目录（.txt / .md）
│   └── pdf/            # PDF 源文件（自动转为 Markdown）
└── chroma_db/          # ChromaDB 向量库（自动生成）
```

## 技术栈

- **LangChain** — 编排框架（ConversationalRetrievalChain）
- **ChromaDB** — 向量数据库
- **智谱 GLM** — Embedding (embedding-3) + LLM (glm-4-flash)
- **MinerU** — PDF 转 Markdown
- **Streamlit** — Web 界面

## Chunk Size 评测结果

使用 10 个测试问题，对比三种 chunk_size 配置（overlap 均为 20%）的 Top3 召回率：

| # | 问题 | 预期文档 | cs=300 | cs=500 | cs=1000 |
|---|------|----------|--------|--------|---------|
| 1 | RAG的完整流程分哪几个步骤？ | RAG学习测试QA.md | ✓ | ✓ | ✓ |
| 2 | RAG检索效果差可以从哪些方向优化？ | RAG学习测试QA.md | ✓ | ✓ | ✓ |
| 3 | HyDE是什么，解决了什么问题？ | RAG学习测试QA.md | ✗ | ✗ | ✗ |
| 4 | Function Calling和MCP协议有什么区别？ | Function Calling与MCP.md | ✓ | ✓ | ✓ |
| 5 | MCP协议的核心设计思路是什么？ | Function Calling与MCP.md | ✓ | ✓ | ✓ |
| 6 | Workflow和Agent的本质区别是什么？ | effective-agent-patterns.md | ✓ | ✓ | ✓ |
| 7 | Orchestrator和Subagent分别是什么角色？ | effective-agent-patterns.md | ✗ | ✗ | ✗ |
| 8 | 什么情况下应该用Workflow而不是Agent？ | effective-agent-patterns.md | ✓ | ✓ | ✓ |
| 9 | Claude Code有哪些核心功能？ | Claude Code入门.md | ✓ | ✓ | ✓ |
| 10 | 如何用Claude Code进行Vibe Coding？ | Claude Code入门.md | ✓ | ✓ | ✓ |
| | **召回率** | | **80%** | **80%** | **80%** |

**结论：** 三种配置召回率一致。未命中的 HyDE、Orchestrator/Subagent 是语义匹配问题，与分块粒度无关。当前使用 cs=500 + overlap=100 作为默认配置。
