# RAG 学习测试 Q&A 整理

---

### Q1：不用 RAG 直接塞文档进 Prompt 有什么问题？RAG 解决了什么？

**问题：**
1. 上下文 Token 限制，文档太多根本塞不进去
2. 成本高、效率低
3. 检索不准确，回答偏离问题
4. LLM 会产生幻觉

**RAG 解决了：**
1. 大模型对特定知识的局限性
2. 知识时效性问题（LLM 训练有截止日期，新文档它不知道）
3. 对内部资料的准确查找
4. 减少幻觉（基于检索到的真实内容回答）

---

### Q2：RAG 两个阶段分别做什么？

**离线建库：**
1. 文档加载
2. **分块（Chunking）**— 切成 500-1000 字小片段，粒度更细更精准
3. 将片段传入 Embedding 模型生成语义向量
4. 将语义向量存入向量数据库

**在线检索：**
1. 用户提问
2. 将问题传入 Embedding 模型生成问题的语义向量
3. 从向量数据库中根据语义向量进行相似度查找
4. 将检索到的相关片段拼接成 Prompt
5. 将 Prompt 传入 LLM
6. LLM 生成回答返回给用户

---

### Q3：RAG 检索效果差如何优化？

| 优化方向 | 专业术语 | 说明 |
|---|---|---|
| 分块策略优化 | Chunking Strategy | 根据内容类型和 Embedding 模型调整 chunk_size |
| 假设性问题与向量一起存入 | HyDE（假设文档嵌入） | 进阶索引技巧，提升检索召回率 |
| 分层索引存入数据库 | Hierarchical Index | 粗粒度+细粒度双层索引 |
| 融合检索（稀疏+稠密） | Hybrid Search（BM25 + Vector） | 关键词检索+语义检索结合 |
| 语句窗口/自动合并检索器 | Sentence Window / Auto-merging Retriever | LlamaIndex 高级检索策略 |
| 检索结果重排过滤 | Rerank | 召回多个片段后用 Rerank 模型选最优 |
| 多路检索结果融合 | RAG Fusion | 生成多个检索结果让 LLM 融合输出 |
| 问题改写优化 | Query Rewriting | 先让 LLM 改写口语化问题再检索，提升召回率 |
| 查询路由 | Query Routing | 让 LLM 根据问题决定走哪条检索路径 |
| Agent 化 RAG | Agentic RAG | 通过 Agent 动态决策检索策略 |
