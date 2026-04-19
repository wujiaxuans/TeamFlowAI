"""
内容生成 Agent
基于 LangChain ReAct 模式，自主调用 RAG 检索 + 文档模板生成结构化内容。
支持 PRD、会议纪要、周报三种文档类型。
"""

import os
from langchain_classic.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from rag import create_qa_chain, ask, get_llm
from prompts import AGENT_PROMPT, FILL_PROMPT, REFINE_PROMPT, TEMPLATES, TYPE_KEYWORDS


def _match_template_type(text: str) -> str:
    """根据输入文本模糊匹配文档类型"""
    text_lower = text.lower()
    for doc_type, keywords in TYPE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return doc_type
    return "PRD"


def create_content_agent(api_key: str) -> AgentExecutor:
    """创建内容生成 Agent，返回 AgentExecutor 实例"""
    qa_chain = create_qa_chain(api_key)
    if qa_chain is None:
        raise ValueError("知识库为空，无法创建 Agent")

    llm = get_llm(api_key)
    last_rag_context = {"text": "", "doc": ""}

    def rag_search(query: str) -> str:
        """搜索知识库，返回与问题相关的文档内容"""
        result = ask(qa_chain, query)
        parts = [f"回答: {result['answer']}"]
        for doc in result["sources"]:
            source = os.path.basename(doc.metadata.get("source", "未知"))
            parts.append(f"\n来源 [{source}]:\n{doc.page_content}")
        context = "\n".join(parts)
        last_rag_context["text"] = context
        return context

    def generate_content(doc_type: str) -> str:
        """根据文档类型和已搜索的背景资料，生成填充好的文档"""
        matched = _match_template_type(doc_type)
        template = TEMPLATES[matched]
        context = last_rag_context["text"]

        if not context:
            last_rag_context["doc"] = template
            return template

        fill_prompt = FILL_PROMPT.format(context=context, template=template)
        result = llm.invoke(fill_prompt).content
        last_rag_context["doc"] = result
        return result

    def refine_content(instruction: str) -> str:
        """根据指令润色当前文档"""
        doc = last_rag_context["doc"]
        if not doc:
            return "当前没有文档，请先调用 generate_content"

        refine_prompt = REFINE_PROMPT.format(doc=doc, instruction=instruction)
        result = llm.invoke(refine_prompt).content
        last_rag_context["doc"] = result
        return result

    tools = [
        Tool(
            name="rag_search",
            func=rag_search,
            description="搜索知识库文档，输入问题或关键词，返回相关的文档内容。",
        ),
        Tool(
            name="generate_content",
            func=generate_content,
            description="生成结构化文档。输入文档类型（PRD/会议纪要/周报），返回填充好的文档。",
        ),
        Tool(
            name="refine_content",
            func=refine_content,
            description="润色当前文档。输入润色指令（如'补充性能指标'、'优化表达'），返回改进后的文档。",
        ),
    ]

    prompt = PromptTemplate(
        template=AGENT_PROMPT,
        input_variables=["input", "agent_scratchpad"],
        partial_variables={
            "tools": "\n".join(f"- {t.name}: {t.description}" for t in tools),
            "tool_names": ", ".join(t.name for t in tools),
        },
    )

    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6,
    )


def run_agent(executor: AgentExecutor, user_input: str) -> str:
    """运行 Agent，返回生成的文档内容"""
    result = executor.invoke({"input": user_input})
    return result["output"]


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GLM_API_KEY")
    if not api_key:
        print("请先在 .env 中配置 GLM_API_KEY")
        exit(1)

    executor = create_content_agent(api_key)
    print("内容生成 Agent 已启动（输入 quit 退出）\n")

    while True:
        user_input = input("你: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue
        output = run_agent(executor, user_input)
        print(f"\nAgent:\n{output}\n")