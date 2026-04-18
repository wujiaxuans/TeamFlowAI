"""
RAG 问答系统 - 命令行交互入口
从 .env 文件读取智谱 API Key，初始化 RAG 问答链，提供交互式问答循环。
"""

import os
from dotenv import load_dotenv
from rag import create_qa_chain, ask

# 从项目根目录的 .env 文件加载环境变量
load_dotenv()


def main():
    # 读取智谱 API Key，优先从环境变量获取
    api_key = os.getenv("GLM_API_KEY")
    if not api_key:
        print("错误: 请设置环境变量 GLM_API_KEY")
        print("  方法1: set GLM_API_KEY=your_key")
        print("  方法2: 在项目根目录创建 .env 文件，写入 GLM_API_KEY=your_key")
        return

    # 初始化 RAG 系统：加载文档 -> 构建向量库 -> 创建问答链
    print("正在初始化 RAG 系统...")
    qa_chain = create_qa_chain(api_key)
    if qa_chain is None:
        return

    print("\n" + "=" * 50)
    print("RAG 问答系统已就绪")
    print("输入问题开始提问，输入 quit 退出")
    print("=" * 50 + "\n")

    # 交互式问答循环
    chat_history = []
    while True:
        question = input("问题: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if question.lower() == "clear":
            chat_history = []
            print("对话已清空\n")
            continue

        # 调用 RAG 链获取回答和来源文档
        result = ask(qa_chain, question, chat_history)
        chat_history.append((question, result["answer"]))

        print(f"\n回答: {result['answer']}")
        # 展示来源文档（去重，同一文件只显示一次）
        if result["sources"]:
            print("\n来源:")
            seen = set()
            for doc in result["sources"]:
                source = doc.metadata.get("source", "未知")
                if source not in seen:
                    seen.add(source)
                    print(f"  - {source}")
        print()


if __name__ == "__main__":
    main()
