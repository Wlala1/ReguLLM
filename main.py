from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi

# 初始化 LLM 和 嵌入模型
llm = ChatTongyi(model="qwen-turbo", temperature=0.1)

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# 加载已经存在的向量数据库
vector_store = Chroma(persist_directory="./compliance_db", embedding_function=embedding_model)

# 创建一个检索器 (Retriever)，它可以根据查询找到相关文档
retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # k=3 表示返回最相关的3个文档片段

# 示例：接收到一个功能描述
feature_description = "Geofences feature rollout in US for market testing"

# 使用检索器找到相关信息
retrieved_docs = retriever.invoke(feature_description)
print(retrieved_docs) # 你会看到返回了包含犹他州法案内容的文档片段

# 设计一个 Prompt 模板
template = """
You are an expert legal compliance analyst. Your task is to determine if a feature needs geo-specific compliance logic based on the provided context and feature description.
You must distinguish between legal requirements and business decisions. Output your answer in a valid JSON format.

CONTEXT FROM KNOWLEDGE BASE:
{context}

FEATURE DESCRIPTION:
{feature_description}

Based on the context and the description, provide your analysis as a JSON object with the keys 'needs_compliance_logic' (boolean), 'reasoning' (string), and 'identified_regulation' (string, or null).
"""

prompt = PromptTemplate.from_template(template)

# 使用 LangChain 的 LCEL 链式语法将所有步骤串起来
rag_chain = (
    {"context": retriever, "feature_description": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 运行 RAG 链
result_json = rag_chain.invoke(feature_description)

print(result_json)