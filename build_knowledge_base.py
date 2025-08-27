from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma



# 1. 加载所有文档
loader = DirectoryLoader(
    "knowledge",
    glob="**/*.txt",
    show_progress=True,
    loader_cls=TextLoader,
    loader_kwargs={"autodetect_encoding": True}
)

documents = loader.load()

# 2. 切分文档
# 将长文档切成更小的、有意义的片段，以便更精确地检索
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
chunks = text_splitter.split_documents(documents)

# 3. 初始化嵌入模型
# BAAI/bge-base-en-v1.5 是一个性能很好的开源嵌入模型
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# 4. 创建并填充向量数据库
# 这会在本地创建一个名为 'compliance_db' 的文件夹来存储向量
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./compliance_db"
)

print("知识库构建完成！")