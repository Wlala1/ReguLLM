# 法规知识库向量数据库

这个项目帮助您将法规文档转换为本地向量数据库，支持语义搜索和知识检索。

## 功能特性

- 🔍 **智能文档处理**: 自动加载和分割法规文档
- 🧠 **向量化存储**: 使用Google Generative AI进行文档嵌入
- 📊 **本地数据库**: 使用ChromaDB存储向量数据
- 🔎 **语义搜索**: 支持自然语言查询法规内容
- 📝 **多编码支持**: 自动检测UTF-8、GBK、GB2312编码

## 安装依赖

```bash
pip install -r requirements.txt
```

## 准备工作

### 1. 获取Google AI API密钥

1. 访问 [Google AI Studio](https://makersuite.google.com/app/apikey)
2. 创建新的API密钥
3. 将API密钥设置为环境变量：

```bash
export GOOGLE_API_KEY="your_api_key_here"
```

或者直接在 `trans_embedding.py` 文件中修改 `GOOGLE_API_KEY` 变量。

### 2. 准备法规文档

将您的5个法规txt文件放入 `knowledge` 目录：

```
knowledge/
├── 法规1.txt
├── 法规2.txt
├── 法规3.txt
├── 法规4.txt
└── 法规5.txt
```

## 使用方法

### 方法1: 运行主脚本

```bash
python trans_embedding.py
```

### 方法2: 运行测试脚本

```bash
python test_knowledge_base.py
```

选择运行模式：
- **自动测试**: 构建知识库并进行预设查询测试
- **交互式搜索**: 手动输入查询进行实时搜索

### 方法3: 在代码中使用

```python
from trans_embedding import LegalDocumentVectorStore

# 初始化
builder = LegalDocumentVectorStore(
    google_api_key="your_api_key",
    knowledge_dir="./knowledge",
    vector_db_dir="./vector_db"
)

# 构建知识库
vectorstore = builder.build_knowledge_base()

# 搜索相关文档
results = builder.search_similar_documents(
    vectorstore, 
    "合同违约责任", 
    k=5
)

for doc in results:
    print(f"来源: {doc.metadata['source']}")
    print(f"内容: {doc.page_content[:200]}...")
```

## 配置参数

可以在初始化时调整以下参数：

- `chunk_size`: 文档分块大小 (默认: 1000)
- `chunk_overlap`: 分块重叠大小 (默认: 200)
- `knowledge_dir`: 法规文件目录 (默认: "./knowledge")
- `vector_db_dir`: 向量数据库目录 (默认: "./vector_db")

## 文件结构

```
.
├── trans_embedding.py      # 主要的向量化脚本
├── test_knowledge_base.py  # 测试和交互脚本
├── config.py              # 配置文件
├── requirements.txt       # 依赖包列表
├── README.md             # 说明文档
├── knowledge/            # 存放法规txt文件
└── vector_db/           # 向量数据库存储目录（自动创建）
```

## 搜索示例

构建完成后，您可以进行以下类型的查询：

- "合同违约的法律后果"
- "行政处罚的程序规定"
- "民事诉讼的管辖范围"
- "刑事责任的认定标准"
- "损害赔偿的计算方法"

## 注意事项

1. **API密钥安全**: 请不要将API密钥提交到版本控制系统
2. **文件编码**: 支持UTF-8、GBK、GB2312编码的txt文件
3. **网络连接**: 首次构建需要网络连接以获取嵌入向量
4. **存储空间**: 向量数据库会占用一定磁盘空间
5. **更新数据**: 如需添加新文档，删除vector_db目录重新构建

## 故障排除

### 常见问题

1. **API密钥错误**
   - 检查密钥是否正确设置
   - 确认API密钥有效且有足够配额

2. **文件编码问题**
   - 确保txt文件使用UTF-8、GBK或GB2312编码
   - 检查文件内容是否完整

3. **依赖安装失败**
   - 使用Python 3.8+版本
   - 考虑使用虚拟环境

4. **向量数据库访问错误**
   - 确保有写入权限
   - 检查磁盘空间是否充足

## 许可证

本项目仅供学习和研究使用。