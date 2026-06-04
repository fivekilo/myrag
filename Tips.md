# myrag 项目说明

## 1. 项目的工作流水线

这个项目是一个分步骤操作的 RAG 系统。用户不是一次性上传文档后直接得到答案，而是沿着“导入 -> 分块/解析 -> 嵌入 -> 建索引 -> 检索 -> 生成”的流水线逐步处理。

### 1.1 文档导入

用户在前端的“文档导入”页面上传 PDF 文件，并选择读取方式。后端会先把文件临时保存到 temp 目录，然后调用 LoadingService 读取 PDF 文本。当前支持的读取方式包括：

- PyMuPDF
- PyPDF
- pdfplumber
- unstructured

导入完成后，系统会把文档按页转成标准化 chunks，并附带页码、字数、页范围、处理时间等元数据，保存到 01-loaded-docs 目录。

这一阶段的目的，是把原始 PDF 转成后续流程可复用的结构化文本。

### 1.2 文件解析

用户也可以在“文件解析”页面单独对 PDF 做结构解析。这个步骤不直接进入向量检索，而是用于观察文档结构。当前支持的解析方式包括：

- all_text：全文连续提取
- by_pages：按页解析
- by_titles：按标题划分章节
- text_and_tables：区分文本和疑似表格内容

解析结果用于展示文档结构，帮助用户理解文档内容组织形式。

### 1.3 知识分块

在“知识分块”页面，用户从已经导入的文档中选择一个文件，再选择分块策略。当前支持：

- by_pages：按页分块
- fixed_size：按固定长度分块
- by_paragraphs：按段落分块
- by_sentences：按句子分块

分块后的结果会保存到 01-chunked-docs。这个步骤的核心作用，是把文档拆成适合做向量表示和检索的知识单元。

### 1.4 文本嵌入

在“向量存储”页面，用户选择一个已导入或已分块的文档，再选择嵌入模型。系统会对每个 chunk 生成 embedding 向量，并把结果保存到 02-embedded-docs。

每个向量都会携带对应的元数据，例如：

- chunk_id
- page_number
- page_range
- content
- embedding_provider
- embedding_model
- vector_dimension

这个阶段的输出是“带有向量表示的知识块”。

### 1.5 向量库索引

在“向量库索引”页面，用户选择一个 embedding 文件，再选择向量库和索引模式。当前代码中保留了 Milvus 和 Chroma 的接口，但当前主路径实际主要使用的是 Chroma。

索引完成后，向量会被写入 03-vector-store，对应形成可检索的 collection。

### 1.6 相似性检索

在“相似性检索”页面，用户输入查询问题，选择向量库中的 collection，并设定检索参数：

- top_k：返回前几个结果
- threshold：相似度阈值
- word_count_threshold：最小文本长度阈值

系统会先用 collection 对应的 embedding 配置把用户问题转成向量，再到向量库中做相似度检索。检索结果会返回文本、得分、来源页码、chunk 编号等信息，并可保存到 04-search-results。

### 1.7 响应生成

在“响应生成”页面，用户可以直接提问，也可以加载已有的检索结果文件。系统会把检索结果拼接成上下文，然后调用生成模型回答问题。

生成后的回答会保存到 05-generation-results。也就是说，最终答案不是直接基于原始 PDF 生成，而是基于“检索出的上下文 + 用户问题”生成，这就是典型的 RAG 工作方式。

### 1.8 整体流程总结

可以把整个项目理解成下面这条链路：

1. 用户上传 PDF
2. 系统读取 PDF 文本并保存为已导入文档
3. 用户选择分块策略，系统把文档拆成知识块
4. 用户选择嵌入模型，系统把知识块转换成向量
5. 用户把向量写入向量库，形成可检索集合
6. 用户输入问题，系统将问题向量化并检索相关知识块
7. 系统把检索结果作为上下文交给生成模型
8. 生成模型输出答案并保存结果

如果只看最终结果，这是一个“知识库问答”系统；如果看过程，它是一个把 RAG 每一步显式展示出来的实验平台。

## 2. 项目的嵌入模型介绍

项目中的嵌入模型用于把文本 chunk 转换为向量，以便后续进行相似度检索。

### 2.1 当前可选的嵌入模型

项目当前支持 3 类嵌入 provider：

#### 1. OpenAI

- text-embedding-3-large
- text-embedding-3-small

#### 2. Bedrock

- cohere.embed-english-v3
- cohere.embed-multilingual-v3

#### 3. HuggingFace

- BAAI/bge-small-zh-v1.5
- sentence-transformers/all-mpnet-base-v2
- sentence-transformers/all-MiniLM-L6-v2
- google-bert/bert-base-uncased

其中，OpenAI 和 Bedrock 属于远程 API 模型，HuggingFace 既可以走远程仓库，也可以走本地模型目录。

### 2.2 嵌入模型 API 调用方式

#### OpenAI

OpenAI 嵌入调用通过 OPENAI_API_KEY 配置密钥。项目后端会创建 OpenAIEmbeddings，然后调用 embed_documents 或 embed_query 生成向量。

这类模型的优点是使用简单、效果稳定，本地几乎不消耗算力；缺点是需要联网，并且有调用成本。

#### Bedrock

Bedrock 通过 AWS_ACCESS_KEY_ID、AWS_SECRET_ACCESS_KEY 和区域配置调用。项目里当前默认区域是 ap-southeast-1。

这类模型同样主要消耗云端资源，本地资源压力很小。

#### HuggingFace

如果配置了 HF_MODEL_PATH，系统会优先从本地目录寻找模型；如果本地不存在，就会把模型名当作远程仓库名处理。

这意味着 HuggingFace 嵌入模型既可以本地部署，也可以联网拉取。

### 2.3 本地部署嵌入模型的硬件建议

嵌入模型通常比生成模型轻得多，对硬件的要求相对低。

#### 轻量级模型

- sentence-transformers/all-MiniLM-L6-v2
- BAAI/bge-small-zh-v1.5

推荐资源：

- 最低可用：4 核到 8 核 CPU，8GB 内存
- 更顺手：16GB 内存
- 如果有独显：4GB 到 6GB 显存即可明显加速

这类模型适合课程实验、小规模文档处理和中文基础检索。

#### 中等体量模型

- google-bert/bert-base-uncased
- sentence-transformers/all-mpnet-base-v2

推荐资源：

- CPU 运行：建议 16GB 内存
- GPU 运行：建议 6GB 到 8GB 显存

如果文档数量较多、分块较细，虽然模型本身不算特别大，但整体嵌入时间会明显增长。

### 2.4 嵌入模型的推荐选择

如果是中文文档，优先建议：

- BAAI/bge-small-zh-v1.5

如果更看重轻量和易跑通，优先建议：

- sentence-transformers/all-MiniLM-L6-v2

如果希望语义效果更强一些，且硬件更充足，可以考虑：

- sentence-transformers/all-mpnet-base-v2

整体上，嵌入模型不需要特别高端的显卡，一台 16GB 内存电脑就可以完成大多数小规模实验；如果有 6GB 到 8GB 显存的独显，体验会更好。

对于这个项目当前已经验证过的一条本地路线，可以直接使用：

- BAAI/bge-small-zh-v1.5
- CUDA
- RTX 3050 Laptop GPU 4GB

在这条配置下，项目后端已经实际跑通了一次本地嵌入，输出向量维度为 512。也就是说，对于小规模中文课程实验，这一级别的显卡可以承担嵌入任务；它更适合做 embedding，不适合硬跑 7B 级生成模型。

## 3. 项目的生成模型介绍

项目中的生成模型用于根据“用户问题 + 检索到的上下文”生成最终答案。

### 3.1 当前可选的生成模型

项目当前支持 3 类生成 provider。

#### 1. HuggingFace 本地/远程模型

- Llama-2-7b-chat
- DeepSeek-7b
- DeepSeek-R1-Distill-Qwen-1.5B
- Qwen-Qwen3-1.7B

#### 2. 阿里云百炼

- qwen-turbo
- qwen3.6-plus

#### 3. DeepSeek API

- deepseek-v4-flash
- deepseek-v4-pro
- deepseek-chat
- deepseek-reasoner

其中，HuggingFace 更适合本地部署；阿里云百炼和 DeepSeek 更适合在线 API 调用。

### 3.2 生成模型 API 调用方式

#### HuggingFace

HuggingFace 生成模型通过 transformers 加载，项目当前使用 AutoModelForCausalLM 和 AutoTokenizer，并以 float16 方式直接装载模型。加载后，系统会把检索结果拼成 context，再把 prompt 送给模型生成回答。

如果配置了 HF_MODEL_PATH，就优先走本地目录；否则会从 HuggingFace 仓库名加载。

#### 阿里云百炼

阿里云百炼通过 OpenAI 兼容接口调用，使用的密钥是 DASHSCOPE_API_KEY，base_url 为阿里云兼容接口地址。项目里还开启了思考模式输出。

这类方式适合本地显卡不足，但又想快速得到较强生成能力的场景。

#### DeepSeek API

DeepSeek 通过 DEEPSEEK_API_KEY 调用，base_url 为 https://api.deepseek.com。

当前更推荐优先选择 deepseek-v4-flash 或 deepseek-v4-pro。前者更适合低成本快速调用，后者更适合质量优先的回答生成。

deepseek-chat 和 deepseek-reasoner 在项目里也仍然保留，主要是为了兼容旧别名和旧配置；如果是新配置，优先按 v4 系列来选。

### 3.3 本地部署生成模型的硬件建议

生成模型比嵌入模型更吃显存。当前项目代码中没有看到 4bit、8bit 量化方案，而是按 float16 直接加载，因此对显卡要求相对更高。

#### 轻量级本地模型

- DeepSeek-R1-Distill-Qwen-1.5B
- Qwen-Qwen3-1.7B

推荐资源：

- 最低可用：6GB 左右显存
- 更稳妥：8GB 显存
- CPU 也能跑，但速度会明显偏慢，建议至少 16GB 内存

这两个模型是当前项目里最适合本地实验和课程演示的选择。

#### 7B 级模型

- Llama-2-7b-chat
- DeepSeek-7b

推荐资源：

- 建议至少 16GB 显存
- 更理想：16GB 到 24GB 显存
- 如果只靠 CPU，虽然理论上能尝试，但生成速度通常不适合日常交互

由于项目当前直接用 float16 加载 7B 模型，所以 8GB 显卡一般不适合直接跑这两个模型。

### 3.4 本地部署推荐模型

如果目标是“先跑通项目”，推荐优先使用：

- DeepSeek-R1-Distill-Qwen-1.5B
- Qwen-Qwen3-1.7B

推荐理由：

- 模型体量小
- 更容易在普通消费级显卡上运行
- 足够支撑课程演示、实验报告和基础问答

如果目标是“提升回答质量”，且你有更强的显卡资源，再考虑：

- Llama-2-7b-chat
- DeepSeek-7b

### 3.5 生成模型的整体建议

可以把生成模型的使用场景简单分成三类：

- 没有高性能显卡：优先用阿里云百炼或 DeepSeek API
- 有普通独显，想本地跑通：优先用 1.5B 到 1.7B 模型
- 有 16GB 以上显存，想要更强本地生成能力：可以尝试 7B 模型

从当前项目配置来看，最推荐的本地部署路线不是 7B，而是 1.5B 或 1.7B。这样更容易成功，也更符合课程项目和实验环境的实际条件。

## 4. 环境配置与启动步骤

这一部分结合 README 的原始建议，以及本次在 Windows 环境下实际启动成功的过程整理而成。目标不是给出一份最大而全的依赖清单，而是给出一条更容易复现的启动路径。

### 4.1 当前建议的环境基线

- 操作系统：Windows
- Python：3.11.9
- Conda 环境名：langchain
- 前端：Node.js + npm
- 后端端口：8001
- 前端端口：5173

项目 README 中原本引用了上游仓库的 requirements_win.txt，但当前工作区并没有把这份文件带进来。因此，本仓库根目录现在补充了一份 requirements.txt，用来覆盖“当前项目在 Windows 下启动”的最小可复现依赖集合。

### 4.2 Python 环境创建步骤

如果你使用 Conda，推荐按下面的顺序执行：

```powershell
conda create -n langchain python=3.11.9
conda activate langchain
```

如果你的 Conda 激活在终端里不稳定，也可以直接使用环境内的解释器和 pip 路径，例如：

```powershell
D:\anaconda\envs\langchain\python.exe
D:\anaconda\envs\langchain\Scripts\pip.exe
```

### 4.3 安装后端依赖

在项目根目录执行：

```powershell
pip install -r requirements.txt
```

这份 requirements.txt 是按“当前仓库代码 + Windows 本地启动验证”整理出来的。它和 README 指向的上游 requirements_win.txt 不完全相同，主要差异是：

- 保留了旧代码所依赖的 LangChain 版本组合
- 明确补上了 pymilvus 运行所需的 environs 和 marshmallow 兼容版本
- 把当前仓库真实用到的 PDF 解析、向量库、API 客户端相关依赖单独列出

### 4.4 安装前端依赖

前端依赖不走 requirements.txt，而是继续由 frontend 目录下的 package.json 管理。执行方式如下：

```powershell
cd frontend
npm install
```

### 4.5 API Key 与本地模型配置

项目可以先启动，再决定是否配置 API Key。

也就是说：

- 如果只是想先把前后端跑起来，不需要先配置 API Key
- 如果后续要调用云端嵌入或云端生成模型，就需要补充对应环境变量
- 如果走本地 HuggingFace 模型路线，则重点是本地模型目录和硬件资源，不一定需要云端密钥

当前代码中实际会读取的环境变量主要包括：

- OPENAI_API_KEY：OpenAI 嵌入
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY：Bedrock 嵌入
- DASHSCOPE_API_KEY：阿里云百炼生成
- DEEPSEEK_API_KEY：DeepSeek 生成
- HF_MODEL_PATH：本地 HuggingFace 模型目录
- EMBEDDING_DEVICE：本地嵌入使用 cpu 或 cuda
- HF_ENDPOINT：HuggingFace 镜像地址

这里最重要的一点是：当前项目已经改成优先从 backend/.env 读取这些配置，不再推荐靠终端临时 set 环境变量。

也就是说，如果你要配置 DeepSeek API Key，直接编辑 backend/.env 即可：

```powershell
DEEPSEEK_API_KEY=你的Key
```

如果你要配置本地嵌入模型，当前已经验证通过的一组配置是：

```powershell
HF_MODEL_PATH=F:/本科/学习/数据挖掘与信息检索/myrag/backend/models
EMBEDDING_DEVICE=cuda
HF_ENDPOINT=https://hf-mirror.com
```

其中，模型 BAAI/bge-small-zh-v1.5 会被解析到下面这个本地目录：

```powershell
backend/models/BAAI/bge-small-zh-v1.5
```

如果该目录下模型文件已经存在，后端就会优先直接从本地加载，而不是再次去线上下载。

### 4.6 启动方式

后端启动：

```powershell
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

如果你是在某些终端工具里启动，工作目录可能没有正确落到 backend，导致出现 Could not import module "main"。这种情况下可以直接使用显式 app-dir：

```powershell
uvicorn main:app --app-dir backend --reload --host 0.0.0.0 --port 8001
```

前端启动：

```powershell
cd frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

本项目当前前端开发环境默认请求地址是：

- http://localhost:8001

因此在本地开发时，前后端启动后一般不需要再改 frontend/src/config/config.js。

### 4.7 Windows 下的额外说明

- 代码里保留了 Milvus 接口，但 Windows 本地开发更适合优先走 Chroma
- 如果只是做课程实验、演示和小规模检索，Chroma 已经足够
- 使用 uvicorn 的 reload 模式时，尽量从 backend 目录启动，避免监视整个仓库导致无关文件变更触发重载

### 4.8 本次实际验证结果

本次已经在 Windows + Conda langchain 环境下完成了以下验证：

1. 后端可以成功 import main
2. 后端 docs 页面可以正常访问
3. 前端开发服务器可以正常启动
4. 前后端地址都返回 200
5. 已完成“导入 -> 分块 -> 嵌入 -> 索引 -> 检索 -> 生成”的一整条流程验证
6. 已验证 Chroma 路线可用，Milvus 接口仍保留但不是当前本地主路径

因此，当前仓库根目录的 requirements.txt 可以视为一份“面向本仓库当前状态的 Windows 启动依赖总结”。

### 4.9 当前项目状态

截至目前，这个项目已经可以按当前仓库代码完整跑通。

本次相较于原始代码的主要调整，集中在以下几类：

- 模型调用链路修正：补齐 backend/.env 和 backend/.env.example 的读取逻辑，优先从 backend/.env 读取 API Key、本地模型路径和设备配置
- 生成模型兼容修正：DeepSeek 路线改为 v4-first，前端可见 deepseek-v4-flash 和 deepseek-v4-pro，同时兼容旧别名
- 数据路径统一：新增基于 backend 目录的固定运行路径，避免因为启动 cwd 不同，把数据写到根目录和 backend 两处
- 向量索引与检索修正：修复 Chroma collection 命名长度限制、metadata 类型不匹配、检索阶段误用默认 384 维 embedding 导致的维度冲突
- 前端显示修正：修复索引页、检索页、生成页白底白字和选中文本可读性问题

### 4.10 当前推荐配置

如果目标是“先稳定复现当前项目状态”，建议直接沿用这一组已经验证通过的组合：

- 嵌入模型：本地部署 BAAI/bge-small-zh-v1.5
- 嵌入设备：CUDA
- 生成模型：DeepSeek API
- 向量库：Chroma

其中，本地 BAAI/bge-small-zh-v1.5 已经在 RTX 3050 Laptop GPU 4GB 上实际跑通过一次嵌入，输出维度为 512；生成部分则已通过 DeepSeek API 跑通。

### 4.11 当前仓库交接说明

目前仓库里已经补齐两类配置文件：

- requirements.txt：面向当前仓库代码和 Windows 本地启动验证整理出的后端依赖集合
- backend/.env.example：供其他开发者复制为 backend/.env 的本地配置模板

日常协作时，建议把真实密钥保留在本地 backend/.env，不要直接提交到远程仓库；需要共享配置结构时，以 backend/.env.example 为准。

当前已经使用一个 PDF 样本完成过端到端测试，因此对开发组来说，这个仓库已经不再是“只能读代码、很难启动”的状态，而是一份可以直接继续迭代功能和界面的可运行基线。
