# Splatoon 武器知识库 RAG 问答系统

基于 RAG (Retrieval-Augmented Generation) 的 Splatoon 武器知识问答系统。录入 167 把武器的详细数据，支持通过 CLI / HTTP API / 独立 .exe 三种方式进行中文自然语言问答。

## 功能特性

- **知识库**：360+ 把武器的完整资料（中英双语），支持俗称/口语化改写，自动识别昵称消歧义
- **智能检索**：混合粒度分词（jieba 词级 + bigram + unigram）+ TF-IDF + 稠密向量混合检索，文件名/俗称匹配，跨语言文件自动扩展
- **Agentic RAG**：LLM 自主判断是否需要换关键词再次检索、全库穷举、或追问用户，最多 4 轮迭代
- **穷举模式**：检测到"哪些武器有 X"类枚举问题时自动触发全库 TF-IDF 扫描
- **三种使用方式**：命令行问答 / HTTP API / 独立 .exe 文件
- **增量构建**：修改知识文件后只重处理变化的文件
- **一键部署**：本地构建索引 → 打包成 .exe → 复制到任意机器运行

## 项目架构

```
用户提问
   │
   ▼
┌──────────────────────────────────────────┐
│  入口层                                   │
│  ask.py (CLI)  /  server.py (HTTP API)    │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  SimpleRAG (simple_rag.py)                │
│  ├─ 安全防护（注入检测 + PII 脱敏）        │
│  ├─ LLM 上下文查询改写 (多轮对话)          │
│  ├─ 口语改写 (glossary.md)                │
│  ├─ 昵称改写（367 武器 × N 个昵称，确定性） │
│  ├─ 昵称消歧义（注入提示词纠正 LLM 混淆）   │
│  ├─ 文件名/俗称匹配 → 缩小检索范围        │
│  ├─ 跨语言文件扩展（中英互补）             │
│  ├─ TF-IDF 检索 (TfidfRetriever)          │
│  │   └─ 混合粒度分词：jieba + bigram + unigram │
│  ├─ 稠密向量检索 (DenseRetriever/ChromaDB) │
│  ├─ Agentic RAG 决策循环 (最多 4 轮)       │
│  │   ├─ 能回答 → 输出答案                 │
│  │   ├─ 信息不够 → SEARCH: 新查询词       │
│  │   ├─ 枚举问题 → EXHAUST: 全库扫描       │
│  │   └─ 问题模糊 → 追问用户               │
│  └─ DeepSeek API 生成答案                 │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  知识库 (knowledge/wiki_cn/*.md           │
│         + knowledge/wiki_en/*.md)         │
│  167 + 188 = 355 个武器 .md 文件           │
│  + glossary.md + 武器俗称及来源.md        │
│                      │                    │
│  离线预处理                                │
│  build_tfidf.py       → TF-IDF 索引         │
│  build_embeddings.py → 稠密向量           │
│  合并保存到 index.pkl + data.db           │
└──────────────────────────────────────────┘
```

## 快速开始

### 方式一：一键运行（推荐）

```bash
# Windows
setup.bat

# macOS / Linux
bash setup.sh
```

脚本自动完成：安装依赖 → 配置 API Key → 爬取中英文数据 → 构建索引。

### 方式二：手动分步

```bash
pip install -r requirements.txt              # 1. 安装依赖
# 创建 .env，写入 DEEPSEEK_API_KEY=xxx       # 2. 配置 API Key
python crawl.py 1-300                        # 3. 爬取中文武器数据
python crawl_en.py                           # 4. 爬取英文武器数据
python build_tfidf.py                        # 5. 构建 TF-IDF 索引
python ask.py                                # 6. 开始问答
```

### 使用方式

**命令行问答**

```bash
python ask.py              # 默认模式
python ask.py --debug      # 调试模式
python ask.py -q           # 安静模式
```

**HTTP API**

```bash
python server.py --port 8000
# 浏览器打开 http://localhost:8000/docs 有 Swagger 调试页面
```

**独立 .exe**

```bash
python build_tfidf.py      # 1. 构建索引
python package.py          # 2. 打包成 dist/ 文件夹
# 把 dist/ 复制到目标机器，创建 .env，双击 rag-server.exe
```

## 使用说明

### build_tfidf.py — 构建/更新 TF-IDF 索引

| 参数 | 说明 |
|------|------|
| *(无)* | 自动增量更新（检测文件变动，只处理变化的） |
| `--file <name>` / `-f <name>` | 强制重新处理指定文件，可多次指定 |

```bash
python build_tfidf.py                         # 增量更新
python build_tfidf.py --file 斯普拉滚筒.md      # 强制重处理单个文件
python build_tfidf.py -f A.md -f B.md          # 强制重处理多个文件
```

- 首次运行：全量构建，保存到 `index.pkl`
- 后续运行：自动对比文件修改时间，只重处理变化的文件
- `--file` 可指定文件名、不带扩展名的名字、或 `knowledge/wiki_cn/xxx.md` 这样的相对路径

### build_embeddings.py — 构建稠密向量索引

| 参数 | 说明 |
|------|------|
| *(无)* | 本地模型模式，`BAAI/bge-small-zh-v1.5`，存入 `index.pkl` |
| `--chroma` | 用 ChromaDB 持久化存储向量（推荐） |
| `--mode local` | 明确指定本地模型模式 |
| `--mode api` | DeepSeek embedding API 模式（无需额外依赖） |
| `--model <name>` | 指定模型名或本地路径（支持微调后的模型） |
| `--force` | 覆盖已有的向量（默认不覆盖） |

```bash
python build_embeddings.py --chroma                    # ChromaDB 持久化（推荐）
python build_embeddings.py                             # 本地模型存入 index.pkl
python build_embeddings.py --mode api                  # API 模式，无需额外安装
python build_embeddings.py --model ./my-finetuned      # 使用微调的本地模型
python build_embeddings.py --chroma --force            # 覆盖已有向量
```

- 必须先运行 `build_tfidf.py`（需要 `index.pkl`）
- `--chroma` 存入 `chroma_db/` 目录，查询时无需重新加载模型
- 不运行时系统退化为纯 TF-IDF 检索，不影响使用

### build_glossary.py — 俗称映射同步

| 参数 | 说明 |
|------|------|
| *(无)* | 将 `knowledge/glossary.md` 全量同步到数据库 |
| `--dry-run` | 预览差异，不写入 |

```bash
python build_glossary.py              # 同步
python build_glossary.py --dry-run    # 预览
```

- `knowledge/glossary.md` 手工维护，格式：`口语词 | 正式名`
- 首次运行 `ask.py` 或 `server.py` 时会自动导入；改了文件后手动运行本脚本同步

### ask.py — 命令行问答

| 参数 | 说明 |
|------|------|
| *(无)* | 默认模式，打印检索日志（改写、匹配等） |
| `--debug` | 额外打印每个 chunk 的匹配分数和来源文件 |
| `-q` | 安静模式，只输出答案，不打印日志 |
| `--mode <mode>` | 检索模式：`tfidf` / `dense` / `hybrid`（默认 hybrid） |
| `--dense-weight <n>` | 混合检索中稠密向量的权重，0.0~1.0（默认 0.5） |
| `--agentic` | 启用 Agentic RAG：LLM 自主判断是否需要多次检索 |

```
交互命令：
  /add 俗称=正式名   添加俗称映射（如 /add 红牙刷=斯普拉射击枪）
  /list              列出所有映射
  /del 俗称          删除映射
  /new               开启新会话（清空对话历史）
  /sessions          列出最近会话
  /switch <id>       切换到指定会话
  /history           查看最近问答记录
  /stats             查看查询统计
  /feedback <id> <1-5> 对指定回答评分
  exit / quit / q    退出
```

```bash
python ask.py                                # 默认模式
python ask.py --debug                        # 调试模式（查看检索细节）
python ask.py -q                             # 安静模式
python ask.py --mode tfidf                   # 纯 TF-IDF 检索
python ask.py --mode dense                   # 纯向量检索
python ask.py --mode hybrid --dense-weight 0.7  # 混合检索，调高向量权重
python ask.py --agentic                      # Agentic RAG 模式
```

### server.py — HTTP API

| 参数 | 说明 |
|------|------|
| `--host <ip>` | 监听地址，默认 `0.0.0.0` |
| `--port <n>` | 监听端口，默认 `8000` |
| `--cache <path>` | 索引文件路径，默认 `index.pkl` |
| `--mode <mode>` | 检索模式：`tfidf` / `dense` / `hybrid`（默认 hybrid） |
| `--dense-weight <n>` | 混合检索中稠密向量的权重，0.0~1.0（默认 0.5） |
| `--agentic` | 启用 Agentic RAG：LLM 自主判断是否需要多次检索 |

| 端点 | 方法 | 说明 |
|------|------|------|
| `/ask` | POST | 问答，body: `{"question": "...", "session_id": 0}` |
| `/health` | GET | 健康检查 |
| `/docs` | GET | Swagger 调试页面 |

```bash
python server.py                                  # 默认 0.0.0.0:8000
python server.py --host 127.0.0.1 --port 8080     # 仅本地
python server.py --cache /path/to/index.pkl       # 指定索引文件
python server.py --mode dense                     # 纯向量检索模式
python server.py --agentic                        # Agentic RAG 模式

# 测试
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"斯普拉滚筒的伤害"}'
curl http://localhost:8000/health
```

服务器上部署只需要 `server.py` + `index.pkl` + `.env` 三个文件。

### package.py — 打包 .exe

无参数，直接运行。生成的 `dist/` 目录可复制到任意 Windows 机器运行。

```bash
python build_tfidf.py      # 1. 先构建索引（含 build_embeddings.py）
python package.py          # 2. 打包

# dist/ 目录内容：
#   rag-server.exe   ← 双击启动
#   index.pkl        ← 知识库索引
#   在 dist/ 里创建 .env，写入 DEEPSEEK_API_KEY=xxx
#   双击 rag-server.exe，服务器启动在 http://0.0.0.0:8000
```

### crawl.py — 爬取武器数据

| 参数 | 说明 |
|------|------|
| `<id>` | 单个武器 ID，如 `208` |
| `<start>-<end>` | 批量爬取 ID 范围 |
| `[output_dir]` | 输出目录，默认 `knowledge/wiki_cn` |

```bash
python crawl.py 208                           # 爬取单个武器
python crawl.py 1-300                         # 批量爬取
python crawl.py 1-300 knowledge/wiki_cn       # 指定输出目录
```

## 多轮对话

系统支持多轮对话，核心机制是 **LLM 上下文查询改写**：

```
用户: 开开的大招是什么
  → LLM 检索不到"开开"，反问"开开是什么武器？"
用户: 是4k
  → LLM 根据对话历史，将"是4k"改写为"4K 特殊武器"
  → 检索命中公升4K.md → 回答：弹跳声呐
用户: 它的副武器呢
  → LLM 将"它的"改写为"公升4K"
  → 检索命中 → 回答：墨汁陷阱
```

### 设计原则

- **后端只做确定性工作**：TF-IDF/向量检索、文件名匹配、glossary 替换
- **语义理解全交给 LLM**：代词消解、澄清合并、意图判断
- 不使用手写规则判断"这是新问题还是追问"——LLM 自己决定怎么改写查询

### CLI 多轮

`python ask.py` 启动后自动从 SQLite 恢复上次会话，所有对话自动保存。对话历史（最近 10 轮）传给 LLM 做上下文。

| 命令 | 功能 |
|------|------|
| `/new` | 开始新会话 |
| `/sessions` | 列出最近 10 个会话 |
| `/switch <id>` | 切换到指定会话 |

### API 多轮

客户端负责传递 `session_id`：

```json
// 第一轮：不带 session_id 或设为 0
{"question": "斯普拉滚筒的伤害是多少？"}
// 响应 → {"answer": "...", "session_id": 5}

// 第二轮：带上 session_id 继续对话
{"question": "它的副武器是什么", "session_id": 5}

// 开启新会话
{"question": "/new"}
```

## 项目文件

| 文件 | 说明 |
|------|------|
| `simple_rag.py` | RAG 核心引擎：安全防护 + LLM 查询改写 + 混合粒度分词 + 昵称消歧义 + TF-IDF + 稠密向量检索 + Agentic RAG 决策循环 + DeepSeek 问答 |
| `database.py` | SQLite 数据库：问答日志 + 俗称映射 + 会话历史 + 知识元数据 |
| `build_tfidf.py` | TF-IDF 关键词索引构建（支持增量更新） |
| `build_embeddings.py` | 稠密向量索引构建（本地模型 / API / ChromaDB） |
| `build_glossary.py` | 俗称映射表同步（glossary.md → data.db） |
| `ask.py` | 命令行问答界面 |
| `server.py` | HTTP API 服务器（FastAPI + uvicorn） |
| `package.py` | 打包成独立 .exe |
| `crawl.py` | 中文武器数据爬取 |
| `crawl_en.py` | 英文武器数据爬取 |
| `setup.bat` / `setup.sh` | 一键设置脚本 |
| `knowledge/glossary.md` | 俗称 → 正式名映射表（手工维护） |
| `requirements.txt` | Python 依赖 |

## 当前的 todo

- [ ] 知识库内容还需要校对（部分武器数据可能有误）
- [ ] 补充所有武器的配置文件（目前 355 把，仍需补充）
- [ ] 支持动态维护 glossary（在多轮对话中自动学习俗称映射）
- [ ] 参考 [小鱿鱿](https://github.com/Cypas/splatoon3-schedule) 的翻译数据，丰富知识来源
- [x] 支持多轮对话（LLM 上下文查询改写 + SQLite 会话管理）
- [x] 引入向量检索（TF-IDF + 稠密向量混合检索，支持本地模型微调）
- [x] 引入 SQLite 数据库（问答日志 + 俗称映射 + 会话历史 + 知识元数据）
- [x] 混合粒度分词（jieba 词级 + bigram + unigram）
- [x] Agentic RAG（LLM 自主判断 SEARCH/EXHAUST/回答/追问，最多 4 轮迭代）
- [x] 昵称消歧义 + 跨语言文件扩展

## 数据来源

- 中文武器数据：[splatoon.com.cn](https://splatoon.com.cn)
- 英文武器数据：[Inkipedia](https://splatoonwiki.org)（CC BY-SA 协议）
- 俗称对照表：手工整理

## License

MIT
