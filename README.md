# Splatoon 武器知识库 RAG 问答系统

基于 RAG (Retrieval-Augmented Generation) 的 Splatoon 武器知识问答系统。录入 167 把武器的详细数据，支持通过 CLI / HTTP API / 独立 .exe 三种方式进行中文自然语言问答。

## 功能特性

- **知识库**：167 把武器的完整资料（伤害、射程、技能等），支持俗称/口语化改写
- **智能检索**：TF-IDF + 稠密向量混合检索，文件名/俗称匹配，跨文件综合回答
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
│  ├─ 口语改写 (glossary.md)                │
│  ├─ 文件名/俗称匹配 → 缩小检索范围        │
│  ├─ TF-IDF 检索 (TfidfRetriever)          │
│  ├─ 稠密向量检索 (DenseRetriever)          │
│  └─ DeepSeek API 生成答案                 │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  知识库 (knowledge/wiki_cn/*.md)          │
│  167 个武器 .md 文件 + glossary.md        │
│                      │                    │
│  离线预处理                                │
│  build_tfidf.py       → TF-IDF 索引         │
│  build_embeddings.py → 稠密向量           │
│  合并保存到 index.pkl                     │
└──────────────────────────────────────────┘
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

在项目根目录创建 `.env` 文件：

```
DEEPSEEK_API_KEY=sk-your-key-here
```

### 3. 构建索引

```bash
python build_tfidf.py                    # 构建 TF-IDF 关键词索引（必须）
python build_embeddings.py         # 构建稠密向量索引（可选，推荐）
```

生成 `index.pkl`。之后如果你修改了 `knowledge/wiki_cn/` 下的 `.md` 文件，重新运行 `python build_tfidf.py` 即可增量更新（向量不需要每次重建）。

`build_embeddings.py` 支持两种模式：
```bash
python build_embeddings.py                     # 本地模型（默认，需 pip install sentence-transformers）
python build_embeddings.py --mode api          # DeepSeek API（无需额外依赖）
python build_embeddings.py --model ./my-model  # 使用微调后的本地模型
```

### 4. 使用

**方式一：命令行问答**

```bash
python ask.py
> 斯普拉滚筒的伤害是多少？
> 有什么适合新手的武器？
> /add 红牙刷=斯普拉射击枪    # 添加俗称映射
> /list                        # 查看所有映射
> exit                         # 退出
```

**方式二：HTTP API**

```bash
python server.py --port 8000
```

```bash
# 提问
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "斯普拉滚筒的伤害是多少？"}'

# 健康检查
curl http://localhost:8000/health

# 浏览器打开 http://localhost:8000/docs 有 Swagger 调试页面
```

**方式三：独立 .exe（无需 Python 环境）**

```bash
# 本地操作
python build_tfidf.py            # 1. 构建 TF-IDF 索引
python build_embeddings.py       # 2. 构建稠密向量（可选）
python package.py                # 3. 打包成 dist/ 文件夹

# 把 dist/ 整个复制到目标机器
# 在 dist/ 里创建 .env（写入 DEEPSEEK_API_KEY=xxx）
# 双击 rag-server.exe 即可
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
| *(无)* | 本地模型模式，`BAAI/bge-small-zh-v1.5` |
| `--mode local` | 明确指定本地模型模式 |
| `--mode api` | DeepSeek embedding API 模式（无需额外依赖） |
| `--model <name>` | 指定模型名或本地路径（支持微调后的模型） |
| `--force` | 覆盖已有的向量（默认不覆盖） |

```bash
python build_embeddings.py                         # 本地模型（需 pip install sentence-transformers）
python build_embeddings.py --mode api              # API 模式，无需额外安装
python build_embeddings.py --model ./my-finetuned  # 使用微调的本地模型
python build_embeddings.py --force                 # 覆盖已有向量
```

- 必须先运行 `build_tfidf.py`（需要 `index.pkl`）
- 不运行时系统退化为纯 TF-IDF 检索，不影响使用

### ask.py — 命令行问答

| 参数 | 说明 |
|------|------|
| *(无)* | 默认模式，打印检索日志（改写、匹配等） |
| `--debug` | 额外打印每个 chunk 的匹配分数和来源文件 |
| `-q` | 安静模式，只输出答案，不打印日志 |

```
交互命令：
  /add 俗称=正式名   添加俗称映射（如 /add 红牙刷=斯普拉射击枪）
  /list              列出所有映射
  /del 俗称          删除映射
  exit / quit / q    退出
```

```bash
python ask.py              # 默认模式
python ask.py --debug      # 调试模式（查看检索细节）
python ask.py -q           # 安静模式
```

### server.py — HTTP API

| 参数 | 说明 |
|------|------|
| `--host <ip>` | 监听地址，默认 `0.0.0.0` |
| `--port <n>` | 监听端口，默认 `8000` |
| `--cache <path>` | 索引文件路径，默认 `index.pkl` |

| 端点 | 方法 | 说明 |
|------|------|------|
| `/ask` | POST | 问答，body: `{"question": "..."}` |
| `/health` | GET | 健康检查 |
| `/docs` | GET | Swagger 调试页面 |

```bash
python server.py                                  # 默认 0.0.0.0:8000
python server.py --host 127.0.0.1 --port 8080     # 仅本地
python server.py --cache /path/to/index.pkl       # 指定索引文件

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

## 项目文件

| 文件 | 说明 |
|------|------|
| `simple_rag.py` | RAG 核心引擎：TF-IDF + 稠密向量检索 + DeepSeek 问答 |
| `build_tfidf.py` | TF-IDF 关键词索引构建（支持增量更新） |
| `build_embeddings.py` | 稠密向量索引构建（本地模型 / API） |
| `ask.py` | 命令行问答界面 |
| `server.py` | HTTP API 服务器（FastAPI + uvicorn） |
| `package.py` | 打包成独立 .exe |
| `crawl.py` | 知识数据爬取脚本 |
| `knowledge/wiki_cn/` | 167 个武器知识 .md 文件 |
| `knowledge/glossary.md` | 俗称 → 正式名映射表 |
| `requirements.txt` | Python 依赖 |

## 当前的 todo

- [ ] 知识库内容还需要校对（部分武器数据可能有误）
- [ ] 补充所有武器的配置文件（目前 167 把，仍需补充）
- [ ] 支持多轮对话（目前是单轮问答）
- [ ] 支持动态维护 glossary（在多轮对话中自动学习俗称映射）
- [x] 引入向量检索（TF-IDF + 稠密向量混合检索，支持本地模型微调）
- [ ] 参考 [小鱿鱿](https://github.com/Cypas/splatoon3-schedule) 的翻译数据，丰富知识来源

## License

MIT
