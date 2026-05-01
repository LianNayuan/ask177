# Splatoon 武器知识库 RAG 问答系统

基于 RAG (Retrieval-Augmented Generation) 的 Splatoon 武器知识问答系统。录入 167 把武器的详细数据，支持通过 CLI / HTTP API / 独立 .exe 三种方式进行中文自然语言问答。

## 功能特性

- **知识库**：167 把武器的完整资料（伤害、射程、技能等），支持俗称/口语化改写
- **智能检索**：TF-IDF 检索 + 文件名/俗称匹配，跨文件综合回答
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
│  └─ DeepSeek API 生成答案                 │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  知识库 (knowledge/wiki_cn/*.md)          │
│  167 个武器 .md 文件 + glossary.md        │
│                      │                    │
│  离线预处理 (build.py)                    │
│  .md → chunk → TF-IDF → index.pkl        │
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
python build.py
```

生成 `index.pkl`。之后如果你修改了 `knowledge/wiki_cn/` 下的 `.md` 文件，重新运行 `python build.py` 即可增量更新。

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
python build.py                 # 构建索引
python package.py               # 打包成 dist/ 文件夹

# 把 dist/ 整个复制到目标机器
# 在 dist/ 里创建 .env（写入 DEEPSEEK_API_KEY=xxx）
# 双击 rag-server.exe 即可
```

## 使用说明

### build.py — 构建/更新索引

```bash
python build.py                    # 自动增量更新（默认）
python build.py --file 斯普拉滚筒.md  # 强制重新处理指定文件
python build.py -f A.md B.md       # 多个文件
```

- 首次运行：全量构建索引，保存到 `index.pkl`
- 后续运行：自动检测文件变动，只重处理变化的文件
- `--file` 可以指定文件名、不带扩展名的名字、或相对路径

### ask.py — 命令行问答

```
命令：
  /add 俗称=正式名   添加俗称映射（如 /add 红牙刷=斯普拉射击枪）
  /list              列出所有映射
  /del 俗称          删除映射
```

### server.py — HTTP API

```bash
python server.py --host 0.0.0.0 --port 8000 --cache index.pkl
```

服务器上部署时，只需要 `server.py` + `index.pkl` + `.env` 三个文件。

## 项目文件

| 文件 | 说明 |
|------|------|
| `simple_rag.py` | RAG 核心引擎：TF-IDF 检索 + DeepSeek 问答 |
| `build.py` | 知识库索引构建（支持增量更新） |
| `ask.py` | 命令行问答界面 |
| `server.py` | HTTP API 服务器（FastAPI + uvicorn） |
| `package.py` | 打包成独立 .exe |
| `crawl.py` | 知识数据爬取脚本 |
| `knowledge/wiki_cn/` | 167 个武器知识 .md 文件 |
| `knowledge/glossary.md` | 俗称 → 正式名映射表 |
| `requirements.txt` | Python 依赖 |
superpower
## 当前的 todo

- [ ] 知识库内容还需要校对（部分武器数据可能有误）
- [ ] 补充所有武器的配置文件（目前 167 把，仍需补充）
- [ ] 支持多轮对话（目前是单轮问答）
- [ ] 支持动态维护 glossary（在多轮对话中自动学习俗称映射）
- [ ] 引入向量检索（目前是纯 TF-IDF 关键词匹配，语义理解有限）
- [ ] 参考 [小鱿鱿](https://github.com/Cypas/splatoon3-schedule) 的翻译数据，丰富知识来源

## License

MIT
