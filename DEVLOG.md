# 开发日记

> 2026-04 ~ 2026-05，记录 Splatoon 武器知识库 RAG 系统的演进过程。

## 第一阶段：跑通 RAG 基本流程 (2026-04)

### init
搭了一个最简 RAG：加载 .md 文件 → 切 chunk → TF-IDF 检索 → 调 DeepSeek 生成答案。核心文件 `simple_rag.py`，不到 300 行。

### 补充知识文档，支持跨文件检索
录入第一批武器 .md 文件（从 splatoon.com.cn 爬）。TF-IDF 检索从单文件扩展到跨文件——一次提问可以命中多把武器。

### 用 gitignore 忽略 .env
`.env` 里有 API key，加进 `.gitignore` 防止泄漏。

### 拆分预处理和问答
把 `build.py`（离线建索引）和 `ask.py`（在线问答）拆成两个文件。索引序列化到 `index.pkl`，问答时加载即用。同时修正了一些武器知识里的错误数据。

### 文件名匹配召回
提问时先搜文件名，能判断"问的是哪个武器文件"再搜对应文件。这是后来俗称匹配的雏形——最早只匹配文件名里的关键词。

### 口语化词汇对照表
加了 `glossary.md`，用户可以自己维护"红牙刷=斯普拉射击枪"这样的映射。提问时先把口语词替换成正式名再检索。这就是 glossary rewrite 机制的起点。

### 俗称字段匹配
武器 .md 文件里加了「俗称：红牙刷」这样的字段。build 时提取俗称，提问时直接匹配——不需要维护 glossary 也能找到对应武器。

### 增量构建
**这是一个重要转折点。** 之前改一个 .md 就要全量重建索引，167 个文件很慢。改成对比文件修改时间（mtime），只重处理变化/新增/删除的文件。加了 `--file` 参数可以强制指定文件重建。

这个改动让开发体验好了很多——改完一个武器文件，3 秒就跑完 build，不用等几十秒。

---

## 第二阶段：部署与服务化 (2026-04)

### HTTP API 服务器
加了 `server.py`，FastAPI + uvicorn，`POST /ask` + `GET /health`。用户之前只有 CLI（`ask.py`），现在可以部署到云服务器上。

争论了一个点：云服务器要不要重新跑 build？结论是不需要——本地 build 好 `index.pkl`，上传到服务器，`server.py` 直接加载。服务器不需要 Python 环境之外的任何东西（不需要 source .md 文件）。

### 打包成 .exe
用户想要"复制过去双击就能跑"。用了 PyInstaller，生成 `rag-server.exe`。把 `index.pkl` 也打进 `dist/` 目录，.exe 启动时从旁边读索引和 .env。

遇到的坑：PyInstaller 检测不到 uvicorn 的内部模块（loops, protocols, lifespan），需要 `--hidden-import` 显式指定。

### README
写了一版 README，画了架构图，三种使用方式（CLI / API / .exe），还有当时的 todo 列表。

---

## 第三阶段：检索质量优化 (2026-04 ~ 2026-05)

### 武器俗称表格
用户贴了一个 TSV 格式的俗称对照表（官中译名 + 俗称 + 由来），转成了 markdown 表格 `武器俗称及来源.md`。这张表里有大量武器文件自身没写的俗称。

解析这张表 → 合并到 `_nicknames` → 提问时命中俗称就能找到对应文件。

### picture_ocr 目录
除了 `wiki_cn`（手动整理的武器数据），又加了 `picture_ocr` 目录参与索引——里面有 OCR 提取的额外信息。

### 日志增强
检索过程像黑盒，用户看不懂"为什么命中了这个文件"。加了三层日志：
- `[Glossary]` — 口语改写过程
- `[Title match]` — 文件名/俗称匹配日志
- `[Rewrite]` — 改写前后的对比

同时把 `--debug` 和 `-q` 拆开，不用每次看满屏的 chunk dump。

### 参考文件始终参与检索
`武器俗称及来源.md` 里有俗称的由来/背景信息。但如果用户的问题命中了具体武器，这个参考文件就被排除了——导致"为什么叫红牙刷"这类问题回答不了。

修复：不管命中什么武器，`武器俗称及来源.md` 始终加入检索范围。

---

## 第四阶段：向量检索 (2026-05)

### 设计决策：为什么加向量检索
TF-IDF 是关键词匹配——"伤害高"命中"伤害"，但"怎么提高存活率"就匹配不到防御型武器。向量检索可以捕捉语义相似性，补上这个短板。

### 零依赖 vs 本地模型
一开始为了保持 .exe 小体积，写了一个纯 Python 的 `DenseRetriever`（L2 归一化 + 点积，15 行代码），chunk 向量用 DeepSeek embedding API 生成。

但用户反问：为什么不用现成库？讨论后决定拆开：

- `build_tfidf.py` — TF-IDF 关键词索引（必须，无额外依赖）
- `build_embeddings.py` — 稠密向量索引（可选，支持本地模型或 API）

本地模型选 `BAAI/bge-small-zh-v1.5`（24M 参数，中文优化，CPU 能跑），也支持微调后用 `--model` 加载。

### 混合检索
两路并行检索 → min-max 归一化 → 加权求和 → 多样性筛选。默认权重 0.5（TF-IDF 和 Dense 各一半）。

### 查询时 embedding 的路由问题
存向量时用的本地模型，但查询时 `_embed_query()` 还往 DeepSeek API 发请求——404。因为 DeepSeek 不认 `BAAI/bge-small-zh-v1.5` 这个名字。

修复：加了模型类型判断——`deepseek-` 开头走 API，其他走本地 sentence-transformers 推理。

### 日志可见性
加了启动时的检索模式提示（`TF-IDF only` / `TF-IDF + Dense (model, weight)`），查询时显示 `[Dense] top-5 dense-only hits`，能清楚看到向量检索独立命中了什么。

---

## 第五阶段：多轮对话 (2026-05)

### SQLite 数据库迁移

把结构化数据从 `index.pkl` 迁移到 SQLite (`data.db`)：chunks、sources、文件元数据、titles、nicknames、glossary。pkl 只保留 TF-IDF 稀疏向量和 dense embeddings。

新增表：`query_logs`（问答日志）、`glossary`（俗称映射）、`feedback`（评分）、`conversations` + `messages`（会话管理）、`knowledge_files` + `knowledge_chunks` + `knowledge_nicknames` + `knowledge_meta`（知识元数据）。

ChromaDB 配置也存入 `knowledge_meta` 表（之前放在 pkl 里）。

### CJK 中文分词修复

TF-IDF 的 `\w+` 正则把整段中文当成一个 token，导致中文查询返回零分。改成 CJK 字符二元组分词：`"特殊武器"` → `["特殊", "殊武", "武器"]`。同时保留单字和英文 `\w+` token。

### 多轮对话 —— 踩坑过程

**第一版：规则式补丁**

加了 `last_matched_files` 兜底（代词消解）、`unknown_weapon_hint` 注入（未知武器名时提示 LLM 反问）、短查询拼接历史（"是4k" 拼上 "开开的大招是什么"）、`_classify_intent()` LLM 分类（区分新问题和追问回答）。

结果：6 层 if/else、3 个内存状态变量、规则和 LLM 分类互相干扰。还是不对。

**第二版：让 LLM 全权负责**

砍掉所有手写意图判断。改成 `_rewrite_with_context()`——把最近 10 轮对话 + 当前消息发给 LLM，让它输出一个自包含的搜索词：

```
用户输入 + 历史 → LLM 改写为搜索词 → 检索 → LLM 生成回答
```

- "是4k" → LLM 看了历史知道在问武器大招 → "4K 特殊武器"
- "它的副武器是什么" → LLM 消解"它" → "公升4K 次要武器"
- "第一个" → LLM 根据上下文解析

后端只做确定性工作（TF-IDF、文件匹配、glossary 替换），语义理解全交给 LLM。代码从 ~120 行缩减到 ~50 行，不再有分支打架。

### HTTP API 会话管理

`server.py` 加了 `session_id` 支持：

- 请求里 `session_id: 0` 或不带 → 自动创建新会话
- 带有效 ID → 继续该会话，注入对话历史
- 问 `/new` → 返回新 session_id

每次问答自动存入 SQLite `messages` 表。SQLite 加了 `check_same_thread=False` 以兼容 FastAPI 线程池。

### CLI 会话命令

`ask.py` 新增 `/new`、`/sessions`、`/switch <id>`、`/history`、`/stats`、`/feedback` 命令。Windows 中文乱码问题通过 `sys.stdout/stdin.reconfigure(encoding='utf-8')` 修复。

### 大小写不敏感匹配

"4k" 无法匹配 "4K" 武器文件 → `_find_relevant_files` 改用 `.lower()` 比较。

---

## 当前状态和下一步

### 已实现
- [x] TF-IDF + 稠密向量混合检索
- [x] 文件名/俗称匹配 + glossary 改写
- [x] 增量构建（只重处理变化的文件）
- [x] CLI 问答 / HTTP API / .exe 三种使用方式
- [x] 本地模型支持（可微调） + API 降级
- [x] SQLite 数据库（问答日志 + 俗称映射 + 会话历史 + 知识元数据）
- [x] 多轮对话（LLM 上下文查询改写 + 会话管理）
- [x] CJK 中文分词（字符二元组）
- [x] HTTP API 会话管理（session_id）

### 待做
- [ ] 知识库内容校对（部分武器数据可能有误）
- [ ] 补充所有武器的配置文件（目前 173 把，仍需补充）
- [ ] 动态维护 glossary（在多轮对话中自动学习俗称映射）
- [ ] 参考小鱿鱿的翻译数据丰富知识来源

---

## 一些想法

**关于数据库**：已用 SQLite 存储问答日志、俗称映射、会话历史、知识元数据。`index.pkl` 仅保留 TF-IDF 向量和 embeddings。

**关于微调**：`BAAI/bge-small-zh-v1.5` 可以微调。收集一批"口语化问题 → 标准武器描述"的 pair，用对比学习就能让模型更懂 Splatoon 领域语义。`--model` 参数已经预留好了。

**关于 git message 当日记**：这个项目一直用 git commit message 记录开发过程，确实比单独维护日记更自然——commit 不会撒谎，改了什么都写在 diff 里。但这篇日记是对整个演进过程的梳理，适合回头复盘。
