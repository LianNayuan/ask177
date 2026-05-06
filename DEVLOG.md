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

### ChromaDB 持久化存储
之前稠密向量存入 `index.pkl`，每次查询要加载全部向量到内存。加了 `--chroma` 参数用 ChromaDB 持久化：
- `python build_embeddings.py --chroma` → 向量存入 `chroma_db/` 目录
- 查询时连接 ChromaDB，不占内存，支持增量添加
- 本地模型只需首次加载，后续查询复用

### 检索模式开关
原来检索模式由"有没有建向量索引"决定，不灵活。加了 `--mode` 运行时开关：
- `--mode tfidf`：纯 TF-IDF，不加载 embedding 模型
- `--mode dense`：纯向量检索，TF-IDF 权重为 0
- `--mode hybrid`：混合检索（默认）
- `--dense-weight <n>`：调节向量权重（0.0~1.0）

三种模式都走同一个 `search_diverse()` 入口，只是 `dense_weight` 不同（0 / 1.0 / 配置值）。`ask.py` 和 `server.py` 都支持。

### build_glossary.py
`knowledge/glossary.md` 的手工维护和数据库同步一直缺独立命令。加了 `build_glossary.py`：
- 把 glossary.md 全量同步到 data.db
- `--dry-run` 预览差异
- 解决"往文件加了词条但数据库不更新"的问题

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

## 第六阶段：混合粒度分词与 Agentic RAG (2026-05-06)

### 为什么需要重新设计分词

之前 CJK 分词用的是纯字符 bigram + unigram。用户问"4K 准星枪"，bigram 把 "4K准星枪" 切成 "4K" + "K准" + "准星" + "星枪"，一个专门武器名被打散了。

用户反问：**"主流方案不是用分词库做的吗？"** 确实，ES 有 IK 分词器，Lucene 有 CJKAnalyzer。但我们的场景不需要反向索引，自建 tokenizer 就够了。

最终方案是**混合粒度**——三种粒度各司其职：

| 粒度 | 来源 | 负责什么 |
|------|------|---------|
| 词级 | `jieba.lcut()` | 准确匹配正式术语 "消防栓旋转枪" |
| bigram | 字符滑动窗口 | 部分匹配 / 模糊搜索 |
| unigram | 单字 | 缩略词匹配 "4K" 等 |

非 CJK 文本（英文、数字）继续用 `\w+` 提取。

**武器名保护**：367 把武器名注册进 jieba 自定义词典（`jieba.add_word()`），确保 "4K准星枪" 不分词，与 "4K"（公升4K 的昵称）区分开。

### Agentic RAG —— 让 LLM 自己判断够不够

**问题**：之前的 RAG 是单轮检索——用户问 "射程比 4K 长的武器有哪些"，系统搜一次就把结果喂给 LLM。但要回答这个问题，需要先搜到 4K 的射程数值，再搜所有射程更长的武器——这是推理链条，单轮检索做不到。

**方案**：Agentic RAG。LLM 在每轮检索后做一个三路判断：

```
LLM 分析当前积累的文档 + 用户问题
  
  ├─ 能回答 → 直接输出答案
  │
  ├─ 信息不够，但知道搜什么 → SEARCH: <新查询词>
  │   （例：搜到 4K 射程 96/100，但没搜到其他武器射程 → SEARCH: 全武器射程对比）
  │
  └─ 问题本身模糊 → 追问用户
      （例："那个武器怎么样" → 不知道"那个"是哪个）
```

后来又加了第四条路径——**穷举模式**：

```
├─ "哪些武器有 X" → EXHAUST: <属性关键词>
    （触发全库 TF-IDF 扫描，取所有超过阈值 10% 的 chunk）
```

**迭代控制**：每次迭代重新 embed 新 query（保持 hybrid 质量），去重积累，最多 4 轮。文件过滤只在第一轮生效（后续轮次不限制，避免漏掉）。

**第一版翻车**：第一次实现把文件过滤完全去掉了，导致 "4K" 命中了 365 个文件，检索引擎吃太多噪音。修成只在第一轮用 file_filter。

**第二版翻车**：LLM 输出 "答案\n\nSEARCH: 新查询" —— `startswith("SEARCH:")` 匹配不上。改成正则 `re.search(r"SEARCH:\s*(.+?)(?:\n|$)", ...)`。

**第三版翻车**：穷举后 LLM 又触发新的 SEARCH/EXHAUST，造成无限循环。改成穷举后立即 `_force_answer()`，不继续循环。

### 模型把 "4K" 当成 "4K准星枪"

用户发现：问 "比4K长的武器有哪些"，回答 "4K准星枪"，把 "4K"（公升4K 的昵称）和 "4K准星枪"（另一把武器）混淆了。

根因：**LLM 不知道 "4K" 在 Splatoon 语境里特指公升4K。**

**修复一：昵称消歧义提示词**

`_find_relevant_files()` 匹配到昵称后，自动生成消歧义提示词注入 LLM 的问题文本：

```
[System: In the user's question, '4K' = 公升4K. 
 These refer ONLY to the canonical weapon, not to other weapons 
 whose names happen to contain the same characters.]
```

这是通用的——不管用户用哪个昵称，只要昵称表里有，就自动生成对应消歧义。

**修复二：昵称改写**

`_rewrite_nicknames()` —— 在检索前把已知昵称替换为正式名。LLM 可能把"中刷"错误改写为"桶装旋转枪"（中加），昵称表会把它纠正回"斯普拉滚筒"。纯规则驱动，不依赖 LLM。

**修复三：顺序 Bug**

昵称改写把"长弓"替换成"三发猎鱼弓"之后才跑消歧义匹配，导致消歧义提示词变成了 `'弓' = 三发猎鱼弓`，丢掉了"长弓"这个关键信息。

修复：在改写**之前**先用原始问题跑一遍昵称匹配，把 `_matched_nicknames` 保存下来，改写后再恢复。

**修复四：跨语言文件扩展**

昵称匹配到中文文件后，检索被限制在该文件。但信息可能在英文文件里——例如 "长弓是哪一天出的" 匹配到了 `三发猎鱼弓.md`（中文），但日期信息只在 `Tri-Stringer.md`（英文）里。

修复：匹配到文件后，扫描所有文件的昵称表，如果其他文件的昵称列表里有相同武器名，自动加入检索范围。

### 改动文件一览

| 文件 | 改动 |
|------|------|
| `simple_rag.py` | Tokenizer 混合粒度重写、`_register_weapon_names()` jieba 自定义词典、`_analyze_or_answer()` / `_agentic_search()` / `_force_answer()` Agentic 三件套、`_rewrite_nicknames()` 昵称改写、`_find_relevant_files()` 消歧义 + 跨语言扩展、`ask()` 流程重构 |
| `ask.py` | 新增 `--agentic` 参数 |
| `server.py` | 新增 `--agentic` 参数，`/health` 返回 agentic 状态 |
| `requirements.txt` | 新增 `jieba>=0.42` |

---

## 当前状态和下一步

### 已实现
- [x] TF-IDF + 稠密向量混合检索（含 ChromaDB 持久化）
- [x] 检索模式开关（纯 TF-IDF / 纯向量 / 混合，支持调权重）
- [x] 文件名/俗称匹配 + glossary 改写
- [x] 昵称消歧义（自动注入提示词纠正 LLM 混淆）
- [x] 跨语言文件扩展（中英文件互相补充检索）
- [x] 混合粒度分词（jieba 词级 + bigram + unigram）+ 武器名保护
- [x] Agentic RAG（SEARCH / EXHAUST / 回答 / 追问 四路决策，最多 4 轮）
- [x] build_glossary.py 俗称映射同步工具
- [x] 增量构建（只重处理变化的文件）
- [x] CLI 问答 / HTTP API / .exe 三种使用方式
- [x] 本地模型支持（可微调） + API 降级
- [x] SQLite 数据库（问答日志 + 俗称映射 + 会话历史 + 知识元数据）
- [x] 多轮对话（LLM 上下文查询改写 + 会话管理）
- [x] HTTP API 会话管理（session_id）

### 待做
- [ ] 知识库内容校对（部分武器数据可能有误）
- [ ] 补充所有武器的配置文件（目前 355 把，仍需补充）
- [ ] 动态维护 glossary（在多轮对话中自动学习俗称映射）
- [ ] 参考小鱿鱿的翻译数据丰富知识来源
- [ ] 日语支持（分词 + 多语言 embedding）

---

## 一些想法

**关于微调**：`BAAI/bge-small-zh-v1.5` 可以微调。收集一批"口语化问题 → 标准武器描述"的 pair，用对比学习就能让模型更懂 Splatoon 领域语义。`--model` 参数已经预留好了。

**关于 Agentic RAG**：目前最多 4 轮迭代，每轮有 LLM 调用成本（~0.001 USD/轮）。对于"比4K长的武器"这类问题 2-3 轮就够，但复杂推理可能不够。后续可以考虑把迭代上限做成可配置参数。

**关于日语支持**：如果知识库加入日文数据，分词器需要接 MeCab/Janome，embedding 模型需要换多语言版本。当前中英检索能工作是因为 Splatoon 武器名本身就是英文——Token 在 `\w+` 阶段直接字面匹配，不依赖翻译。

**关于 git message 当日记**：这个项目一直用 git commit message 记录开发过程，确实比单独维护日记更自然——commit 不会撒谎，改了什么都写在 diff 里。但这篇日记是对整个演进过程的梳理，适合回头复盘。
