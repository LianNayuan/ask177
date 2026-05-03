@echo off
chcp 65001 >nul
echo ============================================
echo   Splatoon 武器知识库 - 一键设置
echo ============================================
echo.

:: ── 1. Python 检查 ──
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 需要 Python 3.10+，请先安装 Python
    pause
    exit /b 1
)
echo [1/5] Python 环境: OK

:: ── 2. 依赖安装 ──
echo [2/5] 安装 Python 依赖...
pip install -r requirements.txt -q
if errorlevel 1 (
    echo [ERROR] 依赖安装失败
    pause
    exit /b 1
)

:: ── 3. API Key ──
if not exist ".env" (
    echo.
    echo 请输入 DeepSeek API Key（从 https://platform.deepseek.com 获取）:
    set /p api_key="> "
    echo DEEPSEEK_API_KEY=!api_key! > .env
    echo .env 已创建
) else (
    echo [.env] 已存在，跳过
)

:: ── 4. 爬取数据 ──
echo.
echo [3/5] 爬取中文武器数据（splatoon.com.cn）...
python crawl.py 1-300
if errorlevel 1 (
    echo [ERROR] 中文数据爬取失败
    pause
    exit /b 1
)

echo.
echo [4/5] 爬取英文武器数据（splatoonwiki.org）...
python crawl_en.py
if errorlevel 1 (
    echo [ERROR] 英文数据爬取失败
    pause
    exit /b 1
)

:: ── 5. 构建索引 ──
echo.
echo [5/5] 构建 TF-IDF 检索索引...
python build_tfidf.py
if errorlevel 1 (
    echo [ERROR] 索引构建失败
    pause
    exit /b 1
)

echo.
echo ============================================
echo   设置完成！
echo   运行 python ask.py 开始问答
echo   运行 python server.py 启动 HTTP API
echo ============================================
echo.
echo （可选）构建稠密向量索引以提升检索质量：
echo   pip install sentence-transformers
echo   python build_embeddings.py

pause
