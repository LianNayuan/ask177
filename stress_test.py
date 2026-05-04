"""Concurrency stress test for the RAG API. Produces a report file.

Usage:
  1. Start the server:   python server.py --port 8000 [--mode tfidf|dense|hybrid]
  2. Run the test:       python stress_test.py
                         python stress_test.py --concurrency 10 --rounds 2
                         python stress_test.py --output report.md

The script fires concurrent requests, measures latency/success, and writes
a self-contained markdown report to stress_report.md (or --output path).
"""

import json
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

API = "http://localhost:8000/ask"
HEALTH = "http://localhost:8000/health"
HEADERS = {"Content-Type": "application/json"}

QUESTIONS = [
    "审查者怎么样",
    "开开的大招是什么",
    "喇叭有几个贴牌",
    "52加仑的副武器是什么",
    "斯普拉射击枪的伤害",
    "What is the range of Splattershot",
    "4K的俗称有哪些",
    "北斋是什么类型的武器",
]


def ask(question: str, session_id: int = 0, timeout: int = 60) -> dict:
    t0 = time.time()
    try:
        body = json.dumps({"question": question, "session_id": session_id}).encode()
        req = urllib.request.Request(API, data=body, headers=HEADERS, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        ms = int((time.time() - t0) * 1000)
        return {"latency_ms": ms, "status": 200,
                "answer_len": len(data.get("answer", "")), "error": None}
    except Exception as e:
        ms = int((time.time() - t0) * 1000)
        return {"latency_ms": ms, "status": -1,
                "answer_len": 0, "error": str(e)[:200]}


def server_info() -> dict | None:
    try:
        req = urllib.request.Request(HEALTH)
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def run(concurrency: int = 5, rounds: int = 1) -> tuple[list[dict], float]:
    total = concurrency * rounds
    tasks = []
    for r in range(rounds):
        for i in range(concurrency):
            q = QUESTIONS[i % len(QUESTIONS)]
            tasks.append((q, r + 1))

    results = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(ask, q, 0, 90): (q, rn) for q, rn in tasks
        }
        for i, f in enumerate(as_completed(futures)):
            r = f.result()
            results.append(r)
            q, _ = futures[f]
            status = "OK" if r["status"] == 200 else "FAIL"
            err = f"  ERR: {r['error']}" if r["error"] else ""
            print(f"  [{i+1:3d}/{total}] {status}  {r['latency_ms']:5d}ms  "
                  f"({q[:35]}...){err}")

    elapsed = time.time() - start
    return results, elapsed


def percentile(sorted_vals: list[float], p: float) -> float:
    """p in 0..1"""
    if not sorted_vals:
        return 0
    idx = int(len(sorted_vals) * p)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def build_report(concurrency: int, rounds: int,
                 results: list[dict], elapsed: float,
                 info: dict | None, out_path: str) -> str:
    tz = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M CST")
    total = len(results)
    latencies = sorted(r["latency_ms"] for r in results)
    success = [r for r in results if r["status"] == 200]
    errors = [r for r in results if r["error"]]

    # ── Build Markdown ──
    lines = []
    lines.append("# RAG API 并发测试报告")
    lines.append(f"**生成时间**: {tz}")
    lines.append(f"**测试命令**: `python stress_test.py --concurrency {concurrency} --rounds {rounds}`")
    lines.append("")

    # Server info
    lines.append("## 1. 服务端配置")
    if info:
        lines.append(f"| 项目 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 接口 | `{API}` |")
        lines.append(f"| 状态 | `{info.get('status', '?')}` |")
        lines.append(f"| 文件数 | {info.get('files', '?')} |")
        lines.append(f"| Chunk 数 | {info.get('chunks', '?')} |")
    else:
        lines.append("(服务器不可达)")
    lines.append("")

    # Test config
    lines.append("## 2. 测试配置")
    lines.append(f"| 项目 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| 并发数 (workers) | {concurrency} |")
    lines.append(f"| 轮次 | {rounds} |")
    lines.append(f"| 总请求数 | {total} |")
    lines.append(f"| 不同问题数 | {len(QUESTIONS)} |")
    lines.append("")

    # Results
    lines.append("## 3. 结果总览")
    fail_count = len(errors)
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| 成功 | {len(success)}/{total} |")
    lines.append(f"| 失败 | {fail_count}/{total} |")
    lines.append(f"| 总耗时 | {elapsed:.1f}s |")
    lines.append(f"| 吞吐量 | {total / elapsed:.2f} req/s |")
    lines.append("")

    if latencies:
        lines.append("## 4. 延迟分布 (ms)")
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 最小 | {min(latencies)} |")
        lines.append(f"| 中位 (P50) | {int(percentile(latencies, 0.5))} |")
        lines.append(f"| P90 | {int(percentile(latencies, 0.90))} |")
        lines.append(f"| P95 | {int(percentile(latencies, 0.95))} |")
        lines.append(f"| P99 | {int(percentile(latencies, 0.99))} |")
        lines.append(f"| 最大 | {max(latencies)} |")
        lines.append(f"| 平均 | {sum(latencies) // len(latencies)} |")
        lines.append("")

    # Error details
    if errors:
        lines.append("## 5. 错误详情")
        for i, e in enumerate(errors[:10]):
            lines.append(f"{i+1}. `{e['error']}`")
        lines.append("")

    # Per-request table
    lines.append("## 6. 逐请求记录")
    lines.append("| # | 耗时(ms) | 状态 | 答案长度 | 问题 |")
    lines.append("|---|----------|------|----------|------|")
    for i, r in enumerate(results):
        status = "OK" if r["status"] == 200 else "FAIL"
        err = r["error"] or ""
        # Use the question from the results; we don't have it stored in r
        # so just show the index
        lines.append(f"| {i+1} | {r['latency_ms']} | {status} | {r['answer_len']} | "
                     f"request #{i+1} {err[:40] if err else ''} |")
    lines.append("")

    # Conclusion
    lines.append("## 7. 结论")
    if fail_count == 0:
        lines.append(f"{concurrency} 并发 × {rounds} 轮，{total} 个请求全部通过。")
    else:
        lines.append(f"{concurrency} 并发 × {rounds} 轮，{total} 个请求，{fail_count} 个失败。")
    if latencies:
        lines.append(f"平均延迟 {sum(latencies)//len(latencies)}ms，P90 延迟 {int(percentile(latencies, 0.90))}ms。")
    lines.append(f"吞吐量 {total / elapsed:.2f} req/s。")
    lines.append("")

    report = "\n".join(lines)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    return report


def main(concurrency: int = 5, rounds: int = 1, out_path: str = "stress_report.md"):
    info = server_info()
    if not info:
        print("ERROR: Server not reachable at http://localhost:8000")
        print("Start it first: python server.py --port 8000 [--mode tfidf]")
        sys.exit(1)

    total = concurrency * rounds
    print(f"Server:  {HEALTH} → {json.dumps(info)}")
    print(f"Config:  concurrency={concurrency}  rounds={rounds}  total={total}")
    print()

    results, elapsed = run(concurrency, rounds)

    report = build_report(concurrency, rounds, results, elapsed, info, out_path)
    print(f"\nReport saved → {out_path}")


if __name__ == "__main__":
    concurrency = 5
    rounds = 1
    out_path = "stress_report.md"
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--concurrency" and i + 1 < len(args):
            concurrency = int(args[i + 1])
            i += 2
        elif args[i] == "--rounds" and i + 1 < len(args):
            rounds = int(args[i + 1])
            i += 2
        elif args[i] == "--output" and i + 1 < len(args):
            out_path = args[i + 1]
            i += 2
        else:
            print(f"Unknown: {args[i]}")
            print("Usage: python stress_test.py [--concurrency N] [--rounds N] [--output path.md]")
            sys.exit(1)
    main(concurrency, rounds, out_path)
