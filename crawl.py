#!/usr/bin/env python3
"""爬取 splatoon.com.cn 武器详情并生成 Markdown 文档。"""

import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://splatoon.com.cn/",
}


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{e.code}") from e


def strip_html(text: str) -> str:
    """去除 HTML 标签和 <br> 转为换行。"""
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def crawl_weapon(weapon_id: int, output_dir: str = "knowledge/wiki_cn") -> str:
    api_url = f"https://splatoon.com.cn/api/datasource/weapon/weapon/detail/{weapon_id}"
    data = fetch_json(api_url)

    if not data.get("status"):
        raise RuntimeError(f"API 返回错误: {data}")

    contents = {c["id"]: c for c in data["data"]["contents"]}

    # --- 武器名称 ---
    name_raw = data["data"]["name"]
    # 去掉 "主武器详情:" 前缀（如果有）
    name = name_raw.replace("主武器详情:", "").replace("主武器详情：", "").strip()

    # --- 简介 (content id=1) ---
    intro_html = contents[1]["content"]
    # 简介部分不保留换行，合并为连续段落
    intro = strip_html(intro_html).replace("\r", "").replace("\n", "")

    # --- 基础信息 (content id=2) ---
    info_items = contents[2]["content"]
    info_map = {}
    for item in info_items:
        title = item["title"]
        content = item.get("content")
        if isinstance(content, dict):
            # 次要武器 / 特殊武器 / 同源武器
            if item.get("type") == "picList":
                if isinstance(content, list):
                    # 同源武器：排除当前武器自身
                    if title == "同源武器":
                        info_map[title] = ", ".join(
                            c["name"] for c in content
                            if c.get("name") and str(c.get("id")) != str(weapon_id)
                        )
                    # 版本：使用 id 而非 name
                    elif title == "版本":
                        info_map[title] = ", ".join(
                            str(c["id"]) for c in content if c.get("id")
                        )
                    else:
                        info_map[title] = ", ".join(
                            c["name"] for c in content if c.get("name")
                        )
                else:
                    info_map[title] = content.get("name", "")
            else:
                info_map[title] = content.get("name", str(content))
        elif isinstance(content, list):
            if title == "版本":
                info_map[title] = ", ".join(
                    str(c["id"]) if isinstance(c, dict) else str(c)
                    for c in content
                )
            elif title == "同源武器":
                info_map[title] = ", ".join(
                    c["name"] for c in content
                    if isinstance(c, dict) and c.get("name") and str(c.get("id")) != str(weapon_id)
                )
            else:
                info_map[title] = ", ".join(
                    c["name"] if isinstance(c, dict) else str(c) for c in content
                )
        else:
            info_map[title] = str(content) if content is not None else ""

    # --- 基础数据 (content id=3) ---
    base_stats = contents[3]["content"]

    # --- 性能数据 (content id=4) ---
    perf_data = contents[4]["content"]  # dict with categories as keys

    # ========== 生成 Markdown ==========
    page_url = f"https://splatoon.com.cn/weapon/detail/{weapon_id}?pId=7"

    lines = []
    lines.append(f"# {name} - 武器详情（Ver.9.3.0）")
    lines.append(f"来源：[{page_url}]({page_url})")
    lines.append("")
    lines.append("")
    lines.append("## 简介")
    lines.append(intro)
    lines.append("")
    lines.append("## 主武器基础信息")

    info_fields = [
        ("简中名", "简中名"),
        ("繁中名", "繁中名"),
        ("日文名", "日文名"),
        ("英文名", "英文名"),
        ("俗称", "俗称"),
        ("次要武器", "次要武器"),
        ("特殊武器", "特殊武器"),
        ("特殊武器需要的点数", "特殊武器点数"),
        ("解锁等级", "解锁等级"),
        ("武器重量", "武器重量"),
        ("同源武器", "同源武器"),
        ("版本", "版本"),
    ]
    for api_key, label in info_fields:
        value = info_map.get(api_key, "")
        if value:
            lines.append(f"+ {label}：{value}")
        else:
            lines.append(f"+ {label}：")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 主武器基础数据")
    for stat in base_stats:
        lines.append(f"+ {stat['name']}：{stat['ratio']}")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 主武器性能数据（Ver.9.3.0）")

    # 分类顺序和显示名称
    category_order = [
        ("射程", "射程"),
        ("精度", "精度"),
        ("伤害", "伤害"),
        ("射击速率", "射击速率"),
        ("机动性能", "机动性能"),
        ("墨汁效率", "墨汁效率"),
        ("其它", "其它"),
    ]

    def _render_categories(data: dict, prefix: str = "") -> None:
        """将 {分类: [条目]} 渲染为 markdown 列表。"""
        for cat_key, cat_display in category_order:
            if cat_key in data:
                items = data[cat_key]
                if prefix:
                    lines.append(f"#### {prefix} / {cat_display}")
                else:
                    lines.append(f"### {cat_display}")
                for item in items:
                    lines.append(f"+ {item['key_zh']}：{item['value']}")
                lines.append("")

    # 收集所有顶层 key：mode（如 "共通", "未蓄力1圈射击", "1圈蓄力射击", "维持蓄力" 等）
    modes = [(k, v) for k, v in perf_data.items() if isinstance(v, dict)]

    if len(modes) == 1 and modes[0][0] == "共通":
        # 单模式（射击枪等）：直接渲染分类
        _render_categories(modes[0][1])
    else:
        # 多模式（弓、蓄力枪等）：先渲染共通部分，再按模式渲染
        for mode_name, mode_data in modes:
            if mode_name == "共通":
                _render_categories(mode_data)
            else:
                lines.append(f"### {mode_name}")
                # 展开该模式下的所有分类
                for cat_key, cat_display in category_order:
                    if cat_key in mode_data:
                        items = mode_data[cat_key]
                        for item in items:
                            lines.append(f"+ {item['key_zh']}：{item['value']}")
                lines.append("")

    md = "\n".join(lines)
    return md, name


def main():
    if len(sys.argv) < 2:
        print("用法: python crawl.py <武器ID> [输出目录]")
        print("      python crawl.py <起始>-<结束> [输出目录]  (批量)")
        print("示例: python crawl.py 208")
        print("      python crawl.py 1-300")
        print("      python crawl.py 1-300 knowledge/wiki_cn")
        sys.exit(1)

    output_dir = sys.argv[2] if len(sys.argv) > 2 else "knowledge/wiki_cn"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    arg = sys.argv[1]
    if "-" in arg:
        start, end = arg.split("-", 1)
        id_range = range(int(start), int(end) + 1)
    else:
        id_range = [int(arg)]

    # Windows 文件名不允许的字符
    INVALID_CHARS = str.maketrans({"\\": "＼", "/": "／", ":": "：", "*": "＊",
                                    "?": "？", "\"": "＂", "<": "＜", ">": "＞", "|": "｜"})

    success = 0
    skip = 0
    for wid in id_range:
        try:
            md_content, name = crawl_weapon(wid, output_dir)
            safe_name = name.translate(INVALID_CHARS)
            output_path = Path(output_dir) / f"{safe_name}.md"
            output_path.write_text(md_content, encoding="utf-8")
            print(f"[{wid}] {safe_name}.md")
            success += 1
            time.sleep(0.15)
        except Exception as e:
            msg = str(e)
            if "429" in msg:
                print(f"[{wid}] 429 限流，等待 3s...")
                time.sleep(3)
                try:
                    md_content, name = crawl_weapon(wid, output_dir)
                    safe_name = name.translate(INVALID_CHARS)
                    output_path = Path(output_dir) / f"{safe_name}.md"
                    output_path.write_text(md_content, encoding="utf-8")
                    print(f"[{wid}] {safe_name}.md (重试成功)")
                    success += 1
                    continue
                except Exception as e2:
                    print(f"[{wid}] 重试失败: {e2}")
            elif "403" in msg or "404" in msg:
                skip += 1
            else:
                skip += 1

    print(f"\n完成: 成功 {success}, 跳过 {skip}")


if __name__ == "__main__":
    main()
