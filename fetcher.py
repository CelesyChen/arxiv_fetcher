#!/usr/bin/env python3
import feedparser
import html
from datetime import datetime, timezone
from pathlib import Path
import json
import os
from typing import List 
import re
CATEGORIES = ["cs.PF", "cs.AR", "cs.DC", "cs.OS"]
NUM_ENTRIES = 20
DOCS = Path("docs/")
HISTORY_DIR = Path("docs/history")
RECORD_FILE = Path("docs/record.json")

def fetch_category(cat):
  url = f"https://export.arxiv.org/rss/{cat}"
  feed = feedparser.parse(url)
  papers = []
  for entry in feed.entries[:NUM_ENTRIES]:
    link = entry.get("link", "")
    arxiv_id_match = re.search(r"arxiv\.org/abs/([0-9]+\.[0-9]+)", link)
    arxiv_id = arxiv_id_match.group(1) if arxiv_id_match else link.split("/")[-1]

    authors = entry.get("author", "Unknown")
    raw_desc = entry.get("description", "")
    # 去掉 HTML 标签
    summary = re.sub(r"<.*?>", "", raw_desc)
    summary = summary.replace("\n", " ").replace("  ", " ").strip()

    if "Abstract:" in summary:
      summary = summary.split("Abstract:", 1)[1].strip()
    else:
      summary = summary.strip()

    # 分类标签（category）
    category = entry.get("tags", [])
    if category:
      category = ", ".join([t["term"] for t in category if "term" in t])
    else:
      category = cat

    papers.append({
      "id": arxiv_id,
      "title": entry.get("title", "").strip(),
      "link": link,
      "authors": authors,
      "summary": summary,
      "category": category
    })
  return papers

def load_record():
  if RECORD_FILE.exists():
    with open(RECORD_FILE, "r", encoding="utf-8") as f:
      return set(json.load(f))
  return set()

def save_record(record):
  with open(RECORD_FILE, "w", encoding="utf-8") as f:
    json.dump(sorted(list(record)), f, ensure_ascii=False, indent=2)

def generate_markdown(papers, date_str):
  lines = [f"# ArXiv 新论文更新（{date_str})\n"]
  grouped = {}
  for p in papers:
    grouped.setdefault(p["category"], []).append(p)
  for cat, group in grouped.items():
    # lines.append(f"\n## {cat}\n")
    for p in group:
      lines.append(f"### [{p['title']}]({p['link']})")
      lines.append(f"**作者**：{p['authors']}\n\n{p['summary']}\n")
      lines.append(f"\n#### {cat}\n")
  return "\n".join(lines)

def generate_html(papers, date_str):
  grouped = {}
  for p in papers:
    grouped.setdefault(p["category"], []).append(p)
  html_items = []
  html_items.append(f"<h1>{date_str}</h1>")
  total = 0
  idx = 1
  for _, group in grouped.items():
    total += len(group)
  for cat, group in grouped.items():
    for p in group:
      html_items.append(
        f"<div><h3><a href='{html.escape(p['link'])}'>{html.escape(p['title'])}</a></h3>"
        f"<h3><a href='{html.escape(p['link'].replace('/abs/', '/pdf/'))}' target='_blank' rel='noopener noreferrer'> <b>[pdf]</b> </a></h3>"
        f"<p>{idx}/{total}</p>"
        f"<p><b>作者：</b>{html.escape(p['authors'])}</p>"
        f"<p>{html.escape(p['summary'])}</p>"
        f"<p><h4>{cat}</h4></p></div><hr>"
      )
      idx += 1
  html_body = "\n".join(html_items)
  template = open("template.html", encoding="utf-8").read()
  return template.replace("{{CONTENT}}", html_body)

if __name__ == "__main__":
  
  DOCS.mkdir(exist_ok=True)
  HISTORY_DIR.mkdir(exist_ok=True)
  date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

  existing = load_record()
  new_papers = []
  for cat in CATEGORIES:
    for p in fetch_category(cat):
      if p["id"] not in existing:
        new_papers.append(p)
        existing.add(p["id"])

  if not new_papers:
    print("No new papers today.")
  else:
    # 保存记录
    save_record(existing)

    # 更新最新 Markdown
    md_text = generate_markdown(new_papers, date_str)
    with open("docs/LATEST.md", "w", encoding="utf-8") as f:
      f.write(md_text)

    # 存档历史
    hist_file = HISTORY_DIR / f"{date_str}.md"
    with open(hist_file, "w", encoding="utf-8") as f:
      f.write(md_text)

    # 生成 HTML 页面
    html_text = generate_html(new_papers, date_str)
    with open("docs/index.html", "w", encoding="utf-8") as f:
      f.write(html_text)
