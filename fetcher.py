#!/usr/bin/env python3
import feedparser
import html
import json
from datetime import datetime, timezone
from pathlib import Path

CATEGORIES = ["cs.CL", "cs.AR", "cs.DC"]
NUM_ENTRIES = 20
HISTORY_DIR = Path("history")
RECORD_FILE = Path("record.json")

def fetch_category(cat):
  url = f"https://export.arxiv.org/rss/{cat}"
  feed = feedparser.parse(url)
  papers = []
  for entry in feed.entries[:NUM_ENTRIES]:
    # 提取 arXiv ID
    if "id" in entry:
      arxiv_id = entry.id.split("/")[-1]
    else:
      arxiv_id = entry.link.split("/")[-1]
    papers.append({
      "id": arxiv_id,
      "title": entry.title,
      "link": entry.link,
      "authors": entry.get("author", "Unknown"),
      "summary": entry.get("summary", "").replace("\n", " ").strip(),
      "category": cat
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
  lines = [f"# ArXiv 新论文更新（{date_str}）\n"]
  grouped = {}
  for p in papers:
    grouped.setdefault(p["category"], []).append(p)
  for cat, group in grouped.items():
    lines.append(f"\n## {cat}\n")
    for p in group:
      lines.append(f"### [{p['title']}]({p['link']})")
      lines.append(f"**作者**：{p['authors']}\n\n{p['summary']}\n")
  return "\n".join(lines)

def generate_html(papers):
  grouped = {}
  for p in papers:
    grouped.setdefault(p["category"], []).append(p)
  html_items = []
  for cat, group in grouped.items():
    html_items.append(f"<h2>{cat}</h2>")
    for p in group:
      html_items.append(
        f"<div><h3><a href='{html.escape(p['link'])}'>{html.escape(p['title'])}</a></h3>"
        f"<p><b>作者：</b>{html.escape(p['authors'])}</p>"
        f"<p>{html.escape(p['summary'])}</p></div><hr>"
      )
  html_body = "\n".join(html_items)
  template = open("template.html", encoding="utf-8").read()
  return template.replace("{{CONTENT}}", html_body)

if __name__ == "__main__":
  HISTORY_DIR.mkdir(exist_ok=True)
  date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
  date_file = datetime.now(timezone.utc).strftime("%Y-%m-%d")

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
    with open("LATEST.md", "w", encoding="utf-8") as f:
      f.write(md_text)

    # 存档历史
    hist_file = HISTORY_DIR / f"{date_file}.md"
    with open(hist_file, "w", encoding="utf-8") as f:
      f.write(md_text)

    # 生成 HTML 页面
    html_text = generate_html(new_papers)
    with open("index.html", "w", encoding="utf-8") as f:
      f.write(html_text)
