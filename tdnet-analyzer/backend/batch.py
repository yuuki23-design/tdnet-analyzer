# backend/batch.py

import os
import re
import time
import requests
from bs4 import BeautifulSoup
import pdfplumber
import pandas as pd
from transformers import pipeline

PDF_DIR = "backend/data/pdfs"
CSV_PATH = "backend/data/tdnet_score_all.csv"
os.makedirs(PDF_DIR, exist_ok=True)

def get_tdnet_links():
    url = "https://www.release.tdnet.info/inbs/I_main_00.html"
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    rows = soup.select("table.tdnet_news_table tr")[1:]

    results = []
    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 4: continue
        title = cols[3].text.strip()
        href = cols[3].find("a")
        if not href: continue
        link = "https://www.release.tdnet.info" + href["href"]
        code = cols[1].text.strip()
        date = cols[0].text.strip()
        results.append({"title": title, "pdf_url": link, "code": code, "date": date})
    return results

def download_pdf(entry):
    date_clean = entry["date"].replace("/", "-")
    title_clean = re.sub(r'[\\/:*?"<>|]', "_", entry["title"])[:30]
    filename = f"{date_clean}_{entry['code']}_{title_clean}.pdf"
    filepath = os.path.join(PDF_DIR, filename)

    if not os.path.exists(filepath):
        res = requests.get(entry["pdf_url"])
        with open(filepath, "wb") as f:
            f.write(res.content)
        time.sleep(0.5)
    return filepath

def extract_text(filepath):
    with pdfplumber.open(filepath) as pdf:
        return "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])

def classify_genre(title, body):
    keywords = {
        "TOB": ["公開買付", "TOB"],
        "自社株買い": ["自己株式取得", "自己株"],
        "増配": ["増配", "配当予想"],
        "上方修正": ["上方修正", "業績予想の修正"],
        "業務提携": ["業務提携", "提携", "協業"],
    }
    for genre, kw_list in keywords.items():
        if any(k in title or k in body for k in kw_list):
            return genre
    return "その他"

def classify_sentiment(text):
    classifier = pipeline("zero-shot-classification", model="nli-deberta-base-japanese")
    result = classifier(text[:512], candidate_labels=["ポジティブ", "ネガティブ", "中立"])
    return result["labels"][0], round(result["scores"][0], 3)

def extract_tob_info(text):
    match = re.search(r"(\d{3,5})円[^\n]{0,10}(買付|買い付け)", text)
    return match.group(1) if match else ""

def extract_buyback_info(text):
    match = re.search(r"([0-9０-９億,，,.円]+)[^\n]{0,10}(取得|買[付い])", text)
    return match.group(1) if match else ""

def run(max_entries=30):
    rows = get_tdnet_links()
    data = []

    for entry in rows[:max_entries]:
        try:
            pdf_path = download_pdf(entry)
            body = extract_text(pdf_path)
            genre = classify_genre(entry["title"], body)
            sentiment, score = classify_sentiment(body)
            tob_price = extract_tob_info(body) if genre == "TOB" else ""
            buyback = extract_buyback_info(body) if genre == "自社株買い" else ""

            base_score = {"TOB": 8, "自社株買い": 6, "業務提携": 4, "上方修正": 5}.get(genre, 0)
            adjustment = 2 if tob_price else 0
            final_score = base_score + adjustment

            data.append({
                "date": entry["date"],
                "code": entry["code"],
                "title": entry["title"],
                "genre": genre,
                "sentiment": sentiment,
                "sentiment_score": score,
                "TOB価格": tob_price,
                "自己株取得額": buyback,
                "スコア": final_score,
                "PDFパス": pdf_path
            })

        except Exception as e:
            print("❌ 処理失敗:", entry["title"], str(e))
            continue

    df = pd.DataFrame(data)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ {CSV_PATH} に保存しました。件数: {len(df)}")

if __name__ == "__main__":
    run()# バッチ処理: PDF取得・解析・CSV出力
