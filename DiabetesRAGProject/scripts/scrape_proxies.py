import os, requests, random, time
from bs4 import BeautifulSoup
from tqdm import tqdm

OUTPUT_DIR = "../data/kb"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROXIES_FILE = "proxies.txt"
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (X11; Linux x86_64)",
]

SOURCES = {
    "lifestyle_guidelines": [
        "https://www.who.int/news-room/fact-sheets/detail/diabetes",
        "https://www.cdc.gov/diabetes/managing/index.html",
        "https://www.niddk.nih.gov/health-information/diabetes/overview"
    ],
    "diet_recommendations": [
        "https://www.cdc.gov/diabetes/managing/eat-well.html",
        "https://www.niddk.nih.gov/health-information/diabetes/overview/diet-eating-physical-activity"
    ],
    "symptoms_and_signs": [
        "https://www.mayoclinic.org/diseases-conditions/diabetes/symptoms-causes/syc-20371444",
        "https://www.who.int/news-room/fact-sheets/detail/diabetes"
    ],
    "complications": [
        "https://www.niddk.nih.gov/health-information/diabetes/overview/preventing-problems",
        "https://www.cdc.gov/diabetes/basics/complications.html"
    ]
}

def get_proxies():
    if not os.path.exists(PROXIES_FILE):
        return []
    with open(PROXIES_FILE) as f:
        return [line.strip() for line in f if line.strip()]

def scrape_page(url, proxy=None):
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    try:
        proxies = {"http": proxy, "https": proxy} if proxy else None
        r = requests.get(url, headers=headers, proxies=proxies, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        return "\n".join(paragraphs)
    except Exception as e:
        print(f"⚠ Failed to scrape {url}: {e}")
        return ""

def main():
    proxies = get_proxies()
    for topic, urls in SOURCES.items():
        combined = []
        for url in tqdm(urls, desc=f"Scraping {topic}"):
            proxy = random.choice(proxies) if proxies else None
            text = scrape_page(url, proxy)
            if text:
                combined.append(f"URL: {url}\n\n{text}\n")
            time.sleep(random.uniform(2, 5))
        outfile = os.path.join(OUTPUT_DIR, f"{topic}.txt")
        with open(outfile, "w") as f:
            f.write("\n".join(combined))
        print(f"✅ Saved {outfile}")

if __name__ == "__main__":
    main()
