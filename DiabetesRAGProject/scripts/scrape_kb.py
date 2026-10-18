import requests
from bs4 import BeautifulSoup
import os
import re

# KB folder path
kb_folder = "../data/"
os.makedirs(kb_folder, exist_ok=True)

# Example URLs to scrape
urls = {
    "lifestyle_guidelines.txt": "https://www.mayoclinic.org/diseases-conditions/diabetes/in-depth/diabetes-management/art-20047963",
    "diet_recommendations.txt": "https://www.diabetes.org/nutrition/healthy-food-choices",
    "symptoms_and_signs.txt": "https://www.cdc.gov/diabetes/basics/symptoms.html"
}

for filename, url in urls.items():
    print(f"Scraping {url} ...")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "lxml")
            # Extract all paragraph text
            paragraphs = soup.find_all('p')
            text = " ".join([p.get_text() for p in paragraphs])
            
            # Clean text
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\[.*?\]', '', text)
            text = re.sub(r'\(.*?\)', '', text)
            
            # Save to file
            with open(os.path.join(kb_folder, filename), "w", encoding="utf-8") as f:
                f.write(text)
            print(f"✅ Saved to {filename}")
        else:
            print(f"Failed to fetch {url}")
    except Exception as e:
        print(f"Error scraping {url}: {e}")
