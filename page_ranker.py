import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')

import requests
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def scrape_content(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])
        return content
    else:
        return ""

def preprocess_content(content):
    tokens = word_tokenize(content.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)

def calculate_similarity(content_list, query):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([query] + content_list)
    
    
    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    return cosine_similarities

def rank_pages(urls, query):
    content_list = []
    for url in urls:
        print(f"Scraping content from: {url}")
        content = scrape_content(url)
        if content:
            processed_content = preprocess_content(content)
            content_list.append(processed_content)
    
    if content_list:
        similarities = calculate_similarity(content_list, query)
        ranked_urls = sorted(zip(urls, similarities), key=lambda x: x[1], reverse=True)
        return ranked_urls
    else:
        return []

if __name__ == "__main__":
    urls = [
        "https://en.wikipedia.org/wiki/Web_scraping",
        "https://realpython.com/beautiful-soup-web-scraper-python/",
        "https://www.geeksforgeeks.org/implementing-web-scraping-python-beautiful-soup/",
        "https://medium.com/geekculture/web-scraping-with-python-a-complete-step-by-step-guide-code-5174e52340ea"
    ]
    
    query = "python web scraping"

    ranked_pages = rank_pages(urls, query)

    print("\nRanked Pages:")
    for idx, (url, score) in enumerate(ranked_pages, start=1):
        print(f"{idx}. {url} - Score: {score:.4f}")
