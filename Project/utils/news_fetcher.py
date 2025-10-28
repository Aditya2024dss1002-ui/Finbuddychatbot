import feedparser
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# Yahoo Finance RSS feeds
YAHOO_RSS_FEEDS = {
    "Top News": "https://finance.yahoo.com/news/rssindex",
    "Markets": "https://finance.yahoo.com/rss/topstories",
    "Crypto": "https://finance.yahoo.com/news/category/crypto/?format=rss",
    "Technology": "https://finance.yahoo.com/news/category/tech/?format=rss",
    "Economy": "https://finance.yahoo.com/news/category/economy/?format=rss",
}


def fetch_article_text(url):
    """Download and extract readable text from a Yahoo Finance article."""
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # Collect all paragraph text
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        article_text = " ".join(paragraphs)

        # Clean up very short or empty pages
        if len(article_text) < 100:
            return ""
        return article_text
    except Exception as e:
        print(f"Error fetching article text: {e}")
        return ""


def fetch_yahoo_finance_news(category="Top News", limit=5):
    """Fetch latest Yahoo Finance news, including full article text."""
    try:
        url = YAHOO_RSS_FEEDS.get(category, YAHOO_RSS_FEEDS["Top News"])
        feed = feedparser.parse(url)

        articles = []
        for entry in feed.entries[:limit]:
            full_text = fetch_article_text(entry.link)

            articles.append({
                "title": entry.title,
                "summary": full_text if full_text else getattr(entry, "summary", "No summary available."),
                "link": entry.link,
                "published": datetime(*entry.published_parsed[:6]).strftime("%Y-%m-%d %H:%M"),
                "source": "Yahoo Finance"
            })

        return pd.DataFrame(articles)
    except Exception as e:
        print(f"Error fetching RSS feed: {e}")
        return pd.DataFrame()
