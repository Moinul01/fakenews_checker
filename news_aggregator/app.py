import joblib
from flask import Flask, render_template
import feedparser
import random
import requests  # Correct import statement for the requests module

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained ML model and vectorizer (ensure these files are in the model folder)
try:
    model_rf = joblib.load('model/Random_Forest.pkl')  # Load your trained Random Forest model
    model_svm = joblib.load('model/SVM_model.pkl')     # Load your trained SVM model
    vectorizer = joblib.load('model/vectorizer.pkl')   # Load your vectorizer
except Exception as e:
    print(f"Error loading model/vectorizer: {e}")
    model_rf = None
    model_svm = None
    vectorizer = None

# RSS sources
news_sources = {
    "World": "http://feeds.bbci.co.uk/news/world/rss.xml",
    "Technology": "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "Sports": "https://feeds.bbci.co.uk/sport/rss.xml"
}

# NewsAPI settings
NEWS_API_KEY = 'c64c7967d8604ab5b7593c64e84b236e'  
NEWS_API_URL = 'https://newsapi.org/v2/top-headlines'

# Function to extract image (if available in the feed entry)
def extract_image(entry):
    """Extract image URL from RSS feed entry."""
    media = entry.get('media_content', None)
    if media and isinstance(media, list) and 'url' in media[0]:
        return media[0]['url']
    # Sometimes enclosure is used
    enclosure = entry.get('enclosures', None)
    if enclosure and isinstance(enclosure, list) and 'href' in enclosure[0]:
        return enclosure[0]['href']
    return None

# Real-time Fake News Prediction using the pre-trained model
def predict_fake_news(text):
    """Use the ML model to predict whether news is fake or real."""
    if model_rf is None or model_svm is None or vectorizer is None:
        return "Real"  # Default to Real if models/vectorizer are not loaded

    # Vectorize the input text using the loaded vectorizer
    text_vectorized = vectorizer.transform([text])
    
    # Get the prediction from the Random Forest model (binary classification: 0 = Real, 1 = Fake)
    prediction_rf = model_rf.predict(text_vectorized)
    # Get the prediction from the SVM model (binary classification: 0 = Real, 1 = Fake)
    prediction_svm = model_svm.predict(text_vectorized)
    
    # For simplicity, choose the Random Forest model prediction as the final prediction
    return "Fake" if prediction_rf[0] == 1 else "Real"

# Function to fetch news from NewsAPI
def fetch_newsapi_articles():
    """Fetch articles from NewsAPI."""
    url = f'{NEWS_API_URL}?country=us&pageSize=9&apiKey={NEWS_API_KEY}'  # Limit to 9 articles
    response = requests.get(url)
    data = response.json()

    articles = []
    if data.get('status') == 'ok':
        for item in data['articles']:
            title = item.get('title') or ''
            description = item.get('description') or ''
            article = {
                'title': title,
                'summary': description,
                'link': item.get('url', '#'),
                'published': item.get('publishedAt', 'N/A'),
                'image': item.get('urlToImage', None),
                'prediction': predict_fake_news(title + " " + description),
                'source': item.get('source', {}).get('name', 'Unknown Source')
            }
            articles.append(article)
    return articles

@app.route('/')
def home():
    """Home route - displays fake news examples first, then random real news."""
    # Create fake news examples (these will always appear first)
    fake_news_examples = [
        {
            'title': "NASA Discovers Evidence of Alien Life on Mars",
            'summary': "NASA has recently uncovered shocking evidence of extraterrestrial life on Mars that could change everything we know about the universe.",
            'link': "#",
            'published': '2023-05-01',
            'image': None,
            'prediction': "Fake",
            'source': "Fake News Example"
        },
        {
            'title': "New Study Reveals the Earth is Actually Flat",
            'summary': "A controversial new study has revealed that the Earth is, in fact, flat, debunking centuries of scientific understanding.",
            'link': "#",
            'published': '2023-05-02',
            'image': None,
            'prediction': "Fake",
            'source': "Fake News Example"
        },
        {
            'title': "World's First Human Cloning Experiment Succeeds",
            'summary': "Scientists have successfully cloned a human being for the first time in history, marking a monumental breakthrough in genetics.",
            'link': "#",
            'published': '2023-05-03',
            'image': None,
            'prediction': "Fake",
            'source': "Fake News Example"
        }
    ]

    # Collect real news from RSS feeds
    real_articles_rss = []
    for url in news_sources.values():
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title = entry.title or ''
            summary = entry.summary or ''
            article = {
                'title': title,
                'link': entry.link,
                'summary': summary,
                'published': getattr(entry, 'published', 'N/A'),
                'image': extract_image(entry),
                'prediction': predict_fake_news(title + " " + summary),
                'source': entry.get('source', {}).get('title', 'Unknown Source')
            }
            real_articles_rss.append(article)

    # Fetch real news from NewsAPI
    real_articles_api = fetch_newsapi_articles()

    # Combine RSS and API real articles
    combined_real_articles = real_articles_rss + real_articles_api
    random.shuffle(combined_real_articles)  # Shuffle for variety

    # Select 9 real articles
    selected_real_articles = combined_real_articles[:9]

    # Combine articles (fake first, then real)
    combined_articles = fake_news_examples + selected_real_articles

    return render_template("home.html", 
                           categories=news_sources.keys(), 
                           articles=combined_articles)

@app.route('/category/<name>')
def category(name):
    """Category route - shows news articles by category."""
    feed_url = news_sources.get(name)
    if not feed_url:
        return "Category not found", 404

    feed = feedparser.parse(feed_url)
    articles = []
    for entry in feed.entries:
        title = entry.title or ''
        summary = entry.summary or ''
        articles.append({
            'title': title,
            'link': entry.link,
            'summary': summary,
            'published': getattr(entry, 'published', 'N/A'),
            'image': extract_image(entry),
            'prediction': predict_fake_news(title + " " + summary),
            'source': entry.get('source', {}).get('title', 'Unknown Source')
        })

    # Fetch real news from NewsAPI for the selected category
    articles_api = fetch_newsapi_articles()

    # Combine and shuffle the articles from RSS and API
    combined_articles = articles + articles_api
    random.shuffle(combined_articles)

    return render_template("category.html", name=name, articles=combined_articles)

if __name__ == '__main__':
    app.run(debug=True)
