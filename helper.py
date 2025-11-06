# Add your utilities or helper functions to this file.

import os
from dotenv import load_dotenv

def get_openai_api_key():
    load_dotenv("api_key.env")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_api_key

def get_news_api_key():
    load_dotenv("api_key.env")
    news_api_key = os.getenv("NEWS_API_KEY")
    return news_api_key