import os

#llama index imports
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, SummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from typing import List, Optional

#streamlit interactivity
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

#website url to txt file imports
import requests
from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import letter

#news article aggregation and website conversion
from newsapi import NewsApiClient
from helper import get_news_api_key

def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def get_doc_tools(file_path: str, name: str) -> str:
    """Get vector query and summary query tools from a document."""

    # load documents
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)
    vector_index = VectorStoreIndex(nodes)
    
    def vector_query(query: str, page_numbers: Optional[List[str]] = None) -> str:
        """Use to answer questions over a given paper.
    
        Useful if you have specific questions over the paper.
        Always leave page_numbers as None UNLESS there is a specific page you want to search for.
    
        Args:
            query (str): the string query to be embedded.
            page_numbers (Optional[List[str]]): Filter by set of pages. Leave as NONE 
                if we want to perform a vector search
                over all pages. Otherwise, filter by the set of specified pages.
        
        """
    
        page_numbers = page_numbers or []
        metadata_dicts = [
            {"key": "page_label", "value": p} for p in page_numbers
        ]
        
        query_engine = vector_index.as_query_engine(
            similarity_top_k=2,
            filters=MetadataFilters.from_dicts(
                metadata_dicts,
                condition=FilterCondition.OR
            )
        )
        response = query_engine.query(query)
        return response
        
    
    vector_query_tool = FunctionTool.from_defaults(
        name=f"vector_tool_{name}",
        fn=vector_query
    )
    
    summary_index = SummaryIndex(nodes)
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    summary_tool = QueryEngineTool.from_defaults(
        name=f"summary_tool_{name}",
        query_engine=summary_query_engine,
        description=(
            f"Useful for summarization questions related to {name}"
        ),
    )

    return vector_query_tool, summary_tool


def website_to_txt(url: str, filename: str = "default.txt", save_dir: str = "documents"):
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, filename)

    #Testing start
    headers = {
        # This header makes the request look like it's coming from a desktop browser
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    # Pass the headers to requests.get()
    response = requests.get(url, headers=headers)
    response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

    html = response.text
    #Testing end
    #html = requests.get(url).text
    soup = BeautifulSoup(html, "lxml")

    # Extract visible text (just paragraphs for simplicity)
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
    text_content = "\n\n".join(paragraphs)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text_content)

    print(f"Saved {filename} to: {output_path}")
    return output_path

#collect files for context on topic
def get_articles(topic: str):
    api_key = get_news_api_key()
    news_api = NewsApiClient(api_key=api_key)

    #get 5 articles on the given topic
    all_articles = news_api.get_everything(
        q=topic,
        language='en',
        sort_by='relevancy',
        page=1,
        page_size=7   # <-- limit to 7 results
    )
    #use the dictionary given by all_articles to get the links to the websites
    urls = [article['url'] for article in all_articles['articles']]
    for i,url in enumerate(urls):
        try:
            website_to_txt(url, f"context_file{i}.txt")
        except requests.exceptions.HTTPError as e:
            print(f"Ran into an error converting website {i+1} to .txt: {e}")
        
def download_uploaded_file(file: UploadedFile ) -> bool:
    #determine where the file will be saved to and its name
    save_path = os.path.join("documents", file.name)

    #write the uploaded file into this location
    with open(save_path, "wb") as f:
        f.write(file.getbuffer())
    
    return os.path.isfile(save_path)

def clear_docs():
    #go through the documents folder and clear the previous context files
    docs_directory = "documents"
    try:
        all_docs = os.listdir(docs_directory)
        for file in all_docs:
            file_path = os.path.join(docs_directory, file)
            if file.endswith(('.txt','.pdf', '.csv')):
                os.remove(file_path)
    except OSError as e:
        print(f"Error reading directory '{docs_directory}': {e}")