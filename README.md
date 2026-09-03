# Study Munch AI

A RAG-powered study buddy built with Streamlit and LlamaIndex. Upload your study materials, give it a topic, and chat with an agent that grounds its answers in your documents *and* automatically-gathered background context — pulling in fresh news articles, and expanding its own knowledge base mid-conversation when it hits a gap.

## Features

- **Document-grounded chat** — upload up to 3 PDF/TXT/CSV files; each gets its own vector search tool and summarization tool via LlamaIndex.
- **Automatic context gathering** — on session start, fetches relevant news articles for your topic (via NewsAPI) and indexes them alongside your uploads.
- **Self-expanding knowledge base** — mid-conversation, the agent can decide it needs more context and pull in new articles on its own (capped by a download limit).
- **Configurable agent** — pick the model (`gpt-3.5-turbo` / `gpt-4.1`), temperature, response length, and a per-session token budget from the UI.
- **Token usage tracking** — counts prompt/completion tokens per query and warns as you approach your session's token limit.
- **Custom themed UI** — dark cyberpunk-styled Streamlit interface (`style.css`, `.streamlit/config.toml`).

## Tech Stack

- [Streamlit](https://streamlit.io/) — UI
- [LlamaIndex](https://www.llamaindex.ai/) — RAG indexing, agent workflow, memory, token counting
- OpenAI (`gpt-3.5-turbo` / `gpt-4.1`, `text-embedding-3-small`) — LLM + embeddings
- [NewsAPI](https://newsapi.org/) — topic-relevant article discovery
- BeautifulSoup + Requests — scraping article text from URLs
- tiktoken — token counting

## Project Structure

```
code/
├── main.py         # Streamlit app: welcome page + chat room UI
├── rag_agent.py     # `agent` class: LlamaIndex FunctionAgent, tools, memory, token tracking
├── utils.py         # Doc-to-tool conversion, article fetching, file upload/cleanup, CSS loader
├── helper.py        # API key loading from api_key.env
├── style.css         # Custom UI theme
├── .streamlit/
│   └── config.toml   # Streamlit theme + server config
├── documents/         # Uploaded/fetched files land here (cleared on exit)
└── api_key.env        # Your API keys (not committed — see Setup)
```

## Setup

1. **Clone and enter the project**
   ```bash
   git clone https://github.com/JR000270/rag-study-buddy
   cd rag-study-buddy/code
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   # source .venv/bin/activate   # macOS/Linux

   pip install -r requirements.txt
   ```

3. **Add your API keys**

   Create `api_key.env` in the `code/` directory:
   ```env
   OPENAI_API_KEY="your-openai-api-key"
   NEWS_API_KEY="your-newsapi-key"
   ```
   This file is already covered by `.gitignore` — never commit it.

4. **Run the app**
   ```bash
   streamlit run main.py
   ```

## Usage

1. Open the app and navigate to **Chat Room** via the sidebar.
2. Upload up to 3 study materials (PDF/TXT/CSV, 1 MB max each) — optional.
3. Enter a study topic/goal.
4. (Optional) Expand **Advanced** to pick the model, creativity, response depth, and token budget.
5. Click **Start Studying** and chat with the agent. Use **New Session** to reset and clear indexed documents.

## Known Limitations

- `get_wikipedia_tool` in [rag_agent.py](../code/rag_agent.py) is a work-in-progress stub (unimplemented, and currently defined outside the `agent` class), so the agent doesn't yet have a working Wikipedia lookup tool.
- The `documents/` folder is wiped on exit (`atexit`), so indexed content doesn't persist across restarts.
