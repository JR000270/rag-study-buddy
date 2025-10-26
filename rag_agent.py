from helper import get_openai_api_key
from utils import get_doc_tools, get_articles
import nest_asyncio
import os
from pathlib import Path
from llama_index.core import Settings
from llama_index.core.memory import Memory 
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.base.llms.types import ChatMessage, MessageRole
nest_asyncio.apply()

class agent:
    OPENAI_API_KEY = get_openai_api_key()
    Settings.openai_api_key = OPENAI_API_KEY
    
    def __init__(self, llm_model: str ="gpt-3.5-turbo", verbose: bool = True, embed_model: str ="text-embedding-3-small", topic : str = ""):
        llm = OpenAI(model = llm_model)
        Settings.embed_model = OpenAIEmbedding(model= embed_model)
        
        self.download_count = 0
        self.download_limit = 5
        #Memory with 2000 token limit, 70% of tokens for raw messages not summarized by llm
        #higher ratio is set = more space for raw messages (more detailed memory)
        self.memory = Memory.from_defaults(
            session_id= "study_session",
            token_limit= 2000,
            chat_history_token_ratio=0.7
        )

        #get tools for all files in documents
        initial_tools = self.get_tools(self.docs())
        external_context_tool = self.get_external_context_tool()

        prefix_prompt = ("Your primary goal is to assist as a study buddy while lightly incorporating "
        "wittiness and humor to be supportive. Importantly keep the user on topic, "
            f"which is: {topic}. Use your specialized tools to answer any specific questions about this topic."
        )

        prefix_msg = ChatMessage(role=MessageRole.SYSTEM, content=prefix_prompt)
        #workflow of agent
        self.workflow = FunctionAgent(
            tools = initial_tools,
            llm = llm,
            verbose = verbose,
            prefix_messages = [prefix_msg],
            memory = self.memory
        )   
    
    def docs(self) -> list[str]:
        #add files in documents folder to docs list
        docs = []
        docs_directory = 'documents'
        
        #make sure the directory is there
        if not os.path.isdir(docs_directory):
            print(f"Directory '{docs_directory}' not found.")
            return docs
        
        #go through the directory and check for the files of specified type and add them into the list
        try:
            all_docs = os.listdir(docs_directory)
            for file in all_docs:
                if file.endswith(('.txt', '.pdf', '.csv')):
                    docs.append(os.path.join(docs_directory, file))
        except OSError as e:
            print(f"Error reading directory '{docs_directory}': {e}")

        return docs

    #returns list of tools
    def get_tools(self, docs: list[str]) -> list:
        paper_to_tools_dict = {}
        for doc in docs:
            #print(f"Getting tools for paper: {doc}")
            try:
                vector_tool, summary_tool = get_doc_tools(doc, Path(doc).stem)
                paper_to_tools_dict[doc] = [vector_tool, summary_tool]

            except Exception as e:
                print(f"Error processing tools for document:{doc} ; {e}") 

        # store all the tools dictionary keys into a list using a list comprehension
        initial_tools = [t for doc in docs for t in paper_to_tools_dict[doc]]
        return initial_tools 
    
    def get_external_context_tool(self):
        #Creates and returns a FunctionTool that handles downloading new context.
        
        def download_and_process_document(query: str) -> str:
            # Downloads a new document based on the query, generates RAG tools for it, 
            #and adds the new tools to the agent's workflow.
            
            if self.download_count >= self.download_limit:
                return f"Sorry, you have reached the maximum limit of {self.download_limit} external documents."
            
            # 1. Topic Focus: Use the LLM's query (its reasoning) as the search term.
            initial_doc_list = self.docs()
            
            # Call the external utility to download the document
            get_articles(query) 
            
            # 2. Tool Generation: Find the newly downloaded file(s) and process them.
            new_doc_list = self.docs()
            new_docs = [doc for doc in new_doc_list if doc not in initial_doc_list]
            
            if not new_docs:
                return f"Could not find any new documents for the query: '{query}'."

            new_tools = []
            for doc in new_docs:
                vector_tool, summary_tool = get_doc_tools(doc, Path(doc).stem)
                new_tools.extend([vector_tool, summary_tool])
                print(f"Added new tools for document: {doc}")

            # 3. Dynamic Update: Add the new tools to the agent's workflow
            self.workflow.tool_runner.add_tools(new_tools)
            self.download_count += 1
            
            return f"Successfully found and added {len(new_docs)} new document(s) to the RAG context for the topic: '{query}'. You can now ask questions about them."

        # Return the FunctionTool wrapper
        return FunctionTool.from_defaults(
            fn=download_and_process_document,
            name="download_external_context",
            description=(
                "Use this tool ONLY when the current context is insufficient to answer a user's question. "
                "The tool takes one argument: 'query', which should be the specific topic or entity "
                "that needs new context (e.g., 'recent developments in fusion power')."
            )
        )
    
    #run a given input query through the agent workflow
    async def __call__(self, query: str) -> str:
        try:
            #convert history 
            response = await self.workflow.run(user_msg = query, memory= self.memory)
        except Exception as e:
            response = f"Error running agent: {e}"
        return response