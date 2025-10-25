from helper import get_openai_api_key
from utils import get_doc_tools
import nest_asyncio
import os
from pathlib import Path
from llama_index.core import Settings
from llama_index.core.memory import Memory 
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.base.llms.types import ChatMessage, MessageRole
nest_asyncio.apply()

class agent:
    OPENAI_API_KEY = get_openai_api_key()
    Settings.openai_api_key = OPENAI_API_KEY
    
    def __init__(self, llm_model: str ="gpt-3.5-turbo", verbose: bool = True, embed_model: str ="text-embedding-3-small", topic : str = ""):
        llm = OpenAI(model = llm_model)
        Settings.embed_model = OpenAIEmbedding(model= embed_model)
        
        #Memory with 2000 token limit, 70% of tokens for raw messages not summarized by llm
        #higher ratio is set = more space for raw messages (more detailed memory)
        self.memory = Memory.from_defaults(
            session_id= "study_session",
            token_limit= 2000,
            chat_history_token_ratio=0.7,
            #llm = llm
        )

        #get tools for all files in documents
        initial_tools = self.get_tools(self.docs())

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
            prefix_messages = [prefix_msg]
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
    
    #run a given input query through the agent workflow
    async def __call__(self, query: str) -> str:
        try:
            #convert history 
            response = await self.workflow.run(user_msg = query, memory= self.memory)
        except Exception as e:
            response = f"Error running agent: {e}"
        return response