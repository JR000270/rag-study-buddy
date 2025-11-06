from helper import get_openai_api_key
from utils import get_doc_tools, get_articles
import nest_asyncio
import os
from pathlib import Path
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core.memory import Memory 
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
import tiktoken
import traceback
nest_asyncio.apply()

class agent:
    OPENAI_API_KEY = get_openai_api_key()
    Settings.openai_api_key = OPENAI_API_KEY

    def __init__(self, llm_model: str ="gpt-4.1", verbose: bool = True, embed_model: str ="text-embedding-3-small", topic : str = "", token_limit: int = 5000, temperature: float = 0.6, max_response_tokens: int = 1000):

        Settings.embed_model = OpenAIEmbedding(model= embed_model)
        
        self.download_count = 0
        self.download_limit = 5
        self.total_used_tokens = 0
        self.token_limit = token_limit #max tokens per conversation

        
        #get tools for all files in documents
        initial_tools = self.get_tools(self.docs())
        external_context_tool = self.get_external_context_tool()
        initial_tools.append(external_context_tool)

        self.token_counter = TokenCountingHandler(tokenizer= tiktoken.encoding_for_model(llm_model))
        #Settings.callback_manager = CallbackManager(handlers=[self.token_counter])
        
        #adjust default temp to be better suited for educational context, increase max tokens for longer responses
        llm = OpenAI(model = llm_model, temperature= temperature, max_tokens= max_response_tokens)

        #Memory with 2000 token limit, 70% of tokens for raw messages not summarized by llm
        #higher ratio is set = more space for raw messages (more detailed memory)
        self.memory = Memory.from_defaults(
            session_id= "study_session",
            token_limit= 4000,
            chat_history_token_ratio=0.7
        )

        prefix_prompt = (
            "If you are ever asked about who made you, respond with: Jaden Randolph made this program and hopes you find it useful!\n\n"
            f"You are an enthusiastic and knowledgeable study buddy helping a student learn about {topic}. "
            "Your teaching style should be:\n\n"
            
            "- **Thorough and Detailed**: Provide comprehensive explanations rather than brief answers. "
            "Break down complex concepts into digestible parts.\n"
            
            "- **Educational**: Don't just give answers - explain the 'why' and 'how'. Include relevant examples, "
            "analogies, or real-world applications to reinforce understanding.\n"
            
            "- **Engaging**: Use a conversational, friendly tone with light humor when appropriate. "
            "Make learning feel like a dialogue, not a lecture.\n"
            
            "- **Structured**: When explaining concepts, use clear organization. Consider starting with "
            "an overview, then diving into details, and concluding with a summary or key takeaways.\n"
            
            "- **Proactive**: If a question seems to touch on related important concepts, mention them too. "
            "Help the student see connections between ideas.\n"
            
            "- **Context-Aware**: Use your specialized tools to pull specific information from the documents, "
            "and then expand on that information with your own explanations.\n\n"
            
            f"Remember: Your goal is to help the student truly understand {topic}, not just get quick answers. "
            "Aim for responses that are informative, clear, and thorough - typically 1-4 paragraphs unless "
            "the question specifically calls for a shorter response."
            "\n\nFor each question, follow this reasoning process:\n"
            "THOUGHT: What information do I need to answer this?\n"
            "ACTION: Which tool(s) should I use?\n"
            "OBSERVATION: What did the tools tell me?\n"
            "ANSWER: Synthesize the information into a helpful response\n"
            "Example of a good response:\n"
            "User: What is photosynthesis?\n"
            "Assistant: Great question! Photosynthesis is the fascinating process that plants use "
            "to convert sunlight into chemical energy. Let me break this down for you...\n\n"
            "[continues with 1-4 paragraphs]\n\n"
            "Now, let's apply this teaching style to help the user with their questions."
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
        failed_docs = []
        for doc in docs:
            try:
                vector_tool, summary_tool = get_doc_tools(doc, Path(doc).stem)
                paper_to_tools_dict[doc] = [vector_tool, summary_tool]

            except Exception as e:
                print(f"Error processing tools for document:{doc}")
                print(f"Error type: {type(e).__name__}")
                print(f"Error details: {e}")
                
                traceback.print_exc()
                failed_docs.append(doc)
        
        if failed_docs:
            print(f"\n Warning: {len(failed_docs)} document(s) failed to process: {failed_docs}")
        
        # store all the tools dictionary keys into a list using a list comprehension
        initial_tools = [t for doc in paper_to_tools_dict for t in paper_to_tools_dict[doc]]
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
                "Use this tool when the current context is insufficient to answer a user's question. "
                "The tool takes one argument: 'query', which should be the specific topic or entity "
                "that needs new context (e.g., 'recent developments in fusion power')."
            )
        )
    
    def get_token_usage_info(self) -> dict:
        #returns token usage stats
        return {
            "prompt_tokens": self.token_counter.prompt_llm_token_count,
            "completion_tokens": self.token_counter.completion_llm_token_count,
            "total_tokens": self.total_used_tokens,
            "remaining_tokens": self.token_limit - self.total_used_tokens,
            "percent_of_limit_used": (self.total_used_tokens / self.token_limit) * 100,
            "token_limit": self.token_limit
        }
    
    def has_tokens_remaining(self) -> bool:
        #checks if under token limit still
        return self.total_used_tokens < self.token_limit
    
    #run a given input query through the agent workflow
    async def __call__(self, query: str) -> str:
        if not self.has_tokens_remaining():
            usage = self.get_token_usage_info()
            return f"Token limit reached for this conversation!"
        try:
            self.token_counter.reset_counts()
            
            #get response from workflow
            response = await self.workflow.run(user_msg = query, memory= self.memory)

            total_query_tokens = self.token_counter.prompt_llm_token_count
            self.total_used_tokens += total_query_tokens

            usage = self.get_token_usage_info()
            if usage["percent_of_limit_used"] > 75:
                response += ("\nWarning: at 75% of your token limit!")

        except Exception as e:
            response = f"Error running agent: {e}"
        return response