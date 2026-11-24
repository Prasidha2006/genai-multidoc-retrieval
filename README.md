## DESIGN AND IMPLEMENTATION OF A MULTIDOCUMENT RETRIEVAL AGENT USING LLAMAINDEX

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:
To develop an intelligent multidocument retrieval system that can access multiple research papers, retrieve relevant information based on user queries, and generate meaningful responses using a Large Language Model (LLM).

### DESIGN STEPS:

#### STEP 1:
Import the required libraries, configure the OpenAI API key, and initialize the asynchronous execution environment using nest_asyncio.

#### STEP 2:
Load three research papers and create vector search tools and summary tools for each document using LlamaIndex. Combine all the generated tools into a single list for multidocument retrieval.

#### STEP 3:
Configure the GPT-4o-mini language model and text embedding model. Create a FunctionAgent using the collected document tools and submit a query about chatbot testing and evaluation results to retrieve and generate relevant information.

### PROGRAM:

```
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()

import nest_asyncio
nest_asyncio.apply()

urls = [
    "https://ieeexplore.ieee.org/document/9297533/",
    "https://ieeexplore.ieee.org/document/9908774/",
    "https://ieeexplore.ieee.org/document/8718222/",
]

papers = [
    "Dynamic_Information_Retrieval.pdf",
    "Answering.pdf",
    "Chatbot.pdf",
]

from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}

for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(
        paper, Path(paper).stem
    )
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

initial_tools = [
    t for paper in papers
    for t in paper_to_tools_dict[paper]
]

import llama_index.llms.openai.utils as openai_utils
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

openai_utils.ALL_AVAILABLE_MODELS["gpt-4o-mini"] = 128000
openai_utils.CHAT_MODELS["gpt-4o-mini"] = 128000

Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-ada-002"
)

len(initial_tools)

from llama_index.core.agent.workflow import FunctionAgent

agent = FunctionAgent(
    tools=initial_tools,
    llm=Settings.llm,
    verbose=True
)

response = await agent.run(
    "Tell me about the testing in chatbots, "
    "and then tell me about the evaluation results"
)

print(str(response))
```

### OUTPUT:

1. Successfully loaded three research papers.
2. Generated six tools, including three vector search tools and three summary tools.
3. Successfully initialized the GPT-4o-mini language model and embedding model.
4. Created the multidocument retrieval agent using FunctionAgent.
5. The agent retrieved information related to chatbot testing and evaluation.
6. Generated a response covering:
   - Unit Testing
   - Integration Testing
   - User Acceptance Testing
   - Accuracy
   - User Satisfaction
   - Response Time
   - Natural Language Understanding

<img width="1322" height="500" alt="image" src="https://github.com/user-attachments/assets/b05a9ac7-64c4-43af-a756-943bc4457f5f" />

### RESULT:

Thus, the multidocument retrieval agent was successfully designed and implemented using LlamaIndex. The agent retrieved relevant information from the loaded research papers and generated meaningful responses to user queries using GPT-4o-mini.
