from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents import load_tools, Tool
# from langchain.tools import DuckDuckGoSearchResults
from tavily import TavilyClient
from langchain.prompts import PromptTemplate
from model import get_gemma
from api_key import TAVILY_API_KEY

gemma_model = get_gemma()


# Create the ReAct template
react_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate(
    template=react_template,
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
)


tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

tavily_search_tool = Tool(
    name="tavily search",
    description = "A web search engine. Use this to as a search engine for general queries.",
    func = lambda x: tavily_client.search(x, max_results=1)
)

# Prepare tools
tools = load_tools(["llm-math"], llm=gemma_model)
tools.append(tavily_search_tool)


# Construct the ReAct agent
agent = create_react_agent(gemma_model, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True
)

def get_urls_from_response(response):
    urls = []
    for step in response["intermediate_steps"]:
        urls.append(step[1]["results"][0]["url"])
    return urls


def search_web(query):
    response = agent_executor.invoke({"input" : query})

    output = response["output"]
    sources = get_urls_from_response(response)

    return output, sources 
    